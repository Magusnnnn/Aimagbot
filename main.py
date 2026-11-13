"""Telegram-бот с ИИ: текст, фото, голосовые, видео, создание картинок, память, рассылка.

Ключи в код НЕ вписываем — они берутся из переменных окружения (FadeHost → Переменные среды).

Обязательные:
  TELEGRAM_BOT_TOKEN  — токен бота от @BotFather
  GROQ_API_KEY        — ключ Groq (текст, зрение, распознавание речи)
Необязательные:
  SYSTEM_PROMPT       — «характер» бота
  ADMIN_IDS           — твой Telegram ID (через запятую): /broadcast, /stats, картинки без лимита
  CF_ACCOUNT_ID, CF_API_TOKEN — Cloudflare Workers AI: стабильное создание картинок (бесплатно)
                        Если не заданы, картинки рисует бесплатный Pollinations (без регистрации)
  IMAGE_DAILY_LIMIT   — сколько картинок в день на человека (по умолчанию 5)
  ALLOWED_USER_IDS    — если задано, бот отвечает только этим людям
  GROQ_MODEL, VISION_MODEL, STT_MODEL, CF_IMAGE_MODEL — смена моделей
  BOT_NAME            — имя в приветствии (по умолчанию MagOne)
  USERS_FILE, MEMORY_DIR — где хранить список пользователей и память чатов
"""
import base64
import io
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from urllib.parse import quote

import requests
import telebot
from telebot.apihelper import ApiTelegramException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ai-bot")


def parse_ids(value):
    """'123, 456' -> {123, 456}"""
    return {int(x) for x in value.replace(" ", "").split(",") if x.isdigit()}


def env_int(name, default):
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


# ---------- Настройки ----------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()  # текст
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b").strip()  # фото и кадры видео
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo").strip()  # голос -> текст
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "").strip()
CF_IMAGE_MODEL = os.getenv("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell").strip()
IMAGE_DAILY_LIMIT = env_int("IMAGE_DAILY_LIMIT", 5)
BOT_NAME = os.getenv("BOT_NAME", "MagOne").strip()
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты — дружелюбный и умный ИИ-ассистент в Telegram. "
    "Отвечай на том языке, на котором пишет пользователь (по умолчанию — по-русски), "
    "кратко и по делу. Не используй Markdown-разметку (**, ##, таблицы): "
    "пиши обычным текстом, для списков используй дефисы или цифры.",
)
ALLOWED_USERS = parse_ids(os.getenv("ALLOWED_USER_IDS", ""))
ADMIN_IDS = parse_ids(os.getenv("ADMIN_IDS", ""))
USERS_FILE = os.getenv("USERS_FILE", "users.json")
MEMORY_DIR = os.getenv("MEMORY_DIR", "memory")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Память: сколько сообщений держим дословно и когда сжимаем старое в краткую сводку
CONTEXT_MESSAGES = 20  # максимум сообщений, которые уходят в ИИ
SUMMARIZE_AT = 20  # если сообщений стало больше — старые сжимаем в сводку
KEEP_MESSAGES = 10  # столько последних сообщений остаётся после сжатия
CLIP_CHARS = 1500  # длинные сообщения в памяти обрезаем
SUMMARY_CHARS = 1500

MAX_IMAGE_BYTES = 3 * 1024 * 1024  # фото для ИИ-зрения
MAX_MEDIA_BYTES = 20 * 1024 * 1024  # голосовые и видео (лимит Telegram для ботов)

if not BOT_TOKEN or not GROQ_API_KEY:
    sys.exit("Нет ключей: задай переменные TELEGRAM_BOT_TOKEN и GROQ_API_KEY")

bot = telebot.TeleBot(BOT_TOKEN, num_threads=4)
chat_locks = defaultdict(threading.Lock)  # сообщения одного чата идут по очереди


def clip(text, limit=CLIP_CHARS):
    return text if len(text) <= limit else text[:limit] + "…"


# ---------- Список пользователей (для рассылки) ----------
users_lock = threading.Lock()
broadcast_lock = threading.Lock()


def load_users():
    try:
        with open(USERS_FILE, encoding="utf-8") as f:
            return {int(x) for x in json.load(f)}
    except (OSError, ValueError, TypeError):
        return set()


def save_users():
    """Вызывать, держа users_lock. Если файл записать нельзя — бот всё равно работает."""
    try:
        tmp = USERS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sorted(users), f)
        os.replace(tmp, USERS_FILE)
    except OSError as e:
        log.error("Не удалось сохранить список пользователей: %s", e)


users = load_users()


def remember_user(message):
    """Запоминаем ID личных чатов — им потом можно отправить рассылку."""
    if message.chat.type != "private":
        return
    with users_lock:
        if message.chat.id not in users:
            users.add(message.chat.id)
            save_users()


# ---------- Память чатов (хранится в файлах, переживает перезапуск) ----------
chats = {}  # chat_id -> {"summary": "...", "messages": [{"role": ..., "content": ...}]}
chats_lock = threading.Lock()


def chat_file(chat_id):
    return os.path.join(MEMORY_DIR, f"{chat_id}.json")


def load_chat(chat_id):
    try:
        with open(chat_file(chat_id), encoding="utf-8") as f:
            data = json.load(f)
        msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in data.get("messages", [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        ]
        return {"summary": str(data.get("summary", "")), "messages": msgs}
    except (OSError, ValueError, AttributeError):
        return {"summary": "", "messages": []}


def get_chat(chat_id):
    with chats_lock:
        if chat_id not in chats:
            chats[chat_id] = load_chat(chat_id)
        return chats[chat_id]


def save_chat(chat_id):
    chat = chats.get(chat_id)
    if chat is None:
        return
    try:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        tmp = chat_file(chat_id) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(chat, f, ensure_ascii=False)
        os.replace(tmp, chat_file(chat_id))
    except OSError as e:
        log.error("Не удалось сохранить память чата %s: %s", chat_id, e)


def forget_chat(chat_id):
    """Полностью стирает память чата (и из RAM, и с диска)."""
    with chats_lock:
        chats.pop(chat_id, None)
    try:
        os.remove(chat_file(chat_id))
    except OSError:
        pass


def trim(msgs, limit=CONTEXT_MESSAGES):
    """Последние сообщения для ИИ; история всегда начинается с сообщения пользователя."""
    msgs = msgs[-limit:]
    while msgs and msgs[0]["role"] != "user":
        msgs = msgs[1:]
    return msgs


def system_for(chat):
    if chat["summary"]:
        return SYSTEM_PROMPT + "\n\nКраткая память об этом пользователе и прошлых разговорах:\n" + chat["summary"]
    return SYSTEM_PROMPT


SUMMARY_PROMPT = (
    "Ты ведёшь краткую память чата ИИ-ассистента. Тебе дают прежнюю сводку и новые сообщения. "
    "Напиши обновлённую сводку по-русски, не длиннее 1200 символов: имя и важные факты о пользователе, "
    "его цели и предпочтения, договорённости, незаконченные задачи. Только по делу, без воды. "
    "Выведи только текст сводки."
)


def update_summary(chat):
    """Старые сообщения сжимаются в краткую сводку, чтобы бот помнил долгий разговор."""
    msgs = chat["messages"]
    if len(msgs) <= SUMMARIZE_AT:
        return
    old, recent = msgs[:-KEEP_MESSAGES], msgs[-KEEP_MESSAGES:]
    transcript = "\n".join(
        ("Пользователь: " if m["role"] == "user" else "Бот: ") + m["content"][:600] for m in old
    )
    prompt = f"Прежняя сводка:\n{chat['summary'] or '(пусто)'}\n\nНовые сообщения:\n{transcript}"
    try:
        summary = ask_ai([{"role": "user", "content": prompt}], system=SUMMARY_PROMPT, max_tokens=2000)
        chat["summary"] = summary[:SUMMARY_CHARS]
        chat["messages"] = trim(recent, KEEP_MESSAGES)
    except Exception as e:
        log.warning("Не получилось обновить сводку памяти: %s", e)
        if len(msgs) > SUMMARIZE_AT * 2:  # чтобы память не росла бесконечно
            chat["messages"] = trim(recent, KEEP_MESSAGES)


# ---------- ИИ (Groq) ----------
class AIError(Exception):
    """Ошибка, текст которой можно показать пользователю."""


def _groq_post(payload):
    try:
        return requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json=payload,
            timeout=90,
        )
    except requests.RequestException as e:
        log.error("Нет связи с Groq: %s", e)
        raise AIError("Не получилось связаться с ИИ. Попробуй ещё раз через минуту.")


def ask_ai(messages, system=None, model=None, extra=None, max_tokens=3000):
    """Отправляет диалог в Groq и возвращает текст ответа."""
    payload = {
        "model": model or MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT if system is None else system}] + messages,
        "max_completion_tokens": max_tokens,
    }
    if extra:
        payload.update(extra)
    r = _groq_post(payload)
    if r.status_code == 400 and extra and "reasoning" in r.text.lower():
        for key in extra:  # модель не знает этих настроек — пробуем без них
            payload.pop(key, None)
        r = _groq_post(payload)

    if r.status_code == 429:
        raise AIError("Лимит запросов к ИИ исчерпан. Подожди минуту и повтори.")
    if r.status_code == 413:
        raise AIError("Слишком длинный диалог. Отправь /clear и спроси ещё раз.")
    if r.status_code != 200:
        log.error("Groq вернул %s: %s", r.status_code, r.text[:500])
        raise AIError("ИИ сейчас недоступен. Попробуй позже.")

    try:
        text = r.json()["choices"][0]["message"].get("content") or ""
    except (ValueError, KeyError, IndexError, AttributeError):
        log.error("Странный ответ Groq: %s", r.text[:500])
        raise AIError("ИИ вернул непонятный ответ. Попробуй ещё раз.")

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()  # недописанные размышления
    if not text:
        raise AIError("ИИ вернул пустой ответ. Попробуй переформулировать вопрос.")
    return text


def split_text(text, limit=4000):
    """Telegram не принимает сообщения длиннее 4096 символов — режем на части."""
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


def reply_with_ai(message, user_content, history_text, model=None, extra=None):
    """Общая логика ответа: память чата -> ИИ -> сохранение -> отправка.

    user_content — то, что видит ИИ (текст или текст + картинки),
    history_text — короткая текстовая запись для памяти (картинки в память не кладём).
    """
    chat_id = message.chat.id
    with chat_locks[chat_id]:
        bot.send_chat_action(chat_id, "typing")
        chat = get_chat(chat_id)
        context = trim(chat["messages"] + [{"role": "user", "content": user_content}])
        try:
            answer = ask_ai(context, system=system_for(chat), model=model, extra=extra)
        except AIError as e:
            bot.reply_to(message, str(e))  # память не меняется
            return
        except Exception:
            log.exception("Непредвиденная ошибка")
            bot.reply_to(message, "Что-то пошло не так. Попробуй ещё раз.")
            return

        chat["messages"] += [
            {"role": "user", "content": clip(history_text)},
            {"role": "assistant", "content": clip(answer)},
        ]
        save_chat(chat_id)
        for part in split_text(answer):
            bot.send_message(chat_id, part)
        if len(chat["messages"]) > SUMMARIZE_AT:  # после ответа, чтобы не задерживать его
            update_summary(chat)
            save_chat(chat_id)


def remember_exchange(chat_id, user_text, bot_text):
    """Записывает в память то, что ИИ не писал сам (например, «нарисовал картинку»)."""
    with chat_locks[chat_id]:
        chat = get_chat(chat_id)
        chat["messages"] += [
            {"role": "user", "content": clip(user_text)},
            {"role": "assistant", "content": clip(bot_text)},
        ]
        save_chat(chat_id)


# ---------- Файлы из Telegram, зрение, речь ----------
def download_telegram_file(file_id, max_bytes):
    try:
        info = bot.get_file(file_id)
        if info.file_size and info.file_size > max_bytes:
            raise AIError(f"Файл слишком большой (максимум {max_bytes // (1024 * 1024)} МБ).")
        return bot.download_file(info.file_path)
    except AIError:
        raise
    except Exception as e:
        log.warning("Не удалось скачать файл: %s", e)
        raise AIError("Не получилось скачать файл. Telegram разрешает ботам только файлы до 20 МБ.")


def image_part(data, mime="image/jpeg"):
    b64 = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def transcribe(data, filename):
    """Голос/звук видео -> текст (Groq Whisper). Пустая строка — речи не найдено."""
    try:
        r = requests.post(
            GROQ_STT_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (filename, data)},
            data={"model": STT_MODEL, "response_format": "json", "temperature": "0"},
            timeout=120,
        )
    except requests.RequestException as e:
        log.error("Нет связи с Groq (речь): %s", e)
        raise AIError("Не получилось связаться с ИИ. Попробуй ещё раз через минуту.")
    if r.status_code == 429:
        raise AIError("Лимит распознавания речи исчерпан. Подожди немного и повтори.")
    if r.status_code == 413:
        raise AIError("Файл слишком большой для распознавания речи.")
    if r.status_code == 400:  # например, в видео нет звуковой дорожки
        log.warning("Groq не принял файл (речь): %s", r.text[:300])
        return ""
    if r.status_code != 200:
        log.error("Groq (речь) вернул %s: %s", r.status_code, r.text[:500])
        raise AIError("Не получилось распознать речь. Попробуй позже.")
    try:
        text = (r.json().get("text") or "").strip()
    except ValueError:
        text = ""
    return text if len(text) >= 2 else ""


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # необязательная библиотека

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extract_frames(data, duration=None):
    """До трёх кадров из видео (JPEG). Если ffmpeg нет — пустой список."""
    exe = find_ffmpeg()
    if not exe:
        return []
    duration = duration or 6
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.mp4")
        with open(src, "wb") as f:
            f.write(data)
        for i, part in enumerate((0.15, 0.5, 0.85)):
            out = os.path.join(tmp, f"f{i}.jpg")
            cmd = [exe, "-v", "error", "-ss", f"{duration * part:.2f}", "-i", src,
                   "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "5", "-y", out]
            try:
                subprocess.run(cmd, capture_output=True, timeout=40)
            except (subprocess.TimeoutExpired, OSError) as e:
                log.warning("ffmpeg не сработал: %s", e)
                continue
            if os.path.exists(out) and os.path.getsize(out) > 0:
                with open(out, "rb") as f:
                    frames.append(f.read())
    return frames


# ---------- Создание картинок ----------
IMG_TRIGGER = re.compile(
    r"^\s*(?:пожалуйста[,\s]+)?(?:нарисуй(?:те)?|(?:сгенерируй|создай|сделай|generate|make|create)(?:те)?\s+"
    r"(?:мне\s+|me\s+)?(?:an?\s+)?(?:картин|изображени|фот|рисун|арт\b|image|picture|photo)\w*)(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)

IMAGE_PROMPT_SYSTEM = (
    "You write prompts for the FLUX text-to-image model. The user's request may be in any language. "
    "Reply with ONE English prompt (max 60 words): subject, setting, style, lighting, composition. "
    "No quotes, no explanations. "
    "If the request is sexual or erotic, shows nudity, graphic violence or gore, hate, self-harm, "
    "or asks for a realistic image of a real private person, reply with exactly: REFUSE"
)

image_usage = {}  # (user_id, "ГГГГ-ММ-ДД") -> сколько картинок уже нарисовано
usage_lock = threading.Lock()
pollinations_lock = threading.Lock()
pollinations_last = [0.0]


def today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def images_used(user_id):
    with usage_lock:
        return image_usage.get((user_id, today()), 0)


def count_image(user_id):
    with usage_lock:
        for key in [k for k in image_usage if k[1] != today()]:
            del image_usage[key]
        key = (user_id, today())
        image_usage[key] = image_usage.get(key, 0) + 1


def enhance_image_prompt(request):
    """Переводит и улучшает запрос для рисующей модели. 'REFUSE' — рисовать нельзя."""
    prompt = ask_ai([{"role": "user", "content": request}], system=IMAGE_PROMPT_SYSTEM, max_tokens=800)
    return prompt.strip().strip('"').strip()[:600]


def image_cloudflare(prompt):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_IMAGE_MODEL}"
    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {CF_API_TOKEN}"},
            json={"prompt": prompt, "steps": 4},
            timeout=120,
        )
    except requests.RequestException as e:
        log.error("Нет связи с Cloudflare: %s", e)
        raise AIError("Не получилось связаться с художником. Попробуй ещё раз позже.")
    if r.headers.get("content-type", "").startswith("image/"):
        return r.content
    low = r.text.lower()
    if r.status_code != 200:
        log.error("Cloudflare вернул %s: %s", r.status_code, r.text[:500])
        if "nsfw" in low:
            raise AIError("Такое я не рисую. Придумай что-нибудь другое 🙂")
        if r.status_code == 429 or "neurons" in low or "4006" in low:
            raise AIError("Бесплатный лимит картинок на сегодня закончился. Попробуй завтра.")
        raise AIError("Не получилось нарисовать картинку. Попробуй ещё раз позже.")
    try:
        image_b64 = (r.json().get("result") or {}).get("image")
        return base64.b64decode(image_b64)
    except (ValueError, TypeError, AttributeError):
        log.error("Странный ответ Cloudflare: %s", r.text[:300])
        raise AIError("Не получилось нарисовать картинку. Попробуй ещё раз позже.")


def image_pollinations(prompt):
    """Бесплатный вариант без регистрации. Без ключа Pollinations пускает ~1 запрос в 15 секунд."""
    url = "https://image.pollinations.ai/prompt/" + quote(prompt[:400], safe="")
    params = {"width": 1024, "height": 1024, "model": "flux", "nologo": "true", "seed": random.randint(1, 10**6)}
    with pollinations_lock:
        wait = 16 - (time.time() - pollinations_last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, params=params, timeout=120)
        except requests.RequestException as e:
            log.error("Нет связи с Pollinations: %s", e)
            raise AIError("Не получилось связаться с художником. Попробуй ещё раз позже.")
        finally:
            pollinations_last[0] = time.time()
    if r.status_code == 429:
        raise AIError("Художник сейчас занят. Подожди полминуты и попробуй снова.")
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
        log.error("Pollinations вернул %s: %s", r.status_code, r.text[:300])
        raise AIError("Не получилось нарисовать картинку. Попробуй ещё раз позже.")
    return r.content


def generate_image(prompt):
    if CF_ACCOUNT_ID and CF_API_TOKEN:
        return image_cloudflare(prompt)
    return image_pollinations(prompt)


def do_image_job(message, request):
    try:
        user_id, chat_id = message.from_user.id, message.chat.id
        unlimited = is_admin(message)
        if not unlimited and images_used(user_id) >= IMAGE_DAILY_LIMIT:
            bot.reply_to(message, f"На сегодня лимит картинок ({IMAGE_DAILY_LIMIT}) закончился. Завтра можно снова!")
            return
        bot.reply_to(message, "🎨 Рисую… это может занять до минуты.")
        bot.send_chat_action(chat_id, "upload_photo")
        try:
            prompt = enhance_image_prompt(request)
        except AIError:
            prompt = request[:400]  # не вышло улучшить — рисуем как есть
        if prompt.upper().startswith("REFUSE"):
            bot.reply_to(message, "Такое я не рисую. Придумай что-нибудь другое 🙂")
            return
        try:
            image = generate_image(prompt)
        except AIError as e:
            bot.reply_to(message, str(e))
            return
        photo = io.BytesIO(image)
        photo.name = "image.jpg"
        bot.send_photo(chat_id, photo, caption=clip(request, 200))
        if not unlimited:
            count_image(user_id)
        remember_exchange(chat_id, request, "[Я нарисовал картинку по этому запросу]")
    except Exception:
        log.exception("Ошибка при создании картинки")
        try:
            bot.reply_to(message, "Не получилось нарисовать картинку. Попробуй ещё раз позже.")
        except Exception:
            pass


def start_image_job(message, request):
    threading.Thread(target=do_image_job, args=(message, request), daemon=True).start()


# ---------- Доступ ----------
def is_allowed(message):
    if not ALLOWED_USERS:
        return True
    return bool(message.from_user and message.from_user.id in ALLOWED_USERS)


def is_admin(message):
    return bool(message.from_user and message.from_user.id in ADMIN_IDS)


def guard(message):
    """True — можно отвечать. Для чужих (если бот приватный) отвечает отказом."""
    if is_allowed(message):
        remember_user(message)
        return True
    bot.reply_to(message, "Извини, этот бот приватный.")
    return False


# ---------- Рассылка ----------
def send_one(uid, text):
    """True — доставлено, False — не вышло, None — пользователя надо убрать из списка."""
    for _ in range(2):
        try:
            bot.send_message(uid, text)
            return True
        except ApiTelegramException as e:
            if e.error_code == 429:  # Telegram просит подождать
                wait = (getattr(e, "result_json", None) or {}).get("parameters", {}).get("retry_after", 5)
                time.sleep(min(wait, 30) + 1)
                continue
            description = (getattr(e, "description", "") or "").lower()
            if e.error_code == 403 or (e.error_code == 400 and "chat not found" in description):
                return None  # пользователь заблокировал бота или удалил аккаунт
            log.warning("Рассылка: ошибка для %s: %s", uid, e)
            return False
        except Exception as e:
            log.warning("Рассылка: ошибка для %s: %s", uid, e)
            return False
    return False


def run_broadcast(admin_chat, text):
    try:
        with users_lock:
            targets = sorted(users)
        bot.send_message(admin_chat, f"Отправляю {len(targets)} пользователям…")
        sent = failed = 0
        gone = []
        for uid in targets:
            result = send_one(uid, text)
            if result is True:
                sent += 1
            elif result is None:
                gone.append(uid)
            else:
                failed += 1
            time.sleep(0.05)  # ~20 сообщений в секунду — в пределах лимитов Telegram
        if gone:
            with users_lock:
                users.difference_update(gone)
                save_users()
        report = f"Готово. Доставлено: {sent}"
        if failed:
            report += f", не доставлено: {failed}"
        if gone:
            report += f", убрано из списка (заблокировали бота): {len(gone)}"
        bot.send_message(admin_chat, report)
    except Exception:
        log.exception("Ошибка рассылки")
    finally:
        broadcast_lock.release()


# ---------- Команды ----------
@bot.message_handler(commands=["start", "help"])
def cmd_start(message):
    if not is_allowed(message):
        return
    remember_user(message)
    bot.reply_to(
        message,
        f"Привет! Я {BOT_NAME} 👋\n\n"
        "Пиши вопросы, присылай фото, голосовые и видео — я пойму.\n"
        "Нарисовать картинку: /img описание или «нарисуй …».\n\n"
        "/clear — стереть память нашего диалога",
    )


@bot.message_handler(commands=["clear", "reset", "forget"])
def cmd_clear(message):
    if not is_allowed(message):
        return
    with chat_locks[message.chat.id]:
        forget_chat(message.chat.id)
    bot.reply_to(message, "Готово, я всё забыл — и диалог, и сводку.")


@bot.message_handler(commands=["img", "draw", "imagine"])
def cmd_img(message):
    if not guard(message):
        return
    parts = message.text.split(maxsplit=1)
    request = parts[1].strip() if len(parts) > 1 else ""
    if not request:
        bot.reply_to(message, "Напиши так: /img рыжий кот в космосе")
        return
    start_image_job(message, request)


@bot.message_handler(commands=["id"])
def cmd_id(message):
    bot.reply_to(message, f"Твой Telegram ID: {message.from_user.id}")


@bot.message_handler(commands=["stats"])
def cmd_stats(message):
    if not is_admin(message):
        return
    with users_lock:
        count = len(users)
    bot.reply_to(message, f"Пользователей в списке рассылки: {count}")


@bot.message_handler(commands=["broadcast"])
def cmd_broadcast(message):
    if not is_admin(message):
        return  # для остальных команды как будто нет
    parts = message.text.split(maxsplit=1)
    text = parts[1].strip() if len(parts) > 1 else ""
    if not text:
        bot.reply_to(message, "Напиши так: /broadcast Текст объявления")
        return
    if len(text) > 4000:
        bot.reply_to(message, "Слишком длинное объявление (максимум 4000 символов).")
        return
    if not broadcast_lock.acquire(blocking=False):
        bot.reply_to(message, "Рассылка уже идёт, подожди, пока закончится.")
        return
    threading.Thread(target=run_broadcast, args=(message.chat.id, text), daemon=True).start()


# ---------- Обычные сообщения ----------
@bot.message_handler(content_types=["text"])
def on_text(message):
    if not guard(message):
        return
    m = IMG_TRIGGER.match(message.text)
    if m:
        if m.group("rest").strip():
            start_image_job(message, message.text.strip())
        else:
            bot.reply_to(message, "Что нарисовать? Например: нарисуй рыжего кота в космосе")
        return
    reply_with_ai(message, message.text, message.text)


def handle_image(message, file_id, file_size, mime):
    if file_size and file_size > MAX_IMAGE_BYTES:
        bot.reply_to(message, "Картинка слишком большая (до 3 МБ). Пришли её как обычное фото, не файлом.")
        return
    try:
        data = download_telegram_file(file_id, MAX_IMAGE_BYTES)
    except AIError as e:
        bot.reply_to(message, str(e))
        return
    caption = (message.caption or "").strip()
    question = caption or "Опиши, что изображено на фото. Если на нём есть текст — перепиши его."
    content = [{"type": "text", "text": question}, image_part(data, mime)]
    reply_with_ai(
        message, content, "[Пользователь прислал фото] " + (caption or "без подписи"),
        model=VISION_MODEL, extra={"reasoning_effort": "none"},
    )


@bot.message_handler(content_types=["photo"])
def on_photo(message):
    if not guard(message):
        return
    photo = message.photo[-1]  # самая большая версия
    handle_image(message, photo.file_id, photo.file_size, "image/jpeg")


@bot.message_handler(content_types=["document"])
def on_document(message):
    if not guard(message):
        return
    doc = message.document
    mime = (doc.mime_type or "").lower()
    if mime in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        handle_image(message, doc.file_id, doc.file_size, mime)
    else:
        bot.reply_to(message, "Такие файлы я пока не читаю. Пришли текст, фото, голосовое или видео.")


def process_media(message, obj, filename, is_video):
    """Голосовое, аудио или видео: расшифровываем звук (и смотрим кадры), потом отвечаем."""
    try:
        if getattr(obj, "file_size", None) and obj.file_size > MAX_MEDIA_BYTES:
            bot.reply_to(message, "Файл слишком большой: я могу скачать до 20 МБ.")
            return
        bot.send_chat_action(message.chat.id, "typing")
        data = download_telegram_file(obj.file_id, MAX_MEDIA_BYTES)
        transcript = transcribe(data, filename)
        caption = (message.caption or "").strip()

        if not is_video:
            if not transcript:
                bot.reply_to(message, "Не расслышал речь. Попробуй записать ещё раз поближе к микрофону.")
                return
            bot.reply_to(message, "🎤 " + clip(transcript, 500))  # показываем, как я понял
            text = f"{caption}\n\n{transcript}" if caption else transcript
            reply_with_ai(message, text, "🎤 " + transcript)
            return

        frames = extract_frames(data, getattr(obj, "duration", None))
        note = f"Подпись к видео: {caption}. " if caption else ""
        speech = f"Расшифровка звука: {transcript}. " if transcript else "Речи в видео нет. "
        history = f"[Пользователь прислал видео] {caption} {clip(transcript, 500)}".strip()
        if frames:
            question = f"Это видео. {note}{speech}Ниже несколько кадров из него. Расскажи, что происходит."
            content = [{"type": "text", "text": question}] + [image_part(f) for f in frames]
            reply_with_ai(message, content, history, model=VISION_MODEL, extra={"reasoning_effort": "none"})
        elif transcript:
            question = (f"Это видео, но картинку я не вижу — только звук. {note}{speech}"
                        "Перескажи суть и ответь на подпись, если она есть.")
            reply_with_ai(message, question, history)
        else:
            bot.reply_to(message, "В этом видео нет речи, а кадры я посмотреть не могу (на сервере нет ffmpeg).")
    except AIError as e:
        bot.reply_to(message, str(e))
    except Exception:
        log.exception("Ошибка при обработке голоса/видео")
        try:
            bot.reply_to(message, "Что-то пошло не так с этим файлом. Попробуй ещё раз.")
        except Exception:
            pass


@bot.message_handler(content_types=["voice", "audio", "video", "video_note"])
def on_media(message):
    if not guard(message):
        return
    if message.voice:
        obj, filename, is_video = message.voice, "voice.ogg", False
    elif message.audio:
        obj, filename, is_video = message.audio, message.audio.file_name or "audio.mp3", False
    elif message.video:
        obj, filename, is_video = message.video, message.video.file_name or "video.mp4", True
    else:
        obj, filename, is_video = message.video_note, "video_note.mp4", True
    threading.Thread(target=process_media, args=(message, obj, filename, is_video), daemon=True).start()


@bot.message_handler(content_types=["sticker", "animation"])
def on_other(message):
    if is_allowed(message):
        bot.reply_to(message, "Стикеры и гифки я пока не понимаю 🙂 Напиши текстом или пришли фото.")


if __name__ == "__main__":
    log.info(
        "Бот запущен. Текст: %s | зрение: %s | речь: %s | картинки: %s | ffmpeg: %s | пользователей: %d",
        MODEL, VISION_MODEL, STT_MODEL,
        "Cloudflare" if (CF_ACCOUNT_ID and CF_API_TOKEN) else "Pollinations",
        "есть" if find_ffmpeg() else "нет",
        len(users),
    )
    bot.remove_webhook()  # на случай, если раньше был включён webhook
    bot.infinity_polling(skip_pending=True)
