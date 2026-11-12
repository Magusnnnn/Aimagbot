"""Telegram-бот с ИИ (Groq).

Ключи в код НЕ вписываем — они берутся из переменных окружения:
  TELEGRAM_BOT_TOKEN  — токен бота от @BotFather
  GROQ_API_KEY        — ключ Groq
Необязательные:
  GROQ_MODEL          — модель (по умолчанию openai/gpt-oss-120b)
  SYSTEM_PROMPT       — «характер» бота
  ALLOWED_USER_IDS    — Telegram ID через запятую; если задано, бот отвечает только им
  ADMIN_IDS           — Telegram ID админов (твой) через запятую: им доступны /broadcast и /stats
  USERS_FILE          — где хранить список пользователей для рассылки (по умолчанию users.json)
"""
import json
import logging
import os
import re
import sys
import threading
import time
from collections import defaultdict

import requests
import telebot
from telebot.apihelper import ApiTelegramException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ai-bot")


def parse_ids(value):
    """'123, 456' -> {123, 456}"""
    return {int(x) for x in value.replace(" ", "").split(",") if x.isdigit()}


# ---------- Настройки ----------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
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
HISTORY_LIMIT = 12  # сколько последних сообщений помнит бот в одном чате
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

if not BOT_TOKEN or not GROQ_API_KEY:
    sys.exit("Нет ключей: задай переменные TELEGRAM_BOT_TOKEN и GROQ_API_KEY")

bot = telebot.TeleBot(BOT_TOKEN, num_threads=4)
history = defaultdict(list)  # chat_id -> [{"role": ..., "content": ...}, ...]
chat_locks = defaultdict(threading.Lock)  # сообщения одного чата идут по очереди


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


# ---------- ИИ ----------
class AIError(Exception):
    """Ошибка, текст которой можно показать пользователю."""


def ask_ai(messages):
    """Отправляет диалог в Groq и возвращает текст ответа."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "max_completion_tokens": 3000,
    }
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json=payload,
            timeout=90,
        )
    except requests.RequestException as e:
        log.error("Нет связи с Groq: %s", e)
        raise AIError("Не получилось связаться с ИИ. Попробуй ещё раз через минуту.")

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

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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


def trim(msgs):
    """Оставляет последние HISTORY_LIMIT сообщений; история начинается с вопроса пользователя."""
    msgs = msgs[-HISTORY_LIMIT:]
    while msgs and msgs[0]["role"] != "user":
        msgs = msgs[1:]
    return msgs


def is_allowed(message):
    if not ALLOWED_USERS:
        return True
    return bool(message.from_user and message.from_user.id in ALLOWED_USERS)


def is_admin(message):
    return bool(message.from_user and message.from_user.id in ADMIN_IDS)


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
        "Привет! Я ИИ-бот. Просто напиши мне вопрос или задачу.\n\n"
        "/clear — забыть историю диалога",
    )


@bot.message_handler(commands=["clear", "reset"])
def cmd_clear(message):
    if not is_allowed(message):
        return
    history.pop(message.chat.id, None)
    bot.reply_to(message, "Готово, диалог очищен.")


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


@bot.message_handler(content_types=["text"])
def on_text(message):
    if not is_allowed(message):
        bot.reply_to(message, "Извини, этот бот приватный.")
        return

    remember_user(message)
    chat_id = message.chat.id
    with chat_locks[chat_id]:
        bot.send_chat_action(chat_id, "typing")
        saved = history[chat_id]
        question = {"role": "user", "content": message.text}
        context = trim(saved + [question])

        try:
            answer = ask_ai(context)
        except AIError as e:
            bot.reply_to(message, str(e))  # история не меняется
            return
        except Exception:
            log.exception("Непредвиденная ошибка")
            bot.reply_to(message, "Что-то пошло не так. Попробуй ещё раз.")
            return

        # запоминаем вопрос и ответ только после успешного ответа ИИ
        saved[:] = trim(context + [{"role": "assistant", "content": answer}])
        for part in split_text(answer):
            bot.send_message(chat_id, part)


@bot.message_handler(
    content_types=["photo", "voice", "audio", "video", "document", "sticker", "video_note"]
)
def on_other(message):
    if is_allowed(message):
        remember_user(message)
        bot.reply_to(message, "Пока я понимаю только текстовые сообщения.")


if __name__ == "__main__":
    log.info("Бот запущен. Модель: %s. Пользователей в списке: %d", MODEL, len(users))
    bot.remove_webhook()  # на случай, если раньше был включён webhook
    bot.infinity_polling(skip_pending=True)
