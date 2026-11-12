"""Telegram-бот с ИИ (Groq).

Ключи в код НЕ вписываем — они берутся из переменных окружения:
  TELEGRAM_BOT_TOKEN  — токен бота от @BotFather
  GROQ_API_KEY        — ключ Groq
Необязательные:
  GROQ_MODEL          — модель (по умолчанию openai/gpt-oss-120b)
  SYSTEM_PROMPT       — «характер» бота
  ALLOWED_USER_IDS    — Telegram ID через запятую; если задано, бот отвечает только им
"""
import logging
import os
import re
import sys
import threading
from collections import defaultdict

import requests
import telebot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ai-bot")

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
ALLOWED_USERS = {
    int(x)
    for x in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(",")
    if x.isdigit()
}
HISTORY_LIMIT = 12  # сколько последних сообщений помнит бот в одном чате
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

if not BOT_TOKEN or not GROQ_API_KEY:
    sys.exit("Нет ключей: задай переменные TELEGRAM_BOT_TOKEN и GROQ_API_KEY")

bot = telebot.TeleBot(BOT_TOKEN, num_threads=4)
history = defaultdict(list)  # chat_id -> [{"role": ..., "content": ...}, ...]
chat_locks = defaultdict(threading.Lock)  # сообщения одного чата идут по очереди


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


@bot.message_handler(commands=["start", "help"])
def cmd_start(message):
    if not is_allowed(message):
        return
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


@bot.message_handler(content_types=["text"])
def on_text(message):
    if not is_allowed(message):
        bot.reply_to(message, "Извини, этот бот приватный.")
        return

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
        bot.reply_to(message, "Пока я понимаю только текстовые сообщения.")


if __name__ == "__main__":
    log.info("Бот запущен. Модель: %s", MODEL)
    bot.remove_webhook()  # на случай, если раньше был включён webhook
    bot.infinity_polling(skip_pending=True)
