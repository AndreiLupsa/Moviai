import asyncio
import datetime
import os

import aiohttp

LOG_FILEPATH = ""


def initialize_log_for_run(timestamp: str):
    global LOG_FILEPATH

    LOG_DIRECTORY = os.path.join("data", "logs")
    os.makedirs(LOG_DIRECTORY, exist_ok=True)

    LOG_FILENAME = f"log_{timestamp}.txt"
    LOG_FILEPATH = os.path.join(LOG_DIRECTORY, LOG_FILENAME)

    log_message(f"--- Logging for run {timestamp} initialized. Output saved to: {LOG_FILEPATH} ---")


def log_message(message: str):
    message_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"[{message_timestamp}] {message}"

    print(log_entry)

    if LOG_FILEPATH:
        with open(LOG_FILEPATH, 'a', encoding='utf-8') as log_file:
            log_file.write(log_entry + "\n")


# log_message(f"--- Logging initialized. Output will be saved to: {LOG_FILEPATH} ---")


async def download_media_file(url: str, path: str) -> bool:
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    with open(path, 'wb') as f:
                        f.write(await response.read())
                    log_message(f"Media file successfully saved to {path}")
                    return True
        except Exception as e:
            log_message(f"--- WARNING: Download attempt {attempt + 1} failed for {url}. Reason: {e}")
            if attempt < 2:
                await asyncio.sleep(2)

    log_message(f"--- ERROR: Failed to download from {url} after 3 attempts.")
    return False
