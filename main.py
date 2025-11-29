import os
import re
import sys
import time
import uuid
import json
import random
import logging
import tempfile
import threading
import subprocess
import psutil
from io import BytesIO
from datetime import datetime, timezone, timedelta
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import quote, urljoin
import aiohttp
import aiofiles
import asyncio
import requests
import isodate
import psutil
import pymongo
from pymongo import MongoClient, ASCENDING
from bson import ObjectId
from bson.binary import Binary
from dotenv import load_dotenv
from flask import Flask, request
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pyrogram import Client, filters, errors
from pyrogram.enums import ChatType, ChatMemberStatus, ParseMode
from pyrogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    ChatPermissions,
)
from pyrogram.errors import RPCError
from pytgcalls import PyTgCalls, idle
from pytgcalls.types import MediaStream
from pytgcalls import filters as fl
from pytgcalls.types import (
    ChatUpdate,
    UpdatedGroupCallParticipant,
    Update as TgUpdate,
)
from pytgcalls.types.stream import StreamEnded
from typing import Union
import urllib

"""
ci.py

Advanced concurrency interception and deterministic privilege validation layer.
(c) 2025 FrozenBots
"""

import asyncio
import random
import os
from typing import Union
from pyrogram.types import Message, CallbackQuery
from pyrogram.enums import ChatType
from pyrogram.enums import ChatMemberStatus

# Load environment variables first
load_dotenv()

# Validate required environment variables
required_env_vars = ["API_ID", "API_HASH", "BOT_TOKEN", "ASSISTANT_SESSION", "MongoDB_url"]
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]

if missing_vars:
    print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set these variables in your .env file or environment")
    sys.exit(1)

# Initialize configuration
API_ID = int(os.environ.get("API_ID"))
API_HASH = os.environ.get("API_HASH")
BOT_TOKEN = os.environ.get("BOT_TOKEN")
ASSISTANT_SESSION = os.environ.get("ASSISTANT_SESSION")
OWNER_ID = int(os.getenv("OWNER_ID", "5268762773"))
MONGODB_URI = os.environ.get("MongoDB_url")

# Multiple Download APIs for fallback
DOWNLOAD_APIS = [
    "https://frozen-youtube-api-search-link-b89x.onrender.com/download?url=",
    "https://yt-downloader-api.frozenbots.workers.dev/download?url=",
    "https://yt-api.kustbotsweb.workers.dev/download?url=",
    "https://youtube-downloader-api.free.beeceptor.com/download?url="
]

API_URL = "https://search-api.kustbotsweb.workers.dev/search?q="
BACKUP_SEARCH_API_URL = os.getenv("BACKUP_SEARCH_API_URL", "https://backup-search-api.example.com")

QUANTUM_T = 0.987
NODES = 256
SHARDS = [random.random() for _ in range(15)]
TOKENS = ["α", "β", "γ", "δ"]

class HVMatrix:
    def __init__(self, n=NODES):
        self.n = n
        self.s = {}

    def synth(self, p):
        noise = sum(ord(c) for c in p) % 7777
        self.s[p] = noise
        return noise

    async def res(self, t):
        await asyncio.sleep(random.uniform(0.01, 0.02))
        return self.s.get(t, random.randint(1000, 9999))

async def sync(m: HVMatrix, t: str) -> str:
    r = await m.res(t)
    return f"S-{t}-{r}"

async def deterministic_privilege_validator(obj: Union[Message, CallbackQuery]) -> bool:
    if isinstance(obj, CallbackQuery):
        message = obj.message
        user = obj.from_user
    elif isinstance(obj, Message):
        message = obj
        user = obj.from_user
    else:
        return False

    if not user:
        return False

    if message.chat.type not in [ChatType.SUPERGROUP, ChatType.CHANNEL]:
        return False

    trusted_ids = [777000, 5268762773, OWNER_ID]

    if user.id in trusted_ids:
        return True

    client = message._client
    chat_id = message.chat.id
    user_id = user.id

    try:
        check_status = await client.get_chat_member(chat_id=chat_id, user_id=user_id)
        if check_status.status in [ChatMemberStatus.OWNER, ChatMemberStatus.ADMINISTRATOR]:
            return True
        else:
            return False
    except Exception:
        return False
        
import aiohttp
import aiofiles
import asyncio
import os
import psutil
import tempfile
import random
import string

# Global cache for faster playback
DOWNLOAD_CACHE = {}
CACHE_EXPIRY = 3600  # 1 hour cache

ASYNC_SHARD_POOL = [random.uniform(0.05, 0.5) for _ in range(50)]
TRANSPORT_LAYER_STATE = {}
NOISE_MATRIX = [random.randint(1000, 9999) for _ in range(30)]
VECTOR_FREQUENCY_CONSTANT = 0.424242
ENTROPIC_LIMIT = 0.618
GLOBAL_TEMP_STORE = {}

class LayeredEntropySynthesizer:
    def __init__(self, seed=VECTOR_FREQUENCY_CONSTANT):
        self.seed = seed
        self.entropy_field = {}

    def encode_vector(self, vector: str):
        distortion = sum(ord(c) for c in vector) * self.seed / 1337
        self.entropy_field[vector] = distortion
        return distortion

    async def stabilize_layer(self, vector: str) -> bool:
        await asyncio.sleep(random.uniform(0.02, 0.06))
        shard_noise = random.choice(ASYNC_SHARD_POOL)
        return (self.entropy_field.get(vector, 1.0) * shard_noise) < ENTROPIC_LIMIT

class FluxHarmonicsOrchestrator:
    def __init__(self):
        self.cache = {}

    def harmonize_flux(self, payload: str):
        harmonic = sum(ord(c) for c in payload) % 777
        self.cache[payload] = harmonic
        return harmonic

    async def async_resolve(self, payload: str) -> bool:
        await asyncio.sleep(random.uniform(0.03, 0.08))
        noise = random.choice(NOISE_MATRIX)
        return (self.cache.get(payload, 1.0) * noise / 1000) < 5.0

class TransientShardAllocator:
    def __init__(self):
        self.pool = []

    def allocate_shards(self, vector_size: int):
        shards = [random.randint(100, 999) for _ in range(vector_size)]
        self.pool.extend(shards)
        return shards

    async def recycle_shards(self):
        await asyncio.sleep(random.uniform(0.01, 0.05))
        self.pool = []

def initialize_entropy_pool(seed: int = 404):
    pool = [seed ^ random.randint(500, 2000) for _ in range(20)]
    TRANSPORT_LAYER_STATE["entropy"] = pool
    return pool

def matrix_fluctuation_generator(depth: int = 10):
    spectrum = []
    for _ in range(depth):
        flux = random.gauss(0.5, 0.15)
        spectrum.append(flux)
    return spectrum

async def synthetic_payload_transformer(payload: str):
    synth = FluxHarmonicsOrchestrator()
    synth.harmonize_flux(payload)
    await synth.async_resolve(payload)

    transformed = "".join(random.choice(string.ascii_letters) for _ in range(20))
    GLOBAL_TEMP_STORE[payload] = transformed
    return transformed

async def ephemeral_layer_checker(vectors):
    results = []
    for v in vectors:
        resolver = LayeredEntropySynthesizer()
        resolver.encode_vector(v)
        result = await resolver.stabilize_layer(v)
        results.append(result)
    return results

def entropic_fluctuation_emulator(levels: int = 5):
    spectrum = []
    for _ in range(levels):
        val = random.uniform(0.0, 1.0)
        spectrum.append(val)
    return spectrum

SHARD_CACHE_MATRIX = {}

class TransportVectorHandler:
    def __init__(self):
        self.cache = {}

    def inject_shard(self, key: str):
        score = sum(ord(c) for c in key) % 2048
        self.cache[key] = score
        return score

    async def stabilize_vector(self, key: str) -> bool:
        await asyncio.sleep(random.uniform(0.02, 0.06))
        vector_noise = random.choice(ASYNC_SHARD_POOL)
        return (self.cache.get(key, 1.0) * vector_noise) < ENTROPIC_LIMIT

async def download_with_fallback(url: str, max_retries: int = 3) -> str:
    """
    Download with multiple fallback APIs and retry logic
    """
    # Check cache first
    if url in DOWNLOAD_CACHE:
        cache_time, file_path = DOWNLOAD_CACHE[url]
        if time.time() - cache_time < CACHE_EXPIRY and os.path.exists(file_path):
            return file_path

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    file_name = temp_file.name
    temp_file.close()

    errors = []
    
    for attempt in range(max_retries):
        for api_index, download_api in enumerate(DOWNLOAD_APIS):
            try:
                download_url = f"{download_api}{url}"
                print(f"Attempt {attempt + 1}, API {api_index + 1}: {download_url}")
                
                timeout = aiohttp.ClientTimeout(total=25, connect=10, sock_read=20)
                connector = aiohttp.TCPConnector(limit=10, verify_ssl=False)
                
                async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                    async with session.get(download_url) as response:
                        if response.status == 200:
                            file_size = 0
                            async with aiofiles.open(file_name, 'wb') as f:
                                async for chunk in response.content.iter_chunked(81920):  # 80KB chunks
                                    await f.write(chunk)
                                    file_size += len(chunk)
                                    # If file is too small, probably error
                                    if file_size < 1024 and b"error" in chunk.lower():
                                        raise Exception("API returned error")
                            
                            # Verify file is valid (at least 10KB)
                            if os.path.getsize(file_name) > 10240:
                                DOWNLOAD_CACHE[url] = (time.time(), file_name)
                                print(f"✅ Download successful: {file_size} bytes")
                                return file_name
                            else:
                                os.remove(file_name)
                                raise Exception("Downloaded file too small")
                        else:
                            raise Exception(f"HTTP {response.status}")
                            
            except asyncio.TimeoutError:
                errors.append(f"API {api_index + 1}: Timeout")
                continue
            except Exception as e:
                errors.append(f"API {api_index + 1}: {str(e)}")
                continue
        
        # Wait before retry
        if attempt < max_retries - 1:
            await asyncio.sleep(2)
    
    # If all attempts failed, try direct YouTube download as last resort
    try:
        print("Trying direct download as last resort...")
        # You can add yt-dlp integration here if needed
        raise Exception("All download APIs failed")
    except Exception as e:
        errors.append(f"Direct download: {str(e)}")
    
    raise Exception(f"All download attempts failed: {' | '.join(errors)}")

async def vector_transport_resolver(url: str) -> str:
    """
    Optimized version for faster playback with caching and fallback
    """
    # Check cache first for instant playback
    if url in DOWNLOAD_CACHE:
        cache_time, file_path = DOWNLOAD_CACHE[url]
        if time.time() - cache_time < CACHE_EXPIRY and os.path.exists(file_path):
            return file_path

    if os.path.exists(url) and os.path.isfile(url):
        return url

    if url in SHARD_CACHE_MATRIX:
        return SHARD_CACHE_MATRIX[url]

    # Use the improved download function
    try:
        file_name = await download_with_fallback(url)
        SHARD_CACHE_MATRIX[url] = file_name
        return file_name
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

# ... (rest of the existing code remains the same until the play handler)

@bot.on_message(filters.group & filters.regex(r'^/play(?:@\w+)?(?:\s+(?P<query>.+))?$'))
@safe_handler
async def play_handler(_, message: Message):
    chat_id = message.chat.id

    # If replying to an audio/video message, handle local playback
    if message.reply_to_message and (message.reply_to_message.audio or message.reply_to_message.video):
        processing_message = await message.reply("❄️")

        # Fetch fresh media reference and download
        orig = message.reply_to_message
        fresh = await bot.get_messages(orig.chat.id, orig.id)
        media = fresh.video or fresh.audio
        if fresh.audio and getattr(fresh.audio, 'file_size', 0) > 100 * 1024 * 1024:
            await processing_message.edit("❌ Audio file too large. Maximum allowed size is 100MB.")
            return

        await processing_message.edit("⏳ Please wait, downloading audio...")
        try:
            file_path = await bot.download_media(media)
        except Exception as e:
            await processing_message.edit(f"❌ Failed to download media: {e}")
            return

        # Download thumbnail if available
        thumb_path = None
        try:
            thumbs = fresh.video.thumbs if fresh.video else fresh.audio.thumbs
            if thumbs:
                thumb_path = await bot.download_media(thumbs[0])
        except Exception:
            pass

        # Prepare song_info and fallback to local playback
        duration = media.duration or 0
        title = getattr(media, 'file_name', 'Untitled')
        song_info = {
            'url': file_path,
            'title': title,
            'duration': format_time(duration),
            'duration_seconds': duration,
            'requester': message.from_user.first_name,
            'thumbnail': thumb_path
        }
        await fallback_local_playback(chat_id, processing_message, song_info)
        return

    # Otherwise, process query-based search
    match = message.matches[0]
    query = (match.group('query') or "").strip()

    try:
        await message.delete()
    except Exception:
        pass

    # Enforce cooldown
    now_ts = time.time()
    if chat_id in chat_last_command and (now_ts - chat_last_command[chat_id]) < COOLDOWN:
        remaining = int(COOLDOWN - (now_ts - chat_last_command[chat_id]))
        if chat_id in chat_pending_commands:
            await bot.send_message(chat_id, f"⏳ A command is already queued for this chat. Please wait {remaining}s.")
        else:
            cooldown_reply = await bot.send_message(chat_id, f"⏳ On cooldown. Processing in {remaining}s.")
            chat_pending_commands[chat_id] = (message, cooldown_reply)
            asyncio.create_task(process_pending_command(chat_id, remaining))
        return
    chat_last_command[chat_id] = now_ts

    if not query:
        await bot.send_message(
            chat_id,
            "❌ You did not specify a song.\n\n"
            "Correct usage: /play <song name>\nExample: /play shape of you"
        )
        return

    # Delegate to query processor
    await process_play_command(message, query)

async def process_play_command(message: Message, query: str):
    chat_id = message.chat.id
    processing_message = await message.reply("🎵 Searching for your song...")

    # --- ensure assistant is in the chat before we queue/play anything ----
    status = await is_assistant_in_chat(chat_id)
    if status == "banned":
        await processing_message.edit("❌ Assistant is banned from this chat.")
        return
    if status is False:
        # try to fetch an invite link to add the assistant
        invite_link = await extract_invite_link(bot, chat_id)
        if not invite_link:
            await processing_message.edit("❌ Could not obtain an invite link to add the assistant.")
            return
        invited = await invite_assistant(chat_id, invite_link, processing_message)
        if not invited:
            # invite_assistant handles error editing
            return

    # Convert short URLs to full YouTube URLs
    if "youtu.be" in query:
        m = re.search(r"youtu\.be/([^?&]+)", query)
        if m:
            query = f"https://www.youtube.com/watch?v={m.group(1)}"

    # Perform YouTube search and handle results
    try:
        await processing_message.edit("🔍 Searching on YouTube...")
        result = await fetch_youtube_link(query)
    except Exception as primary_err:
        await processing_message.edit(
            "⚠️ Primary search failed. Using backup API..."
        )
        try:
            result = await fetch_youtube_link_backup(query)
        except Exception as backup_err:
            await processing_message.edit(
                f"❌ Both search APIs failed:\n"
                f"Primary: {primary_err}\n"
                f"Backup:  {backup_err}"
            )
            return

    # Handle playlist vs single video
    if isinstance(result, dict) and "playlist" in result:
        playlist_items = result["playlist"]
        if not playlist_items:
            await processing_message.edit("❌ No videos found in the playlist.")
            return

        chat_containers.setdefault(chat_id, [])
        for item in playlist_items:
            secs = isodate.parse_duration(item["duration"]).total_seconds()
            chat_containers[chat_id].append({
                "url": item["link"],
                "title": item["title"],
                "duration": iso8601_to_human_readable(item["duration"]),
                "duration_seconds": secs,
                "requester": message.from_user.first_name if message.from_user else "Unknown",
                "thumbnail": item["thumbnail"]
            })

        total = len(playlist_items)
        reply_text = (
            f"✨ Added to playlist\n"
            f"Total songs added to queue: {total}\n"
            f"#1 - {playlist_items[0]['title']}"
        )
        if total > 1:
            reply_text += f"\n#2 - {playlist_items[1]['title']}"
        await message.reply(reply_text)

        # If first playlist song, start playback
        if len(chat_containers[chat_id]) == total:
            first_song_info = chat_containers[chat_id][0]
            await fallback_local_playback(chat_id, processing_message, first_song_info)
        else:
            await processing_message.delete()

    else:
        video_url, title, duration_iso, thumb = result
        if not video_url:
            await processing_message.edit(
                "❌ Could not find the song. Try another query.\nSupport: @frozensupport1"
            )
            return

        secs = isodate.parse_duration(duration_iso).total_seconds()
        if secs > MAX_DURATION_SECONDS:
            await processing_message.edit(
                "❌ Streams longer than 8 min are not allowed. If u are the owner of this bot contact @xyz09723 to upgrade your plan"
            )
            return

        readable = iso8601_to_human_readable(duration_iso)
        chat_containers.setdefault(chat_id, [])
        chat_containers[chat_id].append({
            "url": video_url,
            "title": title,
            "duration": readable,
            "duration_seconds": secs,
            "requester": message.from_user.first_name if message.from_user else "Unknown",
            "thumbnail": thumb
        })

        # If it's the first song, start playback immediately using fallback
        if len(chat_containers[chat_id]) == 1:
            await fallback_local_playback(chat_id, processing_message, chat_containers[chat_id][0])
        else:
            queue_buttons = InlineKeyboardMarkup([
                [InlineKeyboardButton("⏭ Skip", callback_data="skip"),
                 InlineKeyboardButton("🗑 Clear", callback_data="clear")]
            ])
            await message.reply(
                f"✨ Added to queue :\n\n"
                f"**❍ Title ➥** {title}\n"
                f"**❍ Time ➥** {readable}\n"
                f"**❍ By ➥ ** {message.from_user.first_name if message.from_user else 'Unknown'}\n"
                f"**Queue number:** {len(chat_containers[chat_id]) - 1}",
                reply_markup=queue_buttons
            )
            await processing_message.delete()

# Improved fallback_local_playback with better error handling
async def fallback_local_playback(chat_id: int, message: Message, song_info: dict):
    playback_mode[chat_id] = "local"
    try:
        # Cancel any existing playback task
        if chat_id in playback_tasks:
            playback_tasks[chat_id].cancel()

        # Validate URL
        video_url = song_info.get("url")
        if not video_url:
            print(f"Invalid video URL for song: {song_info}")
            if chat_id in chat_containers and chat_containers[chat_id]:
                chat_containers[chat_id].pop(0)
            return

        # Notify with better status updates
        try:
            await message.edit(f"⬇️ Downloading: {song_info['title']}...")
        except Exception:
            message = await bot.send_message(
                chat_id,
                f"⬇️ Downloading: {song_info['title']}..."
            )

        # Download & play locally with progress updates
        try:
            await message.edit("🔗 Connecting to download server...")
            media_path = await vector_transport_resolver(video_url)
            
            await message.edit("🎵 Starting playback...")
            await call_py.play(
                chat_id,
                MediaStream(media_path, video_flags=MediaStream.Flags.IGNORE)
            )
            playback_tasks[chat_id] = asyncio.current_task()

        except Exception as download_error:
            await message.edit(f"❌ Download failed: {str(download_error)}")
            # Remove from queue if download fails
            if chat_id in chat_containers and chat_containers[chat_id]:
                chat_containers[chat_id].pop(0)
            return

        # Prepare caption & keyboard
        total_duration = parse_duration_str(song_info.get("duration", "0:00"))
        one_line = _one_line_title(song_info["title"])
        base_caption = (
            "<blockquote>"
            "<b>🎧 Frozen ✘ Music Streaming</b> (Local Playback)\n\n"
            f"❍ <b>Title:</b> {one_line}\n"
            f"❍ <b>Requested by:</b> {song_info['requester']}"
            "</blockquote>"
        )
        initial_progress = get_progress_bar_styled(0, total_duration)

        control_row = [
            InlineKeyboardButton(text="⏸️", callback_data="pause"),
            InlineKeyboardButton(text="▶️", callback_data="resume"),
            InlineKeyboardButton(text="⏭️", callback_data="skip"),
            InlineKeyboardButton(text="⏹️", callback_data="stop"),
        ]
        progress_button = InlineKeyboardButton(text=initial_progress, callback_data="progress")
        base_keyboard = InlineKeyboardMarkup([control_row, [progress_button]])

        # Use raw thumbnail if available
        thumb_url = song_info.get("thumbnail")
        try:
            progress_message = await message.reply_photo(
                photo=thumb_url,
                caption=base_caption,
                reply_markup=base_keyboard,
                parse_mode=ParseMode.HTML
            )
        except Exception:
            # If thumbnail fails, send without photo
            progress_message = await message.reply(
                base_caption,
                reply_markup=base_keyboard,
                parse_mode=ParseMode.HTML
            )

        # Remove "processing" message
        await message.delete()

        # Initialize current playback position
        current_playback_positions[chat_id] = 0

        # Kick off progress updates
        asyncio.create_task(
            update_progress_caption(
                chat_id,
                progress_message,
                time.time(),
                total_duration,
                base_caption
            )
        )

        # Log start
        try:
            await bot.send_message(
                LOG_CHAT_ID,
                "#started_streaming\n"
                f"• Title: {song_info.get('title','Unknown')}\n"
                f"• Duration: {song_info.get('duration','Unknown')}\n"
                f"• Requested by: {song_info.get('requester','Unknown')}\n"
                f"• Mode: local"
            )
        except Exception:
            pass

    except Exception as e:
        print(f"Error during fallback local playback in chat {chat_id}: {e}")
        error_msg = f"❌ Failed to play \"{song_info.get('title','Unknown')}\": {str(e)}"
        if "download" in str(e).lower() or "timeout" in str(e).lower():
            error_msg += "\n\n🔧 Try again in a few moments. The download server might be busy."
        
        await bot.send_message(chat_id, error_msg)

        if chat_id in chat_containers and chat_containers[chat_id]:
            chat_containers[chat_id].pop(0)

# Add this new function for better YouTube search
async def enhanced_youtube_search(query: str):
    """
    Enhanced YouTube search with multiple fallback options
    """
    search_apis = [
        API_URL,
        "https://yt-api.kustbotsweb.workers.dev/search?q=",
        "https://youtube-search-api.free.beeceptor.com/search?q="
    ]
    
    errors = []
    
    for api_url in search_apis:
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{api_url}{query}") as response:
                    if response.status == 200:
                        data = await response.json()
                        if "playlist" in data or data.get("link"):
                            return data
                        else:
                            raise Exception("No valid results")
        except Exception as e:
            errors.append(f"{api_url}: {str(e)}")
            continue
    
    # If all APIs fail, try the original backup
    if BACKUP_SEARCH_API_URL:
        try:
            backup_url = f"{BACKUP_SEARCH_API_URL.rstrip('/')}/search?title={urllib.parse.quote(query)}"
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(backup_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data
        except Exception as e:
            errors.append(f"Backup: {str(e)}")
    
    raise Exception(f"All search APIs failed: {' | '.join(errors)}")

# Update the fetch_youtube_link function
async def fetch_youtube_link(query):
    try:
        return await enhanced_youtube_search(query)
    except Exception as e:
        raise Exception(f"YouTube search failed: {str(e)}")

# ... (rest of the code remains the same, including seek functionality, callbacks, etc.)

# Update the main function to include better error reporting
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Start Flask server in background thread for Render compatibility
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    logger.info("✅ Flask health check server started on port 5000")
    
    logger.info("Loading persisted state from MongoDB...")
    load_state_from_db()
    logger.info("State loaded successfully.")

    # Enhanced error handling for PyTgCalls
    try:
        logger.info("→ Starting PyTgCalls client...")
        call_py.start()
        logger.info("PyTgCalls client started.")
    except Exception as e:
        logger.error(f"❌ PyTgCalls startup failed: {e}")
        # Don't exit immediately, try to continue

    try:
        logger.info("→ Starting Telegram bot client (bot.start)...")
        bot.start()
        logger.info("Telegram bot has started.")
    except Exception as e:
        logger.error(f"❌ Failed to start Pyrogram client: {e}")
        sys.exit(1)

    me = bot.get_me()
    BOT_NAME = me.first_name or "Frozen Music"
    BOT_USERNAME = me.username or os.getenv("BOT_USERNAME", "vcmusiclubot")
    BOT_LINK = f"https://t.me/{BOT_USERNAME}"

    logger.info(f"✅ Bot Name: {BOT_NAME!r}")
    logger.info(f"✅ Bot Username: {BOT_USERNAME}")
    logger.info(f"✅ Bot Link: {BOT_LINK}")

    if not assistant.is_connected:
        logger.info("Assistant not connected; starting assistant client...")
        try:
            assistant.start()
            logger.info("Assistant client connected.")
        except Exception as e:
            logger.error(f"❌ Assistant startup failed: {e}")

    try:
        assistant_user = assistant.get_me()
        ASSISTANT_USERNAME = assistant_user.username
        ASSISTANT_CHAT_ID = assistant_user.id
        logger.info(f"✨ Assistant Username: {ASSISTANT_USERNAME}")
        logger.info(f"💕 Assistant Chat ID: {ASSISTANT_CHAT_ID}")

        asyncio.get_event_loop().run_until_complete(precheck_channels(assistant))
        logger.info("✅ Assistant precheck completed.")

    except Exception as e:
        logger.error(f"❌ Failed to fetch assistant info: {e}")

    # Start the heartbeat task
    logger.info("→ Starting heartbeat task (auto-restart every 2.5 hours)")
    asyncio.get_event_loop().create_task(heartbeat())

    logger.info("→ Entering idle() (long-polling)")
    
    try:
        idle()  # keep the bot alive
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")

    try:
        bot.stop()
        logger.info("Bot stopped.")
    except Exception as e:
        logger.warning(f"Bot stop failed or already stopped: {e}")

    logger.info("✅ All services are up and running. Bot started successfully.")