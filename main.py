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
from pytgcalls.types.stream import StreamEnded
from typing import Union
import urllib

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

# Fast APIs - Multiple fallbacks
API_URLS = [
    "https://search-api.kustbotsweb.workers.dev/search?q=",
    "https://yt-api.p.rapidapi.com/search?q=",
    "https://youtube-search-results.p.rapidapi.com/youtube-search/?q="
]

# Bot initialization
from pyrogram.errors import UserAlreadyParticipant
import logging

logger = logging.getLogger(__name__)

async def precheck_channels(client):
    targets = ["@kustbots", "@kustbotschat"]
    for chan in targets:
        try:
            await client.join_chat(chan)
            logger.info(f"✓ Joined {chan}")
        except UserAlreadyParticipant:
            logger.info(f"↻ Already in {chan}")
        except Exception as e:
            logger.warning(f"✗ Failed to join {chan}: {e}")

# Safe resolve_peer patch
logging.getLogger("pyrogram").setLevel(logging.ERROR)
_original_resolve_peer = Client.resolve_peer
async def _safe_resolve_peer(self, peer_id):
    try:
        return await _original_resolve_peer(self, peer_id)
    except (KeyError, ValueError) as e:
        if "ID not found" in str(e) or "Peer id invalid" in str(e):
            return None
        raise
Client.resolve_peer = _safe_resolve_peer

# Exception handler
def _custom_exception_handler(loop, context):
    exc = context.get("exception")
    if isinstance(exc, (KeyError, ValueError)) and (
        "ID not found" in str(exc) or "Peer id invalid" in str(exc)
    ):
        return  
    if isinstance(exc, AttributeError) and "has no attribute 'write'" in str(exc):
        return
    loop.default_exception_handler(context)

asyncio.get_event_loop().set_exception_handler(_custom_exception_handler)

# Initialize clients
session_name = os.environ.get("SESSION_NAME", "music_bot1")
bot = Client(session_name, bot_token=BOT_TOKEN, api_id=API_ID, api_hash=API_HASH)
assistant = Client("assistant_account", session_string=ASSISTANT_SESSION)
call_py = PyTgCalls(assistant)

ASSISTANT_USERNAME = None
ASSISTANT_CHAT_ID = None

# MongoDB Setup
try:
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client["music_bot"]
    broadcast_collection = db["broadcast"]
    state_backup = db["state_backup"]
    logger.info("✅ MongoDB connected successfully")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    sys.exit(1)

# Global variables
chat_containers = {}
playback_tasks = {}  
bot_start_time = time.time()
COOLDOWN = 5  # Reduced cooldown
chat_last_command = {}
chat_pending_commands = {}
QUEUE_LIMIT = 20
MAX_DURATION_SECONDS = 480  
current_playback_position = {}

# ULTRA-FAST YOUTUBE SEARCH
async def ultra_fast_youtube_search(query: str):
    """Ultra-fast YouTube search with multiple fallbacks"""
    # Try multiple APIs simultaneously
    tasks = []
    for api_url in API_URLS:
        task = asyncio.create_task(fetch_from_api(api_url + urllib.parse.quote(query)))
        tasks.append(task)
    
    # Wait for first successful response
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    
    # Cancel pending tasks
    for task in pending:
        task.cancel()
    
    for task in done:
        try:
            result = task.result()
            if result:
                return result
        except:
            continue
    
    # If all APIs fail, use direct YouTube search
    return await direct_youtube_search(query)

async def fetch_from_api(url: str):
    """Fetch from API with ultra-fast timeout"""
    timeout = aiohttp.ClientTimeout(total=3, connect=1, sock_connect=1, sock_read=2)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and (data.get('link') or data.get('playlist')):
                        return data
    except:
        pass
    return None

async def direct_youtube_search(query: str):
    """Direct YouTube search as final fallback"""
    try:
        search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
        timeout = aiohttp.ClientTimeout(total=5, connect=2, sock_connect=2, sock_read=3)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(search_url) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Extract first video ID
                    video_ids = re.findall(r'watch\?v=([a-zA-Z0-9_-]{11})', html)
                    if video_ids:
                        video_id = video_ids[0]
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        
                        # Get basic info quickly
                        return {
                            "link": video_url,
                            "title": query,
                            "duration": "PT3M",
                            "thumbnail": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
                        }
    except:
        pass
    
    # Final fallback - return basic structure
    return {
        "link": f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}",
        "title": query,
        "duration": "PT3M", 
        "thumbnail": ""
    }

# ULTRA-FAST PLAYBACK SYSTEM
async def instant_playback(chat_id: int, message: Message, song_info: dict):
    """INSTANT playback - starts within 1-3 seconds"""
    try:
        # Cancel any existing playback
        if chat_id in playback_tasks:
            playback_tasks[chat_id].cancel()

        video_url = song_info.get("url")
        if not video_url:
            return

        # INSTANT PLAYBACK - Direct URL streaming
        start_time = time.time()
        
        await call_py.play(
            chat_id,
            MediaStream(video_url, video_flags=MediaStream.Flags.IGNORE)
        )
        
        playback_time = time.time() - start_time
        print(f"🎵 INSTANT PLAYBACK started in {playback_time:.2f}s")
        
        playback_tasks[chat_id] = asyncio.current_task()
        current_playback_position[chat_id] = {
            'start_time': time.time(),
            'song_info': song_info
        }

        # Show playback UI instantly
        await show_playback_ui(chat_id, message, song_info)
        
    except Exception as e:
        print(f"Instant playback failed: {e}")
        await message.edit(f"❌ Instant play failed, trying fast download...")
        await fast_download_playback(chat_id, message, song_info)

async def fast_download_playback(chat_id: int, message: Message, song_info: dict):
    """Fast download fallback"""
    try:
        video_url = song_info.get("url")
        
        await message.edit("⚡ Fast downloading...")
        start_time = time.time()
        
        # Fast download with 10s timeout
        downloaded_file = await asyncio.wait_for(
            fast_download(video_url),
            timeout=10
        )
        
        await call_py.play(
            chat_id,
            MediaStream(downloaded_file, video_flags=MediaStream.Flags.IGNORE)
        )
        
        download_time = time.time() - start_time
        print(f"⚡ FAST DOWNLOAD completed in {download_time:.2f}s")
        
        playback_tasks[chat_id] = asyncio.current_task()
        current_playback_position[chat_id] = {
            'start_time': time.time(),
            'song_info': song_info,
            'file_path': downloaded_file
        }

        await show_playback_ui(chat_id, message, song_info)
        
    except Exception as e:
        print(f"Fast download failed: {e}")
        await message.edit(f"❌ Playback failed: {e}")
        if chat_id in chat_containers and chat_containers[chat_id]:
            chat_containers[chat_id].pop(0)

async def fast_download(url: str) -> str:
    """Ultra-fast download with 5s timeout"""
    timeout = aiohttp.ClientTimeout(total=5, connect=2, sock_connect=2, sock_read=3)
    
    try:
        # Try multiple download endpoints
        download_urls = [
            f"https://frozen-youtube-api-search-link-b89x.onrender.com/download?url={url}",
            f"https://yt-downloader-api.vercel.app/download?url={url}",
            f"https://youtube-downloader-api.herokuapp.com/download?url={url}"
        ]
        
        for download_url in download_urls:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(download_url) as response:
                        if response.status == 200:
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                            file_name = temp_file.name
                            temp_file.close()
                            
                            # Fast streaming download
                            async with aiofiles.open(file_name, 'wb') as f:
                                async for chunk in response.content.iter_chunked(81920):  # Larger chunks
                                    await f.write(chunk)
                            
                            return file_name
            except:
                continue
                
        raise Exception("All download endpoints failed")
        
    except Exception as e:
        raise Exception(f"Download failed: {e}")

async def show_playback_ui(chat_id: int, message: Message, song_info: dict):
    """Show playback UI instantly"""
    total_duration = parse_duration_str(song_info.get("duration", "0:00"))
    one_line = _one_line_title(song_info["title"])
    
    base_caption = (
        "<blockquote>"
        "<b>🎧 INSTANT PLAYBACK</b>\n\n"
        f"❍ <b>Title:</b> {one_line}\n"
        f"❍ <b>Requested by:</b> {song_info['requester']}\n"
        f"❍ <b>Status:</b> 🔥 Playing Instantly"
        "</blockquote>"
    )

    control_row = [
        InlineKeyboardButton(text="⏸️", callback_data="pause"),
        InlineKeyboardButton(text="▶️", callback_data="resume"), 
        InlineKeyboardButton(text="⏭️", callback_data="skip"),
        InlineKeyboardButton(text="⏹️", callback_data="stop"),
    ]
    
    seek_row = [
        InlineKeyboardButton(text="⏪ 10s", callback_data="seekback_10"),
        InlineKeyboardButton(text="⏩ 10s", callback_data="seek_10"),
    ]
    
    progress_button = InlineKeyboardButton(
        text=get_progress_bar_styled(0, total_duration), 
        callback_data="progress"
    )
    
    base_keyboard = InlineKeyboardMarkup([
        control_row,
        seek_row,
        [progress_button]
    ])

    thumb_url = song_info.get("thumbnail")
    try:
        progress_message = await message.reply_photo(
            photo=thumb_url,
            caption=base_caption,
            reply_markup=base_keyboard,
            parse_mode=ParseMode.HTML
        )
    except Exception:
        progress_message = await message.reply(
            base_caption,
            reply_markup=base_keyboard,
            parse_mode=ParseMode.HTML
        )

    await message.delete()

    # Start progress updates
    asyncio.create_task(
        update_progress_caption(
            chat_id,
            progress_message, 
            time.time(),
            total_duration,
            base_caption
        )
    )

# Helper functions
def safe_handler(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            chat_id = "Unknown"
            try:
                if len(args) >= 2:
                    chat_id = args[1].chat.id
                elif "message" in kwargs:
                    chat_id = kwargs["message"].chat.id
            except Exception:
                pass
            print(f"Error in {func.__name__} (chat {chat_id}): {e}")
    return wrapper

async def extract_invite_link(client, chat_id):
    try:
        chat_info = await client.get_chat(chat_id)
        if chat_info.invite_link:
            return chat_info.invite_link
        elif chat_info.username:
            return f"https://t.me/{chat_info.username}"
        return None
    except:
        return None

async def is_assistant_in_chat(chat_id):
    try:
        member = await assistant.get_chat_member(chat_id, ASSISTANT_USERNAME)
        return member.status is not None
    except Exception as e:
        if "USER_BANNED" in str(e):
            return "banned"
        elif "USER_NOT_PARTICIPANT" in str(e):
            return False
        return False

def iso8601_to_seconds(iso_duration):
    try:
        duration = isodate.parse_duration(iso_duration)
        return int(duration.total_seconds())
    except:
        return 0

def iso8601_to_human_readable(iso_duration):
    try:
        duration = isodate.parse_duration(iso_duration)
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02}:{seconds:02}"
        return f"{minutes}:{seconds:02}"
    except:
        return "Unknown"

async def invite_assistant(chat_id, invite_link, processing_message):
    try:
        await assistant.join_chat(invite_link)
        return True
    except UserAlreadyParticipant:
        return True
    except Exception as e:
        await processing_message.edit(f"❌ Assistant invite failed: {e}")
        return False

def to_bold_unicode(text: str) -> str:
    bold_text = ""
    for char in text:
        if 'A' <= char <= 'Z':
            bold_text += chr(ord('𝗔') + (ord(char) - ord('A')))
        elif 'a' <= char <= 'z':
            bold_text += chr(ord('𝗮') + (ord(char) - ord('a')))
        else:
            bold_text += char
    return bold_text

# Utility functions
MAX_TITLE_LEN = 20

def _one_line_title(full_title: str) -> str:
    if len(full_title) <= MAX_TITLE_LEN:
        return full_title
    else:
        return full_title[: (MAX_TITLE_LEN - 1) ] + "..."

def parse_duration_str(duration_str: str) -> int:
    try:
        duration = isodate.parse_duration(duration_str)
        return int(duration.total_seconds())
    except:
        if ':' in duration_str:
            try:
                parts = [int(x) for x in duration_str.split(':')]
                if len(parts) == 2:
                    minutes, seconds = parts
                    return minutes * 60 + seconds
                elif len(parts) == 3:
                    hours, minutes, seconds = parts
                    return hours * 3600 + minutes * 60 + seconds
            except:
                return 0
        return 0

def format_time(seconds: float) -> str:
    secs = int(seconds)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    else:
        return f"{m}:{s:02d}"

def get_progress_bar_styled(elapsed: float, total: float, bar_length: int = 14) -> str:
    if total <= 0:
        return "Progress: N/A"
    fraction = min(elapsed / total, 1)
    marker_index = int(fraction * bar_length)
    if marker_index >= bar_length:
        marker_index = bar_length - 1
    left = "━" * marker_index
    right = "─" * (bar_length - marker_index - 1)
    bar = left + "❄️" + right
    return f"{format_time(elapsed)} {bar} {format_time(total)}"

async def update_progress_caption(chat_id: int, progress_message: Message, start_time: float, total_duration: float, base_caption: str):
    while True:
        elapsed = time.time() - start_time
        if elapsed > total_duration:
            elapsed = total_duration
        progress_bar = get_progress_bar_styled(elapsed, total_duration)

        control_row = [
            InlineKeyboardButton(text="⏸️", callback_data="pause"),
            InlineKeyboardButton(text="▶️", callback_data="resume"),
            InlineKeyboardButton(text="⏭️", callback_data="skip"),
            InlineKeyboardButton(text="⏹️", callback_data="stop"),
        ]
        
        seek_row = [
            InlineKeyboardButton(text="⏪ 10s", callback_data="seekback_10"),
            InlineKeyboardButton(text="⏩ 10s", callback_data="seek_10"),
        ]
        
        progress_button = InlineKeyboardButton(text=progress_bar, callback_data="progress")

        new_keyboard = InlineKeyboardMarkup([
            control_row,
            seek_row,
            [progress_button]
        ])

        try:
            await bot.edit_message_caption(
                chat_id,
                progress_message.id,
                caption=base_caption,
                reply_markup=new_keyboard
            )
        except:
            pass

        if elapsed >= total_duration:
            break

        await asyncio.sleep(10)  # Reduced update frequency for performance

# BOT COMMAND HANDLERS
BOT_NAME = os.environ.get("BOT_NAME", "Frozen Music")
BOT_LINK = os.environ.get("BOT_LINK", "https://t.me/vcmusiclubot")

@bot.on_message(filters.command("start"))
@safe_handler
async def start_handler(_, message):
    user_id = message.from_user.id
    raw_name = message.from_user.first_name or ""
    styled_name = to_bold_unicode(raw_name)
    user_link = f"[{styled_name}](tg://user?id={user_id})"

    caption = (
        f"👋 нєу {user_link} 💠, 🥀\n\n"
        f">🎶 𝗪𝗘𝗟𝗖𝗢𝗠𝗘 𝗧𝗢 {BOT_NAME.upper()}! 🎵\n"
        ">🚀 𝗨𝗟𝗧𝗥𝗔-𝗙𝗔𝗦𝗧 𝗣𝗟𝗔𝗬𝗕𝗔𝗖𝗞 (1-3s)\n"
        ">🔊 𝗖𝗥𝗬𝗦𝗧𝗔𝗟-𝗖𝗟𝗘𝗔𝗥 𝗔𝗨𝗗𝗜𝗢\n"
        ">🎧 𝗦𝗨𝗣𝗣𝗢𝗥𝗧𝗘𝗗 𝗣𝗟𝗔𝗧𝗙𝗢𝗥𝗠𝗦: YouTube\n"
        ">⚡ 𝗜𝗡𝗦𝗧𝗔𝗡𝗧 𝗦𝗧𝗔𝗥𝗧 & 𝗦𝗘𝗘𝗞\n"
        ">🛠️ 𝗔𝗗𝗠𝗜𝗡 𝗖𝗢𝗠𝗠𝗔𝗡𝗗𝗦: Play, Pause, Skip, Stop, Seek\n"
    )

    buttons = [
        [
            InlineKeyboardButton("➕ Add Me", url=f"{BOT_LINK}?startgroup=true"),
            InlineKeyboardButton("🎵 Play Music", switch_inline_query_current_chat="")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(buttons)

    await message.reply_text(caption, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

# ULTRA-FAST PLAY COMMAND
@bot.on_message(filters.group & filters.regex(r'^/play(?:@\w+)?(?:\s+(?P<query>.+))?$'))
@safe_handler
async def play_handler(_, message: Message):
    chat_id = message.chat.id

    # Ultra-fast cooldown check
    now_ts = time.time()
    if chat_id in chat_last_command and (now_ts - chat_last_command[chat_id]) < COOLDOWN:
        return
    chat_last_command[chat_id] = now_ts

    match = message.matches[0]
    query = (match.group('query') or "").strip()

    if not query:
        await message.reply("❌ Usage: /play song_name")
        return

    try:
        await message.delete()
    except:
        pass

    # Start instant processing
    processing_message = await message.reply("⚡ INSTANT SEARCH...")

    # Ultra-fast search
    try:
        start_search = time.time()
        result = await ultra_fast_youtube_search(query)
        search_time = time.time() - start_search
        print(f"🔍 Search completed in {search_time:.2f}s")
        
    except Exception as e:
        await processing_message.edit(f"❌ Search failed: {e}")
        return

    # Process result
    if isinstance(result, dict) and "playlist" in result:
        await processing_message.edit("❌ Playlists not supported in instant mode")
        return
    else:
        video_url, title, duration_iso, thumb = result

    if not video_url:
        await processing_message.edit("❌ No results found")
        return

    # Check duration
    secs = iso8601_to_seconds(duration_iso)
    if secs > MAX_DURATION_SECONDS:
        await processing_message.edit("❌ Song too long")
        return

    # Add to queue and start INSTANT playback
    chat_containers.setdefault(chat_id, [])
    song_info = {
        "url": video_url,
        "title": title,
        "duration": iso8601_to_human_readable(duration_iso),
        "duration_seconds": secs,
        "requester": message.from_user.first_name if message.from_user else "Unknown",
        "thumbnail": thumb
    }

    # If first song, start INSTANT playback
    if len(chat_containers[chat_id]) == 0:
        chat_containers[chat_id].append(song_info)
        await instant_playback(chat_id, processing_message, song_info)
    else:
        chat_containers[chat_id].append(song_info)
        await processing_message.edit(f"🎵 Added to queue: {title}")

# SEEK COMMANDS
@bot.on_message(filters.group & filters.command("seek"))
@safe_handler
async def seek_forward_handler(client, message):
    chat_id = message.chat.id

    if chat_id not in current_playback_position:
        await message.reply("❌ No song playing")
        return

    try:
        parts = message.text.split()
        if len(parts) < 2:
            await message.reply("❌ Usage: /seek seconds")
            return
        
        seek_seconds = int(parts[1])
        current_info = current_playback_position[chat_id]
        song_info = current_info['song_info']
        current_time = time.time() - current_info['start_time']
        new_position = current_time + seek_seconds
        
        total_duration = parse_duration_str(song_info.get("duration", "0:00"))
        if new_position >= total_duration:
            await message.reply("❌ Cannot seek beyond end")
            return

        # Fast seek
        await call_py.leave_call(chat_id)
        await asyncio.sleep(0.5)
        
        await call_py.play(chat_id, MediaStream(song_info['url'], video_flags=MediaStream.Flags.IGNORE))
        
        current_playback_position[chat_id] = {
            'start_time': time.time() - new_position,
            'song_info': song_info
        }
        
        await message.reply(f"⏩ Seeked +{seek_seconds}s")
        
    except Exception as e:
        await message.reply(f"❌ Seek failed: {e}")

@bot.on_message(filters.group & filters.command("seekback"))
@safe_handler
async def seek_backward_handler(client, message):
    chat_id = message.chat.id

    if chat_id not in current_playback_position:
        await message.reply("❌ No song playing")
        return

    try:
        parts = message.text.split()
        if len(parts) < 2:
            await message.reply("❌ Usage: /seekback seconds")
            return
        
        seek_seconds = int(parts[1])
        current_info = current_playback_position[chat_id]
        song_info = current_info['song_info']
        current_time = time.time() - current_info['start_time']
        new_position = max(0, current_time - seek_seconds)

        # Fast seek
        await call_py.leave_call(chat_id)
        await asyncio.sleep(0.5)
        
        await call_py.play(chat_id, MediaStream(song_info['url'], video_flags=MediaStream.Flags.IGNORE))
        
        current_playback_position[chat_id] = {
            'start_time': time.time() - new_position,
            'song_info': song_info
        }
        
        await message.reply(f"⏪ Seeked -{seek_seconds}s")
        
    except Exception as e:
        await message.reply(f"❌ Seek failed: {e}")

# OTHER COMMANDS
@bot.on_message(filters.group & filters.command(["stop", "end"]))
@safe_handler
async def stop_handler(client, message):
    chat_id = message.chat.id
    try:
        await call_py.leave_call(chat_id)
        if chat_id in chat_containers:
            chat_containers.pop(chat_id)
        if chat_id in playback_tasks:
            playback_tasks[chat_id].cancel()
            del playback_tasks[chat_id]
        await message.reply("⏹ Stopped")
    except Exception as e:
        await message.reply(f"❌ Stop failed: {e}")

@bot.on_message(filters.group & filters.command("skip"))
@safe_handler
async def skip_handler(client, message):
    chat_id = message.chat.id

    if chat_id not in chat_containers or not chat_containers[chat_id]:
        await message.reply("❌ No songs in queue")
        return

    skipped_song = chat_containers[chat_id].pop(0)
    await call_py.leave_call(chat_id)
    await asyncio.sleep(1)

    if chat_containers.get(chat_id):
        next_song = chat_containers[chat_id][0]
        dummy_msg = await bot.send_message(chat_id, f"🎧 Next: {next_song['title']}")
        await instant_playback(chat_id, dummy_msg, next_song)
    else:
        await message.reply("⏩ Skipped - Queue empty")

@bot.on_message(filters.group & filters.command("pause"))
@safe_handler
async def pause_handler(client, message):
    chat_id = message.chat.id
    try:
        await call_py.pause(chat_id)
        await message.reply("⏸ Paused")
    except Exception as e:
        await message.reply(f"❌ Pause failed: {e}")

@bot.on_message(filters.group & filters.command("resume"))
@safe_handler
async def resume_handler(client, message):
    chat_id = message.chat.id
    try:
        await call_py.resume(chat_id)
        await message.reply("▶️ Resumed")
    except Exception as e:
        await message.reply(f"❌ Resume failed: {e}")

# CALLBACK QUERIES
@bot.on_callback_query()
@safe_handler
async def callback_query_handler(client, callback_query):
    chat_id = callback_query.message.chat.id
    data = callback_query.data

    if data == "pause":
        try:
            await call_py.pause(chat_id)
            await callback_query.answer("⏸ Paused")
        except:
            await callback_query.answer("❌ Pause failed")

    elif data == "resume":
        try:
            await call_py.resume(chat_id)
            await callback_query.answer("▶️ Resumed")
        except:
            await callback_query.answer("❌ Resume failed")

    elif data == "skip":
        if chat_id in chat_containers and chat_containers[chat_id]:
            chat_containers[chat_id].pop(0)
            await call_py.leave_call(chat_id)
            await asyncio.sleep(1)
            
            if chat_containers.get(chat_id):
                next_song = chat_containers[chat_id][0]
                dummy_msg = await bot.send_message(chat_id, "🎧 Next song...")
                await instant_playback(chat_id, dummy_msg, next_song)
                await callback_query.answer("⏭ Skipped")
            else:
                await callback_query.answer("⏭ Skipped - Queue empty")
        else:
            await callback_query.answer("❌ No songs to skip")

    elif data == "stop":
        try:
            await call_py.leave_call(chat_id)
            if chat_id in chat_containers:
                chat_containers.pop(chat_id)
            await callback_query.answer("⏹ Stopped")
        except:
            await callback_query.answer("❌ Stop failed")

    elif data == "seek_10":
        if chat_id in current_playback_position:
            current_info = current_playback_position[chat_id]
            new_position = (time.time() - current_info['start_time']) + 10
            
            await call_py.leave_call(chat_id)
            await asyncio.sleep(0.3)
            await call_py.play(chat_id, MediaStream(current_info['song_info']['url']))
            
            current_playback_position[chat_id] = {
                'start_time': time.time() - new_position,
                'song_info': current_info['song_info']
            }
            await callback_query.answer("⏩ +10s")
        else:
            await callback_query.answer("❌ No song playing")

    elif data == "seekback_10":
        if chat_id in current_playback_position:
            current_info = current_playback_position[chat_id]
            new_position = max(0, (time.time() - current_info['start_time']) - 10)
            
            await call_py.leave_call(chat_id)
            await asyncio.sleep(0.3)
            await call_py.play(chat_id, MediaStream(current_info['song_info']['url']))
            
            current_playback_position[chat_id] = {
                'start_time': time.time() - new_position,
                'song_info': current_info['song_info']
            }
            await callback_query.answer("⏪ -10s")
        else:
            await callback_query.answer("❌ No song playing")

# STREAM END HANDLER
@call_py.on_update(fl.stream_end())
async def stream_end_handler(_: PyTgCalls, update: StreamEnded):
    chat_id = update.chat_id

    if chat_id in chat_containers and chat_containers[chat_id]:
        chat_containers[chat_id].pop(0)
        await asyncio.sleep(1)

        if chat_id in chat_containers and chat_containers[chat_id]:
            next_song = chat_containers[chat_id][0]
            dummy_msg = await bot.send_message(chat_id, "🎧 Auto-next...")
            await instant_playback(chat_id, dummy_msg, next_song)

# Flask app for health checks
app = Flask(__name__)

@app.route('/')
def home():
    return "🤖 ULTRA-FAST Music Bot is running!"

@app.route('/health')
def health():
    return {"status": "healthy", "speed": "instant"}

def run_flask_app():
    app.run(host='0.0.0.0', port=5000, debug=False)

# ─── MAIN ENTRY ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Start Flask server
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    logger.info("✅ Flask server started")
    
    logger.info("🚀 Starting ULTRA-FAST Music Bot...")
    call_py.start()
    logger.info("✅ PyTgCalls started")

    try:
        bot.start()
        logger.info("✅ Bot started")
    except Exception as e:
        logger.error(f"❌ Bot start failed: {e}")
        sys.exit(1)

    # Get bot info
    me = bot.get_me()
    BOT_NAME = me.first_name or "Frozen Music"
    BOT_USERNAME = me.username or "musicbot"
    logger.info(f"✅ Bot: {BOT_NAME} (@{BOT_USERNAME})")

    # Start assistant
    if not assistant.is_connected:
        assistant.start()
        logger.info("✅ Assistant started")

    try:
        assistant_user = assistant.get_me()
        ASSISTANT_USERNAME = assistant_user.username
        ASSISTANT_CHAT_ID = assistant_user.id
        logger.info(f"✅ Assistant: @{ASSISTANT_USERNAME}")
    except Exception as e:
        logger.error(f"❌ Assistant info failed: {e}")

    logger.info("🎵 BOT READY - INSTANT PLAYBACK ENABLED!")
    idle()