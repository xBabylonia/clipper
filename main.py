#!/usr/bin/env python3
"""
============================================================
🎬 TELEGRAM VIDEO CLIPPER BOT
============================================================
Automatic video clipping bot that converts long-form videos
into viral short-form content (TikTok, Reels, Shorts).

Single-file production-ready implementation.

Author : xBabylonia
License: MIT
============================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ────────────────────────────────────────────────────────────
# LOGGING SETUP
# ────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "./logs/bot.log")

Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("VideoClipperBot")


# ────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Central configuration loaded from environment variables."""

    # Telegram
    bot_token: str = ""
    allowed_user_ids: List[int] = field(default_factory=list)
    admin_user_ids: List[int] = field(default_factory=list)

    # Processing defaults
    max_video_duration: int = 7200  # seconds
    max_file_size_mb: int = 2000
    default_clip_duration: int = 60
    default_num_clips: int = 5
    default_output_format: str = "9:16"
    default_quality_preset: str = "high"
    default_fps: int = 30

    # Whisper
    whisper_model: str = "base"
    whisper_device: str = "cpu"

    # Storage
    temp_dir: str = "./tmp"
    output_dir: str = "./output"
    db_path: str = "./data/bot.db"
    max_temp_size_gb: int = 10
    auto_cleanup_hours: int = 24

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        def parse_int_list(raw: str) -> List[int]:
            if not raw or not raw.strip():
                return []
            return [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]

        cfg = cls(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            allowed_user_ids=parse_int_list(os.getenv("ALLOWED_USER_IDS", "")),
            admin_user_ids=parse_int_list(os.getenv("ADMIN_USER_IDS", "")),
            max_video_duration=int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "7200")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "2000")),
            default_clip_duration=int(os.getenv("DEFAULT_CLIP_DURATION", "60")),
            default_num_clips=int(os.getenv("DEFAULT_NUM_CLIPS", "5")),
            default_output_format=os.getenv("DEFAULT_OUTPUT_FORMAT", "9:16"),
            default_quality_preset=os.getenv("DEFAULT_QUALITY_PRESET", "high"),
            default_fps=int(os.getenv("DEFAULT_FPS", "30")),
            whisper_model=os.getenv("WHISPER_MODEL", "base"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            temp_dir=os.getenv("TEMP_DIR", "./tmp"),
            output_dir=os.getenv("OUTPUT_DIR", "./output"),
            db_path=os.getenv("DB_PATH", "./data/bot.db"),
            max_temp_size_gb=int(os.getenv("MAX_TEMP_SIZE_GB", "10")),
            auto_cleanup_hours=int(os.getenv("AUTO_CLEANUP_HOURS", "24")),
        )

        if not cfg.bot_token:
            logger.error("TELEGRAM_BOT_TOKEN is not set!")
            sys.exit(1)

        # Ensure dirs exist
        for d in [cfg.temp_dir, cfg.output_dir, str(Path(cfg.db_path).parent)]:
            Path(d).mkdir(parents=True, exist_ok=True)

        return cfg


config = Config.from_env()


# ────────────────────────────────────────────────────────────
# ENUMS & DATA CLASSES
# ────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    DETECTING_SCENES = "detecting_scenes"
    CLIPPING = "clipping"
    CROPPING = "cropping"
    GENERATING_SUBTITLES = "generating_subtitles"
    BURNING_SUBTITLES = "burning_subtitles"
    PROCESSING_AUDIO = "processing_audio"
    GENERATING_THUMBNAILS = "generating_thumbnails"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OutputFormat(str, Enum):
    VERTICAL = "9:16"
    HORIZONTAL = "16:9"
    SQUARE = "1:1"


class QualityPreset(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


QUALITY_MAP = {
    QualityPreset.LOW: {"crf": "28", "preset": "fast", "bitrate": "1M"},
    QualityPreset.MEDIUM: {"crf": "23", "preset": "medium", "bitrate": "3M"},
    QualityPreset.HIGH: {"crf": "18", "preset": "slow", "bitrate": "6M"},
    QualityPreset.ULTRA: {"crf": "15", "preset": "slower", "bitrate": "10M"},
}

FORMAT_RESOLUTIONS = {
    OutputFormat.VERTICAL: (1080, 1920),
    OutputFormat.HORIZONTAL: (1920, 1080),
    OutputFormat.SQUARE: (1080, 1080),
}


@dataclass
class ProcessingJob:
    """Represents a single video processing job."""
    job_id: str = ""
    user_id: int = 0
    chat_id: int = 0
    message_id: int = 0
    source_type: str = ""  # "file" or "url"
    source_path: str = ""
    source_url: str = ""
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    clip_duration: int = 60
    num_clips: int = 5
    output_format: OutputFormat = OutputFormat.VERTICAL
    quality_preset: QualityPreset = QualityPreset.HIGH
    fps: int = 30
    subtitle_enabled: bool = True
    subtitle_lang: str = "auto"
    subtitle_style: str = "viral"
    remove_silence: bool = True
    normalize_audio: bool = True
    smart_crop: bool = True
    generate_thumbnails: bool = True
    output_clips: List[str] = field(default_factory=list)
    output_thumbnails: List[str] = field(default_factory=list)
    output_subtitles: List[str] = field(default_factory=list)
    error_message: str = ""
    created_at: str = ""
    completed_at: str = ""
    virality_scores: List[float] = field(default_factory=list)
    timestamps: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.job_id:
            self.job_id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class UserSettings:
    """Per-user preferences stored in DB."""
    user_id: int = 0
    clip_duration: int = 60
    num_clips: int = 5
    output_format: str = "9:16"
    quality_preset: str = "high"
    subtitle_enabled: bool = True
    subtitle_lang: str = "auto"
    subtitle_style: str = "viral"
    subtitle_font_size: int = 24
    subtitle_font_color: str = "#FFFFFF"
    subtitle_bg_color: str = "#000000AA"
    subtitle_position: str = "bottom"
    remove_silence: bool = True
    normalize_audio: bool = True
    smart_crop: bool = True


# ────────────────────────────────────────────────────────────
# DATABASE
# ────────────────────────────────────────────────────────────

class Database:
    """SQLite database for jobs, user settings, and whitelist."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    source_type TEXT,
                    source_url TEXT,
                    clip_duration INTEGER DEFAULT 60,
                    num_clips INTEGER DEFAULT 5,
                    output_format TEXT DEFAULT '9:16',
                    quality_preset TEXT DEFAULT 'high',
                    subtitle_enabled INTEGER DEFAULT 1,
                    subtitle_lang TEXT DEFAULT 'auto',
                    num_output_clips INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT,
                    completed_at TEXT,
                    metadata TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    settings_json TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS whitelist (
                    user_id INTEGER PRIMARY KEY,
                    added_by INTEGER,
                    added_at TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clips (
                    clip_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    file_path TEXT,
                    telegram_file_id TEXT,
                    thumbnail_path TEXT,
                    subtitle_path TEXT,
                    duration REAL,
                    virality_score REAL DEFAULT 0.0,
                    created_at TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                )
            """)

        # Seed whitelist from config
        for uid in config.allowed_user_ids:
            self.add_to_whitelist(uid, added_by=0)

    def is_whitelisted(self, user_id: int) -> bool:
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM whitelist WHERE user_id=?", (user_id,))
            return cur.fetchone() is not None

    def add_to_whitelist(self, user_id: int, added_by: int = 0):
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO whitelist (user_id, added_by, added_at) VALUES (?, ?, ?)",
                (user_id, added_by, datetime.now().isoformat()),
            )

    def remove_from_whitelist(self, user_id: int):
        with self._cursor() as cur:
            cur.execute("DELETE FROM whitelist WHERE user_id=?", (user_id,))

    def get_whitelist(self) -> List[int]:
        with self._cursor() as cur:
            cur.execute("SELECT user_id FROM whitelist")
            return [row["user_id"] for row in cur.fetchall()]

    def save_job(self, job: ProcessingJob):
        metadata = json.dumps({
            "virality_scores": job.virality_scores,
            "timestamps": job.timestamps,
            "output_clips": job.output_clips,
            "output_thumbnails": job.output_thumbnails,
            "output_subtitles": job.output_subtitles,
        })
        with self._cursor() as cur:
            cur.execute("""
                INSERT OR REPLACE INTO jobs
                (job_id, user_id, chat_id, status, source_type, source_url,
                 clip_duration, num_clips, output_format, quality_preset,
                 subtitle_enabled, subtitle_lang, num_output_clips,
                 error_message, created_at, completed_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id, job.user_id, job.chat_id, job.status.value,
                job.source_type, job.source_url, job.clip_duration,
                job.num_clips, job.output_format.value, job.quality_preset.value,
                int(job.subtitle_enabled), job.subtitle_lang,
                len(job.output_clips), job.error_message,
                job.created_at, job.completed_at, metadata,
            ))

    def get_job(self, job_id: str) -> Optional[Dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_user_jobs(self, user_id: int, limit: int = 10) -> List[Dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_active_jobs(self, user_id: int) -> List[Dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM jobs WHERE user_id=? AND status NOT IN ('completed','failed','cancelled')",
                (user_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def save_clip(self, clip_id: str, job_id: str, user_id: int,
                  file_path: str = "", telegram_file_id: str = "",
                  thumbnail_path: str = "", subtitle_path: str = "",
                  duration: float = 0.0, virality_score: float = 0.0):
        with self._cursor() as cur:
            cur.execute("""
                INSERT OR REPLACE INTO clips
                (clip_id, job_id, user_id, file_path, telegram_file_id,
                 thumbnail_path, subtitle_path, duration, virality_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                clip_id, job_id, user_id, file_path, telegram_file_id,
                thumbnail_path, subtitle_path, duration, virality_score,
                datetime.now().isoformat(),
            ))

    def get_clip(self, clip_id: str) -> Optional[Dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM clips WHERE clip_id=?", (clip_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_user_clips(self, user_id: int, limit: int = 20) -> List[Dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM clips WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def save_user_settings(self, settings: UserSettings):
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO user_settings (user_id, settings_json) VALUES (?, ?)",
                (settings.user_id, json.dumps(asdict(settings))),
            )

    def get_user_settings(self, user_id: int) -> UserSettings:
        with self._cursor() as cur:
            cur.execute("SELECT settings_json FROM user_settings WHERE user_id=?", (user_id,))
            row = cur.fetchone()
            if row:
                data = json.loads(row["settings_json"])
                return UserSettings(**data)
            return UserSettings(user_id=user_id)


db = Database(config.db_path)


# ────────────────────────────────────────────────────────────
# UTILITIES
# ────────────────────────────────────────────────────────────

def check_ffmpeg() -> bool:
    """Verify FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_ffprobe() -> bool:
    """Verify FFprobe is installed and accessible."""
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_video_info(filepath: str) -> Dict[str, Any]:
    """Get video metadata using FFprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        filepath
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)

        video_stream = None
        audio_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = stream

        duration = float(data.get("format", {}).get("duration", 0))
        width = int(video_stream.get("width", 0)) if video_stream else 0
        height = int(video_stream.get("height", 0)) if video_stream else 0
        fps_str = video_stream.get("r_frame_rate", "30/1") if video_stream else "30/1"

        # Parse fps fraction
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 30.0
        else:
            fps = float(fps_str)

        return {
            "duration": duration,
            "width": width,
            "height": height,
            "fps": fps,
            "has_audio": audio_stream is not None,
            "codec": video_stream.get("codec_name", "unknown") if video_stream else "unknown",
            "size_mb": float(data.get("format", {}).get("size", 0)) / (1024 * 1024),
            "bitrate": int(data.get("format", {}).get("bit_rate", 0)),
        }
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {"duration": 0, "width": 0, "height": 0, "fps": 30, "has_audio": False}


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_disk_usage(path: str) -> float:
    """Get disk usage of a directory in GB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)


def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """Remove files older than max_age_hours."""
    cutoff = time.time() - (max_age_hours * 3600)
    count = 0
    for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                if os.path.getmtime(fp) < cutoff:
                    os.remove(fp)
                    count += 1
            except OSError:
                pass
        # Remove empty directories
        for d in dirnames:
            dp = os.path.join(dirpath, d)
            try:
                os.rmdir(dp)
            except OSError:
                pass
    if count > 0:
        logger.info(f"Cleaned up {count} old files from {directory}")


def is_url(text: str) -> bool:
    """Check if text is a URL."""
    url_pattern = re.compile(
        r"https?://(www\.)?"
        r"(youtube\.com|youtu\.be|instagram\.com|tiktok\.com|"
        r"twitter\.com|x\.com|facebook\.com|fb\.watch|"
        r"vimeo\.com|dailymotion\.com|twitch\.tv)"
        r"/\S+"
    )
    return bool(url_pattern.match(text.strip()))


def sanitize_filename(name: str) -> str:
    """Remove unsafe characters from filenames."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)[:100]


# ────────────────────────────────────────────────────────────
# VIDEO DOWNLOADER (yt-dlp)
# ────────────────────────────────────────────────────────────

class VideoDownloader:
    """Download videos from URLs using yt-dlp."""

    @staticmethod
    async def download(url: str, output_dir: str,
                       progress_callback=None) -> Optional[str]:
        """
        Download a video from URL.
        Returns the path to the downloaded file or None on failure.
        """
        output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--merge-output-format", "mp4",
            "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "--max-filesize", f"{config.max_file_size_mb}M",
            "-o", output_template,
            "--no-warnings",
            "--print", "after_move:filepath",
            url,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=600
            )

            if process.returncode != 0:
                error_msg = stderr.decode().strip()
                logger.error(f"yt-dlp failed: {error_msg}")
                return None

            filepath = stdout.decode().strip().split("\n")[-1]
            if os.path.exists(filepath):
                logger.info(f"Downloaded: {filepath}")
                if progress_callback:
                    await progress_callback(100.0, "Download complete")
                return filepath

            # Fallback: find the file in output_dir
            for f in os.listdir(output_dir):
                fp = os.path.join(output_dir, f)
                if os.path.isfile(fp) and f.endswith((".mp4", ".mkv", ".webm")):
                    return fp

            return None

        except asyncio.TimeoutError:
            logger.error("Download timed out (10 min)")
            return None
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None


# ────────────────────────────────────────────────────────────
# SCENE DETECTOR
# ────────────────────────────────────────────────────────────

class SceneDetector:
    """
    Detect scene changes and highlight moments in video.
    Uses FFmpeg scene detection filter + audio energy analysis.
    """

    @staticmethod
    async def detect_scenes(filepath: str, threshold: float = 0.3) -> List[float]:
        """
        Detect scene change timestamps using FFmpeg.
        Returns list of timestamps (seconds) where scenes change.
        """
        cmd = [
            "ffmpeg", "-i", filepath,
            "-vf", f"select='gt(scene,{threshold})',showinfo",
            "-vsync", "vfr",
            "-f", "null", "-"
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                process.communicate(), timeout=300
            )

            timestamps = []
            for line in stderr.decode().split("\n"):
                if "pts_time:" in line:
                    match = re.search(r"pts_time:(\d+\.?\d*)", line)
                    if match:
                        timestamps.append(float(match.group(1)))

            logger.info(f"Detected {len(timestamps)} scene changes")
            return timestamps

        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return []

    @staticmethod
    async def detect_audio_peaks(filepath: str, threshold_db: float = -20.0) -> List[float]:
        """
        Detect loud audio moments (highlights) using FFmpeg.
        Returns timestamps of audio peaks.
        """
        cmd = [
            "ffmpeg", "-i", filepath,
            "-af", f"silencedetect=n={threshold_db}dB:d=0.5",
            "-f", "null", "-"
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                process.communicate(), timeout=300
            )

            # Parse silence_end timestamps (= where noise starts = highlights)
            peaks = []
            for line in stderr.decode().split("\n"):
                if "silence_end" in line:
                    match = re.search(r"silence_end:\s*(\d+\.?\d*)", line)
                    if match:
                        peaks.append(float(match.group(1)))

            logger.info(f"Detected {len(peaks)} audio peaks")
            return peaks

        except Exception as e:
            logger.error(f"Audio peak detection failed: {e}")
            return []

    @classmethod
    async def find_best_clips(
        cls,
        filepath: str,
        clip_duration: int = 60,
        num_clips: int = 5,
        total_duration: float = 0,
    ) -> List[Tuple[float, float]]:
        """
        Find the best clip timestamps combining scene changes and audio peaks.
        Returns list of (start, end) tuples.
        """
        if total_duration <= 0:
            info = get_video_info(filepath)
            total_duration = info["duration"]

        if total_duration <= clip_duration:
            return [(0, total_duration)]

        scenes = await cls.detect_scenes(filepath)
        peaks = await cls.detect_audio_peaks(filepath)

        # Score each potential starting point
        candidates: Dict[int, float] = {}
        step = max(1, int(clip_duration * 0.25))  # 25% overlap steps

        for start_sec in range(0, int(total_duration - clip_duration), step):
            end_sec = start_sec + clip_duration
            score = 0.0

            # Scenes within this range = more visual variety
            scenes_in_range = [s for s in scenes if start_sec <= s <= end_sec]
            score += len(scenes_in_range) * 2.0

            # Audio peaks within range = more energetic
            peaks_in_range = [p for p in peaks if start_sec <= p <= end_sec]
            score += len(peaks_in_range) * 3.0

            # Bonus: strong opening (peak in first 3 seconds)
            early_peaks = [p for p in peaks_in_range if p < start_sec + 3]
            score += len(early_peaks) * 5.0

            # Bonus: scene change near start = good entry point
            early_scenes = [s for s in scenes_in_range if s < start_sec + 2]
            score += len(early_scenes) * 3.0

            candidates[start_sec] = score

        if not candidates:
            # Fallback: evenly split the video
            interval = total_duration / num_clips
            return [
                (i * interval, min((i + 1) * interval, total_duration))
                for i in range(num_clips)
            ]

        # Sort by score and pick top N non-overlapping
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        selected: List[Tuple[float, float]] = []
        for start, score in sorted_candidates:
            if len(selected) >= num_clips:
                break
            end = start + clip_duration
            # Check overlap with already selected
            overlaps = any(
                not (end <= s[0] or start >= s[1])
                for s in selected
            )
            if not overlaps:
                selected.append((float(start), float(min(end, total_duration))))

        # Sort by start time
        selected.sort(key=lambda x: x[0])

        # If we still don't have enough, fill with evenly spaced clips
        if len(selected) < num_clips:
            interval = total_duration / num_clips
            for i in range(num_clips):
                s = i * interval
                e = min(s + clip_duration, total_duration)
                overlaps = any(
                    not (e <= sel[0] or s >= sel[1])
                    for sel in selected
                )
                if not overlaps and len(selected) < num_clips:
                    selected.append((s, e))
            selected.sort(key=lambda x: x[0])

        logger.info(f"Selected {len(selected)} clip segments")
        return selected[:num_clips]


# ────────────────────────────────────────────────────────────
# FACE TRACKER / SMART CROP
# ────────────────────────────────────────────────────────────

class FaceTracker:
    """
    Detect face positions for smart cropping.
    Uses FFmpeg cropdetect + face detection if OpenCV is available.
    """

    @staticmethod
    async def detect_face_region(
        filepath: str, sample_time: float = 5.0
    ) -> Optional[Tuple[int, int]]:
        """
        Detect the primary face/subject position at a given time.
        Returns (center_x, center_y) or None.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.warning("OpenCV not available, using center crop")
            return None

        try:
            # Extract a frame at sample_time
            cap = cv2.VideoCapture(filepath)
            cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return None

            # Try face detection
            cascade_paths = [
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
                cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
                cv2.data.haarcascades + "haarcascade_profileface.xml",
            ]

            for cascade_path in cascade_paths:
                if not os.path.exists(cascade_path):
                    continue
                face_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )

                if len(faces) > 0:
                    # Pick the largest face
                    largest = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest
                    center_x = x + w // 2
                    center_y = y + h // 2
                    logger.info(f"Face detected at ({center_x}, {center_y})")
                    return (center_x, center_y)

            return None

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return None

    @classmethod
    async def get_crop_params(
        cls,
        filepath: str,
        source_w: int,
        source_h: int,
        target_format: OutputFormat,
        sample_time: float = 5.0,
    ) -> Tuple[int, int, int, int]:
        """
        Calculate crop parameters (x, y, w, h) for smart cropping.
        Tries face detection first, falls back to center crop.
        """
        target_w, target_h = FORMAT_RESOLUTIONS[target_format]
        target_ratio = target_w / target_h
        source_ratio = source_w / source_h

        if abs(source_ratio - target_ratio) < 0.01:
            return (0, 0, source_w, source_h)

        # Calculate crop dimensions maintaining target ratio
        if source_ratio > target_ratio:
            # Source is wider — crop width
            crop_h = source_h
            crop_w = int(source_h * target_ratio)
        else:
            # Source is taller — crop height
            crop_w = source_w
            crop_h = int(source_w / target_ratio)

        # Try face detection for positioning
        face_pos = await cls.detect_face_region(filepath, sample_time)

        if face_pos:
            face_x, face_y = face_pos
            # Center crop on face
            crop_x = max(0, min(face_x - crop_w // 2, source_w - crop_w))
            crop_y = max(0, min(face_y - crop_h // 2, source_h - crop_h))
        else:
            # Center crop
            crop_x = (source_w - crop_w) // 2
            crop_y = (source_h - crop_h) // 2

        return (crop_x, crop_y, crop_w, crop_h)


# ────────────────────────────────────────────────────────────
# SUBTITLE GENERATOR
# ────────────────────────────────────────────────────────────

class SubtitleGenerator:
    """Generate subtitles using OpenAI Whisper (faster-whisper)."""

    @staticmethod
    async def generate(
        filepath: str,
        output_srt: str,
        language: str = "auto",
        model_name: str = "",
    ) -> Optional[str]:
        """
        Generate SRT subtitle file from audio.
        Returns path to SRT file or None on failure.
        """
        if not model_name:
            model_name = config.whisper_model

        try:
            # Try faster-whisper first, fall back to openai-whisper
            return await SubtitleGenerator._generate_faster_whisper(
                filepath, output_srt, language, model_name
            )
        except ImportError:
            logger.info("faster-whisper not found, trying openai-whisper")
            try:
                return await SubtitleGenerator._generate_openai_whisper(
                    filepath, output_srt, language, model_name
                )
            except ImportError:
                logger.warning("No whisper library available, skipping subtitles")
                return None

    @staticmethod
    async def _generate_faster_whisper(
        filepath: str, output_srt: str, language: str, model_name: str
    ) -> Optional[str]:
        """Generate subtitles using faster-whisper."""
        from faster_whisper import WhisperModel

        def _run():
            device = config.whisper_device
            compute_type = "float16" if device == "cuda" else "int8"
            model = WhisperModel(model_name, device=device, compute_type=compute_type)

            lang = None if language == "auto" else language
            segments, info = model.transcribe(
                filepath,
                language=lang,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            srt_lines = []
            idx = 1
            for segment in segments:
                start = segment.start
                end = segment.end
                text = segment.text.strip()
                if not text:
                    continue

                start_ts = _seconds_to_srt_time(start)
                end_ts = _seconds_to_srt_time(end)
                srt_lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")
                idx += 1

            with open(output_srt, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_lines))

            return output_srt

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run)

    @staticmethod
    async def _generate_openai_whisper(
        filepath: str, output_srt: str, language: str, model_name: str
    ) -> Optional[str]:
        """Generate subtitles using openai-whisper."""
        import whisper

        def _run():
            model = whisper.load_model(model_name, device=config.whisper_device)
            result = model.transcribe(
                filepath,
                language=None if language == "auto" else language,
                verbose=False,
            )

            srt_lines = []
            idx = 1
            for segment in result.get("segments", []):
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip()
                if not text:
                    continue

                start_ts = _seconds_to_srt_time(start)
                end_ts = _seconds_to_srt_time(end)
                srt_lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")
                idx += 1

            with open(output_srt, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_lines))

            return output_srt

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run)

    @staticmethod
    def get_subtitle_filter(
        srt_path: str,
        style: str = "viral",
        font_size: int = 24,
        font_color: str = "white",
        bg_color: str = "black@0.6",
        position: str = "bottom",
    ) -> str:
        """
        Build FFmpeg subtitle filter string.
        Returns the FFmpeg -vf subtitles filter.
        """
        # Escape special chars in path for FFmpeg
        escaped_path = srt_path.replace("\\", "/").replace(":", "\\:")
        escaped_path = escaped_path.replace("'", "\\'")

        y_positions = {
            "top": "10",
            "center": "(h-text_h)/2",
            "bottom": "h-th-40",
        }
        margin_v = y_positions.get(position, "h-th-40")

        styles = {
            "viral": (
                f"FontSize={font_size},FontName=Arial Black,"
                f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                f"BackColour=&H80000000,Outline=2,Shadow=1,"
                f"MarginV=40,Alignment=2,Bold=1"
            ),
            "minimal": (
                f"FontSize={font_size - 2},FontName=Arial,"
                f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                f"Outline=1,Shadow=0,MarginV=30,Alignment=2"
            ),
            "bold": (
                f"FontSize={font_size + 4},FontName=Impact,"
                f"PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,"
                f"Outline=3,Shadow=2,MarginV=50,Alignment=2,Bold=1"
            ),
            "karaoke": (
                f"FontSize={font_size + 2},FontName=Arial Black,"
                f"PrimaryColour=&H0000FFFF,SecondaryColour=&H00FFFFFF,"
                f"OutlineColour=&H00000000,Outline=2,Shadow=1,"
                f"MarginV=40,Alignment=2,Bold=1"
            ),
        }

        force_style = styles.get(style, styles["viral"])
        return f"subtitles='{escaped_path}':force_style='{force_style}'"


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ────────────────────────────────────────────────────────────
# AUDIO PROCESSOR
# ────────────────────────────────────────────────────────────

class AudioProcessor:
    """Audio processing: silence removal, normalization, music mixing."""

    @staticmethod
    async def remove_silence(
        input_path: str, output_path: str,
        threshold_db: float = -30.0, min_silence_ms: int = 700
    ) -> Optional[str]:
        """Remove silent segments from video."""
        min_silence_sec = min_silence_ms / 1000.0
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", (
                f"silenceremove=stop_periods=-1"
                f":stop_duration={min_silence_sec}"
                f":stop_threshold={threshold_db}dB"
            ),
            "-c:v", "copy",
            output_path,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            if process.returncode == 0 and os.path.exists(output_path):
                return output_path
            logger.error(f"Silence removal failed: {stderr.decode()[-500:]}")
            return None
        except Exception as e:
            logger.error(f"Silence removal error: {e}")
            return None

    @staticmethod
    async def normalize_audio(input_path: str, output_path: str) -> Optional[str]:
        """Normalize audio levels using loudnorm filter."""
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:v", "copy",
            output_path,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            if process.returncode == 0 and os.path.exists(output_path):
                return output_path
            logger.error(f"Audio normalization failed: {stderr.decode()[-500:]}")
            return None
        except Exception as e:
            logger.error(f"Audio normalization error: {e}")
            return None

    @staticmethod
    async def mix_background_music(
        video_path: str, music_path: str, output_path: str,
        music_volume: float = 0.15
    ) -> Optional[str]:
        """Mix background music with video audio."""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", music_path,
            "-filter_complex",
            f"[1:a]volume={music_volume},aloop=loop=-1:size=2e+09[bg];"
            f"[0:a][bg]amix=inputs=2:duration=shortest:dropout_transition=2[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-shortest",
            output_path,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            if process.returncode == 0 and os.path.exists(output_path):
                return output_path
            logger.error(f"Music mixing failed: {stderr.decode()[-500:]}")
            return None
        except Exception as e:
            logger.error(f"Music mixing error: {e}")
            return None


# ────────────────────────────────────────────────────────────
# THUMBNAIL GENERATOR
# ─────────────────────────────────────���──────────────────────

class ThumbnailGenerator:
    """Generate thumbnails from video clips."""

    @staticmethod
    async def generate(
        filepath: str, output_path: str,
        timestamp: float = -1, count: int = 1
    ) -> List[str]:
        """
        Extract thumbnail frames from video.
        If timestamp < 0, auto-select the best frame.
        """
        thumbnails = []

        if timestamp < 0:
            info = get_video_info(filepath)
            duration = info.get("duration", 10)
            # Sample at 10%, 30%, 50% of duration for "best" frame
            sample_points = [duration * p for p in [0.1, 0.3, 0.5]]
        else:
            sample_points = [timestamp]

        for i, ts in enumerate(sample_points[:count]):
            out = output_path.replace(".jpg", f"_{i}.jpg")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(ts),
                "-i", filepath,
                "-vframes", "1",
                "-q:v", "2",
                "-vf", "scale=1080:-1",
                out,
            ]

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(process.communicate(), timeout=30)

                if process.returncode == 0 and os.path.exists(out):
                    thumbnails.append(out)
            except Exception as e:
                logger.error(f"Thumbnail generation error: {e}")

        return thumbnails


# ────────────────────────────────────────────────────────────
# VIRAL ANALYZER
# ────────────────────────────────────────────────────────────

class ViralAnalyzer:
    """Analyze clips for viral potential and generate optimization suggestions."""

    @staticmethod
    def calculate_virality_score(
        clip_duration: float,
        has_strong_hook: bool,
        scene_change_density: float,
        audio_energy: float,
        has_face: bool,
        has_subtitles: bool,
    ) -> float:
        """
        Calculate a 0-100 virality score based on clip characteristics.
        """
        score = 0.0

        # Duration sweet spot (30-60s is ideal)
        if 25 <= clip_duration <= 35:
            score += 20
        elif 45 <= clip_duration <= 65:
            score += 18
        elif 80 <= clip_duration <= 95:
            score += 12
        else:
            score += 5

        # Strong hook (captivating first 3 seconds)
        if has_strong_hook:
            score += 25

        # Scene variety (0.05-0.2 changes per second is ideal)
        if 0.05 <= scene_change_density <= 0.2:
            score += 15
        elif scene_change_density > 0:
            score += 8

        # Audio energy
        score += min(audio_energy * 20, 15)

        # Face presence (personal content performs better)
        if has_face:
            score += 15

        # Subtitles (crucial for engagement)
        if has_subtitles:
            score += 10

        return min(score, 100.0)

    @staticmethod
    def get_platform_recommendations(duration: float) -> Dict[str, str]:
        """Get platform-specific recommendations."""
        recs = {}

        if duration <= 15:
            recs["tiktok"] = "✅ Perfect length for TikTok"
            recs["reels"] = "✅ Great for Reels"
            recs["shorts"] = "✅ Ideal for Shorts"
        elif duration <= 30:
            recs["tiktok"] = "✅ Great length for TikTok"
            recs["reels"] = "✅ Perfect for Reels"
            recs["shorts"] = "✅ Good for Shorts"
        elif duration <= 60:
            recs["tiktok"] = "✅ Good for TikTok"
            recs["reels"] = "✅ Max length for Reels"
            recs["shorts"] = "✅ Perfect for Shorts"
        elif duration <= 90:
            recs["tiktok"] = "✅ Acceptable for TikTok"
            recs["reels"] = "⚠️ Over Reels limit, trim recommended"
            recs["shorts"] = "✅ Within Shorts limit"
        else:
            recs["tiktok"] = "⚠️ Long for TikTok, consider trimming"
            recs["reels"] = "❌ Exceeds Reels 90s limit"
            recs["shorts"] = "❌ Exceeds Shorts 60s limit"

        return recs

    @staticmethod
    def suggest_hashtags(duration: float, has_face: bool) -> List[str]:
        """Suggest generic trending hashtags."""
        tags = ["#fyp", "#viral", "#foryou", "#trending"]
        if has_face:
            tags.extend(["#storytime", "#relatable", "#pov"])
        if duration <= 15:
            tags.append("#quicktips")
        tags.extend(["#clips", "#highlights", "#contentcreator"])
        return tags[:10]


# ────────────────────────────────────────────────────────────
# CORE VIDEO PROCESSOR
# ────────────────────────────────────────────────────────────

class VideoProcessor:
    """
    Core video processing engine.
    Orchestrates all processing steps:
      clip → crop → subtitles → audio → thumbnail → finalize
    """

    def __init__(self):
        self.scene_detector = SceneDetector()
        self.face_tracker = FaceTracker()
        self.subtitle_gen = SubtitleGenerator()
        self.audio_proc = AudioProcessor()
        self.thumb_gen = ThumbnailGenerator()
        self.viral_analyzer = ViralAnalyzer()

    async def process_job(
        self,
        job: ProcessingJob,
        progress_callback=None,
    ) -> ProcessingJob:
        """
        Execute the full processing pipeline for a job.

        Pipeline:
        1. Detect scenes & find best clips
        2. Extract each clip
        3. Smart crop to target format
        4. Generate & burn subtitles
        5. Process audio (silence removal, normalization)
        6. Generate thumbnails
        7. Calculate virality scores
        """
        job_dir = os.path.join(config.temp_dir, job.job_id)
        os.makedirs(job_dir, exist_ok=True)

        try:
            source = job.source_path
            info = get_video_info(source)

            if info["duration"] <= 0:
                raise ValueError("Cannot read video file or duration is 0")

            if info["duration"] > config.max_video_duration:
                raise ValueError(
                    f"Video too long ({format_duration(info['duration'])}). "
                    f"Max: {format_duration(config.max_video_duration)}"
                )

            total_steps = 6
            step = 0

            # ── Step 1: Scene Detection ──
            step += 1
            job.status = JobStatus.DETECTING_SCENES
            if progress_callback:
                await progress_callback(
                    step / total_steps * 100,
                    "🔍 Analyzing video for best moments..."
                )

            if job.timestamps:
                # User provided manual timestamps
                clips_ts = job.timestamps
            else:
                clips_ts = await self.scene_detector.find_best_clips(
                    source, job.clip_duration, job.num_clips, info["duration"]
                )

            job.timestamps = clips_ts
            num_clips = len(clips_ts)

            # ── Step 2: Extract Clips ──
            step += 1
            job.status = JobStatus.CLIPPING
            if progress_callback:
                await progress_callback(
                    step / total_steps * 100,
                    f"✂️ Extracting {num_clips} clips..."
                )

            raw_clips = []
            for i, (start, end) in enumerate(clips_ts):
                clip_path = os.path.join(job_dir, f"clip_{i:02d}_raw.mp4")
                success = await self._extract_clip(source, clip_path, start, end)
                if success:
                    raw_clips.append(clip_path)
                else:
                    logger.warning(f"Failed to extract clip {i} ({start}-{end})")

            if not raw_clips:
                raise ValueError("Failed to extract any clips from video")

            # ── Step 3: Smart Crop ──
            step += 1
            job.status = JobStatus.CROPPING
            if progress_callback:
                await progress_callback(
                    step / total_steps * 100,
                    f"📐 Cropping to {job.output_format.value} format..."
                )

            cropped_clips = []
            for i, clip_path in enumerate(raw_clips):
                cropped_path = os.path.join(job_dir, f"clip_{i:02d}_cropped.mp4")
                result = await self._smart_crop(
                    clip_path, cropped_path, info, job
                )
                cropped_clips.append(result if result else clip_path)

            # ── Step 4: Subtitles ──
            step += 1
            subtitled_clips = []
            if job.subtitle_enabled:
                job.status = JobStatus.GENERATING_SUBTITLES
                if progress_callback:
                    await progress_callback(
                        step / total_steps * 100,
                        "💬 Generating subtitles..."
                    )

                for i, clip_path in enumerate(cropped_clips):
                    srt_path = os.path.join(job_dir, f"clip_{i:02d}.srt")
                    sub_path = os.path.join(job_dir, f"clip_{i:02d}_sub.mp4")

                    srt_result = await self.subtitle_gen.generate(
                        clip_path, srt_path,
                        language=job.subtitle_lang,
                    )

                    if srt_result:
                        job.output_subtitles.append(srt_path)
                        burned = await self._burn_subtitles(
                            clip_path, sub_path, srt_path, job
                        )
                        subtitled_clips.append(burned if burned else clip_path)
                    else:
                        subtitled_clips.append(clip_path)
            else:
                subtitled_clips = cropped_clips

            # ── Step 5: Audio Processing ──
            step += 1
            job.status = JobStatus.PROCESSING_AUDIO
            if progress_callback:
                await progress_callback(
                    step / total_steps * 100,
                    "🔊 Processing audio..."
                )

            final_clips = []
            for i, clip_path in enumerate(subtitled_clips):
                current = clip_path

                if job.remove_silence:
                    desilenced = os.path.join(job_dir, f"clip_{i:02d}_nosil.mp4")
                    result = await self.audio_proc.remove_silence(current, desilenced)
                    if result:
                        current = result

                if job.normalize_audio:
                    normalized = os.path.join(job_dir, f"clip_{i:02d}_norm.mp4")
                    result = await self.audio_proc.normalize_audio(current, normalized)
                    if result:
                        current = result

                # Final encode with quality settings
                final_path = os.path.join(job_dir, f"clip_{i:02d}_final.mp4")
                result = await self._final_encode(current, final_path, job)
                final_clips.append(result if result else current)

            # ── Step 6: Thumbnails & Analysis ──
            step += 1
            job.status = JobStatus.GENERATING_THUMBNAILS
            if progress_callback:
                await progress_callback(
                    step / total_steps * 100,
                    "🖼️ Generating thumbnails & analyzing..."
                )

            for i, clip_path in enumerate(final_clips):
                thumb_path = os.path.join(job_dir, f"thumb_{i:02d}.jpg")
                thumbs = await self.thumb_gen.generate(clip_path, thumb_path)
                job.output_thumbnails.extend(thumbs)

                # Virality scoring
                clip_info = get_video_info(clip_path)
                clip_duration = clip_info.get("duration", job.clip_duration)
                scene_density = len([
                    s for s in (await self.scene_detector.detect_scenes(clip_path, 0.4))
                ]) / max(clip_duration, 1)

                face_pos = await self.face_tracker.detect_face_region(clip_path)

                score = self.viral_analyzer.calculate_virality_score(
                    clip_duration=clip_duration,
                    has_strong_hook=i == 0,  # First clip usually has hook
                    scene_change_density=scene_density,
                    audio_energy=0.7,
                    has_face=face_pos is not None,
                    has_subtitles=job.subtitle_enabled,
                )
                              job.virality_scores.append(score)

            job.output_clips = final_clips

            # ── Move finals to output dir ──
            output_job_dir = os.path.join(config.output_dir, job.job_id)
            os.makedirs(output_job_dir, exist_ok=True)

            final_output_clips = []
            for i, clip_path in enumerate(final_clips):
                dest = os.path.join(output_job_dir, f"clip_{i:02d}.mp4")
                shutil.copy2(clip_path, dest)
                final_output_clips.append(dest)

            job.output_clips = final_output_clips
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now().isoformat()

            if progress_callback:
                await progress_callback(100.0, "✅ Processing complete!")

            # Save to DB
            db.save_job(job)
            for i, clip_path in enumerate(final_output_clips):
                clip_id = f"{job.job_id}_{i:02d}"
                clip_info_data = get_video_info(clip_path)
                db.save_clip(
                    clip_id=clip_id,
                    job_id=job.job_id,
                    user_id=job.user_id,
                    file_path=clip_path,
                    duration=clip_info_data.get("duration", 0),
                    virality_score=job.virality_scores[i] if i < len(job.virality_scores) else 0,
                    thumbnail_path=job.output_thumbnails[i] if i < len(job.output_thumbnails) else "",
                    subtitle_path=job.output_subtitles[i] if i < len(job.output_subtitles) else "",
                )

            return job

        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.error_message = "Job was cancelled by user"
            db.save_job(job)
            raise
        except Exception as e:
            logger.exception(f"Job {job.job_id} failed: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            db.save_job(job)
            return job
        finally:
            # Cleanup temp files for this job
            try:
                if os.path.exists(job_dir):
                    shutil.rmtree(job_dir, ignore_errors=True)
            except Exception:
                pass

    async def _extract_clip(
        self, source: str, output: str, start: float, end: float
    ) -> bool:
        """Extract a clip segment from source video."""
        duration = end - start
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", source,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            output,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=180)

            if process.returncode == 0 and os.path.exists(output):
                return True
            logger.error(f"Clip extraction failed: {stderr.decode()[-300:]}")
            return False
        except Exception as e:
            logger.error(f"Clip extraction error: {e}")
            return False

    async def _smart_crop(
        self, input_path: str, output_path: str,
        video_info: Dict, job: ProcessingJob
    ) -> Optional[str]:
        """Apply smart cropping with face tracking."""
        source_w = video_info.get("width", 1920)
        source_h = video_info.get("height", 1080)
        target_w, target_h = FORMAT_RESOLUTIONS[job.output_format]

        if job.smart_crop:
            crop_x, crop_y, crop_w, crop_h = await self.face_tracker.get_crop_params(
                input_path, source_w, source_h, job.output_format
            )
        else:
            # Center crop
            target_ratio = target_w / target_h
            source_ratio = source_w / source_h
            if source_ratio > target_ratio:
                crop_h = source_h
                crop_w = int(source_h * target_ratio)
            else:
                crop_w = source_w
                crop_h = int(source_w / target_ratio)
            crop_x = (source_w - crop_w) // 2
            crop_y = (source_h - crop_h) // 2

        vf = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={target_w}:{target_h}"

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "copy",
            output_path,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=180)

            if process.returncode == 0 and os.path.exists(output_path):
                return output_path
            logger.error(f"Crop failed: {stderr.decode()[-300:]}")
            return None
        except Exception as e:
            logger.error(f"Crop error: {e}")
            return None

    async def _burn_subtitles(
        self, input_path: str, output_path: str,
        srt_path: str, job: ProcessingJob
    ) -> Optional[str]:
        """Burn subtitles into video."""
        settings = db.get_user_settings(job.user_id)
        sub_filter = SubtitleGenerator.get_subtitle_filter(
            srt_path,
            style=job.subtitle_style,
            font_size=settings.subtitle_font_size,
            font_color=settings.subtitle_font_color,
            position=settings.subtitle_position,
        )

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", sub_filter,
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "copy",
            output_path,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            if process.returncode == 0 and os.path.exists(output_path):
                return output_path
            logger.error(f"Subtitle burn failed: {stderr.decode()[-300:]}")
            return None
        except Exception as e:
            logger.error(f"Subtitle burn error: {e}")
            return None

    async def _final_encode(
        self, input_path: str, output_path: str, job: ProcessingJob
    ) -> Optional[str]:
        """Final encoding pass with quality settings."""
        quality = QUALITY_MAP.get(job.quality_preset, QUALITY_MAP[QualityPreset.HIGH])

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264",
            "-crf", quality["crf"],
            "-preset", quality["preset"],
            "-maxrate", quality["bitrate"],
            "-bufsize", str(int(quality["bitrate"].replace("M", "")) * 2) + "M",
            "-r", str(job.fps),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=600)

            if process.returncode == 0 and os.path.exists(output_path):
                return output_path
            logger.error(f"Final encode failed: {stderr.decode()[-300:]}")
            return None
        except Exception as e:
            logger.error(f"Final encode error: {e}")
            return None


# ────────────────────────────────────────────────────────────
# JOB QUEUE
# ────────────────────────────────────────────────────────────

class JobQueue:
    """Async job queue for managing video processing tasks."""

    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self._queue: asyncio.Queue[ProcessingJob] = asyncio.Queue()
        self._active_jobs: Dict[str, asyncio.Task] = {}
        self._processor = VideoProcessor()
        self._running = False
        self._progress_callbacks: Dict[str, Any] = {}

    async def start(self):
        """Start the job queue worker."""
        self._running = True
        logger.info(f"Job queue started (max concurrent: {self.max_concurrent})")
        asyncio.create_task(self._worker_loop())

    async def stop(self):
        """Stop the job queue worker."""
        self._running = False
        for task in self._active_jobs.values():
            task.cancel()
        logger.info("Job queue stopped")

    async def add_job(self, job: ProcessingJob, progress_callback=None) -> str:
        """Add a job to the queue."""
        db.save_job(job)
        if progress_callback:
            self._progress_callbacks[job.job_id] = progress_callback
        await self._queue.put(job)
        logger.info(f"Job {job.job_id} added to queue (queue size: {self._queue.qsize()})")
        return job.job_id

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job by ID."""
        if job_id in self._active_jobs:
            self._active_jobs[job_id].cancel()
            del self._active_jobs[job_id]
            if job_id in self._progress_callbacks:
                del self._progress_callbacks[job_id]
            logger.info(f"Job {job_id} cancelled")
            return True
        return False

    def get_queue_size(self) -> int:
        return self._queue.qsize()

    def get_active_count(self) -> int:
        return len(self._active_jobs)

    async def _worker_loop(self):
        """Main worker loop that processes jobs from the queue."""
        while self._running:
            try:
                # Wait for available slot
                while len(self._active_jobs) >= self.max_concurrent:
                    # Clean up completed tasks
                    completed = [
                        jid for jid, task in self._active_jobs.items()
                        if task.done()
                    ]
                    for jid in completed:
                        del self._active_jobs[jid]
                        if jid in self._progress_callbacks:
                            del self._progress_callbacks[jid]
                    await asyncio.sleep(1)

                # Get next job (wait up to 2 seconds)
                try:
                    job = await asyncio.wait_for(self._queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue

                # Start processing
                callback = self._progress_callbacks.get(job.job_id)
                task = asyncio.create_task(
                    self._process_with_error_handling(job, callback)
                )
                self._active_jobs[job.job_id] = task

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(1)

    async def _process_with_error_handling(
        self, job: ProcessingJob, progress_callback=None
    ):
        """Process a job with error handling."""
        try:
            await self._processor.process_job(job, progress_callback)
        except asyncio.CancelledError:
            logger.info(f"Job {job.job_id} was cancelled")
        except Exception as e:
            logger.exception(f"Job {job.job_id} failed with error: {e}")


# ────────────────────────────────────────────────────────────
# TELEGRAM BOT HANDLERS
# ────────────────────────────────────────────────────────────

# Import telegram modules
try:
    from telegram import (
        Bot,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Update,
    )
    from telegram.constants import ChatAction, ParseMode
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
except ImportError:
    logger.error(
        "python-telegram-bot not installed. "
        "Run: pip install python-telegram-bot[job-queue]"
    )
    sys.exit(1)


# Global instances
job_queue = JobQueue(max_concurrent=2)
pending_jobs: Dict[int, ProcessingJob] = {}  # chat_id -> pending job config


# ── Auth Middleware ──

def auth_required(func):
    """Decorator to check if user is whitelisted."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not db.is_whitelisted(user_id):
            await update.message.reply_text(
                "⛔ *Access Denied*\n\n"
                "Maaf, kamu belum terdaftar untuk menggunakan bot ini.\n"
                "Hubungi admin untuk mendapatkan akses.\n\n"
                f"User ID kamu: `{user_id}`",
                parse_mode=ParseMode.MARKDOWN,
            )
            logger.warning(f"Unauthorized access attempt by user {user_id}")
            return
        return await func(update, context)
    return wrapper


def admin_required(func):
    """Decorator to check if user is admin."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id not in config.admin_user_ids:
            await update.message.reply_text(
                "⛔ *Admin Only*\n\nPerintah ini hanya untuk admin.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        return await func(update, context)
    return wrapper


# ── /start ──

@auth_required
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user = update.effective_user
    welcome = (
        f"🎬 *Selamat Datang, {user.first_name}!*\n\n"
        "Aku adalah *Video Clipper Bot* — asisten otomatis kamu untuk "
        "mengubah video panjang menjadi konten viral pendek!\n\n"
        "🔥 *Yang bisa aku lakukan:*\n"
        "├ ✂️ Auto-clip video jadi beberapa segmen pendek\n"
        "├ 📐 Crop otomatis ke format 9:16 (TikTok/Reels/Shorts)\n"
        "├ 💬 Generate subtitle otomatis (AI Whisper)\n"
        "├ 🔊 Normalisasi audio & hapus bagian hening\n"
        "├ 🖼️ Generate thumbnail otomatis\n"
        "├ 📊 Analisis skor viralitas\n"
        "└ 🎯 Smart face tracking & cropping\n\n"
        "📌 *Cara Pakai:*\n"
        "1️⃣ Kirim file video atau URL (YouTube, IG, TikTok)\n"
        "2️⃣ Pilih pengaturan (durasi, format, subtitle)\n"
        "3️⃣ Tunggu proses selesai\n"
        "4️⃣ Terima clip-clip siap upload!\n\n"
        "Ketik /help untuk daftar perintah lengkap 🚀"
    )

    keyboard = [
        [
            InlineKeyboardButton("📖 Help", callback_data="help"),
            InlineKeyboardButton("⚙️ Settings", callback_data="settings_menu"),
        ],
        [
            InlineKeyboardButton("📊 Status", callback_data="status"),
            InlineKeyboardButton("📜 History", callback_data="history"),
        ],
    ]

    await update.message.reply_text(
        welcome,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


# ── /help ──

@auth_required
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_text = (
        "📖 *Daftar Perintah*\n\n"
        "*Utama:*\n"
        "/start — Pesan selamat datang\n"
        "/help — Daftar perintah ini\n"
        "/process — Mulai proses video\n"
        "/settings — Atur preferensi\n"
        "/status — Cek antrian & job aktif\n"
        "/cancel — Batalkan proses\n"
        "/history — Riwayat video\n"
        "/download `<clip_id>` — Download ulang clip\n\n"
        "*Admin:*\n"
        "/adduser `<user_id>` — Tambah user ke whitelist\n"
        "/removeuser `<user_id>` — Hapus user dari whitelist\n"
        "/listusers — Lihat daftar user\n"
        "/broadcast `<pesan>` — Kirim pesan ke semua user\n\n"
        "*Input yang Diterima:*\n"
        "📎 Upload file video langsung\n"
        "🔗 Kirim URL (YouTube, Instagram, TikTok, Twitter)\n"
        "⏱️ Kirim timestamp manual: `/process 0:30-1:30 2:00-3:00`\n\n"
        "*Format Output:*\n"
        "📱 9:16 — TikTok, Reels, Shorts\n"
        "🖥️ 16:9 — YouTube\n"
        "⬜ 1:1 — Instagram Feed\n\n"
        "*Tips:*\n"
        "• Video max 2 jam / 2GB\n"
        "• Semakin tinggi kualitas source = hasil lebih baik\n"
        "• Gunakan subtitle untuk boost engagement +40%\n"
    )

    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


# ── /settings ──

@auth_required
async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /settings command — show current settings."""
    user_id = update.effective_user.id
    settings = db.get_user_settings(user_id)

    text = (
        "⚙️ *Pengaturan Kamu*\n\n"
        f"⏱️ Durasi Clip: *{settings.clip_duration}s*\n"
        f"🔢 Jumlah Clip: *{settings.num_clips}*\n"
        f"📐 Format: *{settings.output_format}*\n"
        f"🎨 Kualitas: *{settings.quality_preset}*\n"
        f"💬 Subtitle: *{'✅ On' if settings.subtitle_enabled else '❌ Off'}*\n"
        f"🌐 Bahasa Sub: *{settings.subtitle_lang}*\n"
        f"🎨 Style Sub: *{settings.subtitle_style}*\n"
        f"📏 Font Size: *{settings.subtitle_font_size}*\n"
        f"🤫 Hapus Hening: *{'✅' if settings.remove_silence else '❌'}*\n"
        f"🔊 Normalisasi Audio: *{'✅' if settings.normalize_audio else '❌'}*\n"
        f"👤 Smart Crop: *{'✅' if settings.smart_crop else '❌'}*\n"
    )

    keyboard = [
        [
            InlineKeyboardButton("⏱️ Durasi", callback_data="set_duration"),
            InlineKeyboardButton("🔢 Jumlah", callback_data="set_numclips"),
        ],
        [
            InlineKeyboardButton("📐 Format", callback_data="set_format"),
            InlineKeyboardButton("🎨 Kualitas", callback_data="set_quality"),
        ],
        [
            InlineKeyboardButton("💬 Subtitle", callback_data="set_subtitle"),
            InlineKeyboardButton("🌐 Bahasa", callback_data="set_language"),
        ],
        [
            InlineKeyboardButton("🎨 Style Sub", callback_data="set_substyle"),
            InlineKeyboardButton("📏 Font", callback_data="set_fontsize"),
        ],
        [
            InlineKeyboardButton("🤫 Hapus Hening", callback_data="toggle_silence"),
            InlineKeyboardButton("🔊 Normalize", callback_data="toggle_normalize"),
        ],
        [
            InlineKeyboardButton("👤 Smart Crop", callback_data="toggle_smartcrop"),
        ],
        [
            InlineKeyboardButton("🔄 Reset Default", callback_data="reset_settings"),
        ],
    ]

    await update.message.reply_text(
        text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


# ── /status ──

@auth_required
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command."""
    user_id = update.effective_user.id
    active = db.get_active_jobs(user_id)

    queue_size = job_queue.get_queue_size()
    active_count = job_queue.get_active_count()

    text = (
        "📊 *Status Sistem*\n\n"
        f"🔄 Job Aktif: *{active_count}*\n"
        f"📋 Antrian: *{queue_size}*\n\n"
    )

    if active:
        text += "📌 *Job Kamu:*\n"
        for job_data in active:
            status_emoji = {
                "queued": "⏳",
                "downloading": "📥",
                "detecting_scenes": "🔍",
                "clipping": "✂️",
                "cropping": "📐",
                "generating_subtitles": "💬",
                "burning_subtitles": "🔥",
                "processing_audio": "🔊",
                "generating_thumbnails": "🖼️",
                "finalizing": "⚡",
            }.get(job_data["status"], "⏳")

            text += (
                f"\n{status_emoji} Job `{job_data['job_id']}`\n"
                f"   Status: *{job_data['status']}*\n"
                f"   Dibuat: {job_data['created_at'][:16]}\n"
            )
    else:
        text += "✅ Tidak ada job aktif saat ini."

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


# ── /cancel ──

@auth_required
async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /cancel command."""
    user_id = update.effective_user.id

    # Cancel pending configuration
    if user_id in pending_jobs:
        del pending_jobs[user_id]
        await update.message.reply_text("🗑️ Konfigurasi yang tertunda telah dibatalkan.")
        return

    # Cancel active jobs
    active = db.get_active_jobs(user_id)
    if not active:
        await update.message.reply_text("ℹ️ Tidak ada job aktif untuk dibatalkan.")
        return

    cancelled = 0
    for job_data in active:
        if job_queue.cancel_job(job_data["job_id"]):
            cancelled += 1

    await update.message.reply_text(
        f"🗑️ *{cancelled}* job telah dibatalkan.",
        parse_mode=ParseMode.MARKDOWN,
    )


# ── /history ──

@auth_required
async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /history command."""
    user_id = update.effective_user.id
    jobs = db.get_user_jobs(user_id, limit=10)

    if not jobs:
        await update.message.reply_text("📜 Belum ada riwayat pemrosesan video.")
        return

    text = "📜 *Riwayat Video (10 terakhir):*\n\n"

    for job_data in jobs:
        status_emoji = {
            "completed": "✅",
            "failed": "❌",
            "cancelled": "🗑️",
        }.get(job_data["status"], "⏳")

        meta = json.loads(job_data.get("metadata", "{}") or "{}")
        scores = meta.get("virality_scores", [])
        avg_score = sum(scores) / len(scores) if scores else 0

        text += (
            f"{status_emoji} `{job_data['job_id']}` — "
            f"{job_data['num_output_clips']} clips"
        )

        if job_data["status"] == "completed":
            text += f" | 🔥 {avg_score:.0f}/100"

        text += f"\n   📅 {job_data['created_at'][:16]}\n\n"

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


# ── /download ──

@auth_required
async def cmd_download(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /download <clip_id> command."""
    if not context.args:
        await update.message.reply_text(
            "ℹ️ Gunakan: `/download <clip_id>`\n"
            "Contoh: `/download abc12345_00`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    clip_id = context.args[0]
    clip_data = db.get_clip(clip_id)

    if not clip_data:
        await update.message.reply_text("❌ Clip tidak ditemukan.")
        return

    if clip_data["user_id"] != update.effective_user.id:
        await update.message.reply_text("⛔ Clip ini bukan milikmu.")
        return

    file_path = clip_data.get("file_path", "")

    if clip_data.get("telegram_file_id"):
        await update.message.reply_video(
            video=clip_data["telegram_file_id"],
            caption=f"📎 Clip `{clip_id}`",
            parse_mode=ParseMode.MARKDOWN,
        )
    elif file_path and os.path.exists(file_path):
        await update.message.reply_chat_action(ChatAction.UPLOAD_VIDEO)
        with open(file_path, "rb") as f:
            msg = await update.message.reply_video(
                video=f,
                caption=f"📎 Clip `{clip_id}`",
                parse_mode=ParseMode.MARKDOWN,
            )
            # Save telegram file_id for future downloads
            if msg.video:
                db.save_clip(
                    clip_id=clip_id,
                    job_id=clip_data["job_id"],
                    user_id=clip_data["user_id"],
                    file_path=file_path,
                    telegram_file_id=msg.video.file_id,
                    duration=clip_data.get("duration", 0),
                    virality_score=clip_data.get("virality_score", 0),
                )
    else:
        await update.message.reply_text(
            "❌ File clip sudah tidak tersedia di server.\n"
            "File otomatis dihapus setelah 24 jam."
        )


# ── /process ──

@auth_required
async def cmd_process(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /process command with optional manual timestamps."""
    user_id = update.effective_user.id

    # Parse manual timestamps if provided
    manual_timestamps: List[Tuple[float, float]] = []
    if context.args:
        for arg in context.args:
            match = re.match(r"(\d+):(\d+)-(\d+):(\d+)", arg)
            if match:
                start = int(match.group(1)) * 60 + int(match.group(2))
                end = int(match.group(3)) * 60 + int(match.group(4))
                manual_timestamps.append((float(start), float(end)))

    settings = db.get_user_settings(user_id)

    job = ProcessingJob(
        user_id=user_id,
        chat_id=update.effective_chat.id,
        clip_duration=settings.clip_duration,
        num_clips=settings.num_clips,
        output_format=OutputFormat(settings.output_format),
        quality_preset=QualityPreset(settings.quality_preset),
        subtitle_enabled=settings.subtitle_enabled,
        subtitle_lang=settings.subtitle_lang,
        subtitle_style=settings.subtitle_style,
        remove_silence=settings.remove_silence,
        normalize_audio=settings.normalize_audio,
        smart_crop=settings.smart_crop,
        timestamps=manual_timestamps,
    )

    pending_jobs[user_id] = job

    text = (
        "🎬 *Mode Proses Video*\n\n"
        "Kirim video kamu sekarang:\n"
        "📎 Upload file video langsung, atau\n"
        "🔗 Kirim URL video (YouTube, IG, TikTok, Twitter)\n\n"
        "*Pengaturan saat ini:*\n"
        f"⏱️ Durasi: {settings.clip_duration}s\n"
        f"🔢 Jumlah: {settings.num_clips} clips\n"
        f"📐 Format: {settings.output_format}\n"
        f"💬 Subtitle: {'On' if settings.subtitle_enabled else 'Off'}\n"
    )

    if manual_timestamps:
        text += f"\n📍 Timestamp manual: {len(manual_timestamps)} segmen\n"
        for i, (s, e) in enumerate(manual_timestamps):
            text += f"   {i+1}. {format_duration(s)} → {format_duration(e)}\n"

    text += "\nKetik /cancel untuk membatalkan."

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


# ── Admin Commands ──

@admin_required
async def cmd_adduser(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /adduser <user_id> command."""
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text(
            "ℹ️ Gunakan: `/adduser <user_id>`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    uid = int(context.args[0])
    db.add_to_whitelist(uid, added_by=update.effective_user.id)
    await update.message.reply_text(
        f"✅ User `{uid}` ditambahkan ke whitelist.",
        parse_mode=ParseMode.MARKDOWN,
    )


@admin_required
async def cmd_removeuser(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /removeuser <user_id> command."""
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text(
            "ℹ️ Gunakan: `/removeuser <user_id>`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    uid = int(context.args[0])
    if uid in config.admin_user_ids:
        await update.message.reply_text("⛔ Tidak bisa menghapus admin.")
        return

    db.remove_from_whitelist(uid)
    await update.message.reply_text(
        f"✅ User `{uid}` dihapus dari whitelist.",
        parse_mode=ParseMode.MARKDOWN,
    )


@admin_required
async def cmd_listusers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /listusers command."""
    users = db.get_whitelist()
    text = "👥 *Daftar User Terdaftar:*\n\n"
    for uid in users:
        is_admin = "👑" if uid in config.admin_user_ids else "👤"
        text += f"{is_admin} `{uid}`\n"
    text += f"\nTotal: {len(users)} user"
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


# ── Message Handlers ──

@auth_required
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle video file uploads."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Check disk space
    disk_usage = get_disk_usage(config.temp_dir)
    if disk_usage > config.max_temp_size_gb:
        cleanup_old_files(config.temp_dir, config.auto_cleanup_hours)
        cleanup_old_files(config.output_dir, config.auto_cleanup_hours)

    # Get or create job
    job = pending_jobs.pop(user_id, None)
    if job is None:
        settings = db.get_user_settings(user_id)
        job = ProcessingJob(
            user_id=user_id,
            chat_id=chat_id,
            clip_duration=settings.clip_duration,
            num_clips=settings.num_clips,
            output_format=OutputFormat(settings.output_format),
            quality_preset=QualityPreset(settings.quality_preset),
            subtitle_enabled=settings.subtitle_enabled,
            subtitle_lang=settings.subtitle_lang,
            subtitle_style=settings.subtitle_style,
            remove_silence=settings.remove_silence,
            normalize_audio=settings.normalize_audio,
            smart_crop=settings.smart_crop,
        )

    # Show processing options
    keyboard = [
        [
            InlineKeyboardButton("⏱️ 30s", callback_data=f"dur_30_{job.job_id}"),
            InlineKeyboardButton("⏱️ 60s", callback_data=f"dur_60_{job.job_id}"),
            InlineKeyboardButton("⏱️ 90s", callback_data=f"dur_90_{job.job_id}"),
        ],
        [
            InlineKeyboardButton("🔢 3 clips", callback_data=f"num_3_{job.job_id}"),
            InlineKeyboardButton("🔢 5 clips", callback_data=f"num_5_{job.job_id}"),
            InlineKeyboardButton("🔢 10 clips", callback_data=f"num_10_{job.job_id}"),
        ],
        [
            InlineKeyboardButton("📱 9:16", callback_data=f"fmt_916_{job.job_id}"),
            InlineKeyboardButton("🖥️ 16:9", callback_data=f"fmt_169_{job.job_id}"),
            InlineKeyboardButton("⬜ 1:1", callback_data=f"fmt_11_{job.job_id}"),
        ],
        [
            InlineKeyboardButton(
                f"💬 Sub: {'ON' if job.subtitle_enabled else 'OFF'}",
                callback_data=f"sub_toggle_{job.job_id}",
            ),
            InlineKeyboardButton(
                f"🤫 Silence: {'ON' if job.remove_silence else 'OFF'}",
                callback_data=f"sil_toggle_{job.job_id}",
            ),
        ],
        [
            InlineKeyboardButton(
                "🚀 Proses Sekarang!",
                callback_data=f"start_process_{job.job_id}",
            ),
        ],
        [
            InlineKeyboardButton("❌ Batal", callback_data=f"cancel_{job.job_id}"),
        ],
    ]

    job.source_type = "file"

    # Download file from Telegram
    video = update.message.video or update.message.document
    if not video:
        await update.message.reply_text("❌ Tidak bisa membaca file video.")
        return

    status_msg = await update.message.reply_text(
        "📥 *Mengunduh video dari Telegram...*",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        job_dir = os.path.join(config.temp_dir, job.job_id)
        os.makedirs(job_dir, exist_ok=True)

        file = await context.bot.get_file(video.file_id)
        ext = ".mp4"
        if video.file_name:
            _, ext = os.path.splitext(video.file_name)
            ext = ext or ".mp4"

        source_path = os.path.join(job_dir, f"source{ext}")
        await file.download_to_drive(source_path)

        job.source_path = source_path

        # Get video info
        info = get_video_info(source_path)

        await status_msg.edit_text(
            f"✅ *Video diterima!*\n\n"
            f"📁 Ukuran: {format_size(os.path.getsize(source_path))}\n"
            f"⏱️ Durasi: {format_duration(info['duration'])}\n"
            f"📐 Resolusi: {info['width']}x{info['height']}\n"
            f"🎞️ FPS: {info['fps']:.0f}\n\n"
            f"Pilih pengaturan lalu tekan *🚀 Proses Sekarang!*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

        job.message_id = status_msg.message_id
        pending_jobs[user_id] = job

    except Exception as e:
        logger.exception(f"Video download error: {e}")
        await status_msg.edit_text(
            f"❌ *Gagal mengunduh video:*\n`{str(e)[:200]}`",
            parse_mode=ParseMode.MARKDOWN,
        )


@auth_required
async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle URL messages (YouTube, Instagram, TikTok, etc.)."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    url = update.message.text.strip()

    if not is_url(url):
        return  # Not a recognized video URL

    # Get or create job
    job = pending_jobs.pop(user_id, None)
    if job is None:
        settings = db.get_user_settings(user_id)
        job = ProcessingJob(
            user_id=user_id,
            chat_id=chat_id,
            clip_duration=settings.clip_duration,
            num_clips=settings.num_clips,
            output_format=OutputFormat(settings.output_format),
            quality_preset=QualityPreset(settings.quality_preset),
            subtitle_enabled=settings.subtitle_enabled,
            subtitle_lang=settings.subtitle_lang,
            subtitle_style=settings.subtitle_style,
            remove_silence=settings.remove_silence,
            normalize_audio=settings.normalize_audio,
            smart_crop=settings.smart_crop,
        )

    job.source_type = "url"
    job.source_url = url

    status_msg = await update.message.reply_text(
        "📥 *Mengunduh video dari URL...*\n"
        f"🔗 `{url[:80]}...`",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        job_dir = os.path.join(config.temp_dir, job.job_id)
        os.makedirs(job_dir, exist_ok=True)

        filepath = await VideoDownloader.download(url, job_dir)
        if not filepath:
            await status_msg.edit_text(
                "❌ *Gagal mengunduh video dari URL.*\n"
                "Pastikan URL valid dan video tersedia.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        job.source_path = filepath

        info = get_video_info(filepath)

        keyboard = [
            [
                InlineKeyboardButton("⏱️ 30s", callback_data=f"dur_30_{job.job_id}"),
                InlineKeyboardButton("⏱️ 60s", callback_data=f"dur_60_{job.job_id}"),
                InlineKeyboardButton("⏱️ 90s", callback_data=f"dur_90_{job.job_id}"),
            ],
            [
                InlineKeyboardButton("🔢 3 clips", callback_data=f"num_3_{job.job_id}"),
                InlineKeyboardButton("🔢 5 clips", callback_data=f"num_5_{job.job_id}"),
                InlineKeyboardButton("🔢 10 clips", callback_data=f"num_10_{job.job_id}"),
            ],
            [
                InlineKeyboardButton("📱 9:16", callback_data=f"fmt_916_{job.job_id}"),
                InlineKeyboardButton("🖥️ 16:9", callback_data=f"fmt_169_{job.job_id}"),
                InlineKeyboardButton("⬜ 1:1", callback_data=f"fmt_11_{job.job_id}"),
            ],
            [
                InlineKeyboardButton(
                    f"💬 Sub: {'ON' if job.subtitle_enabled else 'OFF'}",
                    callback_data=f"sub_toggle_{job.job_id}",
                ),
                InlineKeyboardButton(
                    f"🤫 Silence: {'ON' if job.remove_silence else 'OFF'}",
                    callback_data=f"sil_toggle_{job.job_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "🚀 Proses Sekarang!",
                    callback_data=f"start_process_{job.job_id}",
                ),
            ],
            [
                InlineKeyboardButton("❌ Batal", callback_data=f"cancel_{job.job_id}"),
            ],
        ]

        await status_msg.edit_text(
            f"✅ *Video berhasil diunduh!*\n\n"
            f"📁 Ukuran: {format_size(os.path.getsize(filepath))}\n"
            f"⏱️ Durasi: {format_duration(info['duration'])}\n"
            f"📐 Resolusi: {info['width']}x{info['height']}\n"
            f"🎞️ FPS: {info['fps']:.0f}\n\n"
            f"Pilih pengaturan lalu tekan *🚀 Proses Sekarang!*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

        job.message_id = status_msg.message_id
        pending_jobs[user_id] = job

    except Exception as e:
        logger.exception(f"URL download error: {e}")
        await status_msg.edit_text(
            f"❌ *Gagal:*\n`{str(e)[:200]}`",
            parse_mode=ParseMode.MARKDOWN,
        )


# ── Callback Query Handler ──

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all inline keyboard button presses."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    data = query.data

    # ── Simple menu callbacks ──
    if data == "help":
        await query.message.reply_text(
            "Ketik /help untuk melihat daftar perintah lengkap."
        )
        return

    if data == "settings_menu":
        await query.message.reply_text("Ketik /settings untuk mengatur preferensi.")
        return

    if data == "status":
        await query.message.reply_text("Ketik /status untuk cek status.")
        return

    if data == "history":
        await query.message.reply_text("Ketik /history untuk riwayat.")
        return

    # ── Settings toggles ──
    if data.startswith("set_") or data.startswith("toggle_") or data == "reset_settings":
        settings = db.get_user_settings(user_id)

        if data == "toggle_silence":
            settings.remove_silence = not settings.remove_silence
        elif data == "toggle_normalize":
            settings.normalize_audio = not settings.normalize_audio
        elif data == "toggle_smartcrop":
            settings.smart_crop = not settings.smart_crop
        elif data == "reset_settings":
            settings = UserSettings(user_id=user_id)

        elif data == "set_duration":
            keyboard = [
                [
                    InlineKeyboardButton("15s", callback_data="sdur_15"),
                    InlineKeyboardButton("30s", callback_data="sdur_30"),
                    InlineKeyboardButton("45s", callback_data="sdur_45"),
                ],
                [
                    InlineKeyboardButton("60s", callback_data="sdur_60"),
                    InlineKeyboardButton("90s", callback_data="sdur_90"),
                    InlineKeyboardButton("120s", callback_data="sdur_120"),
                ],
            ]
            await query.message.edit_text(
                "⏱️ Pilih durasi clip default:",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
            return

        elif data == "set_numclips":
            keyboard = [
                [
                    InlineKeyboardButton("3", callback_data="snum_3"),
                    InlineKeyboardButton("5", callback_data="snum_5"),
                    InlineKeyboardButton("10", callback_data="snum_10"),
                    InlineKeyboardButton("15", callback_data="snum_15"),
                ],
            ]
            await query.message.edit_text(
                "🔢 Pilih jumlah clip default:",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
            return

        elif data == "set_format":
            keyboard = [
                [
                    InlineKeyboardButton("📱 9:16", callback_data="sfmt_916"),
                    InlineKeyboardButton("🖥️ 16:9", callback_data="sfmt_169"),
                    InlineKeyboardButton("⬜ 1:1", callback_data="sfmt_11"),
                ],
            ]
            await query.message.edit_text(
                "📐 Pilih format output default:",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
            return

        elif data == "set_quality":
            keyboard = [
                [
                    InlineKeyboardButton("Low", callback_data="squal_low"),
                    InlineKeyboardButton("Medium", callback_data="squal_medium"),
                ],
                [
                    InlineKeyboardButton("High", callback_data="squal_high"),
                    InlineKeyboardButton("Ultra", callback_data="squal_ultra"),
                ],
            ]
            await query.message.edit_text(
                "🎨 Pilih kualitas default:",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
            return

        elif data == "set_subtitle":
            settings.subtitle_enabled = not settings.subtitle_enabled

        elif data == "set_language":
            keyboard = [
                [
                    InlineKeyboardButton("🌐 Auto", callback_data="slang_auto"),
                    InlineKeyboardButton("🇮🇩 Indonesia", callback_data="slang_id"),
                ],
                [
                    InlineKeyboardButton("🇬🇧 English", callback_data="slang_en"),
                    InlineKeyboardButton("🇯🇵 Japanese", callback_data="slang_ja"),
                ],
                [
                    InlineKeyboardButton("🇰🇷 Korean", callback_data="slang_ko"),
                    InlineKeyboardButton("🇨🇳 Chinese", callback_data="slang_zh"),
                ],
            ]
            await query.message.edit_text(
                "🌐 Pilih bahasa subtitle:",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
            return

        elif data == "set_substyle":
            keyboard = [
                [
                    InlineKeyboardButton("🔥 Viral", callback_data="sstyle_viral"),
                    InlineKeyboardButton("📝 Minimal", callback_data="sstyle_minimal"),
                ],
                [
                    InlineKeyboardButton("💪 Bold", callback_data="sstyle_bold"),
                    InlineKeyboardButton("🎤 Karaoke", callback_data="sstyle_karaoke"),
                ],
            ]
            await query.message.edit_text(
                "🎨 Pilih style subtitle:",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
            return

        elif data == "set_fontsize":
            keyboard = [
                [
                    InlineKeyboardButton("18", callback_data="sfont_18"),
                    InlineKeyboardButton("22", callback_data="sfont_22"),
                    InlineKeyboardButton("24", callback_data="sfont_24"),
                ],
                [
                    InlineKeyboardButton("28", callback_data="sfont_28"),
                    InlineKeyboardButton("32", callback_data="sfont_32"),
                    InlineKeyboardButton("36", callback_data="sfont_36"),
                ],
            ]
            await query.message.edit_text(
                "📏 Pilih ukuran font subtitle:",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
            return

        db.save_user_settings(settings)
        await query.message.reply_text(
            "✅ Pengaturan disimpan! Ketik /settings untuk lihat.",
        )
        return

    # ── Settings value selections ──
    if data.startswith("sdur_"):
        val = int(data.split("_")[1])
        settings = db.get_user_settings(user_id)
        settings.clip_duration = val
        db.save_user_settings(settings)
        await query.message.edit_text(f"✅ Durasi clip default: *{val}s*", parse_mode=ParseMode.MARKDOWN)
        return

    if data.startswith("snum_"):
        val = int(data.split("_")[1])
        settings = db.get_user_settings(user_id)
        settings.num_clips = val
        db.save_user_settings(settings)
        await query.message.edit_text(f"✅ Jumlah clip default: *{val}*", parse_mode=ParseMode.MARKDOWN)
        return

    if data.startswith("sfmt_"):
        fmt_map = {"916": "9:16", "169": "16:9", "11": "1:1"}
        val = fmt_map.get(data.split("_")[1], "9:16")
        settings = db.get_user_settings(user_id)
        settings.output_format = val
        db.save_user_settings(settings)
        await query.message.edit_text(f"✅ Format output: *{val}*", parse_mode=ParseMode.MARKDOWN)
        return

    if data.startswith("squal_"):
        val = data.split("_")[1]
        settings = db.get_user_settings(user_id)
        settings.quality_preset = val
        db.save_user_settings(settings)
        await query.message.edit_text(f"✅ Kualitas: *{val}*", parse_mode=ParseMode.MARKDOWN)
        return

    if data.startswith("slang_"):
        val = data.split("_")[1]
        settings = db.get_user_settings(user_id)
        settings.subtitle_lang = val
        db.save_user_settings(settings)
        lang_names = {"auto": "Auto-detect", "id": "Indonesia", "en": "English",
                      "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}
        await query.message.edit_text(
            f"✅ Bahasa subtitle: *{lang_names.get(val, val)}*",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    if data.startswith("sstyle_"):
        val = data.split("_")[1]
        settings = db.get_user_settings(user_id)
        settings.subtitle_style = val
        db.save_user_settings(settings)
        await query.message.edit_text(f"✅ Style subtitle: *{val}*", parse_mode=ParseMode.MARKDOWN)
        return

    if data.startswith("sfont_"):
        val = int(data.split("_")[1])
        settings = db.get_user_settings(user_id)
        settings.subtitle_font_size = val
        db.save_user_settings(settings)
        await query.message.edit_text(f"✅ Font size: *{val}*", parse_mode=ParseMode.MARKDOWN)
        return

    # ── Job configuration callbacks ──
    job = pending_jobs.get(user_id)
    if not job:
        await query.message.reply_text("⚠️ Tidak ada job yang tertunda. Kirim video baru.")
        return

    # Extract job_id from callback data (format: action_value_jobid)
    parts = data.split("_")

    if data.startswith("dur_"):
        val = int(parts[1])
        job.clip_duration = val
        pending_jobs[user_id] = job
        await query.answer(f"Durasi clip: {val}s ✅")

    elif data.startswith("num_"):
        val = int(parts[1])
        job.num_clips = val
        pending_jobs[user_id] = job
        await query.answer(f"Jumlah clip: {val} ✅")

    elif data.startswith("fmt_"):
        fmt_map = {"916": OutputFormat.VERTICAL, "169": OutputFormat.HORIZONTAL, "11": OutputFormat.SQUARE}
        val = fmt_map.get(parts[1], OutputFormat.VERTICAL)
        job.output_format = val
        pending_jobs[user_id] = job
        await query.answer(f"Format: {val.value} ✅")

    elif data.startswith("sub_toggle_"):
        job.subtitle_enabled = not job.subtitle_enabled
        pending_jobs[user_id] = job
        await query.answer(f"Subtitle: {'ON' if job.subtitle_enabled else 'OFF'} ✅")

    elif data.startswith("sil_toggle_"):
        job.remove_silence = not job.remove_silence
        pending_jobs[user_id] = job
        await query.answer(f"Hapus Silence: {'ON' if job.remove_silence else 'OFF'} ✅")

    elif data.startswith("cancel_"):
        pending_jobs.pop(user_id, None)
        # Cleanup temp dir
        job_dir = os.path.join(config.temp_dir, job.job_id)
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)
        await query.message.edit_text("🗑️ Proses dibatalkan.")
        return

    elif data.startswith("start_process_"):
        # ── START PROCESSING ──
        pending_jobs.pop(user_id, None)

        if not job.source_path or not os.path.exists(job.source_path):
            await query.message.edit_text(
                "❌ File source tidak ditemukan. Silakan kirim ulang video."
            )
            return

        # Create progress message
        progress_msg = await query.message.edit_text(
            "🚀 *Memulai pemrosesan video...*\n\n"
            "⏳ Status: Queued\n"
            "📊 Progress: 0%\n"
            "⏱️ Estimasi: menghitung...",
            parse_mode=ParseMode.MARKDOWN,
        )

        start_time = time.time()

        async def progress_callback(percent: float, status_text: str):
            """Update progress message in Telegram."""
            try:
                elapsed = time.time() - start_time
                if percent > 0:
                    estimated_total = elapsed / (percent / 100)
                    remaining = estimated_total - elapsed
                    eta_text = format_duration(remaining)
                else:
                    eta_text = "menghitung..."

                bar_length = 20
                filled = int(bar_length * percent / 100)
                bar = "█" * filled + "░" * (bar_length - filled)

                await progress_msg.edit_text(
                    f"🎬 *Processing Video*\n\n"
                    f"{status_text}\n\n"
                    f"[{bar}] {percent:.0f}%\n"
                    f"⏱️ Estimasi sisa: {eta_text}\n"
                    f"⏳ Elapsed: {format_duration(elapsed)}",
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                pass  # Ignore edit errors (rate limits etc)

        # Add to job queue
        await job_queue.add_job(job, progress_callback)

        # Wait for job to complete (poll status)
        while True:
            await asyncio.sleep(3)
            job_data = db.get_job(job.job_id)
            if not job_data:
                continue
            if job_data["status"] in ("completed", "failed", "cancelled"):
                break

        # ── Send Results ──
        job_data = db.get_job(job.job_id)

        if job_data["status"] == "completed":
            metadata = json.loads(job_data.get("metadata", "{}") or "{}")
            output_clips = metadata.get("output_clips", [])
            virality_scores = metadata.get("virality_scores", [])

            elapsed = time.time() - start_time
            avg_score = sum(virality_scores) / len(virality_scores) if virality_scores else 0

            await progress_msg.edit_text(
                f"✅ *Pemrosesan Selesai!*\n\n"
                f"📊 {len(output_clips)} clip dihasilkan\n"
                f"🔥 Rata-rata skor viralitas: {avg_score:.0f}/100\n"
                f"⏱️ Waktu proses: {format_duration(elapsed)}\n\n"
                f"Mengirim clip...",
                parse_mode=ParseMode.MARKDOWN,
            )

            # Send each clip
            for i, clip_path in enumerate(output_clips):
                if not os.path.exists(clip_path):
                    continue

                try:
                    clip_size = os.path.getsize(clip_path)
                    clip_info = get_video_info(clip_path)
                    score = virality_scores[i] if i < len(virality_scores) else 0

                    recs = ViralAnalyzer.get_platform_recommendations(clip_info.get("duration", 0))
                    hashtags = ViralAnalyzer.suggest_hashtags(
                        clip_info.get("duration", 0), score > 60                    )

                    caption = (
                        f"🎬 *Clip {i+1}/{len(output_clips)}*\n"
                        f"📎 ID: `{job.job_id}_{i:02d}`\n"
                        f"⏱️ Durasi: {format_duration(clip_info.get('duration', 0))}\n"
                        f"📁 Ukuran: {format_size(clip_size)}\n"
                        f"🔥 Virality Score: *{score:.0f}/100*\n\n"
                    )

                    for platform, rec in recs.items():
                        caption += f"{rec}\n"

                    caption += f"\n{' '.join(hashtags[:6])}"

                    await query.message.chat.send_action(ChatAction.UPLOAD_VIDEO)

                    # Telegram max video size = 50MB via bot API
                    if clip_size > 50 * 1024 * 1024:
                        await query.message.chat.send_message(
                            f"⚠️ Clip {i+1} terlalu besar ({format_size(clip_size)}) "
                            f"untuk dikirim via Telegram.\n"
                            f"Gunakan `/download {job.job_id}_{i:02d}` nanti.",
                            parse_mode=ParseMode.MARKDOWN,
                        )
                        continue

                    with open(clip_path, "rb") as f:
                        sent_msg = await query.message.chat.send_video(
                            video=f,
                            caption=caption,
                            parse_mode=ParseMode.MARKDOWN,
                            supports_streaming=True,
                            read_timeout=120,
                            write_timeout=120,
                        )

                    # Save telegram file_id for later re-download
                    if sent_msg.video:
                        db.save_clip(
                            clip_id=f"{job.job_id}_{i:02d}",
                            job_id=job.job_id,
                            user_id=job.user_id,
                            file_path=clip_path,
                            telegram_file_id=sent_msg.video.file_id,
                            duration=clip_info.get("duration", 0),
                            virality_score=score,
                        )

                    # Small delay to avoid rate limits
                    await asyncio.sleep(1.5)

                except Exception as e:
                    logger.error(f"Failed to send clip {i}: {e}")
                    await query.message.chat.send_message(
                        f"❌ Gagal mengirim clip {i+1}: `{str(e)[:100]}`",
                        parse_mode=ParseMode.MARKDOWN,
                    )

            # Send summary
            summary_parts = []
            for i, score in enumerate(virality_scores):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
                summary_parts.append(f"{medal} Clip {i+1}: {score:.0f}/100")

            # Sort by score for recommendation
            scored_clips = list(enumerate(virality_scores))
            scored_clips.sort(key=lambda x: x[1], reverse=True)
            best_idx = scored_clips[0][0] if scored_clips else 0

            summary = (
                "📊 *Ringkasan Hasil*\n\n"
                "*Skor Viralitas per Clip:*\n"
                + "\n".join(summary_parts) + "\n\n"
                f"⭐ *Rekomendasi:* Clip {best_idx + 1} memiliki potensi viral tertinggi!\n\n"
                f"💡 *Tips:*\n"
                f"• Upload clip terbaik sebagai video pertama\n"
                f"• Gunakan 3 detik pertama sebagai hook\n"
                f"• Posting di jam prime time (18:00-21:00)\n"
                f"• Tambahkan CTA di caption\n\n"
                f"📎 Re-download clip: `/download <clip_id>`\n"
                f"📜 Riwayat: /history"
            )

            await query.message.chat.send_message(
                summary, parse_mode=ParseMode.MARKDOWN
            )

        elif job_data["status"] == "failed":
            error_msg = job_data.get("error_message", "Unknown error")
            await progress_msg.edit_text(
                f"❌ *Pemrosesan Gagal*\n\n"
                f"Error: `{error_msg[:300]}`\n\n"
                f"💡 Kemungkinan penyebab:\n"
                f"• File video corrupt atau format tidak didukung\n"
                f"• Video terlalu panjang (max {format_duration(config.max_video_duration)})\n"
                f"• Disk space tidak cukup\n"
                f"• FFmpeg error\n\n"
                f"Coba kirim video lain atau hubungi admin.",
                parse_mode=ParseMode.MARKDOWN,
            )

        elif job_data["status"] == "cancelled":
            await progress_msg.edit_text(
                "🗑️ *Pemrosesan dibatalkan.*",
                parse_mode=ParseMode.MARKDOWN,
            )

        return

    # Update the inline keyboard to reflect new settings
    keyboard = [
        [
            InlineKeyboardButton(
                f"⏱️ {'✓ ' if job.clip_duration == 30 else ''}30s",
                callback_data=f"dur_30_{job.job_id}",
            ),
            InlineKeyboardButton(
                f"⏱️ {'✓ ' if job.clip_duration == 60 else ''}60s",
                callback_data=f"dur_60_{job.job_id}",
            ),
            InlineKeyboardButton(
                f"⏱️ {'✓ ' if job.clip_duration == 90 else ''}90s",
                callback_data=f"dur_90_{job.job_id}",
            ),
        ],
        [
            InlineKeyboardButton(
                f"🔢 {'��� ' if job.num_clips == 3 else ''}3",
                callback_data=f"num_3_{job.job_id}",
            ),
            InlineKeyboardButton(
                f"🔢 {'✓ ' if job.num_clips == 5 else ''}5",
                callback_data=f"num_5_{job.job_id}",
            ),
            InlineKeyboardButton(
                f"🔢 {'✓ ' if job.num_clips == 10 else ''}10",
                callback_data=f"num_10_{job.job_id}",
            ),
        ],
        [
            InlineKeyboardButton(
                f"📱 {'✓ ' if job.output_format == OutputFormat.VERTICAL else ''}9:16",
                callback_data=f"fmt_916_{job.job_id}",
            ),
            InlineKeyboardButton(
                f"🖥️ {'✓ ' if job.output_format == OutputFormat.HORIZONTAL else ''}16:9",
                callback_data=f"fmt_169_{job.job_id}",
            ),
            InlineKeyboardButton(
                f"⬜ {'✓ ' if job.output_format == OutputFormat.SQUARE else ''}1:1",
                callback_data=f"fmt_11_{job.job_id}",
            ),
        ],
        [
            InlineKeyboardButton(
                f"💬 Sub: {'ON ✅' if job.subtitle_enabled else 'OFF'}",
                callback_data=f"sub_toggle_{job.job_id}",
            ),
            InlineKeyboardButton(
                f"🤫 Silence: {'ON ✅' if job.remove_silence else 'OFF'}",
                callback_data=f"sil_toggle_{job.job_id}",
            ),
        ],
        [
            InlineKeyboardButton(
                "🚀 Proses Sekarang!",
                callback_data=f"start_process_{job.job_id}",
            ),
        ],
        [
            InlineKeyboardButton("❌ Batal", callback_data=f"cancel_{job.job_id}"),
        ],
    ]

    try:
        info = get_video_info(job.source_path) if job.source_path and os.path.exists(job.source_path) else {}

        text = (
            f"🎬 *Konfigurasi Video*\n\n"
            f"📐 Resolusi: {info.get('width', '?')}x{info.get('height', '?')}\n"
            f"⏱️ Durasi: {format_duration(info.get('duration', 0))}\n\n"
            f"*Pengaturan terpilih:*\n"
            f"├ ⏱️ Durasi clip: {job.clip_duration}s\n"
            f"├ 🔢 Jumlah clip: {job.num_clips}\n"
            f"├ 📐 Format: {job.output_format.value}\n"
            f"├ 💬 Subtitle: {'On' if job.subtitle_enabled else 'Off'}\n"
            f"└ 🤫 Hapus hening: {'On' if job.remove_silence else 'Off'}\n\n"
            f"Tekan *🚀 Proses Sekarang!* untuk mulai."
        )

        await query.message.edit_text(
            text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
    except Exception:
        pass  # Ignore edit errors


# ── Generic Text Message Handler ──

@auth_required
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle plain text messages — check if it's a URL."""
    text = update.message.text.strip()

    if is_url(text):
        await handle_url(update, context)
    else:
        await update.message.reply_text(
            "ℹ️ Kirim file video atau URL untuk memulai.\n"
            "Ketik /help untuk bantuan."
        )


# ── Error Handler ──

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle uncaught errors."""
    logger.error(f"Update {update} caused error: {context.error}", exc_info=context.error)

    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ Terjadi error. Silakan coba lagi.\n"
                "Jika masalah berlanjut, hubungi admin."
            )
        except Exception:
            pass


# ────────────────────────────────────────────────────────────
# SCHEDULED TASKS
# ────────────────────────────────────────────────────────────

async def scheduled_cleanup(context: ContextTypes.DEFAULT_TYPE):
    """Periodic cleanup of old temp and output files."""
    cleanup_old_files(config.temp_dir, config.auto_cleanup_hours)
    cleanup_old_files(config.output_dir, config.auto_cleanup_hours * 2)
    logger.info("Scheduled cleanup completed")


# ────────────────────────────────────────────────────────────
# APPLICATION BUILDER & MAIN
# ────────────────────────────────────────────────────────────

def build_application() -> Application:
    """Build and configure the Telegram bot application."""
    app = (
        Application.builder()
        .token(config.bot_token)
        .concurrent_updates(True)
        .read_timeout(60)
        .write_timeout(60)
        .connect_timeout(30)
        .build()
    )

    # ── Command handlers ──
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("download", cmd_download))
    app.add_handler(CommandHandler("process", cmd_process))

    # Admin commands
    app.add_handler(CommandHandler("adduser", cmd_adduser))
    app.add_handler(CommandHandler("removeuser", cmd_removeuser))
    app.add_handler(CommandHandler("listusers", cmd_listusers))

    # ── Message handlers ──
    app.add_handler(MessageHandler(
        filters.VIDEO | filters.Document.VIDEO | filters.Document.MimeType("video/mp4"),
        handle_video,
    ))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_text,
    ))

    # ── Callback query handler ──
    app.add_handler(CallbackQueryHandler(handle_callback))

    # ── Error handler ──
    app.add_error_handler(error_handler)

    # ── Scheduled jobs ──
    if app.job_queue:
        app.job_queue.run_repeating(
            scheduled_cleanup,
            interval=timedelta(hours=6),
            first=timedelta(minutes=10),
        )

    return app


async def post_init(application: Application):
    """Post-initialization: start the job queue."""
    await job_queue.start()
    logger.info("Job queue started in post_init")


async def post_shutdown(application: Application):
    """Post-shutdown: stop the job queue."""
    await job_queue.stop()
    logger.info("Job queue stopped in post_shutdown")


def main():
    """Main entry point — run the bot."""
    # ── Preflight checks ──
    print("=" * 55)
    print("🎬  TELEGRAM VIDEO CLIPPER BOT")
    print("=" * 55)

    # Check FFmpeg
    if not check_ffmpeg():
        print("❌ FFmpeg not found! Install it:")
        print("   Ubuntu:  sudo apt install ffmpeg")
        print("   macOS:   brew install ffmpeg")
        print("   Docker:  use the provided Dockerfile")
        sys.exit(1)
    print("✅ FFmpeg found")

    if not check_ffprobe():
        print("❌ FFprobe not found! (usually comes with FFmpeg)")
        sys.exit(1)
    print("✅ FFprobe found")

    # Check yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, timeout=5)
        print("✅ yt-dlp found")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("⚠️  yt-dlp not found — URL downloads will not work")
        print("   Install: pip install yt-dlp")

    # Check whisper
    whisper_ok = False
    try:
        import faster_whisper
        print(f"✅ faster-whisper found (model: {config.whisper_model})")
        whisper_ok = True
    except ImportError:
        try:
            import whisper
            print(f"✅ openai-whisper found (model: {config.whisper_model})")
            whisper_ok = True
        except ImportError:
            print("⚠️  No whisper library found — subtitles will be disabled")
            print("   Install: pip install faster-whisper")

    # Check OpenCV
    try:
        import cv2
        print("✅ OpenCV found — smart face tracking enabled")
    except ImportError:
        print("⚠️  OpenCV not found — face tracking disabled, using center crop")
        print("   Install: pip install opencv-python-headless")

    print(f"\n📂 Temp dir:   {os.path.abspath(config.temp_dir)}")
    print(f"📂 Output dir: {os.path.abspath(config.output_dir)}")
    print(f"📂 Database:   {os.path.abspath(config.db_path)}")
    print(f"👥 Whitelisted: {len(config.allowed_user_ids)} users")
    print(f"👑 Admins:      {len(config.admin_user_ids)} users")
    print(f"\n🚀 Bot is starting...\n")

    # ── Build & run ──
    app = build_application()
    app.post_init = post_init
    app.post_shutdown = post_shutdown

    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
