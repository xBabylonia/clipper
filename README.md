# 🎬 Telegram Video Clipper Bot

Bot Telegram otomatis untuk mengubah video panjang menjadi konten viral pendek (TikTok, Reels, Shorts).

## ✨ Fitur

- ✂️ **Auto Scene Detection** — deteksi momen terbaik secara otomatis
- 📐 **Smart Crop** — konversi ke 9:16, 16:9, atau 1:1 dengan face tracking
- 💬 **AI Subtitles** — generate subtitle otomatis via Whisper AI
- 🔊 **Audio Processing** — hapus silence, normalisasi volume
- 🖼️ **Thumbnail Generation** — auto-generate thumbnail
- 📊 **Virality Score** — analisis potensi viral setiap clip
- 🔐 **Whitelist System** — akses terbatas untuk user tertentu
- 📥 **URL Support** — download dari YouTube, Instagram, TikTok, Twitter

## 🚀 Quick Start (15 menit)

### 1. Prerequisites

- Python 3.10+
- FFmpeg
- Telegram Bot Token (dari [@BotFather](https://t.me/BotFather))

### 2. Install

```bash
# Clone / download
git clone <repo-url> && cd telegram-video-clipper

# Install dependencies
pip install -r requirements.txt

# Copy & edit config
cp .env.example .env
nano .env  # isi TELEGRAM_BOT_TOKEN dan ALLOWED_USER_IDS

# Jalankan
python video_clipper_bot.py
```

### 3. Docker (Recommended)

```bash
cp .env.example .env
nano .env

docker-compose up -d --build
docker-compose logs -f
```

## 📱 Cara Pakai

1. Buka bot di Telegram
2. Ketik `/start`
3. Kirim video file atau URL YouTube/IG/TikTok
4. Pilih pengaturan (durasi, format, subtitle)
5. Tekan **🚀 Proses Sekarang!**
6. Terima clip-clip siap upload!

## ⚙️ Commands

| Command | Deskripsi |
|---------|-----------|
| `/start` | Pesan selamat datang |
| `/help` | Daftar perintah |
| `/process` | Mulai proses (opsional: timestamp manual) |
| `/settings` | Atur preferensi |
| `/status` | Cek antrian & job aktif |
| `/cancel` | Batalkan proses |
| `/history` | Riwayat video |
| `/download <id>` | Re-download clip |
| `/adduser <id>` | (Admin) Tambah user |
| `/removeuser <id>` | (Admin) Hapus user |
| `/listusers` | (Admin) Lihat daftar user |

### Manual Timestamp

```
/process 0:30-1:30 2:00-3:00 5:15-6:15
```

## 🔧 Konfigurasi `.env`

| Variable | Default | Deskripsi |
|----------|---------|-----------|
| `TELEGRAM_BOT_TOKEN` | - | Token bot dari BotFather |
| `ALLOWED_USER_IDS` | - | User ID yang diizinkan (comma-separated) |
| `ADMIN_USER_IDS` | - | Admin user IDs |
| `WHISPER_MODEL` | `base` | Model: tiny, base, small, medium, large-v3 |
| `WHISPER_DEVICE` | `cpu` | Device: cpu atau cuda |
| `DEFAULT_CLIP_DURATION` | `60` | Durasi clip default (detik) |
| `DEFAULT_NUM_CLIPS` | `5` | Jumlah clip default |
| `DEFAULT_OUTPUT_FORMAT` | `9:16` | Format: 9:16, 16:9, 1:1 |
| `DEFAULT_QUALITY_PRESET` | `high` | Quality: low, medium, high, ultra |

## 🐛 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `FFmpeg not found` | `sudo apt install ffmpeg` / `brew install ffmpeg` |
| `yt-dlp error` | `pip install -U yt-dlp` |
| Bot tidak merespons | Cek `TELEGRAM_BOT_TOKEN` dan whitelist |
| Subtitle tidak muncul | Install `faster-whisper`: `pip install faster-whisper` |
| Proses lambat | Gunakan GPU (`WHISPER_DEVICE=cuda`) atau model lebih kecil |
| File terlalu besar | Telegram limit 50MB, gunakan quality preset `medium` |
| Out of disk | Kurangi `AUTO_CLEANUP_HOURS` atau `MAX_TEMP_SIZE_GB` |

## 📄 License

MIT
