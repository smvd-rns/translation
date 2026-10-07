import streamlit as st

st.set_page_config(
    page_title="Translation - Audio & Video Transcription",
    page_icon="🎙️",
    layout="centered"
)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

st.markdown("""
<style>
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes pulse-glow {
    0% { box-shadow: 0 0 8px rgba(255, 75, 75, 0.3); border-color: rgba(255, 75, 75, 0.5); }
    50% { box-shadow: 0 0 20px rgba(255, 75, 75, 0.7); border-color: rgba(255, 75, 75, 0.9); }
    100% { box-shadow: 0 0 8px rgba(255, 75, 75, 0.3); border-color: rgba(255, 75, 75, 0.5); }
}
.transcribing-banner {
    background: linear-gradient(135deg, rgba(255, 75, 75, 0.12), rgba(120, 50, 255, 0.12));
    border: 1px dashed #ff4b4b;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 15px 0 25px 0;
    display: flex;
    align-items: center;
    gap: 16px;
    animation: pulse-glow 2.5s infinite ease-in-out;
}
.loader-spinner {
    width: 28px;
    height: 28px;
    border: 3px solid rgba(255, 75, 75, 0.25);
    border-top: 3px solid #ff4b4b;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

import os
import gc
import glob
import time
import base64
import tempfile
import subprocess
import traceback
import requests
import re
import streamlit.components.v1 as components

import yt_dlp
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False


def extract_youtube_video_id(url):
    """Extracts 11-character YouTube video ID from various link formats."""
    if not url:
        return None
    pattern = r'(?:v=|\/([0-9A-Za-z_-]{11}).*|youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1) or match.group(2)
    return None


def fetch_youtube_captions(video_id, target_lang=None):
    """Fetches existing captions/subtitles directly from YouTube API if available."""
    if not YOUTUBE_TRANSCRIPT_AVAILABLE or not video_id:
        return None

    # Method 1: Standard YouTubeTranscriptApi static methods
    try:
        if hasattr(YouTubeTranscriptApi, 'get_transcript'):
            langs = [target_lang, 'en', 'hi', 'es'] if target_lang else ['en', 'hi', 'es']
            try:
                data = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
                return "\n\n".join([item['text'] for item in data if item.get('text')])
            except Exception:
                data = YouTubeTranscriptApi.get_transcript(video_id)
                return "\n\n".join([item['text'] for item in data if item.get('text')])
    except Exception:
        pass

    # Method 2: list_transcripts static method
    try:
        if hasattr(YouTubeTranscriptApi, 'list_transcripts'):
            t_list = YouTubeTranscriptApi.list_transcripts(video_id)
            if target_lang:
                try:
                    t = t_list.find_transcript([target_lang])
                    data = t.fetch()
                    return "\n\n".join([item['text'] for item in data if item.get('text')])
                except Exception:
                    pass
            try:
                t = t_list.find_transcript(['en'])
                data = t.fetch()
                return "\n\n".join([item['text'] for item in data if item.get('text')])
            except Exception:
                for t in t_list:
                    data = t.fetch()
                    return "\n\n".join([item['text'] for item in data if item.get('text')])
    except Exception:
        pass

    # Method 3: Instance API fallback
    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, 'fetch'):
            data = api.fetch(video_id)
            return "\n\n".join([item['text'] for item in data if item.get('text')])
        elif hasattr(api, 'list'):
            t_list = api.list(video_id)
            for t in t_list:
                data = t.fetch()
                return "\n\n".join([item['text'] for item in data if item.get('text')])
    except Exception:
        pass

    return None


def download_youtube_audio(url, output_dir):
    """Downloads audio from YouTube link using yt-dlp with player_client fallback for cloud servers."""
    output_template = os.path.join(output_dir, "yt_audio.%(ext)s")

    # Rotate player_client options (android, ios, mweb, tv, web_embedded) to bypass cloud bot detection
    client_configs = [
        ['android', 'ios'],
        ['mweb', 'tv'],
        ['web_embedded'],
        ['android'],
        ['ios'],
        ['web']
    ]

    last_err = None
    for clients in client_configs:
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_template,
                'nocheckcertificate': True,
                'extractor_args': {
                    'youtube': {
                        'player_client': clients
                    }
                },
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
                    'Accept-Language': 'en-US,en;q=0.9',
                },
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            mp3_path = os.path.join(output_dir, "yt_audio.mp3")
            if os.path.exists(mp3_path):
                return mp3_path

            files = glob.glob(os.path.join(output_dir, "yt_audio.*"))
            if files:
                return files[0]
        except Exception as e:
            last_err = e
            continue

    if last_err:
        raise last_err
    raise Exception("Could not download audio from YouTube URL.")



# Try loading the new Google GenAI SDK (supports AQ. keys and AIza. keys)
try:
    from google import genai
    from google.genai import types
    USE_NEW_SDK = True
except ImportError:
    import google.generativeai as legacy_genai
    USE_NEW_SDK = False

st.title("🎙️ Audio & Video Transcription App")
st.write("Upload an audio/video file or paste a direct **YouTube link** to generate a transcript — powered by Google Gemini AI & Groq Whisper.")


# ── API Key Configuration ─────────────────────────────────────────────────────
api_key = os.environ.get("GEMINI_API_KEY", "").strip().strip("'\"")
raw_groq_keys = os.environ.get("GROQ_API_KEY", "").strip().strip("'\"")

if not api_key:
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown(
        "Get your **free** Gemini API key at "
        "[aistudio.google.com/apikey](https://aistudio.google.com/apikey)"
    )
    user_api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AQ... or AIza...",
        help="Enter your API key from aistudio.google.com/apikey"
    )
    api_key = user_api_key.strip().strip("'\"")

if not raw_groq_keys:
    groq_input = st.sidebar.text_input(
        "Groq API Key(s) (Comma-separated for multi-key speed)",
        type="password",
        placeholder="gsk_key1, gsk_key2, gsk_key3",
        help="Free key(s) from console.groq.com — transcribes whole 1-hr audio in 10 seconds!"
    )
    raw_groq_keys = groq_input.strip().strip("'\"")

# Parse list of Groq keys for multi-key round-robin load balancing
groq_keys = [k.strip() for k in raw_groq_keys.split(",") if k.strip()]

if not api_key and not groq_keys:
    st.info(
        "👈 Enter your **Gemini API Key** or **Groq API Key** in the sidebar to get started."
    )
    st.stop()

if USE_NEW_SDK and api_key:
    client = genai.Client(api_key=api_key)
elif api_key:
    legacy_genai.configure(api_key=api_key)

def transcribe_with_groq(chunk_path, key, language=None):
    """Transcribes audio using Groq Whisper API (whisper-large-v3-turbo)."""
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {key}"}
    with open(chunk_path, "rb") as f:
        files = {"file": (os.path.basename(chunk_path), f, "audio/mp3")}
        data = {"model": "whisper-large-v3-turbo"}
        if language:
            data["language"] = language
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("text", "")
        else:
            raise Exception(f"Groq API Error {resp.status_code}: {resp.text}")

# ── Input Source Selection ───────────────────────────────────────────────────
input_source = st.radio(
    "📥 Select Input Source",
    options=["📁 Upload Audio / Video File", "🔗 Paste YouTube Video Link"],
    horizontal=True
)

uploaded_file = None
youtube_url = ""
yt_video_id = None
yt_transcription_mode = "⚡ Auto-Fetch YouTube Captions (Instant - 1 Second)"

if input_source == "📁 Upload Audio / Video File":
    uploaded_file = st.file_uploader(
        "Choose an audio or video file (Max 30 MB)",
        type=["mp3", "mp4", "wav", "m4a", "aac", "flac", "ogg", "mov", "mkv"],
        help="Maximum file size supported is 30 MB."
    )
    with st.expander("💡 Have a file larger than 30 MB? Click here for quick compression steps"):
        st.markdown(
            "1. 🌐 Go to **[online-audio-converter.com](https://online-audio-converter.com/)**\n"
            "2. Click **'Open files'** and select your file.\n"
            "3. Click **'Advanced settings'**:\n"
            "   - Set **Bitrate** to **`32 kbps`**\n"
            "   - Set **Channels** to **`1`** (Mono)\n"
            "4. Click **'Convert'** and download your compressed file under 30 MB! 🚀"
        )
else:
    youtube_url = st.text_input(
        "🔗 Enter YouTube Video Link",
        placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
        help="Paste any public YouTube lecture, presentation, or video URL."
    ).strip()

    if youtube_url:
        yt_video_id = extract_youtube_video_id(youtube_url)
        if yt_video_id:
            components.html(
                f'''<iframe width="100%" height="350" src="https://www.youtube-nocookie.com/embed/{yt_video_id}" 
                title="YouTube video player" frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
                referrerpolicy="strict-origin-when-cross-origin" allowfullscreen style="border-radius: 10px; width: 100%;"></iframe>''',
                height=360
            )
        else:
            st.warning("⚠️ Invalid YouTube URL format. Please enter a valid YouTube video link.")

    yt_transcription_mode = st.radio(
        "⚡ YouTube Processing Method",
        options=[
            "⚡ Auto-Fetch YouTube Captions (Instant - 1 Second)",
            "🧠 Full AI Audio Transcription (Groq Whisper / Gemini AI)"
        ],
        help="Auto-Fetch retrieves official or auto-generated YouTube captions instantly. Full AI download extracts audio and runs Groq Whisper / Gemini AI."
    )


language_options = {
    "English 🇬🇧 (en)": "en",
    "Auto-Detect (Whisper auto-detects per chunk)": None,
    "Hindi 🇮🇳 (hi)": "hi",
    "Urdu 🇵🇰 (ur)": "ur",
    "Marathi 🇮🇳 (mr)": "mr",
    "Bengali 🇮🇳 (bn)": "bn",
    "Tamil 🇮🇳 (ta)": "ta",
    "Telugu 🇮🇳 (te)": "te",
    "Gujarati 🇮🇳 (gu)": "gu",
    "Kannada 🇮🇳 (kn)": "kn",
    "Malayalam 🇮🇳 (ml)": "ml",
    "Punjabi 🇮🇳 (pa)": "pa",
    "Spanish 🇪🇸 (es)": "es",
    "French 🇫🇷 (fr)": "fr",
    "German 🇩🇪 (de)": "de",
    "Arabic 🇸🇦 (ar)": "ar",
    "Russian 🇷🇺 (ru)": "ru",
    "Japanese 🇯🇵 (ja)": "ja",
    "Chinese 🇨🇳 (zh)": "zh"
}

selected_lang_label = st.selectbox(
    "🌐 Audio Primary Language (Fixes language mixing across chunks)",
    options=list(language_options.keys()),
    index=0,
    help="Selecting a specific language forces Whisper & Gemini to process all chunks in that language and native script, preventing unwanted translation or script switching."
)
selected_language_code = language_options[selected_lang_label]

chunk_duration = st.slider(
    "Audio Chunk Duration (minutes)",
    min_value=5,
    max_value=30,
    value=15,
    step=1,
    help="Long audio files will be automatically split into chunks of this size for optimal processing within Gemini rate limits."
)


def chunk_media_file(input_path, tmp_dir, chunk_minutes=10):
    """Slices input media into chunks in <1 second using ultra-fast ffmpeg stream copy."""
    ext = os.path.splitext(input_path)[1] or ".mp3"
    chunk_pattern_copy = os.path.join(tmp_dir, f"chunk_%03d{ext}")
    segment_seconds = str(chunk_minutes * 60)

    # ⚡ Ultra-fast Stream Copy (-c copy) — takes ~0.5s for a 1-hour file!
    cmd_copy = [
        "ffmpeg", "-y", "-i", input_path,
        "-f", "segment",
        "-segment_time", segment_seconds,
        "-c", "copy",
        chunk_pattern_copy
    ]

    try:
        res = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        chunks = sorted(glob.glob(os.path.join(tmp_dir, f"chunk_*{ext}")))
        if chunks and len(chunks) > 0 and os.path.getsize(chunks[0]) > 0:
            return chunks
    except Exception:
        pass

    # Fallback re-encoding if stream copy is incompatible with container
    chunk_pattern_mp3 = os.path.join(tmp_dir, "chunk_%03d.mp3")
    cmd_reencode = [
        "ffmpeg", "-y", "-i", input_path,
        "-f", "segment",
        "-segment_time", segment_seconds,
        "-c:a", "libmp3lame", "-b:a", "64k",
        chunk_pattern_mp3
    ]
    try:
        res = subprocess.run(cmd_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        chunks = sorted(glob.glob(os.path.join(tmp_dir, "chunk_*.mp3")))
        if chunks:
            return chunks
    except Exception:
        pass

    return [input_path]


# Check if ready to transcribe
can_proceed = False
file_size_mb = 0.0
download_filename = "transcript.txt"

if input_source == "📁 Upload Audio / Video File" and uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if uploaded_file.type.startswith("audio"):
        st.audio(uploaded_file)
    elif uploaded_file.type.startswith("video"):
        st.video(uploaded_file)

    st.caption(f"📁 File size: **{file_size_mb:.1f} MB**")

    MAX_FILE_SIZE_MB = 30
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"⚠️ **File Size Limit Exceeded ({file_size_mb:.1f} MB / Max 30 MB)**\n\n"
            f"Your uploaded file is **{file_size_mb:.1f} MB**, which exceeds the maximum allowed limit of **30 MB**.\n\n"
            f"--- \n\n"
            f"### 💡 How to compress your file under 30 MB in 4 simple steps:\n\n"
            f"1. 🌐 Go to **[online-audio-converter.com](https://online-audio-converter.com/)**\n"
            f"2. Click **'Open files'** and select your audio or video file.\n"
            f"3. Click **'Advanced settings'** button:\n"
            f"   - Change **Bitrate** dropdown to **`32 kbps`**.\n"
            f"   - Change **Channels** dropdown to **`1`** (Mono).\n"
            f"4. Click **'Convert'** and download your compressed file *(brings files down under 30 MB even for 1.5 hour long recordings!)*.\n\n"
            f"Once downloaded, upload your compressed file here to transcribe! 🚀"
        )
        st.stop()

    download_filename = f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt"
    can_proceed = True

elif input_source == "🔗 Paste YouTube Video Link" and youtube_url:
    if not yt_video_id:
        st.warning("⚠️ Please provide a valid YouTube URL to proceed.")
    else:
        download_filename = f"youtube_{yt_video_id}_transcript.txt"
        can_proceed = True


if can_proceed:
    if "transcribing" not in st.session_state:
        st.session_state.transcribing = False

    btn_container = st.empty()

    if st.session_state.transcribing:
        btn_container.button("⏳ Transcription in Progress...", disabled=True, type="secondary")
        start_clicked = False
    else:
        start_clicked = btn_container.button("Start Transcription", type="primary")

    if start_clicked:
        st.session_state.transcribing = True
        btn_container.button("⏳ Transcription in Progress...", disabled=True, type="secondary")

        progress_banner = st.empty()
        progress_banner.markdown("""
        <div class="transcribing-banner">
            <div class="loader-spinner"></div>
            <div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #ff4b4b; display: flex; align-items: center; gap: 8px;">
                    ⚡ Transcription in Progress...
                </div>
                <div style="font-size: 0.88rem; color: #dddddd; margin-top: 4px;">
                    Your audio is being processed and transcribed in real-time. Please stay on this page.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                status_box = st.empty()
                progress_bar = st.progress(0)

                st.subheader("📋 Live Activity Logs")
                log_box = st.empty()

                st.subheader("📝 Live Transcript Preview")
                transcript_preview = st.empty()

                logs = []
                import queue
                log_queue = queue.Queue()

                def log(msg):
                    ts = time.strftime("%H:%M:%S")
                    log_queue.put(f"[{ts}] {msg}")

                def flush_logs():
                    updated = False
                    while not log_queue.empty():
                        try:
                            logs.append(log_queue.get_nowait())
                            updated = True
                        except Exception:
                            break
                    if updated:
                        log_box.code("\n".join(logs[-40:]), language="text")

                final_text = ""
                start_time = time.time()

                # ── Fast Path: YouTube Auto-Fetch Captions ──────────────────────
                if input_source == "🔗 Paste YouTube Video Link" and "Auto-Fetch" in yt_transcription_mode:
                    status_box.info("⚡ Attempting instant fetch of YouTube captions...")
                    log(f"Fetching official/auto captions for YouTube Video ID '{yt_video_id}'...")
                    flush_logs()

                    yt_captions = fetch_youtube_captions(yt_video_id, target_lang=selected_language_code)
                    if yt_captions:
                        final_text = yt_captions
                        log(f"🎉 Successfully fetched YouTube captions instantly!")
                        flush_logs()

                    else:
                        log("⚠️ No captions found via YouTube API. Falling back to Full AI Audio Download & Transcription...")
                        flush_logs()

                # ── Full AI Audio Pipeline (Upload File OR YouTube Fallback) ───
                if not final_text:
                    original_path = ""

                    if input_source == "📁 Upload Audio / Video File":
                        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"
                        original_path = os.path.join(tmp_dir, "input" + suffix)

                        status_box.info("💾 Step 1/3: Saving uploaded file to local memory...")
                        log(f"Saving '{uploaded_file.name}' ({file_size_mb:.1f} MB)...")
                        flush_logs()
                        with open(original_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        log("Saved file successfully.")
                        flush_logs()

                    else:  # YouTube Download
                        status_box.info("📥 Step 1/3: Downloading audio track from YouTube link...")
                        log(f"Downloading YouTube audio from '{youtube_url}' using yt-dlp...")
                        flush_logs()

                        try:
                            original_path = download_youtube_audio(youtube_url, tmp_dir)
                            yt_audio_size_mb = os.path.getsize(original_path) / (1024 * 1024)
                            log(f"Downloaded YouTube audio track ({yt_audio_size_mb:.1f} MB).")
                            flush_logs()
                        except Exception as yt_err:
                            err_str = str(yt_err)
                            log(f"❌ YouTube download error: {err_str}")
                            flush_logs()
                            if "SSL" in err_str or "WRONG_VERSION_NUMBER" in err_str or "Seqrite" in err_str:
                                raise Exception(
                                    "🔒 Local Network Restriction Detected:\n"
                                    "Your laptop's local network/antivirus (e.g. Seqrite Endpoint Protection) blocks outbound YouTube connections from Python scripts.\n\n"
                                    "🌐 LIVE DEPLOYMENT NOTE: Once pushed live (e.g. Render, Streamlit Cloud, Railway, AWS), "
                                    "cloud servers have unrestricted internet access, so YouTube link transcription will work 100% automatically!\n\n"
                                    "💡 For testing on this laptop: Please download the video/audio file locally and use the 'Upload Audio / Video File' tab."
                                )
                            else:
                                raise yt_err

                    # ── Step 2: Chunk media file into segments ────────────────
                    status_box.info(f"⚡ Step 2/3: Slicing audio into {chunk_duration}-minute chunks with ffmpeg...")
                    log(f"Running ffmpeg to split audio into {chunk_duration}-minute chunks...")

                    chunk_files = chunk_media_file(original_path, tmp_dir, chunk_minutes=chunk_duration)
                    total_chunks = len(chunk_files)
                    log(f"Audio split complete! Created {total_chunks} chunk file(s).")
                    status_box.info(f"✅ Prepared **{total_chunks} chunk(s)** for processing.")

                    lang_rule = ""
                    if selected_language_code:
                        lang_name = selected_lang_label.split(" (")[0]
                        lang_rule = f"\n4. The primary spoken language is {lang_name}. Transcribe strictly in {lang_name} using its native script."

                    prompt = (
                        "Please transcribe the speech in this audio file accurately. "
                        "Important rules:\n"
                        "1. Output only the spoken text preserving natural paragraph breaks.\n"
                        "2. If there are repeating chants, mantras, or background music, transcribe the words accurately without repeating the same line over and over endlessly.\n"
                        "3. Do not add commentary, timestamps, or extra formatting."
                        f"{lang_rule}"
                    )

                    # ── Step 3: Process chunks in parallel ────────────────────
                    log("⚡ Launching Parallel Processing...")
                    import concurrent.futures

                    if groq_keys:
                        fallback_models = ["groq-whisper", DEFAULT_GEMINI_MODEL, "gemini-3.5-flash"]
                    else:
                        fallback_models = [DEFAULT_GEMINI_MODEL, "gemini-3.5-flash", "gemini-3.0-flash"]

                    transcripts_dict = {}
                    completed_count = 0

                    def process_chunk_worker(chunk_info):
                        idx, chunk_path = chunk_info
                        chunk_num = idx + 1
                        max_retries = 4

                        for attempt in range(max_retries):
                            current_model = fallback_models[attempt % len(fallback_models)]
                            audio_file = None

                            try:
                                if current_model == "groq-whisper" and groq_keys:
                                    active_groq_key = groq_keys[idx % len(groq_keys)]
                                    log(f"🚀 [Chunk {chunk_num}/{total_chunks}] Transcribing with Groq Whisper (Key #{idx % len(groq_keys) + 1}, Lang: {selected_language_code or 'auto'})...")
                                    text_chunk = transcribe_with_groq(chunk_path, active_groq_key, language=selected_language_code)
                                    if text_chunk:
                                        words = len(text_chunk.split())
                                        log(f"⚡ [Chunk {chunk_num}/{total_chunks}] Groq complete! Transcribed {words} words.")
                                        return (idx, text_chunk)

                                log(f"[Chunk {chunk_num}/{total_chunks}] Uploading to Gemini ({current_model})...")

                                if USE_NEW_SDK:
                                    audio_file = client.files.upload(file=chunk_path)
                                    log(f"[Chunk {chunk_num}/{total_chunks}] Uploaded. Storage ID: {audio_file.name}")

                                    wait_count = 0
                                    while hasattr(audio_file, 'state') and str(getattr(audio_file.state, 'name', audio_file.state)) == "PROCESSING":
                                        time.sleep(2)
                                        audio_file = client.files.get(name=audio_file.name)
                                        wait_count += 1
                                        if wait_count > 20:
                                            break

                                    log(f"🧠 [Chunk {chunk_num}/{total_chunks}] Transcribing with {current_model}...")
                                    config = types.GenerateContentConfig(
                                        temperature=0.0,
                                        max_output_tokens=3000,
                                        system_instruction="You are a precise audio transcription expert. Transcribe spoken words accurately. Never repeat phrases endlessly during music, chants, or silence."
                                    )
                                    response = client.models.generate_content(
                                        model=current_model,
                                        contents=[audio_file, prompt],
                                        config=config
                                    )
                                else:
                                    audio_file = legacy_genai.upload_file(path=chunk_path)
                                    log(f"[Chunk {chunk_num}/{total_chunks}] Uploaded. Storage ID: {audio_file.name}")

                                    wait_count = 0
                                    while audio_file.state.name == "PROCESSING":
                                        time.sleep(2)
                                        audio_file = legacy_genai.get_file(audio_file.name)
                                        wait_count += 1
                                        if wait_count > 20:
                                            break

                                    log(f"🧠 [Chunk {chunk_num}/{total_chunks}] Transcribing with {current_model}...")
                                    model = legacy_genai.GenerativeModel(
                                        model_name=current_model,
                                        generation_config={"temperature": 0.0, "max_output_tokens": 3000}
                                    )
                                    response = model.generate_content(
                                        [prompt, audio_file],
                                        request_options={"timeout": 600}
                                    )

                                if response and response.text:
                                    text_chunk = response.text.strip()
                                    words = len(text_chunk.split())
                                    log(f"✅ [Chunk {chunk_num}/{total_chunks}] Complete! Transcribed {words} words.")
                                    return (idx, text_chunk)

                            except Exception as err:
                                err_msg = str(err)
                                if ("503" in err_msg or "UNAVAILABLE" in err_msg or "429" in err_msg or "high demand" in err_msg) and attempt < max_retries - 1:
                                    next_model = fallback_models[(attempt + 1) % len(fallback_models)]
                                    log(f"⚠️ [Chunk {chunk_num}/{total_chunks}] Server busy (503). Retrying with '{next_model}'...")
                                    time.sleep(3)
                                else:
                                    if attempt == max_retries - 1:
                                        log(f"❌ [Chunk {chunk_num}/{total_chunks}] Error: {err}")
                                        raise err
                                    time.sleep(3)
                            finally:
                                if audio_file:
                                    try:
                                        if USE_NEW_SDK:
                                            client.files.delete(name=audio_file.name)
                                        else:
                                            legacy_genai.delete_file(audio_file.name)
                                    except Exception:
                                        pass

                        return (idx, "")

                    num_workers = min(6, max(3, len(groq_keys)))
                    log(f"⚡ Running with {num_workers} parallel workers across {max(1, len(groq_keys))} key(s)...")

                    chunk_tuples = list(enumerate(chunk_files))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                        future_to_chunk = {executor.submit(process_chunk_worker, item): item for item in chunk_tuples}

                        for future in concurrent.futures.as_completed(future_to_chunk):
                            flush_logs()
                            completed_count += 1
                            idx, text_chunk = future.result()
                            transcripts_dict[idx] = text_chunk

                            ordered_texts = [transcripts_dict[i] for i in range(total_chunks) if i in transcripts_dict]
                            current_combined = "\n\n".join(ordered_texts)
                            transcript_preview.text_area("Live Output", current_combined, height=250, key=f"preview_{completed_count}")

                            progress_bar.progress(completed_count / total_chunks)
                            status_box.info(f"⚡ Parallel Processing: **{completed_count} of {total_chunks} chunks completed**...")
                            flush_logs()

                    final_text = "\n\n".join([transcripts_dict[i] for i in range(total_chunks) if i in transcripts_dict]).strip()

                flush_logs()
                total_time = int(time.time() - start_time)
                status_box.empty()
                progress_bar.empty()

                # ── Step 4: Output ─────────────────────────────────────────────
                if final_text:
                    transcript_preview.text_area("Final Transcript", final_text, height=300, key="final_transcript")
                    st.success(f"🎉 Transcription complete in {total_time} seconds!")
                    st.download_button(
                        label="📥 Download Complete Transcript (.txt)",
                        data=final_text,
                        file_name=download_filename,
                        mime="text/plain"
                    )
                else:
                    st.warning("No transcript was generated. The video/audio may have no speech or could not be processed.")

            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
            finally:
                st.session_state.transcribing = False
                if 'progress_banner' in locals():
                    progress_banner.empty()
                gc.collect()


