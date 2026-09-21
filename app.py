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
import streamlit.components.v1 as components


# Try loading the new Google GenAI SDK (supports AQ. keys and AIza. keys)
try:
    from google import genai
    from google.genai import types
    USE_NEW_SDK = True
except ImportError:
    import google.generativeai as legacy_genai
    USE_NEW_SDK = False

st.title("🎙️ Audio & Video Transcription App")
st.write("Upload an audio or video file to generate a transcript — powered by Google Gemini AI & Groq Whisper.")

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

# ── File uploader & Audio Compressor ─────────────────────────────────────────
component_dir = os.path.join(os.path.dirname(__file__), ".streamlit", "components", "browser_compressor")
browser_compressor = components.declare_component("browser_compressor", path=component_dir)

tab_compressor, tab_direct = st.tabs([
    "⚡ Fast Audio/Video Compressor (Any File Size)",
    "📁 Standard Upload (Max 30 MB)"
])

compressed_result = None
direct_file = None

with tab_compressor:
    compressed_result = browser_compressor()

with tab_direct:
    direct_file = st.file_uploader(
        "Choose an audio or video file (Max 30 MB)",
        type=["mp3", "mp4", "wav", "m4a", "aac", "flac", "ogg", "mov", "mkv"],
        help="Maximum file size for direct upload is 30 MB."
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

# Determine active file payload
active_file_bytes = None
active_file_name = None
file_size_mb = 0.0

if compressed_result and isinstance(compressed_result, dict) and compressed_result.get("base64"):
    active_file_name = compressed_result.get("name", "compressed_audio.mp3")
    active_file_bytes = base64.b64decode(compressed_result["base64"])
    file_size_mb = len(active_file_bytes) / (1024 * 1024)
    orig_mb = compressed_result.get("originalSizeMb", "?")
    
    st.success(
        f"⚡ **Compressed Audio Ready!**\n\n"
        f"📁 `{active_file_name}` | Size: **{file_size_mb:.1f} MB** (Compressed from **{orig_mb} MB** original file)"
    )
    st.audio(active_file_bytes, format="audio/mp3")

elif direct_file is not None:
    active_file_name = direct_file.name
    active_file_bytes = direct_file.getbuffer()
    file_size_mb = direct_file.size / (1024 * 1024)

    if direct_file.type.startswith("audio"):
        st.audio(direct_file)
    elif direct_file.type.startswith("video"):
        st.video(direct_file)

    st.caption(f"📁 File size: **{file_size_mb:.1f} MB**")

    MAX_FILE_SIZE_MB = 30
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"⚠️ **Direct File Size Limit Exceeded ({file_size_mb:.1f} MB / Max {MAX_FILE_SIZE_MB} MB)**\n\n"
            f"Please switch to the **'⚡ Fast Audio/Video Compressor'** tab above to compress your file."
        )
        st.stop()

if active_file_bytes is not None:

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
                    Your file is being uploaded, sliced into chunks, and processed in real-time. Please stay on this page.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        suffix = os.path.splitext(active_file_name)[1] or ".mp3"

        with tempfile.TemporaryDirectory() as tmp_dir:
            original_path = os.path.join(tmp_dir, "input" + suffix)

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

                # ── Step 1: Save uploaded file ────────────────────────────────
                status_box.info("💾 Step 1/3: Saving file to local memory...")
                log(f"Saving '{active_file_name}' ({file_size_mb:.1f} MB)...")
                flush_logs()
                with open(original_path, "wb") as f:
                    f.write(active_file_bytes)
                log("Saved file successfully.")
                flush_logs()

                # Fast ffmpeg audio track extraction & compression to 16kHz mono 32kbps MP3 (<3MB)
                compressed_path = os.path.join(tmp_dir, "compressed.mp3")
                cmd_compress = [
                    "ffmpeg", "-y", "-i", original_path,
                    "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
                    compressed_path
                ]
                try:
                    log("⚡ Optimizing audio stream with ffmpeg...")
                    flush_logs()
                    subprocess.run(cmd_compress, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                    if os.path.exists(compressed_path) and os.path.getsize(compressed_path) > 0:
                        c_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                        log(f"⚡ Audio stream ready! Optimized size: {c_mb:.1f} MB.")
                        processing_path = compressed_path
                    else:
                        processing_path = original_path
                except Exception as c_err:
                    log(f"Optimization note: using original file ({c_err}).")
                    processing_path = original_path
                flush_logs()

                # ── Step 2: Chunk media file into segments ────────────────────
                status_box.info(f"⚡ Step 2/3: Slicing audio into {chunk_duration}-minute chunks with ffmpeg...")
                log(f"Running ffmpeg to split audio into {chunk_duration}-minute chunks...")
                
                chunk_files = chunk_media_file(processing_path, tmp_dir, chunk_minutes=chunk_duration)


                total_chunks = len(chunk_files)
                log(f"Audio split complete! Created {total_chunks} chunk file(s).")
                status_box.info(f"✅ Prepared **{total_chunks} chunk(s)** for processing.")

                transcripts = []

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

                start_time = time.time()

                # ── Step 3: Process chunks in parallel (3 concurrent workers) ──
                log("⚡ Launching Parallel Processing (3 concurrent workers for max speed)...")
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
                    chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)

                    response = None
                    max_retries = 4

                    for attempt in range(max_retries):
                        current_model = fallback_models[attempt % len(fallback_models)]
                        audio_file = None

                        try:
                            # 🚀 Groq Whisper Fast Path (Multi-Key Round Robin)
                            if current_model == "groq-whisper" and groq_keys:
                                active_groq_key = groq_keys[idx % len(groq_keys)]
                                log(f"🚀 [Chunk {chunk_num}/{total_chunks}] Transcribing with Groq Whisper (Key #{idx % len(groq_keys) + 1}, Lang: {selected_language_code or 'auto'})...")
                                text_chunk = transcribe_with_groq(chunk_path, active_groq_key, language=selected_language_code)
                                if text_chunk:
                                    words = len(text_chunk.split())
                                    log(f"⚡ [Chunk {chunk_num}/{total_chunks}] Groq complete in 2s! Transcribed {words} words.")
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
                                # Capped max_output_tokens=3000 to prevent infinite hallucination loop stalls
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

                # Dynamically scale worker threads (e.g. 3 keys = 3 parallel worker threads)
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
                        
                        # Update live preview (chronological order)
                        ordered_texts = [transcripts_dict[i] for i in range(total_chunks) if i in transcripts_dict]
                        current_combined = "\n\n".join(ordered_texts)
                        transcript_preview.text_area("Live Output", current_combined, height=250, key=f"preview_{completed_count}")
                        
                        # Update progress bar
                        progress_bar.progress(completed_count / total_chunks)
                        status_box.info(f"⚡ Parallel Processing: **{completed_count} of {total_chunks} chunks completed**...")
                        flush_logs()

                flush_logs()
                total_time = int(time.time() - start_time)
                log(f"🎉 All {total_chunks} chunk(s) finished in {total_time}s!")
                flush_logs()
                status_box.empty()
                progress_bar.empty()

                # ── Step 4: Show final output ──────────────────────────────────
                final_text = "\n\n".join([transcripts_dict[i] for i in range(total_chunks) if i in transcripts_dict]).strip()

                if final_text:
                    st.success(f"🎉 Transcription complete! Processed {total_chunks} chunk(s) in {total_time} seconds.")
                    st.download_button(
                        label="📥 Download Complete Transcript (.txt)",
                        data=final_text,
                        file_name=f"{os.path.splitext(active_file_name)[0]}_transcript.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No transcript was generated. The file may have no speech or the model could not process it.")

            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
            finally:
                st.session_state.transcribing = False
                if 'progress_banner' in locals():
                    progress_banner.empty()
                gc.collect()


