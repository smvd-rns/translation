import streamlit as st

st.set_page_config(
    page_title="Translation - Audio & Video Transcription",
    page_icon="🎙️",
    layout="centered"
)

import os
import gc
import glob
import time
import tempfile
import subprocess
import traceback
import requests

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

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose an audio or video file",
    type=["mp3", "mp4", "wav", "m4a", "aac", "flac", "ogg", "mov", "mkv"]
)

model_choice = st.selectbox(
    "Select Gemini Model",
    ["gemini-3.5-flash-lite"],
    index=0
)
st.caption("💡 **gemini-3.5-flash-lite** — 500 requests/day free quota, fast, handles hours of audio.")

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

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if uploaded_file.type.startswith("audio"):
        st.audio(uploaded_file)
    elif uploaded_file.type.startswith("video"):
        st.video(uploaded_file)

    st.caption(f"📁 File size: **{file_size_mb:.1f} MB**")

    if st.button("Start Transcription", type="primary"):
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"

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
                status_box.info("💾 Step 1/3: Saving uploaded file to local memory...")
                log(f"Saving '{uploaded_file.name}' ({file_size_mb:.1f} MB)...")
                flush_logs()
                with open(original_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                log(f"Saved file successfully.")
                flush_logs()

                # ── Step 2: Chunk media file into segments ────────────────────
                status_box.info(f"⚡ Step 2/3: Slicing audio into {chunk_duration}-minute chunks with ffmpeg...")
                log(f"Running ffmpeg to split audio into {chunk_duration}-minute chunks...")
                
                chunk_files = chunk_media_file(original_path, tmp_dir, chunk_minutes=chunk_duration)
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
                    fallback_models = ["groq-whisper", model_choice, "gemini-3.5-flash"]
                else:
                    fallback_models = [model_choice, "gemini-3.5-flash", "gemini-3.0-flash"]

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
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No transcript was generated. The file may have no speech or the model could not process it.")

            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
            finally:
                gc.collect()


