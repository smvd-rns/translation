import os
import gc
import glob
import time
import tempfile
import subprocess
import traceback
import streamlit as st

# Try loading the new Google GenAI SDK (supports AQ. keys and AIza. keys)
try:
    from google import genai
    from google.genai import types
    USE_NEW_SDK = True
except ImportError:
    import google.generativeai as legacy_genai
    USE_NEW_SDK = False

st.set_page_config(
    page_title="Audio & Video Transcription App",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Audio & Video Transcription App")
st.write("Upload an audio or video file to generate a transcript — powered by Google Gemini AI.")

# ── Sidebar: API Key ─────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown(
    "Get your **free** Gemini API key at "
    "[aistudio.google.com/apikey](https://aistudio.google.com/apikey)"
)

env_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
user_api_key = st.sidebar.text_input(
    "Gemini API Key",
    value=env_api_key,
    type="password",
    placeholder="AQ... or AIza...",
    help="Enter your API key from aistudio.google.com/apikey"
)

api_key = user_api_key.strip().strip("'\"")

if not api_key:
    st.info(
        "👈 Enter your **Gemini API Key** in the sidebar to get started.\n\n"
        "🔑 Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — takes 30 seconds."
    )
    st.stop()

if USE_NEW_SDK:
    client = genai.Client(api_key=api_key)
else:
    legacy_genai.configure(api_key=api_key)

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

chunk_duration = st.slider(
    "Audio Chunk Duration (minutes)",
    min_value=5,
    max_value=20,
    value=10,
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

                def log(msg):
                    ts = time.strftime("%H:%M:%S")
                    logs.append(f"[{ts}] {msg}")
                    log_box.code("\n".join(logs[-15:]), language="text")

                # ── Step 1: Save uploaded file ────────────────────────────────
                status_box.info("💾 Step 1/3: Saving uploaded file to local memory...")
                log(f"Saving '{uploaded_file.name}' ({file_size_mb:.1f} MB)...")
                with open(original_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                log(f"Saved file successfully.")

                # ── Step 2: Chunk media file into segments ────────────────────
                status_box.info(f"⚡ Step 2/3: Slicing audio into {chunk_duration}-minute chunks with ffmpeg...")
                log(f"Running ffmpeg to split audio into {chunk_duration}-minute chunks...")
                
                chunk_files = chunk_media_file(original_path, tmp_dir, chunk_minutes=chunk_duration)
                total_chunks = len(chunk_files)
                log(f"Audio split complete! Created {total_chunks} chunk file(s).")
                status_box.info(f"✅ Prepared **{total_chunks} chunk(s)** for processing.")

                transcripts = []

                prompt = (
                    "Please transcribe the speech in this audio file accurately. "
                    "Important rules:\n"
                    "1. Output only the spoken text preserving natural paragraph breaks.\n"
                    "2. If there are repeating chants, mantras, or background music, transcribe the words accurately without repeating the same line over and over endlessly.\n"
                    "3. Do not add commentary, timestamps, or extra formatting."
                )

                start_time = time.time()

                # ── Step 3: Process chunks sequentially ───────────────────────
                for idx, chunk_path in enumerate(chunk_files):
                    chunk_num = idx + 1
                    chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
                    
                    # Estimate remaining time
                    elapsed = time.time() - start_time
                    if idx > 0:
                        avg_per_chunk = elapsed / idx
                        rem_chunks = total_chunks - idx
                        eta_sec = int(avg_per_chunk * rem_chunks)
                        eta_str = f"~{eta_sec}s remaining"
                    else:
                        eta_str = "calculating ETA..."

                    status_box.info(f"📤 Processing Chunk **{chunk_num} of {total_chunks}** ({eta_str})...")
                    log(f"--- Chunk {chunk_num}/{total_chunks} ({chunk_size_mb:.1f} MB) ---")
                    log(f"Uploading chunk {chunk_num} to Gemini Files API...")

                    if USE_NEW_SDK:
                        audio_file = client.files.upload(file=chunk_path)
                        log(f"Uploaded to Gemini. Storage ID: {audio_file.name}")

                        # Poll for processing status if needed
                        wait_count = 0
                        while hasattr(audio_file, 'state') and str(getattr(audio_file.state, 'name', audio_file.state)) == "PROCESSING":
                            log(f"Gemini processing chunk audio... (waiting {wait_count*2}s)")
                            time.sleep(2)
                            audio_file = client.files.get(name=audio_file.name)
                            wait_count += 1
                            if wait_count > 30:
                                log(f"❌ Error: Chunk {chunk_num} processing timed out.")
                                st.error(f"Chunk {chunk_num} processing timed out.")
                                st.stop()

                        log(f"🧠 Transcribing speech with gemini-3.5-flash-lite (temp=0.0)...")
                        
                        try:
                            # Use temperature=0.0 for deterministic greedy decoding (prevents repetition loops)
                            config = types.GenerateContentConfig(
                                temperature=0.0,
                                system_instruction="You are a precise audio transcription expert. Transcribe spoken words accurately. Never repeat phrases endlessly during music, chants, or silence."
                            )
                            response = client.models.generate_content(
                                model=model_choice,
                                contents=[audio_file, prompt],
                                config=config
                            )
                            if response and response.text:
                                text_chunk = response.text.strip()
                                transcripts.append(text_chunk)
                                words = len(text_chunk.split())
                                log(f"✅ Chunk {chunk_num} complete! Transcribed {words} words.")
                            else:
                                log(f"⚠️ Chunk {chunk_num} returned empty transcript.")
                        finally:
                            log(f"🧹 Deleting chunk file from Gemini cloud storage...")
                            try:
                                client.files.delete(name=audio_file.name)
                            except Exception:
                                pass
                    else:
                        audio_file = legacy_genai.upload_file(path=chunk_path)
                        log(f"Uploaded to Gemini. Storage ID: {audio_file.name}")

                        wait_count = 0
                        while audio_file.state.name == "PROCESSING":
                            log(f"Gemini processing chunk audio... (waiting {wait_count*2}s)")
                            time.sleep(2)
                            audio_file = legacy_genai.get_file(audio_file.name)
                            wait_count += 1
                            if wait_count > 30:
                                log(f"❌ Error: Chunk {chunk_num} processing timed out.")
                                st.error(f"Chunk {chunk_num} processing timed out.")
                                st.stop()

                        log(f"🧠 Transcribing speech with gemini-3.5-flash-lite (temp=0.0)...")
                        model = legacy_genai.GenerativeModel(
                            model_name=model_choice,
                            generation_config={"temperature": 0.0}
                        )
                        try:
                            response = model.generate_content(
                                [prompt, audio_file],
                                request_options={"timeout": 600}
                            )
                            if response and response.text:
                                text_chunk = response.text.strip()
                                transcripts.append(text_chunk)
                                words = len(text_chunk.split())
                                log(f"✅ Chunk {chunk_num} complete! Transcribed {words} words.")
                            else:
                                log(f"⚠️ Chunk {chunk_num} returned empty transcript.")
                        finally:
                            log(f"🧹 Deleting chunk file from Gemini cloud storage...")
                            try:
                                legacy_genai.delete_file(audio_file.name)
                            except Exception:
                                pass

                    # Update live transcript preview after each chunk!
                    current_combined = "\n\n".join(transcripts)
                    transcript_preview.text_area("Live Output", current_combined, height=250, key=f"preview_{idx}")

                    # Update progress bar
                    progress_bar.progress((idx + 1) / total_chunks)

                    # Respect RPM rate limit (6s pause between chunks = max 10 RPM)
                    if idx < total_chunks - 1:
                        log(f"⏳ Pausing 6 seconds to strictly respect 15 RPM rate limit...")
                        time.sleep(6)

                total_time = int(time.time() - start_time)
                log(f"🎉 All {total_chunks} chunk(s) finished in {total_time}s!")
                status_box.empty()
                progress_bar.empty()

                # ── Step 4: Show final output ──────────────────────────────────
                final_text = "\n\n".join(transcripts).strip()

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


