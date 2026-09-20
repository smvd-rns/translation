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
    """Slices input media into MP3 chunks using ffmpeg."""
    chunk_pattern = os.path.join(tmp_dir, "chunk_%03d.mp3")
    segment_seconds = str(chunk_minutes * 60)
    
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-f", "segment",
        "-segment_time", segment_seconds,
        "-c:a", "libmp3lame", "-b:a", "64k",
        chunk_pattern
    ]
    
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        chunks = sorted(glob.glob(os.path.join(tmp_dir, "chunk_*.mp3")))
        if chunks:
            return chunks
    except Exception as ex:
        st.warning(f"ffmpeg chunking unavailable ({ex}). Processing as single file.")
    
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
                status = st.empty()
                progress_bar = st.progress(0)

                # ── Step 1: Save uploaded file ────────────────────────────────
                status.info("💾 Saving uploaded file...")
                with open(original_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # ── Step 2: Chunk media file into 10-minute segments ──────────
                status.info(f"⚡ Slicing audio into {chunk_duration}-minute chunks with ffmpeg...")
                chunk_files = chunk_media_file(original_path, tmp_dir, chunk_minutes=chunk_duration)
                total_chunks = len(chunk_files)
                status.info(f"✅ Prepared **{total_chunks} chunk(s)** for processing.")

                transcripts = []

                prompt = (
                    "Please transcribe all the speech in this audio file. "
                    "Output only the spoken text, preserving natural paragraph breaks. "
                    "Do not add commentary, timestamps, or extra formatting."
                )

                # ── Step 3: Process chunks sequentially ───────────────────────
                for idx, chunk_path in enumerate(chunk_files):
                    chunk_num = idx + 1
                    status.info(f"📤 [Chunk {chunk_num}/{total_chunks}] Uploading to Gemini API...")
                    
                    if USE_NEW_SDK:
                        audio_file = client.files.upload(file=chunk_path)
                        
                        # Wait for processing if needed
                        wait_count = 0
                        while hasattr(audio_file, 'state') and str(getattr(audio_file.state, 'name', audio_file.state)) == "PROCESSING":
                            time.sleep(2)
                            audio_file = client.files.get(name=audio_file.name)
                            wait_count += 1
                            if wait_count > 30:
                                st.error(f"Chunk {chunk_num} processing timed out.")
                                st.stop()

                        status.info(f"🧠 [Chunk {chunk_num}/{total_chunks}] Transcribing speech...")
                        
                        try:
                            response = client.models.generate_content(
                                model=model_choice,
                                contents=[audio_file, prompt]
                            )
                            if response and response.text:
                                transcripts.append(response.text.strip())
                        finally:
                            try:
                                client.files.delete(name=audio_file.name)
                            except Exception:
                                pass
                    else:
                        audio_file = legacy_genai.upload_file(path=chunk_path)

                        wait_count = 0
                        while audio_file.state.name == "PROCESSING":
                            time.sleep(2)
                            audio_file = legacy_genai.get_file(audio_file.name)
                            wait_count += 1
                            if wait_count > 30:
                                st.error(f"Chunk {chunk_num} processing timed out.")
                                st.stop()

                        status.info(f"🧠 [Chunk {chunk_num}/{total_chunks}] Transcribing speech...")
                        model = legacy_genai.GenerativeModel(model_name=model_choice)
                        try:
                            response = model.generate_content(
                                [prompt, audio_file],
                                request_options={"timeout": 600}
                            )
                            if response and response.text:
                                transcripts.append(response.text.strip())
                        finally:
                            try:
                                legacy_genai.delete_file(audio_file.name)
                            except Exception:
                                pass

                    # Update progress
                    progress_bar.progress((idx + 1) / total_chunks)

                    # Respect RPM rate limit (6s pause between chunks = max 10 RPM)
                    if idx < total_chunks - 1:
                        status.info(f"⏳ Waiting 6s to stay safely within API rate limits...")
                        time.sleep(6)

                status.empty()
                progress_bar.empty()

                # ── Step 4: Show final output ──────────────────────────────────
                final_text = "\n\n".join(transcripts).strip()

                if final_text:
                    st.success(f"🎉 Transcription complete! Processed {total_chunks} chunk(s).")
                    st.text_area("Transcript", final_text, height=400)
                    st.download_button(
                        label="📥 Download Transcript (.txt)",
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


