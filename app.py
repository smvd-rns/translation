import os
import gc
import math
import tempfile
import subprocess
import traceback
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="Audio & Video Transcription App",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Audio & Video Transcription App")
st.write("Upload an audio or video file to generate a clean text transcript — powered by Groq Whisper AI.")

# ── API Key input ────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown(
    "Get your **free** Groq API key at [console.groq.com](https://console.groq.com) → API Keys"
)

# Try environment variable first, then sidebar input
api_key = os.environ.get("GROQ_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_..."
    )

if not api_key:
    st.warning(
        "⚠️ Please enter your **Groq API Key** in the sidebar to use this app.\n\n"
        "👉 Get a free key at [console.groq.com](https://console.groq.com) — takes 1 minute."
    )
    st.stop()

# ── File uploader ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose an audio or video file",
    type=["mp3", "mp4", "wav", "m4a", "aac", "flac", "ogg", "mov", "mkv"]
)

CHUNK_MB = 20  # Groq allows max 25MB per request; use 20MB chunks with safety margin
GROQ_MODEL = "whisper-large-v3-turbo"

if uploaded_file is not None:
    if uploaded_file.type.startswith("audio"):
        st.audio(uploaded_file)
    elif uploaded_file.type.startswith("video"):
        st.video(uploaded_file)

    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.caption(f"📁 File size: **{file_size_mb:.1f} MB**")

    if st.button("Start Transcription", type="primary"):

        with tempfile.TemporaryDirectory() as tmp_dir:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"
            original_path = os.path.join(tmp_dir, "input" + suffix)
            chunks_dir = os.path.join(tmp_dir, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)

            try:
                status = st.empty()

                # ── Step 1: Save file ────────────────────────────────────────
                status.info("💾 Saving uploaded file...")
                with open(original_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # ── Step 2: Get duration via ffprobe ─────────────────────────
                probe = subprocess.run(
                    ["ffprobe", "-v", "error",
                     "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1",
                     original_path],
                    capture_output=True, text=True
                )
                total_duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0

                # ── Step 3: Calculate chunk duration based on file size ───────
                # Each chunk should be ~CHUNK_MB MB
                if file_size_mb <= CHUNK_MB:
                    # Small file — send directly, no splitting needed
                    chunk_files = [original_path]
                    status.info(f"📤 File is {file_size_mb:.1f} MB — sending directly to Groq...")
                else:
                    # Split by duration proportional to size
                    chunk_duration = int((CHUNK_MB / file_size_mb) * total_duration)
                    chunk_duration = max(60, min(chunk_duration, 600))  # between 1 and 10 min
                    num_chunks = math.ceil(total_duration / chunk_duration)
                    status.info(
                        f"🔪 Splitting {total_duration/60:.1f}-min audio into ~{num_chunks} "
                        f"chunks of ~{chunk_duration//60} min each..."
                    )

                    chunk_pattern = os.path.join(chunks_dir, "chunk_%03d.mp3")
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", original_path,
                         "-f", "segment",
                         "-segment_time", str(chunk_duration),
                         "-c:a", "libmp3lame", "-q:a", "5",
                         chunk_pattern],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                    )

                    import glob
                    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.mp3")))
                    status.info(f"✅ Split into {len(chunk_files)} chunks. Starting transcription...")

                # ── Step 4: Transcribe each chunk via Groq API ───────────────
                client = Groq(api_key=api_key)
                total_chunks = len(chunk_files)

                progress_bar = st.progress(0.0)
                chunk_label = st.empty()
                live_box = st.empty()

                all_parts = []

                for i, chunk_path in enumerate(chunk_files):
                    chunk_num = i + 1
                    chunk_label.markdown(
                        f"⚡ **Groq Transcribing** chunk {chunk_num} of {total_chunks} "
                        f"on remote GPU..."
                    )

                    with open(chunk_path, "rb") as audio_file:
                        response = client.audio.transcriptions.create(
                            model=GROQ_MODEL,
                            file=audio_file,
                            response_format="text"
                        )

                    chunk_text = response.strip() if isinstance(response, str) else response.text.strip()
                    if chunk_text:
                        all_parts.append(chunk_text)

                    progress_bar.progress(chunk_num / total_chunks)
                    live_box.text_area(
                        "Live Transcript (building...)",
                        "\n\n".join(all_parts),
                        height=350
                    )

                # ── Step 5: Final result ─────────────────────────────────────
                progress_bar.progress(1.0)
                chunk_label.success("🎉 Transcription complete!")
                status.empty()

                final_text = "\n\n".join(all_parts)
                if final_text:
                    st.text_area("Full Transcript", final_text, height=400)
                    st.download_button(
                        label="📥 Download Transcript (.txt)",
                        data=final_text,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No speech detected in the audio file.")

            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
            finally:
                gc.collect()
