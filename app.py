import os
import gc
import math
import glob
import tempfile
import subprocess
import traceback
import streamlit as st

st.set_page_config(
    page_title="Audio & Video Transcription App",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Audio & Video Transcription App")
st.write("Upload an audio or video file to generate a clean text transcript using Faster-Whisper.")

CHUNK_MINUTES = 5  # Transcribe in 5-minute chunks to stay well under 512 MB RAM

try:
    from faster_whisper import WhisperModel

    uploaded_file = st.file_uploader(
        "Choose an audio or video file",
        type=["mp3", "mp4", "wav", "m4a", "aac", "flac", "ogg", "mov", "mkv"]
    )

    model_size = st.selectbox("Select Model Size", ["tiny", "base"], index=0)
    st.caption(
        f"💡 File will be split into **{CHUNK_MINUTES}-minute chunks** and transcribed one at a time to stay within "
        f"Render's 512 MB free RAM limit. Use **'tiny'** for fastest results."
    )

    if uploaded_file is not None:
        if uploaded_file.type.startswith("audio"):
            st.audio(uploaded_file)
        elif uploaded_file.type.startswith("video"):
            st.video(uploaded_file)

        if st.button("Start Transcription", type="primary"):
            tmp_dir = tempfile.mkdtemp()
            original_path = os.path.join(tmp_dir, "original" + os.path.splitext(uploaded_file.name)[1])
            wav_path = os.path.join(tmp_dir, "audio_16k.wav")
            chunks_dir = os.path.join(tmp_dir, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)

            try:
                status = st.empty()

                # ── Step 1: Save uploaded file ──────────────────────────────────
                status.info("💾 Saving uploaded file...")
                with open(original_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # ── Step 2: Convert to 16 kHz mono WAV ──────────────────────────
                status.info("⚡ Converting to 16 kHz mono WAV with FFmpeg...")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", original_path,
                     "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                )

                # Remove original to free disk space immediately
                os.remove(original_path)
                gc.collect()

                # ── Step 3: Get audio duration ───────────────────────────────────
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
                    capture_output=True, text=True
                )
                total_duration = float(probe.stdout.strip())
                total_chunks = math.ceil(total_duration / (CHUNK_MINUTES * 60))

                status.info(
                    f"🔪 Splitting {total_duration/60:.1f}-minute audio into "
                    f"{total_chunks} chunks of {CHUNK_MINUTES} minutes each..."
                )

                # ── Step 4: Split into chunks ────────────────────────────────────
                chunk_pattern = os.path.join(chunks_dir, "chunk_%03d.wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", wav_path,
                     "-f", "segment",
                     "-segment_time", str(CHUNK_MINUTES * 60),
                     "-c", "copy",
                     chunk_pattern],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                )

                # Remove full wav to free disk/memory
                os.remove(wav_path)
                gc.collect()

                chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.wav")))
                total_chunks = len(chunk_files)

                # ── Step 5: Load Whisper model ONCE ─────────────────────────────
                status.info(f"🧠 Loading Whisper '{model_size}' model...")
                model = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=2,
                    num_workers=1
                )

                # ── Step 6: Transcribe each chunk ────────────────────────────────
                overall_bar = st.progress(0.0)
                chunk_status = st.empty()
                live_box = st.empty()

                all_text_parts = []

                for i, chunk_path in enumerate(chunk_files):
                    chunk_num = i + 1
                    chunk_status.markdown(
                        f"⏳ Transcribing chunk **{chunk_num}/{total_chunks}** "
                        f"({chunk_num * CHUNK_MINUTES - CHUNK_MINUTES}–{chunk_num * CHUNK_MINUTES} min)..."
                    )

                    segments, info = model.transcribe(chunk_path, beam_size=1, vad_filter=True)

                    chunk_texts = []
                    for segment in segments:
                        t = segment.text.strip()
                        if t:
                            chunk_texts.append(t)

                    # Remove chunk file immediately after transcription to free disk
                    os.remove(chunk_path)
                    gc.collect()

                    if chunk_texts:
                        all_text_parts.extend(chunk_texts)

                    # Update progress and live preview
                    overall_bar.progress(chunk_num / total_chunks)
                    live_box.text_area(
                        "Live Transcript (building...)",
                        "\n\n".join(all_text_parts),
                        height=350
                    )

                # ── Step 7: Done ─────────────────────────────────────────────────
                overall_bar.progress(1.0)
                chunk_status.success("🎉 Transcription complete!")

                final_text = "\n\n".join(all_text_parts)

                if final_text:
                    st.download_button(
                        label="📥 Download Full Transcript (.txt)",
                        data=final_text,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No speech detected in the audio file.")

            except subprocess.CalledProcessError as e:
                st.error(f"FFmpeg error: {e}")
                st.code(traceback.format_exc())
            except Exception as e:
                st.error(f"Transcription error: {e}")
                st.code(traceback.format_exc())
            finally:
                # Clean up all temp files
                import shutil
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                gc.collect()

except Exception as global_err:
    st.error("Failed to initialize application.")
    st.exception(global_err)
    st.code(traceback.format_exc())
