import os
import gc
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

# Error boundary container
try:
    from faster_whisper import WhisperModel
    
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass

    compute_type = "float16" if device == "cuda" else "int8"

    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose an audio or video file", 
        type=["mp3", "mp4", "wav", "m4a", "aac", "flac", "ogg", "mov", "mkv"]
    )

    model_size = st.selectbox("Select Model Size", ["tiny", "base", "small", "medium"], index=0)
    st.caption("💡 *Default is set to 'tiny' for maximum speed & memory optimization on free cloud servers (Render free tier).*")

    if uploaded_file is not None:
        if uploaded_file.type.startswith("audio"):
            st.audio(uploaded_file)
        elif uploaded_file.type.startswith("video"):
            st.video(uploaded_file)

        if st.button("Start Transcription", type="primary"):
            with st.spinner("Preparing audio stream... Please wait."):
                suffix = os.path.splitext(uploaded_file.name)[1]
                
                # Create temp file for uploaded upload
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name

                wav_path = temp_path + "_16k.wav"

                try:
                    status_info = st.empty()
                    
                    # 1. Convert to lightweight 16kHz mono WAV using ffmpeg CLI to save memory
                    status_info.info("⚡ Pre-processing audio with FFmpeg (converting to 16kHz mono to save RAM)...")
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-i", temp_path,
                        "-ar", "16000",
                        "-ac", "1",
                        "-c:a", "pcm_s16le",
                        wav_path
                    ]
                    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    
                    # Target the converted lightweight file
                    audio_target = wav_path if os.path.exists(wav_path) else temp_path

                    # 2. Force Garbage Collection to free RAM before loading Whisper model
                    gc.collect()

                    # 3. Load Whisper Model
                    status_info.info(f"🧠 Loading Whisper model ('{model_size}') into memory...")
                    model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=2)
                    
                    # 4. Transcribe with VAD filtering enabled
                    status_info.info("🔄 Detecting language & setting up VAD stream...")
                    segments, info = model.transcribe(audio_target, beam_size=1, vad_filter=True)
                    status_info.empty()
                    
                    st.success(f"Audio loaded ({info.duration:.1f}s / {info.duration/60:.1f} mins). Language: **'{info.language}'** (probability {info.language_probability:.2f})")
                    
                    # UI containers for live streaming transcript
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    live_box = st.empty()
                    
                    full_transcript = []
                    
                    for segment in segments:
                        text = segment.text.strip()
                        if text:
                            full_transcript.append(text)
                        
                        # Calculate progress percentage
                        if info.duration > 0:
                            progress = min(segment.end / info.duration, 1.0)
                            progress_bar.progress(progress)
                            status_text.markdown(f"⏳ **Transcribing:** `{int(progress * 100)}%` completed ({int(segment.end)}s / {int(info.duration)}s)")
                        
                        # Update live transcript text block
                        current_text = "\n\n".join(full_transcript)
                        live_box.text_area("Live Transcript Progress", current_text, height=300)
                    
                    final_text = "\n\n".join(full_transcript)
                    progress_bar.progress(1.0)
                    status_text.success("🎉 Transcription complete!")
                    
                    if final_text:
                        # Download button
                        st.download_button(
                            label="📥 Download Full Transcript (.txt)",
                            data=final_text,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                            mime="text/plain"
                        )
                    else:
                        st.warning("No speech detected in the audio file.")
                        
                except Exception as e:
                    st.error(f"An error occurred during transcription: {e}")
                    st.code(traceback.format_exc())
                finally:
                    # Clean up temporary files
                    for p in [temp_path, wav_path]:
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                    gc.collect()

except Exception as global_err:
    st.error("Failed to initialize application dependencies.")
    st.exception(global_err)
    st.code(traceback.format_exc())
