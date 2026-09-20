import os
import tempfile
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

    model_size = st.selectbox("Select Model Size", ["tiny", "base", "small", "medium"], index=1)
    st.caption("💡 *Note for cloud deployment (e.g. Render free tier): Use 'tiny' or 'base' models to avoid running out of memory (RAM).*")

    if uploaded_file is not None:
        if uploaded_file.type.startswith("audio"):
            st.audio(uploaded_file)
        elif uploaded_file.type.startswith("video"):
            st.video(uploaded_file)

        if st.button("Start Transcription", type="primary"):
            with st.spinner("Transcribing... Please wait (this may take a minute depending on file size and model choice)."):
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name

                try:
                    # Load model
                    status_info = st.empty()
                    status_info.info("Loading Whisper model into memory...")
                    model = WhisperModel(model_size, device=device, compute_type=compute_type)
                    
                    status_info.info("Analyzing audio stream...")
                    segments, info = model.transcribe(temp_path, beam_size=1)
                    
                    st.success(f"Audio loaded ({info.duration:.1f} seconds). Detected language: **'{info.language}'** (probability {info.language_probability:.2f})")
                    
                    # Create UI containers for live streaming
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
                        
                        # Live streaming text update
                        current_text = "\n\n".join(full_transcript)
                        live_box.text_area("Live Transcript Progress", current_text, height=300)
                    
                    final_text = "\n\n".join(full_transcript)
                    progress_bar.progress(1.0)
                    status_text.success("🎉 Transcription complete!")
                    
                    if final_text:
                        # Download button for completed transcript
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
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

except Exception as global_err:
    st.error("Failed to initialize application dependencies.")
    st.exception(global_err)
    st.code(traceback.format_exc())
