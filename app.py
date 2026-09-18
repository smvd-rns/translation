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
                    model = WhisperModel(model_size, device=device, compute_type=compute_type)
                    segments, info = model.transcribe(temp_path, beam_size=1)
                    
                    # Collect text
                    full_transcript = [segment.text.strip() for segment in segments if segment.text.strip()]
                    final_text = "\n\n".join(full_transcript)
                    
                    if final_text:
                        st.success(f"Transcription complete! (Detected language: '{info.language}' with probability {info.language_probability:.2f})")
                        
                        # Display text in a clean box
                        st.text_area("Transcript Result", final_text, height=300)
                        
                        # Download button
                        st.download_button(
                            label="Download Transcript as Text File",
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
