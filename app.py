import os
import streamlit as st
import torch
from faster_whisper import WhisperModel

st.set_page_config(
    page_title="Audio & Video Transcription App",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Audio & Video Transcription App")
st.write("Upload an audio or video file to generate a clean text transcript using Faster-Whisper.")

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose an audio or video file", 
    type=["mp3", "mp4", "wav", "m4a", "aac", "flac", "ogg", "mov", "mkv"]
)

model_size = st.selectbox("Select Model Size", ["tiny", "base", "small", "medium", "large-v3"], index=2)

if uploaded_file is not None:
    temp_filename = "temp_uploaded_file" + os.path.splitext(uploaded_file.name)[1]
    
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    if st.button("Start Transcription", type="primary"):
        with st.spinner("Transcribing... Please wait (this may take a minute depending on file size and model choice)."):
            try:
                # Setup device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                
                # Load model
                model = WhisperModel(model_size, device=device, compute_type=compute_type)
                segments, info = model.transcribe(temp_filename, beam_size=1)
                
                # Collect text
                full_transcript = [segment.text.strip() for segment in segments if segment.text.strip()]
                final_text = "\n\n".join(full_transcript)
                
                st.success(f"Transcription complete! (Detected language: {info.language} with probability {info.language_probability:.2f})")
                
                # Display text in a clean box
                st.text_area("Transcript Result", final_text, height=300)
                
                # Download button
                st.download_button(
                    label="Download Transcript as Text File",
                    data=final_text,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"An error occurred during transcription: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
