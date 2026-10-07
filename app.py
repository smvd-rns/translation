import streamlit as st
import re
import os
import streamlit.components.v1 as components

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False

st.set_page_config(
    page_title="YouTube Caption Fetcher",
    page_icon="🎬",
    layout="centered"
)

def extract_youtube_video_id(url):
    """Extracts 11-character YouTube video ID from various link formats."""
    if not url:
        return None
    pattern = r'(?:v=|\/([0-9A-Za-z_-]{11}).*|youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1) or match.group(2)
    return None

def fetch_youtube_captions(video_id, target_lang=None):
    """Fetches existing captions/subtitles directly from YouTube if available."""
    if not YOUTUBE_TRANSCRIPT_AVAILABLE or not video_id:
        return None

    def _items_to_text(data):
        lines = []
        for item in data:
            if isinstance(item, dict):
                text = item.get('text', '')
            else:
                text = getattr(item, 'text', str(item))
            if text and text.strip():
                lines.append(text.strip())
        return "\n\n".join(lines) if lines else None

    langs_to_try = []
    if target_lang:
        langs_to_try.append(target_lang)
    langs_to_try += ['en', 'hi', 'es', 'fr', 'de', 'ar', 'pt', 'ru', 'ja', 'ko', 'zh']

    if hasattr(YouTubeTranscriptApi, 'get_transcript'):
        try:
            data = YouTubeTranscriptApi.get_transcript(video_id, languages=langs_to_try)
            result = _items_to_text(data)
            if result: return result
        except Exception as e:
            pass
        try:
            data = YouTubeTranscriptApi.get_transcript(video_id)
            result = _items_to_text(data)
            if result: return result
        except Exception as e:
            pass

    if hasattr(YouTubeTranscriptApi, 'list_transcripts'):
        try:
            t_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for t in t_list:
                if not getattr(t, 'is_generated', True):
                    try:
                        data = t.fetch()
                        result = _items_to_text(data)
                        if result: return result
                    except Exception: pass
            for t in t_list:
                try:
                    data = t.fetch()
                    result = _items_to_text(data)
                    if result: return result
                except Exception: pass
        except Exception: pass

    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, 'fetch'):
            try:
                transcript = api.fetch(video_id, languages=langs_to_try)
                result = _items_to_text(transcript)
                if result: return result
            except Exception: pass
            try:
                transcript = api.fetch(video_id)
                result = _items_to_text(transcript)
                if result: return result
            except Exception: pass
        if hasattr(api, 'list'):
            try:
                t_list = api.list(video_id)
                for t in t_list:
                    try:
                        data = t.fetch()
                        result = _items_to_text(data)
                        if result: return result
                    except Exception: pass
            except Exception: pass
    except Exception: pass

    return None

st.title("🎬 YouTube Caption Fetcher")
st.write("Instantly fetch official or auto-generated captions from any public YouTube video.")

youtube_url = st.text_input(
    "🔗 Enter YouTube Video Link",
    placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/..."
).strip()

language_options = {
    "English 🇬🇧 (en)": "en",
    "Hindi 🇮🇳 (hi)": "hi",
    "Spanish 🇪🇸 (es)": "es",
    "French 🇫🇷 (fr)": "fr",
    "German 🇩🇪 (de)": "de",
    "Russian 🇷🇺 (ru)": "ru",
    "Japanese 🇯🇵 (ja)": "ja",
    "Chinese 🇨🇳 (zh)": "zh",
    "Auto-Detect (Any available)": None
}

selected_lang_label = st.selectbox(
    "🌐 Preferred Language (Falls back to English/Auto if unavailable)",
    options=list(language_options.keys()),
    index=0
)
selected_language_code = language_options[selected_lang_label]

if youtube_url:
    yt_video_id = extract_youtube_video_id(youtube_url)
    if yt_video_id:
        components.html(
            f'''<iframe width="100%" height="350" src="https://www.youtube-nocookie.com/embed/{yt_video_id}" 
            title="YouTube video player" frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
            referrerpolicy="strict-origin-when-cross-origin" allowfullscreen style="border-radius: 10px; width: 100%;"></iframe>''',
            height=360
        )
        
        if st.button("🚀 Fetch Captions", type="primary", use_container_width=True):
            with st.spinner("Fetching captions..."):
                captions = fetch_youtube_captions(yt_video_id, target_lang=selected_language_code)
                
                if captions:
                    st.success("🎉 Captions fetched successfully!")
                    st.text_area("Transcript", captions, height=400)
                    st.download_button(
                        label="⬇️ Download Transcript",
                        data=captions,
                        file_name=f"youtube_{yt_video_id}_transcript.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.error("❌ No captions found! This video might not have any captions available, or YouTube is blocking the request from this server IP.")
    else:
        st.warning("⚠️ Invalid YouTube URL format. Please enter a valid YouTube video link.")
