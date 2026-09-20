# Audio & Video Transcription App 🎙️

A Streamlit application using `faster-whisper` for fast audio/video transcription.

## Deploying on Render Free Tier

### YES! This app works on Render Free Tier.

To ensure it runs without any memory issues on Render's 512 MB Free Tier limit:

1. **Select Model Size:** Use **`tiny`** or **`base`** in the app's dropdown.
   - `tiny`: Uses ~150 MB RAM (Very fast & reliable on free server)
   - `base`: Uses ~250 MB RAM (Higher accuracy, fits in free memory)
   - *Avoid `small` or `medium` on the free tier as they require >1 GB RAM.*

2. **Deployment Method:** Docker deployment (Render will auto-detect the `Dockerfile` in this repo).

### Steps to Deploy:
1. Push this repository to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** -> **Web Service**.
4. Connect your GitHub repository.
5. Environment will be auto-detected as **Docker**.
6. Select **Free Tier** and click **Create Web Service**.