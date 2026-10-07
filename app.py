import streamlit as st

st.set_page_config(
    page_title="Translation - Audio & Video Transcription",
    page_icon="🎙️",
    layout="centered"
)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

st.markdown("""
<style>
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes pulse-glow {
    0% { box-shadow: 0 0 8px rgba(255, 75, 75, 0.3); border-color: rgba(255, 75, 75, 0.5); }
    50% { box-shadow: 0 0 20px rgba(255, 75, 75, 0.7); border-color: rgba(255, 75, 75, 0.9); }
    100% { box-shadow: 0 0 8px rgba(255, 75, 75, 0.3); border-color: rgba(255, 75, 75, 0.5); }
}
.transcribing-banner {
    background: linear-gradient(135deg, rgba(255, 75, 75, 0.12), rgba(120, 50, 255, 0.12));
    border: 1px dashed #ff4b4b;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 15px 0 25px 0;
    display: flex;
    align-items: center;
    gap: 16px;
    animation: pulse-glow 2.5s infinite ease-in-out;
}
.loader-spinner {
    width: 28px;
    height: 28px;
    border: 3px solid rgba(255, 75, 75, 0.25);
    border-top: 3px solid #ff4b4b;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

import os
import gc
import glob
import time
import base64
import tempfile
import subprocess
import traceback
import requests
import re
import streamlit.components.v1 as components

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

# Free caption providers have no app-level daily/monthly quotas.
# RapidAPI is an optional last-resort backup and inherits RapidAPI plan quotas.
PIPED_INSTANCE_FALLBACKS = [
    "https://pipedapi.kavin.rocks",
    "https://pipedapi-libre.kavin.rocks",
    "https://piped-api.privacy.com.de",
    "https://pipedapi.adminforge.de",
    "https://api.piped.yt",
    "https://pipedapi.drgns.space",
    "https://pipedapi.ducks.party",
    "https://piped-api.codespace.cz",
    "https://pipedapi.reallyaweso.me",
    "https://api.piped.private.coffee",
    "https://pipedapi.nosebs.ru",
    "https://pipedapi.leptons.xyz",
]
INVIDIOUS_INSTANCE_FALLBACKS = [
    "https://inv.nadeko.net",
    "https://invidious.f5.si",
    "https://invidious.private.coffee",
    "https://yt.artemislena.eu",
    "https://invidious.jing.rocks",
    "https://inv.tux.pizza",
]
PIPED_INSTANCES_DOC_URL = (
    "https://raw.githubusercontent.com/TeamPiped/documentation/"
    "main/content/docs/public-instances/index.md"
)
INVIDIOUS_INSTANCES_URL = "https://api.invidious.io/instances.json"


def extract_youtube_video_id(url):
    """Extracts 11-character YouTube video ID from various link formats."""
    if not url:
        return None
    pattern = r'(?:v=|\/([0-9A-Za-z_-]{11}).*|youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1) or match.group(2)
    return None


def _dedupe_caption_lines(lines):
    clean_lines = []
    last_line = ""
    for line in lines:
        if line and line != last_line:
            clean_lines.append(line)
            last_line = line
    return clean_lines


def _parse_vtt_to_text(vtt_text):
    """Convert WEBVTT / SRT-like caption text into plain paragraphs."""
    lines = []
    for line in (vtt_text or "").splitlines():
        line = line.strip()
        if (
            not line
            or line.startswith("WEBVTT")
            or line.startswith("NOTE")
            or "-->" in line
            or line.startswith("Kind:")
            or line.startswith("Language:")
            or re.match(r"^\d+$", line)
        ):
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line and not re.match(r"^[\d:.,]+$", line):
            lines.append(line)
    clean = _dedupe_caption_lines(lines)
    return "\n\n".join(clean) if clean else None


def _transcript_items_to_text(data):
    lines = []
    # New API: FetchedTranscript is iterable of snippets with .text
    iterable = getattr(data, "snippets", None) or data
    try:
        iterable = list(iterable)
    except TypeError:
        return None
    for item in iterable:
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = getattr(item, "text", str(item))
        if text and str(text).strip():
            lines.append(str(text).strip())
    return "\n\n".join(lines) if lines else None


def _load_youtube_cookie_session():
    """Build a requests Session with Netscape cookies.txt when present."""
    cookie_file = "cookies.txt"
    if not os.path.exists(cookie_file):
        return None, False
    try:
        from http.cookiejar import MozillaCookieJar
        from requests import Session
        from requests.cookies import RequestsCookieJar

        jar = MozillaCookieJar(cookie_file)
        jar.load(ignore_discard=True, ignore_expires=True)
        session = Session()
        req_jar = RequestsCookieJar()
        for cookie in jar:
            req_jar.set_cookie(cookie)
        session.cookies = req_jar
        session.headers.update({
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })
        return session, True
    except Exception:
        return None, True


def _preferred_langs(target_lang=None):
    langs = []
    if target_lang:
        langs.append(target_lang)
    langs += ["en", "hi", "es", "fr", "de", "ar", "pt", "ru", "ja", "ko", "zh"]
    # Preserve order, drop duplicates
    seen = set()
    ordered = []
    for lang in langs:
        if lang and lang not in seen:
            seen.add(lang)
            ordered.append(lang)
    return ordered


def _is_piped_api_root(url):
    """True for Piped API base URLs; false for badge/asset paths scraped from docs."""
    if not url or not url.startswith("https://"):
        return False
    lower = url.lower().rstrip("/")
    if any(bad in lower for bad in ("/badge", "/registered", "/assets", ".png", ".svg")):
        return False
    host_path = lower.split("://", 1)[-1]
    # API roots are host-only (optional trailing slash already stripped)
    if "/" in host_path:
        return False
    return (
        ("pipedapi" in host_path)
        or ("piped-api" in host_path)
        or host_path.startswith("api.piped.")
    )


def _rewrite_timedtext_url(base_url):
    """
    Rewrite YouTube timedtext URLs to video.google.com/timedtext.
    Useful when www.youtube.com TLS is blocked but Google endpoints still work.
    """
    from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

    if not base_url:
        return None
    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query)
    # Force VTT for consistent parsing
    qs["fmt"] = ["vtt"]
    flat = {k: v[0] for k, v in qs.items() if v}
    return urlunparse(("https", "video.google.com", "/timedtext", "", urlencode(flat), ""))


def _get_dynamic_piped_instances():
    """Refresh Piped API URLs from the official public-instances doc when possible."""
    instances = list(PIPED_INSTANCE_FALLBACKS)
    try:
        resp = requests.get(PIPED_INSTANCES_DOC_URL, timeout=8)
        if resp.status_code == 200:
            found = re.findall(r"https://[a-zA-Z0-9._/-]*piped[a-zA-Z0-9._/-]*", resp.text)
            api_like = [u.rstrip("/") for u in found if _is_piped_api_root(u)]
            if api_like:
                # Prefer curated fallbacks first (more reliable than scraped order)
                instances = list(dict.fromkeys(instances + api_like))
    except Exception:
        pass
    return instances


def _get_dynamic_invidious_instances():
    """Refresh Invidious instances from api.invidious.io when possible."""
    # Prefer curated fallbacks first — public directory is often sparse/flaky
    instances = list(INVIDIOUS_INSTANCE_FALLBACKS)
    try:
        resp = requests.get(INVIDIOUS_INSTANCES_URL, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            live = []
            for entry in data:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                meta = entry[1] if isinstance(entry[1], dict) else {}
                uri = meta.get("uri") or entry[0]
                if (
                    isinstance(uri, str)
                    and uri.startswith("https://")
                    and meta.get("type") == "https"
                    and meta.get("api", True)
                ):
                    live.append(uri.rstrip("/"))
            if live:
                instances = list(dict.fromkeys(instances + live))
    except Exception:
        pass
    return instances


def fetch_rapidapi_transcript(video_id, keys, target_lang=None, log_func=None):
    """
    Fetches YouTube transcript via RapidAPI using multi-key rotation / fallback.
    Subject to RapidAPI plan quotas (not app-enforced).
    """
    if not keys:
        return None

    for idx, key in enumerate(keys):
        key = key.strip()
        if not key:
            continue
        try:
            if log_func: log_func(f"Trying RapidAPI Key #{idx+1}...")
            # Endpoint 1: youtube-transcriptor
            url = f"https://youtube-transcriptor.p.rapidapi.com/transcript?video_id={video_id}"
            headers = {
                "x-rapidapi-key": key,
                "x-rapidapi-host": "youtube-transcriptor.p.rapidapi.com"
            }
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                lines = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "text" in item:
                            lines.append(str(item["text"]).strip())
                        elif isinstance(item, dict) and "transcription" in item:
                            lines.extend([str(t.get("text", "")).strip() for t in item["transcription"] if isinstance(t, dict) and t.get("text") is not None])
                        elif isinstance(item, list):
                            lines.extend([str(t.get("text", "")).strip() for t in item if isinstance(t, dict) and t.get("text") is not None])
                elif isinstance(data, dict):
                    if "transcript" in data and isinstance(data["transcript"], list):
                        lines = [str(t.get("text", "")).strip() for t in data["transcript"] if isinstance(t, dict) and t.get("text") is not None]
                    elif "text" in data:
                        lines = [str(data["text"]).strip()]
                if lines:
                    if log_func: log_func(f"✅ Success with Key #{idx+1} (youtube-transcriptor)")
                    return "\n\n".join([l for l in lines if l])
            else:
                if log_func: log_func(f"⚠️ Key #{idx+1} (youtube-transcriptor) returned {resp.status_code}: {resp.text[:100]}")

            # Endpoint 2: youtube-transcript3
            url2 = f"https://youtube-transcript3.p.rapidapi.com/api/transcript?videoId={video_id}"
            headers2 = {
                "x-rapidapi-key": key,
                "x-rapidapi-host": "youtube-transcript3.p.rapidapi.com"
            }
            resp2 = requests.get(url2, headers=headers2, timeout=10)
            if resp2.status_code == 200:
                data2 = resp2.json()
                if isinstance(data2, dict) and "transcript" in data2:
                    lines = [str(t.get("text", "")).strip() for t in data2["transcript"] if isinstance(t, dict) and t.get("text") is not None]
                    if lines:
                        if log_func: log_func(f"✅ Success with Key #{idx+1} (youtube-transcript3)")
                        return "\n\n".join(lines)
            else:
                if log_func: log_func(f"⚠️ Key #{idx+1} (youtube-transcript3) returned {resp2.status_code}: {resp2.text[:100]}")
        except Exception as e:
            if log_func: log_func(f"⚠️ Key #{idx+1} error: {e}")
            continue
    return None


def fetch_piped_transcript(video_id, target_lang=None, log_func=None):
    """Fetches transcripts from public Piped API instances (VTT parsing). Free / no app quotas."""
    instances = _get_dynamic_piped_instances()
    # Cap attempts so a long dead list does not stall the UI
    instances = instances[:12]

    for idx, instance in enumerate(instances):
        try:
            if log_func: log_func(f"🌐 Trying Piped API Instance #{idx+1} ({instance})...")
            url = f"{instance.rstrip('/')}/streams/{video_id}"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                subtitles = data.get("subtitles", []) or []

                if not subtitles:
                    if log_func: log_func(f"⚠️ Instance #{idx+1} returned no subtitles.")
                    continue

                selected_sub = None
                for sub in subtitles:
                    code = (sub.get("code") or "").lower()
                    if target_lang and code.startswith(target_lang.lower()) and not sub.get("autoGenerated"):
                        selected_sub = sub
                        break
                    if not target_lang and code.startswith("en") and not sub.get("autoGenerated"):
                        selected_sub = sub
                        break

                if not selected_sub:
                    for sub in subtitles:
                        code = (sub.get("code") or "").lower()
                        if target_lang and code.startswith(target_lang.lower()):
                            selected_sub = sub
                            break
                        if not target_lang and code.startswith("en"):
                            selected_sub = sub
                            break

                if not selected_sub:
                    selected_sub = subtitles[0]

                sub_url = selected_sub.get("url")
                if sub_url:
                    sub_resp = requests.get(sub_url, timeout=8)
                    if sub_resp.status_code == 200:
                        text = _parse_vtt_to_text(sub_resp.text)
                        if text:
                            if log_func: log_func(f"✅ Success via Piped API ({selected_sub.get('name')})!")
                            return text
            else:
                if log_func: log_func(f"⚠️ Instance #{idx+1} failed with status {resp.status_code}")
        except Exception as e:
            if log_func: log_func(f"⚠️ Instance #{idx+1} error: {e}")
            continue
    return None


def fetch_invidious_transcript(video_id, target_lang=None, log_func=None):
    """Fetches captions via public Invidious instances. Free / no app quotas."""
    instances = _get_dynamic_invidious_instances()[:10]
    preferred = _preferred_langs(target_lang)

    for idx, instance in enumerate(instances):
        base = instance.rstrip("/")
        try:
            if log_func: log_func(f"🆓 Trying Invidious Instance #{idx+1} ({base})...")
            list_resp = requests.get(f"{base}/api/v1/captions/{video_id}", timeout=8)
            if list_resp.status_code != 200:
                if log_func: log_func(f"⚠️ Invidious #{idx+1} list failed: {list_resp.status_code}")
                continue
            try:
                payload = list_resp.json()
            except Exception:
                if log_func: log_func(f"⚠️ Invidious #{idx+1} returned non-JSON.")
                continue

            captions = payload.get("captions") if isinstance(payload, dict) else None
            if not captions:
                if log_func: log_func(f"⚠️ Invidious #{idx+1} has no captions for this video.")
                continue

            selected = None
            for lang in preferred:
                for cap in captions:
                    code = (cap.get("languageCode") or "").lower()
                    if code == lang.lower() or code.startswith(lang.lower() + "-"):
                        selected = cap
                        break
                if selected:
                    break
            if not selected:
                selected = captions[0]

            lang_code = (selected.get("languageCode") or preferred[0] or "en").split("-")[0]
            # Try label URL from API, then lang= shortcut (some instances only fill one)
            candidate_urls = []
            cap_url = selected.get("url") or ""
            if cap_url.startswith("/"):
                cap_url = base + cap_url
            if cap_url:
                candidate_urls.append(cap_url)
            candidate_urls.append(f"{base}/api/v1/captions/{video_id}?lang={lang_code}")

            got_body = False
            for try_url in candidate_urls:
                cap_resp = requests.get(try_url, timeout=10)
                if cap_resp.status_code != 200:
                    continue
                if not (cap_resp.text or "").strip():
                    continue
                got_body = True
                text = _parse_vtt_to_text(cap_resp.text)
                if text:
                    label = selected.get("label") or selected.get("languageCode") or "captions"
                    if log_func: log_func(f"✅ Success via Invidious ({label})!")
                    return text

            if log_func:
                if got_body:
                    log_func(f"⚠️ Invidious #{idx+1} returned caption body but parse failed.")
                else:
                    log_func(f"⚠️ Invidious #{idx+1} caption body empty (instance likely can't reach YouTube).")
        except Exception as e:
            if log_func: log_func(f"⚠️ Invidious #{idx+1} error: {e}")
            continue
    return None


def fetch_innertube_transcript(video_id, target_lang=None, log_func=None):
    """
    Free caption fetch via YouTube InnerTube (youtubei.googleapis.com) +
    video.google.com/timedtext. Avoids www.youtube.com (often TLS-blocked).
    No app-level quotas.
    """
    preferred = _preferred_langs(target_lang)
    if log_func:
        log_func("🆓 Trying InnerTube + video.google.com timedtext (free, no quota)...")

    try:
        body = {
            "context": {
                "client": {
                    "clientName": "ANDROID",
                    "clientVersion": "20.10.38",
                }
            },
            "videoId": video_id,
        }
        resp = requests.post(
            "https://youtubei.googleapis.com/youtubei/v1/player?prettyPrint=false",
            json=body,
            timeout=15,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "com.google.android.youtube/20.10.38 (Linux; U; Android 14)",
            },
        )
        if resp.status_code != 200:
            if log_func:
                log_func(f"⚠️ InnerTube player failed: {resp.status_code}")
            return None

        data = resp.json()
        tracks = (
            ((data.get("captions") or {}).get("playerCaptionsTracklistRenderer") or {})
            .get("captionTracks")
        )
        if not tracks:
            if log_func:
                log_func("⚠️ InnerTube returned no caption tracks.")
            return None

        selected = None
        for lang in preferred:
            for track in tracks:
                code = (track.get("languageCode") or "").lower()
                if code == lang.lower() or code.startswith(lang.lower() + "-"):
                    selected = track
                    break
            if selected:
                break
        if not selected:
            selected = tracks[0]

        timedtext_url = _rewrite_timedtext_url(selected.get("baseUrl"))
        if not timedtext_url:
            if log_func:
                log_func("⚠️ InnerTube track missing baseUrl.")
            return None

        cap_resp = requests.get(timedtext_url, timeout=15)
        if cap_resp.status_code != 200 or not (cap_resp.text or "").strip():
            if log_func:
                log_func(f"⚠️ timedtext download failed: {cap_resp.status_code}")
            return None

        text = _parse_vtt_to_text(cap_resp.text)
        if text:
            name_obj = selected.get("name")
            if isinstance(name_obj, dict):
                label = name_obj.get("simpleText") or selected.get("languageCode") or "captions"
            else:
                label = name_obj or selected.get("languageCode") or "captions"
            if log_func:
                log_func(f"✅ Success via InnerTube timedtext ({label})!")
            return text

        if log_func:
            log_func("⚠️ InnerTube timedtext parse returned empty.")
    except Exception as e:
        if log_func:
            log_func(f"⚠️ InnerTube error: {e}")
    return None


def fetch_ytdlp_transcript(video_id, target_lang=None, log_func=None):
    """Fetch official/auto subtitles with yt-dlp (no audio download). Free / no app quotas."""
    if not YT_DLP_AVAILABLE:
        return None

    preferred = _preferred_langs(target_lang)
    outdir = tempfile.mkdtemp(prefix="yt_subs_")
    cookie_file = "cookies.txt" if os.path.exists("cookies.txt") else None
    try:
        if log_func: log_func("🆓 Trying yt-dlp subtitle extract (free, no quota)...")
        opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": preferred[:6],
            "subtitlesformat": "vtt",
            "outtmpl": os.path.join(outdir, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        if cookie_file:
            opts["cookiefile"] = cookie_file

        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        # Prefer requested language files, then any .vtt
        candidates = []
        for lang in preferred:
            candidates.extend(glob.glob(os.path.join(outdir, f"{video_id}.{lang}*.vtt")))
        candidates.extend(glob.glob(os.path.join(outdir, f"{video_id}*.vtt")))
        seen = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    text = _parse_vtt_to_text(fh.read())
                if text:
                    if log_func: log_func(f"✅ Success via yt-dlp ({os.path.basename(path)})!")
                    return text
            except Exception:
                continue
        if log_func: log_func("⚠️ yt-dlp found no usable subtitle files.")
    except Exception as e:
        if log_func: log_func(f"⚠️ yt-dlp error: {e}")
    finally:
        try:
            for path in glob.glob(os.path.join(outdir, "*")):
                try:
                    os.remove(path)
                except Exception:
                    pass
            os.rmdir(outdir)
        except Exception:
            pass
    return None


def fetch_direct_youtube_transcript(video_id, target_lang=None, log_func=None):
    """Uses youtube-transcript-api (v1.x instance API). Free / no app quotas."""
    if not YOUTUBE_TRANSCRIPT_AVAILABLE:
        return None

    langs_to_try = _preferred_langs(target_lang)
    session, cookies_present = _load_youtube_cookie_session()
    if cookies_present and session and log_func:
        log_func("🍪 'cookies.txt' found — attaching to youtube-transcript-api session...")
    elif cookies_present and log_func:
        log_func("🍪 'cookies.txt' found but could not be loaded; continuing without cookies...")
    elif log_func:
        log_func("🌐 Trying direct youtube-transcript-api without cookies...")

    try:
        api = YouTubeTranscriptApi(http_client=session) if session else YouTubeTranscriptApi()
    except TypeError:
        # Extremely old fallback — should not hit with requirements pin
        api = YouTubeTranscriptApi()

    # Modern API (1.x): instance .fetch / .list
    last_err = None
    if hasattr(api, "fetch"):
        try:
            data = api.fetch(video_id, languages=langs_to_try)
            result = _transcript_items_to_text(data)
            if result:
                if log_func: log_func("✅ Success via direct youtube-transcript-api!")
                return result
        except Exception as e:
            last_err = e
        try:
            # Any available transcript language
            if hasattr(api, "list"):
                t_list = api.list(video_id)
                for transcript in t_list:
                    try:
                        data = transcript.fetch()
                        result = _transcript_items_to_text(data)
                        if result:
                            if log_func: log_func("✅ Success via youtube-transcript-api list()!")
                            return result
                    except Exception as e:
                        last_err = e
                        continue
        except Exception as e:
            last_err = e

    # Legacy classmethods (pre-1.0) if somehow still present
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        try:
            data = YouTubeTranscriptApi.get_transcript(video_id, languages=langs_to_try)
            result = _transcript_items_to_text(data)
            if result:
                if log_func: log_func("✅ Success via legacy get_transcript!")
                return result
        except Exception as e:
            last_err = e

    if log_func:
        if last_err:
            log_func(f"❌ Direct youtube-transcript-api failed: {type(last_err).__name__}: {last_err}")
        else:
            log_func("❌ Direct youtube-transcript-api failed or blocked.")
    return None


def fetch_youtube_captions(video_id, target_lang=None, log_func=None):
    """
    Fetch existing captions/subtitles for a YouTube video.

    Priority (free / unlimited first, RapidAPI last):
      1. youtube-transcript-api (direct)
      2. InnerTube + video.google.com timedtext
      3. Invidious public instances
      4. Piped public instances
      5. yt-dlp subtitle extract
      6. RapidAPI (quota-limited backup)
      7. Optional Cloudflare Worker proxy
    """
    if not video_id:
        return None

    # ── Method 1: Direct YouTube (free, no app quotas) ───────────────────────
    res = fetch_direct_youtube_transcript(video_id, target_lang=target_lang, log_func=log_func)
    if res:
        return res

    # ── Method 2: InnerTube / video.google.com (bypasses youtube.com TLS) ────
    res = fetch_innertube_transcript(video_id, target_lang=target_lang, log_func=log_func)
    if res:
        return res

    # ── Method 3: Invidious (free public mirrors) ────────────────────────────
    if log_func: log_func("🆓 Trying Free Invidious Instances...")
    res = fetch_invidious_transcript(video_id, target_lang=target_lang, log_func=log_func)
    if res:
        return res
    elif log_func:
        log_func("❌ Invidious instances failed or returned no captions.")

    # ── Method 4: Piped (free public mirrors, dynamically refreshed) ─────────
    if log_func: log_func("🌐 Trying Free Piped API Instances...")
    res = fetch_piped_transcript(video_id, target_lang=target_lang, log_func=log_func)
    if res:
        return res
    elif log_func:
        log_func("❌ Piped API instances failed or returned no captions.")

    # ── Method 5: yt-dlp (free local extract) ────────────────────────────────
    res = fetch_ytdlp_transcript(video_id, target_lang=target_lang, log_func=log_func)
    if res:
        return res

    # ── Method 6: RapidAPI (external plan quotas — last resort) ──────────────
    rapidapi_keys = []
    raw_r_env = os.environ.get("RAPIDAPI_KEY", "").strip()
    if raw_r_env:
        rapidapi_keys = [k.strip() for k in raw_r_env.split(",") if k.strip()]
    if hasattr(st, "session_state") and "rapidapi_keys" in st.session_state and st.session_state.rapidapi_keys:
        rapidapi_keys = st.session_state.rapidapi_keys

    if rapidapi_keys:
        if log_func: log_func(f"🔑 Falling back to RapidAPI (Pool of {len(rapidapi_keys)} keys, quota-limited)...")
        res = fetch_rapidapi_transcript(video_id, rapidapi_keys, target_lang=target_lang, log_func=log_func)
        if res:
            return res
        elif log_func:
            log_func("❌ All RapidAPI keys failed.")

    # ── Method 7: Cloudflare Worker Proxy (Bypasses IP Blocks) ───────────────
    cf_worker_url = os.environ.get("CF_WORKER_TRANSCRIPT_URL", "").strip()
    if not cf_worker_url and hasattr(st, "session_state") and "cf_worker_url" in st.session_state:
        cf_worker_url = st.session_state.cf_worker_url.strip()

    if cf_worker_url:
        if log_func: log_func("🌐 Trying Cloudflare Proxy Worker...")
        try:
            cf_worker_url = ("https://" + cf_worker_url) if not cf_worker_url.startswith("http") else cf_worker_url
            cf_worker_url = cf_worker_url.rstrip("/")
            resp = requests.get(f"{cf_worker_url}/?v={video_id}", timeout=12)
            if resp.status_code == 200:
                json_data = resp.json()
                if "transcript" in json_data and isinstance(json_data["transcript"], list):
                    lines = [item.get("text", "").strip() for item in json_data["transcript"] if item.get("text")]
                    if lines:
                        if log_func: log_func("✅ Success via Cloudflare Worker!")
                        return "\n\n".join(lines)
            else:
                if log_func: log_func(f"⚠️ Cloudflare Worker failed: {resp.status_code}")
        except Exception as cf_err:
            if log_func: log_func(f"⚠️ Cloudflare error: {cf_err}")

    return None



# Removed download_youtube_audio

# Try loading the new Google GenAI SDK (supports AQ. keys and AIza. keys)
try:
    from google import genai
    from google.genai import types
    USE_NEW_SDK = True
except ImportError:
    import google.generativeai as legacy_genai
    USE_NEW_SDK = False

st.title("🎙️ Audio & Video Transcription App")
st.write("Upload an audio/video file or paste a direct **YouTube link** to generate a transcript — powered by Google Gemini AI & Groq Whisper.")


# ── API Key Configuration ─────────────────────────────────────────────────────
api_key = os.environ.get("GEMINI_API_KEY", "").strip().strip("'\"")
raw_groq_keys = os.environ.get("GROQ_API_KEY", "").strip().strip("'\"")
raw_rapidapi_keys = os.environ.get("RAPIDAPI_KEY", "").strip().strip("'\"")

if not api_key:
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown(
        "Get your **free** Gemini API key at "
        "[aistudio.google.com/apikey](https://aistudio.google.com/apikey)"
    )
    user_api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AQ... or AIza...",
        help="Enter your API key from aistudio.google.com/apikey"
    )
    api_key = user_api_key.strip().strip("'\"")

if not raw_groq_keys:
    groq_input = st.sidebar.text_input(
        "Groq API Key(s) (Comma-separated for multi-key speed)",
        type="password",
        placeholder="gsk_key1, gsk_key2, gsk_key3",
        help="Free key(s) from console.groq.com — transcribes whole 1-hr audio in 10 seconds!"
    )
    raw_groq_keys = groq_input.strip().strip("'\"")

# Parse list of Groq keys for multi-key round-robin load balancing
groq_keys = [k.strip() for k in raw_groq_keys.split(",") if k.strip()]

# Read RapidAPI Key from backend environment variables (hidden from public UI)
if raw_rapidapi_keys:
    st.session_state.rapidapi_keys = [k.strip() for k in raw_rapidapi_keys.split(",") if k.strip()]
else:
    st.session_state.rapidapi_keys = []

# Read Cloudflare Proxy URL from backend environment variables (hidden from public UI)
cf_worker_env = os.environ.get("CF_WORKER_TRANSCRIPT_URL", "").strip()
if cf_worker_env:
    st.session_state.cf_worker_url = cf_worker_env

if not api_key and not groq_keys:
    st.info(
        "👈 Enter your **Gemini API Key** or **Groq API Key** in the sidebar to get started."
    )
    st.stop()

if USE_NEW_SDK and api_key:
    client = genai.Client(api_key=api_key)
elif api_key:
    legacy_genai.configure(api_key=api_key)

def transcribe_with_groq(chunk_path, key, language=None):
    """Transcribes audio using Groq Whisper API (whisper-large-v3-turbo)."""
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {key}"}
    with open(chunk_path, "rb") as f:
        files = {"file": (os.path.basename(chunk_path), f, "audio/mp3")}
        data = {"model": "whisper-large-v3-turbo"}
        if language:
            data["language"] = language
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("text", "")
        else:
            raise Exception(f"Groq API Error {resp.status_code}: {resp.text}")

# ── Input Source Selection ───────────────────────────────────────────────────
input_source = st.radio(
    "📥 Select Input Source",
    options=["📁 Upload Audio / Video File", "🔗 Paste YouTube Video Link"],
    horizontal=True
)

uploaded_file = None
youtube_url = ""
yt_video_id = None
yt_transcription_mode = "⚡ Auto-Fetch YouTube Captions (Instant - 1 Second)"

if input_source == "📁 Upload Audio / Video File":
    uploaded_file = st.file_uploader(
        "Choose an audio or video file (Max 30 MB)",
        type=["mp3", "mp4", "wav", "m4a", "aac", "flac", "ogg", "mov", "mkv"],
        help="Maximum file size supported is 30 MB."
    )
    with st.expander("💡 Have a file larger than 30 MB? Click here for quick compression steps"):
        st.markdown(
            "1. 🌐 Go to **[online-audio-converter.com](https://online-audio-converter.com/)**\n"
            "2. Click **'Open files'** and select your file.\n"
            "3. Click **'Advanced settings'**:\n"
            "   - Set **Bitrate** to **`32 kbps`**\n"
            "   - Set **Channels** to **`1`** (Mono)\n"
            "4. Click **'Convert'** and download your compressed file under 30 MB! 🚀"
        )
else:
    youtube_url = st.text_input(
        "🔗 Enter YouTube Video Link",
        placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
        help="Paste any public YouTube lecture, presentation, or video URL."
    ).strip()

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
        else:
            st.warning("⚠️ Invalid YouTube URL format. Please enter a valid YouTube video link.")

    yt_transcription_mode = "⚡ Auto-Fetch YouTube Captions (Instant - 1 Second)"


language_options = {
    "English 🇬🇧 (en)": "en",
    "Auto-Detect (Whisper auto-detects per chunk)": None,
    "Hindi 🇮🇳 (hi)": "hi",
    "Urdu 🇵🇰 (ur)": "ur",
    "Marathi 🇮🇳 (mr)": "mr",
    "Bengali 🇮🇳 (bn)": "bn",
    "Tamil 🇮🇳 (ta)": "ta",
    "Telugu 🇮🇳 (te)": "te",
    "Gujarati 🇮🇳 (gu)": "gu",
    "Kannada 🇮🇳 (kn)": "kn",
    "Malayalam 🇮🇳 (ml)": "ml",
    "Punjabi 🇮🇳 (pa)": "pa",
    "Spanish 🇪🇸 (es)": "es",
    "French 🇫🇷 (fr)": "fr",
    "German 🇩🇪 (de)": "de",
    "Arabic 🇸🇦 (ar)": "ar",
    "Russian 🇷🇺 (ru)": "ru",
    "Japanese 🇯🇵 (ja)": "ja",
    "Chinese 🇨🇳 (zh)": "zh"
}

selected_lang_label = st.selectbox(
    "🌐 Audio Primary Language (Fixes language mixing across chunks)",
    options=list(language_options.keys()),
    index=0,
    help="Selecting a specific language forces Whisper & Gemini to process all chunks in that language and native script, preventing unwanted translation or script switching."
)
selected_language_code = language_options[selected_lang_label]

chunk_duration = st.slider(
    "Audio Chunk Duration (minutes)",
    min_value=5,
    max_value=30,
    value=15,
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


# Check if ready to transcribe
can_proceed = False
file_size_mb = 0.0
download_filename = "transcript.txt"

if input_source == "📁 Upload Audio / Video File" and uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if uploaded_file.type.startswith("audio"):
        st.audio(uploaded_file)
    elif uploaded_file.type.startswith("video"):
        st.video(uploaded_file)

    st.caption(f"📁 File size: **{file_size_mb:.1f} MB**")

    MAX_FILE_SIZE_MB = 30
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"⚠️ **File Size Limit Exceeded ({file_size_mb:.1f} MB / Max 30 MB)**\n\n"
            f"Your uploaded file is **{file_size_mb:.1f} MB**, which exceeds the maximum allowed limit of **30 MB**.\n\n"
            f"--- \n\n"
            f"### 💡 How to compress your file under 30 MB in 4 simple steps:\n\n"
            f"1. 🌐 Go to **[online-audio-converter.com](https://online-audio-converter.com/)**\n"
            f"2. Click **'Open files'** and select your audio or video file.\n"
            f"3. Click **'Advanced settings'** button:\n"
            f"   - Change **Bitrate** dropdown to **`32 kbps`**.\n"
            f"   - Change **Channels** dropdown to **`1`** (Mono).\n"
            f"4. Click **'Convert'** and download your compressed file *(brings files down under 30 MB even for 1.5 hour long recordings!)*.\n\n"
            f"Once downloaded, upload your compressed file here to transcribe! 🚀"
        )
        st.stop()

    download_filename = f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt"
    can_proceed = True

elif input_source == "🔗 Paste YouTube Video Link" and youtube_url:
    if not yt_video_id:
        st.warning("⚠️ Please provide a valid YouTube URL to proceed.")
    else:
        download_filename = f"youtube_{yt_video_id}_transcript.txt"
        can_proceed = True


if can_proceed:
    if "transcribing" not in st.session_state:
        st.session_state.transcribing = False

    btn_container = st.empty()

    if st.session_state.transcribing:
        btn_container.button("⏳ Transcription in Progress...", disabled=True, type="secondary")
        start_clicked = False
    else:
        start_clicked = btn_container.button("Start Transcription", type="primary")

    if start_clicked:
        st.session_state.transcribing = True
        btn_container.button("⏳ Transcription in Progress...", disabled=True, type="secondary")

        progress_banner = st.empty()
        progress_banner.markdown("""
        <div class="transcribing-banner">
            <div class="loader-spinner"></div>
            <div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #ff4b4b; display: flex; align-items: center; gap: 8px;">
                    ⚡ Transcription in Progress...
                </div>
                <div style="font-size: 0.88rem; color: #dddddd; margin-top: 4px;">
                    Your audio is being processed and transcribed in real-time. Please stay on this page.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                status_box = st.empty()
                progress_bar = st.progress(0)

                st.subheader("📋 Live Activity Logs")
                log_box = st.empty()

                st.subheader("📝 Live Transcript Preview")
                transcript_preview = st.empty()

                logs = []
                import queue
                log_queue = queue.Queue()

                def log(msg):
                    ts = time.strftime("%H:%M:%S")
                    log_queue.put(f"[{ts}] {msg}")

                def flush_logs():
                    updated = False
                    while not log_queue.empty():
                        try:
                            logs.append(log_queue.get_nowait())
                            updated = True
                        except Exception:
                            break
                    if updated:
                        log_box.code("\n".join(logs[-40:]), language="text")

                final_text = ""
                start_time = time.time()

                # ── Fast Path: YouTube Auto-Fetch Captions ──────────────────────
                if input_source == "🔗 Paste YouTube Video Link" and "Auto-Fetch" in yt_transcription_mode:
                    status_box.info("⚡ Attempting instant fetch of YouTube captions...")
                    log(f"Fetching official/auto captions for YouTube Video ID '{yt_video_id}'...")
                    flush_logs()

                    try:
                        yt_captions = fetch_youtube_captions(yt_video_id, target_lang=selected_language_code, log_func=log)
                    except Exception as cap_err:
                        yt_captions = None
                        log(f"⚠️ Caption fetch error: {type(cap_err).__name__}: {cap_err}")
                        flush_logs()

                    if yt_captions:
                        final_text = yt_captions
                        log(f"🎉 Successfully fetched YouTube captions instantly!")
                        flush_logs()

                    else:
                        log("⚠️ No captions found via YouTube API. Falling back to Full AI Audio Download & Transcription...")
                        flush_logs()


                # ── Full AI Audio Pipeline (Upload File OR YouTube Fallback) ───
                if not final_text:
                    original_path = ""

                    if input_source == "📁 Upload Audio / Video File":
                        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"
                        original_path = os.path.join(tmp_dir, "input" + suffix)

                        status_box.info("💾 Step 1/3: Saving uploaded file to local memory...")
                        log(f"Saving '{uploaded_file.name}' ({file_size_mb:.1f} MB)...")
                        flush_logs()
                        with open(original_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        log("Saved file successfully.")
                        flush_logs()

                    else:  # YouTube (Auto-fetch failed)
                        raise Exception(
                            "❌ YouTube Auto-Captions not found or blocked!\n\n"
                            "This video might not have captions, or YouTube is blocking the server IP.\n\n"
                            "**Workaround:** Download the video locally and use the '📁 Upload Audio / Video File' tab to transcribe it."
                        )

                    # ── Step 2: Chunk media file into segments ────────────────
                    status_box.info(f"⚡ Step 2/3: Slicing audio into {chunk_duration}-minute chunks with ffmpeg...")
                    log(f"Running ffmpeg to split audio into {chunk_duration}-minute chunks...")

                    chunk_files = chunk_media_file(original_path, tmp_dir, chunk_minutes=chunk_duration)
                    total_chunks = len(chunk_files)
                    log(f"Audio split complete! Created {total_chunks} chunk file(s).")
                    status_box.info(f"✅ Prepared **{total_chunks} chunk(s)** for processing.")

                    lang_rule = ""
                    if selected_language_code:
                        lang_name = selected_lang_label.split(" (")[0]
                        lang_rule = f"\n4. The primary spoken language is {lang_name}. Transcribe strictly in {lang_name} using its native script."

                    prompt = (
                        "Please transcribe the speech in this audio file accurately. "
                        "Important rules:\n"
                        "1. Output only the spoken text preserving natural paragraph breaks.\n"
                        "2. If there are repeating chants, mantras, or background music, transcribe the words accurately without repeating the same line over and over endlessly.\n"
                        "3. Do not add commentary, timestamps, or extra formatting."
                        f"{lang_rule}"
                    )

                    # ── Step 3: Process chunks in parallel ────────────────────
                    log("⚡ Launching Parallel Processing...")
                    import concurrent.futures

                    if groq_keys:
                        fallback_models = ["groq-whisper", DEFAULT_GEMINI_MODEL, "gemini-3.5-flash"]
                    else:
                        fallback_models = [DEFAULT_GEMINI_MODEL, "gemini-3.5-flash", "gemini-3.0-flash"]

                    transcripts_dict = {}
                    completed_count = 0

                    def process_chunk_worker(chunk_info):
                        idx, chunk_path = chunk_info
                        chunk_num = idx + 1
                        max_retries = 4

                        for attempt in range(max_retries):
                            current_model = fallback_models[attempt % len(fallback_models)]
                            audio_file = None

                            try:
                                if current_model == "groq-whisper" and groq_keys:
                                    active_groq_key = groq_keys[idx % len(groq_keys)]
                                    log(f"🚀 [Chunk {chunk_num}/{total_chunks}] Transcribing with Groq Whisper (Key #{idx % len(groq_keys) + 1}, Lang: {selected_language_code or 'auto'})...")
                                    text_chunk = transcribe_with_groq(chunk_path, active_groq_key, language=selected_language_code)
                                    if text_chunk:
                                        words = len(text_chunk.split())
                                        log(f"⚡ [Chunk {chunk_num}/{total_chunks}] Groq complete! Transcribed {words} words.")
                                        return (idx, text_chunk)

                                log(f"[Chunk {chunk_num}/{total_chunks}] Uploading to Gemini ({current_model})...")

                                if USE_NEW_SDK:
                                    audio_file = client.files.upload(file=chunk_path)
                                    log(f"[Chunk {chunk_num}/{total_chunks}] Uploaded. Storage ID: {audio_file.name}")

                                    wait_count = 0
                                    while hasattr(audio_file, 'state') and str(getattr(audio_file.state, 'name', audio_file.state)) == "PROCESSING":
                                        time.sleep(2)
                                        audio_file = client.files.get(name=audio_file.name)
                                        wait_count += 1
                                        if wait_count > 20:
                                            break

                                    log(f"🧠 [Chunk {chunk_num}/{total_chunks}] Transcribing with {current_model}...")
                                    config = types.GenerateContentConfig(
                                        temperature=0.0,
                                        max_output_tokens=3000,
                                        system_instruction="You are a precise audio transcription expert. Transcribe spoken words accurately. Never repeat phrases endlessly during music, chants, or silence."
                                    )
                                    response = client.models.generate_content(
                                        model=current_model,
                                        contents=[audio_file, prompt],
                                        config=config
                                    )
                                else:
                                    audio_file = legacy_genai.upload_file(path=chunk_path)
                                    log(f"[Chunk {chunk_num}/{total_chunks}] Uploaded. Storage ID: {audio_file.name}")

                                    wait_count = 0
                                    while audio_file.state.name == "PROCESSING":
                                        time.sleep(2)
                                        audio_file = legacy_genai.get_file(audio_file.name)
                                        wait_count += 1
                                        if wait_count > 20:
                                            break

                                    log(f"🧠 [Chunk {chunk_num}/{total_chunks}] Transcribing with {current_model}...")
                                    model = legacy_genai.GenerativeModel(
                                        model_name=current_model,
                                        generation_config={"temperature": 0.0, "max_output_tokens": 3000}
                                    )
                                    response = model.generate_content(
                                        [prompt, audio_file],
                                        request_options={"timeout": 600}
                                    )

                                if response and response.text:
                                    text_chunk = response.text.strip()
                                    words = len(text_chunk.split())
                                    log(f"✅ [Chunk {chunk_num}/{total_chunks}] Complete! Transcribed {words} words.")
                                    return (idx, text_chunk)

                            except Exception as err:
                                err_msg = str(err)
                                if ("503" in err_msg or "UNAVAILABLE" in err_msg or "429" in err_msg or "high demand" in err_msg) and attempt < max_retries - 1:
                                    next_model = fallback_models[(attempt + 1) % len(fallback_models)]
                                    log(f"⚠️ [Chunk {chunk_num}/{total_chunks}] Server busy (503). Retrying with '{next_model}'...")
                                    time.sleep(3)
                                else:
                                    if attempt == max_retries - 1:
                                        log(f"❌ [Chunk {chunk_num}/{total_chunks}] Error: {err}")
                                        raise err
                                    time.sleep(3)
                            finally:
                                if audio_file:
                                    try:
                                        if USE_NEW_SDK:
                                            client.files.delete(name=audio_file.name)
                                        else:
                                            legacy_genai.delete_file(audio_file.name)
                                    except Exception:
                                        pass

                        return (idx, "")

                    num_workers = min(6, max(3, len(groq_keys)))
                    log(f"⚡ Running with {num_workers} parallel workers across {max(1, len(groq_keys))} key(s)...")

                    chunk_tuples = list(enumerate(chunk_files))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                        future_to_chunk = {executor.submit(process_chunk_worker, item): item for item in chunk_tuples}

                        for future in concurrent.futures.as_completed(future_to_chunk):
                            flush_logs()
                            completed_count += 1
                            idx, text_chunk = future.result()
                            transcripts_dict[idx] = text_chunk

                            ordered_texts = [transcripts_dict[i] for i in range(total_chunks) if i in transcripts_dict]
                            current_combined = "\n\n".join(ordered_texts)
                            transcript_preview.text_area("Live Output", current_combined, height=250, key=f"preview_{completed_count}")

                            progress_bar.progress(completed_count / total_chunks)
                            status_box.info(f"⚡ Parallel Processing: **{completed_count} of {total_chunks} chunks completed**...")
                            flush_logs()

                    final_text = "\n\n".join([transcripts_dict[i] for i in range(total_chunks) if i in transcripts_dict]).strip()

                flush_logs()
                total_time = int(time.time() - start_time)
                status_box.empty()
                progress_bar.empty()

                # ── Step 4: Output ─────────────────────────────────────────────
                if final_text:
                    transcript_preview.text_area("Final Transcript", final_text, height=300, key="final_transcript")
                    st.success(f"🎉 Transcription complete in {total_time} seconds!")
                    st.download_button(
                        label="📥 Download Complete Transcript (.txt)",
                        data=final_text,
                        file_name=download_filename,
                        mime="text/plain"
                    )
                else:
                    st.warning("No transcript was generated. The video/audio may have no speech or could not be processed.")

            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
            finally:
                st.session_state.transcribing = False
                if 'progress_banner' in locals():
                    progress_banner.empty()
                gc.collect()


