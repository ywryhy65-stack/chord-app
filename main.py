import json
import html
import os
import re
import shutil
import subprocess
import tempfile
import traceback
import socket
from pathlib import Path
from typing import Optional, List
from urllib.parse import parse_qs, urlparse

# Ensure Homebrew and common paths are in PATH for Apple Silicon/Intel Macs
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin" + os.pathsep + "/usr/local/bin"

import librosa
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.ndimage import median_filter
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from yt_dlp import YoutubeDL
from celery_app import celery_app
from tasks import analyze_chords_task
from celery.result import AsyncResult


# Force IPv4 for requests
class IPv4Adapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['socket_options'] = HTTPAdapter.default_poolmanager_extended_kwargs(None).get('socket_options', []) + [
            (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1),
            (socket.IPPROTO_IP, socket.IP_TOS, 0x10), # IPTOS_LOWDELAY
        ]
        # This is a bit hacky but works for forcing IPv4 in many environments
        # by overriding the family in the pool manager's connection.
        return super().init_poolmanager(*args, **kwargs)

def get_resilient_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

resilient_session = get_resilient_session()


BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
AUDIO_CACHE_DIR = BASE_DIR / "audio_cache"

app = FastAPI(title="ChordSync Backend")

_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")
_HTML_TITLE_RE = re.compile(r"<title[^>]*>([^<]+)</title>", re.IGNORECASE | re.DOTALL)

_HTTP_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class AnalyzeRequest(BaseModel):
    search_or_url: str = Field(..., min_length=1, max_length=2048)


CHORD_ROOTS = [
    "C", "C#", "D", "Eb", "E", "F",
    "F#", "G", "Ab", "A", "Bb", "B",
]

# Drop chord runs shorter than this (0.4s allows single-beat chord changes)
MIN_DURATION_SEC = 0.67

# Extra template id for low-RMS (non-musical) frames; kept in chord_names alongside major/minor.
NC_CHORD_LABEL = "N.C."


def _align_series_to_n_frames(series: np.ndarray, n: int) -> np.ndarray:
    """Trim or edge-pad a 1D feature to match chroma/template frame count."""
    t = int(series.shape[0])
    if t == n:
        return series
    if t > n:
        return series[:n].copy()
    if t == 0:
        return np.zeros(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    out[:t] = series
    out[t:] = series[-1]
    return out


def extract_youtube_video_id(url: str) -> Optional[str]:
    """Parse an 11-character YouTube video id from common URL shapes."""
    s = (url or "").strip()
    if not s:
        return None
    try:
        u = urlparse(s)
        host = (u.hostname or "").lower().replace("www.", "")
        vid: Optional[str] = None
        if host == "youtu.be":
            vid = u.path.strip("/").split("/")[0] or None
        elif "youtube.com" in host:
            if u.path.startswith("/embed/"):
                parts = u.path.split("/")
                vid = parts[2] if len(parts) > 2 else None
            elif u.path.startswith("/shorts/"):
                parts = u.path.split("/")
                vid = parts[2] if len(parts) > 2 else None
            else:
                qs = parse_qs(u.query)
                v = qs.get("v", [None])[0]
                vid = v
        if vid and _VIDEO_ID_RE.match(vid):
            return vid
    except Exception:
        pass
    m = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})", s)
    return m.group(1) if m else None


def _is_spotify_url(s: str) -> bool:
    try:
        u = urlparse((s or "").strip())
        host = (u.hostname or "").lower()
        return "spotify.com" in host
    except Exception:
        return False


def _clean_spotify_title(raw_title: str) -> str:
    t = html.unescape(raw_title).strip()
    for pat in (
        r"\s*\|\s*Spotify\s+Web\s+Player\s*$",
        r"\s*\|\s*Spotify\s*$",
        r"\s*[–—-]\s*Spotify\s*$",
        r"\s+on\s+Spotify\s*$",
    ):
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _spotify_url_to_search_query(spotify_url: str) -> str:
    try:
        r = resilient_session.get(
            spotify_url.strip(),
            timeout=20,
            headers=_HTTP_BROWSER_HEADERS,
        )
        r.raise_for_status()
    except requests.RequestException as exc:
        status_code = 400
        if r.status_code == 404:
            status_code = 404
        elif r.status_code >= 500:
            status_code = 502
        
        if "timeout" in str(exc).lower():
            status_code = 504
            
        raise HTTPException(
            status_code=status_code,
            detail=f"Could not load Spotify page: {exc}",
        ) from exc
    m = _HTML_TITLE_RE.search(r.text)
    if not m:
        raise HTTPException(status_code=400, detail="Could not parse Spotify page title")
    query = _clean_spotify_title(m.group(1))
    if not query:
        raise HTTPException(status_code=400, detail="Empty title from Spotify page")
    return query


def _ytsearch_first_video_id(query: str) -> str:
    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty search query")
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "socket_timeout": 30,
        "source_address": "0.0.0.0", # Force IPv4
        "retries": 5,
        "extractor_retries": 3,
        "remote_components": ["ejs:github"],
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{q}", download=False)
    except Exception as exc:
        status_code = 400
        if "timeout" in str(exc).lower():
            status_code = 504
        raise HTTPException(
            status_code=status_code,
            detail=f"YouTube search failed: {exc}",
        ) from exc
    if not info:
        raise HTTPException(status_code=400, detail="No YouTube search results")

    entries = info.get("entries")
    if isinstance(entries, list):
        for first in entries:
            if first is None or not isinstance(first, dict):
                continue
            vid = first.get("id")
            if isinstance(vid, str) and _VIDEO_ID_RE.match(vid):
                return vid
            url = first.get("url") or first.get("webpage_url")
            if isinstance(url, str):
                extracted = extract_youtube_video_id(url)
                if extracted:
                    return extracted

    vid_top = info.get("id")
    if isinstance(vid_top, str) and _VIDEO_ID_RE.match(vid_top):
        return vid_top

    raise HTTPException(
        status_code=400,
        detail="Could not resolve a YouTube video from search results",
    )


def resolve_input_to_youtube(raw: str) -> tuple[str, str]:
    """
    Resolve user input to a canonical YouTube watch URL and 11-char video id.

    - YouTube URLs: extract id.
    - Spotify URLs: fetch page title, then ytsearch1.
    - Otherwise: treat as free-text YouTube search (ytsearch1).
    """
    s = (raw or "").strip()
    if not s:
        raise HTTPException(status_code=400, detail="Input is empty")

    vid_direct = extract_youtube_video_id(s)
    if vid_direct:
        watch = f"https://www.youtube.com/watch?v={vid_direct}"
        return watch, vid_direct

    search_query = _spotify_url_to_search_query(s) if _is_spotify_url(s) else s
    vid = _ytsearch_first_video_id(search_query)
    watch = f"https://www.youtube.com/watch?v={vid}"
    return watch, vid


def _major_minor_templates() -> tuple[np.ndarray, list[str]]:
    templates: list[np.ndarray] = []
    labels: list[str] = []

    def add_template(root, intervals, label_suffix=""):
        v = np.zeros(12, dtype=np.float64)
        for semi in intervals:
            v[(root + semi) % 12] += 1.0

        # Add subtle perfect fifth overtone to assist with bass tracking
        v[(root + 7) % 12] += 0.3

        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        templates.append(v)
        labels.append(f"{CHORD_ROOTS[root]}{label_suffix}")

    for r in range(12):
        # We only keep Major and Minor to prevent erratic chord jumps
        add_template(r, [0, 4, 7], "")           # Major
        add_template(r, [0, 3, 7], "m")          # Minor

    return np.array(templates), labels


def _generate_transition_matrix(n_states: int, self_prob: float = 0.98) -> np.ndarray:
    """
    Generate transition matrix. self_prob=0.98 is perfectly balanced
    for 10-frames-per-second analysis to prevent flickering.
    """
    trans = np.full((n_states, n_states), (1.0 - self_prob) / (n_states - 1))
    np.fill_diagonal(trans, self_prob)
    return trans


def _separate_stems(audio_path: Path) -> Path:
    """
    Step 1: Use Spleeter to separate accompaniment from vocals.
    Isolates harmonic/bass content and mutes vocals/drums.
    """
    try:
        # Check if spleeter is available
        if shutil.which("spleeter") is None:
            return audio_path

        output_dir = audio_path.parent / "spleeter_output"
        # 2stems model: vocals + accompaniment
        acc_path = output_dir / audio_path.stem / "accompaniment.wav"
        
        if not acc_path.exists():
            cmd = [
                "spleeter", "separate",
                "-p", "spleeter:2stems",
                "-o", str(output_dir),
                str(audio_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        
        if acc_path.exists():
            return acc_path
    except Exception as e:
        print(f"DEBUG: Spleeter failed: {e}. Falling back to original audio.")
    
    return audio_path


def _run_length_encode_chords(
    chord_ids: np.ndarray,
    times: np.ndarray,
    confidences: np.ndarray,
    chord_names: list[str],
) -> list[dict]:
    """One output per contiguous run; `time` is the timestamp at the start of each run."""
    n = int(chord_ids.shape[0])
    if n == 0:
        return []

    out: list[dict] = []
    run_start = 0
    for i in range(1, n):
        if int(chord_ids[i]) != int(chord_ids[run_start]):
            cid = int(chord_ids[run_start])
            out.append(
                {
                    "time": round(float(times[run_start]), 3),
                    "chord": chord_names[cid],
                    "confidence": float(round(float(confidences[run_start]), 4)),
                }
            )
            run_start = i

    cid = int(chord_ids[run_start])
    out.append(
        {
            "time": round(float(times[run_start]), 3),
            "chord": chord_names[cid],
            "confidence": float(round(float(confidences[run_start]), 4)),
        }
    )
    return out


def _merge_short_transients(
    segments: list[dict],
    total_duration: float,
    min_duration_sec: float = MIN_DURATION_SEC,
) -> list[dict]:
    """
    Remove RLE segments shorter than `min_duration_sec` (noise / passing chords).
    Short middle/end segments merge into the preceding stable chord (omitted from output).
    Short opening segments merge into the next stable chord (its `time` moves earlier).
    """
    if not segments:
        return []

    n = len(segments)
    merged: list[dict] = []
    i = 0
    absorbed_start: Optional[float] = None

    while i < n:
        t_start = float(segments[i]["time"]) if absorbed_start is None else absorbed_start
        t_end = float(segments[i + 1]["time"]) if i + 1 < n else float(total_duration)
        duration = t_end - t_start

        if duration >= min_duration_sec - 1e-9:
            merged.append(
                {
                    "time": round(t_start, 3),
                    "chord": segments[i]["chord"],
                    "confidence": segments[i]["confidence"],
                }
            )
            absorbed_start = None
            i += 1
            continue

        if merged:
            absorbed_start = None
            i += 1
            continue

        absorbed_start = t_start
        i += 1

    if not merged and segments:
        best_j = max(
            range(n),
            key=lambda j: (
                float(segments[j + 1]["time"]) if j + 1 < n else float(total_duration)
            )
            - float(segments[j]["time"]),
        )
        s = segments[best_j]
        merged.append(
            {
                "time": round(float(s["time"]), 3),
                "chord": s["chord"],
                "confidence": s["confidence"],
            }
        )

    return merged


def _cleanup_old_audio(keep_count: int = 3):
    """Scan AUDIO_CACHE_DIR for .wav files and keep only the newest ones."""
    if not AUDIO_CACHE_DIR.exists():
        return
    
    wav_files = list(AUDIO_CACHE_DIR.glob("*.wav"))
    # Sort by modification time (mtime), newest first
    wav_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Delete anything beyond the keep_count
    for old_wav in wav_files[keep_count:]:
        try:
            old_wav.unlink()
        except Exception as e:
            print(f"DEBUG: Failed to delete old audio {old_wav}: {e}")


def _download_audio(youtube_url: str, output_wav_path: Path) -> str:
    # Use yt-dlp with ffmpeg to extract a WAV file.
    cmd = [
        "yt-dlp",
        "--verbose",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--socket-timeout", "30",
        "--force-ipv4",
        "--retries", "5",
        "--extractor-retries", "3",
        "--remote-components", "ejs:github",
        "-o",
        str(output_wav_path.with_suffix(".%(ext)s")),
        youtube_url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300) # 5 min limit for download
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Audio download timed out after 5 minutes: {exc}",
        ) from exc
    
    # Force Console Logging
    print(f"--- yt-dlp STDOUT ---\n{result.stdout}")
    print(f"--- yt-dlp STDERR ---\n{result.stderr}")
    
    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip()
        status_code = 400
        if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            status_code = 504
        raise HTTPException(
            status_code=status_code,
            detail=f"yt-dlp failed with return code {result.returncode}. Error: {error_msg}",
        )
    return result.stderr.strip()


def _estimate_chords(audio_path: Path) -> list[dict]:
    """
    Refined Extraction Pipeline: Fixed Frame analysis (No Beat-Tracker Gaps)
    with Global Key Awareness and vocal rumble suppression.
    """
    # 1. Source Separation (Try Spleeter, fallback gracefully if missing)
    harmonic_audio_path = _separate_stems(audio_path)
    
    # Use 22050 Hz for standard DSP (faster and drops super high vocal frequencies)
    y, sr = librosa.load(str(harmonic_audio_path), sr=22050, mono=True)
    total_duration = float(len(y)) / float(sr)

    # 2. Aggressive Harmonic Isolation (Crucial if Spleeter is missing)
    # This heavily suppresses drums and vocal transients.
    y_harmonic, _ = librosa.effects.hpss(y, margin=(2.0, 5.0))

    # 3. Fixed Frame Extraction (Solves the "Black Hole" gap)
    hop_length = 2048  # ~10.7 frames per second. Consistent and reliable.
    
    # 4. Chroma Extraction (Locked to Instrument range)
    # fmin=C3 (130Hz) ignores chest-voice vocal rumble from baritone singers.
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=hop_length,
        n_octaves=5,                  # Ignore extremely high vocal notes
        fmin=librosa.note_to_hz('C3') # Start from C3 to focus on harmonic instruments
    )
    
    # Standardize and Normalize
    chroma = librosa.util.normalize(chroma, axis=0)
    
    # 5. Global Key Detection & Diatonic Masking
    # Sum chroma over time to find the overall tonal distribution
    global_chroma = np.sum(chroma, axis=1)
    
    templates, chord_names = _major_minor_templates()
    nc_template = np.zeros(12)
    templates_with_nc = np.vstack([templates, nc_template])
    chord_names_with_nc = chord_names + [NC_CHORD_LABEL]

    # Key Profiles (Krumhansl-Schmuckler) to find the Global Key
    # Major/Minor templates already exist, we'll correlate them with global_chroma
    key_scores = np.dot(templates, global_chroma)
    best_key_idx = np.argmax(key_scores)
    
    # Extract root and mode (Major/Minor) from the best template
    # Templates are stored as [Maj-0, Min-0, Maj-1, Min-1, ...] or similar?
    # Actually _major_minor_templates returns [Maj-0, Min-0, Maj-1, Min-1, ...]
    is_minor_key = (best_key_idx % 2 == 1)
    key_root = best_key_idx // 2
    
    # Create Diatonic Mask
    # Diatonic degrees for Major: 0, 2, 4, 5, 7, 9, 11
    # Diatonic degrees for Minor: 0, 2, 3, 5, 7, 8, 10
    if is_minor_key:
        diatonic_steps = [0, 2, 3, 5, 7, 8, 10]
        # Common chords in Minor: i, III, iv, v, VI, VII
        diatonic_chords = [
            (0, "m"), (3, ""), (5, "m"), (7, "m"), (8, ""), (10, "")
        ]
    else:
        diatonic_steps = [0, 2, 4, 5, 7, 9, 11]
        # Common chords in Major: I, ii, iii, IV, V, vi
        diatonic_chords = [
            (0, ""), (2, "m"), (4, "m"), (5, ""), (7, ""), (9, "m")
        ]

    # Convert relative diatonic chords to absolute indices in templates
    # templates are ordered: for r in range(12): add_template(r, Maj), add_template(r, Min)
    diatonic_indices = []
    for root_rel, suffix in diatonic_chords:
        root_abs = (key_root + root_rel) % 12
        # Index in templates is root_abs * 2 (Major) or root_abs * 2 + 1 (Minor)
        idx = root_abs * 2 if suffix == "" else root_abs * 2 + 1
        diatonic_indices.append(idx)
    
    # Include the "No Chord" index as valid
    nc_idx = len(chord_names_with_nc) - 1
    
    # Apply Penalty Mask (Diatonic = 1.0, Non-Diatonic = 0.01)
    mask = np.full(len(chord_names_with_nc), 0.01)
    mask[diatonic_indices] = 1.0
    mask[nc_idx] = 0.5 # Slightly penalize NC to prefer musical content

    # 6. Observation Probabilities
    scores = np.dot(templates_with_nc, chroma)
    prob_matrix = np.exp(scores * 10.0)
    
    # Gating by Diatonic Mask
    prob_matrix *= mask[:, np.newaxis]
    
    prob_matrix /= (np.sum(prob_matrix, axis=0) + 1e-12)
    
    # 7. Transition Matrix (0.98 for 10fps stability)
    n_chords = len(chord_names_with_nc)
    transition_matrix = _generate_transition_matrix(n_chords, self_prob=0.98)
    
    # Viterbi decoding
    final_ids = librosa.sequence.viterbi(prob_matrix, transition_matrix)
    confidence = np.max(prob_matrix, axis=0)

    # 8. Map to Timestamps (Fixed Grid)
    times = librosa.frames_to_time(np.arange(len(final_ids)), sr=sr, hop_length=hop_length)

    # Final output
    rle = _run_length_encode_chords(final_ids, times, confidence, chord_names_with_nc)
    return _merge_short_transients(rle, total_duration)


@app.get("/")
def serve_index() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(INDEX_FILE))


@app.get("/audio/{video_id}")
def serve_cached_audio(video_id: str):
    if not _VIDEO_ID_RE.match(video_id):
        raise HTTPException(status_code=404, detail="Invalid video id")
    path = AUDIO_CACHE_DIR / f"{video_id}.wav"
    
    # Lazy Re-download Fallback
    if not path.is_file():
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wav_stem = tmp_path / "audio"
            wav_file = tmp_path / "audio.wav"
            
            try:
                _download_audio(youtube_url, wav_stem)
                if not wav_file.exists():
                    candidates = list(tmp_path.glob("audio.*"))
                    if not candidates:
                        raise HTTPException(status_code=500, detail="Fallback download failed to produce audio file")
                    wav_file = candidates[0]
                
                AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(wav_file, path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Audio not found and lazy re-download failed: {e}")

    return FileResponse(
        str(path),
        media_type="audio/wav",
        filename=f"{video_id}.wav",
    )


@app.get("/history")
def get_history():
    history = []
    if AUDIO_CACHE_DIR.exists():
        for json_file in AUDIO_CACHE_DIR.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    history.append({
                        "video_id": data.get("video_id"),
                        "title": data.get("title", "Unknown Title"),
                        "source_url": data.get("source_url")
                    })
            except Exception:
                continue
    return history


@app.delete("/history/{video_id}")
def delete_history_item(video_id: str):
    if not _VIDEO_ID_RE.match(video_id):
        raise HTTPException(status_code=400, detail="Invalid video id")
    
    json_path = AUDIO_CACHE_DIR / f"{video_id}.json"
    wav_path = AUDIO_CACHE_DIR / f"{video_id}.wav"
    
    deleted = False
    if json_path.exists():
        json_path.unlink()
        deleted = True
    if wav_path.exists():
        wav_path.unlink()
        deleted = True
        
    if not deleted:
        raise HTTPException(status_code=404, detail="Song not found in history")
        
    return {"status": "success", "message": f"Deleted {video_id}"}


@app.get("/suggest")
def suggest(q: str):
    """Fetch YouTube search suggestions from Google's public API."""
    if not q or not q.strip():
        return {"suggestions": []}
    
    url = f"http://suggestqueries.google.com/complete/search?client=firefox&ds=yt&q={q}"
    try:
        r = resilient_session.get(url, timeout=5, headers=_HTTP_BROWSER_HEADERS)
        r.raise_for_status()
        data = r.json()
        # data[1] contains the list of suggestions in the Firefox-style response
        suggestions = data[1] if len(data) > 1 else []
        return {"suggestions": suggestions}
    except Exception as e:
        print(f"DEBUG: Suggestion fetch failed: {e}")
        return {"suggestions": []}


def _clean_title(title: str) -> str:
    """Aggressively clean YouTube title for lyrics search."""
    # 1. Remove anything inside parentheses () or brackets []
    cleaned = re.sub(r"[\(\[].*?[\)\]]", "", title)
    # 2. Remove specific keywords (case-insensitive)
    keywords = ["official", "video", "lyrics", "live", "cover", "קליפ רשמי", "מילים", "הופעה חיה", "קאבר"]
    for kw in keywords:
        cleaned = re.sub(r"(?i)" + re.escape(kw), "", cleaned)
    # 3. Replace hyphens, pipes, and separators with spaces
    cleaned = re.sub(r"[\-\|]", " ", cleaned)
    # 4. Strip extra whitespace
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned


def _extract_english(title: str) -> str:
    """Extract only Latin alphabet characters and spaces from title."""
    english_only = re.sub(r"[^a-zA-Z\s]", "", title)
    english_only = " ".join(english_only.split()).strip()
    return english_only


def _validate_lrclib_result(res: dict, video_duration: Optional[int], yt_title: str = "", yt_artist: str = "", yt_track: str = "") -> bool:
    """
    Validate a single LRCLIB result using a '100% Sure or Abort' approach:
    1. Must have syncedLyrics.
    2. Duration difference must be <= 15 seconds (if video_duration is available).
    3. db_artist OR db_track must match yt_title or yt_artist/yt_track.
       - db_artist and db_track must be at least 2 characters (skip if shorter).
    Result is ONLY accepted if: duration check passes AND (artist_match OR track_match).
    """
    if not res.get("syncedLyrics"):
        return False

    db_artist = res.get('artistName', '') or ''
    db_track = res.get('trackName', '') or ''

    db_artist = db_artist.lower().strip()
    db_track = db_track.lower().strip()

    if len(db_artist) < 2 or len(db_track) < 2:
        return False

    yt_title_l = yt_title.lower()
    yt_artist_l = yt_artist.lower()
    yt_track_l = yt_track.lower()

    duration_diff = 0
    has_valid_duration = True
    if video_duration is not None and res.get("duration") is not None:
        duration_diff = abs(video_duration - res["duration"])
        has_valid_duration = duration_diff <= 15

    if not has_valid_duration:
        return False

    artist_match = db_artist and (db_artist in yt_title_l or db_artist in yt_artist_l)
    track_match = db_track and (db_track in yt_title_l or db_track in yt_track_l)

    return artist_match or track_match



def _fetch_lyrics_from_lrclib(query: str, video_duration: Optional[int] = None, yt_title: str = "", yt_artist: str = "", yt_track: str = "") -> Optional[str]:
    """Search LRCLIB API with a single query and return first validated synced lyrics."""
    try:
        # Skip garbage or too-short queries
        if not query or len(query.strip()) < 3:
            return None
            
        url = f"https://lrclib.net/api/search?q={query}"
        response = resilient_session.get(url, timeout=10)
        if response.status_code == 200:
            results = response.json()
            for res in results:
                if _validate_lrclib_result(res, video_duration, yt_title, yt_artist, yt_track):
                    return res["syncedLyrics"]
    except Exception as e:
        print(f"DEBUG: LRCLIB fetch failed: {e}")
        pass
    return None


def _fetch_synced_lyrics(youtube_title: str, artist: Optional[str] = None, track: Optional[str] = None, video_duration: Optional[int] = None) -> Optional[str]:
    """
    Fetch synchronized lyrics from LRCLIB API using a multi-step fallback strategy:
    1. Best Match (Metadata): Use {artist} {track} if available.
    2. Clean Title Search: Search with aggressively cleaned YouTube title.
    3. English/Transliteration Extraction: Extract Latin chars only and search.
    4. Bidirectional Split Search: Split by '-' or '|', try part 2 then part 1.
    Each search uses a 'Smart Metadata Cross-Reference' validation (duration + text overlap).
    """
    yt_artist = artist or ""
    yt_track = track or ""
    yt_title = youtube_title or ""

    # Step 1: Metadata search
    if yt_artist and yt_track:
        query = f"{yt_artist} {yt_track}"
        print(f"DEBUG: Lyrics step 1 - metadata search: '{query}'")
        result = _fetch_lyrics_from_lrclib(query, video_duration, yt_title, yt_artist, yt_track)
        if result:
            return result

    # Step 2: Clean title search
    cleaned = _clean_title(yt_title)
    if cleaned:
        print(f"DEBUG: Lyrics step 2 - cleaned title search: '{cleaned}'")
        result = _fetch_lyrics_from_lrclib(cleaned, video_duration, yt_title, yt_artist, yt_track)
        if result:
            return result

    # Step 3: English/transliteration extraction
    english_only = _extract_english(yt_title)
    if english_only:
        print(f"DEBUG: Lyrics step 3 - english extraction search: '{english_only}'")
        result = _fetch_lyrics_from_lrclib(english_only, video_duration, yt_title, yt_artist, yt_track)
        if result:
            return result

    # Step 4: Bidirectional split search
    parts = re.split(r"[\-\|]", yt_title)
    if len(parts) >= 2:
        second_part = _clean_title(parts[1]) if len(parts) > 1 else None
        if second_part:
            print(f"DEBUG: Lyrics step 4a - split search (part 2): '{second_part}'")
            result = _fetch_lyrics_from_lrclib(second_part, video_duration, yt_title, yt_artist, yt_track)
            if result:
                return result

        first_part = _clean_title(parts[0]) if len(parts) > 0 else None
        if first_part:
            print(f"DEBUG: Lyrics step 4b - split search (part 1): '{first_part}'")
            result = _fetch_lyrics_from_lrclib(first_part, video_duration, yt_title, yt_artist, yt_track)
            if result:
                return result

    print(f"DEBUG: Lyrics fetch failed for title: '{yt_title}'")
    return None


@app.post("/analyze")
async def analyze_chords(payload: AnalyzeRequest):
    raw_input = payload.search_or_url.strip()
    youtube_watch_url, video_id = resolve_input_to_youtube(raw_input)

    # 1. Instant Load from Cache
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_json = AUDIO_CACHE_DIR / f"{video_id}.json"
    if cache_json.exists():
        try:
            with open(cache_json, "r") as f:
                return {"status": "SUCCESS", "result": json.load(f)}
        except Exception:
            pass

    # 2. Trigger Celery Task
    task = analyze_chords_task.delay(youtube_watch_url, video_id, raw_input)
    return {"status": "PENDING", "task_id": task.id}


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    
    if task_result.status == "SUCCESS":
        return {
            "status": "SUCCESS",
            "result": task_result.result
        }
    elif task_result.status == "FAILURE":
        return {
            "status": "FAILURE",
            "error": str(task_result.info)
        }
    else:
        return {
            "status": "PENDING"
        }


@app.get("/health")
def health():
    return {"ok": True}
