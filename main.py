import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import librosa
import numpy as np
import requests
from scipy.ndimage import median_filter
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from yt_dlp import YoutubeDL


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


CHORD_LABELS = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]

# Drop chord runs shorter than this (passing notes / glissandi); time folds into prior stable chord.
MIN_DURATION_SEC = 0.8

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
        r = requests.get(
            spotify_url.strip(),
            timeout=20,
            headers=_HTTP_BROWSER_HEADERS,
        )
        r.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=400,
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
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{q}", download=False)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
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
    """24 chroma templates (12 major + 12 minor), L2-normalized for cosine similarity."""
    majors: list[np.ndarray] = []
    minors: list[np.ndarray] = []
    labels: list[str] = []

    for r in range(12):
        v = np.zeros(12, dtype=np.float64)
        v[r] = 1.0
        v[(r + 4) % 12] += 1.0
        v[(r + 7) % 12] += 1.0
        majors.append(v / np.linalg.norm(v))

    for r in range(12):
        v = np.zeros(12, dtype=np.float64)
        v[r] = 1.0
        v[(r + 3) % 12] += 1.0
        v[(r + 7) % 12] += 1.0
        minors.append(v / np.linalg.norm(v))

    for r in range(12):
        labels.append(CHORD_LABELS[r])
    for r in range(12):
        labels.append(f"{CHORD_LABELS[r]}m")

    templates = np.stack(majors + minors, axis=0)
    return templates, labels


def _median_kernel_frames(sr: int, hop_length: int, window_sec: float, n_frames: int) -> int:
    """Odd kernel length in frames (~window_sec); capped to array length, at least 1."""
    k = max(3, int(round(window_sec * sr / float(hop_length))))
    if k % 2 == 0:
        k += 1
    k = min(k, max(1, n_frames))
    if k % 2 == 0:
        k -= 1
    return max(k, 1)


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


def _download_audio(youtube_url: str, output_wav_path: Path) -> None:
    # Use yt-dlp with ffmpeg to extract a WAV file.
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        str(output_wav_path.with_suffix(".%(ext)s")),
        youtube_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail=f"yt-dlp failed: {result.stderr.strip() or result.stdout.strip()}",
        )


def _estimate_chords(audio_path: Path) -> list[dict]:
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    y_harmonic, _y_perc = librosa.effects.hpss(y)
    total_duration = float(len(y)) / float(sr)
    hop_length = 512
    estimated_tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
    chroma = librosa.feature.chroma_cens(
        y=y_harmonic,
        sr=sr,
        hop_length=hop_length,
        tuning=estimated_tuning,
    )
    times = librosa.times_like(chroma, sr=sr, hop_length=hop_length)

    templates, chord_names = _major_minor_templates()
    # Unit-normalize each frame so dot product == cosine similarity to templates.
    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-12)
    # (24, T) similarity scores
    scores = templates @ chroma_norm
    best_idx = np.argmax(scores, axis=0).astype(np.int32)
    confidence = np.max(scores, axis=0)

    n_frames = int(best_idx.shape[0])
    if n_frames == 0:
        return []

    rms = librosa.feature.rms(y=y_harmonic, hop_length=hop_length)[0]
    rms = _align_series_to_n_frames(np.asarray(rms, dtype=np.float64), n_frames)
    rms_max = float(np.max(rms)) if rms.size else 0.0
    threshold = rms_max * 0.05
    low_energy = rms < threshold
    nc_id = len(chord_names)
    chord_names_nc = chord_names + [NC_CHORD_LABEL]
    best_idx = np.where(low_energy, np.int32(nc_id), best_idx)
    confidence = np.where(low_energy, np.float64(0.0), confidence)

    # ~1.25s median window suppresses split-second label noise (odd kernel, edge-safe mode).
    median_window_sec = 1.25
    kernel = _median_kernel_frames(sr, hop_length, median_window_sec, n_frames)
    smoothed_ids = median_filter(best_idx, size=kernel, mode="nearest").astype(np.int32)

    rle = _run_length_encode_chords(smoothed_ids, times, confidence, chord_names_nc)
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
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found for this video")
    return FileResponse(
        str(path),
        media_type="audio/wav",
        filename=f"{video_id}.wav",
    )


@app.post("/analyze")
def analyze_chords(payload: AnalyzeRequest):
    raw_input = payload.search_or_url.strip()
    youtube_watch_url, video_id = resolve_input_to_youtube(raw_input)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        wav_stem = tmp_path / "audio"
        wav_file = tmp_path / "audio.wav"

        _download_audio(youtube_watch_url, wav_stem)
        if not wav_file.exists():
            # Some ffmpeg/yt-dlp setups may produce a non-wav extension despite flags.
            candidates = list(tmp_path.glob("audio.*"))
            if not candidates:
                raise HTTPException(status_code=500, detail="Downloaded audio file not found")
            wav_file = candidates[0]

        analyze_path = wav_file
        AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = AUDIO_CACHE_DIR / f"{video_id}.wav"
        shutil.copy2(wav_file, cached)
        analyze_path = cached

        try:
            chords = _estimate_chords(analyze_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"librosa analysis failed: {exc}") from exc

    return {
        "source_url": raw_input,
        "video_id": video_id,
        "chords": chords,
    }


@app.get("/health")
def health():
    return {"ok": True}
