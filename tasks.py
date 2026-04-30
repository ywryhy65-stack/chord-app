import os
import json
import shutil
import tempfile
import traceback
from pathlib import Path
from yt_dlp import YoutubeDL
import librosa
import numpy as np
from scipy.ndimage import median_filter
from celery_app import celery_app

# --- Re-importing constants and helpers from main.py logic ---
# (In a real production app, these should be in a shared utils.py)

CHORD_ROOTS = [
    "C", "C#", "D", "Eb", "E", "F",
    "F#", "G", "Ab", "A", "Bb", "B",
]
MIN_DURATION_SEC = 0.67
NC_CHORD_LABEL = "N.C."
AUDIO_CACHE_DIR = Path(__file__).resolve().parent / "audio_cache"

def _major_minor_templates() -> tuple[np.ndarray, list[str]]:
    templates: list[np.ndarray] = []
    labels: list[str] = []
    def add_template(root, intervals, label_suffix=""):
        v = np.zeros(12, dtype=np.float64)
        for semi in intervals:
            v[(root + semi) % 12] += 1.0
        v[(root + 7) % 12] += 0.3
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        templates.append(v)
        labels.append(f"{CHORD_ROOTS[root]}{label_suffix}")
    for r in range(12):
        add_template(r, [0, 4, 7], "")
        add_template(r, [0, 3, 7], "m")
    return np.array(templates), labels

def _generate_transition_matrix(n_states: int, self_prob: float = 0.98) -> np.ndarray:
    trans = np.full((n_states, n_states), (1.0 - self_prob) / (n_states - 1))
    np.fill_diagonal(trans, self_prob)
    return trans

def _separate_stems(audio_path: Path) -> Path:
    try:
        if shutil.which("spleeter") is None:
            return audio_path
        output_dir = audio_path.parent / "spleeter_output"
        acc_path = output_dir / audio_path.stem / "accompaniment.wav"
        if not acc_path.exists():
            import subprocess
            cmd = ["spleeter", "separate", "-p", "spleeter:2stems", "-o", str(output_dir), str(audio_path)]
            subprocess.run(cmd, check=True, capture_output=True)
        if acc_path.exists():
            return acc_path
    except Exception as e:
        print(f"DEBUG: Spleeter failed: {e}. Falling back to original audio.")
    return audio_path

def _run_length_encode_chords(chord_ids: np.ndarray, times: np.ndarray, confidences: np.ndarray, chord_names: list[str]) -> list[dict]:
    n = int(chord_ids.shape[0])
    if n == 0:
        return []
    out: list[dict] = []
    run_start = 0
    for i in range(1, n):
        if int(chord_ids[i]) != int(chord_ids[run_start]):
            cid = int(chord_ids[run_start])
            out.append({"time": round(float(times[run_start]), 3), "chord": chord_names[cid], "confidence": float(round(float(confidences[run_start]), 4))})
            run_start = i
    cid = int(chord_ids[run_start])
    out.append({"time": round(float(times[run_start]), 3), "chord": chord_names[cid], "confidence": float(round(float(confidences[run_start]), 4))})
    return out

def _merge_short_transients(segments: list[dict], total_duration: float, min_duration_sec: float = MIN_DURATION_SEC) -> list[dict]:
    if not segments: return []
    n = len(segments)
    merged: list[dict] = []
    i = 0
    absorbed_start = None
    while i < n:
        t_start = float(segments[i]["time"]) if absorbed_start is None else absorbed_start
        t_end = float(segments[i + 1]["time"]) if i + 1 < n else float(total_duration)
        duration = t_end - t_start
        if duration >= min_duration_sec - 1e-9:
            merged.append({"time": round(t_start, 3), "chord": segments[i]["chord"], "confidence": segments[i]["confidence"]})
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
        best_j = max(range(n), key=lambda j: (float(segments[j + 1]["time"]) if j + 1 < n else float(total_duration)) - float(segments[j]["time"]))
        s = segments[best_j]
        merged.append({"time": round(float(s["time"]), 3), "chord": s["chord"], "confidence": s["confidence"]})
    return merged

def _estimate_chords(audio_path: Path) -> list[dict]:
    harmonic_audio_path = _separate_stems(audio_path)
    y, sr = librosa.load(str(harmonic_audio_path), sr=22050, mono=True)
    total_duration = float(len(y)) / float(sr)
    y_harmonic, _ = librosa.effects.hpss(y, margin=(2.0, 5.0))
    hop_length = 2048
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length, n_octaves=5, fmin=librosa.note_to_hz('C3'))
    chroma = librosa.util.normalize(chroma, axis=0)
    global_chroma = np.sum(chroma, axis=1)
    templates, chord_names = _major_minor_templates()
    nc_template = np.zeros(12)
    templates_with_nc = np.vstack([templates, nc_template])
    chord_names_with_nc = chord_names + [NC_CHORD_LABEL]
    key_scores = np.dot(templates, global_chroma)
    best_key_idx = np.argmax(key_scores)
    is_minor_key = (best_key_idx % 2 == 1)
    key_root = best_key_idx // 2
    if is_minor_key:
        diatonic_chords = [(0, "m"), (3, ""), (5, "m"), (7, "m"), (8, ""), (10, "")]
    else:
        diatonic_chords = [(0, ""), (2, "m"), (4, "m"), (5, ""), (7, ""), (9, "m")]
    diatonic_indices = []
    for root_rel, suffix in diatonic_chords:
        root_abs = (key_root + root_rel) % 12
        idx = root_abs * 2 if suffix == "" else root_abs * 2 + 1
        diatonic_indices.append(idx)
    nc_idx = len(chord_names_with_nc) - 1
    mask = np.full(len(chord_names_with_nc), 0.01)
    mask[diatonic_indices] = 1.0
    mask[nc_idx] = 0.5
    scores = np.dot(templates_with_nc, chroma)
    prob_matrix = np.exp(scores * 10.0)
    prob_matrix *= mask[:, np.newaxis]
    prob_matrix /= (np.sum(prob_matrix, axis=0) + 1e-12)
    transition_matrix = _generate_transition_matrix(len(chord_names_with_nc), self_prob=0.98)
    final_ids = librosa.sequence.viterbi(prob_matrix, transition_matrix)
    confidence = np.max(prob_matrix, axis=0)
    times = librosa.frames_to_time(np.arange(len(final_ids)), sr=sr, hop_length=hop_length)
    rle = _run_length_encode_chords(final_ids, times, confidence, chord_names_with_nc)
    return _merge_short_transients(rle, total_duration)

@celery_app.task(bind=True)
def analyze_chords_task(self, youtube_watch_url, video_id, raw_input):
    # This task handles the heavy lifting
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_json = AUDIO_CACHE_DIR / f"{video_id}.json"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        wav_stem = tmp_path / "audio"
        wav_file = tmp_path / "audio.wav"

        # Fetch metadata
        try:
            ydl_opts = {
                "quiet": True,
                "socket_timeout": 30,
                "source_address": "0.0.0.0",
                "retries": 5,
                "extractor_retries": 3,
                "remote_components": ["ejs:github"],
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_watch_url, download=False)
                title = info.get("title", "Unknown Title")
                artist = info.get("artist")
                track = info.get("track")
                video_duration = info.get("duration")
        except Exception:
            title = "Unknown Title"
            artist = track = video_duration = None

        # Download audio (re-using main.py logic)
        import subprocess
        cmd = ["yt-dlp", "--extract-audio", "--audio-format", "wav", "--audio-quality", "0", youtube_watch_url, "-o", str(wav_stem) + ".%(ext)s"]
        subprocess.run(cmd, check=True, capture_output=True)
        
        if not wav_file.exists():
            candidates = list(tmp_path.glob("audio.*"))
            if candidates: wav_file = candidates[0]
            else: raise Exception("Download failed")

        # Chord estimation
        chords = _estimate_chords(wav_file)

        # We can't easily import _fetch_synced_lyrics from main due to circularity
        # For now, we skip lyrics or re-implement if critical. 
        # (Recommendation: move all core logic to a shared file)
        synced_lyrics = None 

        result = {
            "title": title,
            "source_url": raw_input,
            "video_id": video_id,
            "chords": chords,
            "lyrics": synced_lyrics,
        }

        # Cache it
        cached_wav = AUDIO_CACHE_DIR / f"{video_id}.wav"
        shutil.copy2(wav_file, cached_wav)
        with open(cache_json, "w") as f:
            json.dump(result, f)

    return result
