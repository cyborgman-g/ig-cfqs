import random
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio

EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
CORPORA = ("voxceleb1", "librispeech", "vctk")
SKIP = {"wav", "wav48", "wav16", "flac", "test", "test-clean", "test-other", "audio", "wav48_silence_trimmed"}


def speaker_id(path):
    """Speaker folder from an official corpus path."""
    p = Path(path)
    for cand in (p.parent.parent.name, p.parent.name):
        if cand and cand not in SKIP:
            return cand
    return p.parent.name


def collect(root):
    """Return {corpus/speaker: [audio paths]} under MIX_DATA."""
    root = Path(root)
    spk_map = defaultdict(list)
    for corpus in CORPORA:
        croot = root / corpus
        if not croot.is_dir():
            continue
        for p in croot.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXT:
                spk_map[f"{corpus}/{speaker_id(p)}"].append(str(p))
    return dict(spk_map)


def load_wav(path, duration=3.0, sr=16000):
    """Load mono 16 kHz audio cropped or padded to duration seconds."""
    wav, s = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
    n = int(duration * sr)
    if wav.shape[1] < n:
        wav = torch.nn.functional.pad(wav, (0, n - wav.shape[1]))
    else:
        wav = wav[:, :n]
    return wav


def load_input(model, path, itype, device, duration=3.0):
    """Model input tensor from a file (waveform or features)."""
    wav = load_wav(path, duration).to(device)
    if itype == "waveform":
        return wav
    if hasattr(model, "wav_to_feat"):
        return model.wav_to_feat(wav)
    return wav


def sample_trials(spk_map, n_target, n_impostor, rng):
    """Balanced same-speaker and different-speaker (path, path, label) trials."""
    speakers = [s for s, fs in spk_map.items() if len(fs) >= 2]
    if len(speakers) < 2:
        speakers = list(spk_map)
    trials = []
    for _ in range(n_target):
        spk = rng.choice(speakers)
        files = spk_map[spk]
        if len(files) < 2:
            continue
        a, b = rng.sample(files, 2)
        trials.append((1, a, b, spk))
    keys = list(spk_map)
    for _ in range(n_impostor):
        sa, sb = rng.sample(keys, 2)
        trials.append((0, rng.choice(spk_map[sa]), rng.choice(spk_map[sb]), f"{sa}_vs_{sb}"))
    rng.shuffle(trials)
    return trials


def score_pair(model, itype, device, duration, p1, p2):
    """Cosine similarity of a trial pair."""
    e1 = model.forward(load_input(model, p1, itype, device, duration))
    e2 = model.forward(load_input(model, p2, itype, device, duration))
    e1, e2 = torch.nn.functional.normalize(e1.flatten(), dim=0), torch.nn.functional.normalize(e2.flatten(), dim=0)
    return float(torch.clamp(torch.dot(e1, e2), -1, 1))


def eer_threshold(scores, labels):
    """EER operating point on cosine scores (label 1 = target)."""
    import numpy as np
    from sklearn.metrics import roc_curve
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(thr[i])


def calibrate_threshold(model, itype, device, duration, trials):
    """EER threshold from labeled trials."""
    scores, labels = [], []
    for y, a, b, _ in trials:
        try:
            scores.append(score_pair(model, itype, device, duration, a, b))
            labels.append(y)
        except Exception:
            continue
    if len(scores) < 8:
        return 0.5
    return eer_threshold(scores, labels)
