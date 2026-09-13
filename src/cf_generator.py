import asyncio
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients

from src.metrics import flipped, l2, pesq_stoi

ROOT = Path(__file__).resolve().parent.parent


def _len(x, n):
    x = x[..., :n]
    return F.pad(x, (0, n - x.shape[-1])) if x.shape[-1] < n else x


def dtw_align(src, tgt):
    """Warp src onto tgt's duration with DTW on log-mel frames."""
    import librosa
    n = tgt.shape[-1]
    a = src.squeeze().detach().cpu().numpy()
    b = tgt.squeeze().detach().cpu().numpy()
    fa = librosa.feature.melspectrogram(y=a, sr=16000, n_mels=40)
    fb = librosa.feature.melspectrogram(y=b, sr=16000, n_mels=40)
    _, wp = librosa.sequence.dtw(X=fa, Y=fb, metric="euclidean")
    wp = np.asarray(wp)[::-1]
    src_t = wp[:, 0] * (len(a) / max(fa.shape[1], 1))
    tgt_t = wp[:, 1] * (len(b) / max(fb.shape[1], 1))
    nt = np.clip(np.interp(np.arange(len(b)), tgt_t, src_t), 0, len(a) - 1)
    y = np.interp(nt, np.arange(len(a)), a)
    out = torch.tensor(y, dtype=src.dtype, device=src.device).unsqueeze(0)
    return _len(out, n)


def align_any(src, tgt):
    """DTW waveforms; linear time-align feature maps."""
    if src.shape == tgt.shape:
        return src
    if src.dim() == 2:
        try:
            return dtw_align(src, tgt)
        except Exception:
            return _len(src, tgt.shape[-1])
    t = tgt.shape[1]
    return F.interpolate(src.permute(0, 2, 1), size=t, mode="linear", align_corners=False).permute(0, 2, 1)


def to01(attr):
    a = attr.abs()
    return (a - a.min()) / (a.max() - a.min() + 1e-9)


def hard_mask(attr, tau_m=0.5):
    """M in [0, 1] then bins with M > tau_m."""
    return (to01(attr) > tau_m).to(attr.dtype)


def random_mask_like(mask):
    """Same density as mask, random locations."""
    p = float(mask.mean().clamp(1e-4, 1 - 1e-4))
    return (torch.rand_like(mask) < p).to(mask.dtype)


class IntegratedGradientsAttributor:
    STRATEGIES = ("acoustic_silence", "average_speaker", "expected_gradients", "content_controlled")

    def __init__(self, model, itype="waveform", n_steps=50):
        self.model, self.itype, self.n_steps = model, itype, n_steps

    def _score_fn(self, enroll_emb):
        return lambda x: F.cosine_similarity(self.model.forward(x), enroll_emb)

    def _silence(self, test):
        if self.itype == "waveform":
            return torch.zeros_like(test)
        return torch.zeros_like(test)

    def compute(self, enroll, test, strategy, refs=None):
        """Return (attribution, cosine score). refs are baseline tensors."""
        enroll_emb = self.model.forward(enroll).detach()
        fn = self._score_fn(enroll_emb)
        test_g = test.detach().clone().requires_grad_(True)
        ig = IntegratedGradients(fn)
        if strategy == "acoustic_silence":
            attr = ig.attribute(test_g, baselines=self._silence(test), n_steps=self.n_steps)
        elif strategy == "average_speaker":
            aligned = [align_any(r.to(test.device), test) for r in (refs or [])]
            bl = torch.stack(aligned).mean(0) if aligned else self._silence(test)
            attr = ig.attribute(test_g, baselines=bl, n_steps=self.n_steps)
        elif strategy == "expected_gradients":
            acc = None
            pool = list(refs or [])
            for r in pool:
                bl = align_any(r.to(test.device), test)
                a = ig.attribute(test_g, baselines=bl, n_steps=self.n_steps)
                acc = a if acc is None else acc + a
            attr = acc / max(len(pool), 1) if acc is not None else ig.attribute(
                test_g, baselines=self._silence(test), n_steps=self.n_steps
            )
        elif strategy == "content_controlled":
            bl = align_any(refs[0].to(test.device), test) if refs else self._silence(test)
            attr = ig.attribute(test_g, baselines=bl, n_steps=self.n_steps)
        else:
            raise ValueError(strategy)
        with torch.no_grad():
            score = fn(test).item()
        return attr.detach(), score


@dataclass
class CFResult:
    strategy: str
    direction: str
    score_before: float
    score_after: float
    flipped: bool
    l2: float
    waveform: Optional[torch.Tensor] = None
    pesq: Optional[float] = None
    stoi: Optional[float] = None
    iters: Optional[int] = None


class CounterfactualGenerator:
    """Salience masking, identity transplant, masked gradient ascent, random-mask control."""

    def __init__(self, model, itype="waveform", sr=16000, l2_budget=5.0, tau=0.5,
                 max_iters=300, lr=0.005, n_fft=512, hop=128, use_hifigan=False, tau_m=0.5):
        self.m, self.device = model, model.device
        self.itype, self.sr = itype, sr
        self.l2_budget, self.tau = l2_budget, tau
        self.max_iters, self.lr = max_iters, lr
        self.n_fft, self.hop, self.tau_m = n_fft, hop, tau_m
        self._hifigan = None
        self._use_hifigan = use_hifigan and itype != "waveform"

    def _hifi(self):
        if self._hifigan is not None or not self._use_hifigan:
            return self._hifigan
        try:
            from speechbrain.inference.vocoder import HIFIGAN
            self._hifigan = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-ljspeech",
                savedir=str(ROOT / "pretrained" / "hifigan"),
                run_opts={"device": str(self.device)},
            )
        except Exception:
            try:
                from speechbrain.pretrained import HIFIGAN
                self._hifigan = HIFIGAN.from_hparams(
                    source="speechbrain/tts-hifigan-ljspeech",
                    savedir=str(ROOT / "pretrained" / "hifigan"),
                    run_opts={"device": str(self.device)},
                )
            except Exception:
                self._hifigan = None
        return self._hifigan

    def _score(self, feat, enroll_emb):
        with torch.no_grad():
            return F.cosine_similarity(self.m.forward(feat), enroll_emb).item()

    def _clamp(self, orig, pert):
        d = pert - orig
        n = d.norm()
        return orig + d * (self.l2_budget / (n + 1e-9)) if n > self.l2_budget else pert

    def _stft(self, x):
        win = torch.hann_window(self.n_fft, device=x.device)
        S = torch.stft(x.squeeze(), n_fft=self.n_fft, hop_length=self.hop, window=win, return_complex=True)
        return S.abs(), S.angle(), win

    def _istft(self, mag, phase, win):
        y = torch.istft(torch.polar(mag, phase).unsqueeze(0), n_fft=self.n_fft, hop_length=self.hop, window=win)
        return y if y.dim() == 2 else y.unsqueeze(0)

    def _to_wav(self, feat):
        if self.itype == "waveform":
            return feat.detach().cpu()
        h = self._hifi()
        if h is None:
            return None
        try:
            return h.decode_batch(feat.detach().squeeze(0).T.unsqueeze(0)).squeeze().unsqueeze(0).cpu()
        except Exception:
            return None

    def _pack(self, strat, direction, s0, s1, feat, test, wav, ref_wav, iters=None):
        wav = wav if wav is not None else self._to_wav(feat)
        p, s = pesq_stoi(ref_wav, wav, self.sr)
        return CFResult(
            strat, direction, s0, s1, flipped(s0, s1, self.tau, direction),
            l2(feat, test), wav, p, s, iters,
        )

    def _apply_spec(self, test, mag_fn):
        mag, phase, win = self._stft(test)
        mag_cf = mag_fn(mag)
        wav = self._istft(mag_cf, phase, win).to(self.device)
        return self._clamp(test, wav)

    def _masking(self, test, enroll_emb, attr, direction, ref_wav, mask=None, name="masking"):
        s0 = self._score(test, enroll_emb)
        sup = direction == "accept_to_reject"
        if self.itype == "waveform":
            m = mask if mask is not None else hard_mask(self._stft(attr)[0], self.tau_m)
            feat = self._apply_spec(test, lambda mag: mag * (1 - m) if sup else mag * (1 + m))
            wav = feat.detach().cpu()
        else:
            m = mask if mask is not None else hard_mask(attr, self.tau_m)
            feat = self._clamp(test, test * (1 - m) if sup else test * (1 + m))
            wav = self._to_wav(feat)
        return self._pack(name, direction, s0, self._score(feat, enroll_emb), feat, test, wav, ref_wav)

    def _transplant(self, test, enroll, enroll_emb, attr, direction, ref_wav):
        s0 = self._score(test, enroll_emb)
        if self.itype == "waveform":
            mag_t, phase_t, win = self._stft(test)
            m = hard_mask(self._stft(attr)[0], self.tau_m)
            mag_e = self._stft(enroll)[0]
            if mag_e.shape != mag_t.shape:
                mag_e = F.interpolate(mag_e.unsqueeze(0).unsqueeze(0), size=mag_t.shape, mode="bilinear", align_corners=False).squeeze()
            mag_cf = torch.where(m.bool(), mag_e, mag_t)
            wav = self._istft(mag_cf, phase_t, win).to(self.device)
            feat = self._clamp(test, wav)
            wav = feat.detach().cpu()
        else:
            m = hard_mask(attr, self.tau_m)
            enroll_al = align_any(enroll, test)
            feat = self._clamp(test, torch.where(m.bool(), enroll_al, test))
            wav = self._to_wav(feat)
        return self._pack("transplant", direction, s0, self._score(feat, enroll_emb), feat, test, wav, ref_wav)

    def _ga(self, test, enroll_emb, attr, direction, ref_wav):
        s0 = self._score(test, enroll_emb)
        m = (to01(attr) > 0).to(test.dtype)
        delta = torch.zeros_like(test, requires_grad=True)
        opt = torch.optim.AdamW([delta], lr=self.lr)
        sign = 1.0 if direction == "accept_to_reject" else -1.0
        iters = 0
        for i in range(self.max_iters):
            opt.zero_grad()
            emb = F.normalize(self.m.forward(test + m * delta), dim=-1)
            (sign * F.cosine_similarity(emb, enroll_emb).mean()).backward()
            opt.step()
            with torch.no_grad():
                n = delta.norm()
                if n > self.l2_budget:
                    delta.data *= self.l2_budget / (n + 1e-9)
                cs = F.cosine_similarity(F.normalize(self.m.forward(test + m * delta), dim=-1), enroll_emb).item()
            iters = i + 1
            if flipped(s0, cs, self.tau, direction):
                break
        with torch.no_grad():
            feat = self._clamp(test, test + m * delta)
        return self._pack("gradient_ascent", direction, s0, self._score(feat, enroll_emb), feat, test, None, ref_wav, iters)

    def generate(self, test, enroll, attr, ref_wav=None,
                 strategies=("masking", "transplant", "gradient_ascent", "random_mask"),
                 directions=("accept_to_reject", "reject_to_accept")):
        """Run requested strategies and directions; keys are strategy__direction."""
        enroll_emb = self.m.forward(enroll.to(self.device)).detach()
        test, attr = test.to(self.device), attr.to(self.device)
        out = {}
        for d in directions:
            for s in strategies:
                k = f"{s}__{d}"
                try:
                    if s == "masking":
                        out[k] = self._masking(test, enroll_emb, attr, d, ref_wav)
                    elif s == "transplant":
                        out[k] = self._transplant(test, enroll.to(self.device), enroll_emb, attr, d, ref_wav)
                    elif s == "gradient_ascent":
                        out[k] = self._ga(test, enroll_emb, attr, d, ref_wav)
                    elif s == "random_mask":
                        rm = random_mask_like(hard_mask(attr if self.itype != "waveform" else self._stft(attr)[0], self.tau_m))
                        out[k] = self._masking(test, enroll_emb, attr, d, ref_wav, mask=rm, name="random_mask")
                except Exception:
                    continue
        return out


class ContentControlledBaseline:
    """Whisper transcription + Edge-TTS + DTW alignment (same text, other voice)."""

    def __init__(self, sr=16000, device="cpu"):
        self.sr, self.device, self._asr = sr, device, None

    def _load(self):
        if self._asr is not None:
            return True
        try:
            import whisper
            self._asr = whisper.load_model(
                "base", device=self.device, download_root=str(ROOT / "pretrained" / "whisper"),
            )
            return True
        except Exception:
            return False

    def __call__(self, test_path, duration, device=None, test_wav=None):
        """Return DTW-aligned TTS waveform or None."""
        if not self._load():
            return None
        try:
            import edge_tts
            from src.data import load_wav
            wav = test_wav if test_wav is not None else load_wav(test_path, duration, self.sr)
            text = (self._asr.transcribe(wav.squeeze().numpy(), language="en", fp16=False).get("text") or "").strip()
            if not text:
                return None
            voice = random.choice(["en-US-GuyNeural", "en-GB-RyanNeural", "en-US-JennyNeural"])

            async def _synth():
                buf = io.BytesIO()
                async for chunk in edge_tts.Communicate(text, voice).stream():
                    if chunk["type"] == "audio":
                        buf.write(chunk["data"])
                return buf.getvalue()

            try:
                data = asyncio.run(_synth())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                data = loop.run_until_complete(_synth())
                loop.close()
            import torchaudio
            out, osr = torchaudio.load(io.BytesIO(data))
            if osr != self.sr:
                out = torchaudio.functional.resample(out, osr, self.sr)
            out = out.mean(0, keepdim=True)
            return dtw_align(out.to(wav.device), wav)
        except Exception:
            return None
