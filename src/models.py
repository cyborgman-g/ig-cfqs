"""Six paper ASV wrappers."""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party"))


def _patch_torchaudio():
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: [""]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda _: None


def _dev(device):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


class FBank:
    """Kaldi 80-dim fbank used by CAM++ and E-Res2Net."""

    def __init__(self, n_mels=80, sample_rate=16000, mean_nor=True):
        self.n_mels, self.sr, self.mean_nor = n_mels, sample_rate, mean_nor

    def __call__(self, wav):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav[:1].cpu()
        feat = torchaudio.compliance.kaldi.fbank(
            wav, num_mel_bins=self.n_mels, sample_frequency=self.sr, dither=0.0
        )
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat


class _SBWav:
    def __init__(self, source, device=None):
        _patch_torchaudio()
        from speechbrain.inference.speaker import EncoderClassifier
        self.device = _dev(device)
        self._clf = EncoderClassifier.from_hparams(
            source=source, savedir=str(ROOT / "pretrained" / source.split("/")[-1]),
            run_opts={"device": str(self.device)},
        )
        self.model = self._clf.mods.embedding_model
        self.model.eval()

    def forward(self, x):
        feats = self._clf.mods.compute_features(x)
        feats = self._clf.mods.mean_var_norm(feats, torch.ones(feats.shape[0], device=self.device))
        return F.normalize(self.model(feats).squeeze(1), dim=-1)


class ECAPA_TDNN(_SBWav):
    """SpeechBrain ECAPA-TDNN (VoxCeleb)."""

    def __init__(self, device=None):
        super().__init__("speechbrain/spkrec-ecapa-voxceleb", device)


class ResNet34_SB(_SBWav):
    """SpeechBrain ResNet-34 (VoxCeleb)."""

    def __init__(self, device=None):
        super().__init__("speechbrain/spkrec-resnet-voxceleb", device)


def _ms(model_id, local_name, revision):
    from modelscope.hub.snapshot_download import snapshot_download
    dest = ROOT / "pretrained" / local_name
    if dest.exists():
        return dest
    return Path(snapshot_download(model_id, revision=revision, cache_dir=str(ROOT / "pretrained")))


class _FbankSV:
    def __init__(self, device=None):
        self.device = _dev(device)
        self._fbank = FBank()

    def wav_to_feat(self, wav):
        if isinstance(wav, (str, Path)):
            w, sr = torchaudio.load(str(wav))
            if sr != 16000:
                w = torchaudio.functional.resample(w, sr, 16000)
            wav = w[:1]
        feat = self._fbank(wav.to("cpu"))
        return feat.unsqueeze(0).to(self.device)

    def forward(self, x):
        return F.normalize(self.model(x), dim=-1)


class CAMPlus(_FbankSV):
    """CAM++ VoxCeleb (ModelScope English)."""

    def __init__(self, device=None):
        super().__init__(device)
        from speakerlab.models.campplus.DTDNN import CAMPPlus as _Net
        cache = _ms("iic/speech_campplus_sv_en_voxceleb_16k", "speech_campplus_sv_en_voxceleb_16k", "v1.0.2")
        ckpt = torch.load(cache / "campplus_voxceleb.bin", map_location="cpu")
        self.model = _Net(feat_dim=80, embedding_size=512)
        self.model.load_state_dict(ckpt)
        self.model.to(self.device).eval()


class ERes2Net_Large(_FbankSV):
    """E-Res2Net-Large (ModelScope English VoxCeleb)."""

    def __init__(self, device=None):
        super().__init__(device)
        from speakerlab.models.eres2net.ERes2Net import ERes2Net
        cache = _ms("iic/speech_eres2net_large_sv_en_voxceleb_16k", "speech_eres2net_large_sv_en_voxceleb_16k", "v1.0.0")
        ckpt = torch.load(next(cache.glob("*.ckpt")), map_location="cpu")
        for k in ("state_dict", "model"):
            if isinstance(ckpt, dict) and k in ckpt:
                ckpt = ckpt[k]
        self.model = ERes2Net(feat_dim=80, embedding_size=512, m_channels=64)
        self.model.load_state_dict(ckpt, strict=False)
        self.model.to(self.device).eval()


class WavLM_BasePlus_SV:
    """WavLM-Base+ speaker verification."""

    def __init__(self, device=None, model_name="microsoft/wavlm-base-plus-sv"):
        from transformers import WavLMForXVector
        self.device = _dev(device)
        cache = ROOT / "pretrained" / "wavlm-base-plus-sv"
        self.model = WavLMForXVector.from_pretrained(model_name, cache_dir=str(cache)).to(self.device).eval()

    def forward(self, x):
        return F.normalize(self.model(input_values=x).embeddings, dim=-1)


class W2VBertSV:
    """W2V-BERT 2.0 speaker model (filterbank features)."""

    def __init__(self, device=None, ckpt_path=None):
        from transformers import Wav2Vec2BertModel, AutoConfig, AutoFeatureExtractor
        self.device = _dev(device)
        cache = str(ROOT / "pretrained" / "w2vbert")
        cfg = AutoConfig.from_pretrained("facebook/w2v-bert-2.0", cache_dir=cache)
        self._processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0", cache_dir=cache)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.front = Wav2Vec2BertModel(cfg)
                self.adapter_layers = nn.ModuleList([
                    nn.Sequential(nn.Linear(1024, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 128))
                    for _ in range(25)
                ])
                self.attention = nn.Sequential(nn.Conv1d(3200, 128, 1), nn.ReLU(), nn.BatchNorm1d(128), nn.Conv1d(128, 3200, 1))
                self.bottleneck = nn.Linear(6400, 256)

            def forward(self, input_features, attention_mask=None):
                hs = self.front(input_features=input_features, attention_mask=attention_mask, output_hidden_states=True).hidden_states
                x = torch.cat([adp(h) for adp, h in zip(self.adapter_layers, hs)], dim=-1)
                x = x.transpose(1, 2)
                w = torch.softmax(self.attention(x), dim=-1)
                mean = (w * x).sum(-1)
                std = torch.clamp((w * x ** 2).sum(-1) - mean ** 2, 1e-9).sqrt()
                return F.normalize(self.bottleneck(torch.cat([mean, std], dim=-1)), dim=-1)

        self.model = Net()
        path = Path(ckpt_path) if ckpt_path else ROOT / "pretrained" / "w2vbert" / "model_lmft_0.14.pth"
        if path.is_file():
            sd = torch.load(path, map_location="cpu")
            sd = sd["modules"]["spk_model"] if isinstance(sd, dict) and "modules" in sd else sd
            self.model.load_state_dict(sd, strict=False)
        else:
            self.model.front = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", cache_dir=str(ROOT / "pretrained" / "w2vbert"))
        self.model.to(self.device).eval()

    def wav_to_feat(self, wav):
        if isinstance(wav, (str, Path)):
            w, sr = torchaudio.load(str(wav))
            if sr != 16000:
                w = torchaudio.functional.resample(w, sr, 16000)
            wav = w
        arr = wav.mean(0).detach().cpu().numpy()
        return self._processor(arr, sampling_rate=16000, return_tensors="pt")["input_features"].to(self.device)

    def forward(self, x):
        return self.model(input_features=x)


MODEL_REGISTRY = {
    "ecapa": (ECAPA_TDNN, "waveform"),
    "resnet34": (ResNet34_SB, "waveform"),
    "camplus": (CAMPlus, "melspec"),
    "eres2net": (ERes2Net_Large, "melspec"),
    "wavlm": (WavLM_BasePlus_SV, "waveform"),
    "w2vbert": (W2VBertSV, "melspec"),
}

CNN_HIFIGAN = {"ecapa", "resnet34", "camplus", "eres2net"}
