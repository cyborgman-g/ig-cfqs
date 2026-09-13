# Attribution-steered counterfactual audit for speaker verification

Code for *Beyond Saliency: Phonetically Disentangled Counterfactuals for Speaker Verification Interpretability*.

This repo does not ship results. Numbers differ across machines (float, TTS voice, GPU).

## Setup

```bash
pip install -r requirements.txt
```

Put official **test** audio under:

| Path | Corpus |
|---|---|
| `MIX_DATA/voxceleb1/` | VoxCeleb1 test (`idXXXXX/<video>/*.wav`) |
| `MIX_DATA/librispeech/` | LibriSpeech test-clean (and test-other if available) |
| `MIX_DATA/vctk/` | VCTK (`pXXX/*.wav`) |

Any nested `.wav` / `.flac` under those three names is picked up. Speaker id is the speaker folder name. See `MIX_DATA/README.txt`.

## Run

```bash
python main.py --config configs/paper.yaml --model ecapa
python main.py --config configs/paper.yaml --all
```

Models: `ecapa`, `resnet34`, `camplus`, `eres2net`, `wavlm`, `w2vbert`.

Weights download on first use into `pretrained/` (SpeechBrain, ModelScope, Hugging Face). Optional W2V-BERT speaker head: `pretrained/w2vbert/model_lmft_0.14.pth`.

Outputs:

- `outputs/<model>/results.csv`
- `outputs/<model>/summary.json`
- `outputs/<model>/ig_masks.pt`

## Config (`configs/paper.yaml`)

| Key | Role |
|---|---|
| `score_threshold: 0.0` | EER τ from a calibration split of `MIX_DATA` |
| `n_baselines: 30` | reference count for average-speaker / expected-gradients |
| `tau_m` | hard mask threshold on min-max \|IG\| |
| `baselines` / `strategies` / `directions` | what to run |

## Methods

**`src.cf_generator.IntegratedGradientsAttributor.compute`** — attribution map for `cosine(enroll, test)` under one IG baseline:

- `acoustic_silence`
- `average_speaker`
- `expected_gradients`
- `content_controlled` — Whisper ASR + Edge-TTS + DTW time-align

**`src.cf_generator.CounterfactualGenerator.generate`**

| Strategy | What it does |
|---|---|
| `masking` | suppress or boost bins with `M > tau_m` |
| `transplant` | copy enrollment bins where `M > tau_m` into the test utterance |
| `gradient_ascent` | AdamW step confined to `M > 0` until the decision flips or `max_iters` |
| `random_mask` | same density as the IG mask, random locations (control) |

**`src.metrics.flipped`** — true only if the original score was on one side of τ and the counterfactual crossed it.

**`src.metrics.l2` / `pesq_stoi`** — distortion on the CF representation; PESQ/STOI on reconstructed 16 kHz wav (iSTFT for waveform models; HiFi-GAN for fbank models when available).

**`src.metrics.summarize`** — success rate, mean L2, mean \|Δscore\|, PESQ, STOI, iters, grouped by baseline × strategy × direction.

**`src.metrics.mask_cosine` / `mask_cosine_matrix`** — cosine of binarized IG maps across models or baselines.
