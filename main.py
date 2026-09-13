"""Run attribution-steered counterfactual audits on MIX_DATA."""
import argparse
import json
import logging
import random
from pathlib import Path

import torch
import torchaudio
import yaml

from src.cf_generator import ContentControlledBaseline, CounterfactualGenerator, IntegratedGradientsAttributor
from src.data import calibrate_threshold, collect, load_input, load_wav, sample_trials
from src.metrics import summarize, write_csv
from src.models import CNN_HIFIGAN, MODEL_REGISTRY

log = logging.getLogger("audit")
ROOT = Path(__file__).resolve().parent


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def refs_for(model, itype, device, duration, paths, n):
    out = []
    for p in paths[:n]:
        try:
            out.append(load_input(model, p, itype, device, duration))
        except Exception:
            pass
    return out


def run(cfg):
    """Audit one model over MIX_DATA; write CSV and a JSON summary under out_dir."""
    name = cfg["model"]
    cls, itype = MODEL_REGISTRY[name]
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    duration = float(cfg.get("duration", 3))
    n_bl = int(cfg.get("n_baselines", 30))
    seed = int(cfg.get("seed", 0))
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = cls(device=device)
    spk_map = collect(cfg.get("data_root", "MIX_DATA"))
    if not spk_map:
        raise FileNotFoundError("No audio under MIX_DATA/{voxceleb1,librispeech,vctk}")
    n_tgt, n_imp = int(cfg.get("n_target", 50)), int(cfg.get("n_impostor", 50))
    n_cal = int(cfg.get("n_calibrate", 40))
    trials = sample_trials(spk_map, n_tgt + n_cal // 2, n_imp + n_cal // 2, rng)
    cal, use = trials[:n_cal], trials[n_cal:] or trials
    tau = cfg.get("score_threshold", 0.0)
    tau = calibrate_threshold(model, itype, device, duration, cal) if not tau else float(tau)
    ig = IntegratedGradientsAttributor(model, itype, n_steps=int(cfg.get("n_steps", 50)))
    cf = CounterfactualGenerator(
        model, itype, l2_budget=float(cfg.get("l2_budget", 5.0)), tau=tau,
        max_iters=int(cfg.get("max_iters", 300)), lr=float(cfg.get("lr", 0.005)),
        use_hifigan=bool(cfg.get("use_hifigan", name in CNN_HIFIGAN)),
        tau_m=float(cfg.get("tau_m", 0.5)),
    )
    all_files = [p for fs in spk_map.values() for p in fs]
    ref_pool = refs_for(model, itype, device, duration, rng.sample(all_files, min(n_bl, len(all_files))), n_bl)
    cc = ContentControlledBaseline(device=device) if "content_controlled" in cfg.get("baselines", []) else None
    baselines = cfg.get("baselines", list(IntegratedGradientsAttributor.STRATEGIES))
    strategies = cfg.get("strategies", ["masking", "transplant", "gradient_ascent", "random_mask"])
    directions = cfg.get("directions", ["accept_to_reject", "reject_to_accept"])
    rows, masks = [], {}
    out_dir = Path(cfg.get("out_dir", "outputs")) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (lab, enroll_p, test_p, tag) in enumerate(use):
        try:
            enroll = load_input(model, enroll_p, itype, device, duration)
            test = load_input(model, test_p, itype, device, duration)
            ref_wav = load_wav(test_p, duration) if itype == "waveform" else None
            for bl in baselines:
                refs = None
                if bl in ("average_speaker", "expected_gradients"):
                    refs = ref_pool
                elif bl == "content_controlled":
                    w = cc(test_p, duration, device, load_wav(test_p, duration)) if cc else None
                    if w is None:
                        continue
                    refs = [w.to(device) if itype == "waveform" else model.wav_to_feat(w)]
                attr, s0 = ig.compute(enroll, test, bl, refs)
                if i == 0:
                    masks[bl] = attr.cpu()
                res = cf.generate(test, enroll, attr, ref_wav, strategies, directions)
                for key, r in res.items():
                    st, d = key.split("__", 1)
                    rows.append({
                        "model": name, "baseline": bl, "strategy": st, "direction": d,
                        "pair": i, "same_speaker": bool(lab), "spk": tag, "tau": tau,
                        "score_before": r.score_before, "score_after": r.score_after,
                        "flipped": r.flipped, "l2": r.l2, "pesq": r.pesq, "stoi": r.stoi, "iters": r.iters,
                    })
                    if r.waveform is not None and cfg.get("save_wav"):
                        torchaudio.save(str(out_dir / f"p{i:03d}_{bl}_{key}.wav"), r.waveform.float(), 16000)
        except Exception as e:
            log.error("trial %s failed: %s", i, e)
    csv_path = out_dir / "results.csv"
    write_csv(csv_path, rows)
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if masks:
        torch.save(masks, out_dir / "ig_masks.pt")
    log.info("wrote %s (%d rows) tau=%.4f", csv_path, len(rows), tau)
    return rows, summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "paper.yaml"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    names = list(MODEL_REGISTRY) if args.all else [args.model or cfg.get("model", "ecapa")]
    for n in names:
        run({**cfg, "model": n, "out_dir": cfg.get("out_dir", "outputs")})
