"""Run zjt's trained FlowCast (VAE-latent conditional-flow model) on 10 test
samples and save the forecasts + VAE reconstruction + persistence to dcm's own
folder, for a blur comparison against the pixel-space EDM baseline.

This script does NOT modify any of zjt's code. It imports zjt's model loaders
from zjt/cira-diff/flowcast_v2/datasets/goes/{runtime,normalization}.py and the
`forecast`/`decode` helpers below are copied verbatim from zjt's
    zjt/cira-diff/flowcast_v2/scripts/evaluate.py
so the sampling is byte-faithful to zjt's evaluation protocol.

Outputs (all Kelvin, float32), test order:
    flowcast_forecasts.npy  [sample, lead, member, H, W] = (10, 18, 8, 256, 256)
    vae_recon.npy           [sample, lead, H, W]         = (10, 18, 256, 256)
    persistence.npy         [sample, lead, H, W]
    truth.npy               [sample, lead, H, W]
    input.npy               [sample, 2, H, W]
    meta.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import zarr
from tqdm import tqdm

ZJT = Path("/home/group1/26fall_aiclass/zjt/cira-diff/flowcast_v2")
sys.path.insert(0, str(ZJT))

from datasets.goes.runtime import load_vae, load_flowcast, read_config, sha256  # noqa: E402
from datasets.goes.normalization import to_kelvin, normalize_to_neg_one_one, denormalize_to_kelvin  # noqa: E402


# ---------------------------------------------------------------------------
# Verbatim from zjt/.../scripts/evaluate.py
# ---------------------------------------------------------------------------
def decode(vae, latent, norm, frame_batch=8):
    b, t, h, w, c = latent.shape
    frames = latent.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    decoded = torch.cat([vae.decode(chunk).sample for chunk in frames.split(frame_batch)])
    return denormalize_to_kelvin(decoded[:, 0], norm).reshape(b, t, *decoded.shape[-2:])


def forecast(model, condition, target_shape, samples, steps, horizon):
    # Roll the model out to the full target horizon; horizon==1 for multi-step models.
    condition = model.normalize(condition)
    for _ in range(samples):
        cond, states = condition, []
        for _ in range(horizon):
            state = torch.randn(target_shape, device=cond.device)
            for step in range(steps):
                time = torch.full((len(state),), step / steps, device=state.device)
                state = state + model(time, state, cond) / steps
            states.append(model.denormalize(state))
            # Feed the prediction back, keeping the lag window fixed at lag_time frames.
            cond = torch.cat([cond, state], dim=1)[:, -condition.shape[1]:]
        yield torch.cat(states, dim=1)


def latent_shape(latent, key):
    _, t, c, h, w = latent[key].shape
    return (t, h, w, c)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vae_checkpoint", default=str(ZJT / "runs/vae/vae_1005_150523/models/early_stopping_model.pt"))
    p.add_argument("--vae_config", default=str(ZJT / "runs/vae/vae_1005_150523/config_snapshot.yaml"))
    p.add_argument("--cfm_checkpoint", default=str(ZJT / "runs/cfm/cfm_1007_010810/models/early_stopping_model.pt"))
    p.add_argument("--cfm_config", default=str(ZJT / "runs/cfm/cfm_1007_010810/config_snapshot.yaml"))
    p.add_argument("--latent_path", default=str(ZJT / "runs/latent/test.zarr"))
    p.add_argument("--data", default=str(ZJT.parent / "datasets/satcast/edm_GOES_ch13_test_dataset.zarr"))
    p.add_argument("--output_dir", default="runs/evaluation/flowcast_compare")
    p.add_argument("--max_samples", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--samples", type=int, default=8, help="ensemble members")
    p.add_argument("--steps", type=int, default=10, help="CFM Euler steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    vae, norm = load_vae(args.vae_checkpoint, args.vae_config, device)
    raw = zarr.open(args.data, mode="r")
    latent = zarr.open(args.latent_path, mode="r")
    # Provenance guard (same checks zjt's evaluate.py enforces).
    assert latent.attrs.get("complete") is True and latent.attrs.get("encoding") == "mode"
    assert latent.attrs.get("vae_sha256") == sha256(args.vae_checkpoint)
    assert latent.attrs.get("normalization") == norm
    assert Path(latent.attrs["source_path"]).resolve() == Path(args.data).resolve()

    model = load_flowcast(args.cfm_checkpoint, args.cfm_config,
                          latent_shape(latent, "input_images"),
                          latent_shape(latent, "output_images"), device)

    n_total = raw["input_images"].shape[0]
    count = min(args.max_samples, n_total)
    leads = raw["output_images"].shape[1]
    assert leads == 18, f"expected 18 lead frames, got {leads}"
    h, w = raw["output_images"].shape[-2:]

    truth = np.zeros((count, leads, h, w), dtype=np.float32)
    persistence = np.zeros_like(truth)
    vae_recon = np.zeros_like(truth)
    input_ = np.zeros((count, 2, h, w), dtype=np.float32)
    flowcast = np.zeros((count, leads, args.samples, h, w), dtype=np.float32)

    with torch.inference_mode():
        for start in tqdm(range(0, count, args.batch_size), desc="batch"):
            stop = min(start + args.batch_size, count)
            x = np.asarray(raw["input_images"][start:stop], dtype=np.float32)
            y = np.asarray(raw["output_images"][start:stop], dtype=np.float32)
            b = y.shape[0]

            truth[start:stop] = to_kelvin(y, norm)
            persistence[start:stop] = np.repeat(to_kelvin(x[:, -1:], norm), leads, axis=1)
            input_[start:stop] = x

            # Deterministic VAE reconstruction (encode mode -> decode): the VAE's
            # own detail ceiling, before any latent-space generative model.
            frames = torch.from_numpy(normalize_to_neg_one_one(y, norm)).reshape(-1, 1, h, w).to(device)
            rec = torch.cat([vae.decode(vae.encode(chunk).latent_dist.mode()).sample
                             for chunk in frames.split(8)])
            vae_recon[start:stop] = denormalize_to_kelvin(rec[:, 0], norm).reshape(b, leads, h, w).cpu().numpy()

            # FlowCast autoregressive rollout in latent space.
            condition = torch.from_numpy(np.asarray(latent["input_images"][start:stop])).permute(0, 1, 3, 4, 2).to(device)
            target_shape = (b, *latent_shape(latent, "output_images"))
            horizon = -(-leads // latent_shape(latent, "output_images")[0])
            preds = np.stack([decode(vae, z, norm).cpu().numpy()
                              for z in forecast(model, condition, target_shape, args.samples, args.steps, horizon)])
            preds = preds[:, :, :leads]                      # [member, B, lead, H, W]
            flowcast[start:stop] = preds.transpose(1, 2, 0, 3, 4)  # [B, lead, member, H, W]

    # Quick sanity check against zjt's reported aggregate (~3.93 K MAE).
    fc_mean = flowcast.mean(axis=2)
    mae_fc = float(np.abs(fc_mean - truth).mean())
    mae_pers = float(np.abs(persistence - truth).mean())
    mae_vae = float(np.abs(vae_recon - truth).mean())
    print(f"sanity MAE (K): flowcast={mae_fc:.4f}  vae_recon={mae_vae:.4f}  persistence={mae_pers:.4f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "flowcast_forecasts.npy", flowcast)
    np.save(out / "vae_recon.npy", vae_recon)
    np.save(out / "persistence.npy", persistence)
    np.save(out / "truth.npy", truth)
    np.save(out / "input.npy", input_)
    meta = dict(
        vae_checkpoint=str(Path(args.vae_checkpoint).resolve()),
        cfm_checkpoint=str(Path(args.cfm_checkpoint).resolve()),
        vae_config=str(Path(args.vae_config).resolve()),
        cfm_config=str(Path(args.cfm_config).resolve()),
        latent_path=str(Path(args.latent_path).resolve()),
        data=str(Path(args.data).resolve()),
        vae_sha256=sha256(args.vae_checkpoint),
        cfm_sha256=sha256(args.cfm_checkpoint),
        normalization=norm,
        samples_evaluated=count,
        ensemble_members=args.samples,
        euler_steps=args.steps,
        seed=args.seed,
        batch_size=args.batch_size,
        lead_frames=leads,
        lead_spacing_minutes=10,
        sanity_mae_k=dict(flowcast=mae_fc, vae_recon=mae_vae, persistence=mae_pers),
        note="flowcast_forecasts.npy shape [sample, lead, member, H, W]; all arrays Kelvin float32",
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}/flowcast_forecasts.npy {flowcast.shape} + vae_recon/persistence/truth/input + meta.json")


if __name__ == "__main__":
    main()
