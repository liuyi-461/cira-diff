"""生成全量数据的 VAE 重建图：把 18 帧 truth 逐帧 encode→decode，反归一化成 Kelvin。

与 zjt 的 evaluate.py 中 "vae" 路径逐字一致：
    normalize_to_neg_one_one -> vae.encode().latent_dist.mode() -> vae.decode().sample
    -> denormalize_to_kelvin
输出 vae_recon.npy (1024,18,256,256) float32 Kelvin，即 VAE 的"细节上限"参考。

运行（flowcast-goes 环境，单卡约 8~12 分钟）：
    /home/group1/miniconda3/envs/flowcast-goes/bin/python code/make_vae_recon.py --device cuda:3
"""
import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import zarr

import os

# 根目录可用环境变量覆盖；默认值是共享旧机 test-ai 上的位置
SATCAST = os.environ.get("SATCAST_DIR", "/data1/satcast")
DCM = os.environ.get("DCM_ROOT", "/home/group1/26fall_aiclass/dcm")
ZJT_ROOT = os.environ.get("ZJT_ROOT", "/home/group1/26fall_aiclass/zjt")

ZJT = Path(f"{ZJT_ROOT}/cira-diff/flowcast_v2")
sys.path.insert(0, str(ZJT))
from datasets.goes.runtime import load_vae  # noqa: E402
from datasets.goes.normalization import to_kelvin, normalize_to_neg_one_one, denormalize_to_kelvin  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vae_checkpoint", default=str(ZJT / "runs/vae/vae_1005_150523/models/early_stopping_model.pt"))
    p.add_argument("--vae_config", default=str(ZJT / "runs/vae/vae_1005_150523/config_snapshot.yaml"))
    p.add_argument("--data", default=f"{SATCAST}/edm_GOES_ch13_test_dataset.zarr")
    p.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "vae_recon/vae_recon.npy"))
    p.add_argument("--chunk_samples", type=int, default=64, help="每轮处理的样本数（控制内存）")
    p.add_argument("--batch_frames", type=int, default=8, help="VAE 编解码帧 batch（与 evaluate.py 一致）")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vae, norm = load_vae(args.vae_checkpoint, args.vae_config, device)
    raw = zarr.open(args.data, mode="r")
    n, leads, h, w = raw["output_images"].shape
    assert leads == 18, f"expected 18 lead frames, got {leads}"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    recon = np.zeros((n, leads, h, w), dtype=np.float32)

    se = ae = 0.0
    total = n * leads
    done = 0
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, n, args.chunk_samples):
            stop = min(start + args.chunk_samples, n)
            y = np.asarray(raw["output_images"][start:stop], dtype=np.float32)
            b = y.shape[0]
            truth = to_kelvin(y, norm)
            frames = torch.from_numpy(normalize_to_neg_one_one(y, norm)).reshape(-1, 1, h, w).to(device)
            rec = torch.cat([vae.decode(vae.encode(chunk).latent_dist.mode()).sample
                             for chunk in frames.split(args.batch_frames)])
            rec = denormalize_to_kelvin(rec[:, 0], norm).reshape(b, leads, h, w).cpu().numpy()
            recon[start:stop] = rec
            err = rec - truth
            se += np.square(err).sum(); ae += np.abs(err).sum()
            done += b * leads
            print(f"[{done}/{total}] {time.time() - t0:5.0f}s  mae={ae / (done * h * w):.4f}K", flush=True)

    np.save(out, recon)
    pixels = total * h * w
    mae = ae / pixels
    rmse = float(np.sqrt(se / pixels))
    print(f"wrote {out}  {recon.shape} {recon.dtype}")
    print(f"VAE recon: MAE={mae:.4f}K  RMSE={rmse:.4f}K")

    metric_path = out.with_name(out.stem + "_metrics.json")
    metric_path.write_text(json.dumps(dict(
        model="vae_recon", samples_evaluated=n, mae_k=float(mae), rmse_k=rmse,
        note="VAE reconstruction of the 18 truth frames (encode->mode->decode), Kelvin detail ceiling.",
    ), indent=2), encoding="utf-8")
    print(f"wrote {metric_path}")


if __name__ == "__main__":
    main()
