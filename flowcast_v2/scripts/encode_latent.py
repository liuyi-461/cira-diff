"""Stream deterministic latent encodings to a new Zarr store."""
import argparse
from pathlib import Path


def encode(zarr_path, ckpt_path, config, out_path, n_samples=None, batch_size=8, device=None):
    import numpy as np
    import torch
    import zarr
    from datasets.goes.normalization import normalize_to_neg_one_one, sample_count
    from datasets.goes.runtime import load_vae, sha256
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    vae, norm = load_vae(ckpt_path, config, device)
    source = zarr.open(str(zarr_path), mode='r')
    n = sample_count(source, n_samples)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out = zarr.open_group(str(out_path), mode='w-')
    out.attrs.update(dict(complete=False, source_path=str(Path(zarr_path).resolve()),
                          vae_checkpoint=str(Path(ckpt_path).resolve()), vae_sha256=sha256(ckpt_path),
                          normalization=norm, encoding='mode', samples=n))
    with torch.inference_mode():
        for key in ('input_images', 'output_images'):
            if source[key].ndim != 4:
                raise ValueError('Expected raw GOES (N,T,H,W) arrays')
            for start in range(0, n, batch_size):
                raw = np.asarray(source[key][start:min(start + batch_size, n)], dtype=np.float32)
                if not np.isfinite(raw).all():
                    raise ValueError(f'Nonfinite data at {key}:{start}')
                b, t, h, w = raw.shape
                frames = torch.from_numpy(normalize_to_neg_one_one(raw, norm)).reshape(-1, 1, h, w).to(device)
                # Limit GPU frame batch size independently of the sequence length.
                latent = torch.cat([vae.encode(chunk).latent_dist.mode() for chunk in frames.split(batch_size)])
                array = latent.reshape(b, t, *latent.shape[1:]).cpu().numpy().astype('float32')
                if key not in out:
                    out.create_dataset(key, shape=(n, *array.shape[1:]),
                                       chunks=(1, *array.shape[1:]), dtype='float32')
                out[key][start:start + b] = array
                print(f'{key}: {start + b}/{n}', flush=True)
    out.attrs['complete'] = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('zarr_path', 'ckpt_path', 'config', 'out_path'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--n_samples', type=int)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--device')
    encode(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
