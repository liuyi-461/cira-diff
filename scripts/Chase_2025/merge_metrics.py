"""Merge sharded eval_edm_Chase2025.py outputs into one zjt-format metrics.json.

Each shard's metrics.json carries a `_accumulator` field with the raw running sums
of the streaming Metrics. Summing them across shards and re-deriving the metrics is
exactly equivalent to a single-process run over the full test set. The same applies
to `_accumulator_by_lead` (per-lead-time sums). If `--forecasts` is given, the
per-shard forecasts_{index}_{count}.npy slices are also assembled (in test order)
into one [1024, 18, 8, 256, 256] float32-Kelvin array via a memory map.

Usage:
    python scripts/Chase_2025/merge_metrics.py \
        --shards runs/evaluation/shard_*/metrics.json \
        --output runs/evaluation/metrics.json \
        --forecasts runs/evaluation/forecasts.npy
"""
import argparse
import json
from math import log10, sqrt
from pathlib import Path

import numpy as np


def result_from_state(state, data_range):
    n, frames = state["n"], state["frames"]
    se, ae, crps, ssim = state["se"], state["ae"], state["crps"], state["ssim"]
    hits, union = state["hits"], state["union"]
    fss_error, fss_ref = state["fss_error"], state["fss_ref"]
    mse = se / n
    return dict(mse_k2=mse, mae_k=ae / n, rmse_k=float(sqrt(mse)),
                psnr_db=float(10 * log10(data_range ** 2 / mse)) if mse else None,
                ssim=ssim / frames, crps_k=crps / n,
                csi=hits / union if union else None,
                fss=1 - fss_error / fss_ref if fss_ref else None,
                pixels=int(n))


def _sum_states(reports, key, metric_names):
    """Sum the named `key` state dicts across shards; returns {name: summed_state}."""
    merged = {name: {k: 0.0 for k in reports[0][key][name]} for name in metric_names}
    for r in reports:
        for name in metric_names:
            for k, v in r[key][name].items():
                merged[name][k] += v
    return merged


def _sum_lead_states(reports, key, metric_names, leads):
    """Sum per-lead state lists across shards; returns {name: [state x leads]}."""
    merged = {name: [{k: 0.0 for k in reports[0][key][name][0]} for _ in range(leads)]
              for name in metric_names}
    for r in reports:
        for name in metric_names:
            for t in range(leads):
                for k, v in r[key][name][t].items():
                    merged[name][t][k] += v
    return merged


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", nargs="+", required=True, help="shard metrics.json files to merge")
    p.add_argument("--output", required=True)
    p.add_argument("--forecasts", default=None,
                   help="assemble shard forecasts_{index}_{count}.npy slices into this file")
    args = p.parse_args(argv)

    reports = [json.loads(Path(f).read_text()) for f in args.shards]
    if not reports:
        p.error("no shards provided")

    # All shards share the same normalization and threshold/fss_window.
    norm = reports[0]["normalization"]
    data_range = norm["kelvin_max"] - norm["kelvin_min"]
    metric_names = [k for k in reports[0]["metrics"].keys()]
    total_samples = 0
    for r in reports:
        total_samples += r["samples_evaluated"]

    merged_state = _sum_states(reports, "_accumulator", metric_names)
    final = dict(
        arguments=dict(reports[0]["arguments"]),
        normalization=norm,
        samples_evaluated=total_samples,
        metric_definition=reports[0]["metric_definition"],
        metrics={name: result_from_state(merged_state[name], data_range) for name in metric_names},
    )
    final["arguments"]["index"] = 0
    final["arguments"]["max_samples"] = None
    final["arguments"]["num_shards"] = len(reports)

    if all("_accumulator_by_lead" in r for r in reports):
        leads = len(reports[0]["_accumulator_by_lead"][metric_names[0]])
        merged_lead = _sum_lead_states(reports, "_accumulator_by_lead", metric_names, leads)
        final["metrics_by_lead"] = {
            name: [result_from_state(merged_lead[name][t], data_range) for t in range(leads)]
            for name in metric_names
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(final["metrics"], indent=2))
    print(f"merged {len(reports)} shards ({total_samples} samples) -> {out}")

    if args.forecasts:
        assemble_forecasts(reports, Path(args.forecasts))


def assemble_forecasts(reports, final_path):
    slices = []  # (index, count, path) in test order
    for r in reports:
        a = r["arguments"]
        count = r["samples_evaluated"]
        path = Path(a["output_dir"]) / f"forecasts_{a['index']}_{count}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing forecast slice {path} (run with --save_forecasts)")
        slices.append((a["index"], count, path))
    slices.sort(key=lambda t: t[0])
    expected = [slices[0][0]] + [slices[i - 1][0] + slices[i - 1][1] for i in range(1, len(slices))]
    for (index, count, path), start in zip(slices, expected):
        if index != start:
            raise ValueError(f"forecast slice gap/overlap at index {index} (expected {start})")

    first = np.load(slices[0][2], mmap_mode="r")
    shape = (sum(c for _, c, _ in slices),) + first.shape[1:]
    total_bytes = int(np.prod(shape)) * first.dtype.itemsize
    print(f"assembling {len(slices)} forecast slices into {final_path} "
          f"(shape {shape}, {first.dtype}, {total_bytes / 2**30:.1f} GiB)")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    dst = np.lib.format.open_memmap(str(final_path), mode="w+", dtype=first.dtype, shape=shape)
    for index, count, path in slices:
        dst[index:index + count] = np.load(path, mmap_mode="r")
    dst.flush()
    print(f"wrote {final_path}")


if __name__ == "__main__":
    main()
