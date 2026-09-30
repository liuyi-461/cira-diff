"""Isolated VAE -> train/validation latent -> CFM sweeps; never edit source configs."""
import argparse
import copy
import csv
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid
import yaml
from datasets.goes.normalization import fit_normalization, require_distinct, validate_normalization

ROOT = Path(__file__).resolve().parents[1]


def expand(value):
    if isinstance(value, dict):
        start, end, step = (float(value[k]) for k in ('start', 'end', 'step'))
        if not all(math.isfinite(v) for v in (start, end, step)) or step <= 0 or end < start:
            raise ValueError('Sweep ranges require finite start <= end and step > 0')
        return [start + i * step for i in range(int((end - start) / step + 1e-9) + 1)]
    return value if isinstance(value, list) else [value]


def generate_combos(section):
    return [dict(zip(section, values)) for values in itertools.product(*(expand(v) for v in section.values()))]


def configure(base, params, stage):
    cfg = copy.deepcopy(base)
    mapping = {'learning_rate': 'optimizer_params', 'kl_weight': 'loss_params',
               'disc_weight': 'loss_params', 'sigma': 'flow_matching_params'}
    for key, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{stage}.{key} must be numeric; write 1.0e-4 instead of 1e-4')
        if key in ('num_epochs', 'micro_batch_size', 'warmup_generator_epochs'):
            if int(value) != value or value < (0 if key == 'warmup_generator_epochs' else 1):
                raise ValueError(f'Invalid integer parameter: {key}')
            value = int(value)
        elif value < 0:
            raise ValueError(f'{key} must be nonnegative')
        cfg[mapping.get(key, 'training_params')][key] = value
    return cfg


def run_cmd(cmd, log_path, env):
    print('[CMD]', subprocess.list2cmdline([str(x) for x in cmd]), flush=True)
    with open(log_path, 'w', encoding='utf-8') as handle:
        return subprocess.run([str(x) for x in cmd], cwd=ROOT, env=env,
                              stdout=handle, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'sweep_config.yaml'))
    parser.add_argument('--train_file')
    parser.add_argument('--val_file')
    parser.add_argument('--test_file')
    parser.add_argument('--output_dir', default=str(ROOT / 'runs'))
    args = parser.parse_args()
    with open(args.config, encoding='utf-8-sig') as handle:
        sweep = yaml.safe_load(handle)
    common = sweep['common']
    paths = {split: getattr(args, split + '_file') or common.get(split + '_file') for split in ('train','val','test')}
    if not paths['train'] or not paths['val']:
        parser.error('--train_file and --val_file (independent GOES stores) are required')
    require_distinct(*paths.values())
    paths = {k: str(Path(v).resolve()) if v else None for k,v in paths.items()}
    vae_base = yaml.safe_load((ROOT / 'configs/goes/autoencoder_kl.yaml').read_text())
    cfm_base = yaml.safe_load((ROOT / 'configs/goes/flowcast.yaml').read_text())
    combos = list(itertools.product(generate_combos(sweep['vae']), generate_combos(sweep['cfm'])))
    # Reject malformed sweep values before expensive scans or training.
    configs = [(configure(vae_base, v, 'vae'), configure(cfm_base, c, 'cfm')) for v,c in combos]
    norm = common.get('normalization')
    norm = validate_normalization(norm) if norm else fit_normalization(paths['train'],
        mean=common.get('source_mean',279.0699), std=common.get('source_std',19.3297),
        max_samples=common.get('max_samples'))
    env = os.environ.copy()
    if common.get('gpu_ids') is not None and 'CUDA_VISIBLE_DEVICES' not in env:
        env['CUDA_VISIBLE_DEVICES'] = str(common['gpu_ids'])
    visible = env.get('CUDA_VISIBLE_DEVICES')
    ngpu = int(common.get('num_gpus') or (len(visible.split(',')) if visible else 1))
    if ngpu <= 0 or (visible and ngpu > len(visible.split(','))):
        raise ValueError('num_gpus exceeds the allocated visible GPUs')
    session = Path(args.output_dir).resolve() / ('sweep_' + uuid.uuid4().hex[:12])
    session.mkdir(parents=True)
    result_path = session / 'results.csv'
    fields = ['run_id','vae_rc','encode_train_rc','encode_val_rc','encode_test_rc','cfm_rc','cfm_val_loss','error','out_dir']
    with result_path.open('w',newline='',encoding='utf-8') as handle:
        csv.DictWriter(handle,fieldnames=fields).writeheader()
    failed = False
    for i, (vae_cfg,cfm_cfg) in enumerate(configs):
        run = session / f'run_{i:03d}'
        run.mkdir()
        row = dict.fromkeys(fields)
        row.update(run_id=i,out_dir=str(run))
        vae_ckpt = run / 'vae/models/early_stopping_model.pt'
        for cfg in (vae_cfg,cfm_cfg):
            cfg['normalization'] = norm
            cfg['data_provenance'] = paths
        cfm_cfg['autoencoder_params']['autoencoder_checkpoint'] = str(vae_ckpt)
        for name,cfg in [('vae',vae_cfg),('cfm',cfm_cfg),('config_snapshot',dict(sweep=sweep,paths=paths,normalization=norm))]:
            (run / (name + '.yaml')).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        launcher = [sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={ngpu}']
        def train_cmd(stage,train,val):
            module = 'experiments.goes.autoencoder.dist_train_autoencoder_kl' if stage == 'vae' else 'experiments.goes.runner.flowcast.dist_train_flowcast'
            cmd = launcher + ['--module',module,'--config',run / (stage+'.yaml'),'--train_file',train,'--val_file',val,'--output_dir',run/stage]
            for option,key in [('max_samples','max_samples'),('val_max_samples','val_max_samples')]:
                if common.get(key) is not None:
                    cmd += ['--'+option,str(common[key])]
            return cmd
        try:
            row['vae_rc'] = run_cmd(train_cmd('vae',paths['train'],paths['val']),run/'vae.log',env)
            if row['vae_rc'] != 0 or not vae_ckpt.is_file():
                raise RuntimeError('VAE failed or its checkpoint was not produced')
            for split,source in paths.items():
                if not source:
                    continue
                cmd = [sys.executable,'-m','scripts.encode_latent','--zarr_path',source,'--ckpt_path',vae_ckpt,
                       '--config',run/'vae.yaml','--out_path',run/(split+'.zarr'),
                       '--batch_size',str(common.get('encode_batch_size',8))]
                limit = common.get('max_samples' if split=='train' else split+'_max_samples')
                if limit is not None:
                    cmd += ['--n_samples',str(limit)]
                key = 'encode_'+split+'_rc'
                row[key] = run_cmd(cmd,run/(key+'.log'),env)
                if row[key] != 0:
                    raise RuntimeError(f'{split} encoding failed; CFM skipped')
            row['cfm_rc'] = run_cmd(train_cmd('cfm',run/'train.zarr',run/'val.zarr'),run/'cfm.log',env)
            if row['cfm_rc'] != 0:
                raise RuntimeError('CFM training failed')
            with (run/'cfm/losses.csv').open() as handle:
                row['cfm_val_loss'] = min(float(r['val_loss']) for r in csv.DictReader(handle))
        except Exception as error:
            row['error'] = str(error)
            failed = True
        finally:
            with result_path.open('a',newline='',encoding='utf-8') as handle:
                csv.DictWriter(handle,fieldnames=fields).writerow(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    print(f'Results: {result_path}')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
