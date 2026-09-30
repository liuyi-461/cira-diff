"""Small GOES pipeline smoke test. Run --help for options; requires a CUDA training environment."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def run_logged(command, logfile, env):
    print('[RUN]', subprocess.list2cmdline(command), flush=True)
    with logfile.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_file', required=True, help='Raw training Zarr directory')
    parser.add_argument('--val_file', required=True, help='Independent raw validation Zarr directory')
    parser.add_argument('--train_samples', type=positive, default=16)
    parser.add_argument('--val_samples', type=positive, default=4)
    parser.add_argument('--vae_epochs', type=positive, default=2, help='At least 2 to produce a VAE checkpoint')
    parser.add_argument('--cfm_epochs', type=positive, default=1)
    parser.add_argument('--gpu', help='Optional single GPU ID; otherwise use the first allocated visible GPU')
    parser.add_argument('--output_dir', default=str(ROOT / 'runs' / 'small_tests'))
    parser.add_argument('--dry_run', action='store_true', help='Write config and show the command without loading data or training')
    args = parser.parse_args(argv)
    if args.vae_epochs < 2:
        parser.error('--vae_epochs must be >= 2: epoch 0 activates the discriminator; epoch 1 saves the checkpoint')
    if args.gpu is not None and (not args.gpu.strip() or ',' in args.gpu):
        parser.error('--gpu must name exactly one GPU')
    train, val = Path(args.train_file).resolve(), Path(args.val_file).resolve()
    if train == val or (train.exists() and val.exists() and train.samefile(val)):
        parser.error('Training and validation must use independent datasets')
    if not args.dry_run:
        if not train.is_dir() or not val.is_dir():
            parser.error('Both data paths must be existing Zarr directories')
    env = os.environ.copy()
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.gpu.strip()
    env['PYTHONUNBUFFERED'] = '1'
    # Run the probe in a child so --gpu takes effect before CUDA initialization.
    if not args.dry_run:
        probe = subprocess.run([sys.executable, '-c',
            'import torch; print("CUDA:", torch.cuda.is_available()); '
            'raise SystemExit(0 if torch.cuda.is_available() else 1)'], env=env, cwd=ROOT)
        if probe.returncode:
            print('A CUDA GPU and the project training dependencies are required.', file=sys.stderr)
            return probe.returncode
    run_dir = Path(args.output_dir).resolve() / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    config = {
        'common': {'num_gpus': 1, 'gpu_ids': None, 'train_file': str(train), 'val_file': str(val),
                   'test_file': None, 'max_samples': args.train_samples, 'val_max_samples': args.val_samples,
                   'encode_batch_size': 1, 'source_mean': 279.0699, 'source_std': 19.3297, 'normalization': None},
        'vae': {'learning_rate': [0.0001], 'num_epochs': [args.vae_epochs], 'micro_batch_size': [1],
                'num_workers': [0], 'kl_weight': [1.0e-4], 'disc_weight': [0.5], 'warmup_generator_epochs': [0]},
        'cfm': {'learning_rate': [0.0005], 'num_epochs': [args.cfm_epochs], 'micro_batch_size': [1],
                'num_workers': [0], 'sigma': [0.01]},
    }
    config_path = run_dir / 'small_test_config.yaml'
    # JSON is valid YAML, keeping --help and --dry_run free of training dependencies.
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    command = [sys.executable, '-m', 'tools.run_sweep', '--config', str(config_path), '--output_dir', str(run_dir)]
    print(f'Output: {run_dir}', flush=True)
    print('Smoke test only: these short runs do not measure model quality.', flush=True)
    if args.dry_run:
        print('[DRY RUN]', subprocess.list2cmdline(command))
        return 0
    summary = {'config': str(config_path), 'pipeline_rc': None, 'evaluation_rc': None, 'status': 'running'}
    try:
        summary['pipeline_rc'] = run_logged(command, run_dir / 'pipeline.log', env)
        if summary['pipeline_rc'] != 0:
            raise RuntimeError('Training/encoding failed; inspect pipeline.log and the per-stage logs')
        results = list(run_dir.glob('sweep_*/results.csv'))
        if len(results) != 1:
            raise RuntimeError('Expected exactly one results.csv in this isolated test directory')
        with results[0].open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1 or any(rows[0][k] != '0' for k in ('vae_rc','encode_train_rc','encode_val_rc','cfm_rc')):
            raise RuntimeError('Not all training/encoding stages succeeded')
        stage = Path(rows[0]['out_dir'])
        evaluation = [sys.executable, '-m', 'scripts.evaluate', '--data', str(val), '--split', 'val',
            '--latent_path', str(stage / 'val.zarr'),
            '--vae_checkpoint', str(stage / 'vae/models/early_stopping_model.pt'),
            '--vae_config', str(stage / 'vae/config_snapshot.yaml'),
            '--cfm_checkpoint', str(stage / 'cfm/models/early_stopping_model.pt'),
            '--cfm_config', str(stage / 'cfm/config_snapshot.yaml'),
            '--max_samples', str(min(2, args.val_samples)), '--batch_size', '1', '--samples', '2', '--steps', '4',
            '--figure', '--output_dir', str(stage / 'smoke_evaluation')]
        summary['evaluation_rc'] = run_logged(evaluation, run_dir / 'evaluation.log', env)
        if summary['evaluation_rc'] != 0:
            raise RuntimeError('Prediction/evaluation failed; inspect evaluation.log')
        metrics = stage / 'smoke_evaluation/metrics.json'
        figure = stage / 'smoke_evaluation/comparison.png'
        if not metrics.is_file() or not figure.is_file():
            raise RuntimeError('Evaluation did not produce the expected metrics and figure')
        summary.update(status='passed', metrics=str(metrics), figure=str(figure), results=str(results[0]))
        print(f'PASS: {metrics}\nFigure: {figure}', flush=True)
        return 0
    except Exception as error:
        summary.update(status='failed', error=str(error))
        print(f'FAIL: {error}', file=sys.stderr)
        return 1
    finally:
        (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
