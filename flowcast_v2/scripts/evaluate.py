"""Evaluate GOES forecasts against raw Kelvin targets, including VAE and persistence baselines."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import zarr
from datasets.goes.normalization import sample_count, to_kelvin, normalize_to_neg_one_one, denormalize_to_kelvin
from datasets.goes.runtime import load_vae, load_flowcast, read_config, sha256
from datasets.goes.metrics import Metrics


def decode(vae, latent, norm, frame_batch=8):
    b,t,h,w,c = latent.shape
    frames = latent.permute(0,1,4,2,3).reshape(b*t,c,h,w)
    decoded = torch.cat([vae.decode(chunk).sample for chunk in frames.split(frame_batch)])
    return denormalize_to_kelvin(decoded[:,0], norm).reshape(b,t,*decoded.shape[-2:])


def forecast(model, condition, target_shape, samples, steps):
    condition = model.normalize(condition)
    # Fixed-step Euler, consistent with training's normalized latent space.
    for _ in range(samples):
        state = torch.randn(target_shape,device=condition.device)
        for step in range(steps):
            time = torch.full((len(state),),step/steps,device=state.device)
            state = state + model(time,state,condition)/steps
        yield model.denormalize(state)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='Independent raw validation/test Zarr')
    p.add_argument('--vae_checkpoint',required=True)
    p.add_argument('--vae_config',required=True)
    p.add_argument('--cfm_checkpoint')
    p.add_argument('--cfm_config')
    p.add_argument('--latent_path')
    p.add_argument('--output_dir',default='runs/evaluation')
    p.add_argument('--max_samples',type=int)
    p.add_argument('--batch_size',type=int,default=1)
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threshold',type=float,default=235.)
    p.add_argument('--fss_window',type=int,default=9)
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--figure',action='store_true')
    p.add_argument('--split',choices=['val','test'],default='test')
    args=p.parse_args(argv)
    if min(args.batch_size,args.samples,args.steps) <= 0 or args.index < 0:
        p.error('batch_size/samples/steps must be positive and index nonnegative')
    if args.cfm_checkpoint and not (args.cfm_config and args.latent_path):
        p.error('CFM evaluation requires --cfm_config and --latent_path')
    torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae,norm=load_vae(args.vae_checkpoint,args.vae_config,device)
    raw=zarr.open(args.data,mode='r')
    n=sample_count(raw)
    if args.index >= n:
        p.error('index is outside the dataset')
    count=sample_count(raw,args.max_samples)
    end=min(n,args.index+count)
    configs=[read_config(args.vae_config)]
    latent=model=None
    if args.cfm_checkpoint:
        config=read_config(args.cfm_config); configs.append(config)
        snapshot=Path(args.cfm_checkpoint).resolve().parent.parent/'config_snapshot.yaml'
        if not snapshot.is_file() or read_config(snapshot) != config:
            raise ValueError('Use the CFM checkpoint training config_snapshot.yaml')
        latent=zarr.open(args.latent_path,mode='r')
        if latent.attrs.get('complete') is not True or latent.attrs.get('encoding') != 'mode':
            raise ValueError('Incomplete or legacy latent store; re-encode')
        if latent.attrs.get('vae_sha256') != sha256(args.vae_checkpoint) or latent.attrs.get('normalization') != norm:
            raise ValueError('Latent encoder does not match VAE checkpoint/config')
        if Path(latent.attrs['source_path']).resolve() != Path(args.data).resolve():
            raise ValueError('Latent source differs from raw ground truth')
        if config.get('normalization') != norm or sha256(config['autoencoder_params']['autoencoder_checkpoint']) != latent.attrs['vae_sha256']:
            raise ValueError('CFM was trained with a different VAE/normalization')
        if end > latent['input_images'].shape[0]:
            raise ValueError('Not enough latent samples for selected raw range; set --max_samples')
        def shape(key):
            _,t,c,h,w=latent[key].shape
            return (t,h,w,c)
        model=load_flowcast(args.cfm_checkpoint,args.cfm_config,shape('input_images'),shape('output_images'),device)
    # Test data must be independent of both training and model selection data.
    for config in configs:
        provenance=config.get('data_provenance',{})
        for split in (('train','val') if args.split=='test' else ('train',)):
            if provenance.get(split) and Path(provenance[split]).resolve()==Path(args.data).resolve():
                raise ValueError(f'{args.split} evaluation source overlaps {split} data')
    metrics={name:Metrics(norm['kelvin_max']-norm['kelvin_min'],args.threshold,args.fss_window)
             for name in (['vae','persistence','flowcast'] if model else ['vae','persistence'])}
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for start in range(args.index,end,args.batch_size):
            stop=min(start+args.batch_size,end)
            x=np.asarray(raw['input_images'][start:stop],dtype=np.float32)
            y=np.asarray(raw['output_images'][start:stop],dtype=np.float32)
            truth=to_kelvin(y,norm)
            persistence=np.repeat(to_kelvin(x[:,-1:],norm),y.shape[1],axis=1)
            metrics['persistence'].update(persistence[None],truth)
            b,t,h,w=y.shape
            frames=torch.from_numpy(normalize_to_neg_one_one(y,norm)).reshape(-1,1,h,w).to(device)
            rec=torch.cat([vae.decode(vae.encode(chunk).latent_dist.mode()).sample for chunk in frames.split(8)])
            rec=denormalize_to_kelvin(rec[:,0],norm).reshape(b,t,h,w).cpu().numpy()
            metrics['vae'].update(rec[None],truth)
            predictions=None
            if model:
                condition=torch.from_numpy(np.asarray(latent['input_images'][start:stop])).permute(0,1,3,4,2).to(device)
                target_shape=(b,*shape('output_images'))
                predictions=np.stack([decode(vae,z,norm).cpu().numpy() for z in forecast(model,condition,target_shape,args.samples,args.steps)])
                metrics['flowcast'].update(predictions,truth)
            if args.figure and start==args.index:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                pictures=[to_kelvin(x[0,-1],norm),truth[0,0],rec[0,0]]
                titles=['Last observation (K)','Raw target (K)','VAE reconstruction (K)']
                if predictions is not None:
                    pictures.extend([predictions[:,0,0].mean(0),np.quantile(predictions[:,0,0],.1,axis=0),np.quantile(predictions[:,0,0],.9,axis=0)])
                    titles.extend(['Ensemble mean (K)','10% quantile (K)','90% quantile (K)'])
                fig,axes=plt.subplots(1,len(pictures),figsize=(5*len(pictures),5))
                for ax,picture,title in zip(axes,pictures,titles):
                    ax.imshow(picture,cmap='Spectral_r',vmin=norm['kelvin_min'],vmax=norm['kelvin_max']);ax.set_title(title);ax.axis('off')
                fig.tight_layout();fig.savefig(out/'comparison.png',dpi=120);plt.close(fig)
            print(f'Evaluated {stop-args.index}/{end-args.index}',flush=True)
    report=dict(arguments=vars(args),normalization=norm,samples_evaluated=end-args.index,
                metric_definition='Kelvin; CSI/FSS on ensemble-mean Tb <= threshold; reflected spatial windows; PSNR null for zero MSE; CSI/FSS null for no events',
                metrics={k:v.result() for k,v in metrics.items()})
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report['metrics'],indent=2))


if __name__=='__main__':
    main()
