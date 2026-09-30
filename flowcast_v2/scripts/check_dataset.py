"""Dataset smoke test with explicit paths and shared normalization."""
import argparse
import yaml
from torch.utils.data import DataLoader
from datasets.goes.goes_zarr_autoencoder_dataset import GOESZarrAutoencoderDataset, autoencoder_collate
from datasets.goes.goes_zarr_latent_dataset import GOESZarrLatentDataset, goes_latent_collate


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--kind',choices=['raw','latent'],required=True)
    p.add_argument('--config')
    args=p.parse_args(argv)
    if args.kind=='raw':
        if not args.config:
            p.error('raw data requires --config with fitted normalization')
        with open(args.config,encoding='utf-8-sig') as f:
            norm=yaml.safe_load(f)['normalization']
        ds=GOESZarrAutoencoderDataset(args.data,max_samples=2,normalization=norm)
        frames,metas=next(iter(DataLoader(ds,batch_size=3,collate_fn=autoencoder_collate)))
        print(frames.shape,frames.min().item(),frames.max().item(),metas)
    else:
        ds=GOESZarrLatentDataset(args.data,max_samples=2)
        x,y,metas=next(iter(DataLoader(ds,batch_size=2,collate_fn=goes_latent_collate)))
        print(x.shape,y.shape,metas)


if __name__=='__main__':
    main()
