"""Fit training-only brightness-temperature bounds into a VAE config."""
import argparse
from pathlib import Path
import yaml
from datasets.goes.normalization import fit_normalization

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train_file',required=True)
    p.add_argument('--config',default='configs/goes/autoencoder_kl.yaml')
    p.add_argument('--out',required=True)
    p.add_argument('--mean',type=float,default=279.0699)
    p.add_argument('--std',type=float,default=19.3297)
    args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8-sig'))
    config['normalization']=fit_normalization(args.train_file,args.mean,args.std)
    output=Path(args.out)
    if output.resolve()==Path(args.config).resolve():
        p.error('--out must not overwrite the template')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as handle:
        yaml.safe_dump(config,handle,sort_keys=False)

if __name__=='__main__':
    main()
