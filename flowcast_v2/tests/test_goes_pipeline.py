import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
import zarr
from datasets.goes.normalization import (fit_normalization,normalize_to_neg_one_one,
    denormalize_to_kelvin,to_kelvin,require_distinct)
from datasets.goes.goes_zarr_autoencoder_dataset import GOESZarrAutoencoderDataset,autoencoder_collate
from datasets.goes.goes_zarr_latent_dataset import GOESZarrLatentDataset
from datasets.goes.metrics import Metrics
from datasets.goes.runtime import compute_mean_std,validate_latent_pair
from tools.run_sweep import configure,expand


@pytest.fixture
def raw(tmp_path):
    path=tmp_path/'raw.zarr'
    data=zarr.open_group(str(path),mode='w')
    x=np.arange(3*2*8*8,dtype=np.float32).reshape(3,2,8,8)/100-3
    y=np.arange(3*8*8,dtype=np.float32).reshape(3,1,8,8)/100
    data.create_dataset('input_images',data=x)
    data.create_dataset('output_images',data=y)
    return path,x,y


def test_normalization_reversible_without_clipping(raw):
    path,x,y=raw
    norm=fit_normalization(path)
    values=np.concatenate([x.ravel(),y.ravel(),[-10.,10.]])
    np.testing.assert_allclose(denormalize_to_kelvin(normalize_to_neg_one_one(values,norm),norm),to_kelvin(values,norm),rtol=1e-6)
    assert normalize_to_neg_one_one(np.array([10.]),norm)[0]>1
    train=np.concatenate([x.ravel(),y.ravel()])
    assert normalize_to_neg_one_one(train,norm).min()==pytest.approx(-1)
    assert normalize_to_neg_one_one(train,norm).max()==pytest.approx(1)


def test_all_frames_and_collation(raw):
    path,x,y=raw
    norm=fit_normalization(path)
    ds=GOESZarrAutoencoderDataset(path,normalization=norm,max_samples=2)
    assert len(ds)==6
    for i in range(len(ds)):
        frame,meta=ds[i]
        original=np.concatenate([x[i//3],y[i//3]])[i%3]
        np.testing.assert_allclose(frame[0],normalize_to_neg_one_one(original,norm))
        assert meta=={'idx':i//3,'frame':i%3}
    frames,metas=autoencoder_collate([ds[0],ds[1]])
    assert frames.shape==(2,1,8,8) and len(metas)==2
    assert torch.equal(ds[0][0],ds[0][0])
    with pytest.raises(ValueError):
        GOESZarrAutoencoderDataset(path,normalization=norm,max_samples=0)


def test_split_overlap_rejected(raw):
    path,_,_=raw
    with pytest.raises(ValueError,match='independent'):
        require_distinct(path,path.parent/'.'/path.name)


def test_stream_encoder_mode_and_partial_failure(raw,tmp_path,monkeypatch):
    from scripts.encode_latent import encode
    import datasets.goes.runtime as runtime
    path,x,y=raw
    norm=fit_normalization(path)
    seen=[]
    class FakeVAE:
        def encode(self,frames):
            seen.append(frames.cpu().clone())
            class Distribution:
                def mode(self):return frames*2
                def sample(self):raise AssertionError('sample must not be called')
            return SimpleNamespace(latent_dist=Distribution())
    monkeypatch.setattr(runtime,'load_vae',lambda *a:(FakeVAE(),norm))
    monkeypatch.setattr(runtime,'sha256',lambda *a:'hash')
    out=tmp_path/'latent.zarr'
    encode(path,'ckpt','config',out,batch_size=2,device='cpu')
    data=zarr.open(str(out),mode='r')
    assert data.attrs['complete'] is True
    np.testing.assert_allclose(data['input_images'][:,:,0],normalize_to_neg_one_one(x,norm)*2)
    assert max(len(s) for s in seen)<=2
    ds=GOESZarrLatentDataset(out)
    assert len(ds)==3 and ds[0][0].shape==(2,8,8,1)
    with pytest.raises((ValueError,FileExistsError)):
        encode(path,'ckpt','config',out,batch_size=2,device='cpu')
    broken=zarr.open(str(out),mode='a');broken.attrs['complete']=False
    with pytest.raises(ValueError,match='Incomplete'):
        GOESZarrLatentDataset(out)
    class BrokenVAE:
        def encode(self,frames):raise RuntimeError('injected encoding failure')
    monkeypatch.setattr(runtime,'load_vae',lambda *a:(BrokenVAE(),norm))
    failed=tmp_path/'failed.zarr'
    with pytest.raises(RuntimeError):
        encode(path,'ckpt','config',failed,device='cpu')
    assert zarr.open(str(failed),mode='r').attrs['complete'] is False


def test_metrics_perfect_and_crps():
    target=np.ones((1,1,8,8))*230
    perfect=Metrics(100)
    perfect.update(target[None],target)
    result=perfect.result()
    assert result['mse_k2']==0 and result['crps_k']==0
    assert result['ssim']==pytest.approx(1) and result['fss']==pytest.approx(1) and result['csi']==1
    scored=Metrics(100)
    scored.update(np.stack([target-2,target+2]),target)
    assert scored.result()['crps_k']==pytest.approx(1)
    absent=Metrics(100,threshold=200)
    absent.update(target[None],target)
    assert absent.result()['csi'] is None and absent.result()['fss'] is None


def test_sweep_validation_and_no_template_mutation():
    base={'loss_params':{'kl_weight':.1},'training_params':{}}
    result=configure(base,{'kl_weight':1e-4},'vae')
    assert base['loss_params']['kl_weight']==.1
    assert result['loss_params']['kl_weight']==1e-4
    with pytest.raises(ValueError,match='numeric'):
        configure(base,{'kl_weight':'1e-4'},'vae')
    with pytest.raises(ValueError):expand({'start':0,'end':1,'step':0})


def test_weighted_latent_statistics():
    batches=[(torch.tensor([1.,2.]),torch.tensor([3.]),None),
             (torch.tensor([4.]),torch.tensor([5.]),None)]
    mean,std=compute_mean_std(batches)
    assert mean.item()==3
    assert std.item()==pytest.approx(np.std([1,2,3,4,5]))


def test_latent_provenance_mismatch():
    train=SimpleNamespace(attrs=dict(vae_sha256='a',normalization={},encoding='mode',source_path='train'))
    val=SimpleNamespace(attrs=dict(vae_sha256='b',normalization={},encoding='mode',source_path='val'))
    with pytest.raises(ValueError,match='vae_sha256'):
        validate_latent_pair(train,val,{})


def test_forecast_normalizes_condition_and_denormalizes_output():
    from scripts.evaluate import forecast
    class Model:
        def normalize(self,x):return (x-10)/2
        def denormalize(self,x):return x*2+10
        def __call__(self,t,state,condition):
            assert torch.equal(condition,torch.ones_like(condition))
            return torch.zeros_like(state)
    torch.manual_seed(3)
    noise=torch.randn(1,1,2,2,1)
    torch.manual_seed(3)
    output=list(forecast(Model(),torch.full((1,2,2,2,1),12.),(1,1,2,2,1),1,2))[0]
    torch.testing.assert_close(output,noise*2+10)


def test_sweep_failure_records_and_skips_cfm(raw,tmp_path,monkeypatch):
    import tools.run_sweep as sweep
    train,_,_=raw
    val=tmp_path/'val.zarr';val.mkdir()
    cfg={'common':{'train_file':str(train),'val_file':str(val),'normalization':fit_normalization(train)},
         'vae':{'kl_weight':[1.0e-4]},'cfm':{'sigma':[.01]}}
    config=tmp_path/'sweep.yaml';config.write_text(yaml.safe_dump(cfg))
    commands=[]
    def fail(cmd,log,env):commands.append(cmd);return 9
    monkeypatch.setattr(sweep,'run_cmd',fail)
    monkeypatch.setattr('sys.argv',['sweep','--config',str(config),'--output_dir',str(tmp_path/'runs')])
    assert sweep.main()==1
    assert len(commands)==1
    rows=list((tmp_path/'runs').glob('*/results.csv'))
    import csv
    with rows[0].open() as handle:row=list(csv.DictReader(handle))[0]
    assert row['vae_rc']=='9' and row['cfm_rc']=='' and row['encode_train_rc']==''


def test_evaluate_uses_raw_targets_and_persistence(raw,tmp_path,monkeypatch):
    import scripts.evaluate as evaluation
    path,x,y=raw
    norm=fit_normalization(path)
    config=tmp_path/'vae.yaml'
    config.write_text(yaml.safe_dump({'normalization':norm,'data_provenance':{'train':'independent_train'}}))
    class VAE:
        def encode(self,frames):return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:frames))
        def decode(self,frames):return SimpleNamespace(sample=frames + .1)
    monkeypatch.setattr(evaluation,'load_vae',lambda *a:(VAE(),norm))
    out=tmp_path/'evaluation'
    evaluation.main(['--data',str(path),'--vae_checkpoint','dummy','--vae_config',str(config),
                     '--output_dir',str(out),'--batch_size','2'])
    report=json.loads((out/'metrics.json').read_text())
    delta=.05*(norm['kelvin_max']-norm['kelvin_min'])
    assert report['metrics']['vae']['mae_k']==pytest.approx(delta,rel=1e-4)
    expected=np.mean((to_kelvin(x[:,-1:],norm)-to_kelvin(y,norm))**2)
    assert report['metrics']['persistence']['mse_k2']==pytest.approx(float(expected),rel=1e-5)
    assert report['samples_evaluated']==3


@pytest.mark.parametrize('fail_encoding',[True,False])
def test_sweep_isolated_paths_and_encoding_failure(raw,tmp_path,monkeypatch,fail_encoding):
    import tools.run_sweep as sweep
    import csv
    train,_,_=raw
    val=tmp_path/'val.zarr';val.mkdir()
    cfg={'common':{'train_file':str(train),'val_file':str(val),'normalization':fit_normalization(train)},
         'vae':{'kl_weight':[1.0e-4]},'cfm':{'sigma':[.01]}}
    config=tmp_path/'sweep.yaml';config.write_text(yaml.safe_dump(cfg))
    templates={p:p.read_bytes() for p in (sweep.ROOT/'configs/goes').glob('*.yaml')}
    commands=[]
    def run(cmd,log,env):
        cmd=[str(c) for c in cmd];commands.append(cmd)
        if 'scripts.encode_latent' in cmd:
            return 8 if fail_encoding else 0
        output=Path(cmd[cmd.index('--output_dir')+1])
        (output/'models').mkdir(parents=True)
        (output/'models/early_stopping_model.pt').touch()
        (output/'losses.csv').write_text('epoch,train_loss,val_loss\n0,1,.75\n1,.5,.6\n')
        return 0
    monkeypatch.setattr(sweep,'run_cmd',run)
    monkeypatch.setattr('sys.argv',['sweep','--config',str(config),'--output_dir',str(tmp_path/'runs')])
    assert sweep.main()==int(fail_encoding)
    assert all(p.read_bytes()==content for p,content in templates.items())
    with next((tmp_path/'runs').glob('*/results.csv')).open() as handle:row=list(csv.DictReader(handle))[0]
    if fail_encoding:
        assert row['encode_train_rc']=='8' and row['cfm_rc']==''
        assert len(commands)==2
    else:
        cfm=commands[-1]
        assert cfm[cfm.index('--train_file')+1] != cfm[cfm.index('--val_file')+1]
        assert row['cfm_val_loss']=='0.6'
        assert len(commands)==4


def test_zero_step_warmup():
    # Load only the scheduler helper, avoiding unrelated optional metric imports.
    import ast
    source=Path(__file__).resolve().parents[1]/'common/utils/utils.py'
    module=ast.parse(source.read_text(encoding='utf-8'))
    function=next(node for node in module.body if isinstance(node,ast.FunctionDef) and node.name=='warmup_lambda')
    scope={}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),scope)
    assert scope['warmup_lambda'](0)(0)==1.0
    assert scope['warmup_lambda'](10)(0)==pytest.approx(.1)
    assert scope['warmup_lambda'](10)(10)==pytest.approx(1.)


def test_small_test_dry_run(tmp_path):
    from run_small_test import main
    assert main(['--train_file',str(tmp_path/'train.zarr'),'--val_file',str(tmp_path/'val.zarr'),
                 '--output_dir',str(tmp_path/'output'),'--dry_run'])==0
    config=json.loads(next((tmp_path/'output').glob('*/small_test_config.yaml')).read_text())
    assert config['common']['num_gpus']==1
    assert config['common']['max_samples']==16 and config['common']['val_max_samples']==4
    assert config['vae']['num_epochs']==[2] and config['vae']['warmup_generator_epochs']==[0]
    assert config['cfm']['num_epochs']==[1]


def test_small_test_rejects_shared_data(tmp_path):
    from run_small_test import main
    with pytest.raises(SystemExit) as error:
        main(['--train_file',str(tmp_path/'same'),'--val_file',str(tmp_path/'same'),'--dry_run'])
    assert error.value.code==2
