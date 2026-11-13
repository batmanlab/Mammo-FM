"""Original MSAE optimization recipe with explicit provenance and safe validation.

Matches original model, loss, AdamW, decoder constraints, batch size, seed,
30 epochs and LinearDecayLR. Validation uses training normalization. last.pt
is the original final-epoch selection; best.pt is a separate diagnostic.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import numpy as np
import torch
from mammo_sae.core import SpatialSAE
from mammo_sae.data import read_features
from mammo_sae.impl.sae import MatryoshkaAutoencoder
from mammo_sae.impl.utils import LinearDecayLR
from mammo_sae.training import Features, statistics, evaluate


def train_step(sae,x,optimizer):
    optimizer.zero_grad()
    normalized=(x-sae.mean)*sae.scale
    recons,_,_,_=sae.model(normalized)
    weights=sae.model.relative_importance
    loss=sum(w*torch.nn.functional.mse_loss(r,normalized) for w,r in zip(weights,recons))/sum(weights)
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward()
    sae.model.scale_to_unit_norm()
    sae.model.project_grads_decode()
    torch.nn.utils.clip_grad_norm_(sae.model.parameters(),1.0)
    optimizer.step()
    return float(loss.detach())


def train(train_path,val_path,config,output,epochs=30,batch_size=4096,device='cuda',provenance=None):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if Path(train_path).resolve()==Path(val_path).resolve():raise ValueError('Training and validation must differ')
    random.seed(12345);np.random.seed(12345);torch.manual_seed(12345)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(12345)
    values,_=read_features(train_path);validation,_=read_features(val_path)
    if values.shape[1]!=config['n_inputs'] or validation.shape[1]!=config['n_inputs']:
        raise ValueError('Feature dimensions do not match SAE')
    if epochs<1 or batch_size<1 or len(values)<batch_size:raise ValueError('Invalid epochs/batch or fewer than one complete training batch')
    mean,scale,target=statistics(values,False,None)
    sae=SpatialSAE(MatryoshkaAutoencoder(**config),mean,scale).to(device)
    optimizer=torch.optim.AdamW(sae.model.parameters(),lr=1e-4,betas=(.9,.999),eps=6.25e-10,
                                weight_decay=0.,fused=device.startswith('cuda'))
    scheduler=LinearDecayLR(optimizer,epochs,decay_time=.8)
    loader=torch.utils.data.DataLoader(Features(values),batch_size=batch_size,shuffle=True,drop_last=True,num_workers=0)
    validation_loader=torch.utils.data.DataLoader(Features(validation),batch_size=batch_size,shuffle=False,num_workers=0)
    output.mkdir(parents=True)
    run=dict(model_config=config,epochs=epochs,batch_size=batch_size,seed=12345,lr=.0001,
             scheduler='LinearDecayLR',decay_time=.8,decoder_unit_norm=True,decoder_gradient_projection=True,
             train_features=str(Path(train_path).resolve()),val_features=str(Path(val_path).resolve()),
             validation_normalization='training statistics',primary_checkpoint='last.pt',
             optimizer='AdamW',betas=[.9,.999],eps=6.25e-10,weight_decay=0.,clip_grad=1.,drop_last=True,
             device=device,torch_version=str(torch.__version__),provenance=provenance)
    (output/'run_config.json').write_text(json.dumps(run,indent=2)+'\n')
    best=float('inf')
    with (output/'history.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['epoch','lr','train_loss','val_mse']+['mse_k%d'%k for k in sae.model.nesting_list]);f.flush()
        for epoch in range(epochs):
            sae.train();total=0.;n=0;lr=optimizer.param_groups[0]['lr']
            for x in loader:
                total+=train_step(sae,x.to(device),optimizer)*len(x);n+=len(x)
            scheduler.step()
            errors=evaluate(sae,validation_loader);score=float(np.mean(errors))
            if not np.isfinite(score):raise ValueError('Nonfinite validation MSE')
            writer.writerow([epoch+1,lr,total/n,score]+errors);f.flush()
            checkpoint=dict(model={k:v.detach().cpu() for k,v in sae.model.state_dict().items()},
                            model_config=config,mean_center=sae.mean.detach().cpu(),scaling_factor=scale,
                            target_norm=target,epoch=epoch+1,val_mse=score,run_config=run)
            temporary=output/'last.pt.tmp';torch.save(checkpoint,temporary);temporary.replace(output/'last.pt')
            if score<best:
                best=score;temporary=output/'best.pt.tmp';torch.save(checkpoint,temporary);temporary.replace(output/'best.pt')
            print(f'epoch={epoch+1} lr={lr:.7g} train_loss={total/n:.6g} val_mse={score:.6g}',flush=True)
    (output/'COMPLETE.json').write_text(json.dumps(dict(epochs=epochs,primary_checkpoint='last.pt',best_val_mse=best),indent=2)+'\n')
    return sae


def main():
    p=argparse.ArgumentParser()
    for name in ['train','val','model-config','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=4096)
    p.add_argument('--device',default='cuda');p.add_argument('--provenance')
    a=p.parse_args()
    train(a.train,a.val,json.loads(Path(a.model_config).read_text()),a.output,a.epochs,a.batch_size,a.device,
          json.loads(Path(a.provenance).read_text()) if a.provenance else None)

if __name__=='__main__':main()
