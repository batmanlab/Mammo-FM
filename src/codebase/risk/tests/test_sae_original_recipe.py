import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from mammo_sae.core import SpatialSAE
from mammo_sae.impl.sae import MatryoshkaAutoencoder
from mammo_sae.impl.utils import LinearDecayLR
from mammo_sae.impl.loss import SAELoss
from mammo_sae.training import statistics
from mammo_sae.original_training import train_step,train

CONFIG=dict(n_inputs=3,n_latents=8,activation='TopKReLU',tied=False,normalize=False,
            latent_soft_cap=0.,nesting_list=[2,4,8],relative_importance=[1.,1.,1.])

class RecipeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(12345)

    def test_update_matches_original_training_equations(self):
        actual=SpatialSAE(MatryoshkaAutoencoder(**CONFIG),torch.zeros(3),1.7)
        reference=copy.deepcopy(actual)
        def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),eps=6.25e-10,weight_decay=0.)
        oa,ob=optimizer(actual.model),optimizer(reference.model)
        x=torch.randn(16,3);loss_fn=SAELoss(reconstruction_loss='mse',sparse_weight=0.)
        recons,z,all_recons,all_z=reference.model(x*1.7)
        loss=sum(loss_fn(r,x*1.7,latent)[1] for r,latent in zip(recons,z))/len(recons)
        ob.zero_grad();loss.backward();reference.model.scale_to_unit_norm();reference.model.project_grads_decode()
        torch.nn.utils.clip_grad_norm_(reference.model.parameters(),1.);ob.step()
        observed=train_step(actual,x,oa)
        self.assertAlmostEqual(observed,float(loss.detach()),places=6)
        for k,v in reference.model.state_dict().items():torch.testing.assert_close(actual.model.state_dict()[k],v)

    def test_original_linear_decay_schedule(self):
        weight=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.AdamW([weight],lr=1e-4)
        schedule=LinearDecayLR(optimizer,30,.8);rates=[]
        for _ in range(30):rates.append(optimizer.param_groups[0]['lr']);optimizer.step();schedule.step()
        np.testing.assert_allclose(rates[:25],1e-4)
        self.assertAlmostEqual(rates[-1],1e-4/6)
        self.assertEqual(optimizer.param_groups[0]['lr'],0.)

    def test_saved_model_uses_training_normalization_and_final_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);values=np.arange(144,dtype=np.float32).reshape(48,3)/10+1
            np.save(p/'train.npy',values[:32]);np.save(p/'val.npy',values[32:])
            train(p/'train.npy',p/'val.npy',CONFIG,p/'fit',epochs=2,batch_size=8,device='cpu')
            model=SpatialSAE.load(p/'fit/last.pt');_,scale,_=statistics(values[:32])
            self.assertAlmostEqual(float(model.scale),scale,places=6)
            self.assertTrue(torch.isfinite(model(values[32:])).all())
            saved=torch.load(p/'fit/last.pt',weights_only=False)
            self.assertEqual(saved['epoch'],2)
            self.assertEqual(saved['run_config']['primary_checkpoint'],'last.pt')
            self.assertTrue((p/'fit/COMPLETE.json').exists())

if __name__=='__main__':unittest.main()
