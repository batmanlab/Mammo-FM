"""Spatial extraction from the exact image encoder embedded in a risk checkpoint."""
import numpy as np
import torch
from PIL import Image
from breastclip.model.modules import EfficientNet
from mammo_sae.core import load_checkpoint
from mayo_mammofm_risk.model import compatible_model_state


def risk_checkpoint_spatial(checkpoint, device='cpu', mean=0.3089279, std=0.25053555408335154):
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError('Invalid normalization parameters')
    saved=load_checkpoint(checkpoint)
    if saved.get('stage') not in (None, 'stage2'):
        raise ValueError('Expected a complete Stage-2 risk checkpoint')
    state=compatible_model_state(saved['model'])
    weights={k[len('image_encoder.'):]:v for k,v in state.items() if k.startswith('image_encoder.')}
    encoder=EfficientNet.from_name('efficientnet-b5',num_classes=1)
    encoder.load_state_dict(weights,strict=True)
    encoder.to(device).eval().requires_grad_(False)
    del saved,weights
    @torch.no_grad()
    def extract(row):
        with Image.open(row['image_path']) as image:
            pixels=np.asarray(image.convert('RGB'),dtype=np.float32)
        pixels-=pixels.min()
        maximum=pixels.max()
        if maximum>0:pixels/=maximum
        x=torch.from_numpy((pixels-mean)/std).permute(2,0,1)[None].to(device)
        return encoder.extract_features(x)[0]
    return extract
