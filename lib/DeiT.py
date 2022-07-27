# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import numpy as np


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DeiT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

    def forward(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)
        pe = self.pos_embed

        x = x + pe
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = DeiT(
        patch_size=16, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load('pretrained/deit_small_patch16_224-cd65a155.pth')
        model.load_state_dict(ckpt['model'], strict=False)
    
    # Positional embedding will be different
    # If you want to pre-trained, you resize (interpolate)
    # Interpolate is applied on image, steps:  # (embed_size, height, width)
    #  1. Change the shape of pos_embed into image shape
    #    Shape of pos_embed: (height * width, embed_size), Shape of img (after patch embedding): (embed_size, height, width)
    #    New shape of pos_embed (same as img): (embed_dim, height, width)
    #  2. Resize the positional embedding using interpolate method
    pe = model.pos_embed[:, 1:, :].detach() # Remove cls token
    pe = pe.transpose(-1, -2) # Shape of pe: (batch_size, embed_dim, height * width)
    
    # Shape of pe: (batch_size, embed_dim, height, width)
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))
    
    # Interpolate will function only last two dimension, model: cubulic
    # Shape of pe: (batch_size, embed_dim, new_height, new_width)
    pe = F.interpolate(pe, size=(22, 22), mode='bilinear', align_corners=True)
    
    # Shape of pe: (batch_size, embed_dim, new_height * new_width)
    pe = pe.flatten(2)
    
    # Shape of pe: (batch_size, new_height * new_width, embed_dim)
    pe = pe.transpose(-1, -2)
    
    # Note: the parameter here are freezed
    model.pos_embed = nn.Parameter(pe)
    
    # Remove the head parameters for downstream task
    model.head = nn.Identity()
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = DeiT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load('pretrained/deit_base_patch16_224-b5f2ef4d.pth')
        model.load_state_dict(ckpt['model'], strict=False)
    
    pe = model.pos_embed[:, 1:, :].detach()
    pe = pe.transpose(-1, -2)
    pe = pe.view(pe.shape[0], pe.shape[1], int(np.sqrt(pe.shape[2])), int(np.sqrt(pe.shape[2])))
    pe = F.interpolate(pe, size=(22, 22), mode='bilinear', align_corners=True)
    pe = pe.flatten(2)
    pe = pe.transpose(-1, -2)
    model.pos_embed = nn.Parameter(pe)
    model.head = nn.Identity()
    return model
