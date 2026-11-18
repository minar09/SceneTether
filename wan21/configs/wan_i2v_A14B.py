# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Ported verbatim from the official Wan2.2 source vendored at
# wan22_turbo/wan22/configs/wan_i2v_A14B.py, with the model_type note
# added. Used as the DMD/Re-DMD TEACHER (real score); the student stays the cfpp 2-step
# causal checkpoint.
#
# Why this teacher pairs with the cfpp student at all: A14B ships Wan2.1_VAE.pth with
# vae_stride (4, 8, 8) and out_dim 16 -- the SAME latent space as the cfpp student. The
# 48-channel / (4,16,16) VAE belongs to TI2V-5B only, which is what made that pairing
# painful. Teacher and student differ in size (5120x40 vs 1536x30) and in conditioning
# interface, but they score in one latent space, which is what DMD requires.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

# ------------------------ Wan I2V A14B ------------------------#

i2v_A14B = EasyDict(__name__='Config: Wan I2V A14B')
i2v_A14B.update(wan_shared_cfg)

i2v_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
i2v_A14B.t5_tokenizer = 'google/umt5-xxl'

# NO clip entries: Wan2.2-I2V-A14B has no CLIP branch. See 'i2v22' in wan/modules/model.py.
i2v_A14B.model_type = 'i2v22'
i2v_A14B.in_dim = 36
i2v_A14B.out_dim = 16

# vae
i2v_A14B.vae_checkpoint = 'Wan2.1_VAE.pth'
i2v_A14B.vae_stride = (4, 8, 8)

# transformer
i2v_A14B.patch_size = (1, 2, 2)
i2v_A14B.dim = 5120
i2v_A14B.ffn_dim = 13824
i2v_A14B.freq_dim = 256
i2v_A14B.num_heads = 40
i2v_A14B.num_layers = 40
i2v_A14B.window_size = (-1, -1)
i2v_A14B.qk_norm = True
i2v_A14B.cross_attn_norm = True
i2v_A14B.eps = 1e-6

# MoE: two 14B experts, ~27B total but 14B active per step. The high-noise expert handles
# the early layout stage, the low-noise expert the detail stage; the switch is at the
# timestep where SNR falls to half of SNR_min.
i2v_A14B.low_noise_checkpoint = 'low_noise_model'
i2v_A14B.high_noise_checkpoint = 'high_noise_model'

# inference
i2v_A14B.sample_shift = 5.0
i2v_A14B.sample_steps = 40
i2v_A14B.boundary = 0.900
i2v_A14B.sample_guide_scale = (3.5, 3.5)  # low noise, high noise
