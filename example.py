import torch
import torch.nn as nn
from torchvision import transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt

import random
from Swin_pytorch import SwinTransformer
from Visualization import AttentionMap, img2tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SwinTransformer(img_size = 224,
                        patch_size = 4,
                        in_chans = 3,
                        num_classes = 2,
                        embed_dim = 96,
                        depths = [2, 2, 6, 2],
                        num_heads = [3, 6, 12, 24],
                        window_size = 7,
                        mlp_ratio=4., 
                        qkv_bias=True, 
                        qk_scale=None,
                        drop_rate=0., 
                        attn_drop_rate=0., 
                        drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm
                        ).to(device)

img = cv2.imread('./1.jpg')
test_input, img = img2tensor(img)
#test_input = torch.randn((4, 3, 224, 224))
test_output, attention_maps = model(test_input)

print("Input image size : ", test_input.shape)
print("Output image size : ", test_output.shape)        # (B, num_classes)
print("Output Attention Map : ", attention_maps.shape)  # (B, num_heads[-1], window_size**2, window_size**2)

AttentionMap(attention_maps, image = img)