import torch.nn as nn
from torchvision import transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt

import random

def normalize_image(image):
    max_value = np.max(image)
    min_value = np.min(image)

    normalized_image = (image - min_value) / (max_value - min_value)

    return normalized_image

def AttentionMap(attention_maps, image_size = 224, N = 3, image = None) :
    B, Heads, H, W = attention_maps.shape
    
    assert int(H**(1/2)) == int(W**(1/2)), 'Attention Map size Wrong'
    window_size = int(H**(1/2))

    # Batch = 0, Random N head
    random_head = [random.randint(0, Heads - 1) for _ in range(N)]
    for i in random_head :
        n_head_attention_map = attention_maps[0, i, :, :]       # (window_size^2, window_size^2)
        fig, axs = plt.subplots(window_size, window_size)

        for idx, q in enumerate(n_head_attention_map) :
            q = q.reshape(window_size, window_size)             # q-th Query Attention Score
            q = nn.functional.interpolate(q.unsqueeze(0).unsqueeze(0), 
                scale_factor = image_size // window_size, mode = 'nearest')[0][0].detach().cpu().numpy() # (image_size, image_size)
            if image is not None :
                q = normalize_image(q)
                c0 = (image[:, :, 0] * q).astype(np.uint8)
                c1 = (image[:, :, 1] * q).astype(np.uint8)
                c2 = (image[:, :, 2] * q).astype(np.uint8)
                q = (np.stack([c0, c1, c2], axis = -1))

            axs[int(idx // window_size), int(idx % window_size)].imshow(q)
            axs[int(idx // window_size), int(idx % window_size)].axis('off')

        plt.savefig(f'./Attention_map/{i}-th_head_attention_map.png')
    
    return None

def img2tensor(img, image_size = 224):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    convert_tensor = transforms.ToTensor()
    test_input = convert_tensor(img).unsqueeze(0)

    return test_input, img