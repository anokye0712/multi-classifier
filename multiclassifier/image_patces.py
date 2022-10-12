import torch
import patchify
import numpy as np 

def patches(test_data, w, h, ch, step=1):
    patch = patchify.patchify(np.array(test_data), (w, h, ch), step=step)
    patches = patch
    patches = torch.from_numpy(patches).squeeze(2)
    return patches