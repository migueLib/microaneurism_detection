import torch
import PIL
import numpy as np
from PIL import Image
from tqdm import tqdm

def cluster(x):
    out = []
    # Iterate each row
    for i in tqdm(range(len(x))):
        if i-1 < 0:
            out.append(torch.mean(x[i:i+2], dim=0))

        elif i+1 < len(x):
            frame = torch.stack([x[i-1], x[i+1]]).to("cuda")
            diffs = torch.stack([ abs(x[i]-x[i-1]), abs(x[i]-x[i+1])])
            index = torch.min(diffs, dim=0).indices[:,0]
            minvl = torch.stack([frame[j,i,:] for i,j in enumerate(index)])
            out.append(torch.mean(torch.stack([minvl, x[i]]), dim=0))

        else:
            out.append(torch.mean(x[i-1:i+1], dim=0))
    # Stack 
    out = torch.stack(out, dim=0)
    
    return out.type(torch.uint8)


def cluster_transpose(x):
    x = torch.transpose(x, 0, 1)
    x = cluster(x)
    return torch.transpose(x, 0, 1)


def nearest_neighbour_mean(x):
    x = torch.cuda.HalfTensor(x)

    # Run clustering on data
    meaned_v = cluster(x)
    meaned_h = cluster_transpose(x)
    
    # Differrence between original and meaned
    diff_v = x - meaned_v
    diff_h = x - meaned_h
    diff = abs(diff_h) > abs(diff_v)
    
    # Propate flags
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            diff[i,j,:]=diff[i,j,0]
    
    diff2cal = torch.where(diff, meaned_v,meaned_h)
    
    return Image.fromarray(diff2cal.to("cpu").numpy().astype(np.uint8))