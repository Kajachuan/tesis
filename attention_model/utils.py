import math
import torch
import torch.nn.functional as F

def pos_encoding(embed_dim, length, device):
    pos_enc = torch.zeros(length, embed_dim, device=device)
    pos = torch.arange(0, length).unsqueeze(1)
    div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pos_enc[:, 0::2] = torch.sin(pos * div)
    pos_enc[:, 1::2] = torch.cos(pos * div)
    return pos_enc.unsqueeze(0)

def center_trim(tensor, size):
    delta = tensor.size(-1) - size
    if delta:
        l_dif = delta // 2
        r_dif = delta - l_dif
        tensor = tensor[..., l_dif:-r_dif]
    return tensor

def pad_tensor(tensor, size):
    delta = size - tensor.size(-1)
    if delta:
        l_dif = delta // 2
        r_dif = delta - l_dif
        tensor = F.pad(tensor, (l_dif, r_dif))
    return tensor