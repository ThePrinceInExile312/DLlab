import torch
import torch.nn.functional as F

def init_multihead_params(embed_size, heads):
    head_dim = embed_size // heads
    assert embed_size % heads == 0, "Embedding size must be divisible by number of heads"
    
    # Define weight matrices for Q, K, V and output
    W_q = torch.nn.Linear(embed_size, embed_size)
    W_k = torch.nn.Linear(embed_size, embed_size)
    W_v = torch.nn.Linear(embed_size, embed_size)
    W_o = torch.nn.Linear(embed_size, embed_size)
    
    return {
        'W_q': W_q,
        'W_k': W_k,
        'W_v': W_v,
        'W_o': W_o,
        'embed_size': embed_size,
        'heads': heads,
        'head_dim': head_dim
    }

def multihead_attention(value, key, query, mask, params):
    N = query.size(0)
    value_len, key_len, query_len = value.size(1), key.size(1), query.size(1)
    H, D = params['heads'], params['head_dim']

    # Project inputs
    Q = params['W_q'](query).view(N, query_len, H, D).transpose(1, 2)
    K = params['W_k'](key).view(N, key_len, H, D).transpose(1, 2)
    V = params['W_v'](value).view(N, value_len, H, D).transpose(1, 2)

    # Scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-1e20'))

    attention = F.softmax(scores, dim=-1)
    out = torch.matmul(attention, V)

    # Concatenate heads
    out = out.transpose(1, 2).contiguous().view(N, query_len, H * D)
    out = params['W_o'](out)

    return out

# Example usage
embed_size = 128
heads = 8
params = init_multihead_params(embed_size, heads)

x = torch.rand(2, 10, embed_size)  # (batch, seq_len, embed_size)
output = multihead_attention(x, x, x, mask=None, params=params)
print(output.shape)


import torch 
import torch.nn as nn

x = torch.rand(2, 10, 128)  
mha = nn.MultiheadAttention(128, 8,True)
out, _ = mha(x, x, x)  
print(out.shape)


