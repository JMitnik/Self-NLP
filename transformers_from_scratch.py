
#%%
# Based on http://peterbloem.nl/blog/transformers

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#%%
# Generate X
batch_size = 5
seq_len = 10
vec_dim = 5
x: torch.Tensor = torch.rand(batch_size, seq_len, vec_dim)

#%%
# SIMPLE ATTENTION

# Torch.bmm applies a batch matrix multiplication across all batches
#, thus ignoring the 0th dimensions of the matrcies.

# By multiplying the matrix with its transpose, we get the relationship between
# a vector and each other vector. The resulting matrxi has Batch x rows from the
# pespective of X, and then to all the other possible vectors.
raw_weights = torch.bmm(x, x.transpose(1, 2))

# Batch x X x all possible relations to X. All these relations need to sum up to
# 1. Thus, we perform softmax over all elements in the row
weights = F.softmax(raw_weights, dim=2)

# Now we multiply the relations from X to all other vectors by the actual rows
# of X.
y = torch.bmm(weights, x)
#%%


#%%
# Self-attention module

# For each head, we have
class SelfAttention(nn.Module):
    ''' Self-attention module.

        Notes:
            - For each head, we have a separate set of W_q, W_k, W_v
            concatenated into 3 matrices with (k, head * k) dimension.

            - We also unify the three matrices into one big tensor.
    '''
    def __init__(self, k, heads=8):
        super().__init__()
        self.k, self.heads = k, heads

        self.to_keys = nn.Linear(k, k * heads, bias=False)
        self.to_queries = nn.Linear(k, k * heads, bias=False)
        self.to_values = nn.Linear(k, k * heads, bias=False)

        self.unify = nn.Linear(k * heads, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        # Now we map each of
        queries = self.to_queries(x).view(b, t, h, k)
        keys = self.to_keys(x).view(b, t, h, k)
        values = self.to_values(x).view(b, t, h, k)


