import torch
import torch.nn as nn


class FreqEmbedder(nn.Module):
    # Generate Positional Encoding in NeRF (section 5.1)
    def __init__(self, in_dim=3, multi_res=10, use_log_bands=True, include_input=True):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = multi_res
        self.max_freq_log2 = multi_res - 1
        self.use_log_bands = use_log_bands
        self.periodic_fns = [torch.sin, torch.cos]
        self.include_input = include_input

        self.embed_fns = None
        self.out_dim = None
        self.num_embed_fns = None
        self.create_embedding_fn()

    def create_embedding_fn(self):
        self.embed_fns = []
        self.out_dim = self.num_freqs * len(self.periodic_fns) * self.in_dim
        if self.include_input:
            self.embed_fns.append(lambda x: x)
            self.out_dim += self.in_dim

        if self.use_log_bands:
            freq_bands = 2. ** torch.linspace(0., self.max_freq_log2 , steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.**0, 2. ** self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                self.embed_fns.append(lambda x, p_fn=p_fn,freq=freq: p_fn(x * freq)) # e.g., torch.cos(x*(2^5))
        self.num_embed_fns = len(self.embed_fns)

    def forward(self, x):
        """
        x: [..., in_dim]; xyz or view direction
        embedding: [..., out_dim]; the corresponding frequency encoding
        """
        embed_lst = [embed_fn(x) for embed_fn in self.embed_fns] # [list of [..., in_dim]]
        embedding = torch.cat(embed_lst, dim=-1) # [..., out_dim]
        return embedding
