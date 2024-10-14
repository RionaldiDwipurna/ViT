import torch
from torch import nn


class MLP_Block(nn.Module):
    def __init__(self, embed_dim:int=768, mlp_size:int=3072, dropout:float=0.1):
        super().__init__()
        self.layer1 = nn.Linear(embed_dim, mlp_size)
        self.layer2 = nn.Linear(mlp_size, embed_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = nn.functional.gelu(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.dropout2(x)

        return x

        

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int=12, dropout:float=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(p=dropout)

        self.division = self.head_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, num_heads, head_dim]
        # [batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_heads, num_patches, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # [batch_size, num_heads, num_patches, num_patches]
        attn_weight = (Q @ K.transpose(-1,-2)) / self.division
        attn_weight = nn.functional.softmax(attn_weight, dim=-1)

        # [batch_size, num_heads, num_patches, head_dim]
        
        attn_weight = self.dropout_layer(attn_weight)
        attn = attn_weight @ V

        if attn.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Output size should be {(batch_size, self.num_heads, seq_len, self.head_dim)}, but it is"
                f"{attn.size()}"
            )

        attn = attn.transpose(1, 2).contiguous().view(batch_size,seq_len,self.embed_dim)

        attn = self.out_proj(attn)
        return attn

        


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim:int):
        super().__init__()
        self.MLP = MLP_Block(embed_dim=embed_dim)
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.Attention = MultiHeadAttention(embed_dim=embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        # img: N x (P^2 x C)
        # N = Height * Width / Patch^2;
        # batch_size RGB: [batch_size, Height, Width, Channel] -> [batch_size, N num of patch, Patch^2 * Channel]
        # ViT base: D = 768; image_size = (224,224), patch_size (16*16)
        # [batch_size, 224,224, 3] -> [batch_size, (224*224)/(16*16), 16*16*3] = [batch_size, 196, 7]
        # [batch_size, 14x16,14x16, 3] -> [batch_size, 14 * 14, 16 * 16 * 3]
        # [batch_size, 196, 768]
        residual = x
        x = self.layernorm_1(x)
        x = self.Attention(x)
        x += residual

        residual_2 = x
        x = self.layernorm_2(x)
        x = self.MLP(x)
        x += residual_2
        return x


