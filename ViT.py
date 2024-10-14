import torch
from torch import nn
from transformer import TransformerEncoder

class ViT(nn.Module):
    def __init__(self, 
        in_channels:int=3,
        embed_dims:int=768,
        patch_size:int=16,
        image_size:int=224,
        num_transformer_layer:int=12,
        num_classes:int =1000,
        ):
        super().__init__()
        assert image_size % patch_size == 0, "image must be divisible by 0"

        self.embedding = PatchEmbedding(in_channels=in_channels, embed_dims=embed_dims, patch_size=patch_size, image_size=image_size)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(embed_dim=embed_dims) for _ in range(num_transformer_layer)]
        )
        self.final_head = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(in_features=embed_dims,out_features=num_classes)
        )
    def forward(self, x):
        # (batch, 3 , 224, 224)
        batch = x.shape[0]

        # (batch, 197, 768)
        x = self.embedding(x)

        x = self.transformer_encoder(x)

        x = self.final_head(x[:,0])

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels:int=3, embed_dims:int=768, patch_size:int=16, image_size:int=224, embedding_dropout:float=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=embed_dims,kernel_size=patch_size,stride=patch_size,padding=0)
        self.num_patches = (image_size//patch_size) ** 2
        self.class_embedding = nn.Parameter(torch.randn(1,1,embed_dims))
        self.positional_embedding = nn.Parameter(torch.randn(1,self.num_patches + 1,embed_dims))
        self.dropout = nn.Dropout(p=embedding_dropout)

    def forward(self, x):
        # (batch, 3 , 224, 224) -> (batch, 768, 14,14)
        batch, _, _, _ = x.size()

        x = self.conv1(x)

        # (batch, 768, 192)
        x = x.view(batch, 768, -1)

        x = x.permute(0, 2, 1)

        # (batch, 1, 768)
        class_token = self.class_embedding.expand(batch, -1,-1)

        x = torch.cat((class_token, x), dim=1)

        x += self.positional_embedding

        x = self.dropout(x)


        return x
