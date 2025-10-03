import torch
import torch.nn as nn
from einops import rearrange


class VisionTransformerMTL(nn.Module):
    def __init__(self, Csh=8, Cm=4, Fm=256, Cs=4, Fs=256):
        super().__init__()

    
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(3, Csh, kernel_size=3, padding=1), 
            nn.BatchNorm2d(Csh),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1)  # Output size [B, Csh, 17, 17]
        )

        # Position embedding and transformer encoder
        self.pos_embedding = nn.Parameter(torch.randn(1, 17 * 17, Csh))  
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=Csh,
                nhead=4,
                dim_feedforward=32,
                batch_first=True,
                dropout=0.2
            ),
            num_layers=2
        )

        # Modulation classification head
        self.mod_head = nn.Sequential(
            nn.Conv2d(Csh, Cm, kernel_size=3, padding=1),
            nn.BatchNorm2d(Cm),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(Cm * 17 * 17, Fm),
            nn.BatchNorm1d(Fm),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Fm, 6)  
        )

        # Signal classification head
        self.sig_head = nn.Sequential(
            nn.Conv2d(Csh, Cs, kernel_size=3, padding=1),
            nn.BatchNorm2d(Cs),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(Cs * 17 * 17, Fs),
            nn.BatchNorm1d(Fs),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Fs, 8)  
        )

    def forward(self, x):  # Input shape: [B, 3, 16, 16]
        x = self.shared_cnn(x)  # [B, Csh, 17, 17]
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, 289, Csh]
        x = x + self.pos_embedding[:, :x.size(1), :]  # Add position embedding
        x = self.transformer(x)  # [B, 289, Csh]
        x = rearrange(x, 'b (h w) c -> b c h w', h=17)  # [B, Csh, 17, 17]
        return self.mod_head(x), self.sig_head(x)
