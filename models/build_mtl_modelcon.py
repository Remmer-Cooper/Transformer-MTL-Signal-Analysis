import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F  

class VisionTransformerMTL(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1)  # [B,8,17,17]
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 17*17, 8))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=8,
                nhead=4,
                dim_feedforward=32,
                batch_first=True,
                dropout=0.2
            ),
            num_layers=2
        )
        
        self.mod_head = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(4 * 17 * 17, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)
        )
        self.sig_head = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(4 * 17 * 17, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 8)
        )


    def forward(self, x):  #[B,3,16,16]
        x = self.shared_cnn(x)  # [B,8,17,17]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=17)
        return self.mod_head(x), self.sig_head(x)
    


