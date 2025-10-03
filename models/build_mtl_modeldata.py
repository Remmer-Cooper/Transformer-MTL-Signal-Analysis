import torch
import torch.nn as nn
from einops import rearrange

class VisionTransformerMTL(nn.Module):    
    def __init__(self, config="(8,4,256,4,256)"):
        super(VisionTransformerMTL, self).__init__()
        
        # (Csh, Cm, Fm, Cs, Fs)
        if isinstance(config, str):
            config = config.strip("()").split(",")
            config = [int(x.strip()) for x in config]
        
        Csh, Cm, Fm, Cs, Fs = config
        
        # --------------------------
        # Shared CNN Feature Extraction
        # --------------------------
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(3, Csh, kernel_size=3, padding=1),  # Input channels 3 for I+Q+Constellation
            nn.BatchNorm2d(Csh),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1)  # Output size [B, Csh, 17, 17]
        )
        
        # --------------------------
        # Position Embedding and Transformer Encoder
        # --------------------------
        self.pos_embedding = nn.Parameter(torch.randn(1, 17*17, Csh))
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
        
        # --------------------------
        # Modulation Classification Head
        # --------------------------
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
        
        # --------------------------
        # Signal Classification Head
        # --------------------------
        self.sig_head = nn.Sequential(
            # 信号分支卷积层
            nn.Conv2d(Csh, Cs, kernel_size=3, padding=1),  
            nn.BatchNorm2d(Cs),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3),
            # 全连接层
            nn.Linear(Cs * 17 * 17, Fs),  
            nn.BatchNorm1d(Fs),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Fs, 8)  
        )


    def forward(self, x):  # Input shape: [B, 3, 16, 16]
    
        x = self.shared_cnn(x)  # [B,8,17,17]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=17)
        return self.mod_head(x), self.sig_head(x)
