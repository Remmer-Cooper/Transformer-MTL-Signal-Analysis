import torch.nn as nn

class MTLModel(nn.Module):
    def __init__(self):
        super(MTLModel, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),  # [B, 8, 17, 17]
            nn.Dropout(0.25)
        )
        

        self.mod_head = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1),  # [B, 4, 17, 17]
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
            nn.Conv2d(8, 4, kernel_size=3, padding=1),  # [B, 4, 17, 17]
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

    def forward(self, x):
        x = self.shared_layers(x)
        mod_output = self.mod_head(x)
        sig_output = self.sig_head(x)
        return mod_output, sig_output