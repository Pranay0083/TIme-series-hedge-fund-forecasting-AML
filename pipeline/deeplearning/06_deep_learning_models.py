import torch
import torch.nn as nn

class BottleneckAutoencoderMLP(nn.Module):
    """
    Bottleneck Autoencoder Multi-Layer Perceptron architecture for tabular financial series.
    Compresses high-collinearity inputs, concatenates with original strong features,
    and applies heavy Swish activation mapping to targets.
    """
    def __init__(self, input_dim, keep_original_dim, latent_dim=32, drop_rate=0.3):
        super(BottleneckAutoencoderMLP, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(), 
            nn.Dropout(drop_rate),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.SiLU()
        )
        
        combined_dim = latent_dim + keep_original_dim 
        
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(drop_rate + 0.1), 
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 1) 
        )
        
    def forward(self, x_all, x_critical):
        """
        x_all: The full noisy feature matrix [batch, input_dim]
        x_critical: Specific engineered strong features [batch, keep_original_dim]
        """
        latent = self.encoder(x_all)
        combined = torch.cat([latent, x_critical], dim=1)
        output = self.mlp(combined)
        return output
