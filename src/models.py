import torch
import torch.nn as nn

# ==========================================
# 1. Time Embedding & Modulation Components.
# ==========================================

class TimeEmbedding(nn.Module):
    """
    Embeds a scalar time-step (dt) into a high-dimensional vector.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # MLP to map scalar dt to a vector
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        # t is expected to be (Batch,) or (Batch, 1)
        return self.mlp(t.view(-1, 1))

class ModulatedTransformerBlock(nn.Module):
    """
    Core Component: Transformer Block modulated by dt via AdaLN (Adaptive Layer Norm).
    """
    def __init__(self, dim, num_heads, time_emb_dim, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        # 1. Multi-Head Self-Attention.
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # 2. Feed Forward Network (MLP).
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

        # 3. AdaLN Modulation Layers.
        # Outputs 6 parameters: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 6 * dim, bias=True)
        )

        # Base LayerNorm (elementwise_affine=False is crucial for AdaLN).
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, time_emb):
        """
        Args:
            x: Input tensor (Batch, Num_Tokens, Dim)
            time_emb: Time embedding vector (Batch, Time_Dim)
        """
        # Generate modulation parameters.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(time_emb).chunk(6, dim=1)

        # --- Modulated Attention Block ---
        # Apply normalization and modulation (shift and scale).
        x_norm1 = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)

        # Apply gating and residual connection.
        x = x + gate_msa.unsqueeze(1) * attn_out

        # --- Modulated FFN Block ---
        # Apply normalization and modulation.
        x_norm2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        ffn_out = self.ffn(x_norm2)
        
        # Apply gating and residual connection.
        x = x + gate_mlp.unsqueeze(1) * ffn_out
        
        return x

# ==========================================
# 2. The Bottleneck Bridge.
# ==========================================

class TransformerBottleneck3D(nn.Module):
    """
    Acts as the bridge between the Encoder and Decoder.
    1. Flattens 3D feature maps to tokens.
    2. Processes tokens via Time-Modulated Transformer Blocks.
    3. Reshapes tokens back to 3D feature maps.
    """
    def __init__(self, in_channels, dim, num_layers, num_heads, time_emb_dim, spatial_size):
        """
        Args:
            spatial_size: Tuple (D, H, W) of the feature map at the bottleneck (e.g., (8, 8, 8)).
        """
        super().__init__()

        self.in_channels = in_channels
        self.dim = dim
        self.spatial_size = spatial_size
        self.num_tokens = spatial_size[0] * spatial_size[1] * spatial_size[2]

        # Projection: CNN Channels -> Transformer Embedding Dimension.
        self.to_tokens = nn.Linear(in_channels, dim)
        
        # Learnable Position Embedding.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, dim))

        # Stack of Modulated Transformer Blocks.
        self.blocks = nn.ModuleList([
            ModulatedTransformerBlock(dim, num_heads, time_emb_dim)
            for _ in range(num_layers)
        ])

        # Projection: Transformer Embedding Dimension -> CNN Channels.
        self.to_grid = nn.Linear(dim, in_channels)

        # Initialize weights.
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize pos_embed with small noise.
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Zero-Initialize the last layer of AdaLN (gate, shift, scale) for stable start.
        for block in self.blocks:
            # Set weight and bias of the last Linear layer in adaLN_modulation to zero.
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, time_emb):
        # Input x: (Batch, Channels, D, H, W)
        B, C, D, H, W = x.shape
        
        # 1. Flatten.
        x_flat = x.flatten(2).transpose(1, 2) 
        
        # 2. Projection & Pos Embed.
        x_tokens = self.to_tokens(x_flat)
        x_tokens = x_tokens + self.pos_embed

        # 3. Modulated Transformer Blocks.
        for block in self.blocks:
            x_tokens = block(x_tokens, time_emb)

        # 4. Unpatchify / Reshape.
        x_out = self.to_grid(x_tokens) 
        x_out = x_out.transpose(1, 2).reshape(B, C, D, H, W)
        
        return x_out

# ==========================================
# 3. Standard U-Net Components.
# ==========================================

class DoubleConv3D(nn.Module):
    """
    (Conv3D -> GroupNorm -> SiLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """
    Downscaling with MaxPool then DoubleConv.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """
    Upscaling then DoubleConv, handling skip connections.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Upsample the incoming feature map (in_channels -> in_channels).
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # The conv layer must accept the concatenated channels (in_channels/2 + skip_channels).
        self.conv = DoubleConv3D(in_channels // 2 + skip_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ==========================================
# 4. Full Hybrid U-Net Architecture.
# ==========================================

class HybridUNet(nn.Module):
    """
    Hybrid TransU-Net for 64x64x64 input.
    - Encoder/Decoder: 3D CNN.
    - Bottleneck: Time-Modulated Transformer (8x8x8 resolution).
    """
    def __init__(self, in_channels=4, out_channels=4, base_dim=32, time_emb_dim=128):
        super().__init__()
        
        # Time Embedding Module
        self.time_mlp = TimeEmbedding(time_emb_dim)
        
        # --- Encoder (Conv3D) ---
        # Input size: 64x64x64
        self.inc = DoubleConv3D(in_channels, base_dim)
        self.down1 = Down3D(base_dim, base_dim * 2)
        self.down2 = Down3D(base_dim * 2, base_dim * 4)
        self.down3 = Down3D(base_dim * 4, base_dim * 8)

        # --- Bottleneck (Time-Modulated Transformer) ---
        self.bottleneck = TransformerBottleneck3D(
            in_channels=base_dim * 8,
            dim=512,
            num_layers=4,
            num_heads=8,
            time_emb_dim=time_emb_dim,
            spatial_size=(8, 8, 8)
        )

        # --- Decoder (Conv3D) ---
        self.up1 = Up3D(in_channels=base_dim * 8, skip_channels=base_dim * 4, out_channels=base_dim * 4)
        self.up2 = Up3D(in_channels=base_dim * 4, skip_channels=base_dim * 2, out_channels=base_dim * 2)
        self.up3 = Up3D(in_channels=base_dim * 2, skip_channels=base_dim, out_channels=base_dim)
        
        # Final Output Projection.
        self.outc = nn.Conv3d(base_dim, out_channels, kernel_size=1)

    def forward(self, x, dt):
        """
        Args:
            x: Input feature map (Batch, Channels, 64, 64, 64)
            dt: Time step scalar (Batch,)
        """
        # 1. Generate Time Embedding.
        t_emb = self.time_mlp(dt)
        
        # 2. Encoder (Downsampling).
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 3. Bottleneck (Transformer with Time Modulation).
        x_mid = self.bottleneck(x4, t_emb)
        
        # 4. Decoder (Upsampling with Skip Connections)
        x_up1 = self.up1(x_mid, x3)
        x_up2 = self.up2(x_up1, x2)
        x_up3 = self.up3(x_up2, x1)
        
        # 5. Output.
        return self.outc(x_up3)
