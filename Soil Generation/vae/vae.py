import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import math

# ------------------ Self-Attention ------------------ #
class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        batch_size, sequence_length, d_embed = x.shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, d_embed)
        return self.out_proj(output)

# ------------------ VAE Blocks ------------------ #
class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor):
        residue = x
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        return x + residue

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_layer = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)

# ------------------ VAE Encoder ------------------ #
class VAE_Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.Conv1d(128, 128, kernel_size=3, stride=2),
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            nn.Conv1d(256, 256, kernel_size=3, stride=2),
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),
            nn.SiLU(),

            nn.Conv1d(512, 8, kernel_size=3, padding=1),
            nn.Conv1d(8, 8, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor = None):
        # x: (B, L, 1) -> (B, 1, L)
        x = x.transpose(1, 2)
        orig_length = x.shape[-1]

        # 计算补齐长度
        pad_len = (8 - orig_length % 8) % 8
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))  # 只右侧补0

        for module in self.encoder:
            if getattr(module, 'stride', None) == (2,):
                x = F.pad(x, (0, 1))  # 适配非偶数长度
            x = module(x)

        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        std = torch.exp(log_var * 0.5)

        if noise is None:
            noise = torch.randn_like(std)

        z = mean + std * noise
        return z * 0.18215, mean, log_var, orig_length, pad_len


# ------------------ VAE Decoder ------------------ #
class VAE_Decoder(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(4, 4, kernel_size=1),
            nn.Conv1d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            nn.Upsample(scale_factor=2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),

            nn.Conv1d(128, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, z: torch.Tensor, orig_length: int, pad_len: int):
        # z: (B, 4, L/8)
        z = z / 0.18215

        for module in self.decoder:
            z = module(z)

        x = z.transpose(1, 2)  # (B, C, L) -> (B, L, C)

        if pad_len > 0:
            x = x[:, :orig_length, :]

        return x


# ------------------ Loss Function ------------------ #
def loss_function(recons, x, mu, log_var, kld_weight=1.0):
    recons_loss = F.mse_loss(recons, x, reduction='sum') / x.size(0)
    q_z_x = Normal(mu, torch.exp(0.5 * log_var))
    p_z = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
    kld_loss = kl_divergence(q_z_x, p_z).sum(1).mean()
    return {
        'loss': recons_loss + kld_weight * kld_loss,
        'mse': recons_loss.detach(),
        'KLD': kld_loss.detach()
    }



# def loss_function(recons, x, mu, log_var, kld_weight=1) -> dict:
#     """
#     Computes the VAE loss function.
#     KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
#     :para recons: reconstruction vector
#     :para x: original vector
#     :para mu: mean of latent gaussian distribution
#     :log_var: log of latent gaussian distribution variance
#     :kld_weight: weight of kl-divergence term
#     """
#     # recons, x: (B, L, 12) -> number, batch wise average
#     recons_loss = F.mse_loss(recons, x, reduction='sum').div(x.size(0))
#
#     # (old) mu, log_var: (B, 4, L/8) -> number
#     # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 2), dim=1).sum()
#
#     # q(z|x): distribution learned by encoder
#     q_z_x = Normal(mu, log_var.mul(.5).exp())
#     # p(z): prior of z, intended to be standard Gaussian
#     p_z = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
#     # kld_loss: batch wise average
#     kld_loss = kl_divergence(q_z_x, p_z).sum(1).mean()
#
#     loss = recons_loss + kld_weight * kld_loss
#
#     return {'loss': loss, 'mse':recons_loss.detach(), 'KLD':kld_loss.detach()}
if __name__ == '__main__':
    x = torch.randn(8, 2100, 1)  # 任意长度都可，不一定要2100

    encoder = VAE_Encoder()
    decoder = VAE_Decoder()

    z, mu, log_var, orig_len, pad_len = encoder(x)
    x_recon = decoder(z, orig_len, pad_len)

    print(x.shape, x_recon.shape)  # 应该都是 [8, 2100, 1]

