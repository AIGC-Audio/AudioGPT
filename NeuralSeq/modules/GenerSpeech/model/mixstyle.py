from modules.commons.common_layers import *
import random


class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, hidden_size=256):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self._activated = True
        self.hidden_size = hidden_size
        self.affine_layer = LinearNorm(
            hidden_size,
            2 * hidden_size, # For both b (bias) g (gain)
        )

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})'

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x, spk_embed):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu, sig = torch.mean(x, dim=-1, keepdim=True), torch.std(x, dim=-1, keepdim=True)
        x_normed = (x - mu) / (sig + 1e-6)  # [B, T, H_m]

        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        # Get Bias and Gain
        mu1, sig1 = torch.split(self.affine_layer(spk_embed), self.hidden_size, dim=-1)  # [B, 1, 2 * H_m] --> 2 * [B, 1, H_m]

        # MixStyle
        perm = torch.randperm(B)
        mu2, sig2 = mu1[perm], sig1[perm]

        mu_mix = mu1*lmda + mu2 * (1-lmda)
        sig_mix = sig1*lmda + sig2 * (1-lmda)

        # Perform Scailing and Shifting
        return sig_mix * x_normed + mu_mix # [B, T, H_m]
