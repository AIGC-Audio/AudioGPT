import math
import torch
from torch import nn
from torch.nn import functional as F


class GMMLogLoss(nn.Module):
    ''' compute the GMM loss between model output and the groundtruth data.
    Args:
        ncenter: numbers of gaussian distribution
        ndim: dimension of each gaussian distribution
        sigma_min:  current we do not use it.
    '''
    def __init__(self, ncenter, ndim, sigma_min=0.03):
        super(GMMLogLoss,self).__init__()
        self.ncenter = ncenter
        self.ndim = ndim
        self.sigma_min = sigma_min 
    
    def forward(self, output, target):
        '''
        Args:
            output: [b, T, ncenter + ncenter * ndim * 2]:
                [:, :,  : ncenter] shows each gaussian probability 
                [:, :, ncenter : ncenter + ndim * ncenter] shows the average values of each dimension of each gaussian 
                [: ,:, ncenter + ndim * ncenter : ncenter + ndim * 2 * ncenter] show the negative log sigma of each dimension of each gaussian 
            target: [b, T, ndim], the ground truth target landmark data is shown here 
        To maximize the log-likelihood equals to minimize the negative log-likelihood. 
        NOTE: It is unstable to directly compute the log results of sigma, e.g. ln(-0.1) as we need to clip the sigma results 
        into positive. Hence here we predict the negative log sigma results to avoid numerical instablility, which mean:
            `` sigma = 1/exp(predict), predict = -ln(sigma) ``
        Also, it will be just the 'B' term below! 
        Currently we only implement single gaussian distribution, hence the first values of pred are meaningless.
        For single gaussian distribution:
            L(mu, sigma) = -n/2 * ln(2pi * sigma^2) - 1 / (2 x sigma^2) * sum^n (x_i - mu)^2  (n for prediction times, n=1 for one frame, x_i for gt)
                         = -1/2 * ln(2pi) - 1/2 * ln(sigma^2) - 1/(2 x sigma^2) * (x - mu)^2
        == min -L(mu, sgima) = 0.5 x ln(2pi) + 0.5 x ln(sigma^2) + 1/(2 x sigma^2) * (x - mu)^2
                             = 0.5 x ln_2PI + ln(sigma) + 0.5 x (MU_DIFF/sigma)^2
                             = A - B + C
            In batch and Time sample, b and T are summed and averaged.
        '''
        b, T, _ = target.shape
        # read prediction paras
        mus = output[:, :, self.ncenter : (self.ncenter + self.ncenter * self.ndim)].view(b, T, self.ncenter, self.ndim)  # [b, T, ncenter, ndim]
        
        # apply min sigma
        neg_log_sigmas_out = output[:, :, (self.ncenter + self.ncenter * self.ndim):].view(b, T, self.ncenter, self.ndim)  # [b, T, ncenter, ndim]   
        inv_sigmas_min = torch.ones(neg_log_sigmas_out.size()).cuda() * (1. / self.sigma_min)
        inv_sigmas_min_log = torch.log(inv_sigmas_min)
        neg_log_sigmas = torch.min(neg_log_sigmas_out, inv_sigmas_min_log)
        
        inv_sigmas = torch.exp(neg_log_sigmas)
        # replicate the target of ncenter to minus mu
        target_rep = target.unsqueeze(2).expand(b, T, self.ncenter, self.ndim)  # [b, T, ncenter, ndim]

        MU_DIFF = target_rep - mus  # [b, T, ncenter, ndim]
        # sigma process
        # A = 0.5 * math.log(2 * math.pi)   # 0.9189385332046727
        # B = neg_log_sigmas  # [b, T, ncenter, ndim]
        # C = 0.5 * (MU_DIFF * inv_sigmas)**2  # [b, T, ncenter, ndim]
        # negative_loglikelihood =  A - B + C  # [b, T, ncenter, ndim]
        
        # return negative_loglikelihood.mean()
        return (MU_DIFF**2).mean()

def Sample_GMM(gmm_params, ncenter, ndim, weight_smooth=0.0, sigma_scale=0.0):
    '''Sample values from a given a GMM distribution.
    Args:
        gmm_params: [b, target_length, (2 * ndim + 1) * ncenter], including the 
        distribution weights, average and sigma
        ncenter: numbers of gaussian distribution
        ndim: dimension of each gaussian distribution 
        weight_smooth: float, smooth the gaussian distribution weights
        sigma_scale: float, adjust the gaussian scale, larger for sharper prediction,
            0 for zero sigma which always return average values
    Returns:
        current_sample: [b,t,c=ndim]
    '''
    b, T, _ = gmm_params.shape
    gmm_params_cpu = gmm_params.cpu().view(-1, (2 * ndim + 1) * ncenter)
    # compute each distrubution probability
    prob = F.softmax(gmm_params_cpu[:, : ncenter] * (1 + weight_smooth), dim=1)
    # select the gaussian distribution according to their weights
    selected_idx = torch.multinomial(prob, num_samples=1, replacement=True)
    
    mu = gmm_params_cpu[:, ncenter : ncenter + ncenter * ndim]
    # please note that we use -logsigma as output, hence here we need to take the negative
    sigma = torch.exp(-gmm_params_cpu[:, ncenter + ncenter * ndim:]) * sigma_scale
    # print('sigma average:', sigma.mean())
    
    selected_sigma = torch.empty(b*T, ndim).float()
    selected_mu = torch.empty(b*T, ndim).float()
    current_sample = torch.randn(b*T, ndim).float()

    for i in range(b*T):
        idx = selected_idx[i, 0]
        selected_sigma[i, :] = sigma[i, idx * ndim:(idx + 1) * ndim]
        selected_mu[i, :] = mu[i, idx * ndim:(idx + 1) * ndim]

    # sample with sel sigma and sel mean
    current_sample = current_sample * selected_sigma + selected_mu

    return  current_sample.reshape(b, T, -1)
