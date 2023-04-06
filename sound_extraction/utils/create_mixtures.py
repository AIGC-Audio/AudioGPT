import torch
import numpy as np

def add_noise_and_scale(front, noise, snr_l=0, snr_h=0, scale_lower=1.0, scale_upper=1.0):
    """
    :param front: front-head audio, like vocal [samples,channel], will be normlized so any scale will be fine
    :param noise: noise, [samples,channel], any scale
    :param snr_l: Optional
    :param snr_h: Optional
    :param scale_lower: Optional
    :param scale_upper: Optional
    :return: scaled front and noise (noisy = front + noise), all_mel_e2e outputs are noramlized within [-1 , 1]
    """
    snr = None
    noise, front = normalize_energy_torch(noise), normalize_energy_torch(front)  # set noise and vocal to equal range [-1,1]
    # print("normalize:",torch.max(noise),torch.max(front))
    if snr_l is not None and snr_h is not None:
        front, noise, snr = _random_noise(front, noise, snr_l=snr_l, snr_h=snr_h)  # remix them with a specific snr

    noisy, noise, front = unify_energy_torch(noise + front, noise, front)   # normalize noisy, noise and vocal energy into [-1,1]
  
    # print("unify:", torch.max(noise), torch.max(front), torch.max(noisy))
    scale = _random_scale(scale_lower, scale_upper) # random scale these three signal
     
    # print("Scale",scale)
    noisy, noise, front = noisy * scale, noise * scale, front * scale  # apply scale
    # print("after scale", torch.max(noisy), torch.max(noise), torch.max(front), snr, scale)
    
    front, noise = _to_numpy(front), _to_numpy(noise) # [num_samples]
    mixed_wav = front + noise
    
    return front, noise, mixed_wav, snr, scale

def _random_scale(lower=0.3, upper=0.9):
    return float(uniform_torch(lower, upper))

def _random_noise(clean, noise, snr_l=None, snr_h=None):
    snr = uniform_torch(snr_l,snr_h)
    clean_weight = 10 ** (float(snr) / 20)
    return clean, noise/clean_weight, snr
    
def _to_numpy(wav):
    return np.transpose(wav, (1, 0))[0].numpy()  # [num_samples]

def normalize_energy(audio, alpha = 1):
    '''
    :param audio: 1d waveform, [batchsize, *],
    :param alpha: the value of output range from: [-alpha,alpha]
    :return: 1d waveform which value range from: [-alpha,alpha]
    '''
    val_max = activelev(audio)
    return (audio / val_max) * alpha

def normalize_energy_torch(audio, alpha = 1):
    '''
    If the signal is almost empty(determined by threshold), if will only be divided by 2**15
    :param audio: 1d waveform, 2**15
    :param alpha: the value of output range from: [-alpha,alpha]
    :return: 1d waveform which value range from: [-alpha,alpha]
    '''
    val_max = activelev_torch([audio])
    return (audio / val_max) * alpha

def unify_energy(*args):
    max_amp = activelev(args)
    mix_scale = 1.0/max_amp
    return [x * mix_scale for x in args]

def unify_energy_torch(*args):
    max_amp = activelev_torch(args)
    mix_scale = 1.0/max_amp
    return [x * mix_scale for x in args]

def activelev(*args):
    '''
        need to update like matlab
    '''
    return np.max(np.abs([*args]))

def activelev_torch(*args):
    '''
        need to update like matlab
    '''
    res = []
    args = args[0]
    for each in args:
        res.append(torch.max(torch.abs(each)))
    return max(res)

def uniform_torch(lower, upper):
    if(abs(lower-upper)<1e-5):
        return upper
    return (upper-lower)*torch.rand(1)+lower

if __name__ == "__main__":
    wav1 = torch.randn(1, 32000)
    wav2 = torch.randn(1, 32000)
    target, noise, snr, scale = add_noise_and_scale(wav1, wav2)
