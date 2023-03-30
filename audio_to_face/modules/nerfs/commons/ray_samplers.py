import torch
import numpy as np
import math
import random
from audio_to_face.utils.commons.hparams import hparams


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_rays(H, W, focal, c2w, cx=None, cy=None):
    """
    Get the rays emitted from camera to all pixels. 
    The ray is represented in world coordinate
    input:
        H: height of the image (in pixel)
        W: width of the image (in pixel)
        focal: focal length of the camera (in pixel)
        c2w: a 3x4 camera-to-world matrix, it should be something like this:
            [[r11, r12, r13, t1],
            [r21, r22, r23, t2],
            [r31, r32, r33, t3],]
        cx: center of camera in Width axis
        cy: center of camera in Height axis
    return:
        rays_o: the start point of the ray
        rays_d: the direction of the ray. so you can sample the point in the ray with:
            xyz = rays_o + rays_d * z_val, where z_val is the distance.
    """
    # pytorch's meshgrid has indexing='ij'
    i_pixels, j_pixels = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i_pixels = i_pixels.t().to(device)
    j_pixels = j_pixels.t().to(device)
    if cx is None:
        cx = W*.5
    if cy is None:
        cy = H*.5
    directions = torch.stack([(i_pixels-cx)/focal, -(j_pixels-cy)/focal, -torch.ones_like(i_pixels)], dim=-1) 
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1) # rays_delta
    # origin point of all ray, the camera center in the world coordinate
    rays_o = c2w[:3, -1].expand(rays_d.shape) 
    return rays_o, rays_d


class BaseRaySampler:
    def __init__(self, N_rays):
        super(BaseRaySampler, self).__init__()
        self.N_rays = N_rays

    def __call__(self, H, W, focal, c2w):
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        select_coords = self.sample_rays(H, W).to(device)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        return rays_o, rays_d, select_coords

    def sample_rays(self, H, W, **kwargs):
        raise NotImplementedError


class UniformRaySampler(BaseRaySampler):
    def __init__(self, N_rays=None):
        """
        Uniform RaySampler in vanilla NeRF and AD-NeRF
            We use it in the photo reconstruction training (i.e., calculating MSE).
        """
        super().__init__(N_rays=N_rays)        

    def sample_rays(self, H, W, n_rays=None, rect=None, in_rect_percent=0.9, **kwargs):
        """
        rect: [w1, h1, delta_w, delta_h]
        """
        if n_rays is None:
            n_rays = self.N_rays
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1, 2]).to(device)  # (H * W, 2)
        if rect is None:
            # uniformly sample the whole image
            select_inds = np.random.choice(coords.shape[0], size=[n_rays], replace=False) # (n_rays, )
            select_coords = coords[select_inds].long().to(device)
        else:
            # uniformly sample from the rect reigon and out-rect, respectively.
            w1, h1, delta_w, delta_h = rect
            w2 = w1 + delta_w
            h2 = h1 + delta_h
            rect_inds = (coords[:, 0] >= h1) & (coords[:, 0] <= h2) & (coords[:, 1] >= w1) & ( coords[:, 1] <= w2) # (H*W), boolean mask
            
            coords_rect = coords[rect_inds] # [num_idx_in_mask, 2]
            coords_norect = coords[~rect_inds]
            num_rays_in_rect = int(n_rays * in_rect_percent)
            num_rays_out_rect = n_rays - num_rays_in_rect
            select_inds_rect = np.random.choice(coords_rect.shape[0], size=[num_rays_in_rect], replace=False)  # (num_rays_in_rect,)
            select_coords_rect = coords_rect[select_inds_rect].long() # (num_rays_in_rect, 2)
            select_inds_norect = np.random.choice(coords_norect.shape[0], size=[num_rays_out_rect], replace=False)  # (num_rays_out_rect,)
            select_coords_norect = coords_norect[select_inds_norect].long() # (num_rays_in_rect)
            select_coords = torch.cat((select_coords_rect, select_coords_norect), dim=0) 
        return select_coords  # (n_rays, 2)
    
    def __call__(self, H, W, focal, c2w, n_rays=None, select_coords=None, rect=None, in_rect_percent=0.9, **kwargs):
        rays_o, rays_d = get_rays(H, W, focal, c2w) # [H, W, 3]
        if select_coords is None:
            select_coords = self.sample_rays(H, W, n_rays, rect, in_rect_percent) # [N_rand, 2]
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        return rays_o, rays_d, select_coords

    def sample_pixels_from_img_with_select_coords(self, img, select_coords):
        """
        img: [H*W, 3]
        """
        return img[select_coords[:, 0], select_coords[:, 1]] 


class TorsoUniformRaySampler(BaseRaySampler):
    def __init__(self, N_rays=None):
        """
        Uniform RaySampler for Torso
            We use it in the photo reconstruction training (i.e., calculating MSE).
        """
        super().__init__(N_rays=N_rays)        

    def sample_rays(self, H, W, n_rays=None, rect=None, in_rect_percent=0.9, **kwargs):
        """
        rect: [w1, h1, delta_w, delta_h]
        """
        if n_rays is None:
            n_rays = self.N_rays
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1, 2]).to(device)  # (H * W, 2)
        if rect is None:
            rect = [0, H/2, W, H/2]            
        # uniformly sample from the rect reigon and out-rect, respectively.
        w1, h1, delta_w, delta_h = rect
        w2 = w1 + delta_w
        h2 = h1 + delta_h
        rect_inds = (coords[:, 0] >= h1) & (coords[:, 0] <= h2) & (coords[:, 1] >= w1) & ( coords[:, 1] <= w2) # (H*W), boolean mask
        
        coords_rect = coords[rect_inds] # [num_idx_in_mask, 2]
        coords_norect = coords[~rect_inds]
        num_rays_in_rect = int(n_rays * in_rect_percent)
        num_rays_out_rect = n_rays - num_rays_in_rect
        select_inds_rect = np.random.choice(coords_rect.shape[0], size=[num_rays_in_rect], replace=False)  # (num_rays_in_rect,)
        select_coords_rect = coords_rect[select_inds_rect].long() # (num_rays_in_rect, 2)
        select_inds_norect = np.random.choice(coords_norect.shape[0], size=[num_rays_out_rect], replace=False)  # (num_rays_out_rect,)
        select_coords_norect = coords_norect[select_inds_norect].long() # (num_rays_in_rect)
        select_coords = torch.cat((select_coords_rect, select_coords_norect), dim=0) 

        return select_coords  # (n_rays, 2)
    
    def __call__(self, H, W, focal, c2w, n_rays=None, select_coords=None, rect=None, in_rect_percent=0.9, **kwargs):
        rays_o, rays_d = get_rays(H, W, focal, c2w) # [H, W, 3]
        if select_coords is None:
            select_coords = self.sample_rays(H, W, n_rays, rect, in_rect_percent, **kwargs) # [N_rand, 2]
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        return rays_o, rays_d, select_coords

    def sample_image_with_select_coords(self, img, select_coords):
        """
        img: [H*W, 3]
        """
        return img[select_coords[:, 0], select_coords[:, 1]] 


class FullRaySampler(BaseRaySampler):
    def __init__(self, **kwargs):
        """
        This sampler directly get all rays to render the whole image.
            We only use it in the inference phase.
        """
        super().__init__(N_rays=None)

    def sample_rays(self, H, W, **kwargs):
        num_h_points = int(H*hparams['infer_scale_factor'])
        num_w_points = int(W*hparams['infer_scale_factor'])
        h, w = torch.meshgrid([torch.linspace(0,H-1,num_h_points), torch.linspace(0,W-1,num_w_points)])
        h = h.reshape([-1,1]).long()
        w = w.reshape([-1,1]).long()
        select_coords = torch.cat([h, w], dim=-1)# (n_rays, 2)
        return select_coords

    def __call__(self, H, W, focal, c2w):
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        select_coords = self.sample_rays(H, W).to(device)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        return rays_o, rays_d, select_coords


class PatchRaySampler(BaseRaySampler):
    def __init__(self, N_rays, min_scale=0.2, max_scale=1., scale_anneal=-0.1):
        """
        A modified version of FlexGridSampler in GRAF
            We utilize this sampler to get patch-wise rays for adversarial training
        """
        self.N_rays_sqrt = int(math.sqrt(N_rays))
        super(PatchRaySampler, self).__init__(self.N_rays_sqrt**2)

        self.min_scale = min_scale
        self.max_scale = max_scale

        # nn.functional.grid_sample grid value range in [-1,1]
        self.w, self.h = torch.meshgrid([torch.linspace(-1,1,self.N_rays_sqrt),
                                         torch.linspace(-1,1,self.N_rays_sqrt)])
        self.h = self.h.unsqueeze(2) # [n_rays_sqrt, n_rays_sqrt, 1]
        self.w = self.w.unsqueeze(2) # [n_rays_sqrt, n_rays_sqrt, 1]

        # directly return grid for grid_sample
        self.return_indices = False

        self.scale_anneal = scale_anneal

    def sample_rays(self, H, W, iterations, n_rays=None, rect=None, **kwargs):
        """
        get the uv coordinates of the rays that belongs to a image patch
        return: a uv mesh grid of [H, W, 2], 
            note that different from index-based sampler which 
            generates grid with values in [0,H] and [0,W]
            to utilize F.grid_search, we generate grid with values in [-1,1]
        """
        # get the unit mesh grid first, 
        # 0 denotes the center and -1~1 denotes the boundary of the whole image
        if n_rays is None:
            sqrt_n_rays = self.N_rays_sqrt
            unit_w = self.w # value in [-1,1]
            unit_h = self.h # value in [-1,1]
        else:
            sqrt_n_rays = int(math.sqrt(n_rays))
            unit_w, unit_h = torch.meshgrid([torch.linspace(-1, 1, sqrt_n_rays),
                                            torch.linspace(-1,1,sqrt_n_rays)])
            unit_w = unit_w.unsqueeze(2) # [n_rays_sqrt, n_rays_sqrt, 1]
            unit_h = unit_h.unsqueeze(2)

        # then get the scale factor of the unit mesh grid
        # k_iter = iterations // 1000 * 3 * self.scale_anneal
        # min_scale = max(self.min_scale, self.max_scale * math.exp(k_iter))
        # min_scale = min(0.9, min_scale)
        min_scale = self.min_scale
        scale_factor = torch.Tensor(1).uniform_(min_scale, self.max_scale)
        w = unit_w * scale_factor
        h = unit_h * scale_factor
        
        if rect is None:
            max_offset_w = 1-scale_factor.item()
            max_offset_h = 1-scale_factor.item()
            h_offset = torch.Tensor(1).uniform_(0, max_offset_h) * (torch.randint(high=2,size=(1,)).float()-0.5)*2
            w_offset = torch.Tensor(1).uniform_(0, max_offset_w) * (torch.randint(high=2,size=(1,)).float()-0.5)*2
        else:
            w1, h1, delta_w, delta_h = rect
            w2 = w1 + delta_w
            h2 = h1 + delta_h

            # rule1. the edge of patch shall not out of the image
            # rule2. the center of patch shall not out of the rect
            min_offset_w = max(scale_factor.item()-1, (w1-W//2)/(W//2)) 
            min_offset_h = max(scale_factor.item()-1, (h1-H//2)/(H//2))
            max_offset_w = min(1-scale_factor.item(), (w2-W//2)/(W//2))
            max_offset_h = min(1-scale_factor.item(), (h2-H//2)/(H//2))
            h_offset = torch.Tensor(1).uniform_(min_offset_h, max_offset_h)
            w_offset = torch.Tensor(1).uniform_(min_offset_w, max_offset_w)
        h += h_offset
        w += w_offset
        hw = torch.cat([h,w], dim=2) # [H, W, 2], it is necessary to keep the H,W axis for grid_sampling
        select_coords = hw
        return select_coords

    def __call__(self, H, W, focal, c2w, iterations, n_rays=None, rect=None, **kwargs):
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        select_coords = self.sample_rays(H, W, iterations, n_rays, rect).to(rays_o.device)
        # instead of int index, this is float index, and we need to sample rays with interpolation
        rays_o = torch.nn.functional.grid_sample(rays_o.permute(2,0,1).unsqueeze(0), 
                                select_coords.unsqueeze(0), mode='bilinear', align_corners=True)[0]
        rays_d = torch.nn.functional.grid_sample(rays_d.permute(2,0,1).unsqueeze(0), 
                                select_coords.unsqueeze(0), mode='bilinear', align_corners=True)[0]
        rays_o = rays_o.permute(1,2,0).view(-1, 3)
        rays_d = rays_d.permute(1,2,0).view(-1, 3)
        return rays_o, rays_d, select_coords

    def sample_image_with_select_coords(self, img, select_coords):
        """
        img: [H, W, 3]
        """
        img = img.permute(2, 0, 1) # [3, H, W]
        sampled_patch = torch.nn.functional.grid_sample(img.unsqueeze(0), 
                                select_coords.unsqueeze(0), mode='bilinear', align_corners=True)[0]
        sampled_patch = sampled_patch.permute(1, 2, 0) # [H, W, 3]
        return sampled_patch


if __name__ == '__main__':
    H, W = 800, 800
    focal = 1200
    rect = [0,0,100,100]
    c2w = torch.FloatTensor([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    ray_sampler = BaseRaySampler(10000)
    uniform_ray_sampler = UniformRaySampler(10000)
    full_ray_sampler = FullRaySampler()
    flex_ray_sampler = PatchRaySampler(10000)
    uniform_ray_sampler(H,W,focal,c2w,10000,rect=rect)
    import cv2
    img = cv2.imread("experimiental_yerfor/r_0.png")
    img = torch.from_numpy(img).permute([2,0,1]).float()
    flex_ray_sampler.iterations = 50000
    rays_o, rays_d, select_coords = flex_ray_sampler(H,W,focal, c2w,rect=[0,0,800,800], iterations=1000)
    # gt_patch = flex_ray_sampler.sample_image_with_select_coords(select_coords, img)
    # gt_patch = gt_patch.permute([1,2,0]).numpy().astype(np.uint8)
    # cv2.imwrite("experimiental_yerfor/sampled_iteration=50000.png", gt_patch)