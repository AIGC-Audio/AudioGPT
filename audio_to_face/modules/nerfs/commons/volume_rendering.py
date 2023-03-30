import torch
import torch.nn.functional as F
import numpy as np

from audio_to_face.modules.nerfs.commons.ray_samplers import get_rays

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def raw2outputs(raw, z_vals, rays_d, bc_rgb, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model, rgb+sigma
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    def raw2alpha(raw, dists, act_fn=F.relu): 
        """
        Args:
            raw: predicted sigma; [N_rays, N_samples]
            dists: delta distance; [N_rays, N_samples]
        return:
            alpha: normalized volume density (occulusion degree) in the delta distance; [N_rays, N_samples]
        """
        return 1. - torch.exp(-(act_fn(raw)+1e-6)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1] # delta_z_vals ,[N_rays, N_samples-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1)  # add infintely far, [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1) #  [N_rays, N_samples]

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    rgb = torch.cat((rgb[:, :-1, :], bc_rgb.unsqueeze(1)), dim=1) # replace the last sample point with background color
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    importance_weights = alpha * \
        torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(importance_weights[..., None] * rgb, -2)  # [N_rays, 3]
    
    rgb_map_fg = torch.sum(importance_weights[:, :-1, None]*rgb[:, :-1, :], -2)

    depth_map = torch.sum(importance_weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(importance_weights, -1))
    accu_map = torch.sum(importance_weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-accu_map[..., None])

    return rgb_map, disp_map, accu_map, importance_weights, depth_map, rgb_map_fg


def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])


    # Invert CDF
    u = u.to(device).contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples

def render_rays(ray_batch,
                bc_rgb,
                cond,
                network_fn,
                N_samples,
                return_raw=False,
                linear_disp=False,
                perturb=1.,
                N_importance=0,
                white_bkgd=False,
                raw_noise_std=0.,
                **kwargs
                ):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2]) # [near, far]
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device) # [64,]
    if not linear_disp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples]) # [1024, 64]

    if perturb > 0.: # default 1., set it to add noise in z_vals
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])  # [1024, 63]
        upper = torch.cat([mids, z_vals[..., -1:]], -1)  # [1024, 64]
        lower = torch.cat([z_vals[..., :1], mids], -1)  # [1024, 64]
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(device)

        t_rand[..., -1] = 1.0
        z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # coarse_network_out = network_fn.forward(pts, cond, viewdirs, run_model_fine=False) # [N_rays, N_samples, rgbd]
    coarse_network_out = network_fn.forward(pts, cond, viewdirs,run_model_fine=False, **kwargs) # [N_rays, N_samples, rgbd]
    raw = coarse_network_out['rgb_sigma']

    rgb_map, disp_map, acc_map, weights, depth_map, rgb_map_fg = raw2outputs(
        raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd)

    if N_importance > 0: # default 128
        # additionally run on fine model
        rgb_map_0, disp_map_0, acc_map_0, last_weight_0, rgb_map_fg_0 = rgb_map, disp_map, acc_map, weights[..., -1], rgb_map_fg

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        # fine_network_out = network_fn.forward(pts, cond, viewdirs, run_model_fine=True)
        fine_network_out = network_fn.forward(pts, cond, viewdirs, run_model_fine=True, **kwargs) # [N_rays, N_samples, rgbd]

        raw = fine_network_out['rgb_sigma']

        rgb_map, disp_map, acc_map, weights, depth_map, rgb_map_fg = raw2outputs(
            raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'rgb_map_fg': rgb_map_fg}
    if return_raw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb_map_coarse'] = rgb_map_0
        ret['disp_map_coarse'] = disp_map_0
        ret['accu_map_coarse'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['last_weight'] = weights[..., -1]
        ret['last_weight0'] = last_weight_0
        ret['rgb_map_fg0'] = rgb_map_fg_0

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
            print(f"! [Numerical Error] {k} contains nan or inf.")
    return ret


def batchify_render_rays(rays_flat, bc_rgb, cond, chunk, network_fn, N_samples,  N_importance, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        if cond.squeeze().ndim == 1:    
            ret = render_rays(rays_flat[i:i+chunk], bc_rgb[i:i+chunk],
                          cond, network_fn, N_samples, N_importance=N_importance, **kwargs)
        elif cond.squeeze().ndim == 2:
            ret = render_rays(rays_flat[i:i+chunk], bc_rgb[i:i+chunk],
                          cond[i:i+chunk], network_fn, N_samples, N_importance=N_importance, **kwargs)

        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render_dynamic_face(H, W, focal, cx, cy, chunk=1024, rays_o=None, rays_d=None, bc_rgb=None, cond=None,
                        c2w=None, near=0., far=1., use_viewdirs=True, c2w_staticcam=None,
                        network_fn=None, N_samples=None, N_importance=None,
                        **kwargs):
    """
    bc_rgb: [H,W,3]
    """
    if bc_rgb is not None:
        bc_rgb = bc_rgb.reshape(-1, 3) # [H*W, 3]

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w, cx, cy)
    else:
        # use provided ray batch
        rays_o, rays_d = rays_o, rays_d

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam, cx, cy)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * \
        torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1) # [N, 8]
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1) # [N,11=rays_o3+rays_d3+nearfar2+viewdir3]

    # Render and reshape
    all_ret = batchify_render_rays(rays, bc_rgb, cond, chunk, network_fn=network_fn, N_samples=N_samples, N_importance=N_importance, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'last_weight', 'rgb_map_fg']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


if __name__ == '__main__':
    get_rays(450, 450, 1200, )