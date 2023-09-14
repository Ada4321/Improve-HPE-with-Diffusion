import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/root/Improve-HPE-with-Diffusion')
from core.criterion import build_criterion


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':  # use this -- sample beta values linearly between linear_start=1e-4 and linear_end=2e-2
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        regressor,   # regression function -- mlps
        denoise_fn,  # denoise function -- denoise_transformer
        diff_opt,
        loss_opt
        # schedule_opt=None
    ):
        super().__init__()
        self.regressor = regressor
        self.denoise_fn = denoise_fn
        self.loss_opt = loss_opt
        self.condition_on_preds = diff_opt['condition_on_preds']
        self.is_ddim = diff_opt.get('is_ddim', True)
        self.clip_denoised = diff_opt.get('clip_denoised', False)
        self.predict_x_start = diff_opt.get('predict_x_start', False)
        # if schedule_opt is not None:
        #     pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        self.loss_fn = build_criterion(self.loss_opt, device)

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end']) 
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        if 'sample_step' in schedule_opt:
            self.sample_steps = schedule_opt['sample_step']

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        self.register_buffer('sqrt_recip_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. / (1. - alphas_cumprod))))
        self.register_buffer('sqrt_recipm1_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. / (1. - alphas_cumprod) - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise      # produce X_0 from X_t
    
    def predict_noise_from_start(self, x_t, t, x_start):
        return self.sqrt_recip_one_minus_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_one_minus_alphas_cumprod[t] * x_start      # produce X_0 from X_t
    
    def predict_start_from_noise_continuous(self, x_t, gamma, noise):
        a = 1. / gamma
        b = torch.sqrt(1. / torch.pow(gamma, 2) - 1)
        return a * x_t - b * noise      # produce X_0 from X_t

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t                         # estimated mu_{t-1}
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]  # estimated sigma_{t-1}
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance_ddpm(self, x, t,  image_embed, cur_preds):
        batch_size = x.shape[0]

        noise_level = torch.FloatTensor(  # repeat noise_level for each batch
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        
        if not self.predict_x_start:
            if not self.condition_on_preds:
                x_recon = self.predict_start_from_noise(
                    x, t=t, noise=self.denoise_fn(x, image_embed, noise_level))  # X_0
            else:
                x_recon = self.predict_start_from_noise(
                    x, t=t, noise=self.denoise_fn(x, image_embed, noise_level, cur_preds=cur_preds))
        else:
            if not self.condition_on_preds:
                x_recon = self.denoise_fn(x, image_embed, noise_level)          # X_0
            else:
                x_recon = self.denoise_fn(x, image_embed, noise_level, cur_preds=cur_preds)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    def p_mean_variance_ddim(self, x, t, alpha, alpha_next, image_embed, cur_preds):
        batch_size = x.shape[0]

        noise_level = torch.FloatTensor(  # repeat noise_level for each batch
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        
        if not self.predict_x_start:
            if not self.condition_on_preds:
                pred_noise = self.denoise_fn(x, image_embed, noise_level)
            else:
                pred_noise = self.denoise_fn(x, image_embed, noise_level, cur_preds=cur_preds)
            x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)     # X_0
        else:
            if not self.condition_on_preds:
                x_recon = self.denoise_fn(x.reshape(batch_size,-1,2), image_embed, noise_level)
            else:
                x_recon = self.denoise_fn(x.reshape(batch_size,-1,2), image_embed, noise_level, cur_preds=cur_preds.reshape(batch_size,-1,2))
            pred_noise = self.predict_noise_from_start(x, t=t, x_start=x_recon)  # noise

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)

        posterior_variance = ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = ((1 - alpha_next) - torch.square(posterior_variance)).sqrt()

        return x_recon, pred_noise, c, posterior_variance


    @torch.no_grad()
    def p_sample_ddpm(self, x, t, image_embed, cur_preds):
        model_mean, model_log_variance = self.p_mean_variance_ddpm(
            x=x, t=t, image_embed=image_embed, cur_preds=cur_preds)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()  # return X_{t-1}
    
    @torch.no_grad()
    def p_sample_ddim(self, x, t, t_next, alpha, alpha_next, image_embed, cur_preds):
        x_recon, pred_noise, c, model_variance = self.p_mean_variance_ddim(
            x=x, t=t, alpha=alpha, alpha_next=alpha_next, image_embed=image_embed, cur_preds=cur_preds)
        
        if t_next < 0:
            return x_recon
        
        noise = torch.randn_like(x_recon) if t_next > 0 else 0.

        return x_recon * alpha_next.sqrt() + \
                    model_variance * noise + \
                    c * pred_noise               # X_{t-1}

    @torch.no_grad()
    def p_sample_loop_ddpm(self, x_in):
        image_embed = x_in['im_feats']
        cur_preds = x_in['cur_preds']

        device = cur_preds.device
        
        shape = cur_preds.shape                  # it should be (b,34)
        res = torch.randn(shape, device=device)  # initialize residual as noise

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            res = self.p_sample_ddpm(res, i, image_embed, cur_preds)
            
        return res  # the final predicted residual

    @torch.no_grad()
    def p_sample_loop_ddim(self, x_in):
        image_embed = x_in['im_feats']
        cur_preds = x_in['cur_preds']

        shape, device, alphas = cur_preds.shape, cur_preds.device, self.alphas_cumprod_prev

        res = torch.randn(shape, device=device)  # initialize residual as noise

        times = torch.linspace(-1., self.num_timesteps, steps = self.sample_steps + 1)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = alphas[time+1]
            alpha_next = alphas[time_next+1]

            res = self.p_sample_ddim(res, time, time_next, alpha, alpha_next, image_embed, cur_preds)

        return res
    
    @torch.no_grad()
    def p_sample_loop(self, x_in):
        if not self.is_ddim:
            res = self.p_sample_loop_ddpm(x_in)
        else:
            res = self.p_sample_loop_ddim(x_in)

        return res

    @torch.no_grad()
    def sample(self, images):
        preds, imfeats = self.regressor(images)
        if self.loss_opt['type'] == 'sanity_check':
            return {'preds': preds}
            #pred_jts = preds['raw_pred_jts']
            #return pred_jts + torch.ones_like(pred_jts) * 0.
        else:
            x_in = {
                'im_feats': imfeats,
                'cur_preds': preds['raw_pred_jts']
            }
            res = self.p_sample_loop(x_in)

            return {'preds': preds, 'res': res}
            #return preds + res
            #return preds

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):  # sample X_t from X_0
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gamma
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def regress(self, images):
        return self.regressor(images)
        # preds, imfeats = self.regressor(images)
        # if self.training:
        #     return preds, imfeats, self.reg_loss_func(preds, gt)
        # else:
        #     return preds, imfeats
    
    def diffuse(self, x_in, noise=None):
        x_start = x_in['gt_res']        # x_start should be gt_res, the goal of diffusion 
        image_embed = x_in['im_feats']
        cur_preds = x_in['cur_preds']
        
        # [b, c, h, w] = x_start.shape
        b, num_kp_coords = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)  # sample a 't' value -- total diffusion step T
        continuous_sqrt_alpha_cumprod = torch.FloatTensor( # sample one alpha value for each sample in the batch
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start)) # sample a noise from N(0,1)
        x_noisy = self.q_sample(                                  # sample X_t
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod, noise=noise)

        if not self.condition_on_preds:
            x_recon = self.denoise_fn(x_noisy.reshape(b,-1,2), image_embed, continuous_sqrt_alpha_cumprod) # x_recon -- reconstructed noise
        else:
            x_recon = self.denoise_fn(
                x_noisy.reshape(b,-1,2), image_embed, continuous_sqrt_alpha_cumprod, cur_preds.reshape(b,-1,2))
            
        if not self.predict_x_start:    
            res_recon = self.predict_start_from_noise_continuous(x_noisy, continuous_sqrt_alpha_cumprod.view(-1)[0], x_recon)
            return x_recon, noise, res_recon
        else:
            return x_recon

    # def forward(self, x, *args, **kwargs):
    def forward(self, images, gt):
        # preds, im_feats, l_reg = self.regress(images=images, gt=gt)
        if self.loss_opt['type'] == 'fixed_res_and_diff':
            with torch.no_grad():
                preds, im_feats = self.regress(images=images)
        else:
            preds, im_feats = self.regress(images=images)

        pred_jts = preds['raw_pred_jts']
        pred_sigmas = preds['pred_sigmas'] if 'pred_sigmas' in preds else None
        pred_sigmas = self.denoise_fn.sigma_linear(pred_sigmas.unsqueeze(-1)).squeeze(-1)
        # gt_res = torch.abs(pred_jts - gt)
        gt_res = gt - pred_jts
        # pred_jts = pred_jts.reshape(images.shape[0],-1,2)
        # gt_res = gt_res.reshape(images.shape[0],-1,2)
        x_in = {
            'im_feats': im_feats.detach(), 
            'gt_res': gt_res.detach(), 
            'cur_preds': pred_jts.detach()
        }

        if not self.predict_x_start:
            pred_noise, gt_noise, res_recon = self.diffuse(x_in=x_in)
            losses = self.loss_fn(
                preds=pred_jts, 
                gt=gt, 
                pred_noise=pred_noise, 
                gt_noise=gt_noise, 
                res=res_recon,
                predict_x_start=self.predict_x_start,
                rle_loss=self.loss_opt['rle_loss'],
                sigma=pred_sigmas
                )
        else:
            res_recon = self.diffuse(x_in=x_in)
            losses = self.loss_fn(
                preds=pred_jts, 
                gt=gt,  
                res_recon=res_recon,
                gt_res=gt_res,
                predict_x_start=self.predict_x_start,
                rle_loss=self.loss_opt['rle_loss'],
                sigma=pred_sigmas
                )

        return losses