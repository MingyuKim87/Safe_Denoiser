import os
import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from .utils.lshash_torch import LSHash
import torch
import torch.nn.functional as F

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_repellency_method(name: str, ref_data, embed_fn, forward_fn, num_timesteps, max_idx, beta_min, beta_max, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](ref_data, embed_fn, forward_fn, num_timesteps, max_idx, beta_min, beta_max, **kwargs)

class RepellencyMethod(ABC):
    def __init__(self, ref_data, embed_fn, forward_fn, num_timesteps, max_idx, beta_min, beta_max, n_embed, **kwargs):
        self.ref_data = ref_data
        self.embed_fn = embed_fn
        self.forward_fn = forward_fn
        self.num_timesteps = num_timesteps
        self.max_idx = max_idx
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_embed = n_embed
        
        # params
        self.sigma = kwargs.get('sigma', 1.0)
        self.scale = kwargs.get('scale', 1.0)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        self.quantile = kwargs.get('quantile', 0.0)
        self.beta_threshold = kwargs.get('beta_threshold', False)
        self.beta_threshold_margin = kwargs.get('beta_threshold_margin', 0.0)
        
        self.proj_ref_path = kwargs.get('proj_ref_path', None)
        self.proj_beta_ref_path = kwargs.get('proj_noisy_ref_path_for_beta', None)
        self.cache_proj_ref = kwargs.get('cache_proj_ref', False)
        self.cache_proj_beta_ref = kwargs.get('cache_noisy_ref_path_for_beta', False)
        
        # cached for proj_ref (only for negation data)
        if self.cache_proj_ref:
            self.proj_refs = self.import_proj_ref(self.proj_ref_path)
        else:
            self.proj_refs = self.set_proj_ref()
                        
    def project(self, data, **kwargs):
        results = []
        if len(data) > self.n_embed:
            for i in range(0, len(data), self.n_embed):
                temp = self.embed_fn(data[i:min(i + self.n_embed, len(data))])
                results.append(temp)
            results = torch.cat(results, 0)
            
            # normalization
            out = torch.norm(results, dim=1, keepdim=True)
            results = results / out
            return results
        else:
            results = self.embed_fn(data)
            
            # normalization
            out = torch.norm(results, dim=1, keepdim=True)
            results = results / out
            return results
        
    def discrete_to_continous_time(self, idx, **kwargs):
        if idx == 0:
            result = 0.001
        else:
            result = idx / self.max_idx
        return result
    
    def sigma_edm(self, cont_time, **kwargs):
        return torch.sqrt(torch.exp(0.5*self.beta_max*cont_time**2 + self.beta_min * cont_time) - 1.)
        
    def mkdir_cache(self, path):
        dir_path = os.path.split(path)[0]
        os.makedirs(dir_path, exist_ok=True)
        
    def set_proj_ref(self):
        result = []
        with torch.no_grad():
            emb = self.project(self.ref_data)
            result.append(emb.cpu())

            # del
            del emb
            torch.cuda.empty_cache()
        
        # Save the result
        result = torch.cat(result, 0)
        print("[Proj_Ref] Save the cached proj_ref")
        self.mkdir_cache(self.proj_ref_path)
        torch.save(result, self.proj_ref_path)

        # map location (cuda)
        result = result.to('cuda')
        return result
    
    def set_noisy_proj_ref(self, scheduler, num_timesteps=None, **kwargs):
        # container
        results = {}
        
        # random seed
        num_inference_steps = num_timesteps if num_timesteps is not None else 50
        device = kwargs.get("device", "cuda")
        generator = kwargs.get("generator", torch.Generator(device=device).manual_seed(42))
        
        # reference dataset
        xs_0 = self.proj_refs
        
        # Prepare timesteps
        scheduler.set_timesteps(num_inference_steps, device=device)
        
        # timesteps
        timesteps = scheduler.timesteps
        
        with torch.no_grad():
            for t in timesteps:
                temp_container = []
                for bs in range(0, len(xs_0), self.n_embed):
                    batch_xs_0 = xs_0[bs:bs+self.n_embed]
                    
                    # noise
                    noise = torch.randn(batch_xs_0.shape, 
                                        generator=generator, 
                                        device=device, dtype=torch.float32)

                    # update latents with repellency
                    latents = scheduler.add_noise(batch_xs_0, noise, t)
                    temp_container.append(latents)
                    
                    # del
                    del latents
                    torch.cuda.empty_cache()
                    
                temp_container = torch.cat(temp_container, 0)
                results[t.item()] = temp_container
                
                # del
                del temp_container
                torch.cuda.empty_cache()
            
        print("[Proj_Ref] Save the cached proj_beta_ref")
        self.mkdir_cache(self.proj_beta_ref_path)
        torch.save(results, self.proj_beta_ref_path)
        return results

    def import_proj_ref(self, proj_ref_path):
        result = torch.load(proj_ref_path, map_location=self.ref_data.device)
        return result
        
    def get_proj_ref(self):
        return self.proj_refs
    
    def get_noisy_proj_refs(self):
        return self.noisy_proj_refs
                
    @abstractmethod
    def empirical_denoiser(self, data, **kwargs):
        pass
        
    def conditioning(self, x_0_hat, **kwargs):
        if kwargs.get("beta_threshold", False):
            return self.conditioning_threshold(x_0_hat, **kwargs)
        else:
            return self.conditioning_1(x_0_hat, **kwargs)
            
    def conditioning_threshold(self, x_0_hat, **kwargs):    
        # empirical denoiser
        negative_x_0_hat, negative_x_0_hat_item = self.empirical_denoiser(x_t=x_0_hat, sigma=self.sigma, **kwargs)
        
        # \beta threshold
            # timestep -> 0
        beta_threshold = self.beta_threshold - self.beta_threshold_margin
        if negative_x_0_hat_item["denominator"] > beta_threshold: is_negation = True
        else: is_negation = False
        
        x_0_hat -= self.scale * negative_x_0_hat
        return {"x_0_hat" : x_0_hat, "mean_x_0_hat" : negative_x_0_hat_item, "is_negation" : is_negation}
    
    def conditioning_1(self, x_0_hat, **kwargs):
        negative_x_0_hat, negative_x_0_hat_item = self.empirical_denoiser(x_t=x_0_hat, sigma=self.sigma, **kwargs)
        x_0_hat -= self.scale * negative_x_0_hat
        return {"x_0_hat" : negative_x_0_hat, "mean_x_0_hat" : negative_x_0_hat_item, "is_negation" : True}
            
@register_conditioning_method(name='euclidean')
class EuclideanRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, forward_fn, max_idx, beta_min, beta_max, **kwargs):
        super().__init__(ref_data, embed_fn, forward_fn, max_idx, beta_min, beta_max, **kwargs)
        self.scale = kwargs.get('scale', 1.0)

    def empirical_denoiser(self, x_t, time, sigma=3.0, **kwargs):
        # shape
        M, C, H, W = self.ref_data.shape
        N, _, _, _ = x_t.shape
        
        # reshape
        x_t_proj = x_t.reshape(-1, C*H*W)
        ref_data_proj = self.ref_data.reshape(-1, C*H*W)
        
        # denominator
        ones = torch.ones(M)[..., None].to(ref_data_proj.device)
        
        # empicial denoiser by kernel tricks
        kernel = - (torch.cdist(x_t_proj, ref_data_proj)[0]).reshape(N, M, 1).repeat(1, 1, C*H*W+1) / (2. * sigma ** 2) #- 2. * np.pi * (sigma ** 2)
        kernel = (kernel.exp() * (torch.cat((ref_data_proj, ones), 1).reshape(1, -1, C*H*W+1))).sum(dim=1)
        
        # Compute the log of the kernel
        log_kernel = kernel.log()  # Assume kernel values are all positive

        # Compute log of the denominator
        log_denominator = log_kernel[:, -1].reshape(-1, 1)  # Extract the last column in log space
        log_nominator = log_kernel[:, :-1]  # Extract all but the last column in log space
        log_negative_score = log_nominator - log_denominator
        
        # negative_score
        negative_score = log_negative_score.exp()
        negative_score_item = negative_score.mean().item()

        # reshape
        negative_score = negative_score.reshape(-1, C, H, W)
        return negative_score, negative_score_item


@register_conditioning_method(name='kernel')
class RBFKernelRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, forward_fn, max_idx, beta_min, beta_max, **kwargs):
        super().__init__(ref_data, embed_fn, forward_fn, max_idx, beta_min, beta_max, **kwargs)
        
        # cached for noisy_proj_refs (# of noisy_proj_refs greater than # of proj_refs)
        if self.cache_proj_beta_ref:
            self.noisy_proj_refs = self.import_proj_ref(self.proj_beta_ref_path)
        else:
            scheduler = kwargs.get("scheduler", None) 
            assert scheduler != None, "We need scheduler for computing \beta reference"
            self.noisy_proj_refs = self.set_noisy_proj_ref(scheduler, self.num_timesteps)
        self.noisy_refs_beta_quantitle = self.empirical_beta(sigma=self.sigma, quantitle=self.quantile)   
        
        self.scale = kwargs.get('scale', 1.0)

    def empirical_denoiser(self, x_t, time, sigma=1.0, **kwargs):
        # Embedding for kernel distance
        x_t_proj = self.project(x_t, time)
        ref_data_proj = self.project(self.ref_data, time)
        
        # reshape
        N, D = x_t_proj.shape
        M, D = ref_data_proj.shape
        _, C, H, W = self.ref_data.shape
        x_t_proj = x_t_proj.reshape(1, -1, D)
        ref_data_proj = ref_data_proj.reshape(1, -1, D)
        
        # data augmentation for denominator
        ref_data_org = self.ref_data.reshape(M, -1)
        ones = torch.ones(M)[..., None].to(ref_data_org.device)
        
        # empicial denoiser by kernel tricks
        kernel = - (torch.cdist(x_t_proj, ref_data_proj)[0]).reshape(N, M, 1).repeat(1, 1, C*H*W+1) / (2. * sigma ** 2) #- 2. * np.pi * (sigma ** 2)
        kernel = (kernel.exp() * (torch.cat((ref_data_org, ones), dim=1).reshape(1, -1, C*H*W+1))).sum(dim=1)
        
        # Avoid dividing by zero or very small numbers
        denominator = kernel[:, -1].reshape(-1, 1) + self.epsilon  # Add epsilon to prevent division by zero
        nominator = kernel[:, :-1]

        # Compute negative_score with enhanced stability
        negative_score = nominator / denominator
        negative_score_item = negative_score.clamp(min=-1e10, max=1e10).mean().item()
        
        # reshape
        negative_score = negative_score.reshape(-1, C, H, W)
        return negative_score, negative_score_item

@register_conditioning_method(name='kernel_fast')
class RBFKernelRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, forward_fn, num_timesteps, max_idx, beta_min, beta_max, **kwargs):
        super().__init__(ref_data, embed_fn, forward_fn, num_timesteps, max_idx, beta_min, beta_max, **kwargs)
        
        # params for safer 
        self.scale = kwargs.get('scale', 1.0)
        self.beta_threshold = kwargs.get('beta_threshold', -1.0)
            
        if self.beta_threshold <= 0:
            if self.cache_proj_beta_ref:
                self.noisy_proj_refs = self.import_proj_ref(self.proj_beta_ref_path)
            else:
                # compute beta reference
                scheduler = kwargs.get("scheduler", None) 
                assert scheduler != None, "We need scheduler for computing \beta reference"
                self.noisy_proj_refs = self.set_noisy_proj_ref(scheduler, self.num_timesteps)
            
            self.noisy_refs_beta_quantitle = self.empirical_beta(sigma=self.sigma, quantitle=self.quantile)   
            # pick the minimum timesteps (t=0) of beta in reference
            self.beta_threshold = self.noisy_refs_beta_quantitle[list(self.noisy_refs_beta_quantitle.keys())[-1]]
            
            # del
            del self.noisy_proj_refs, self.noisy_refs_beta_quantitle
            torch.cuda.empty_cache()
        
        
    def empirical_denoiser(self, x_t, sigma=1.0, **kwargs):
        '''
            args:
                x_t: latent (no need to be embedded)
            returns:
                negative_score: negative score
                negative_score_item: mean of negative score
        '''
        
        # Embedding for kernel distance
        x_t_proj = x_t.reshape(x_t.shape[0], -1)
        ref_data_latent = self.get_proj_ref()
        ref_data_proj = ref_data_latent.reshape(ref_data_latent.shape[0], -1)
        
        # reshape
        N, D = x_t_proj.shape
        M, D = ref_data_proj.shape
        _, C, H, W = ref_data_latent.shape
        x_t_proj = x_t_proj.reshape(1, -1, D)
        ref_data_proj = ref_data_proj.reshape(1, -1, D)
        
        # data augmentation for denominator
        ref_data_org = ref_data_latent.reshape(M, -1)
        ones = torch.ones(M)[..., None].to(ref_data_org.device)
        
        # empicial denoiser by kernel tricks
        kernel = - (torch.cdist(x_t_proj, ref_data_proj)[0]).reshape(N, M, 1).repeat(1, 1, C*H*W+1) / (2. * sigma ** 2) #- 2. * np.pi * (sigma ** 2)
        kernel = (kernel.exp() * (torch.cat((ref_data_org, ones), dim=1).reshape(1, -1, C*H*W+1))).sum(dim=1)
        
        # Avoid dividing by zero or very small numbers
        denominator = kernel[:, -1].reshape(-1, 1) + self.epsilon  # Add epsilon to prevent division by zero
        nominator = kernel[:, :-1]

        # Compute negative_score with enhanced stability
        negative_score = nominator / denominator
        negative_score_item = negative_score.clamp(min=-1e10, max=1e10).mean().item()
        
        # reshape
        negative_score = negative_score.reshape(-1, C, H, W)
        return negative_score, {"negative_score_item" : negative_score_item, "denominator" : denominator.item(), 
                                "nominator" : nominator}
        
    def empirical_beta(self, sigma=1.0, quantitle=0.25, **kwargs):
        print("*"*10, f"Set Beta Thresholds", "*"*10)
        # container
        results = {}
        
        # Embedding for kernel distance
        x_t = self.get_noisy_proj_refs()
        ref_data = self.get_proj_ref()
        
        # Compute \beta for each time step
        for t, latents in x_t.items():
            print(f"[Empirical Betas] Computing empirical beta for {t}-th step")
                    
            # Embedding for kernel distance
            x_t_proj = latents.reshape(latents.shape[0], -1)
            ref_data_latent = ref_data.clone().detach()
            ref_data_proj = ref_data_latent.reshape(ref_data_latent.shape[0], -1)
            
            # reshape
            N, D = x_t_proj.shape
            M, D = ref_data_proj.shape
            _, C, H, W = ref_data_latent.shape
            x_t_proj = x_t_proj.reshape(1, -1, D)
            ref_data_proj = ref_data_proj.reshape(1, -1, D)
            
            # empicial betas by kernel tricks
            kernel = - (torch.cdist(x_t_proj, ref_data_proj)[0]).reshape(N, M, 1) / (2. * sigma ** 2) #- 2. * np.pi * (sigma ** 2)
            beta = kernel.exp().sum(dim=(1,2)) + self.epsilon
            
            # Avoid dividing by zero or very small numbers
            q1_val_beta = torch.quantile(beta, quantitle)
            print(f"Top {100*(1-quantitle):.1f} % of radius at t={t}: {q1_val_beta.item():.3f}")
            results[t] = q1_val_beta
        return results
            
@register_conditioning_method(name='sparse')
class SparseRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, forward_fn, num_timesteps, max_idx, beta_min, beta_max, **kwargs):
        super().__init__(ref_data, embed_fn, forward_fn, num_timesteps, max_idx, beta_min, beta_max, **kwargs)
        
        # params for sparse repellency
        self.radius = kwargs.get('radius', -1.0)
        self.scale = kwargs.get('scale', 1.0) # overcompensation
        
        # cached for noisy_proj_refs (# of noisy_proj_refs greater than # of proj_refs)
        if self.radius <= 0:
            if self.cache_proj_beta_ref:
                self.noisy_proj_refs = self.import_proj_ref(self.proj_beta_ref_path)
                
            else:
                # compute radius reference
                scheduler = kwargs.get("scheduler", None) 
                assert scheduler != None, "We need scheduler for computing \beta reference"
                self.noisy_proj_refs = self.set_noisy_proj_ref(scheduler, self.num_timesteps)
                
            self.noisy_refs_beta_quantitle = self.empirical_radius(quantitle=self.quantile)   

            # set radius (pick the minimum timesteps (t=0) of radius in reference)
            self.radius = self.noisy_refs_beta_quantitle[list(self.noisy_refs_beta_quantitle.keys())[-1]]
            
            # del
            del self.noisy_proj_refs, self.noisy_refs_beta_quantitle
            torch.cuda.empty_cache()
                
    def find_neighbors_within_radius(self, x_0_hat):
        ref_data_latent = self.get_proj_ref() # [N, C, emb_H, emb_W]
        
        if x_0_hat.dim() != ref_data_latent.dim():
            x_0_hat = x_0_hat.unsqueeze(0)
        
        diff = x_0_hat - ref_data_latent
        
        # condition
        repellency_data_idx = torch.norm(diff, p=2, dim=(1,2,3)) < self.radius
        
        # find neighbors
        repellency_data = ref_data_latent[repellency_data_idx]
        return repellency_data
        
    def repellency_force(self, x_0_hat, **kwargs):
        repellency_data = self.find_neighbors_within_radius(x_0_hat)
        diff_vec = x_0_hat.unsqueeze(1) - repellency_data.unsqueeze(0) #[N, M, C, H, W]
        weight = torch.norm(diff_vec, p=2 ,dim=(2,3,4))
        trunc_weight = F.relu((self.radius/weight) - 1.)
        
        # Dimension matching
        trunc_weight = trunc_weight[..., None, None, None]
        repellency_term = (diff_vec * trunc_weight).sum(dim=1)
        return repellency_term, {"repellency_force" : repellency_term.norm(p=2).item(), "trunc_weight" : trunc_weight}
    
    def empirical_denoiser(self, x_0_hat, **kwargs):
        '''wrapped function'''
        result = self.repellency_force(x_0_hat, **kwargs)
        return result
    
    def conditioning_1(self, x_0_hat, **kwargs):
        repellency_term, repellency_kwargs = self.repellency_force(x_0_hat, **kwargs)
        x_0_hat += self.scale * repellency_term
        
        if repellency_kwargs["trunc_weight"].sum() == 0.0:
            is_negation = False
        else:
            is_negation = True
        return {"x_0_hat" : x_0_hat, "mean_x_0_hat" : repellency_kwargs["repellency_force"], "is_negation" : is_negation}
    
    '''wrapping function (for sparse repellency)'''
    def conditioning_threshold(self, x_0_hat, **kwargs):    
        results = self.conditioning_1(x_0_hat, **kwargs)
        return results
    
    def empirical_radius(self, quantitle=0.25, **kwargs):
        print("*"*10, f"Set Radius Thresholds", "*"*10)
        # container
        results = {}
        
        # Embedding for kernel distance
        x_t = self.get_noisy_proj_refs()
        ref_data = self.get_proj_ref()
        
        with torch.no_grad():
            # Compute \beta for each time step
            for t, latents in x_t.items():
                print(f"[Empirical Betas] Computing empirical radius for {t}-th step")
                container = []
                
                # compute euclidean distance
                for latents_i in latents:
                    diff = latents_i.unsqueeze(0) - ref_data
                    distance = torch.norm(diff, p=2, dim=(1,2,3)) 
                    container.append(distance)
                    
                    del distance
                    torch.cuda.empty_cache()
            
                distances = torch.cat(container, 0) # [N, 1]
                # Avoid dividing by zero or very small numbers
                q1_val_beta = torch.quantile(distances, quantitle)
                print(f"Top {100*(1-quantitle):.1f} % of beta at t={t}: {q1_val_beta.item():.3f}")
                results[t] = q1_val_beta
        return results
    
