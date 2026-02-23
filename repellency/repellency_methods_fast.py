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
        self.scale = kwargs.get('scale', 1.0)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        
        self.proj_ref_path = kwargs.get('proj_ref_path', None)
        self.cache_proj_ref = kwargs.get('cache_proj_ref', False)
        
        if self.cache_proj_ref:
            self.proj_refs = self.import_proj_ref(self.proj_ref_path)
        else:
            self.proj_refs = self.set_proj_ref()
        
    @torch.no_grad()
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
            
            # type casting
            results = results.float()
            return results
        else:
            results = self.embed_fn(data)
            
            # normalization
            out = torch.norm(results, dim=1, keepdim=True)
            results = results / out
            
            # type casting
            results = results.float()
            return results
        
    def discrete_to_continous_time(self, idx, **kwargs):
        if idx == 0:
            result = 0.001
        else:
            result = idx / self.max_idx
        return result
    
    def sigma_edm(self, cont_time, **kwargs):
        return torch.sqrt(torch.exp(0.5*self.beta_max*cont_time**2 + self.beta_min * cont_time) - 1.)
    
    def sigma(self, cont_time, **kwargs):
        pass
        
    def mkdir_cache(self):
        dir_path = os.path.split(self.proj_ref_path)[0]
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
        self.mkdir_cache()
        torch.save(result, self.proj_ref_path)

        # map location (cuda)
        result = result.to('cuda')
        return result

    def import_proj_ref(self, proj_ref_path):
        result = torch.load(proj_ref_path, map_location=self.ref_data.device)
        return result
        
    def get_proj_ref(self):
        return self.proj_refs
    
    @abstractmethod
    def empirical_denoiser(self, data, **kwargs):
        pass
   
    def conditioning(self, x_0_hat, **kwargs):
        if x_0_hat.dtype != self.ref_data.dtype:
            x_0_hat = x_0_hat.to(self.ref_data.dtype)
        
        if kwargs.get("guidance_scale", None) is not None and kwargs.get("guidance_scale", None) > 0.0:
            return self.conditioning_2(x_0_hat, **kwargs)
        else:
            return self.conditioning_1(x_0_hat, **kwargs)
            
    def conditioning_1(self, x_0_hat, **kwargs):
        negative_x_0_hat, negative_x_0_hat_item = self.empirical_denoiser(x_t=x_0_hat, **kwargs)
        x_0_hat -= self.scale * negative_x_0_hat
        return {"x_0_hat" : x_0_hat, "mean_x_0_hat" : negative_x_0_hat_item}
    
    def conditioning_2(self, x_0_hat, **kwargs):
        negative_x_0_hat, negative_x_0_hat_item = self.empirical_denoiser(x_t=x_0_hat, **kwargs)
        x_0_hat -= negative_x_0_hat
        return {"x_0_hat" : negative_x_0_hat, "mean_x_0_hat" : negative_x_0_hat_item}
            

@register_conditioning_method(name='euclidean')
class EuclideanRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, forward_fn, max_idx, beta_min, beta_max, **kwargs):
        super().__init__(ref_data, embed_fn, forward_fn, max_idx, beta_min, beta_max, **kwargs)
        self.scale = kwargs.get('scale', 1.0)

    def empirical_denoiser(self, x_t, time, sigma=1.0, **kwargs):
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
        self.scale = kwargs.get('scale', 1.0)
        
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
        return negative_score, negative_score_item    
    
@register_conditioning_method(name='random_noise')
class RandomNoiseRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, forward_fn, num_timesteps, max_idx, beta_min, beta_max, **kwargs):
        super().__init__(ref_data, embed_fn, forward_fn, num_timesteps, max_idx, beta_min, beta_max, **kwargs)
        self.scale = kwargs.get('scale', 1.0)
        
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
        
        # Random Noise
        negative_score = torch.randn(size=(1, C*H*W)).to(ref_data_latent.device)
        negative_score_item = negative_score.clamp(min=-1e10, max=1e10).mean().item()
        
        # reshape
        negative_score = negative_score.reshape(-1, C, H, W)
        return negative_score, negative_score_item    

@register_conditioning_method(name='sparse')
class SparseRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, forward_fn, num_timesteps, max_idx, beta_min, beta_max, **kwargs):
        super().__init__(ref_data, embed_fn, forward_fn, num_timesteps, max_idx, beta_min, beta_max, **kwargs)
        self.radius = kwargs.get('radius', 1.0)
        self.scale = kwargs.get('scale', 1.0) # overcompensation

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
        return repellency_term, repellency_term.norm(p=2).item()
    
    def empirical_denoiser(self, x_0_hat, **kwargs):
        '''wrapped function'''
        result = self.repellency_force(x_0_hat, **kwargs)
        return result
    
    def conditioning_1(self, x_0_hat, **kwargs):
        repellency_term, repellency_term_item = self.repellency_force(x_0_hat, **kwargs)
        x_0_hat += self.scale * repellency_term
        return {"x_0_hat" : x_0_hat, "mean_x_0_hat" : repellency_term_item}
    
@register_conditioning_method(name='lsh')
class LSHRepellency(RepellencyMethod):
    def __init__(self, ref_data, embed_fn, forward_fn, max_idx, beta_min, beta_max, **kwargs):
        super().__init__(ref_data, embed_fn, forward_fn, max_idx, beta_min, beta_max, **kwargs)
        self.omega = kwargs.get('omega', 1.0)
        self.pca_dim = kwargs.get('pca_dim', 5)
        self.kappa = kwargs.get('kappa', 6)
        self.num_buckets = kwargs.get('num_buckets', 10)
        self.num_hashtables = kwargs.get("num_hashtables", 100)
        self.data_dim = kwargs.get("data_dim", 3)
        self.filename = kwargs.get("filename")
        
        # PCA
        self.pca, self.pca_reduced_data = self.init_pca(ref_data)
        self.lsh = self.init_lsh(ref_data, self.pca_reduced_data, 
                                 self.filename, ref_data.device)
        
    def init_pca(self, data):
        device = data.device
        pca = PCA(n_components=self.pca_dim)
        if type(data) == torch.Tensor:
            data = data.cpu().numpy()
            data_flatten = data.reshape(data.shape[0], -1)
        
        # fit
        pca.fit(data_flatten) # principal components
        
        # transformed data (reconstruct)
        pca_reduced_data = torch.from_numpy(pca.transform(data_flatten)).to(device)
        print(f"PCA components/variance: {np.sum(pca.explained_variance_ratio_)}")
        return pca, pca_reduced_data
    
    def init_lsh(self, data, pca_reduced_data, filename, device):
        lsh = LSHash(self.kappa, self.pca_dim, num_hashtables=self.num_hashtables, 
                      max_buckets=self.num_buckets, omega=self.omega, data_dim=self.data_dim, 
                      hashtable_filename=filename, device=device)
        
        
        
        data_flatten = data.reshape(data.shape[0], -1)
        
        x_one_cats = torch.cat((data_flatten.t(), torch.ones(1, data.shape[0]).to(device)), dim=0)
        lsh.index(x_one_cats, feature=pca_reduced_data.t(), retrieve_samples=False)
        return lsh
        
    def apply_rff(self, m, sigma):
        return torch.sqrt(2) * torch.cos(torch.dot(m / torch.sqrt(2) / sigma, self.rff) + self.rff_shift) # m = (n_data, dimension), rff = (dimension, num_features), rff_shift = (1, num_features)

    def apply_rff_sin(self, m, sigma):
        return torch.sqrt(2) * torch.sin(torch.dot(m / torch.sqrt(2) / sigma, self.rff) + self.rff_shift) # m = (n_data, dimension), rff = (dimension, num_features), rff_shift = (1, num_features)

    def non_private_kde(self, x_0_hat, ref_imgs, sigma):
        return torch.dot(self.apply_rff(ref_imgs, sigma), self.apply_rff(x_0_hat, sigma).T) / self.num_feature  # rff_kde = (n_data, num_features), rff_kde_queries = (n_query, num_features)
    
    def empirical_denoiser(self, x_0_hat, time, sigma=1.0, **kwargs):
        # shape
        M, C, H, W = self.ref_data.shape
        N, _, _, _ = x_0_hat.shape
        
        # reshape
        x_0_hat_proj = x_0_hat.reshape(-1, C*H*W)
        ref_data_proj = self.ref_data.reshape(-1, C*H*W)
        
        # denominator
        ones = torch.ones(M)[..., None].to(ref_data_proj.device)
        
        # denominator
        ref_data_org = self.ref_data.reshape(M, -1)
        ones = torch.ones(M)[..., None].to(ref_data_org.device)
        
        # empicial denoiser by kernel tricks
        kernel = - (torch.cdist(x_0_hat_proj, ref_data_proj)[0] ** 2).reshape(N, M, 1).repeat(1, 1, C*H*W+1) / (2. * sigma ** 2) #- 2. * np.pi * (sigma ** 2)
        kernel = (kernel.exp() * (torch.cat((ref_data_org, ones), 1).reshape(1, -1, C*H*W+1))).sum(dim=1)
        
        #############
        # LOGSUMEXP #
        #############
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