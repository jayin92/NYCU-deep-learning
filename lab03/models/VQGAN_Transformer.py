import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, _ = self.vqgan.encode(x)

        return codebook_mapping, codebook_indices.reshape(codebook_mapping.shape[0], -1)
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            def linear_func(x):
                return 1 - x
            return linear_func
        elif mode == "cosine":
            def cosine_func(x):
                return np.cos(np.pi * x / 2)
            return cosine_func            
        elif mode == "square":
            def square_func(x):
                return 1 - x ** 2
            return square_func   
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        _, z_indices = self.encode_to_z(x)  # ground truth
        mask_tokens = torch.full_like(z_indices, self.mask_token_id)  # mask token
        mask = torch.bernoulli(torch.full(z_indices.shape, 0.5)).bool()  # mask ratio

        new_z_indices = z_indices.clone()
        new_z_indices[mask] = mask_tokens[mask]

        logits = self.transformer(new_z_indices)  # transformer predict the probability of tokens
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask, mask_num, ratio, mask_func):

        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        ##At the end of the decoding process, add back the original(non-masked) token values
        
        masked_z_indices = z_indices.clone()
        masked_z_indices[mask] = self.mask_token_id

        logits = self.transformer(masked_z_indices) # B x num_image_tokens x num_codebook_vectors
        # Apply softmax to convert logits into a probability distribution across the last dimension.
        probs = logits.softmax(dim=-1)
        # Get the predicted probabilities for the masked tokens
        z_indices_predict = torch.distributions.Categorical(logits=logits).sample()  # sample from the predicted distribution
        while torch.any(z_indices_predict == self.mask_token_id):
            z_indices_predict = torch.distributions.Categorical(logits=logits).sample()

        # FIND MAX probability for each token value
        z_indices_predict[~mask] = z_indices[~mask]  # keep the original tokens for unmasked positions
        z_indices_predict_prob = probs.gather(-1, z_indices_predict.unsqueeze(-1)).squeeze(-1)  # B x num_image_tokens
        z_indices_predict_prob = torch.where(mask, z_indices_predict_prob, torch.full_like(z_indices_predict_prob, float('inf')))  # set unmasked positions to inf

        mask_ratio = self.gamma_func(mask_func)(ratio)  # apply the mask function to get the ratio
        print(f"mask ratio: {mask_ratio}")
        # Calculate the number of masked tokens based on the ratio
        mask_len = int(mask_num * mask_ratio)  # number of masked tokens
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = torch.distributions.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices_predict_prob.device)  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        sorted_confidence, _ = torch.sort(confidence, dim=-1)  # sort the confidence scores
        threshold = sorted_confidence[:, mask_len].unsqueeze(-1)
        mask_bc = confidence < threshold  # new mask has confiedence less than the threshold
        # Set the masked tokens to the mask token id

        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
