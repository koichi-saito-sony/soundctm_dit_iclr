import typing as tp
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from einops import rearrange
from torch import nn
from torch.nn import functional as F
from x_transformers import ContinuousTransformerWrapper, Encoder
from .blocks import FourierFeatures
from .transformer import ContinuousTransformer

class CTMDiffusionTransformer(nn.Module):
    def __init__(self, 
        io_channels=32, 
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["x-transformers", "continuous_transformer"] = "x-transformers",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        time_embedding_type: str = "positional",
        **kwargs):

        super().__init__()
        
        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        timestep_features_dim = 256

        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        if cond_token_dim > 0:
            # Conditioning tokens

            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
        else:
            cond_embed_dim = 0

        # COMMENTED THIS PART OF THE CODE DUE TO THE BELOW ERROR (SINCE WE ARE ALSO NOT USING GLOBAL EMBEDDINGS)
        # RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. 
        #               This error indicates that your module has parameters that were not used in producing loss. 
        # if global_cond_dim > 0:
        #     # Global conditioning
        #     global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
        #     self.to_global_embed = nn.Sequential(
        #         nn.Linear(global_cond_dim, global_embed_dim, bias=False),
        #         nn.SiLU(),
        #         nn.Linear(global_embed_dim, global_embed_dim, bias=False)
        #     )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size

        # Transformer

        self.transformer_type = transformer_type

        self.global_cond_type = global_cond_type

        if self.transformer_type == "x-transformers":
            self.transformer = ContinuousTransformerWrapper(
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                max_seq_len=0, #Not relevant without absolute positional embeds
                attn_layers = Encoder(
                    dim=embed_dim,
                    depth=depth,
                    heads=num_heads,
                    attn_flash = True,
                    cross_attend = cond_token_dim > 0,
                    dim_context=None if cond_embed_dim == 0 else cond_embed_dim,
                    zero_init_branch_output=True,
                    use_abs_pos_emb = False,
                    rotary_pos_emb=True,
                    ff_swish = True,
                    ff_glu = True,
                    **kwargs
                )
            )

        elif self.transformer_type == "continuous_transformer":

            global_dim = None

            if self.global_cond_type == "adaLN":
                # The global conditioning is projected to the embed_dim already at this point
                global_dim = embed_dim

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                global_cond_dim=global_dim,
                **kwargs
            )
             
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def _forward(
        self, 
        x, 
        t,
        s_timestep, # newly added for ctm
        embedd_cfg, # newly added for ctm
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        **kwargs):

        if cross_attn_cond is not None:
            # print("Going through if 1...........")
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)
        

        # COMMENTED THIS PART OF THE CODE DUE TO THE BELOW ERROR (SINCE WE ARE ALSO NOT USING GLOBAL EMBEDDINGS)        # RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. 
        #               This error indicates that your module has parameters that were not used in producing loss. 
        # if global_embed is not None:
        #     # Project the global conditioning to the embedding dimension
        #     global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)
            
            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

        if input_concat_cond is not None:
            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None])) # (b, embed_dim)
        s_timestep_embed = self.to_timestep_embed(self.timestep_features(s_timestep[:, None])) # (b, embed_dim)

        # COMMENTED THIS PART OF THE CODE DUE TO THE BELOW ERROR (SINCE WE ARE ALSO NOT USING GLOBAL EMBEDDINGS)        # RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. 
        #               This error indicates that your module has parameters that were not used in producing loss. 
        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists
        # if global_embed is not None:
            
        #     global_embed = global_embed + timestep_embed
        # else:
        # MOVED BELOW LINE OUT OF ELSE LOOP         
        global_embed = timestep_embed

        global_embed = global_embed + s_timestep_embed
        global_embed = global_embed + embedd_cfg

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend":
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        if self.transformer_type == "x-transformers":
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, **extra_args, **kwargs)
        elif self.transformer_type == "continuous_transformer":
            output = self.transformer(x, 
                                       prepend_embeds=prepend_inputs,
                                       context=cross_attn_cond, 
                                       context_mask=cross_attn_cond_mask, 
                                       mask=mask, 
                                       prepend_mask=prepend_mask, 
                                       return_info=return_info, **extra_args, **kwargs)

            if return_info:
                output, info = output
        elif self.transformer_type == "mm_transformer":
            output = self.transformer(x, context=cross_attn_cond, mask=mask, context_mask=cross_attn_cond_mask, **extra_args, **kwargs)

        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        if return_info:
            return output, info

        return output
