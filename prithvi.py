import torch
import torch.nn as nn
import sys
import json
from einops import rearrange
from geobreeze.engine.model import EvalModelWrapper


class Prithvi(EvalModelWrapper):

    def _load_encoder(self, blk_indices, model_dir, checkpoint_name='Prithvi_EO_V2_600M_TL.pt'):
        sys.path.insert(0, model_dir)
        from prithvi_mae import PrithviMAE

        with open(f'{model_dir}/config.json') as f:
            cfg = json.load(f)['pretrained_cfg']

        model = PrithviMAE(**cfg)
        ckpt  = torch.load(f'{model_dir}/{checkpoint_name}',
                           map_location='cpu', weights_only=False)
        state = ckpt.get('state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'Prithvi loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}')

        self.encoder  = model
        self.norm     = model.encoder.norm
        self.blk_indices = blk_indices

        # PASTIS bands B02-B07 are indices 0-5 in the 10-band array
        self.band_indices = [0, 1, 2, 3, 4, 5]

    def _prepare_input(self, x_dict):
        """
        geobreeze passes [B, C, H, W] (10-band, mean-collapsed).
        Prithvi expects [B, C, T, H, W] with C=6, T=num_frames.
        We select 6 bands and fake T=1.
        """
        x = x_dict['imgs']  # [B, 10, H, W]
        x = x[:, self.band_indices, :, :]  # [B, 6, H, W]
        x = x.unsqueeze(2)  # [B, 6, 1, H, W] — fake T=1
        return x

    def get_blocks(self, x_dict):
        x = self._prepare_input(x_dict)
        blocks = self.encoder.forward_features(x)  # list of 32 x [B, 1025, 1280]
        return [blocks[i] for i in self.blk_indices]

    def default_blocks_to_featurevec(self, block_list):
        # Use cls token (index 0) from last block
        return self.norm(block_list[-1])[:, 0, :]

    def get_segm_blks(self, x_dict):
        """
        Standard ViT segmentation — drop cls token, reshape to [B, D, H, W].
        With T=1 and patch_size=14, 224/14=16 patches per side → 16*16=256 patches.
        """
        x = self._prepare_input(x_dict)
        blocks = self.encoder.forward_features(x)
        selected = [blocks[i] for i in self.blk_indices]

        # Drop cls token and reshape to spatial
        patch_size = int((selected[0].size(1) - 1) ** 0.5)
        out = []
        for blk in selected:
            b = blk[:, 1:, :]  # drop cls token [B, P, D]
            b = rearrange(b, 'b (h w) d -> b d h w', h=patch_size, w=patch_size)
            out.append(b)
        return out

    def replace_pe(self, num_channels):
        raise NotImplementedError('Prithvi uses fixed 6-band input')
