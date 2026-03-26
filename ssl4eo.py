import torch
import torch.nn as nn
from geobreeze.engine.model import EvalModelWrapper


# PASTIS S2 bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
# SSL4EO all-13 S2 order: B01(0), B02(1), B03(2), B04(3), B05(4), B06(5),
#                          B07(6), B08(7), B8A(8), B09(9), B10(10), B11(11), B12(12)
PASTIS_TO_S2_ALL13 = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]


class SSL4EO(EvalModelWrapper):
    """
    Wrapper for SSL4EO-S12 pretrained ViT-S/16 models from torchgeo.

    Available weights_id values:
        SENTINEL2_ALL_DINO  - ViT-S/16 trained with DINO on all 13 S2 bands
        SENTINEL2_ALL_MOCO  - ViT-S/16 trained with MoCo on all 13 S2 bands

    Input: PASTIS S2 (10 bands). Missing bands B01, B09, B10 are zero-padded
    to match the model's expected 13-channel input.
    """

    def _load_encoder(self, blk_indices, weights_id='SENTINEL2_ALL_DINO'):
        from torchgeo.models import ViTSmall16_Weights, vit_small_patch16_224

        weights = getattr(ViTSmall16_Weights, weights_id)
        model = vit_small_patch16_224(weights=weights, dynamic_img_size=True)

        # Detach norm from encoder to avoid double-optimization
        self.norm = model.norm
        model.norm = nn.Identity()

        self.encoder = model
        self.blk_indices = blk_indices

    def _prepare_input(self, x_dict):
        """Zero-pad PASTIS 10-band input to 13-band for SSL4EO."""
        x = x_dict['imgs']  # [B, 10, H, W]
        B, C, H, W = x.shape
        x13 = torch.zeros(B, 13, H, W, device=x.device, dtype=x.dtype)
        x13[:, PASTIS_TO_S2_ALL13, :, :] = x
        return x13

    def get_blocks(self, x_dict):
        x = self._prepare_input(x_dict)
        blocks = self.encoder.get_intermediate_layers(x, n=self.blk_indices)
        return list(blocks)

    def default_blocks_to_featurevec(self, block_list):
        # get_intermediate_layers returns patch tokens only (no CLS) — use mean pool
        return self.norm(block_list[-1]).mean(dim=1)

    def get_segm_blks(self, x):
        """Override: get_intermediate_layers returns patch tokens only, no CLS to drop."""
        from einops import rearrange
        block_list = self.get_blocks(x)
        h = w = int(block_list[0].size(1) ** 0.5)
        out = [rearrange(blk, 'b (h w) d -> b d h w', h=h, w=w)
               for blk in block_list]
        return out
