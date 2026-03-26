import torch
import torch.nn as nn
from einops import rearrange

from geobreeze.engine.model import EvalModelWrapper

# PASTIS band ordering: B02(0), B03(1), B04(2), B05(3), B06(4), B07(5), B08(6), B8A(7), B11(8), B12(9)
# OLMoEarth S2_L2A band ordering: B02(0), B03(1), B04(2), B08(3), B05(4), B06(5), B07(6), B8A(7), B11(8), B12(9), B01(10), B09(11)
# Maps PASTIS index → OLMoEarth index
PASTIS_TO_OLMOEARTH_IDX = [0, 1, 2, 4, 5, 6, 3, 7, 8, 9]


class OlmoEarth(EvalModelWrapper):
    """OLMoEarth wrapper for geobreeze evaluation.

    Accepts 10-band PASTIS Sentinel-2 input [B, 10, H, W] and maps it to
    the 12-band S2_L2A format expected by OLMoEarth, zero-padding B01 and B09.
    """

    def _load_encoder(self, blk_indices, model_id='OLMOEARTH_V1_BASE', model_path=None):
        from olmoearth_pretrain.model_loader import ModelID, load_model_from_id, load_model_from_path

        if model_path is not None:
            mae_model = load_model_from_path(model_path)
        else:
            mae_model = load_model_from_id(ModelID[model_id])

        self.encoder = mae_model.encoder
        # Use Identity norm to avoid double-optimization with the encoder's own norm
        self.norm = nn.Identity()
        self.blk_indices = blk_indices

    def _prepare_input(self, x_dict):
        """Convert geobreeze [B, 10, H, W] to OLMoEarth MaskedOlmoEarthSample."""
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample

        x = x_dict['imgs']  # [B, 10, H, W]
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # Build 12-band S2_L2A tensor [B, H, W, T=1, 12] (channels-last)
        s2 = torch.zeros(B, H, W, 1, 12, device=device, dtype=dtype)
        for pastis_idx, olmo_idx in enumerate(PASTIS_TO_OLMOEARTH_IDX):
            s2[:, :, :, 0, olmo_idx] = x[:, pastis_idx, :, :]

        # Dummy timestamps [B, T=1, 3] = [day=15, month=6, year=2020]
        # Must be Long because month is used as an embedding index
        timestamps = torch.zeros(B, 1, 3, device=device, dtype=torch.long)
        timestamps[:, 0, 0] = 15    # day
        timestamps[:, 0, 1] = 6     # month (0-indexed)
        timestamps[:, 0, 2] = 2020  # year

        # Mask: all ONLINE_ENCODER (0), same shape as s2
        mask = torch.zeros_like(s2)

        return MaskedOlmoEarthSample(
            timestamps=timestamps,
            sentinel2_l2a=s2,
            sentinel2_l2a_mask=mask,
        )

    def _run_encoder(self, x_dict):
        """Run encoder and return spatial features [B, H', W', D]."""
        sample = self._prepare_input(x_dict)
        result = self.encoder(sample, patch_size=4)
        tam = result['tokens_and_masks']

        # tam.sentinel2_l2a: [B, H', W', T=1, band_sets=3, D]
        feat = tam.sentinel2_l2a       # [B, H', W', 1, 3, D]
        feat = feat[:, :, :, 0, :, :]  # [B, H', W', 3, D]
        feat = feat.mean(dim=3)        # [B, H', W', D]
        return feat

    def get_blocks(self, x_dict):
        """Return final encoder tokens as list of [B, P, D]."""
        feat = self._run_encoder(x_dict)
        B, H, W, D = feat.shape
        tokens = feat.reshape(B, H * W, D)  # [B, P, D]
        return [tokens]

    def default_blocks_to_featurevec(self, block_list):
        """Mean-pool spatial tokens to get [B, D] feature vector."""
        return block_list[-1].mean(dim=1)

    def get_segm_blks(self, x):
        """Return 4 copies of spatial feature [B, D, H', W'] for FPN."""
        feat = self._run_encoder(x)
        feat = rearrange(feat, 'b h w d -> b d h w')  # [B, D, H', W']
        return [feat, feat, feat, feat]
