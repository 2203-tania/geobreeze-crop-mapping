import torch
import torch.nn as nn
from geobreeze.engine.model import EvalModelWrapper


class SatlasPretrain(EvalModelWrapper):

    def _load_encoder(self, blk_indices, model_identifier='Sentinel2_SwinB_SI_MS', fpn=False):
        import satlaspretrain_models

        weights = satlaspretrain_models.Weights()
        self.encoder = weights.get_pretrained_model(
            model_identifier, fpn=fpn, device='cpu'
        )
        self.fpn = fpn
        self.blk_indices = blk_indices
        self.norm = nn.Identity()

        # Channel dims for SwinB without FPN: 128, 256, 512, 1024
        # Project all to embed_dim for geobreeze's Feature2Pyramid
        in_dims = [128, 256, 512, 1024]
        self.channel_projections = nn.ModuleList([
            nn.Conv2d(in_dim, self.embed_dim, kernel_size=1)
            for in_dim in in_dims
        ])

    def _drop_b8a(self, x):
        """Drop B8A (index 7) to go from 10 PASTIS bands to 9 SatlasPretrain bands."""
        if x.shape[1] == 10:
            return torch.cat([x[:, :7], x[:, 8:]], dim=1)
        return x

    def _prepare_input(self, x_dict):
        x = x_dict['imgs']
        if x.dim() == 5:
            # (B, T, C, H, W) -> encode each frame, then max-pool over T
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self._drop_b8a(x)
            out = self.encoder(x)
            out = [
                feat.view(B, T, *feat.shape[1:]).max(dim=1).values
                for feat in out
            ]
        else:
            x = self._drop_b8a(x)
            out = self.encoder(x)
        return out

    def get_blocks(self, x_dict):
        out = self._prepare_input(x_dict)
        blocks = []
        for i, idx in enumerate(self.blk_indices):
            feat = out[idx]                              # [B, D, H', W']
            feat = self.channel_projections[idx](feat)  # [B, embed_dim, H', W']
            B, D, H, W = feat.shape
            blocks.append(feat.flatten(2).transpose(1, 2))  # [B, P, D]
        return blocks

    def get_segm_blks(self, x_dict):
        """Returns list of [B, embed_dim, H', W'] — all projected to same dim."""
        out = self._prepare_input(x_dict)
        return [
            self.channel_projections[idx](out[idx])
            for idx in self.blk_indices
        ]

    def default_blocks_to_featurevec(self, block_list):
        return block_list[-1].mean(dim=1)

    def replace_pe(self, num_channels):
        raise NotImplementedError('SatlasPretrain uses fixed channel projection')
