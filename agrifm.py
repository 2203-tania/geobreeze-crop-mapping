import torch
import torch.nn as nn
import sys
from geobreeze.engine.model import EvalModelWrapper


class AgriFM(EvalModelWrapper):

    def _load_encoder(self, blk_indices, checkpoint_path, agrifm_repo_path):
        # Mock mmcv before importing AgriFM to avoid mmcv dependency
        sys.path.insert(0, '/mnt/tania/geobreeze')
        import mmcv_mock  # installs mock into sys.modules

        sys.path.insert(0, agrifm_repo_path)
        sys.path.insert(0, f'{agrifm_repo_path}/AgriFM')

        from models.video_swin_transformer import PretrainingSwinTransformer3DEncoder
        from mmseg.registry.registry import MODELS

        patch_emd_cfg = dict(
            type='SwinPatchEmbed3D',
            patch_size=(2, 4, 4),
            in_chans=10,
            embed_dim=128,
        )
        backbone_cfg = dict(
            type='SwinTransformer3D',
            pretrained=None,
            pretrained2d=False,
            patch_size=(2, 4, 4),
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=(8, 7, 7),
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=False,
            frozen_stages=-1,
            use_checkpoint=False,
            downsample_steps=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
            feature_fusion='cat',
            mean_frame_down=True,
        )

        encoder = MODELS.build(dict(
            type='PretrainingSwinTransformer3DEncoder',
            patch_emd_cfg=patch_emd_cfg,
            backbone_cfg=backbone_cfg,
        ))

        # Load checkpoint with key remapping: S2_patch_emd -> patch_emd
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('S2_patch_emd'):
                new_k = k.replace('S2_patch_emd', 'patch_emd')
            elif k.startswith('HLSL30_patch_emd') or k.startswith('Modis_patch_emd'):
                continue
            else:
                new_k = k
            new_state_dict[new_k] = v

        missing, unexpected = encoder.load_state_dict(new_state_dict, strict=False)
        print(f'AgriFM loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}')

        self.encoder = encoder
        self.norm = encoder.backbone.norm
        self.blk_indices = blk_indices

    def get_blocks(self, x_dict):
        x = x_dict['imgs']  # [B, C, H, W]
        x = x.unsqueeze(1)  # [B, 1, C, H, W]
        features_dict = self.encoder(x)
        feat = features_dict['encoder_features']  # [B, D, H', W']
        B, D, Hp, Wp = feat.shape
        feat = feat.flatten(2).transpose(1, 2)  # [B, P, D]
        return [feat] * len(self.blk_indices)

    def default_blocks_to_featurevec(self, block_list):
        return block_list[-1].mean(dim=1)  # [B, D]

    def get_segm_blks(self, x_dict):
        x = x_dict['imgs']  # [B, C, H, W]
        x = x.unsqueeze(1)  # [B, 1, C, H, W]
        features_dict = self.encoder(x)
        feat = features_dict['encoder_features']  # [B, D, H', W']
        return [feat] * len(self.blk_indices)

    def replace_pe(self, num_channels):
        raise NotImplementedError('AgriFM uses SwinPatchEmbed3D with fixed in_chans=10')
