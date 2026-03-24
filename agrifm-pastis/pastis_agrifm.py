# AgriFM config for PASTIS crop type mapping
custom_imports = dict(imports=['AgriFM'], allow_failed_imports=False)

img_size = 128
num_frames = 32
num_classes = 21
swin_in_embed_dim = 128
embed_dim = 2048

pretrained_weights_path = '/mnt/tania/AgriFM/AgriFM/datasets/example_dataset/AgriFM.pth'
data_path = '/mnt/tania/pastis_agrifm/h5_samples'
data_list_path = '/mnt/tania/pastis_agrifm/data_list'
work_dir = '/mnt/tania/agrifm_pastis_run'

mean = {'S2': [4179.192015478227, 4065.9106675194444, 3957.274910960156, 5207.452475253116,
               4327.12234687, 4873.16102239, 5049.1637925, 5111.07806856, 3056.86349163, 2490.9675032]}
std  = {'S2': [4041.5212325268735, 3691.003119315892, 3629.331318356375, 2973.5178530908756,
               3569.73343885, 3085.9151435, 2937.56005119, 2806.04462314, 1808.30013156, 1694.20220774]}

train_pipelines = [
    dict(type='MapNormalize', mean=mean, std=std),
    dict(type='MapResize', size={'S2': (img_size, img_size), 'label': (img_size, img_size)}),
]
valid_pipelines = [
    dict(type='MapNormalize', mean=mean, std=std),
    dict(type='MapResize', size={'S2': (img_size, img_size), 'label': (img_size, img_size)}),
]

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='MappingDataset',
        data_toot_path=data_path,
        data_list_file=data_list_path + '/train.txt',
        data_pipelines=train_pipelines,
        data_keys=('S2',),
        label_key='label',
    ))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='MappingDataset',
        data_toot_path=data_path,
        data_list_file=data_list_path + '/val.txt',
        data_pipelines=valid_pipelines,
        data_keys=('S2',),
        label_key='label',
    ))

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='MappingDataset',
        data_toot_path=data_path,
        data_list_file=data_list_path + '/test.txt',
        data_pipelines=valid_pipelines,
        data_keys=('S2',),
        label_key='label',
    ))

model = dict(
    type='MultiUnifiedModel',
    encoders=dict(
        type='MultiModalEncoder',
        encoders_cfg=dict(
            S2=dict(
                type='PretrainingSwinTransformer3DEncoder',
                patch_emd_cfg=dict(
                    type='SwinPatchEmbed3D',
                    patch_size=(2, 4, 4),
                    in_chans=10,
                    embed_dim=swin_in_embed_dim,
                ),
                backbone_cfg=dict(
                    type='SwinTransformer3D',
                    pretrained=None,
                    pretrained2d=False,
                    patch_size=(2, 4, 4),
                    embed_dim=swin_in_embed_dim,
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
                ),
                init_cfg=dict(
                    type='pretrained',
                    checkpoint=pretrained_weights_path,
                    revise_keys=[('S2_patch_emd', 'patch_emd')],
                ),
            ),
        ),
    ),
    neck=dict(
        type='MultiFusionNeck',
        embed_dim=embed_dim,
        in_feature_key=('S2',),
        feature_size=(4, 4),
        out_size=(img_size, img_size),
        in_fusion_key_list=(
            {'S2': 2048},
            {'S2': 2048},
            {'S2': 2048},
        ),
    ),
    head=dict(
        type='CropFCNHead',
        num_classes=num_classes,
        embed_dim=embed_dim,
        loss_model=dict(type='CropCEloss'),
    ),
)

resume = False
optimizer = dict(type='AdamW', lr=6e-5, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=1e-6, by_epoch=False, begin=500),
]

log_processor = dict(by_epoch=True)
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='CropIoUMetric', iou_metrics=['mFscore', 'mIoU'],
                     num_classes=num_classes, ignore_index=-1)
test_cfg = dict(type='TestLoop')
test_evaluator = dict(type='CropIoUMetric', iou_metrics=['mFscore', 'mIoU'],
                      num_classes=num_classes, ignore_index=-1)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1,
                    max_keep_ckpts=1, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
