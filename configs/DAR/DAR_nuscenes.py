_base_ = [
     '../_base_/models/lss_swin_nuscenes.py','../_base_/datasets/nuscenes_without_occlusion.py',
    '../_base_/default_runtime.py','../_base_/schedules/schedule_20k_LSSnusecenes.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='force_lss_BEVSegmentor',
    pretrained='./configs/_base_/models/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0,1,2,3,),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.0,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        output_missing_index_as_none=True),
    transformer = dict(type='DAR',use_light = True),
    decode_head=dict(
        type='DAR_head',
        num_classes=14,
        align_corners=True),
    test_cfg=dict(mode='whole',output_type='iou',positive_thred=0.5)
    )

load_from = r"./models_dir/lss_pon_bs_nuscenes/iter_88000.pth"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        reduce_zero_label=False,
        with_calib=True,
        imdecode_backend='pyramid'),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        resize_gt=False,
        keep_ratio=False),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'img_norm_cfg', 'calib'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        reduce_zero_label=False,
        with_calib=True,
        imdecode_backend='pyramid'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False, resize_gt=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg', 'gt_semantic_seg',
                           'calib'))
        ])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        type='NuscenesDataset',
        data_root='./data/nuscenes/',
        img_dir='train/img_dir/',
        ann_dir='train/ann_bev_dir/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                reduce_zero_label=False,
                with_calib=True,
                imdecode_backend='pyramid'),
            dict(
                type='Resize',
                img_scale=(1024, 1024),
                resize_gt=False,
                keep_ratio=False),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_semantic_seg'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor',
                           'img_norm_cfg', 'calib'))
        ]),
    val=dict(
        type='NuscenesDataset',
        data_root='./data/nuscenes/',
        img_dir='val/img_dir',
        ann_dir='val/ann_bev_dir',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                reduce_zero_label=False,
                with_calib=True,
                imdecode_backend='pyramid'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False, resize_gt=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                   'img_shape', 'pad_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'img_norm_cfg',
                                   'gt_semantic_seg', 'calib'))
                ])
        ]),
    test=dict(
        type='NuscenesDataset',
        data_root='./data/nuscenes/',
        img_dir='img_dir/test',
        ann_dir='ann_bev_dir/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                reduce_zero_label=False,
                with_calib=True,
                imdecode_backend='pyramid'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False, resize_gt=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=('filename', 'ori_filename', 'ori_shape',
                                   'img_shape', 'pad_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'img_norm_cfg',
                                   'gt_semantic_seg', 'calib'))
                ])
        ]))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00018,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
