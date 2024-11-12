# dataset settings
dataset_type = 'NuscenesDataset'
data_root = "./data/nuscenes/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False, with_calib=True, imdecode_backend='pyramid'),
    dict(type='Resize', img_scale=(800, 600), resize_gt=False, keep_ratio=False),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'calib')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False, with_calib=True, imdecode_backend='pyramid'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 600),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False, resize_gt=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'gt_semantic_seg', 'calib')),
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/img_dir',
        ann_dir='train/ann_bev_dir',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/img_dir',
        ann_dir='val/ann_bev_dir',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/img_dir',
        ann_dir='val/ann_bev_dir',
        pipeline=test_pipeline))
