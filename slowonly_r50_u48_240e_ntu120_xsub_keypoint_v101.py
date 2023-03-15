model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=18,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1),
        with_cp=True),
    cls_head=dict(
        type='I3DHead',
        in_channels=128,
        num_classes=8,
        spatial_type='avg',
        dropout_ratio=0.5),
    train_cfg=dict(),
    test_cfg=dict(average_clips='prob'))
dataset_type = 'PoseDataset'
ann_file_train = '/home/will/Desktop/Thesis/Dataset/annotations/annotation_train.pkl'
ann_file_val = '/home/will/Desktop/Thesis/Dataset/annotations/annotation_val.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(
        type='Flip',
        flip_ratio=0.5,
        left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='UniformSampleFrames', clip_len=48, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=True,
        left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='PoseDataset',
        ann_file=
        '/home/will/Desktop/Thesis/Dataset/annotations/annotation_train.pkl',
        data_prefix='',
        class_prob=dict({
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 1,
            9: 1,
            10: 1,
            11: 1,
            12: 1,
            13: 1,
            14: 1,
            15: 1,
            16: 1,
            17: 1,
            18: 1,
            19: 1,
            20: 1,
            21: 1,
            22: 1,
            23: 1,
            24: 1,
            25: 1,
            26: 1,
            27: 1,
            28: 1,
            29: 1,
            30: 1,
            31: 1,
            32: 1,
            33: 1,
            34: 1,
            35: 1,
            36: 1,
            37: 1,
            38: 1,
            39: 1,
            40: 1,
            41: 1,
            42: 1,
            43: 1,
            44: 1,
            45: 1,
            46: 1,
            47: 1,
            48: 1,
            49: 1,
            50: 1,
            51: 1,
            52: 1,
            53: 1,
            54: 1,
            55: 1,
            56: 1,
            57: 1,
            58: 1,
            59: 1,
            60: 2,
            61: 2,
            62: 2,
            63: 2,
            64: 2,
            65: 2,
            66: 2,
            67: 2,
            68: 2,
            69: 2,
            70: 2,
            71: 2,
            72: 2,
            73: 2,
            74: 2,
            75: 2,
            76: 2,
            77: 2,
            78: 2,
            79: 2,
            80: 2,
            81: 2,
            82: 2,
            83: 2,
            84: 2,
            85: 2,
            86: 2,
            87: 2,
            88: 2,
            89: 2,
            90: 2,
            91: 2,
            92: 2,
            93: 2,
            94: 2,
            95: 2,
            96: 2,
            97: 2,
            98: 2,
            99: 2,
            100: 2,
            101: 2,
            102: 2,
            103: 2,
            104: 2,
            105: 2,
            106: 2,
            107: 2,
            108: 2,
            109: 2,
            110: 2,
            111: 2,
            112: 2,
            113: 2,
            114: 2,
            115: 2,
            116: 2,
            117: 2,
            118: 2,
            119: 2
        }),
        pipeline=[
            dict(type='UniformSampleFrames', clip_len=48),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(-1, 64)),
            dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
            dict(type='Resize', scale=(56, 56), keep_ratio=False),
            dict(
                type='Flip',
                flip_ratio=0.5,
                left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
                right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
            dict(
                type='GeneratePoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=True,
                with_limb=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='PoseDataset',
        ann_file=
        '/home/will/Desktop/Thesis/Dataset/annotations/annotation_val.pkl',
        data_prefix='',
        pipeline=[
            dict(
                type='UniformSampleFrames',
                clip_len=48,
                num_clips=1,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(-1, 64)),
            dict(type='CenterCrop', crop_size=64),
            dict(
                type='GeneratePoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=True,
                with_limb=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='PoseDataset',
        ann_file=
        '/home/will/Desktop/Thesis/Dataset/annotations/annotation_val.pkl',
        data_prefix='',
        pipeline=[
            dict(
                type='UniformSampleFrames',
                clip_len=48,
                num_clips=1,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(-1, 64)),
            dict(type='CenterCrop', crop_size=64),
            dict(
                type='GeneratePoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=True,
                with_limb=False,
                double=False,
                left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
                right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0003)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 240
checkpoint_config = dict(interval=10)
workflow = [('train', 10)]
evaluation = dict(
    interval=10,
    metrics=['top_k_accuracy', 'mean_class_accuracy'],
    topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/slowonly_r50_u48_240e_ntu120_xsub_keypoint'
load_from = None
resume_from = None
find_unused_parameters = False
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
