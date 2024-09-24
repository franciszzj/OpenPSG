file_dir = './data/psg/processed/'
data_dir = './data/coco/'
load_from = './work_dirs/checkpoints/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth'
resume_from = None
pretrained_transformer = './work_dirs/checkpoints/bert-base-uncased'
work_dir = './work_dirs/kings_sgg_baseline_v2'

custom_imports = dict(imports=[
    'kings_sgg.datasets.coco_panoptic_relation',
    'kings_sgg.datasets.pipelines.loading',
    'kings_sgg.models.detectors.mask2former_relation_v2',
    'kings_sgg.models.relation_heads.mask2former_relation_head',
    'kings_sgg.models.relation_heads.relation_transformer_head_v2',
    'kings_sgg.models.seg_heads.maskformer_fusion_relation_head',
], allow_failed_imports=False)

num_things_classes = 80
num_stuff_classes = 53
num_object_classes = num_things_classes + num_stuff_classes
num_relation_classes = 56

#########
# model #
#########
model = dict(
    type='Mask2FormerRelationV2',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
    ),
    panoptic_head=dict(
        type='Mask2FormerRelationHead',
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_object_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            use_pan_seg_losses=False)
    ),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionRelationHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None
    ),
    relation_head=dict(
        type='RelationTransformerHeadV2',
        pretrained_transformer=pretrained_transformer,
        load_pretrained_weights=True,
        use_adapter=False,
        input_feature_size=256,
        output_feature_size=768,
        num_transformer_layer=2,
        num_object_classes=num_object_classes,
        num_relation_classes=num_relation_classes,
        text_embedding_size=1536,  # openai embedding service
        max_object_num=30,
        use_object_vision_only=True,
        use_pair_vision_only=False,
        use_pair_text_vision_cross=False,
        use_pair_vision_text_cross=False,
        use_triplet_vision_text_cross=False,
        use_moe=False,
        moe_weight_type='v1',
        embedding_add_cls=True,
        merge_cls_type='add',
        positional_encoding=None,
        use_background_feature=False,
        loss_type='v1',
        loss_weight=50,
        loss_alpha=1,
    ),
    text_info_db_dir='./data/psg/openai/gpt-3.5-turbo',
    text_embed_db_dir='./data/psg/openai/gpt-3.5-turbo_text-embedding-ada-002',
    text_embedding_size=1536,  # openai embedding service
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler'),
        freeze_layers=['backbone', 'panoptic_head'],
    ),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
    init_cfg=None)

###########
# dataset #
###########
image_size = (1024//2, 1024//2)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='LoadPanopticRelationAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True,
        with_rel=True,
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=[(1500, 400), (1500, 1400)],
        multiscale_mode='range',
        keep_ratio=True),
    # RandomCrop is not suitable for relation.
    # dict(
    #     type='RandomCrop',
    #     crop_size=image_size,
    #     crop_type='absolute',
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
                   'gt_rels', 'masks_info')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = 'CocoPanopticRelationDataset'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=f'{file_dir}/psg_tra.json',
        img_prefix=data_dir,
        seg_prefix=data_dir,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=f'{file_dir}/psg_val.json',
        img_prefix=data_dir,
        seg_prefix=data_dir,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=f'{file_dir}/psg_val.json',
        img_prefix=data_dir,
        seg_prefix=data_dir,
        pipeline=test_pipeline))

#########################
# optimizer, lr, runner #
#########################
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6, 10])

runner = dict(type='EpochBasedRunner', max_epochs=12)

log_level = 'INFO'
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 1)]
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
evaluation = dict(metric=['PQ'], classwise=True)
find_unused_parameters = True

custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
