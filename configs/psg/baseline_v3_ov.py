file_dir = './data/psg/processed/'
data_dir = './data/coco/'
load_from = None
resume_from = None
pretrained_transformer = './work_dirs/checkpoints/bert-base-uncased'
work_dir = './work_dirs/ov_psg_baseline'

custom_imports = dict(imports=[
    'kings_sgg.datasets.coco_panoptic_relation',
    'kings_sgg.datasets.pipelines.loading',
    'kings_sgg.models.detectors.openseed_relation',
    'kings_sgg.models.relation_heads.mask2former_relation_head',
    'kings_sgg.models.relation_heads.relation_transformer_head_v3',
], allow_failed_imports=False)

thing_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
                 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
stuff_classes = ['banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door',
                 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror', 'net',
                 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof',
                 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick',
                 'wall-stone', 'wall-tile', 'wall-wood', 'water', 'window-blind', 'window',
                 'tree', 'fence', 'ceiling', 'sky', 'cabinet', 'table', 'floor', 'pavement',
                 'mountain', 'grass', 'dirt', 'paper', 'food', 'building', 'rock', 'wall', 'rug']
num_things_classes = 80
num_stuff_classes = 53
num_object_classes = num_things_classes + num_stuff_classes
relation_classes = ['over', 'in front of', 'beside', 'on', 'in', 'attached to',
                    'hanging from', 'on back of', 'falling off', 'going down', 'painted on',
                    'walking on', 'running on', 'crossing', 'standing on', 'lying on',
                    'sitting on', 'flying over', 'jumping over', 'jumping from', 'wearing',
                    'holding', 'carrying', 'looking at', 'guiding', 'kissing', 'eating',
                    'drinking', 'feeding', 'biting', 'catching', 'picking', 'playing with',
                    'chasing', 'climbing', 'cleaning', 'playing', 'touching', 'pushing',
                    'pulling', 'opening', 'cooking', 'talking to', 'throwing', 'slicing',
                    'driving', 'riding', 'parked on', 'driving on', 'about to hit',
                    'kicking', 'swinging', 'entering', 'exiting', 'enclosing', 'leaning on']
num_relation_classes = 56

#########
# model #
#########
model = dict(
    type='OpenSeeDRelation',
    openseed_config_path='./3rdparty/OpenSeeD/configs/openseed/openseed_swint_lang.yaml',
    openseed_pretrained_path='./work_dirs/checkpoints/openseed/model_state_dict_swint_51.2ap.pt',
    thing_classes=thing_classes,
    stuff_classes=stuff_classes,
    relation_head=dict(
        type='RelationTransformerHeadV3',
        llm_path='./work_dirs/checkpoints/llama2/llama-2-7b-chat',
        tokenizer_path='./work_dirs/checkpoints/llama2/tokenizer.model',
        shave_language_decoder_at=6,
        causal_mask=False,
        ov_relation=True,
        relation_classes=relation_classes,
        sub_obj_merge_type='concat',
        num_object_in_layers=0,
        num_object_out_layers=0,
        num_relation_out_layers=0,
    ),
    text_info_db_dir='./data/psg/openai/gpt-3.5-turbo',
    text_embed_db_dir='./data/psg/openai/gpt-3.5-turbo_text-embedding-ada-002',
    text_embedding_size=1536,  # openai embedding service
    train_cfg=dict(
        freeze_layers=['openseed', 'relation_head.llama_model'],
    ),
    test_cfg=None,
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
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999))
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
