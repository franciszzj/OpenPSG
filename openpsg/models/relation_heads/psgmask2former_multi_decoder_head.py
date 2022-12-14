# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import os
import sys
import cv2
import copy
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.cnn import Conv2d, Linear, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import force_fp32, ModuleList

from mmdet.core import (build_assigner, build_sampler, multi_apply, reduce_mean,
                        bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from openpsg.models.relation_heads.psgmaskformer_head import PSGMaskFormerHead
from openpsg.models.relation_heads.psgtr_head import MLP


@HEADS.register_module()
class PSGMask2FormerMultiDecoderHead(PSGMaskFormerHead):

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_classes=133,
                 num_relations=56,
                 object_classes=None,
                 predicate_classes=None,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 sync_cls_avg_factor=False,
                 bg_cls_weight=0.02,
                 use_mask=True,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 positional_encoding=None,
                 transformer_decoder=None,
                 decoder_cfg=dict(use_query_pred=True,
                                  ignore_masked_attention_layers=[]),
                 use_shared_query=False,
                 test_forward_output_type='high2low',
                 use_consistency_loss=False,
                 consistency_loss_weight=10.0,
                 use_decoder_parameter_mapping=False,
                 decoder_parameter_mapping_num_layers=3,
                 optim_global_decoder=False,
                 aux_loss_list=[0, 1, 2, 3, 4, 5, 6, 7],
                 sub_loss_cls=dict(type='CrossEntropyLoss',
                                   use_sigmoid=False,
                                   loss_weight=1.0,
                                   class_weight=1.0),
                 sub_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 sub_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 sub_loss_focal=dict(type='BCEFocalLoss', loss_weight=1.0),
                 sub_loss_dice=dict(type='psgtrDiceLoss', loss_weight=1.0),
                 obj_loss_cls=dict(type='CrossEntropyLoss',
                                   use_sigmoid=False,
                                   loss_weight=1.0,
                                   class_weight=1.0),
                 obj_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 obj_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 obj_loss_focal=dict(type='BCEFocalLoss', loss_weight=1.0),
                 obj_loss_dice=dict(type='psgtrDiceLoss', loss_weight=1.0),
                 rel_loss_cls=dict(type='CrossEntropyLoss',
                                   use_sigmoid=False,
                                   loss_weight=2.0,
                                   class_weight=1.0),
                 # TODO: mask sure which assigner to use, we need use mask-based assigner.
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        # 1. Config
        self.num_classes = num_classes  # 133 for COCO
        self.num_relations = num_relations  # 56 for PSG Dataset
        self.object_classes = object_classes
        self.predicate_classes = predicate_classes
        self.num_queries = num_queries  # 100
        self.num_transformer_feat_level = num_transformer_feat_level
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.bg_cls_weight = bg_cls_weight
        self.use_mask = use_mask
        self.decoder_cfg = decoder_cfg
        self.use_shared_query = use_shared_query
        self.test_forward_output_type = test_forward_output_type
        self.use_consistency_loss = use_consistency_loss
        self.consistency_loss_weight = consistency_loss_weight
        self.use_decoder_parameter_mapping = use_decoder_parameter_mapping
        self.decoder_parameter_mapping_num_layers = decoder_parameter_mapping_num_layers
        self.optim_global_decoder = optim_global_decoder
        self.aux_loss_list = aux_loss_list
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 2. Head, Pixel and Transformer Decoder
        assert pixel_decoder.encoder.transformerlayers.\
            attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]

        transformer_decoder_ = copy.deepcopy(transformer_decoder)
        self.num_heads = transformer_decoder_.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder_.num_layers
        if self.optim_global_decoder or self.use_decoder_parameter_mapping:
            self.global_transformer_decoder = build_transformer_layer_sequence(
                transformer_decoder_)
        self.high2low_transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder_)
        self.low2high_transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder_)
        self.decoder_embed_dims = self.high2low_transformer_decoder.embed_dims

        if self.use_decoder_parameter_mapping:
            self.build_parameter_mapping()

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_pe = build_positional_encoding(positional_encoding)
        if self.use_shared_query:
            self.query_embed = nn.Embedding(
                self.num_queries, out_channels)
            self.query_feat = nn.Embedding(
                self.num_queries, feat_channels)
        else:
            if self.optim_global_decoder or self.use_decoder_parameter_mapping:
                self.global_query_embed = nn.Embedding(
                    self.num_queries, out_channels)
                self.global_query_feat = nn.Embedding(
                    self.num_queries, feat_channels)
            self.high2low_query_embed = nn.Embedding(
                self.num_queries, out_channels)
            self.high2low_query_feat = nn.Embedding(
                self.num_queries, feat_channels)
            self.low2high_query_embed = nn.Embedding(
                self.num_queries, out_channels)
            self.low2high_query_feat = nn.Embedding(
                self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        # 3. Pred
        self.sub_cls_out_channels = self.num_classes if sub_loss_cls['use_sigmoid'] \
            else self.num_classes + 1
        self.obj_cls_out_channels = self.num_classes if obj_loss_cls['use_sigmoid'] \
            else self.num_classes + 1
        # self.rel_cls_out_channels = self.num_relations if rel_loss_cls['use_sigmoid'] \
        #     else self.num_relations + 1
        # if rel_loss_cls.use_sigmoid is True, the out_channels is still num_relations + 1
        # because the begin index for relation is 1, not 0. so we can set the out_channels
        # index 0 to be ignored when use_sigmoid=True, and index 0 to be no_relation when
        # use_sigmoid=False.
        self.rel_cls_out_channels = self.num_relations + 1

        self.sub_cls_embed = Linear(
            self.decoder_embed_dims, self.sub_cls_out_channels)
        self.sub_box_embed = MLP(
            self.decoder_embed_dims, self.decoder_embed_dims, 4, 3)
        self.obj_cls_embed = Linear(
            self.decoder_embed_dims, self.obj_cls_out_channels)
        self.obj_box_embed = MLP(
            self.decoder_embed_dims, self.decoder_embed_dims, 4, 3)
        if self.optim_global_decoder:
            self.global_rel_cls_embed = Linear(
                self.decoder_embed_dims, self.rel_cls_out_channels)
        self.high2low_rel_cls_embed = Linear(
            self.decoder_embed_dims, self.rel_cls_out_channels)
        self.low2high_rel_cls_embed = Linear(
            self.decoder_embed_dims, self.rel_cls_out_channels)
        self.sub_mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))
        self.obj_mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        # 4. Assigner and Sampler
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert sub_loss_cls['loss_weight'] == assigner['s_cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert obj_loss_cls['loss_weight'] == assigner['o_cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert rel_loss_cls['loss_weight'] == assigner['r_cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert sub_loss_bbox['loss_weight'] == assigner['s_reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert obj_loss_bbox['loss_weight'] == assigner['o_reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert sub_loss_iou['loss_weight'] == assigner['s_iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            assert obj_loss_iou['loss_weight'] == assigner['o_iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            if train_cfg.assigner.type == 'HTriMatcher':
                if 's_focal_cost' in assigner.keys():
                    del assigner['s_focal_cost']
                if 's_dice_cost' in assigner.keys():
                    del assigner['s_dice_cost']
                if 'o_focal_cost' in assigner.keys():
                    del assigner['o_focal_cost']
                if 'o_dice_cost' in assigner.keys():
                    del assigner['o_dice_cost']
            if train_cfg.assigner.type == 'MaskHTriMatcher':
                assert sub_loss_focal['loss_weight'] == assigner['s_focal_cost']['weight'], \
                    'The mask focal loss weight for loss and matcher should be exactly the same.'
                assert sub_loss_dice['loss_weight'] == assigner['s_dice_cost']['weight'], \
                    'The mask dice loss weight for loss and matcher should be exactly the same.'
                assert obj_loss_focal['loss_weight'] == assigner['o_focal_cost']['weight'], \
                    'The mask focal loss weight for loss and matcher should be exactly the same.'
                assert obj_loss_dice['loss_weight'] == assigner['o_dice_cost']['weight'], \
                    'The mask dice loss weight for loss and matcher should be exactly the same.'
            self.assigner = build_assigner(assigner)
            # following DETR sampling=False, so use PseudoSampler
            sampler_cfg = train_cfg.get('sampler', dict(type='PseudoSampler'))
            self.sampler = build_sampler(sampler_cfg, context=self)
            # NOTE: Not use
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        # 5. Loss
        # NOTE following the official DETR rep0, bg_cls_weight means
        # relative classification weight of the no-object class.
        if not sub_loss_cls.use_sigmoid:
            s_class_weight = sub_loss_cls.get('class_weight', None)
            s_class_weight = torch.ones(num_classes + 1) * s_class_weight
            # NOTE set background class as the last indice
            s_class_weight[-1] = bg_cls_weight
            sub_loss_cls.update({'class_weight': s_class_weight})
        if not obj_loss_cls.use_sigmoid:
            o_class_weight = obj_loss_cls.get('class_weight', None)
            o_class_weight = torch.ones(num_classes + 1) * o_class_weight
            # NOTE set background class as the last indice
            o_class_weight[-1] = bg_cls_weight
            obj_loss_cls.update({'class_weight': o_class_weight})
        if not rel_loss_cls.use_sigmoid:
            rel_bg_cls_weight = bg_cls_weight
        else:
            # When use_sigmoid=True, then ignore the index 0
            rel_bg_cls_weight = 0.
        r_class_weight = rel_loss_cls.get('class_weight', None)
        r_class_weight = torch.ones(num_relations + 1) * r_class_weight
        # NOTE set background class as the first indice for relations as they are 1-based
        r_class_weight[0] = rel_bg_cls_weight
        rel_loss_cls.update({'class_weight': r_class_weight})
        if 'bg_cls_weight' in rel_loss_cls:
            rel_loss_cls.pop('bg_cls_weight')

        self.sub_loss_cls = build_loss(sub_loss_cls)  # cls
        self.sub_loss_bbox = build_loss(sub_loss_bbox)  # bbox
        self.sub_loss_iou = build_loss(sub_loss_iou)  # bbox
        self.sub_loss_focal = build_loss(sub_loss_focal)  # mask
        self.sub_loss_dice = build_loss(sub_loss_dice)  # mask
        self.obj_loss_cls = build_loss(obj_loss_cls)  # cls
        self.obj_loss_bbox = build_loss(obj_loss_bbox)  # bbox
        self.obj_loss_iou = build_loss(obj_loss_iou)  # bbox
        self.obj_loss_focal = build_loss(obj_loss_focal)  # mask
        self.obj_loss_dice = build_loss(obj_loss_dice)  # mask
        self.rel_loss_cls = build_loss(rel_loss_cls)  # rel

        if self.use_consistency_loss:
            self.consistency_cls_loss = nn.KLDivLoss(
                reduction='batchmean', log_target=True)
            self.consistency_reg_loss = nn.SmoothL1Loss()

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        if self.use_decoder_parameter_mapping or self.optim_global_decoder:
            for p in self.global_transformer_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.use_decoder_parameter_mapping:
            for p in self.pm_dict.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for p in self.high2low_transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.low2high_transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def build_parameter_mapping(self):
        self.pm_dict = nn.ModuleDict()
        for n, p in self.high2low_transformer_decoder.named_parameters():
            p.requires_grad = False
            p_size = p.shape[0]
            if len(p.shape) == 2:
                pm = MLConv1d(p_size, p_size, p_size,
                              self.decoder_parameter_mapping_num_layers)
            elif len(p.shape) == 1:
                pm = MLP(p_size, p_size, p_size,
                         self.decoder_parameter_mapping_num_layers)
            self.pm_dict.update({n.replace('.', '_'): pm})
        for n, p in self.low2high_transformer_decoder.named_parameters():
            p.requires_grad = False
            p_size = p.shape[0]
            if len(p.shape) == 2:
                pm = MLConv1d(p_size, p_size, p_size,
                              self.decoder_parameter_mapping_num_layers)
            elif len(p.shape) == 1:
                pm = MLP(p_size, p_size, p_size,
                         self.decoder_parameter_mapping_num_layers)
            self.pm_dict.update({n.replace('.', '_'): pm})

    @force_fp32(apply_to=('high2low_all_cls_scores', 'high2low_all_bbox_preds', 'high2low_all_mask_preds',
                          'low2high_all_cls_scores', 'low2high_all_bbox_preds', 'low2high_all_mask_preds'))
    def loss(self,
             high2low_all_cls_scores,
             high2low_all_bbox_preds,
             high2low_all_mask_preds,
             low2high_all_cls_scores,
             low2high_all_bbox_preds,
             low2high_all_mask_preds,
             gt_labels_list,
             gt_bboxes_list,
             gt_masks_list,
             gt_rels_list,
             high2low_gt_rels_list,
             low2high_gt_rels_list,
             img_metas,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        high2low_all_s_cls_scores = high2low_all_cls_scores['sub']
        high2low_all_o_cls_scores = high2low_all_cls_scores['obj']
        high2low_all_r_cls_scores = high2low_all_cls_scores['rel']
        high2low_all_s_bbox_preds = high2low_all_bbox_preds['sub']
        high2low_all_o_bbox_preds = high2low_all_bbox_preds['obj']
        high2low_all_s_mask_preds = high2low_all_mask_preds['sub']
        high2low_all_o_mask_preds = high2low_all_mask_preds['obj']

        low2high_all_s_cls_scores = low2high_all_cls_scores['sub']
        low2high_all_o_cls_scores = low2high_all_cls_scores['obj']
        low2high_all_r_cls_scores = low2high_all_cls_scores['rel']
        low2high_all_s_bbox_preds = low2high_all_bbox_preds['sub']
        low2high_all_o_bbox_preds = low2high_all_bbox_preds['obj']
        low2high_all_s_mask_preds = low2high_all_mask_preds['sub']
        low2high_all_o_mask_preds = low2high_all_mask_preds['obj']

        num_dec_layers = len(high2low_all_s_cls_scores)

        # all_gt_rels_list = [
        #     gt_rels_list for _ in range(num_dec_layers)]

        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        high2low_all_gt_rels_list = [
            high2low_gt_rels_list for _ in range(num_dec_layers)]
        low2high_all_gt_rels_list = [
            low2high_gt_rels_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        loss_dict = dict()

        ############
        # high2low #
        ############
        high2low_s_losses_cls, high2low_o_losses_cls, high2low_r_losses_cls, \
            high2low_s_losses_bbox, high2low_o_losses_bbox, high2low_s_losses_iou, high2low_o_losses_iou, \
            high2low_s_losses_focal, high2low_o_losses_focal, high2low_s_losses_dice, high2low_o_losses_dice = \
            multi_apply(self.loss_single,
                        high2low_all_s_cls_scores, high2low_all_o_cls_scores, high2low_all_r_cls_scores,
                        high2low_all_s_bbox_preds, high2low_all_o_bbox_preds,
                        high2low_all_s_mask_preds, high2low_all_o_mask_preds,
                        # use high2low_all_gt_rels_list
                        high2low_all_gt_rels_list, all_gt_bboxes_list, all_gt_labels_list,
                        all_gt_masks_list, img_metas_list, all_gt_bboxes_ignore_list)

        # loss from the last decoder layer
        loss_dict['high2low_s_loss_cls'] = high2low_s_losses_cls[-1]
        loss_dict['high2low_o_loss_cls'] = high2low_o_losses_cls[-1]
        loss_dict['high2low_r_loss_cls'] = high2low_r_losses_cls[-1]
        loss_dict['high2low_s_loss_bbox'] = high2low_s_losses_bbox[-1]
        loss_dict['high2low_o_loss_bbox'] = high2low_o_losses_bbox[-1]
        loss_dict['high2low_s_loss_iou'] = high2low_s_losses_iou[-1]
        loss_dict['high2low_o_loss_iou'] = high2low_o_losses_iou[-1]
        if self.use_mask:
            loss_dict['high2low_s_loss_focal'] = high2low_s_losses_focal[-1]
            loss_dict['high2low_s_loss_dice'] = high2low_s_losses_dice[-1]
            loss_dict['high2low_o_loss_focal'] = high2low_o_losses_focal[-1]
            loss_dict['high2low_o_loss_dice'] = high2low_o_losses_dice[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for high2low_s_loss_cls_i, high2low_o_loss_cls_i, high2low_r_loss_cls_i, \
            high2low_s_loss_bbox_i, high2low_o_loss_bbox_i, \
            high2low_s_loss_iou_i, high2low_o_loss_iou_i, \
            high2low_s_loss_focal_i, high2low_o_loss_focal_i, \
            high2low_s_loss_dice_i, high2low_o_loss_dice_i in zip(high2low_s_losses_cls[:-1], high2low_o_losses_cls[:-1], high2low_r_losses_cls[:-1],
                                                                  high2low_s_losses_bbox[:-1], high2low_o_losses_bbox[:-1],  # noqa
                                                                  high2low_s_losses_iou[:- 1], high2low_o_losses_iou[:-1],  # noqa
                                                                  high2low_s_losses_focal[:- 1], high2low_s_losses_dice[:-1],  # noqa
                                                                  high2low_o_losses_focal[:-1], high2low_o_losses_dice[:-1]):  # noqa
            loss_dict[f'd{num_dec_layer}.high2low_s_loss_cls'] = high2low_s_loss_cls_i
            loss_dict[f'd{num_dec_layer}.high2low_o_loss_cls'] = high2low_o_loss_cls_i
            loss_dict[f'd{num_dec_layer}.high2low_r_loss_cls'] = high2low_r_loss_cls_i
            loss_dict[f'd{num_dec_layer}.high2low_s_loss_bbox'] = high2low_s_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.high2low_o_loss_bbox'] = high2low_o_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.high2low_s_loss_iou'] = high2low_s_loss_iou_i
            loss_dict[f'd{num_dec_layer}.high2low_o_loss_iou'] = high2low_o_loss_iou_i
            loss_dict[f'd{num_dec_layer}.high2low_s_loss_focal'] = high2low_s_loss_focal_i
            loss_dict[f'd{num_dec_layer}.high2low_o_loss_focal'] = high2low_o_loss_focal_i
            loss_dict[f'd{num_dec_layer}.high2low_s_loss_dice'] = high2low_s_loss_dice_i
            loss_dict[f'd{num_dec_layer}.high2low_o_loss_dice'] = high2low_o_loss_dice_i
            num_dec_layer += 1

        ############
        # low2high #
        ############
        low2high_s_losses_cls, low2high_o_losses_cls, low2high_r_losses_cls, \
            low2high_s_losses_bbox, low2high_o_losses_bbox, low2high_s_losses_iou, low2high_o_losses_iou, \
            low2high_s_losses_focal, low2high_o_losses_focal, low2high_s_losses_dice, low2high_o_losses_dice = \
            multi_apply(self.loss_single,
                        low2high_all_s_cls_scores, low2high_all_o_cls_scores, low2high_all_r_cls_scores,
                        low2high_all_s_bbox_preds, low2high_all_o_bbox_preds,
                        low2high_all_s_mask_preds, low2high_all_o_mask_preds,
                        # use low2high_all_gt_rels_list
                        low2high_all_gt_rels_list, all_gt_bboxes_list, all_gt_labels_list,
                        all_gt_masks_list, img_metas_list, all_gt_bboxes_ignore_list)

        # loss from the last decoder layer
        loss_dict['low2high_s_loss_cls'] = low2high_s_losses_cls[-1]
        loss_dict['low2high_o_loss_cls'] = low2high_o_losses_cls[-1]
        loss_dict['low2high_r_loss_cls'] = low2high_r_losses_cls[-1]
        loss_dict['low2high_s_loss_bbox'] = low2high_s_losses_bbox[-1]
        loss_dict['low2high_o_loss_bbox'] = low2high_o_losses_bbox[-1]
        loss_dict['low2high_s_loss_iou'] = low2high_s_losses_iou[-1]
        loss_dict['low2high_o_loss_iou'] = low2high_o_losses_iou[-1]
        if self.use_mask:
            loss_dict['low2high_s_loss_focal'] = low2high_s_losses_focal[-1]
            loss_dict['low2high_s_loss_dice'] = low2high_s_losses_dice[-1]
            loss_dict['low2high_o_loss_focal'] = low2high_o_losses_focal[-1]
            loss_dict['low2high_o_loss_dice'] = low2high_o_losses_dice[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for low2high_s_loss_cls_i, low2high_o_loss_cls_i, low2high_r_loss_cls_i, \
            low2high_s_loss_bbox_i, low2high_o_loss_bbox_i, \
            low2high_s_loss_iou_i, low2high_o_loss_iou_i, \
            low2high_s_loss_focal_i, low2high_o_loss_focal_i, \
            low2high_s_loss_dice_i, low2high_o_loss_dice_i in zip(low2high_s_losses_cls[:-1], low2high_o_losses_cls[:-1], low2high_r_losses_cls[:-1],
                                                                  low2high_s_losses_bbox[:-1], low2high_o_losses_bbox[:-1],  # noqa
                                                                  low2high_s_losses_iou[:- 1], low2high_o_losses_iou[:-1],  # noqa
                                                                  low2high_s_losses_focal[:- 1], low2high_s_losses_dice[:-1],  # noqa
                                                                  low2high_o_losses_focal[:-1], low2high_o_losses_dice[:-1]):  # noqa
            loss_dict[f'd{num_dec_layer}.low2high_s_loss_cls'] = low2high_s_loss_cls_i
            loss_dict[f'd{num_dec_layer}.low2high_o_loss_cls'] = low2high_o_loss_cls_i
            loss_dict[f'd{num_dec_layer}.low2high_r_loss_cls'] = low2high_r_loss_cls_i
            loss_dict[f'd{num_dec_layer}.low2high_s_loss_bbox'] = low2high_s_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.low2high_o_loss_bbox'] = low2high_o_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.low2high_s_loss_iou'] = low2high_s_loss_iou_i
            loss_dict[f'd{num_dec_layer}.low2high_o_loss_iou'] = low2high_o_loss_iou_i
            loss_dict[f'd{num_dec_layer}.low2high_s_loss_focal'] = low2high_s_loss_focal_i
            loss_dict[f'd{num_dec_layer}.low2high_o_loss_focal'] = low2high_o_loss_focal_i
            loss_dict[f'd{num_dec_layer}.low2high_s_loss_dice'] = low2high_s_loss_dice_i
            loss_dict[f'd{num_dec_layer}.low2high_o_loss_dice'] = low2high_o_loss_dice_i
            num_dec_layer += 1

        if self.use_consistency_loss:
            high2low_sub_input = F.log_softmax(high2low_all_s_cls_scores[-1].flatten(start_dim=0, end_dim=1), dim=1)  # noqa
            low2high_sub_input = F.log_softmax(low2high_all_s_cls_scores[-1].flatten(start_dim=0, end_dim=1), dim=1)  # noqa
            high2low_obj_input = F.log_softmax(high2low_all_o_cls_scores[-1].flatten(start_dim=0, end_dim=1), dim=1)  # noqa
            low2high_obj_input = F.log_softmax(low2high_all_o_cls_scores[-1].flatten(start_dim=0, end_dim=1), dim=1)  # noqa
            sub_consistency_cls1_loss = self.consistency_cls_loss(high2low_sub_input, low2high_sub_input) * self.consistency_loss_weight  # noqa
            sub_consistency_cls2_loss = self.consistency_cls_loss(low2high_sub_input, high2low_sub_input) * self.consistency_loss_weight  # noqa
            obj_consistency_cls1_loss = self.consistency_cls_loss(high2low_obj_input, low2high_obj_input) * self.consistency_loss_weight  # noqa
            obj_consistency_cls2_loss = self.consistency_cls_loss(low2high_obj_input, high2low_obj_input) * self.consistency_loss_weight  # noqa
            # bbox has already done sigmoid(), but mask not.
            sub_consistency_bbox_loss = self.consistency_reg_loss(high2low_all_s_bbox_preds[-1], low2high_all_s_bbox_preds[-1]) * self.consistency_loss_weight  # noqa
            obj_consistency_bbox_loss = self.consistency_reg_loss(high2low_all_o_bbox_preds[-1], low2high_all_o_bbox_preds[-1]) * self.consistency_loss_weight  # noqa
            sub_consistency_mask_loss = self.consistency_reg_loss(high2low_all_s_mask_preds[-1].sigmoid(), low2high_all_s_mask_preds[-1].sigmoid()) * self.consistency_loss_weight  # noqa
            obj_consistency_mask_loss = self.consistency_reg_loss(high2low_all_o_mask_preds[-1].sigmoid(), low2high_all_o_mask_preds[-1].sigmoid()) * self.consistency_loss_weight  # noqa
            loss_dict['sub_consistency_cls1_loss'] = sub_consistency_cls1_loss
            loss_dict['sub_consistency_cls2_loss'] = sub_consistency_cls2_loss
            loss_dict['obj_consistency_cls1_loss'] = obj_consistency_cls1_loss
            loss_dict['obj_consistency_cls2_loss'] = obj_consistency_cls2_loss
            loss_dict['sub_consistency_bbox_loss'] = sub_consistency_bbox_loss
            loss_dict['obj_consistency_bbox_loss'] = obj_consistency_bbox_loss
            loss_dict['sub_consistency_mask_loss'] = sub_consistency_mask_loss
            loss_dict['obj_consistency_mask_loss'] = obj_consistency_mask_loss

        return loss_dict

    def _merge_gt_with_gt(self, gt_rels_list1, gt_rels_list2):
        gt_rels_list = []
        for gt_rels1, gt_rels2 in zip(gt_rels_list1, gt_rels_list2):
            assert gt_rels1.shape[0] == gt_rels2.shape[0]
            gt_rels = []
            for idx in range(gt_rels1.shape[0]):
                gt_rel1 = gt_rels1[idx]
                gt_rel2 = gt_rels2[idx]
                assert gt_rel1[0] == gt_rel1[0]
                assert gt_rel1[1] == gt_rel1[1]
                if gt_rel1[2] == gt_rel2[2]:
                    gt_rels.append(gt_rel1)
                else:
                    gt_rels.append(gt_rel1)
                    gt_rels.append(gt_rel2)
            gt_rels = torch.stack(gt_rels, dim=0)
            gt_rels_list.append(gt_rels)
        return gt_rels_list

    def _merge_pred_with_pred(self,
                              s_cls_scores1, o_cls_scores1, r_cls_scores1,
                              s_bbox_preds1, o_bbox_preds1,
                              s_mask_preds1, o_mask_preds1,
                              s_cls_scores2, o_cls_scores2, r_cls_scores2,
                              s_bbox_preds2, o_bbox_preds2,
                              s_mask_preds2, o_mask_preds2, img_metas):

        results = []
        for img_idx in range(len(img_metas)):
            img_shape = img_metas[img_idx]['img_shape']
            scale_factor = img_metas[img_idx]['scale_factor']
            # 1. get result1
            rel_pairs1, labels1, scores1, bboxes1, masks1, r_labels1, r_scores1, r_dists1 = \
                self._get_results_single_easy(
                    s_cls_scores1[img_idx].detach(), o_cls_scores1[img_idx].detach(), r_cls_scores1[img_idx].detach(),  # noqa
                    s_bbox_preds1[img_idx].detach(), o_bbox_preds1[img_idx].detach(),  # noqa
                    s_mask_preds1[img_idx].detach(), o_mask_preds1[img_idx].detach(),  # noqa
                    img_shape, scale_factor, rescale=False)
            num1 = rel_pairs1.shape[0]
            # 2. get result2
            rel_pairs2, labels2, scores2, bboxes2, masks2, r_labels2, r_scores2, r_dists2 = \
                self._get_results_single_easy(
                    s_cls_scores2[img_idx].detach(), o_cls_scores2[img_idx].detach(), r_cls_scores2[img_idx].detach(),  # noqa
                    s_bbox_preds2[img_idx].detach(), o_bbox_preds2[img_idx].detach(),  # noqa
                    s_mask_preds2[img_idx].detach(), o_mask_preds2[img_idx].detach(),  # noqa
                    img_shape, scale_factor, rescale=False)
            num2 = rel_pairs2.shape[0]

            # 3. reshape result to (n, ...)
            labels1 = labels1.reshape((2, num1)).permute((1, 0))
            scores1 = scores1.reshape((2, num1)).permute((1, 0))
            bboxes1 = bboxes1.reshape((2, num1, 5)).permute((1, 0, 2))
            h, w = masks1.shape[-2:]
            masks1 = masks1.reshape((2, num1, h, w)).permute((1, 0, 2, 3))

            labels2 = labels2.reshape((2, num2)).permute((1, 0))
            scores2 = scores2.reshape((2, num2)).permute((1, 0))
            bboxes2 = bboxes2.reshape((2, num2, 5)).permute((1, 0, 2))
            h, w = masks2.shape[-2:]
            masks2 = masks2.reshape((2, num2, h, w)).permute((1, 0, 2, 3))

            # 4. concat result1 and result2
            labels_all = torch.cat((labels1, labels2), dim=0)
            scores_all = torch.cat((scores1, scores2), dim=0)
            bboxes_all = torch.cat((bboxes1, bboxes2), dim=0)
            masks_all = torch.cat((masks1, masks2), dim=0)
            r_labels_all = torch.cat((r_labels1, r_labels2), dim=0)
            r_scores_all = torch.cat((r_scores1, r_scores2), dim=0)
            r_dists_all = torch.cat((r_dists1, r_dists2), dim=0)

            # 5. re-arrange based on r_scores, output r_idxes
            r_idxes = torch.argsort(r_scores_all, dim=0, descending=True)

            # 6. re-arrange based on r_idxes
            labels = labels_all[r_idxes]
            scores = scores_all[r_idxes]
            bboxes = bboxes_all[r_idxes]
            masks = masks_all[r_idxes]
            r_labels = r_labels_all[r_idxes]
            r_scores = r_scores_all[r_idxes]
            r_dists = r_dists_all[r_idxes]

            # 7. dedup
            keep_tri = self._dedup_triplets_based_on_iou(
                labels[:, 0], labels[:, 1], r_labels, masks[:, 0], masks[:, 1])
            rel_pairs = torch.asarray([i for i in range(keep_tri.sum()*2)],
                                      dtype=r_labels.dtype, device=r_labels.device)
            labels = labels[keep_tri]
            scores = scores[keep_tri]
            bboxes = bboxes[keep_tri]
            masks = masks[keep_tri]
            r_labels = r_labels[keep_tri]
            r_scores = r_scores[keep_tri]
            r_dists = r_dists[keep_tri]

            # 8. reshape to (n*2, ...)
            labels = labels.permute((1, 0)).reshape((-1))
            scores = scores.permute((1, 0)).reshape((-1))
            bboxes = bboxes.permute((1, 0, 2)).reshape((-1, 5))
            masks = masks.permute((1, 0, 2, 3)).reshape((-1, h, w))

            results.append([rel_pairs, labels, scores, bboxes,
                           masks, r_labels, r_scores, r_dists])
        return results

    def _merge_gt_with_pred(self, gt_labels_list, gt_bboxes_list, gt_masks_list, gt_rels_list,
                            pred_results, img_metas):
        merge_gt_labels_list = []
        merge_gt_bboxes_list = []
        merge_gt_masks_list = []
        merge_gt_rels_list = []
        for img_idx in range(len(img_metas)):
            rel_pairs, labels, scores, det_bboxes, output_masks, r_labels, r_scores, r_dists = \
                pred_results[img_idx]

            merge_gt_labels = torch.cat(
                (gt_labels_list[img_idx], labels), dim=0)
            merge_gt_bboxes = torch.cat(
                (gt_bboxes_list[img_idx], det_bboxes[:, :4]), dim=0)
            merge_gt_masks = torch.cat(
                (gt_masks_list[img_idx], output_masks), dim=0)
            if rel_pairs.shape[0] > 0 and r_labels.shape[0] > 0:
                rel_pairs += gt_labels_list[img_idx].shape[0]
                rel_pairs = rel_pairs.reshape(2, -1).T
                rels = torch.cat((rel_pairs, r_labels.unsqueeze(1)), dim=1)
                merge_gt_rels = torch.cat((gt_rels_list[img_idx], rels), dim=0)
            else:
                merge_gt_rels = gt_rels_list[img_idx]

            merge_gt_labels_list.append(merge_gt_labels)
            merge_gt_bboxes_list.append(merge_gt_bboxes)
            merge_gt_masks_list.append(merge_gt_masks)
            merge_gt_rels_list.append(merge_gt_rels)
        return merge_gt_labels_list, merge_gt_bboxes_list, merge_gt_masks_list, merge_gt_rels_list

    @force_fp32(apply_to=('global_all_cls_scores', 'global_all_bbox_preds', 'global_all_mask_preds',
                          'high2low_all_cls_scores', 'high2low_all_bbox_preds', 'high2low_all_mask_preds',
                          'low2high_all_cls_scores', 'low2high_all_bbox_preds', 'low2high_all_mask_preds'))
    def loss_global(self,
                    global_all_cls_scores,
                    global_all_bbox_preds,
                    global_all_mask_preds,
                    high2low_all_cls_scores,
                    high2low_all_bbox_preds,
                    high2low_all_mask_preds,
                    low2high_all_cls_scores,
                    low2high_all_bbox_preds,
                    low2high_all_mask_preds,
                    gt_labels_list,
                    gt_bboxes_list,
                    gt_masks_list,
                    gt_rels_list,
                    high2low_gt_rels_list,
                    low2high_gt_rels_list,
                    img_metas,
                    gt_bboxes_ignore=None):
        global_all_s_cls_scores = global_all_cls_scores['sub']
        global_all_o_cls_scores = global_all_cls_scores['obj']
        global_all_r_cls_scores = global_all_cls_scores['rel']
        global_all_s_bbox_preds = global_all_bbox_preds['sub']
        global_all_o_bbox_preds = global_all_bbox_preds['obj']
        global_all_s_mask_preds = global_all_mask_preds['sub']
        global_all_o_mask_preds = global_all_mask_preds['obj']

        high2low_all_s_cls_scores = high2low_all_cls_scores['sub']
        high2low_all_o_cls_scores = high2low_all_cls_scores['obj']
        high2low_all_r_cls_scores = high2low_all_cls_scores['rel']
        high2low_all_s_bbox_preds = high2low_all_bbox_preds['sub']
        high2low_all_o_bbox_preds = high2low_all_bbox_preds['obj']
        high2low_all_s_mask_preds = high2low_all_mask_preds['sub']
        high2low_all_o_mask_preds = high2low_all_mask_preds['obj']

        low2high_all_s_cls_scores = low2high_all_cls_scores['sub']
        low2high_all_o_cls_scores = low2high_all_cls_scores['obj']
        low2high_all_r_cls_scores = low2high_all_cls_scores['rel']
        low2high_all_s_bbox_preds = low2high_all_bbox_preds['sub']
        low2high_all_o_bbox_preds = low2high_all_bbox_preds['obj']
        low2high_all_s_mask_preds = low2high_all_mask_preds['sub']
        low2high_all_o_mask_preds = low2high_all_mask_preds['obj']

        num_dec_layers = len(high2low_all_s_cls_scores)
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        #####################
        # global gt prepare #
        #####################
        # 1. merge high2low and low2high gt
        global_gt_rels_list = self._merge_gt_with_gt(
            high2low_gt_rels_list, low2high_gt_rels_list)
        # 2. merge high2low and low2high pred
        pred_results = self._merge_pred_with_pred(
            high2low_all_s_cls_scores[-1], high2low_all_o_cls_scores[-1], high2low_all_r_cls_scores[-1],
            high2low_all_s_bbox_preds[-1], high2low_all_o_bbox_preds[-1],
            high2low_all_s_mask_preds[-1], high2low_all_o_mask_preds[-1],
            low2high_all_s_cls_scores[-1], low2high_all_o_cls_scores[-1], low2high_all_r_cls_scores[-1],
            low2high_all_s_bbox_preds[-1], low2high_all_o_bbox_preds[-1],
            low2high_all_s_mask_preds[-1], low2high_all_o_mask_preds[-1],
            img_metas)
        # 3. merge gt and pred
        global_gt_labels_list, global_gt_bboxes_list, global_gt_masks_list, global_gt_rels_list = self._merge_gt_with_pred(
            gt_labels_list, gt_bboxes_list, gt_masks_list, global_gt_rels_list, pred_results, img_metas)

        global_all_gt_labels_list = [
            global_gt_labels_list for _ in range(num_dec_layers)]
        global_all_gt_bboxes_list = [
            global_gt_bboxes_list for _ in range(num_dec_layers)]
        global_all_gt_masks_list = [
            global_gt_masks_list for _ in range(num_dec_layers)]
        global_all_gt_rels_list = [
            global_gt_rels_list for _ in range(num_dec_layers)]

        loss_dict = dict()

        ##########
        # global #
        ##########
        global_s_losses_cls, global_o_losses_cls, global_r_losses_cls, \
            global_s_losses_bbox, global_o_losses_bbox, global_s_losses_iou, global_o_losses_iou, \
            global_s_losses_focal, global_o_losses_focal, global_s_losses_dice, global_o_losses_dice = \
            multi_apply(self.loss_single,
                        global_all_s_cls_scores, global_all_o_cls_scores, global_all_r_cls_scores,
                        global_all_s_bbox_preds, global_all_o_bbox_preds,
                        global_all_s_mask_preds, global_all_o_mask_preds,
                        global_all_gt_rels_list, global_all_gt_bboxes_list, global_all_gt_labels_list,
                        global_all_gt_masks_list, img_metas_list, all_gt_bboxes_ignore_list)

        # loss from the last decoder layer
        loss_dict['global_s_loss_cls'] = global_s_losses_cls[-1]
        loss_dict['global_o_loss_cls'] = global_o_losses_cls[-1]
        loss_dict['global_r_loss_cls'] = global_r_losses_cls[-1]
        loss_dict['global_s_loss_bbox'] = global_s_losses_bbox[-1]
        loss_dict['global_o_loss_bbox'] = global_o_losses_bbox[-1]
        loss_dict['global_s_loss_iou'] = global_s_losses_iou[-1]
        loss_dict['global_o_loss_iou'] = global_o_losses_iou[-1]
        if self.use_mask:
            loss_dict['global_s_loss_focal'] = global_s_losses_focal[-1]
            loss_dict['global_s_loss_dice'] = global_s_losses_dice[-1]
            loss_dict['global_o_loss_focal'] = global_o_losses_focal[-1]
            loss_dict['global_o_loss_dice'] = global_o_losses_dice[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for global_s_loss_cls_i, global_o_loss_cls_i, global_r_loss_cls_i, \
            global_s_loss_bbox_i, global_o_loss_bbox_i, \
            global_s_loss_iou_i, global_o_loss_iou_i, \
            global_s_loss_focal_i, global_o_loss_focal_i, \
            global_s_loss_dice_i, global_o_loss_dice_i in zip(global_s_losses_cls[:-1], global_o_losses_cls[:-1], global_r_losses_cls[:-1],
                                                                global_s_losses_bbox[:-1], global_o_losses_bbox[:-1],  # noqa
                                                                global_s_losses_iou[:- 1], global_o_losses_iou[:-1],  # noqa
                                                                global_s_losses_focal[:- 1], global_s_losses_dice[:-1],  # noqa
                                                                global_o_losses_focal[:-1], global_o_losses_dice[:-1]):  # noqa
            loss_dict[f'd{num_dec_layer}.global_s_loss_cls'] = global_s_loss_cls_i
            loss_dict[f'd{num_dec_layer}.global_o_loss_cls'] = global_o_loss_cls_i
            loss_dict[f'd{num_dec_layer}.global_r_loss_cls'] = global_r_loss_cls_i
            loss_dict[f'd{num_dec_layer}.global_s_loss_bbox'] = global_s_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.global_o_loss_bbox'] = global_o_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.global_s_loss_iou'] = global_s_loss_iou_i
            loss_dict[f'd{num_dec_layer}.global_o_loss_iou'] = global_o_loss_iou_i
            loss_dict[f'd{num_dec_layer}.global_s_loss_focal'] = global_s_loss_focal_i
            loss_dict[f'd{num_dec_layer}.global_o_loss_focal'] = global_o_loss_focal_i
            loss_dict[f'd{num_dec_layer}.global_s_loss_dice'] = global_s_loss_dice_i
            loss_dict[f'd{num_dec_layer}.global_o_loss_dice'] = global_o_loss_dice_i
            num_dec_layer += 1

        return loss_dict

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size, decoder_layer_idx=0,
                     transformer_decoder=None, rel_cls_embed=None):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.
            decoder_layer_idx (int): the index number of decoder layer.
                0 layer means the embedding output from encoder and before
                feed into the decoder.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h*w). \
                This attn_mask is used for cross attention.
        """
        debug = False

        decoder_out = transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)

        sub_output_class = self.sub_cls_embed(decoder_out)
        obj_output_class = self.obj_cls_embed(decoder_out)
        rel_output_class = rel_cls_embed(decoder_out)
        all_cls_score = dict(sub=sub_output_class,
                             obj=obj_output_class,
                             rel=rel_output_class)

        sub_output_coord = self.sub_box_embed(decoder_out).sigmoid()
        obj_output_coord = self.obj_box_embed(decoder_out).sigmoid()
        all_bbox_pred = dict(sub=sub_output_coord,
                             obj=obj_output_coord)

        sub_mask_embed = self.sub_mask_embed(decoder_out)
        sub_output_mask = torch.einsum(
            'bqc,bchw->bqhw', sub_mask_embed, mask_feature)
        obj_mask_embed = self.obj_mask_embed(decoder_out)
        obj_output_mask = torch.einsum(
            'bqc,bchw->bqhw', obj_mask_embed, mask_feature)
        all_mask_pred = dict(sub=sub_output_mask,
                             obj=obj_output_mask)

        sub_attn_mask = F.interpolate(
            sub_output_mask,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (batch_size, num_queries, h, w) ->
        #   (batch_size * num_head, num_queries, h*w)
        sub_attn_mask = sub_attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        sub_attn_mask = sub_attn_mask.sigmoid() < 0.5
        sub_attn_mask = sub_attn_mask.detach()

        if debug:
            for i in range(sub_attn_mask.shape[0]):
                for j in range(sub_attn_mask.shape[1]):
                    mask = sub_attn_mask[i, j].reshape(
                        attn_mask_target_size).float().cpu().numpy() * 255
                    cv2.imwrite('vis/decoder_layer_{}_sub_attn_mask_head_{}_query_{}.png'.format(
                        decoder_layer_idx, i, j), mask)

        obj_attn_mask = F.interpolate(
            obj_output_mask,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (batch_size, num_queries, h, w) ->
        #   (batch_size * num_head, num_queries, h*w)
        obj_attn_mask = obj_attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        obj_attn_mask = obj_attn_mask.sigmoid() < 0.5
        obj_attn_mask = obj_attn_mask.detach()

        if debug:
            for i in range(obj_attn_mask.shape[0]):
                for j in range(obj_attn_mask.shape[1]):
                    mask = obj_attn_mask[i, j].reshape(
                        attn_mask_target_size).float().cpu().numpy() * 255
                    cv2.imwrite('vis/decoder_layer_{}_obj_attn_mask_head_{}_query_{}.png'.format(
                        decoder_layer_idx, i, j), mask)

        attn_mask = torch.mul(sub_attn_mask, obj_attn_mask)

        if debug:
            for i in range(attn_mask.shape[0]):
                for j in range(attn_mask.shape[1]):
                    mask = attn_mask[i, j].reshape(
                        attn_mask_target_size).float().cpu().numpy() * 255
                    cv2.imwrite('vis/decoder_layer_{}_attn_mask_head_{}_query_{}.png'.format(
                        decoder_layer_idx, i, j), mask)

        all_attn_mask = attn_mask

        return all_cls_score, all_bbox_pred, all_mask_pred, all_attn_mask

    def forward(self, feats, img_metas, **kwargs):
        debug = False
        if debug:
            print('\n img_metas for this test image: {}'.format(img_metas))

        # 1. Forward
        batch_size = len(img_metas)

        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_pe(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)

        if self.use_shared_query:
            query_feat = self.query_feat.weight.unsqueeze(1).repeat(
                (1, batch_size, 1))
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(
                (1, batch_size, 1))
            if self.optim_global_decoder:
                global_query_feat = query_feat
                global_query_embed = query_embed
            high2low_query_feat = query_feat
            high2low_query_embed = query_embed
            low2high_query_feat = query_feat
            low2high_query_embed = query_embed
        else:
            if self.optim_global_decoder:
                global_query_feat = self.global_query_feat.weight.unsqueeze(1).repeat(
                    (1, batch_size, 1))
                global_query_embed = self.global_query_embed.weight.unsqueeze(1).repeat(
                    (1, batch_size, 1))
            high2low_query_feat = self.high2low_query_feat.weight.unsqueeze(1).repeat(
                (1, batch_size, 1))
            high2low_query_embed = self.high2low_query_embed.weight.unsqueeze(1).repeat(
                (1, batch_size, 1))
            low2high_query_feat = self.low2high_query_feat.weight.unsqueeze(1).repeat(
                (1, batch_size, 1))
            low2high_query_embed = self.low2high_query_embed.weight.unsqueeze(1).repeat(
                (1, batch_size, 1))

        if self.optim_global_decoder:
            global_cls_pred_list = []
            global_bbox_pred_list = []
            global_mask_pred_list = []
        high2low_cls_pred_list = []
        high2low_bbox_pred_list = []
        high2low_mask_pred_list = []
        low2high_cls_pred_list = []
        low2high_bbox_pred_list = []
        low2high_mask_pred_list = []

        if self.decoder_cfg.get('use_query_pred', True):
            if self.optim_global_decoder:
                cls_pred, bbox_pred, mask_pred, global_cross_attn_mask = self.forward_head(
                    global_query_feat, mask_features, multi_scale_memorys[
                        0].shape[-2:], decoder_layer_idx=0,
                    transformer_decoder=self.global_transformer_decoder, rel_cls_embed=self.global_rel_cls_embed)
                global_cls_pred_list.append(cls_pred)
                global_bbox_pred_list.append(bbox_pred)
                global_mask_pred_list.append(mask_pred)
            cls_pred, bbox_pred, mask_pred, high2low_cross_attn_mask = self.forward_head(
                high2low_query_feat, mask_features, multi_scale_memorys[
                    0].shape[-2:], decoder_layer_idx=0,
                transformer_decoder=self.high2low_transformer_decoder, rel_cls_embed=self.high2low_rel_cls_embed)
            high2low_cls_pred_list.append(cls_pred)
            high2low_bbox_pred_list.append(bbox_pred)
            high2low_mask_pred_list.append(mask_pred)
            cls_pred, bbox_pred, mask_pred, low2high_cross_attn_mask = self.forward_head(
                low2high_query_feat, mask_features, multi_scale_memorys[
                    0].shape[-2:], decoder_layer_idx=0,
                transformer_decoder=self.low2high_transformer_decoder, rel_cls_embed=self.low2high_rel_cls_embed)
            low2high_cls_pred_list.append(cls_pred)
            low2high_bbox_pred_list.append(bbox_pred)
            low2high_mask_pred_list.append(mask_pred)
        else:
            if self.optim_global_decoder:
                global_cross_attn_mask = None
            high2low_cross_attn_mask = None
            low2high_cross_attn_mask = None

        ignore_masked_attention_layers = self.decoder_cfg.get(
            'ignore_masked_attention_layers', [])

        if self.use_decoder_parameter_mapping:
            self.update_perspective_decoder()

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            if self.optim_global_decoder and (global_cross_attn_mask is not None):
                global_cross_attn_mask[torch.where(
                    global_cross_attn_mask.sum(-1) == global_cross_attn_mask.shape[-1])] = False
            if high2low_cross_attn_mask is not None:
                high2low_cross_attn_mask[torch.where(
                    high2low_cross_attn_mask.sum(-1) == high2low_cross_attn_mask.shape[-1])] = False
            if low2high_cross_attn_mask is not None:
                low2high_cross_attn_mask[torch.where(
                    low2high_cross_attn_mask.sum(-1) == low2high_cross_attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            if self.optim_global_decoder:
                global_layer = self.global_transformer_decoder.layers[i]
            high2low_layer = self.high2low_transformer_decoder.layers[i]
            low2high_layer = self.low2high_transformer_decoder.layers[i]
            # cross_attn_mask is prepared for cross attention
            # if we use relation_query, then we need setup another self_attn_mask
            # for self attention, to decide whether interact triplet query
            # with relation query.
            if i in ignore_masked_attention_layers:
                if self.optim_global_decoder:
                    global_cross_attn_mask = None
                high2low_cross_attn_mask = None
                low2high_cross_attn_mask = None
            self_attn_mask = None
            if self.optim_global_decoder:
                global_attn_masks = [global_cross_attn_mask, self_attn_mask]
            high2low_attn_masks = [high2low_cross_attn_mask, self_attn_mask]
            low2high_attn_masks = [low2high_cross_attn_mask, self_attn_mask]
            if self.optim_global_decoder:
                global_query_feat = global_layer(
                    query=global_query_feat,
                    key=decoder_inputs[level_idx],
                    value=decoder_inputs[level_idx],
                    query_pos=global_query_embed,
                    key_pos=decoder_positional_encodings[level_idx],
                    attn_masks=global_attn_masks,
                    query_key_padding_mask=None,
                    # here we do not apply masking on padded region
                    key_padding_mask=None)
            high2low_query_feat = high2low_layer(
                query=high2low_query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=high2low_query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=high2low_attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            low2high_query_feat = low2high_layer(
                query=low2high_query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=low2high_query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=low2high_attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)

            if (i in self.aux_loss_list) or (i+1 == self.num_transformer_decoder_layers):
                if self.optim_global_decoder:
                    cls_pred, bbox_pred, mask_pred, global_cross_attn_mask = self.forward_head(
                        global_query_feat, mask_features, multi_scale_memorys[(
                            i + 1) % self.num_transformer_feat_level].shape[-2:],
                        decoder_layer_idx=i+1, transformer_decoder=self.global_transformer_decoder, rel_cls_embed=self.global_rel_cls_embed)
                    global_cls_pred_list.append(cls_pred)
                    global_bbox_pred_list.append(bbox_pred)
                    global_mask_pred_list.append(mask_pred)
                cls_pred, bbox_pred, mask_pred, high2low_cross_attn_mask = self.forward_head(
                    high2low_query_feat, mask_features, multi_scale_memorys[(
                        i + 1) % self.num_transformer_feat_level].shape[-2:],
                    decoder_layer_idx=i+1, transformer_decoder=self.high2low_transformer_decoder, rel_cls_embed=self.high2low_rel_cls_embed)
                high2low_cls_pred_list.append(cls_pred)
                high2low_bbox_pred_list.append(bbox_pred)
                high2low_mask_pred_list.append(mask_pred)
                cls_pred, bbox_pred, mask_pred, low2high_cross_attn_mask = self.forward_head(
                    low2high_query_feat, mask_features, multi_scale_memorys[(
                        i + 1) % self.num_transformer_feat_level].shape[-2:],
                    decoder_layer_idx=i+1, transformer_decoder=self.low2high_transformer_decoder, rel_cls_embed=self.low2high_rel_cls_embed)
                low2high_cls_pred_list.append(cls_pred)
                low2high_bbox_pred_list.append(bbox_pred)
                low2high_mask_pred_list.append(mask_pred)
            else:
                if self.optim_global_decoder:
                    global_cross_attn_mask = None
                high2low_cross_attn_mask = None
                low2high_cross_attn_mask = None

        if self.optim_global_decoder:
            global_all_cls_scores, global_all_bbox_preds, global_all_mask_preds = self.pack_model_outputs(
                global_cls_pred_list, global_bbox_pred_list, global_mask_pred_list)
        high2low_all_cls_scores, high2low_all_bbox_preds, high2low_all_mask_preds = self.pack_model_outputs(
            high2low_cls_pred_list, high2low_bbox_pred_list, high2low_mask_pred_list)
        low2high_all_cls_scores, low2high_all_bbox_preds, low2high_all_mask_preds = self.pack_model_outputs(
            low2high_cls_pred_list, low2high_bbox_pred_list, low2high_mask_pred_list)

        if self.optim_global_decoder:
            return global_all_cls_scores, global_all_bbox_preds, global_all_mask_preds, \
                high2low_all_cls_scores, high2low_all_bbox_preds, high2low_all_mask_preds, \
                low2high_all_cls_scores, low2high_all_bbox_preds, low2high_all_mask_preds

        if os.getenv('MERGE_PREDICT', 'false').lower() == 'true' and not self.training:
            merge_type = 'soft_merge'
            if merge_type == 'hard_merge':
                pred1 = (high2low_all_cls_scores,
                         high2low_all_bbox_preds, high2low_all_mask_preds)
                pred2 = (low2high_all_cls_scores,
                         low2high_all_bbox_preds, low2high_all_mask_preds)

                gt_labels = kwargs['gt_labels'][0]
                gt_bboxes = kwargs['gt_bboxes'][0]
                gt_masks = kwargs['gt_masks'][0]
                gt_rels = kwargs['gt_rels'][0]
                # high2low_gt_rels = kwargs['high2low_gt_rels'][0]
                # low2high_gt_rels = kwargs['low2high_gt_rels'][0]
                gt_bboxes_ignore = None
                gt = (gt_labels, gt_bboxes, gt_masks,
                      gt_rels, gt_rels, gt_rels, img_metas, gt_bboxes_ignore)

                input1 = pred1 + pred1 + gt
                input2 = pred2 + pred2 + gt
                loss1 = self.loss(*input1)
                loss2 = self.loss(*input2)

                loss1_value = sum([x.cpu().detach().item()
                                   for x in loss1.values()])
                loss2_value = sum([x.cpu().detach().item()
                                   for x in loss2.values()])

                if loss1_value < loss2_value:
                    all_cls_scores = high2low_all_cls_scores
                    all_bbox_preds = high2low_all_bbox_preds
                    all_mask_preds = high2low_all_mask_preds
                else:
                    all_cls_scores = low2high_all_cls_scores
                    all_bbox_preds = low2high_all_bbox_preds
                    all_mask_preds = low2high_all_mask_preds
            elif merge_type == 'soft_merge':
                all_cls_scores = {
                    'sub': high2low_all_cls_scores['sub'] * 0.5 + low2high_all_cls_scores['sub'] * 0.5,
                    'obj': high2low_all_cls_scores['obj'] * 0.5 + low2high_all_cls_scores['obj'] * 0.5,
                    'rel': high2low_all_cls_scores['rel'] * 0.5 + low2high_all_cls_scores['rel'] * 0.5, }
                all_bbox_preds = {
                    'sub': high2low_all_bbox_preds['sub'] * 0.5 + low2high_all_bbox_preds['sub'] * 0.5,
                    'obj': high2low_all_bbox_preds['obj'] * 0.5 + low2high_all_bbox_preds['obj'] * 0.5, }
                all_mask_preds = {
                    'sub': high2low_all_mask_preds['sub'] * 0.5 + low2high_all_mask_preds['sub'] * 0.5,
                    'obj': high2low_all_mask_preds['obj'] * 0.5 + low2high_all_mask_preds['obj'] * 0.5, }

            return all_cls_scores, all_bbox_preds, all_mask_preds, all_cls_scores, all_bbox_preds, all_mask_preds

        return high2low_all_cls_scores, high2low_all_bbox_preds, high2low_all_mask_preds, low2high_all_cls_scores, low2high_all_bbox_preds, low2high_all_mask_preds

    @torch.no_grad()
    def update_perspective_decoder(self):
        # use global decoder and parameter mapping to update high2low and low2high decoder parameters.
        for (global_n, global_p), (high2low_n, high2low_p), (low2high_n, low2high_p) in zip(
                self.global_transformer_decoder.named_parameters(),
                self.high2low_transformer_decoder.named_parameters(),
                self.low2high_transformer_decoder.named_parameters()):
            assert global_n.replace('global', 'high2low') == high2low_n
            assert global_n.replace('global', 'low2high') == low2high_n
            high2low_p.data = self.pm_dict[high2low_n.replace('.', '_')](
                global_p.data)
            low2high_p.data = self.pm_dict[low2high_n.replace('.', '_')](
                global_p.data)

    def pack_model_outputs(self, cls_pred_list, bbox_pred_list, mask_pred_list):
        all_s_cls_scores = []
        all_o_cls_scores = []
        all_r_cls_scores = []
        all_s_bbox_preds = []
        all_o_bbox_preds = []
        all_s_mask_preds = []
        all_o_mask_preds = []
        for cls_pred, bbox_pred, mask_pred in zip(cls_pred_list, bbox_pred_list, mask_pred_list):
            sub_cls_pred = cls_pred['sub']
            obj_cls_pred = cls_pred['obj']
            rel_cls_pred = cls_pred['rel']
            sub_bbox_pred = bbox_pred['sub']
            obj_bbox_pred = bbox_pred['obj']
            sub_mask_pred = mask_pred['sub']
            obj_mask_pred = mask_pred['obj']
            all_s_cls_scores.append(sub_cls_pred)
            all_o_cls_scores.append(obj_cls_pred)
            all_r_cls_scores.append(rel_cls_pred)
            all_s_bbox_preds.append(sub_bbox_pred)
            all_o_bbox_preds.append(obj_bbox_pred)
            all_s_mask_preds.append(sub_mask_pred)
            all_o_mask_preds.append(obj_mask_pred)
        all_s_cls_scores = torch.stack(all_s_cls_scores, dim=0)
        all_o_cls_scores = torch.stack(all_o_cls_scores, dim=0)
        all_r_cls_scores = torch.stack(all_r_cls_scores, dim=0)
        all_s_bbox_preds = torch.stack(all_s_bbox_preds, dim=0)
        all_o_bbox_preds = torch.stack(all_o_bbox_preds, dim=0)
        all_s_mask_preds = torch.stack(all_s_mask_preds, dim=0)
        all_o_mask_preds = torch.stack(all_o_mask_preds, dim=0)
        all_cls_scores = dict(
            sub=all_s_cls_scores,
            obj=all_o_cls_scores,
            rel=all_r_cls_scores)
        all_bbox_preds = dict(
            sub=all_s_bbox_preds,
            obj=all_o_bbox_preds)
        all_mask_preds = dict(
            sub=all_s_mask_preds,
            obj=all_o_mask_preds)
        return all_cls_scores, all_bbox_preds, all_mask_preds

    def forward_train(self,
                      feats,
                      img_metas,
                      gt_rels,
                      high2low_gt_rels,
                      low2high_gt_rels,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # not consider ignoring bboxes
        assert gt_bboxes_ignore is None

        outs = self(feats, img_metas)
        # forward
        if self.optim_global_decoder:
            assert len(outs) == 9, 'If optim_global_decoder=True, forward output number should be 9, but now is {}'.format(
                len(outs))
        else:
            assert len(outs) == 6, 'If optim_global_decoder=False, forward output number should be 6, but now is {}'.format(
                len(outs))

        gts = (gt_labels, gt_bboxes, gt_masks, gt_rels, high2low_gt_rels, low2high_gt_rels, img_metas, gt_bboxes_ignore)  # noqa
        # loss
        losses_global = dict()
        if self.optim_global_decoder:
            loss_inputs = outs[3:9] + gts
            loss_global_inputs = outs + gts
            losses_global = self.loss_global(*loss_global_inputs)
        else:
            loss_inputs = outs + gts
        losses = self.loss(*loss_inputs)
        losses.update(losses_global)

        return losses

    def simple_test(self, feats, img_metas, rescale=False, **kwargs):
        """Test without augmentaton.

        Args:
            feats (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two tensors.

            - mask_cls_results (Tensor): Mask classification logits,\
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            - mask_pred_results (Tensor): Mask logits, shape \
                (batch_size, num_queries, h, w).
        """
        outs = self(feats, img_metas, **kwargs)
        if len(outs) == 6:
            if self.test_forward_output_type == 'high2low':
                outs = outs[0:3]
            elif self.test_forward_output_type == 'low2high':
                outs = outs[3:6]
        if len(outs) == 9:
            if self.test_forward_output_type == 'global':
                outs = outs[0:3]
            elif self.test_forward_output_type == 'high2low':
                outs = outs[3:6]
            elif self.test_forward_output_type == 'low2high':
                outs = outs[6:9]
        results_list = self.get_results(*outs, img_metas, rescale=rescale)
        return results_list

    def _dedup_triplets_based_on_iou(self, s_labels, o_labels, r_labels, s_mask_pred, o_mask_pred):
        relation_classes = defaultdict(lambda: [])
        for k, (s_l, o_l, r_l) in enumerate(zip(s_labels, o_labels, r_labels)):
            relation_classes[(s_l.item(), o_l.item(),
                              r_l.item())].append(k)
        s_binary_masks = s_mask_pred.to(torch.float).flatten(1)
        o_binary_masks = o_mask_pred.to(torch.float).flatten(1)

        def dedup_triplets(triplets_ids, s_binary_masks, o_binary_masks, keep_tri):
            while len(triplets_ids) > 1:
                base_s_mask = s_binary_masks[triplets_ids[0]].unsqueeze(0)
                base_o_mask = o_binary_masks[triplets_ids[0]].unsqueeze(0)
                other_s_masks = s_binary_masks[triplets_ids[1:]]
                other_o_masks = o_binary_masks[triplets_ids[1:]]
                # calculate ious
                s_ious = base_s_mask.mm(other_s_masks.transpose(
                    0, 1))/((base_s_mask+other_s_masks) > 0).sum(-1)
                o_ious = base_o_mask.mm(other_o_masks.transpose(
                    0, 1))/((base_o_mask+other_o_masks) > 0).sum(-1)
                ids_left = []
                for s_iou, o_iou, other_id in zip(s_ious[0], o_ious[0], triplets_ids[1:]):
                    if (s_iou > 0.5) & (o_iou > 0.5):
                        keep_tri[other_id] = False
                    else:
                        ids_left.append(other_id)
                triplets_ids = ids_left
            return keep_tri

        keep_tri = torch.ones_like(
            r_labels, dtype=torch.bool, device=r_labels.device)
        for triplets_ids in relation_classes.values():
            if len(triplets_ids) > 1:
                keep_tri = dedup_triplets(
                    triplets_ids, s_binary_masks, o_binary_masks, keep_tri)

        return keep_tri

    def _get_results_single_easy(self,
                                 s_cls_score, o_cls_score, r_cls_score,
                                 s_bbox_pred, o_bbox_pred,
                                 s_mask_pred, o_mask_pred,
                                 img_shape, scale_factor, rescale=False):

        # because input is half size of mask, here should follow pre-process, not post-process.
        # mask_size = (round(img_shape[0] / scale_factor[1]),
        #              round(img_shape[1] / scale_factor[0]))
        mask_size = (img_shape[0] // 2, img_shape[1] // 2)
        max_per_img = self.num_queries

        ###################
        # sub/obj/rel cls #
        ###################
        s_logits = F.softmax(s_cls_score, dim=-1)[..., :-1]
        o_logits = F.softmax(o_cls_score, dim=-1)[..., :-1]
        s_scores, s_labels = s_logits.max(-1)
        o_scores, o_labels = o_logits.max(-1)

        r_lgs = F.softmax(r_cls_score, dim=-1)
        r_logits = r_lgs[..., 1:]
        r_scores, r_indexes = r_logits.reshape(-1).topk(max_per_img)
        r_labels = r_indexes % self.num_relations + 1
        triplet_index = r_indexes // self.num_relations

        s_scores = s_scores[triplet_index]
        s_labels = s_labels[triplet_index]
        s_bbox_pred = s_bbox_pred[triplet_index]
        s_mask_pred = s_mask_pred[triplet_index]

        o_scores = o_scores[triplet_index]
        o_labels = o_labels[triplet_index]
        o_bbox_pred = o_bbox_pred[triplet_index]
        o_mask_pred = o_mask_pred[triplet_index]

        r_dists = r_lgs.reshape(-1, self.num_relations + 1)[triplet_index]

        # same as post-process
        keep = (s_scores > 0.5) & (o_scores > 0.5) & (r_scores > 0.3)
        s_scores = s_scores[keep]
        s_labels = s_labels[keep]
        s_bbox_pred = s_bbox_pred[keep]
        s_mask_pred = s_mask_pred[keep]
        o_scores = o_scores[keep]
        o_labels = o_labels[keep]
        o_bbox_pred = o_bbox_pred[keep]
        o_mask_pred = o_mask_pred[keep]
        r_scores = r_scores[keep]
        r_labels = r_labels[keep]
        r_dists = r_dists[keep]

        ################
        # sub/obj bbox #
        ################
        s_det_bboxes = bbox_cxcywh_to_xyxy(s_bbox_pred)
        s_det_bboxes[:, 0::2] = s_det_bboxes[:, 0::2] * img_shape[1]
        s_det_bboxes[:, 1::2] = s_det_bboxes[:, 1::2] * img_shape[0]
        s_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        s_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            s_det_bboxes /= s_det_bboxes.new_tensor(scale_factor)
        s_det_bboxes = torch.cat((s_det_bboxes, s_scores.unsqueeze(1)), -1)

        o_det_bboxes = bbox_cxcywh_to_xyxy(o_bbox_pred)
        o_det_bboxes[:, 0::2] = o_det_bboxes[:, 0::2] * img_shape[1]
        o_det_bboxes[:, 1::2] = o_det_bboxes[:, 1::2] * img_shape[0]
        o_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        o_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            o_det_bboxes /= o_det_bboxes.new_tensor(scale_factor)
        o_det_bboxes = torch.cat((o_det_bboxes, o_scores.unsqueeze(1)), -1)

        ################
        # sub/obj mask #
        ################
        s_mask_pred = F.interpolate(s_mask_pred.unsqueeze(1),
                                    size=mask_size).squeeze(1)
        o_mask_pred = F.interpolate(o_mask_pred.unsqueeze(1),
                                    size=mask_size).squeeze(1)
        s_mask_pred = torch.sigmoid(s_mask_pred) > 0.85
        o_mask_pred = torch.sigmoid(o_mask_pred) > 0.85

        ### triplets deduplicate ###
        keep_tri = self._dedup_triplets_based_on_iou(
            s_labels, o_labels, r_labels, s_mask_pred, o_mask_pred)

        scores = torch.cat((s_scores[keep_tri], o_scores[keep_tri]), 0)
        # object, (2*n)
        labels = torch.cat((s_labels[keep_tri], o_labels[keep_tri]), 0)
        # object bbox, (2*n, 5)
        det_bboxes = torch.cat(
            (s_det_bboxes[keep_tri], o_det_bboxes[keep_tri]), 0)
        # object mask, (2*n, h, w)
        output_masks = torch.cat(
            (s_mask_pred[keep_tri], o_mask_pred[keep_tri]), 0)
        # relation (n)
        r_labels = r_labels[keep_tri]
        r_scores = r_scores[keep_tri]
        r_dists = r_dists[keep_tri]
        # (n, 2)
        rel_pairs = torch.arange(keep_tri.sum()*2,
                                 dtype=r_labels.dtype,
                                 device=r_labels.device).reshape(2, -1).T

        return rel_pairs, labels, scores, det_bboxes, output_masks, r_labels, r_scores, r_dists


class MLConv1d(nn.Module):
    """Very simple multi-layer conv1d."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, kernel_size=1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv1d(n, k, kernel_size) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
