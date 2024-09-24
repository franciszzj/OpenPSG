import os
from pathlib import Path
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from mmcv.runner import BaseModule
from mmdet.core import INSTANCE_OFFSET
from mmdet.models.builder import HEADS
from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel, AutoTokenizer, AutoModelForCausalLM
from timm.layers import PatchEmbed

from kings_sgg.models.detectors.mask2former_relation_v2 import object_categories, relation_categories


@HEADS.register_module()
class RelationTransformerHeadV4(BaseModule):
    def __init__(self,
                 # relation qformer
                 qformer_model_name='Salesforce/instructblip-vicuna-7b',
                 qformer_instruction='Is there a relation between {} and {}?',
                 patch_size=16,
                 qformer_layer_num=2,
                 qformer_feature_size=768,
                 sampled_qformer_batch_size=32,
                 qformer_neg_over_pos=3,
                 rel_cls_type='binary',  # binary or multiclass or binary+multiclass
                 rel_cls_loss_weight=50.0,
                 # llm
                 llm_model_name='meta-llama/Llama-2-7b-hf',
                 llm_instruction='What are the relations between {} and {}? Assistant: ',
                 llm_truncate_num=-1,
                 llm_feature_size=4096,
                 max_llm_forward_num=4,
                 pair_selector_threshold=0.5,
                 # object and relation
                 num_object_classes=133,
                 object_feature_size=256,
                 relation_classes=relation_categories,
                 max_object_num=30,
                 **kwargs):
        super().__init__()
        ##########
        # Config #
        ##########
        # relation qformer
        self.qformer_instruction = qformer_instruction
        self.patch_size = patch_size
        self.qformer_layer_num = qformer_layer_num
        self.qformer_feature_size = qformer_feature_size
        self.sampled_qformer_batch_size = sampled_qformer_batch_size
        self.qformer_neg_over_pos = qformer_neg_over_pos
        self.rel_cls_type = rel_cls_type
        self.rel_cls_loss_weight = rel_cls_loss_weight
        # llm
        self.llm_instruction = llm_instruction
        self.llm_truncate_num = llm_truncate_num
        self.llm_feature_size = llm_feature_size
        self.max_llm_forward_num = max_llm_forward_num
        self.pair_selector_threshold = pair_selector_threshold
        # object and relation
        self.num_object_classes = num_object_classes
        self.object_feature_size = object_feature_size
        self.relation_classes = relation_classes
        self.num_relation_classes = len(relation_classes)
        self.max_object_num = max_object_num

        #########
        # Model #
        #########
        self.patch_embed = PatchEmbed(
            img_size=None, patch_size=patch_size, in_chans=object_feature_size, embed_dim=object_feature_size)
        # relation qformer
        relation_qformer_config = InstructBlipQFormerConfig(
            hidden_size=qformer_feature_size,
            num_hidden_layers=qformer_layer_num,
            cross_attention_frequency=1,
            encoder_hidden_size=object_feature_size,)
        self.relation_qformer = InstructBlipQFormerModel(
            relation_qformer_config)
        self.relation_qformer_tokenizer = AutoTokenizer.from_pretrained(
            qformer_model_name, subfolder="qformer_tokenizer")
        self.relation_query = nn.Parameter(
            torch.randn(1, 32, qformer_feature_size))
        self.rel_cls_query = nn.Parameter(
            torch.randn(1, 1, qformer_feature_size))
        if 'binary' in self.rel_cls_type:
            self.binary_rel_cls_pred = nn.Linear(qformer_feature_size, 1)
        if 'multiclass' in self.rel_cls_type:
            self.multiclass_rel_cls_pred = nn.Linear(
                qformer_feature_size, self.num_relation_classes)
        # llm
        self.language_projection = nn.Linear(
            qformer_feature_size, llm_feature_size)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name, low_cpu_mem_usage=True, trust_remote_code=True)
        if self.llm_truncate_num > 0:
            self.language_model.model.layers = self.language_model.model.layers[
                :self.llm_truncate_num]
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

    def forward(self, inputs, is_generation=None):
        # parse inputs
        image_feature = inputs['mask_features']
        batch_size = image_feature.shape[0]
        meta_info = inputs['img_metas'][0]
        assert batch_size == 1, 'only support batch size 1 for now.'

        if self.training:
            object_num = len(meta_info['masks_info'])
            object_names = [object_categories[x['category']]
                            for x in meta_info['masks_info']]
            gt_thing_masks = inputs['gt_masks'][0]
            gt_thing_labels = inputs['gt_labels'][0]
            gt_semantic_seg = inputs['gt_semantic_seg'][0]

            # construct relation label
            relation_target = image_feature.new_zeros(
                (object_num, object_num, self.num_relation_classes))
            for ii, jj, relation_class in meta_info['gt_rels'][0]:
                relation_target[ii, jj, relation_class] = 1
            if 'binary' in self.rel_cls_type:
                binary_rel_cls_label = (relation_target.sum(
                    2) > 0).to(relation_target.dtype)
            if 'multiclass' in self.rel_cls_type:
                multiclass_rel_cls_label = relation_target.permute((2, 0, 1))
            relation_label_index = torch.nonzero(
                relation_target, as_tuple=False)
        else:
            object_info = inputs['object_info'][0]
            object_id_list = object_info['object_id_list'][:self.max_object_num]
            object_num = len(object_id_list)
            object_names = [object_categories[x.item() % INSTANCE_OFFSET]
                            for x in object_id_list]
            pan_masks = object_info['pan_results']

        ##############################
        #      relation qformer      #
        ##############################
        # qformer instruction builder
        qformer_batch_size = object_num * object_num
        qformer_instructions = [self.qformer_instruction.format(
            object_names[i // object_num], object_names[i % object_num]) for i in range(qformer_batch_size)]
        qformer_instructions = self.relation_qformer_tokenizer(
            qformer_instructions, return_tensors="pt", padding=True, return_attention_mask=True)
        qformer_input_ids = qformer_instructions['input_ids'].cuda()
        qformer_input_mask = qformer_instructions['attention_mask'].cuda()

        # relation exist cls
        rel_cls_query = self.rel_cls_query.expand(qformer_batch_size, -1, -1)
        relation_query = self.relation_query.expand(qformer_batch_size, -1, -1)
        qformer_query = torch.cat([rel_cls_query, relation_query], dim=1)
        qformer_attention_mask = torch.cat([torch.ones(
            qformer_batch_size, qformer_query.shape[1]).cuda(), qformer_input_mask], dim=1)

        # pairwise and patching
        if self.training:
            image_patches, pair_masks = self.prepare_train(
                image_feature, meta_info, gt_thing_masks, gt_thing_labels, gt_semantic_seg)
        else:
            image_patches, pair_masks = self.prepare_inference(
                image_feature, meta_info, object_id_list, pan_masks)
        image_patches = image_patches.expand(qformer_batch_size, -1, -1)
        # cross-attention mask
        pair_masks = pair_masks.expand(-1, qformer_query.shape[1], -1)

        if self.training:
            qformer_sampled_idxes = self.qformer_sampler(relation_target)
        else:
            qformer_sampled_idxes = torch.arange(qformer_batch_size).cuda()

        # qformer forward
        qformer_outputs = torch.zeros_like(qformer_query)
        sampled_qformer_outputs = self.relation_qformer(
            input_ids=qformer_input_ids[qformer_sampled_idxes],
            attention_mask=qformer_attention_mask[qformer_sampled_idxes],
            query_embeds=qformer_query[qformer_sampled_idxes],
            encoder_hidden_states=image_patches[qformer_sampled_idxes],
            encoder_attention_mask=pair_masks[qformer_sampled_idxes],
        )['last_hidden_state'][:, :qformer_query.shape[1]]
        qformer_outputs[qformer_sampled_idxes] = sampled_qformer_outputs
        if self.training:
            cls_feature = sampled_qformer_outputs[:, 0]
            if 'binary' in self.rel_cls_type:
                binary_rel_cls_pred = self.binary_rel_cls_pred(cls_feature)
                binary_rel_cls_pred = binary_rel_cls_pred.view(-1)
                binary_rel_cls_label = binary_rel_cls_label.view(-1)[
                    qformer_sampled_idxes]
                binary_rel_cls_loss = self.loss_for_rel_cls_pred(
                    binary_rel_cls_pred, binary_rel_cls_label)
            if 'multiclass' in self.rel_cls_type:
                multiclass_rel_cls_pred = self.multiclass_rel_cls_pred(
                    cls_feature)
                multiclass_rel_cls_pred = multiclass_rel_cls_pred.view(
                    (-1, self.num_relation_classes))
                multiclass_rel_cls_label = multiclass_rel_cls_label.view(
                    (self.num_relation_classes, -1)).permute((1, 0))[qformer_sampled_idxes]
                multiclass_rel_cls_loss = self.loss_for_rel_cls_pred(
                    multiclass_rel_cls_pred, multiclass_rel_cls_label)
        else:
            cls_feature = qformer_outputs[:, 0]
            if 'binary' in self.rel_cls_type:
                binary_rel_cls_pred = self.binary_rel_cls_pred(cls_feature)
                binary_rel_cls_pred = torch.sigmoid(binary_rel_cls_pred)
            if 'multiclass' in self.rel_cls_type:
                multiclass_rel_cls_pred = self.multiclass_rel_cls_pred(
                    cls_feature)
                multiclass_rel_cls_pred = torch.sigmoid(
                    multiclass_rel_cls_pred)
        pair_feature = qformer_outputs[:, 1:]

        #########################
        #      llm forward      #
        #########################
        # selector
        if self.training:
            selected_idxes = [x[0] * object_num + x[1]
                              for x in relation_label_index.tolist()]
            selected_idxes = random.sample(selected_idxes, min(
                len(selected_idxes), self.max_llm_forward_num))
            if len(selected_idxes) == 0:
                selected_idxes = random.sample(
                    list(range(qformer_batch_size)), min(qformer_batch_size, self.max_llm_forward_num))
        else:
            # selected_idxes = torch.nonzero(
            #     rel_cls_pred.squeeze() > self.pair_selector_threshold, as_tuple=False).squeeze().tolist()
            # if len(selected_idxes) < max(self.max_llm_forward_num, qformer_batch_size):
            #     more_selected_idxes = rel_cls_pred.squeeze().topk(self.max_llm_forward_num).indices.tolist()
            #     selected_idxes = list(set(selected_idxes) | set(more_selected_idxes))
            if 'binary' in self.rel_cls_type:
                selected_idxes = binary_rel_cls_pred.squeeze(1).topk(
                    qformer_batch_size).indices.tolist()[:20]
            if 'multiclass' in self.rel_cls_type:
                # rel_cls_pred: (batch_size, object_num, object_num, relation_classes)
                for i in range(object_num):
                    multiclass_rel_cls_pred[:, i, i, :] = 0
                multiclass_rel_cls_pred = multiclass_rel_cls_pred.view(
                    (qformer_batch_size, self.num_relation_classes))
                _, topk_indices = torch.topk(
                    multiclass_rel_cls_pred.reshape(-1), k=100)
                rel_pred_list = []
                rel_score_list = []
                for idx in topk_indices:
                    sub_obj_idx = idx // qformer_batch_size
                    rel_idx = idx % qformer_batch_size
                    sub_idx = sub_obj_idx // object_num
                    obj_idx = sub_obj_idx % object_num
                    rel_pred = [sub_idx, obj_idx, rel_idx]
                    rel_score = multiclass_rel_cls_pred.reshape(-1)[idx]
                    if rel_pred not in rel_pred_list:
                        rel_pred_list.append(rel_pred)
                        rel_score_list.append(rel_score)

        # llm instruction builder
        llm_instructions = [self.llm_instruction.format(
            object_names[i // object_num], object_names[i % object_num]) for i in selected_idxes]
        self.llm_tokenizer.padding_side = 'left'
        llm_input = self.llm_tokenizer(
            llm_instructions, return_tensors="pt", padding=True, return_attention_mask=True)
        llm_input_ids = llm_input['input_ids'].cuda()
        llm_attention_masks = llm_input['attention_mask'].cuda()
        if self.training:
            llm_labels = ['' for _ in selected_idxes]
            relation_target_list = relation_target.view(
                (-1, self.num_relation_classes)).tolist()
            for i, si in enumerate(selected_idxes):
                rel_target = relation_target_list[si]
                for rel_label_id, exist in enumerate(rel_target):
                    if exist:
                        llm_labels[i] += ' {} </s>'.format(
                            self.relation_classes[rel_label_id])
            self.llm_tokenizer.padding_side = 'right'
            llm_label = self.llm_tokenizer(
                llm_labels, return_tensors="pt", padding=True, return_attention_mask=True)
            llm_label_input_ids = llm_label['input_ids'].cuda()
            llm_label_attention_mask = llm_label['attention_mask'].cuda()
            llm_input_ids = torch.cat(
                [llm_input_ids, llm_label_input_ids], dim=1)
            llm_attention_masks = torch.cat(
                [llm_attention_masks, llm_label_attention_mask], dim=1)

        # llm forward
        if self.training:
            rel_llm_loss_list = []
        else:
            llm_rel_pred_list = []
            llm_rel_score_list = []
        for i, si in enumerate(selected_idxes):
            llm_input = self.language_projection(pair_feature[si])
            llm_input_id = llm_input_ids[i]
            llm_input_embeds = self.language_model.get_input_embeddings()(llm_input_id)
            llm_input_embeds = torch.cat([llm_input, llm_input_embeds], dim=0)
            llm_attention_mask = torch.cat([torch.ones(
                llm_input.size()[:-1], dtype=torch.long).cuda(), llm_attention_masks[i]], dim=0)
            llm_input_embeds = llm_input_embeds.unsqueeze(0)
            llm_attention_mask = llm_attention_mask.unsqueeze(0)
            if is_generation == None:
                is_generation = not self.training
            if is_generation:  # inference
                outputs = self.language_model.generate(
                    inputs_embeds=llm_input_embeds,
                    attention_mask=llm_attention_mask,
                    max_new_tokens=16,
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                relation_pred_str = self.llm_tokenizer.batch_decode(outputs.sequences)[
                    0]
                relation_pred = relation_pred_str.split(
                    '<s>')[1].split('</s>')[0].strip()
                for relation_name in relation_pred.split('  '):
                    if relation_name in relation_categories:
                        sub_idx = si // object_num
                        obj_idx = si % object_num
                        rel_idx = relation_categories.index(relation_name)
                        rel_pred = [sub_idx, obj_idx, rel_idx]
                        if rel_pred not in llm_rel_pred_list:
                            llm_rel_pred_list.append(
                                [sub_idx, obj_idx, rel_idx])
                            llm_rel_score_list.append(1)
            else:  # train
                outputs = self.language_model(
                    inputs_embeds=llm_input_embeds,
                    attention_mask=llm_attention_mask,
                )
                logits = outputs.logits[:, -llm_label_input_ids.size(1):, :]
                if self.training:
                    labels = torch.where(
                        llm_label_attention_mask[i:i+1].bool(), llm_label_input_ids[i:i+1], torch.ones_like(llm_label_input_ids[i:i+1]) * -100)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss(reduction="mean")
                    this_rel_llm_loss = loss_fct(
                        shift_logits.view(-1, self.language_model.config.vocab_size), shift_labels.view(-1))
                    rel_llm_loss_list.append(this_rel_llm_loss)

        # output
        output_dict = dict()
        if self.training:
            if 'binary' in self.rel_cls_type:
                output_dict['binary_rel_cls_loss'] = binary_rel_cls_loss
            if 'multiclass' in self.rel_cls_type:
                output_dict['multiclass_rel_cls_loss'] = multiclass_rel_cls_loss
            output_dict['rel_llm_loss'] = torch.stack(
                rel_llm_loss_list).mean()
            # output_dict['multiclass_rel_cls_recall@20'] = self.get_recall_N(
            #     multiclass_rel_cls_pred.view((1, object_num, object_num, self.num_relation_classes)), relation_target.view((1, object_num, object_num, self.num_relation_classes)))
        else:
            output_dict['rel_pred'] = llm_rel_pred_list + rel_pred_list
            output_dict['rel_score'] = llm_rel_score_list + rel_score_list

        return output_dict

    def prepare_train(self, image_feature, meta_info, gt_thing_masks, gt_thing_labels, gt_semantic_seg):
        # patching
        image_patches = self.patch_embed(image_feature)

        # pair mask
        dtype = image_feature.dtype
        device = image_feature.device
        feat_h, feat_w = image_feature.shape[-2:]
        masks_info = meta_info['masks_info']
        object_num = len(masks_info)

        gt_thing_masks = gt_thing_masks.to_tensor(dtype, device)
        # gt_thing_masks already padded
        # img_h, img_w = meta_info['img_shape'][:2]
        # gt_thing_masks = F.interpolate(gt_thing_masks.unsqueeze(1).float(), size=(
        #     img_h, img_w), mode='bilinear', align_corners=False).squeeze(1) > 0.5
        # pad_h, pad_w = meta_info['pad_shape'][:2]
        # gt_thing_masks = F.pad(gt_thing_masks.unsqueeze(
        #     1), (0, pad_w-img_w, 0, pad_h-img_h), value=0).squeeze(1)
        gt_thing_masks = F.interpolate(gt_thing_masks.unsqueeze(0).float(), size=(
            feat_h // self.patch_size, feat_w // self.patch_size), mode='bilinear', align_corners=False).squeeze(0) > 0.5

        gt_semantic_seg = gt_semantic_seg.to(dtype).to(device)
        # gt_semantic_seg already padded
        # gt_semantic_seg = F.interpolate(gt_semantic_seg.unsqueeze(1).float(), size=(
        #     img_h, img_w), mode='nearest').squeeze(1)
        # gt_semantic_seg = F.pad(gt_semantic_seg.unsqueeze(1), (0, pad_w-img_w, 0, pad_h-img_h), value=0).squeeze(1)
        gt_semantic_seg = F.interpolate(gt_semantic_seg.unsqueeze(0).float(), size=(
            feat_h // self.patch_size, feat_w // self.patch_size), mode='nearest').squeeze(0)

        mask_list = []
        thing_idx = 0
        for mask_info in masks_info:
            label = mask_info['category']
            if mask_info['is_thing']:
                mask = gt_thing_masks[thing_idx:thing_idx+1]
                thing_idx += 1
            else:
                mask = gt_semantic_seg == label
            mask_list.append(mask)
        masks = torch.cat(mask_list, dim=0)
        pair_masks = [torch.logical_or(masks[i], masks[j]) for i in range(
            len(masks)) for j in range(len(masks))]
        pair_masks = torch.stack(pair_masks, dim=0)
        pair_masks = pair_masks.reshape((object_num * object_num, 1, -1))

        return image_patches, pair_masks

    def prepare_inference(self, image_feature, meta_info, object_id_list, pan_masks):
        # patching
        image_patches = self.patch_embed(image_feature)

        # pair mask
        feat_h, feat_w = image_feature.shape[-2:]
        object_num = len(object_id_list)

        img_h, img_w = meta_info['img_shape'][:2]
        pan_masks = F.interpolate(pan_masks[None, None, ...].float(), size=(
            img_h, img_w), mode='nearest').squeeze()
        pad_h, pad_w = meta_info['pad_shape'][:2]
        pan_masks = F.pad(
            pan_masks[None, None, ...], (0, pad_w-img_w, 0, pad_h-img_h), value=0).squeeze()
        pan_masks = F.interpolate(pan_masks[None, None, ...].float(), size=(
            feat_h // self.patch_size, feat_w // self.patch_size), mode='nearest').squeeze()

        mask_list = []
        for object_id in object_id_list:
            mask = pan_masks[None, ...] == object_id
            mask_list.append(mask)
        masks = torch.cat(mask_list, dim=0)
        pair_masks = [torch.logical_or(masks[i], masks[j]) for i in range(
            len(masks)) for j in range(len(masks))]
        pair_masks = torch.stack(pair_masks, dim=0)
        pair_masks = pair_masks.reshape((object_num * object_num, 1, -1))

        return image_patches, pair_masks

    def qformer_sampler(self, relation_target):
        '''
        relation_target: [object_num, object_num, relation_classes]
        '''
        relation_target = relation_target.reshape(-1,
                                                  self.num_relation_classes)
        relation_target = relation_target.sum(1)
        positive_idxes = torch.nonzero(
            relation_target, as_tuple=False)[:, 0]
        negative_idxes = torch.nonzero(
            relation_target == 0, as_tuple=False)[:, 0]
        positive_num = positive_idxes.shape[0]
        negative_num = negative_idxes.shape[0]
        if positive_num < self.sampled_qformer_batch_size:
            sampled_positive_idxes = positive_idxes
            sampled_negative_idxes = negative_idxes[torch.randint(
                0, negative_num, (min(self.sampled_qformer_batch_size - positive_num, positive_num * self.qformer_neg_over_pos),))]
        else:
            sampled_positive_idxes = positive_idxes[torch.randint(
                0, positive_num, (self.sampled_qformer_batch_size // (self.qformer_neg_over_pos + 1),))]
            sampled_negative_idxes = negative_idxes[torch.randint(
                0, negative_num, (self.sampled_qformer_batch_size * self.qformer_neg_over_pos // (self.qformer_neg_over_pos + 1),))]
        sampled_idxes = torch.cat(
            [sampled_positive_idxes, sampled_negative_idxes], dim=0)
        return sampled_idxes

    def loss_for_rel_cls_pred(self, rel_cls_pred, rel_cls_label):
        '''
        # case 1:
        rel_cls_pred: [pair_num]
        rel_cls_label: [pair_num]
        # case 2:
        rel_cls_pred: [relation_num, pair_num]
        rel_cls_label: [relation_num, pair_num]
        '''
        if rel_cls_pred.dim() == 1:
            loss = F.binary_cross_entropy_with_logits(
                rel_cls_pred, rel_cls_label)
        elif rel_cls_pred.dim() == 2:
            loss = self.multilabel_categorical_crossentropy(
                rel_cls_label, rel_cls_pred)
            weight = (loss / loss.max()) ** 1
            loss = loss * weight
            loss = torch.mean(loss)
        loss *= self.rel_cls_loss_weight
        return loss

    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        """https://kexue.fm/archives/7359
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 9999
        y_pred_pos = y_pred - (1 - y_true) * 9999
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss

    def get_recall_N(self, y_pred, y_true, object_num=20):
        """
            y_pred: [batch_size, relation_num, object_num, object_num]
            y_true: [batch_size, relation_num, object_num, object_num]
        """

        device = y_pred.device
        dtype = y_pred.dtype
        recall_list = []

        for idx in range(len(y_true)):
            sample_y_true = []
            sample_y_pred = []

            # find topk
            _, topk_indices = torch.topk(
                y_true[idx:idx+1].reshape([-1, ]), k=object_num)
            for index in topk_indices:
                # pred_relation = index // (y_true.shape[2] ** 2)
                pred_relation = torch.div(
                    index, y_true.shape[2] ** 2, rounding_mode='floor')
                index_subject_object = index % (y_true.shape[2] ** 2)
                # pred_subject = index_subject_object // y_true.shape[2]
                pred_subject = torch.div(
                    index_subject_object, y_true.shape[2], rounding_mode='floor')
                pred_object = index_subject_object % y_true.shape[2]
                if y_true[idx, pred_relation, pred_subject, pred_object] == 0:
                    continue
                sample_y_true.append(
                    [pred_subject, pred_object, pred_relation])

            # find topk
            _, topk_indices = torch.topk(
                y_pred[idx:idx+1].reshape([-1, ]), k=object_num)
            for index in topk_indices:
                # pred_relation = index // (y_pred.shape[2] ** 2)
                pred_relation = torch.div(
                    index, y_pred.shape[2] ** 2, rounding_mode='floor')
                index_subject_object = index % (y_pred.shape[2] ** 2)
                # pred_subject = index_subject_object // y_pred.shape[2]
                pred_subject = torch.div(
                    index_subject_object, y_pred.shape[2], rounding_mode='floor')
                pred_object = index_subject_object % y_pred.shape[2]
                sample_y_pred.append(
                    [pred_subject, pred_object, pred_relation])

            recall = len([x for x in sample_y_pred if x in sample_y_true]
                         ) / (len(sample_y_true) + 1e-8)
            recall_list.append(recall)

        mean_recall = torch.tensor(recall_list).to(device).mean() * 100
        return mean_recall
