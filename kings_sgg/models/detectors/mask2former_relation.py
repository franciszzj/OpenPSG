import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
# Use ground truth panoptic segmentation to evaluate the performance of visual relation detection
import os
import cv2
import json
from panopticapi.utils import rgb2id, id2rgb
# '''

from mmdet.core import INSTANCE_OFFSET, bbox2result
from mmdet.models.builder import DETECTORS, build_head
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.detectors import Mask2Former


@DETECTORS.register_module()
class Mask2FormerRelation(Mask2Former):
    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 relation_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):

        if train_cfg is not None:
            freeze_layers = train_cfg.pop('freeze_layers', [])
        else:
            freeze_layers = []

        super(Mask2FormerRelation, self).__init__(
            backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg
        )

        self.object_token_size = 1
        self.relation_head = build_head(relation_head)
        self.object_cls_embed = nn.Embedding(
            self.relation_head.num_object_classes, self.relation_head.input_feature_size)
        self.max_object_num = self.relation_head.max_object_num
        self.embedding_add_cls = self.relation_head.embedding_add_cls
        self.merge_cls_type = self.relation_head.merge_cls_type
        if hasattr(self.relation_head, 'postional_encoding_layer') and self.relation_head.postional_encoding_layer is not None:
            self.add_postional_encoding = True
        else:
            self.add_postional_encoding = False
        self.use_background_feature = self.relation_head.use_background_feature

        for layer_name in freeze_layers:
            for n, p in self.named_parameters():
                if n.startswith(layer_name):
                    p.requires_grad = False

        '''
        # Use ground truth panoptic segmentation to evaluate the performance of visual relation detection
        psg_file = './data/psg/psg.json'
        self.test_data_dict = {}
        test_data = load_json(psg_file)
        test_id_list = get_test_id(psg_file)
        for d in test_data['data']:
            image_id = d['image_id']
            if image_id not in test_id_list:
                continue
            self.test_data_dict[d['file_name'].split('/')[-1]] = d
        # '''

    def _id_is_thing(self, old_idx, gt_thing_label):
        if old_idx < len(gt_thing_label):
            return True
        else:
            return False

    def _mask_pooling(self, feature, mask, output_size=1):
        '''
        feature [256, h, w]
        mask [1, h, w]
        output_size == 1: mean
        '''
        # 1. 根据mask选出mask_feature中的对应点的特征。
        if mask.sum() <= 0:
            return feature.new_zeros([output_size, feature.shape[0]])
        mask_bool = (mask >= 0.5)[0]
        feats = feature[:, mask_bool]

        # 2. 根据output_size构造输出的feature
        if feats.shape[1] < output_size:
            feats = torch.cat(
                [feats] * int(np.ceil(output_size/feats.shape[1])), dim=1)
            feats = feats[:, :output_size]

        split_list = [feats.shape[1] // output_size] * output_size
        for idx in range(feats.shape[1] - sum(split_list)):
            split_list[idx] += 1
        feats_list = torch.split(feats, split_list, dim=1)
        feats_mean_list = [feat.mean(dim=1)[None] for feat in feats_list]
        feats_tensor = torch.cat(feats_mean_list, dim=0)
        # [output_size, 256]
        return feats_tensor

    def _thing_embedding(self, idx, feature, gt_thing_mask, gt_thing_label, meta_info):
        device = feature.device
        dtype = feature.dtype

        gt_mask = gt_thing_mask.to_ndarray()
        gt_mask = gt_mask[idx: idx + 1]
        gt_mask = torch.from_numpy(gt_mask).to(device).to(dtype)

        h_img, w_img = meta_info['img_shape'][:2]
        gt_mask = F.interpolate(gt_mask[:, None], size=(h_img, w_img))[:, 0]
        h_pad, w_pad = meta_info['pad_shape'][:2]
        gt_mask = F.pad(gt_mask[:, None],
                        (0, w_pad-w_img, 0, h_pad-h_img))[:, 0]
        h_feature, w_feature = feature.shape[-2:]
        gt_mask = F.interpolate(
            gt_mask[:, None], size=(h_feature, w_feature))[:, 0]

        embedding_thing = self._mask_pooling(
            feature, gt_mask, output_size=self.object_token_size)  # [output_size, 256]

        cls_feature_thing = self.object_cls_embed(
            gt_thing_label[idx: idx + 1].reshape([-1, ]))  # [1, 256]

        if self.embedding_add_cls:
            if self.merge_cls_type == 'cat':
                embedding_thing = torch.cat(
                    [embedding_thing, cls_feature_thing], dim=-1)
            elif self.merge_cls_type == 'add':
                embedding_thing = embedding_thing + cls_feature_thing

        if self.add_postional_encoding:
            # [1, h, w]
            pos_embed_zeros = feature.new_zeros((1, ) + feature.shape[-2:])
            # [1, 256, h, w]
            pos_embed = self.relation_head.postional_encoding_layer(
                pos_embed_zeros)
            pos_embed_mask_pooling = self._mask_pooling(
                pos_embed[0], gt_mask, output_size=self.object_token_size)
            embedding_thing = embedding_thing + pos_embed_mask_pooling

        if self.use_background_feature:
            background_feature = self._mask_pooling(
                feature, 1 - gt_mask, output_size=self.object_token_size)  # [output_size, 256]
            embedding_thing = embedding_thing + background_feature

        # [output_size, 256]
        return embedding_thing

    def _stuff_embedding(self, idx, feature, masks_info, gt_semantic_seg):
        device = feature.device
        dtype = feature.dtype

        category_stuff = masks_info[idx]['category']
        mask_stuff = gt_semantic_seg == category_stuff
        mask_stuff = mask_stuff.to(dtype)
        mask_stuff = F.interpolate(mask_stuff[None], size=(
            feature.shape[1], feature.shape[2]))[0]
        label_stuff = torch.tensor(category_stuff).to(device).to(torch.long)

        embedding_stuff = self._mask_pooling(
            feature, mask_stuff, output_size=self.object_token_size)  # [output_size, 256]

        cls_feature_stuff = self.object_cls_embed(
            label_stuff.reshape([-1, ]))  # [1, 256]

        if self.embedding_add_cls:
            if self.merge_cls_type == 'cat':
                embedding_stuff = torch.cat(
                    [embedding_stuff, cls_feature_stuff], dim=-1)
            elif self.merge_cls_type == 'add':
                embedding_stuff = embedding_stuff + cls_feature_stuff

        if self.add_postional_encoding:
            # [1, h, w]
            pos_embed_zeros = feature.new_zeros((1, ) + feature.shape[-2:])
            # [1, 256, h, w]
            pos_embed = self.relation_head.postional_encoding_layer(
                pos_embed_zeros)
            pos_embed_mask_pooling = self._mask_pooling(
                pos_embed[0], mask_stuff, output_size=self.object_token_size)
            embedding_stuff = embedding_stuff + pos_embed_mask_pooling

        if self.use_background_feature:
            background_feature = self._mask_pooling(
                feature, 1 - mask_stuff, output_size=self.object_token_size)  # [output_size, 256]
            embedding_stuff = embedding_stuff + background_feature

        # [output_size, 256]
        return embedding_stuff

    def _get_input_and_target(self, feature, meta_info, gt_thing_mask, gt_thing_label, gt_semantic_seg):
        """
        Input:
            feature: [256, h, w]
            meta_info: dict
            gt_thing_mask: bitmap, n
            gt_thing_label:
        Output:
            embedding: [1, n, 256]
            rel_target: [1, 56, n, n]
        """

        masks_info = meta_info['masks_info']
        num_keep = min(self.max_object_num, len(masks_info))
        keep_idx_list = random.sample(list(range(len(masks_info))), num_keep)
        old2new_dict = {old: new for new, old in enumerate(keep_idx_list)}

        embedding_list = []
        for idx, old in enumerate(keep_idx_list):
            if self._id_is_thing(old, gt_thing_label):
                embedding = self._thing_embedding(
                    old, feature, gt_thing_mask, gt_thing_label, meta_info)
            else:
                embedding = self._stuff_embedding(
                    old, feature, masks_info, gt_semantic_seg)
            embedding_list.append(embedding[None])
        # [1, n * self.object_token_size, 256]
        embedding = torch.cat(embedding_list, dim=1)

        if self.relation_head.loss_type == 'v0_softmax':
            rel_target = feature.new_zeros(
                [1, 1, embedding.shape[1], embedding.shape[1]])
            # last one is background, so background index should - 1.
            rel_target += self.relation_head.num_relation_classes - 1
            for ii, jj, cls_relationship in meta_info['gt_rels'][0]:
                if not (ii in old2new_dict and jj in old2new_dict):
                    continue
                new_ii, new_jj = old2new_dict[ii], old2new_dict[jj]
                rel_target[0, 0, new_ii, new_jj] = cls_relationship
        else:
            rel_target = feature.new_zeros(
                [1, self.relation_head.num_relation_classes, embedding.shape[1], embedding.shape[1]])
            for ii, jj, cls_relationship in meta_info['gt_rels'][0]:
                if not (ii in old2new_dict and jj in old2new_dict):
                    continue
                new_ii, new_jj = old2new_dict[ii], old2new_dict[jj]
                rel_target[0, cls_relationship, new_ii, new_jj] = 1

        return embedding, rel_target

    def _get_input(self, pan_result, object_id_list, object_score_list, feature_map, meta_info):
        device = feature_map.device
        dtype = feature_map.dtype

        ori_height, ori_width = meta_info['ori_shape'][:2]
        resize_height, resize_width = meta_info['img_shape'][:2]
        pad_height, pad_width = meta_info['pad_shape'][:2]

        mask_list = []
        class_list = []
        for object_id in object_id_list:
            if object_id == 133:
                continue
            mask = pan_result == object_id
            cls = object_id % INSTANCE_OFFSET
            mask_list.append(mask)
            class_list.append(cls)

        if len(mask_list) == 0:
            return None

        class_tensor = torch.tensor(class_list).to(device).to(torch.long)[None]
        object_cls_embedding = self.object_cls_embed(class_tensor)

        mask_tensor = torch.stack(mask_list)[None]
        mask_tensor = (mask_tensor * 1).to(dtype)
        h_img, w_img = resize_height, resize_width
        mask_tensor = F.interpolate(mask_tensor, size=(h_img, w_img))
        h_pad, w_pad = pad_height, pad_width
        mask_tensor = F.pad(mask_tensor, (0, w_pad-w_img, 0, h_pad-h_img))
        h_feature, w_feature = feature_map.shape[-2:]
        mask_tensor = F.interpolate(mask_tensor, size=(h_feature, w_feature))
        mask_tensor = mask_tensor[0][:, None]

        # feature_map [bs, 256, h, w]
        # mask_tensor [n, 1, h, w]
        object_embedding = (
            feature_map * mask_tensor).sum(dim=[2, 3]) / (mask_tensor.sum(dim=[2, 3]) + 1e-8)
        object_embedding = object_embedding[None]
        if self.embedding_add_cls:
            if self.merge_cls_type == 'cat':
                object_embedding = torch.cat(
                    [object_embedding, object_cls_embedding], dim=-1)
            elif self.merge_cls_type == 'add':
                object_embedding = object_embedding + object_cls_embedding

        if self.add_postional_encoding:
            pos_embed_zeros = feature_map[0].new_zeros(
                (1, ) + feature_map[0].shape[-2:])
            pos_embed = self.relationship.postional_encoding_layer(
                pos_embed_zeros)
            for idx in range(object_embedding.shape[1]):
                pos_embed_mask_pooling = self._mask_pooling(
                    pos_embed[0], mask_tensor[idx], output_size=self.object_token_size)
                object_embedding[0, idx] = \
                    object_embedding[0, idx] + pos_embed_mask_pooling

        if self.use_background_feature:
            background_mask = 1 - mask_tensor
            background_feature = (feature_map * background_mask).sum(
                dim=[2, 3]) / (background_mask.sum(dim=[2, 3]) + 1e-8)
            background_feature = background_feature[None]
            # object_embedding [1, n, 256]
            object_embedding = object_embedding + background_feature

        # object_embedding [1, n, 256]
        return object_embedding, object_id_list, object_score_list

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        # add batch_input_shape in img_metas
        super(SingleStageDetector, self).forward_train(img, img_metas)
        feats = self.extract_feat(img)
        losses, mask_features = self.panoptic_head.forward_train(feats, img_metas, gt_bboxes,
                                                                 gt_labels, gt_masks,
                                                                 gt_semantic_seg,
                                                                 gt_bboxes_ignore)

        rel_input_list = []
        rel_target_list = []

        num_imgs = len(img_metas)

        for idx in range(num_imgs):
            embedding, rel_target = self._get_input_and_target(
                mask_features[idx],
                img_metas[idx],
                gt_masks[idx],
                gt_labels[idx],
                gt_semantic_seg[idx],
            )
            rel_input_list.append(embedding)
            rel_target_list.append(rel_target)

        max_length = max([e.shape[1]
                          for e in rel_input_list])
        mask_attention = mask_features.new_zeros([num_imgs, max_length])
        for idx in range(num_imgs):
            mask_attention[idx, :rel_input_list[idx].shape[1]] = 1.
        rel_input_list = [
            F.pad(e, [0, 0, 0, max_length-e.shape[1]])
            for e in rel_input_list
        ]
        rel_target_list = [
            F.pad(t, [0, max_length-t.shape[3], 0, max_length-t.shape[2]])
            for t in rel_target_list
        ]
        rel_input = torch.cat(rel_input_list, dim=0)
        rel_target = torch.cat(rel_target_list, dim=0)
        rel_output = self.relation_head(
            rel_input, mask_attention)
        loss_relation = self.relation_head.loss(
            rel_output, rel_target, mask_attention)
        losses.update(loss_relation)
        torch.cuda.empty_cache()
        return losses

    def simple_test(self, imgs, img_metas, **kwargs):
        feats = self.extract_feat(imgs)
        cls_pred_results, mask_pred_results, mask_features = self.panoptic_head.simple_test(
            feats, img_metas, **kwargs)

        '''
        # Use ground truth panoptic segmentation to evaluate the performance of visual relation detection
        key = img_metas[0]['filename'].split('/')[-1]
        meta_info = self.test_data_dict[key]
        pan_seg = cv2.imread(os.path.join(
            './data/coco', meta_info['pan_seg_file_name']))
        pan_seg = cv2.cvtColor(pan_seg, cv2.COLOR_BGR2RGB)
        pan_seg = cv2.resize(
            pan_seg, (img_metas[0]['img_shape'][1], img_metas[0]['img_shape'][0]), interpolation=cv2.INTER_NEAREST)
        pan_seg_id = rgb2id(pan_seg)
        cls_gt_results = []
        mask_gt_results = []
        for seg_info in meta_info['segments_info']:
            cls_id = seg_info['category_id']
            cls_gt_results.append(cls_id)
            padded_mask = np.zeros(img_metas[0]['pad_shape'][:2], dtype=np.float32)
            mask = (pan_seg_id == seg_info['id']).astype(np.float32)
            padded_mask[:mask.shape[0], :mask.shape[1]] = mask
            mask_gt_results.append(padded_mask)
        cls_gt_results = torch.from_numpy(np.array(cls_gt_results))
        cls_gt_results = F.one_hot(
            cls_gt_results, num_classes=self.num_classes + 1) * 100.
        cls_gt_results = cls_gt_results.to(
            cls_pred_results.dtype).to(cls_pred_results.device)
        mask_gt_results = torch.from_numpy(np.array(mask_gt_results)) * 100. - 50.
        mask_gt_results = mask_gt_results.to(
            mask_pred_results.dtype).to(mask_pred_results.device)
        cls_pred_results = cls_gt_results.unsqueeze(0)
        mask_pred_results = mask_gt_results.unsqueeze(0)
        # '''

        results = self.panoptic_fusion_head.simple_test(
            cls_pred_results, mask_pred_results, img_metas, **kwargs)

        if not self.test_cfg.get('predict_relation', False):
            for i in range(len(results)):
                if 'pan_results' in results[i]:
                    results[i]['pan_results'] = results[i]['pan_results'].detach(
                    ).cpu().numpy()

                if 'ins_results' in results[i]:
                    labels_per_image, bboxes, mask_pred_binary = results[i][
                        'ins_results']
                    bbox_results = bbox2result(bboxes, labels_per_image,
                                               self.num_things_classes)
                    mask_results = [[] for _ in range(self.num_things_classes)]
                    for j, label in enumerate(labels_per_image):
                        mask = mask_pred_binary[j].detach().cpu().numpy()
                        mask_results[label].append(mask)
                    results[i]['ins_results'] = bbox_results, mask_results

                assert 'sem_results' not in results[i], 'segmantic segmentation '\
                    'results are not supported yet.'

            if self.num_stuff_classes == 0:
                results = [res['ins_results'] for res in results]

        else:
            # predict relation
            assert len(img_metas) == 1
            batch_idx = 0
            device = mask_features.device
            dtype = mask_features.dtype

            res = results[batch_idx]
            pan_results = res['pan_results']
            object_id_list = res['object_id_list']
            object_score_list = res['object_score_list']

            object_res = self._get_input(
                pan_result=pan_results,
                object_id_list=object_id_list,
                object_score_list=object_score_list,
                feature_map=mask_features,
                meta_info=img_metas[batch_idx]
            )

            relation_res = []
            relation_scores = []
            if object_res is not None:
                object_embedding, object_id_list, object_score_list = object_res
                # relation_output: [batch_size, num_relation_classes, object_num, object_num]
                relation_output = self.relation_head(
                    object_embedding, attention_mask=None)
                relation_output = relation_output[0]

                # Discard the relationship corresponding to the object itself
                for idx_i in range(relation_output.shape[1]):
                    relation_output[:, idx_i, idx_i] = -9999

                loss_type = self.relation_head.loss_type
                if loss_type == 'v0_softmax':
                    relation_output = torch.softmax(relation_output, dim=1)
                    relation_output = relation_output[:(
                        self.relation_head.num_relation_classes-1)]
                elif loss_type == 'v0_sigmoid':
                    relation_output = torch.sigmoid(relation_output)
                elif loss_type in ['v1', 'v1_no_bs_limit']:
                    relation_output = torch.exp(relation_output)
                    # relation_output = torch.sigmoid(relation_output)

                # relation_output * subject score * object score
                object_score_tensor = torch.tensor(
                    object_score_list, device=device, dtype=dtype)
                relation_output = relation_output * \
                    object_score_tensor[None, :, None]
                relation_output = relation_output * \
                    object_score_tensor[None, None, :]

                # find topk
                if relation_output.shape[1] > 1:
                    _, topk_indices = torch.topk(
                        relation_output.reshape([-1, ]), k=100)

                    # subject, object, relation
                    for index in topk_indices:
                        pred_relation = index // (
                            relation_output.shape[1] ** 2)
                        index_subject_object = index % (
                            relation_output.shape[1] ** 2)
                        relation_score = relation_output.reshape(
                            [-1, ])[index].item()
                        pred_subject = index_subject_object // relation_output.shape[1]
                        pred_object = index_subject_object % relation_output.shape[1]
                        pred = [pred_subject.item(),
                                pred_object.item(),
                                pred_relation.item()]
                        relation_res.append(pred)
                        relation_scores.append(relation_score)

            rl = dict(
                object_id_list=[oid.item() for oid in object_id_list],
                relation=relation_res)
            res['rel_results'] = rl
            res['pan_results'] = res['pan_results'].detach().cpu().numpy()
            res['rel_scores'] = relation_scores

            results = [res]

        return results


'''
# Use ground truth panoptic segmentation to evaluate the performance of visual relation detection
def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def get_test_id(test_file):
    dataset = load_json(test_file)
    test_id_list = [
        d['image_id'] for d in dataset['data'] if (d['image_id'] in dataset['test_image_ids']) and (len(d['relations']) != 0)
    ]
    return test_id_list
# '''
