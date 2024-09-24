import os
import random
import numpy as np
import dbm
import yaml
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import INSTANCE_OFFSET, bbox2result
from mmdet.models.builder import DETECTORS, build_head
from mmdet.models.detectors.base import BaseDetector

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from openseed.BaseModel import BaseModel
from openseed import build_model

from kings_sgg.models.detectors.mask2former_relation_v2 import object_categories, relation_categories


@DETECTORS.register_module()
class OpenSeeDRelation(BaseDetector):
    def __init__(self,
                 # openseed
                 openseed_config_path='',
                 openseed_pretrained_path='',
                 thing_classes=[],
                 stuff_classes=[],
                 # relation
                 relation_head=None,
                 # text
                 text_info_db_dir='./data/psg/openai/gpt-3.5-turbo',
                 text_embed_db_dir='./data/psg/openai/gpt-3.5-turbo_text-embedding-ada-002',
                 text_embedding_size=1536,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(OpenSeeDRelation, self).__init__(init_cfg)
        ##################################
        #            OpenSeeD            #
        ##################################
        with open(openseed_config_path, encoding='utf-8') as f:
            openseed_config = yaml.safe_load(f)
        openseed_config['WEIGHT'] = openseed_pretrained_path
        self.openseed = BaseModel(openseed_config,
                                  build_model(openseed_config)).from_pretrained(openseed_pretrained_path).cuda().eval()
        thing_colors = [random_color(rgb=True, maximum=255).astype(
            int).tolist() for _ in range(len(thing_classes))]
        stuff_colors = [random_color(rgb=True, maximum=255).astype(
            int).tolist() for _ in range(len(stuff_classes))]
        thing_dataset_id_to_contiguous_id = {
            x: x for x in range(len(thing_classes))}
        stuff_dataset_id_to_contiguous_id = {
            x+len(thing_classes): x for x in range(len(stuff_classes))}

        MetadataCatalog.get("demo").set(
            thing_colors=thing_colors,
            thing_classes=thing_classes,
            thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
            stuff_colors=stuff_colors,
            stuff_classes=stuff_classes,
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
        )
        self.openseed.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            thing_classes + stuff_classes, is_eval=False)
        metadata = MetadataCatalog.get('demo')
        self.openseed.model.metadata = metadata
        self.openseed.model.sem_seg_head.num_classes = len(
            thing_classes + stuff_classes)

        ##################################
        #            Relation            #
        ##################################
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

        if train_cfg is not None:
            freeze_layers = train_cfg.pop('freeze_layers', [])
        else:
            freeze_layers = []
        for layer_name in freeze_layers:
            for n, p in self.named_parameters():
                if n.startswith(layer_name):
                    p.requires_grad = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.text_info_db = dbm.open(
            os.path.join(text_info_db_dir, 'kv.db'), 'r')
        self.text_embedding_size = text_embedding_size
        self.text_embed_db = dbm.open(
            os.path.join(text_embed_db_dir, 'kv.db'), 'r')

    def extract_feat(self, img):
        pass

    def aug_test(self, imgs, img_metas, rescale=False):
        pass

    def onnx_export(self, img, img_metas, with_nms=True):
        pass

    def forward_openseed(self, imgs, img_metas, mode='test'):
        assert imgs.size(0) == 1, 'only support batch size 1'
        img = imgs[0]
        img_meta = img_metas[0]
        # 1. build openseed batch_inputs
        # the input scale for openseed is 0~255, while mmdet is -2.x~2.x. Note OpenSeed's mean and std are same as mmdet.
        pixel_mean = self.openseed.model.pixel_mean.clone().to(img.device).view(3, 1, 1)
        pixel_std = self.openseed.model.pixel_std.clone().to(img.device).view(3, 1, 1)
        img = img * pixel_std + pixel_mean
        # input to openseed should not be padded.
        rm_pad_h, rm_pad_w = img_meta['img_shape'][:2]
        img = img[:, :rm_pad_h, :rm_pad_w]
        batch_inputs = [{'image': img, 'height': img_meta['ori_shape']
                         [0], 'width': img_meta['ori_shape'][1]}]
        # 2. forward openseed
        openseed_outputs, mask_features = self.openseed.forward(batch_inputs)
        openseed_output = openseed_outputs[0]
        if mode == 'train':
            losses = dict()
            return losses, mask_features
        # 3. change openseed output to mmdet format
        _pan_results = openseed_output['panoptic_seg'][0].cpu().numpy()
        pan_results = np.zeros_like(_pan_results)
        record = dict()
        _object_id_list = []
        for inst_dict in openseed_output['panoptic_seg'][1]:
            id = inst_dict['id']
            category_id = inst_dict['category_id']
            if category_id not in record:
                record[category_id] = 0
            else:
                record[category_id] += 1
            _object_id_list.append((id, category_id, record[category_id]))
            index = np.where(_pan_results == id)
            pan_results[index] = category_id + \
                INSTANCE_OFFSET * record[category_id]
        pan_results = torch.from_numpy(pan_results).to(img.device)
        object_id_list = [torch.tensor(
            x[1] + INSTANCE_OFFSET*x[2], dtype=torch.int32) for x in _object_id_list]
        object_score_list = [torch.tensor(1.0)
                             for _ in openseed_output['panoptic_seg'][1]]
        ins_results_class = openseed_output['instances'].get('pred_classes')
        ins_results_box = openseed_output['instances'].get('pred_boxes').tensor
        ins_results_score = openseed_output['instances'].get('scores')
        ins_results_bbox = torch.cat(
            [ins_results_box, ins_results_score[:, None]], dim=1)
        ins_results_mask = openseed_output['instances'].get('pred_masks')
        ins_results = (ins_results_class, ins_results_bbox, ins_results_mask)
        result = {'pan_results': pan_results, 'object_id_list': object_id_list,
                  'object_score_list': object_score_list, 'ins_results': ins_results}
        results = [result]
        return results, mask_features

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
        # 1. select corresponding features in mask_features according to mask.
        if mask.sum() <= 0:
            return feature.new_zeros([output_size, feature.shape[0]])
        mask_bool = (mask >= 0.5)[0]
        feats = feature[:, mask_bool]

        # 2. construct output feature according to output_size.
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
            gt_thing_label: [n]
            gt_semantic_seg: [h, w]
        Output:
            object_embedding: [1, n, 256]
            sub_obj_embedding: [1, n * n, 256]
            sub_obj_str_list: [n * n], list of str
            sub_obj_text_embedding: [1, n * n, 1536]
            relation_target: [1, 56, n, n]
        """

        masks_info = meta_info['masks_info']
        num_keep = min(self.max_object_num, len(masks_info))
        keep_idx_list = random.sample(list(range(len(masks_info))), num_keep)
        old2new_dict = {old: new for new, old in enumerate(keep_idx_list)}

        # 1. object embedding
        object_embedding_list = []
        for idx, old in enumerate(keep_idx_list):
            if self._id_is_thing(old, gt_thing_label):
                object_embedding = self._thing_embedding(
                    old, feature, gt_thing_mask, gt_thing_label, meta_info)
            else:
                object_embedding = self._stuff_embedding(
                    old, feature, masks_info, gt_semantic_seg)
            object_embedding_list.append(object_embedding[None])
        # [1, n * self.object_token_size, 256]
        object_embedding = torch.cat(object_embedding_list, dim=1)

        use_text_db = self.relation_head.use_pair_text_vision_cross or \
            self.relation_head.use_pair_vision_text_cross or \
            self.relation_head.use_triplet_vision_text_cross

        # 2. sub obj pair input (embedding, str_list)
        sub_obj_embedding_list = []
        sub_obj_name_list = []
        sub_obj_str_list = []
        sub_obj_text_embedding_list = []
        obj_label_list = [0 for _ in keep_idx_list]
        for old_idx in keep_idx_list:
            new_idx = old2new_dict[old_idx]
            obj_label_list[new_idx] = masks_info[old_idx]['category']
        obj_name_list = [object_categories[object_id]
                         for object_id in obj_label_list]
        for i, sub_name in enumerate(obj_name_list):
            for j, obj_name in enumerate(obj_name_list):
                key = sub_name + '#' + obj_name
                sub_obj_name = key
                if use_text_db and key in self.text_info_db:
                    sub_obj_str = pickle.loads(self.text_info_db[key])
                else:
                    sub_obj_str = ''
                if use_text_db and key in self.text_embed_db:
                    sub_obj_text_embedding = torch.from_numpy(np.array(pickle.loads(
                        self.text_embed_db[key])))
                else:
                    sub_obj_text_embedding = feature.new_zeros(
                        [self.text_embedding_size])
                # cat sub and obj embedding
                sub_obj_embedding = torch.cat(
                    [object_embedding[0, i], object_embedding[0, j]], dim=0)
                sub_obj_embedding_list.append(sub_obj_embedding)
                sub_obj_name_list.append(sub_obj_name)
                sub_obj_str_list.append(sub_obj_str)
                sub_obj_text_embedding_list.append(sub_obj_text_embedding)
        sub_obj_embedding = torch.stack(sub_obj_embedding_list)
        sub_obj_embedding = sub_obj_embedding.unsqueeze(0)
        sub_obj_text_embedding = torch.stack(sub_obj_text_embedding_list)
        sub_obj_text_embedding = sub_obj_text_embedding.unsqueeze(0)
        sub_obj_text_embedding = sub_obj_text_embedding.to(
            feature.device).to(feature.dtype)

        # 3. sub obj rel triplet input (embedding, str_list)
        sub_obj_rel_name_list = []
        sub_obj_rel_str_list = []
        sub_obj_rel_text_embedding_list = []
        for i, sub_name, in enumerate(obj_name_list):
            for j, obj_name in enumerate(obj_name_list):
                for k, rel_name in enumerate(relation_categories):
                    key = sub_name + '#' + obj_name + '#' + rel_name
                    sub_obj_rel_name = key
                    if use_text_db and key in self.text_info_db:
                        sub_obj_rel_str = pickle.loads(self.text_info_db[key])
                    else:
                        sub_obj_rel_str = ''
                    if use_text_db and key in self.text_embed_db:
                        sub_obj_rel_text_embedding = torch.from_numpy(np.array(pickle.loads(
                            self.text_embed_db[key])))
                    else:
                        sub_obj_rel_text_embedding = feature.new_zeros(
                            [self.text_embedding_size])
                    sub_obj_rel_name_list.append(sub_obj_rel_name)
                    sub_obj_rel_str_list.append(sub_obj_rel_str)
                    sub_obj_rel_text_embedding_list.append(
                        sub_obj_rel_text_embedding)
        sub_obj_rel_text_embedding = torch.stack(
            sub_obj_rel_text_embedding_list)
        sub_obj_rel_text_embedding = sub_obj_rel_text_embedding.unsqueeze(0)
        sub_obj_rel_text_embedding = sub_obj_rel_text_embedding.to(
            feature.device).to(feature.dtype)

        # 4. relation target
        if self.relation_head.loss_type == 'v0_softmax':
            relation_target = feature.new_zeros(
                [1, 1, object_embedding.shape[1], object_embedding.shape[1]])
            # last one is background, so background index should - 1.
            relation_target += self.relation_head.num_relation_classes - 1
            for ii, jj, cls_relationship in meta_info['gt_rels'][0]:
                if not (ii in old2new_dict and jj in old2new_dict):
                    continue
                new_ii, new_jj = old2new_dict[ii], old2new_dict[jj]
                relation_target[0, 0, new_ii, new_jj] = cls_relationship
        else:
            relation_target = feature.new_zeros(
                [1, self.relation_head.num_relation_classes, object_embedding.shape[1], object_embedding.shape[1]])
            for ii, jj, cls_relationship in meta_info['gt_rels'][0]:
                if not (ii in old2new_dict and jj in old2new_dict):
                    continue
                new_ii, new_jj = old2new_dict[ii], old2new_dict[jj]
                relation_target[0, cls_relationship, new_ii, new_jj] = 1

        return_dict = dict()
        return_dict['object_embedding'] = object_embedding
        return_dict['sub_obj_embedding'] = sub_obj_embedding
        return_dict['sub_obj_name_list'] = sub_obj_name_list
        return_dict['sub_obj_str_list'] = sub_obj_str_list
        return_dict['sub_obj_text_embedding'] = sub_obj_text_embedding
        return_dict['sub_obj_rel_name_list'] = sub_obj_rel_name_list
        return_dict['sub_obj_rel_str_list'] = sub_obj_rel_str_list
        return_dict['sub_obj_rel_text_embedding'] = sub_obj_rel_text_embedding
        return_dict['relation_target'] = relation_target
        return return_dict

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

        use_text_db = self.relation_head.use_pair_text_vision_cross or self.relation_head.use_pair_vision_text_cross
        sub_obj_embedding_list = []
        sub_obj_name_list = []
        sub_obj_str_list = []
        sub_obj_text_embedding_list = []
        obj_name_list = [object_categories[object_id if isinstance(object_id, int) else object_id.item()]
                         for object_id in class_list]
        for i, sub_name in enumerate(obj_name_list):
            for j, obj_name in enumerate(obj_name_list):
                key = sub_name + '#' + obj_name
                sub_obj_name = key
                if use_text_db and key in self.text_info_db:
                    sub_obj_str = pickle.loads(self.text_info_db[key])
                else:
                    sub_obj_str = ''
                if use_text_db and key in self.text_embed_db:
                    sub_obj_text_embedding = torch.from_numpy(np.array(pickle.loads(
                        self.text_embed_db[key])))
                else:
                    sub_obj_text_embedding = feature_map.new_zeros(
                        [self.text_embedding_size])

                sub_obj_embedding = torch.cat(
                    [object_embedding[0, i], object_embedding[0, j]], dim=0)
                sub_obj_embedding_list.append(sub_obj_embedding)
                sub_obj_name_list.append(sub_obj_name)
                sub_obj_str_list.append(sub_obj_str)
                sub_obj_text_embedding_list.append(sub_obj_text_embedding)
        sub_obj_embedding = torch.stack(sub_obj_embedding_list)
        sub_obj_embedding = sub_obj_embedding.unsqueeze(0)
        sub_obj_text_embedding = torch.stack(sub_obj_text_embedding_list)
        sub_obj_text_embedding = sub_obj_text_embedding.unsqueeze(0)
        sub_obj_text_embedding = sub_obj_text_embedding.to(device).to(dtype)

        # sub obj rel triplet input (embedding, str_list)
        sub_obj_rel_name_list = []
        sub_obj_rel_str_list = []
        sub_obj_rel_text_embedding_list = []
        for i, sub_name, in enumerate(obj_name_list):
            for j, obj_name in enumerate(obj_name_list):
                for k, rel_name in enumerate(relation_categories):
                    key = sub_name + '#' + obj_name + '#' + rel_name
                    sub_obj_rel_name = key
                    if use_text_db and key in self.text_info_db:
                        sub_obj_rel_str = pickle.loads(self.text_info_db[key])
                    else:
                        sub_obj_rel_str = ''
                    if use_text_db and key in self.text_embed_db:
                        sub_obj_rel_text_embedding = torch.from_numpy(np.array(pickle.loads(
                            self.text_embed_db[key])))
                    else:
                        sub_obj_rel_text_embedding = feature_map.new_zeros(
                            [self.text_embedding_size])
                    sub_obj_rel_name_list.append(sub_obj_rel_name)
                    sub_obj_rel_str_list.append(sub_obj_rel_str)
                    sub_obj_rel_text_embedding_list.append(
                        sub_obj_rel_text_embedding)
        sub_obj_rel_text_embedding = torch.stack(
            sub_obj_rel_text_embedding_list)
        sub_obj_rel_text_embedding = sub_obj_rel_text_embedding.unsqueeze(0)
        sub_obj_rel_text_embedding = sub_obj_rel_text_embedding.to(
            device).to(dtype)

        return_dict = dict()
        # object_embedding [1, n, 256]
        return_dict['object_embedding'] = object_embedding
        return_dict['object_id_list'] = object_id_list
        return_dict['object_score_list'] = object_score_list
        return_dict['sub_obj_embedding'] = sub_obj_embedding
        return_dict['sub_obj_name_list'] = sub_obj_name_list
        return_dict['sub_obj_str_list'] = sub_obj_str_list
        return_dict['sub_obj_text_embedding'] = sub_obj_text_embedding
        return_dict['sub_obj_rel_name_list'] = sub_obj_rel_name_list
        return_dict['sub_obj_rel_str_list'] = sub_obj_rel_str_list
        return_dict['sub_obj_rel_text_embedding'] = sub_obj_rel_text_embedding
        return return_dict

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        losses, mask_features = self.forward_openseed(
            img, img_metas, mode='train')

        object_embedding_list = []
        sub_obj_pair_embedding_list = []
        sub_obj_pair_name_list = []
        sub_obj_pair_str_list = []
        sub_obj_pair_text_embedding_list = []
        sub_obj_rel_triplet_name_list = []
        sub_obj_rel_triplet_str_list = []
        sub_obj_rel_triplet_text_embedding_list = []
        relation_target_list = []

        num_imgs = len(img_metas)
        for idx in range(num_imgs):
            input_and_target_dict = self._get_input_and_target(
                mask_features[idx],
                img_metas[idx],
                gt_masks[idx],
                gt_labels[idx],
                gt_semantic_seg[idx],
            )
            object_embedding = input_and_target_dict['object_embedding']
            sub_obj_embedding = input_and_target_dict['sub_obj_embedding']
            sub_obj_name_list = input_and_target_dict['sub_obj_name_list']
            sub_obj_str_list = input_and_target_dict['sub_obj_str_list']
            sub_obj_text_embedding = input_and_target_dict['sub_obj_text_embedding']
            sub_obj_rel_name_list = input_and_target_dict['sub_obj_rel_name_list']
            sub_obj_rel_str_list = input_and_target_dict['sub_obj_rel_str_list']
            sub_obj_rel_text_embedding = input_and_target_dict['sub_obj_rel_text_embedding']
            relation_target = input_and_target_dict['relation_target']
            object_embedding_list.append(object_embedding)
            relation_target_list.append(relation_target)
            sub_obj_pair_embedding_list.append(sub_obj_embedding)
            sub_obj_pair_text_embedding_list.append(sub_obj_text_embedding)
            sub_obj_pair_name_list.append(sub_obj_name_list)
            sub_obj_pair_str_list.append(sub_obj_str_list)
            sub_obj_rel_triplet_name_list.append(sub_obj_rel_name_list)
            sub_obj_rel_triplet_str_list.append(sub_obj_rel_str_list)
            sub_obj_rel_triplet_text_embedding_list.append(
                sub_obj_rel_text_embedding)

        max_length = max([e.shape[1]
                          for e in object_embedding_list])
        mask_attention = mask_features.new_zeros([num_imgs, max_length])
        sub_obj_pair_mask_attention = mask_features.new_zeros(
            [num_imgs, max_length * max_length])
        for idx in range(num_imgs):
            mask_attention[idx, :object_embedding_list[idx].shape[1]] = 1.
            sub_obj_pair_mask_attention[idx,
                                        :sub_obj_pair_embedding_list[idx].shape[1]] = 1.
        object_embedding_list = [
            F.pad(e, [0, 0, 0, max_length-e.shape[1]])
            for e in object_embedding_list
        ]
        sub_obj_pair_embedding_list = [
            F.pad(e, [0, 0, 0, max_length*max_length-e.shape[1]])
            for e in sub_obj_pair_embedding_list
        ]
        sub_obj_pair_name_list = [
            e + [''] * (max_length*max_length - len(e))
            for e in sub_obj_pair_name_list
        ]
        sub_obj_pair_str_list = [
            e + [''] * (max_length*max_length - len(e))
            for e in sub_obj_pair_str_list
        ]
        sub_obj_pair_text_embedding_list = [
            F.pad(e, [0, 0, 0, max_length*max_length-e.shape[1]])
            for e in sub_obj_pair_text_embedding_list
        ]
        sub_obj_rel_triplet_name_list = [
            e + [''] * (max_length*max_length *
                        self.relation_head.num_relation_classes - len(e))
            for e in sub_obj_rel_triplet_name_list
        ]
        sub_obj_rel_triplet_str_list = [
            e + [''] * (max_length*max_length *
                        self.relation_head.num_relation_classes - len(e))
            for e in sub_obj_rel_triplet_str_list
        ]
        sub_obj_rel_triplet_text_embedding_list = [
            F.pad(e, [0, 0, 0, max_length*max_length *
                  self.relation_head.num_relation_classes-e.shape[1]])
            for e in sub_obj_rel_triplet_text_embedding_list
        ]
        relation_target_list = [
            F.pad(t, [0, max_length-t.shape[3], 0, max_length-t.shape[2]])
            for t in relation_target_list
        ]
        object_embedding = torch.cat(object_embedding_list, dim=0)
        sub_obj_pair_embedding = torch.cat(
            sub_obj_pair_embedding_list, dim=0)
        sub_obj_pair_text_embedding = torch.cat(
            sub_obj_pair_text_embedding_list, dim=0)
        sub_obj_rel_triplet_text_embedding = torch.cat(
            sub_obj_rel_triplet_text_embedding_list, dim=0)
        relation_target = torch.cat(relation_target_list, dim=0)

        relation_head_input = dict()
        # vision feature
        relation_head_input['object_embedding'] = object_embedding
        relation_head_input['sub_obj_pair_embedding'] = sub_obj_pair_embedding
        # pair text feature
        relation_head_input['sub_obj_pair_name_list'] = sub_obj_pair_name_list
        relation_head_input['sub_obj_pair_str_list'] = sub_obj_pair_str_list
        relation_head_input['sub_obj_pair_text_embedding'] = sub_obj_pair_text_embedding
        # triplet text feature
        relation_head_input['sub_obj_rel_triplet_name_list'] = sub_obj_rel_triplet_name_list
        relation_head_input['sub_obj_rel_triplet_str_list'] = sub_obj_rel_triplet_str_list
        relation_head_input['sub_obj_rel_triplet_text_embedding'] = sub_obj_rel_triplet_text_embedding
        # mask
        relation_head_input['mask_attention'] = mask_attention
        relation_head_input['sub_obj_pair_mask_attention'] = sub_obj_pair_mask_attention
        # forward relation head
        relation_head_output = self.relation_head(relation_head_input)
        object_vision_only_pred = relation_head_output['object_vision_only_pred']
        pair_vision_only_pred = relation_head_output['pair_vision_only_pred']
        pair_text_vision_cross_pred = relation_head_output['pair_text_vision_cross_pred']
        pair_vision_text_cross_pred = relation_head_output['pair_vision_text_cross_pred']
        triplet_vision_text_cross_pred = relation_head_output['triplet_vision_text_cross_pred']
        moe_pred = relation_head_output['moe_pred']
        # calculate loss
        if object_vision_only_pred is not None:
            object_vision_only_loss = self.relation_head.loss(
                object_vision_only_pred, relation_target, mask_attention, 'object_vision_only')
            losses.update(object_vision_only_loss)
        if pair_vision_only_pred is not None:
            pair_vision_only_loss = self.relation_head.loss(
                pair_vision_only_pred, relation_target, mask_attention, 'pair_vision_only')
            losses.update(pair_vision_only_loss)
        if pair_text_vision_cross_pred is not None:
            pair_text_vision_cross_loss = self.relation_head.loss(
                pair_text_vision_cross_pred, relation_target, sub_obj_pair_mask_attention, 'pair_text_vision_cross')
            losses.update(pair_text_vision_cross_loss)
        if pair_vision_text_cross_pred is not None:
            pair_vision_text_cross_loss = self.relation_head.loss(
                pair_vision_text_cross_pred, relation_target, sub_obj_pair_mask_attention, 'pair_vision_text_cross')
            losses.update(pair_vision_text_cross_loss)
        if triplet_vision_text_cross_pred is not None:
            triplet_vision_text_cross_loss = self.relation_head.loss(
                triplet_vision_text_cross_pred, relation_target, sub_obj_pair_mask_attention, 'triplet_vision_text_cross')
            losses.update(triplet_vision_text_cross_loss)
        if moe_pred is not None:
            moe_loss = self.relation_head.loss(
                moe_pred, relation_target, sub_obj_pair_mask_attention, 'moe')
            losses.update(moe_loss)
        torch.cuda.empty_cache()
        return losses

    def simple_test(self, imgs, img_metas, **kwargs):
        results, mask_features = self.forward_openseed(
            imgs, img_metas, mode='test')

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

            input_dict = self._get_input(
                pan_result=pan_results,
                object_id_list=object_id_list,
                object_score_list=object_score_list,
                feature_map=mask_features,
                meta_info=img_metas[batch_idx]
            )

            relation_res = []
            relation_scores = []
            if input_dict is not None:
                object_embedding = input_dict['object_embedding']
                object_id_list = input_dict['object_id_list']
                object_score_list = input_dict['object_score_list']
                sub_obj_embedding = input_dict['sub_obj_embedding']
                sub_obj_name_list = input_dict['sub_obj_name_list']
                sub_obj_str_list = input_dict['sub_obj_str_list']
                sub_obj_text_embedding = input_dict['sub_obj_text_embedding']
                sub_obj_rel_name_list = input_dict['sub_obj_rel_name_list']
                sub_obj_rel_str_list = input_dict['sub_obj_rel_str_list']
                sub_obj_rel_text_embedding = input_dict['sub_obj_rel_text_embedding']
                relation_head_input = dict()
                relation_head_input['object_embedding'] = object_embedding
                relation_head_input['sub_obj_pair_embedding'] = sub_obj_embedding
                relation_head_input['sub_obj_pair_name_list'] = sub_obj_name_list
                relation_head_input['sub_obj_pair_str_list'] = sub_obj_str_list
                relation_head_input['sub_obj_pair_text_embedding'] = sub_obj_text_embedding
                relation_head_input['sub_obj_rel_triplet_name_list'] = sub_obj_rel_name_list
                relation_head_input['sub_obj_rel_triplet_str_list'] = sub_obj_rel_str_list
                relation_head_input['sub_obj_rel_triplet_text_embedding'] = sub_obj_rel_text_embedding
                relation_head_input['mask_attention'] = torch.ones(
                    [1, object_embedding.shape[1]], device=device, dtype=dtype)
                relation_head_input['sub_obj_pair_mask_attention'] = torch.ones(
                    [1, sub_obj_embedding.shape[1]], device=device, dtype=dtype)
                # forward relation head
                relation_head_output = self.relation_head(relation_head_input)
                object_vision_only_pred = relation_head_output['object_vision_only_pred']
                pair_vision_only_pred = relation_head_output['pair_vision_only_pred']
                pair_text_vision_cross_pred = relation_head_output['pair_text_vision_cross_pred']
                pair_vision_text_cross_pred = relation_head_output['pair_vision_text_cross_pred']
                triplet_vision_text_cross_pred = relation_head_output['triplet_vision_text_cross_pred']
                moe_pred = relation_head_output['moe_pred']
                if object_vision_only_pred is not None:
                    relation_output = object_vision_only_pred[0]
                if pair_vision_only_pred is not None:
                    relation_output = pair_vision_only_pred[0]
                if pair_text_vision_cross_pred is not None:
                    relation_output = pair_text_vision_cross_pred[0]
                if pair_vision_text_cross_pred is not None:
                    relation_output = pair_vision_text_cross_pred[0]
                if triplet_vision_text_cross_pred is not None:
                    relation_output = triplet_vision_text_cross_pred[0]
                if moe_pred is not None:
                    relation_output = moe_pred[0]

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
                        # pred_relation = index // (
                        #     relation_output.shape[1] ** 2)
                        pred_relation = torch.div(
                            index, (relation_output.shape[1] ** 2), rounding_mode='floor')
                        index_subject_object = index % (
                            relation_output.shape[1] ** 2)
                        # pred_subject = index_subject_object // relation_output.shape[1]
                        pred_subject = torch.div(
                            index_subject_object, relation_output.shape[1], rounding_mode='floor')
                        pred_object = index_subject_object % relation_output.shape[1]
                        pred = [pred_subject.item(),
                                pred_object.item(),
                                pred_relation.item()]
                        relation_score = relation_output.reshape(
                            [-1, ])[index].item()
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
