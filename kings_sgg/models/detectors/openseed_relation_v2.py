import os
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import INSTANCE_OFFSET
from mmdet.models.builder import DETECTORS, build_head
from mmdet.models.detectors.base import BaseDetector

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from openseed.BaseModel import BaseModel
from openseed import build_model


@DETECTORS.register_module()
class OpenSeeDRelationV2(BaseDetector):
    def __init__(self,
                 # openseed
                 openseed_config_path='',
                 openseed_pretrained_path='',
                 thing_classes=[],
                 stuff_classes=[],
                 # relation
                 relation_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(OpenSeeDRelationV2, self).__init__(init_cfg)
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
        self.relation_head = build_head(relation_head)

        if train_cfg is not None:
            freeze_layers = train_cfg.pop('freeze_layers', [])
        else:
            freeze_layers = []
        for layer_name in freeze_layers:
            for n, p in self.named_parameters():
                if n.startswith(layer_name):
                    p.requires_grad = False
        self.freeze_layers = freeze_layers
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

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

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        # forward segmenter (openseed)
        losses, mask_features = self.forward_openseed(
            img, img_metas, mode='train')

        # forward relation head
        relation_head_input = dict()
        relation_head_input['mask_features'] = mask_features
        relation_head_input['img_metas'] = img_metas
        relation_head_input['gt_labels'] = gt_labels
        relation_head_input['gt_masks'] = gt_masks
        relation_head_input['gt_semantic_seg'] = gt_semantic_seg
        relation_head_output = self.relation_head(relation_head_input)
        losses.update(relation_head_output)
        torch.cuda.empty_cache()
        return losses

    @torch.no_grad()
    def simple_test(self, imgs, img_metas, **kwargs):
        # forward segmenter (openseed)
        results, mask_features = self.forward_openseed(
            imgs, img_metas, mode='test')

        # forward relation head
        relation_head_input = dict()
        relation_head_input['mask_features'] = mask_features
        relation_head_input['img_metas'] = img_metas
        relation_head_input['object_info'] = results
        relation_head_output = self.relation_head(relation_head_input)

        res = results[0]
        res['pan_results'] = results[0]['pan_results'].detach().cpu().numpy()
        res['rel_results'] = dict(
            object_id_list=[oid.item() for oid in res['object_id_list']],
            relation=relation_head_output['rel_pred'],)
        res['rel_scores'] = relation_head_output['rel_score']

        return [res]
