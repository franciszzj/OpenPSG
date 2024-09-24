import torch
import torch.nn.functional as F

from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
from mmdet.models.builder import HEADS
from mmdet.models.seg_heads import MaskFormerFusionHead


@HEADS.register_module()
class MaskFormerFusionRelationHead(MaskFormerFusionHead):
    def panoptic_relation_postprocess(self, mask_cls, mask_pred):
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
        iou_thr = self.test_cfg.get('iou_thr', 0.8)
        filter_low_score = self.test_cfg.get('filter_low_score', False)

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > object_mask_thr)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)
        object_id_list = []
        object_score_list = []
        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            mode = 'raw'
            if mode == 'area':
                areas = (cur_masks >= 0.5).sum(dim=[1, 2])
                area_tensor, idx_tensor = torch.sort(areas)
                instance_id = 1
                for area, idx_cur in zip(area_tensor.flip(dims=[0]), idx_tensor.flip(dims=[0])):
                    if area <= 0:
                        continue
                    pred_class = cur_classes[idx_cur].to(torch.long)
                    if pred_class == 133:
                        continue
                    pred_mask = cur_masks[idx_cur] >= 0.5
                    pred_score = cur_prob_masks[idx_cur][pred_mask].mean()

                    isthing = pred_class < self.num_things_classes
                    if not isthing:
                        panoptic_seg[pred_mask] = panoptic_seg[pred_mask] * \
                            0 + pred_class
                        oid = pred_class
                    else:
                        panoptic_seg[pred_mask] = panoptic_seg[pred_mask] * \
                            0 + (pred_class + instance_id * INSTANCE_OFFSET)
                        oid = pred_class + instance_id * INSTANCE_OFFSET
                        instance_id += 1

                    object_id_list.append(oid)
                    object_score_list.append(pred_score)
            elif mode == 'raw':
                # cur_mask_ids = cur_prob_masks.argmax(0)
                cur_mask_score, cur_mask_ids = cur_prob_masks.max(dim=0)
                instance_id = 1
                for k in range(cur_classes.shape[0]):
                    # pred_class = int(cur_classes[k].item())
                    pred_class = cur_classes[k].to(torch.long)
                    isthing = pred_class < self.num_things_classes
                    mask = cur_mask_ids == k
                    mask_area = mask.sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()

                    score = cur_mask_score[mask].mean()

                    if filter_low_score:
                        mask = mask & (cur_masks[k] >= 0.5)
                        # mask_area = mask.sum().item()

                    if mask_area > 0 and original_area > 0:
                        if mask_area / original_area < iou_thr:
                            continue

                        if not isthing:
                            # different stuff regions of same class will be
                            # merged here, and stuff share the instance_id 0.
                            panoptic_seg[mask] = panoptic_seg[mask] * \
                                0 + pred_class
                            # object_id_list.append(pred_class)
                            # object_score_list.append(score)
                        else:
                            panoptic_seg[mask] = panoptic_seg[mask] * 0 + \
                                (pred_class + instance_id * INSTANCE_OFFSET)
                            # object_id_list.append(pred_class + instance_id * INSTANCE_OFFSET)
                            # object_score_list.append(score)
                            instance_id += 1

                object_id_list = []
                object_score_list = []
                for oid in torch.unique(panoptic_seg):
                    if oid == 133:
                        continue
                    mask = panoptic_seg == oid
                    score = cur_mask_score[mask].mean()
                    object_id_list.append(oid)
                    object_score_list.append(score)

        # return panoptic_seg
        return panoptic_seg, object_id_list, object_score_list

    def simple_test(self,
                    mask_cls_results,
                    mask_pred_results,
                    img_metas,
                    rescale=False,
                    **kwargs):
        panoptic_on = self.test_cfg.get('panoptic_on', True)
        semantic_on = self.test_cfg.get('semantic_on', False)
        instance_on = self.test_cfg.get('instance_on', False)
        assert not semantic_on, 'segmantic segmentation '\
            'results are not supported yet.'

        results = []
        for mask_cls_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_pred_results, img_metas):
            # remove padding
            img_height, img_width = meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]

            if rescale:
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]

            result = dict()
            if panoptic_on:
                pan_results, object_id_list, object_score_list = self.panoptic_relation_postprocess(
                    mask_cls_result, mask_pred_result)
                result['pan_results'] = pan_results
                result['object_id_list'] = object_id_list
                result['object_score_list'] = object_score_list

            if instance_on:
                ins_results = self.instance_postprocess(
                    mask_cls_result, mask_pred_result)
                result['ins_results'] = ins_results

            if semantic_on:
                sem_results = self.semantic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['sem_results'] = sem_results

            results.append(result)

        return results
