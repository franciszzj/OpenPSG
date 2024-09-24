from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.loading import LoadPanopticAnnotations


@PIPELINES.register_module()
class LoadPanopticRelationAnnotations(LoadPanopticAnnotations):
    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=True,
                 with_seg=True,
                 with_rel=True,
                 file_client_args=dict(backend='disk')):
        super(LoadPanopticRelationAnnotations, self).__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            file_client_args=file_client_args)

        self.with_rel = with_rel

    def _load_relations(self, results):
        gt_rels = results['ann_info']['relations']
        masks_info = results['ann_info']['masks']
        results['gt_rels'] = gt_rels
        results['masks_info'] = masks_info
        return results

    def __call__(self, results):
        results = super(LoadPanopticRelationAnnotations,
                        self).__call__(results)
        if self.with_rel:
            results = self._load_relations(results)

        return results
