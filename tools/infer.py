import os
import sys
import cv2
import copy
import json
import random
import numpy as np
import torch
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.core import INSTANCE_OFFSET
from panopticapi.utils import rgb2id, id2rgb
# for visualization
import seaborn as sns
from skimage.segmentation import find_boundaries
from prettytable import PrettyTable


def replace_name(text):
    # for visualization
    if '-stuff' in text:
        text = text.replace('-stuff', '')
    if '-merged' in text:
        text = text.replace('-merged', '')
    if '-other' in text:
        text = text.replace('-other', '')
    return text


def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def get_model(config, checkpoint, test_pipeline_img_scale):
    config = mmcv.Config.fromfile(config)

    test_pipeline = copy.deepcopy(config['data']['test']['pipeline'])
    test_pipeline[1]['img_scale'] = test_pipeline_img_scale
    config['data']['test']['pipeline'] = test_pipeline

    if config['model']['test_cfg'] is None:
        config['model']['test_cfg'] = dict()
    config['model']['test_cfg']['predict_relation'] = True
    if 'backbone' not in config['model']:
        config['model']['backbone'] = dict()

    model = init_detector(config, checkpoint)
    return model


def get_test_id(test_file):
    dataset = load_json(test_file)
    test_id_list = [
        d['image_id'] for d in dataset['data'] if (d['image_id'] in dataset['test_image_ids']) and (len(d['relations']) != 0)
    ]
    thing_classes = dataset['thing_classes']
    stuff_classes = dataset['stuff_classes']
    object_classes = thing_classes + stuff_classes
    predicate_classes = dataset['predicate_classes']
    return test_id_list, object_classes, predicate_classes


def inference(test_pipeline_img_scale, config, checkpoint, test_file, data_dir, output_dir, use_vis=False):

    panseg_output_dir = os.path.join(output_dir, 'submission/panseg')
    json_output_dir = os.path.join(output_dir, 'submission')

    os.makedirs(panseg_output_dir, exist_ok=True)

    test_data = load_json(test_file)
    test_id_list, object_classes, predicate_classes = get_test_id(test_file)
    # for visualization, palette
    palette = sns.color_palette('pastel', len(object_classes))
    model = get_model(config, checkpoint, test_pipeline_img_scale)

    print('Inference begin.')
    test_idx = -1
    prog_bar = mmcv.ProgressBar(len(test_id_list))
    all_result_dict = []
    for d in test_data['data']:
        image_id = d['image_id']
        if image_id not in test_id_list:
            continue
        test_idx += 1
        prog_bar.update()

        img_file = os.path.join(data_dir, d['file_name'])
        img = cv2.imread(img_file)
        results = inference_detector(model, img_file)

        if use_vis:
            vis(palette, object_classes, predicate_classes,
                img_file, results, output_path='./vis_test')

        pan_results = results['pan_results']
        rel_results = results['rel_results']
        object_id_list = rel_results['object_id_list']
        relation = rel_results['relation']

        '''
        # Use ground truth relation, to find the upper bound of the performance.
        try:
            # GT
            pan_seg = cv2.imread(os.path.join(
                './data/coco', d['pan_seg_file_name']))
            pan_seg = cv2.cvtColor(pan_seg, cv2.COLOR_BGR2RGB)
            pan_seg_id = rgb2id(pan_seg)
            gt_cls_list = []
            gt_mask_list = []
            if len(d['segments_info']) > 0:
                for seg_info in d['segments_info']:
                    cls_id = seg_info['category_id']
                    gt_cls_list.append(cls_id)
                    mask = (pan_seg_id == seg_info['id']).astype(np.float32)
                    gt_mask_list.append(mask)
                gt_masks = torch.from_numpy(
                    np.stack(gt_mask_list, axis=0)).flatten(1)
            gt_rel_list = d['relations']
            # Pred
            pred_cls_list = []
            pred_mask_list = []
            if len(object_id_list) > 0:
                for object_id in object_id_list:
                    cls_id = object_id % INSTANCE_OFFSET
                    pred_cls_list.append(cls_id)
                    mask = (pan_results == object_id).astype(np.float32)
                    pred_mask_list.append(mask)
                pred_masks = torch.from_numpy(
                    np.stack(pred_mask_list, axis=0)).flatten(1)
            pred_rel_list = copy.deepcopy(relation)

            masks_all = torch.cat([gt_masks, pred_masks], dim=0)
            ious = masks_all.mm(masks_all.transpose(0, 1)) / \
                ((masks_all+masks_all) > 0).sum(-1)
            ious = ious[:len(gt_cls_list), len(gt_cls_list):]
            for (gt_s, gt_o, gt_r) in gt_rel_list:
                for idx, (pred_s, pred_o, pred_r) in enumerate(relation):
                    if ious[gt_s, pred_s] > 0.5 and ious[gt_o, pred_o] > 0.5:
                        if gt_r != pred_r:
                            pred_rel_list[idx][2] = gt_r

            relation = copy.deepcopy(pred_rel_list)
        except:
            pass
        # '''

        panseg_output = np.zeros_like(img)
        segments_info = []
        for object_id in object_id_list:
            # object_id == 133 background
            mask = pan_results == object_id
            if object_id == 133:
                continue
            r, g, b = random.choices(range(0, 255), k=3)

            mask = mask[..., None]
            mask = mask.astype(int)
            coloring_mask = np.concatenate([mask]*3, axis=-1)
            color = np.array([b, g, r]).reshape([1, 1, 3])
            coloring_mask = coloring_mask * color
            panseg_output = panseg_output + coloring_mask
            idx_class = object_id % INSTANCE_OFFSET + 1
            segment = dict(category_id=int(idx_class), id=rgb2id((r, g, b)))
            segments_info.append(segment)

        panseg_output = panseg_output.astype(np.uint8)
        cv2.imwrite(f'{panseg_output_dir}/{test_idx}.png', panseg_output)

        if len(relation) == 0:
            relation = [[0, 0, 0]]
        if len(segments_info) == 0:
            r, g, b = random.choices(range(0, 255), k=3)
            segments_info = [dict(category_id=1, id=rgb2id((r, g, b)))]

        single_result_dict = dict(
            # image_id=image_id,
            relations=[[s, o, r + 1] for s, o, r in relation],
            segments_info=segments_info,
            pan_seg_file_name='%d.png' % test_idx,
        )
        all_result_dict.append(single_result_dict)

    print('Inference finish.')
    with open(f'{json_output_dir}/relation.json', 'w') as outfile:
        json.dump(all_result_dict, outfile, default=str)
    print('Dump results to {}'.format(json_output_dir))


def vis(palette, object_classes, predicate_classes, img_file, results, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_name = img_file.split('/')[-1]
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    seg_id = results['pan_results']
    object_id_list = results['rel_results']['object_id_list']
    object_label_list = [x % INSTANCE_OFFSET for x in object_id_list]
    boundaries = find_boundaries(seg_id, mode='thick')
    relations = results['rel_results']['relation']

    position_name_color_list = []
    new_seg = copy.deepcopy(image)
    boundaries_for_object_list = []
    for idx, object_id in enumerate(object_id_list):
        object_label = object_label_list[idx]
        object_name = object_classes[object_label]
        object_name = replace_name(object_name)
        index = np.where(seg_id == object_id)
        mean_y, mean_x = int(index[0].mean()), int(index[1].mean())
        position = [mean_x, mean_y]
        # whole image
        color = [int(x * 255)
                 for x in palette[object_label]]
        new_seg[index] = color
        # for each object
        mask_for_object = np.zeros_like(seg_id)
        mask_for_object[index] = 1
        boundaries_for_object = find_boundaries(mask_for_object, mode='inner')
        boundaries_for_object_list.append(boundaries_for_object)
        position_name_color_list.append([position, object_name, color])

    new_image = image * 0.5 + new_seg * 0.5
    new_image = new_image.astype(np.uint8)
    new_image[boundaries] = [64, 64, 64]

    for idx, (position, name, color) in enumerate(position_name_color_list):
        name_size = cv2.getTextSize(
            f'{idx}_{name}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        new_image = cv2.circle(
            new_image, (position[0], position[1]), 3, color, -1, cv2.LINE_AA)
        new_image = cv2.rectangle(new_image, (position[0], position[1]), (
            position[0] + name_size[0][0], position[1] + name_size[0][1] + 2), [255-color[0], 255-color[1], 255-color[2]], -1)
        new_image = cv2.putText(new_image, f'{idx}_{name}', (
            position[0], position[1] + name_size[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('{}/{}_pan_seg.jpg'.format(output_path,
                file_name.split('.')[0]), new_image)

    table = PrettyTable(['subject', 'relation', 'object'])
    for sub_id, obj_id, rel_label in relations[:20]:
        sub_label = object_label_list[sub_id]
        obj_label = object_label_list[obj_id]
        sub_name = object_classes[sub_label]
        obj_name = object_classes[obj_label]
        rel_name = predicate_classes[rel_label]
        table.add_row(
            [f'{sub_id}_{sub_name}', rel_name, f'{obj_id}_{obj_name}'])
    print(img_file)
    print(table)


if __name__ == '__main__':
    exp_tag = sys.argv[1]
    epoch = sys.argv[2]
    try:
        use_vis = sys.argv[3]
        use_vis = True
    except:
        use_vis = False
    data_dir = './data/coco'
    test_file = './data/psg/psg.json'
    root_path = './work_dirs/ov_psg_{}'.format(exp_tag)
    output_dir = '{}/epoch_{}_results'.format(root_path, epoch)
    config_file = '{}/{}.py'.format(root_path, exp_tag)
    checkpoint_file = '{}/epoch_{}.pth'.format(root_path, epoch)

    inference(
        test_pipeline_img_scale=(1333, 1333),
        config=config_file,
        checkpoint=checkpoint_file,
        test_file=test_file,
        data_dir=data_dir,
        output_dir=output_dir,
        use_vis=use_vis)
