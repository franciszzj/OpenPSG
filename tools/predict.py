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


def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def get_model(config, checkpoint, test_pipeline_img_scale):
    config = mmcv.Config.fromfile(config)

    test_pipeline = copy.deepcopy(config['data']['test']['pipeline'])
    test_pipeline[1]['img_scale'] = test_pipeline_img_scale
    config['data']['test']['pipeline'] = test_pipeline

    config['model']['test_cfg']['predict_relation'] = True

    model = init_detector(config, checkpoint)
    return model


def inference(test_pipeline_img_scale, config, checkpoint, test_file, data_dir, output_dir):

    panseg_output_dir = os.path.join(output_dir, 'panseg')
    json_output_dir = output_dir

    os.makedirs(panseg_output_dir, exist_ok=True)

    test_data = load_json(test_file)
    model = get_model(config, checkpoint, test_pipeline_img_scale)

    print('Inference begin.')
    test_idx = -1
    prog_bar = mmcv.ProgressBar(len(test_data['data']))
    all_result_dict = []
    for d in test_data['data']:
        test_idx += 1
        prog_bar.update()

        img_file = os.path.join(data_dir, d['file_name'])
        img = cv2.imread(img_file)
        results = inference_detector(model, img_file)

        pan_results = results['pan_results']
        rel_results = results['rel_results']
        object_id_list = rel_results['object_id_list']
        relation = rel_results['relation']
        relation_scores = results['rel_scores']

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
        panseg_name = d['file_name'].split('/')[-1].split('.')[0]
        cv2.imwrite(f'{panseg_output_dir}/{panseg_name}.png', panseg_output)

        if len(relation) == 0:
            relation = [[0, 0, 0]]
        if len(segments_info) == 0:
            r, g, b = random.choices(range(0, 255), k=3)
            segments_info = [dict(category_id=1, id=rgb2id((r, g, b)))]

        single_result_dict = dict()
        single_result_dict.update(d)
        single_result_dict['relations'] = [[s, o, r + 1]
                                           for s, o, r in relation]
        single_result_dict['segments_info'] = segments_info
        single_result_dict['relation_scores'] = relation_scores
        all_result_dict.append(single_result_dict)

    print('Inference finish.')
    with open(f'{json_output_dir}/relation.json', 'w') as outfile:
        json.dump(all_result_dict, outfile, default=str)
    print('Dump results to {}'.format(json_output_dir))


if __name__ == '__main__':
    exp_tag = 'v99_0'
    epoch = 12
    data_dir = 'path/to/data'
    test_file = 'path/to/test.json'
    root_path = './work_dirs/kings_sgg_{}'.format(exp_tag)
    output_dir = 'path/to/output/{}_output_results'.format(
        exp_tag)
    config_file = '{}/{}.py'.format(root_path, exp_tag)
    checkpoint_file = '{}/epoch_{}.pth'.format(root_path, epoch)

    inference(
        test_pipeline_img_scale=(1333, 1333),
        config=config_file,
        checkpoint=checkpoint_file,
        test_file=test_file,
        data_dir=data_dir,
        output_dir=output_dir)
