import os
import sys
import cv2
import json
import numpy as np

thing_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
stuff_classes = ['banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
                 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
object_classes = thing_classes + stuff_classes
relation_classes = ['over', 'in front of', 'beside', 'on', 'in', 'attached to', 'hanging from', 'on back of', 'falling off', 'going down', 'painted on', 'walking on', 'running on', 'crossing', 'standing on', 'lying on', 'sitting on', 'flying over', 'jumping over', 'jumping from', 'wearing', 'holding', 'carrying', 'looking at', 'guiding', 'kissing',
                    'eating', 'drinking', 'feeding', 'biting', 'catching', 'picking', 'playing with', 'chasing', 'climbing', 'cleaning', 'playing', 'touching', 'pushing', 'pulling', 'opening', 'cooking', 'talking to', 'throwing', 'slicing', 'driving', 'riding', 'parked on', 'driving on', 'about to hit', 'kicking', 'swinging', 'entering', 'exiting', 'enclosing', 'leaning on']


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


def parse_single(data, panseg_dir):
    file_name = data['file_name']
    relations = data['relations']
    segments_info = data['segments_info']
    relation_scores = data['relation_scores']
    panseg_path = os.path.join(
        panseg_dir, file_name.split('/')[-1].split('.')[0] + '.png')
    panseg = cv2.imread(panseg_path)
    panseg = rgb2id(panseg)
    object_list = []
    for object in segments_info:
        object_name = object_classes[object['category_id'] - 1]
        object_mask = panseg == object['id']
        object_list.append((object_name, object_mask))

    triplet_list = []
    for relation, relation_score in zip(relations, relation_scores):
        subject_name = object_list[relation[0] - 1][0]
        object_name = object_list[relation[1] - 1][0]
        relation_name = relation_classes[relation[2] - 1]
        subject_mask = object_list[relation[0] - 1][1]
        object_mask = object_list[relation[1] - 1][1]
        triplet_list.append((subject_name, relation_name,
                            object_name, subject_mask, object_mask, relation_score))

    return triplet_list


def parse(in_file, panseg_dir):
    with open(in_file, 'r') as f:
        data_list = json.load(f)

    for data in data_list:
        triplet_list = parse_single(data, panseg_dir)
        yield triplet_list


if __name__ == '__main__':
    # Usage: python parse_predict.py in_file panseg_dir
    # Example: python parse_predict.py ./v99_0_output_results/relation.json ./v99_0_output_results/panseg/
    in_file = sys.argv[1]
    panseg_dir = sys.argv[2]
    for triplet_list in parse(in_file, panseg_dir):
        # Example:
        # triplet_list: (subject_name, relation_name, object_name, subject_mask, object_mask, relation_score)
        #     subject_name: str
        #     relation_name: str
        #     object_name: str
        #     subject_mask: np.ndarray (h, w)
        #     object_mask: np.ndarray (h, w)
        #     relation_score: float
        input("Press Enter to continue...")
        print(triplet_list)
