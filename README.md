# OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models

Official code implementation of OpenPSG, [arXiv](https://arxiv.org/abs/2311.16492).

## Abstract
Panoptic Scene Graph Generation (PSG) aims to segment objects and recognize their relations, enabling the structured understanding of an image. Previous methods focus on predicting predefined object and relation categories, hence limiting their applications in the open world scenarios. With the rapid development of large multimodal models (LMMs), significant progress has been made in open-set object detection and segmentation, yet open-set relation prediction in PSG remains unexplored. In this paper, we focus on the task of open-set relation prediction integrated with a pretrained open-set panoptic segmentation model to achieve true open-set panoptic scene graph generation (OpenPSG). Our OpenPSG leverages LMMs to achieve open-set relation prediction in an autoregressive manner. We introduce a relation query transformer to efficiently extract visual features of object pairs and estimate the existence of relations between them. The latter can enhance the prediction efficiency by filtering irrelevant pairs. Finally, we design the generation and judgement instructions to perform open-set relation prediction in PSG autoregressively. To our knowledge, we are the first to propose the open-set PSG task. Extensive experiments demonstrate that our method achieves state-of-the-art performance in open-set relation prediction and panoptic scene graph generation.

## How to train
Please use config: `configs/psg/baseline_v4_ov.py`.
```
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_port=27500 \
  tools/train.py \
  $CONFIG \
  --auto-resume \
  --no-validate \
  --launcher pytorch
```

## How to test
Execute the following command, and you will get a submission file that can be used to evaluate the model.
```
PYTHONPATH=".":$PYTHONPATH \
python tools/infer.py \
  $EXP_TAG \
  $EPOCH_NUM
```

## How to evaluate
Please install [HiLo](https://github.com/franciszzj/HiLo).
```
cd ${HiLo_ROOT}
# SUBMISSION_PATH looks like "work_dirs/kings_sgg_v1_1/epoch_12_results/submission/"
python tools/grade.py $SUBMISSION_PATH
```

## Resource
Json used to train and test our method: [psg_train.json](https://emckclac-my.sharepoint.com/:u:/g/personal/k21163430_kcl_ac_uk/EUDvXDxSEexJnkBfy_1yr34BvJfimWQTUfOKEMTPwxyF0w?e=7tbF1R) and [psg_val.json](https://emckclac-my.sharepoint.com/:u:/g/personal/k21163430_kcl_ac_uk/Ecau5X4R8ylHsGc543BuqJsBggqhN8l3pLXT3-5TlVvzDg?e=5t1xVW).

## Citation
```
@article{zhou2024openpsg,
  title={OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models},
  author={Zhou, Zijian and Zhu, Zheng and Caesar, Holger and Shi, Miaojing},
  journal={arXiv preprint arXiv:2407.11213},
  year={2024}
}
```
