# Learning Granularity-Aware Affordances from Human-Tool Interaction for Tool-Based Functional Grasping in Dexterous Robotics（GAAF-DEX）

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2303.09665-b31b1b.svg)](https://arxiv.org/abs/2303.09665)
[![GitHub](https://img.shields.io/website?label=Project%20Page&up_message=page&url=https://reagan1311.github.io/locate/)](https://reagan1311.github.io/locate/)
[![ ](https://img.shields.io/youtube/views/RLHansdFxII?label=Video&style=flat)](https://www.youtube.com/watch?v=RLHansdFxII)  -->

We will release our code and dataset soon...

## Abstract

To enable robots to use tools, the initial step is teaching robots to employ dexterous gestures for touching specific areas precisely where tasks are performed. Affordance features of objects serve as a bridge in the functional interaction between agents and objects. However, leveraging these affordance cues to help robots achieve functional tool grasping remains
unresolved. To address this, we propose a granularity-aware affordance feature extraction method for locating functional affordance areas and predicting dexterous coarse gestures. We
study the intrinsic mechanisms of human tool use. On one hand, we use fine-grained affordance features of object-functional finger contact areas to locate functional affordance regions.
On the other hand, we use highly activated coarse-grained affordance features in hand-object interaction regions to predict grasp gestures. Additionally, we introduce a model-based post-
processing module that includes functional finger coordinate localization, finger-to-end coordinate transformation, and force feedback-based coarse-to-fine grasping. This forms a complete
dexterous robotic functional grasping framework GAAF-Dex, which Learning Granularity-Aware Affordances from Human-Tool Interaction for Tool-Based Functional Grasping in Dexterous Robotics. Unlike fully-supervised methods that require extensive data annotation, we employ a weakly supervised approach to extract relevant cues from exocentric (Exo) images of hand-object interactions to supervise feature extraction in egocentric (Ego) images. Correspondingly, we have constructed a small-scale dataset, FAH, which includes near 6𝐾 images of functional hand-object interaction Exo images and Ego images of 18 commonly used tools performing 6 tasks. Extensive experiments on the dataset demonstrate that our method outperforms state-of-the-art methods, and real-world localization and grasping experiments validate the practical applicability of our approach.

<p align="center">
 <img src="./assets/pipeline.png" align=center>
</p>

## Usage

### 1. Requirements

Code is tested under Pytorch 1.12.1, python 3.7, and CUDA 11.6

```
pip install -r requirements.txt
```

### 2. Dataset

Download the FAH dataset
.

### 3. Train and Test

Our pretrained model can be downloaded
  from .... Run following commands to start training or testing:

```
python train.py --data_root <PATH_TO_DATA>
python test.py --data_root <PATH_TO_DATA> --model_file <PATH_TO_MODEL>
```

## Citation

```

```

## Anckowledgement

This repo is based on [Cross-View-AG](https://github.com/lhc1224/Cross-View-AG)
, [LOCATE](https://github.com/Reagan1311/LOCATE) Thanks for their great work!
