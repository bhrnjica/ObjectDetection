# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
import os, sys

import numpy as np
import utils.od_utils as od
from utils.config_helpers import merge_configs

def get_configuration():
    # load configs for detector, base network and data set
    from FasterRCNN.FasterRCNN_config import cfg as detector_cfg

    # for VGG16 base model use:         from utils.configs.VGG16_config import cfg as network_cfg
    # for AlexNet base model use:       from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    # for Pascal VOC 2007 data set use: from utils.configs.Pascal_config import cfg as dataset_cfg
    # for the Grocery data set use:     from utils.configs.Grocery_config import cfg as dataset_cfg
    # from utils.configs.Grocery_config import cfg as dataset_cfg
    ###################################################################
    # Custom Dataset NO3310
    from utils.configs.NO3310_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg, {'DETECTOR': 'FasterRCNN'}])

if __name__ == '__main__':
    # Get Configuration
    cfg = get_configuration()

    # train and test
    eval_model = od.train_object_detector(cfg)
    # eval_results = od.evaluate_test_set(eval_model, cfg)

   
    # detect objects in single image
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"DataSets/NO3310/testImages/img30.jpg")
    regressed_rois, cls_probs = od.evaluate_single_image(eval_model, img_path, cfg)
    bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs, cfg)

    # write detection results to output
    fg_boxes = np.where(labels > 0)
    print("#bboxes: before nms: {}, after nms: {}, foreground: {}".format(len(regressed_rois), len(bboxes), len(fg_boxes[0])))
    for i in fg_boxes[0]: print("{:<12} (label: {:<2}), score: {:.3f}, box: {}".format(
                                cfg["DATA"].CLASSES[labels[i]], labels[i], scores[i], [int(v) for v in bboxes[i]]))
    # visualize detections on image
    od.visualize_results(img_path, bboxes, labels, scores, cfg)
    # measure inference time
   # od.measure_inference_time(eval_model, img_path, cfg, num_repetitions=100)
