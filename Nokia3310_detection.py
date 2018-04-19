# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
import os
import numpy as np
import cntk
from FasterRCNN.FasterRCNN_train import prepare, train_faster_rcnn, store_eval_model_with_native_udf
from FasterRCNN.FasterRCNN_eval import compute_test_set_aps, FasterRCNN_Evaluator
import utils.od_utils as od
from utils.config_helpers import merge_configs
from utils.plot_helpers import plot_test_set_results

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

    # train 
    trained_model = od.train_object_detector(cfg)
    # test model with some images
    eval_results = od.evaluate_test_set(trained_model, cfg)

    # Plot results on test set images
    if cfg.VISUALIZE_RESULTS:
        num_eval = min(cfg["DATA"].NUM_TEST_IMAGES, 100)
        results_folder = os.path.join(cfg.OUTPUT_PATH, cfg["DATA"].DATASET)
        evaluator = FasterRCNN_Evaluator(trained_model, cfg)
        plot_test_set_results(evaluator, num_eval, results_folder, cfg)

    if cfg.STORE_EVAL_MODEL_WITH_NATIVE_UDF:
        store_eval_model_with_native_udf(trained_model, cfg)

    # detect objects in single image
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"DataSets/NO3310/testImages/img30.jpg")
    regressed_rois, cls_probs = od.evaluate_single_image(trained_model, img_path, cfg)
    bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs, cfg)

    # write detection results to output
    fg_boxes = np.where(labels > 0)
    print("#bboxes: before nms: {}, after nms: {}, foreground: {}".format(len(regressed_rois), len(bboxes), len(fg_boxes[0])))
    for i in fg_boxes[0]: print("{:<12} (label: {:<2}), score: {:.3f}, box: {}".format(
                                cfg["DATA"].CLASSES[labels[i]], labels[i], scores[i], [int(v) for v in bboxes[i]]))
    # visualize detections on image
    od.visualize_results(img_path, bboxes, labels, scores, cfg)
    # measure inference time
   # od.measure_inference_time(trained_model, img_path, cfg, num_repetitions=100)
