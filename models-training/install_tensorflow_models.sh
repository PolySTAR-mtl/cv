#!/bin/bash

ROOT_DIR=`pwd`
MODELS_DIR=$ROOT_DIR/../models
PYTHON='poetry run python'
PIP='poetry run pip'

# make sure tensorflow has been installed
$PIP list | grep tensorflow
if [ $? -ne 0 ]; then
    echo "TensorFlow doesn't seem to be installed!"
    exit
fi

# Download tensorflow models, with the working commit
cd ..
git clone https://github.com/tensorflow/models
cd $MODELS_DIR || exit
git checkout 6518c1c

# Fix the code for python3
cd research || exit
sed -i "" -e "157s/print '--annotation_type expected value is 1 or 2.'/print('--annotation_type expected value is 1 or 2.')/" \
       object_detection/dataset_tools/oid_hierarchical_labels_expansion.py
sed -i "" -e "516s/print num_classes, num_anchors/print(num_classes, num_anchors)/" \
       object_detection/meta_architectures/ssd_meta_arch_test.py
sed -i "" -e "282s/losses_dict.itervalues()/losses_dict.values()/" \
       object_detection/model_lib.py
sed -i "" -e "381s/category_index.values(),/list(category_index.values()),/" \
       object_detection/model_lib.py
sed -i "" -e "391s/eval_metric_ops.iteritems()/eval_metric_ops.items()/" \
       object_detection/model_lib.py
sed -i "" -e "225s/reversed(zip(output_feature_map_keys, output_feature_maps_list)))/reversed(list(zip(output_feature_map_keys, output_feature_maps_list))))/" \
       object_detection/models/feature_map_generators.py
sed -i "" -e "842s/print 'Scores and tpfp per class label: {}'.format(class_index)/print('Scores and tpfp per class label: {}'.format(class_index))/" \
       object_detection/utils/object_detection_evaluation.py
sed -i "" -e "843s/print tp_fp_labels/print(tp_fp_labels)/" \
       object_detection/utils/object_detection_evaluation.py
sed -i "" -e "844s/print scores/print(scores)/" \
       object_detection/utils/object_detection_evaluation.py
sed -n '31p' object_detection/eval_util.py | grep -q vis_utils &&
    ex -s -c 31m23 -c w -c q object_detection/eval_util.py

protoc object_detection/protos/*.proto --python_out=.

# run a basic test to make sure tensorflow object detection is working
echo Running model_builder_test.py
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$MODELS_DIR/research:$MODELS_DIR/research/slim \
    $PYTHON $MODELS_DIR/research/object_detection/builders/model_builder_test.py

echo add tf models, research and slim to your PYTHONPATH
