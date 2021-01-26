#!/bin/bash
set -e

cd ../../../../../

# 1. Params
task_name="armor-digit"

read -rp "Experiment name:" experiment_name
job_id=${experiment_name}_$(date +'%Y%m%d_%H%M%S')
job_dir="gs://poly-cv-dev/experiments/$task_name/$job_id"
author=$(git config user.name | tr " " - | tr '[:upper:]' '[:lower:]')
echo Running job "$job_id" for task "$task_name" by "$author"

# 2. build source
poetry build -f wheel

# 3. start job
gcloud ai-platform jobs submit training "${job_id}" \
    --config src/research/robots/armor_digit/gcloud/hptuning_config.yaml \
    --job-dir="${job_dir}" \
    --packages ./dist/src-0.2.0-py3-none-any.whl \
    --module-name=research.robots.armor_digit.gcloud.train_cnn \
    --labels task=${task_name},author="${author}"

# 4. logs
echo "logs:  https://console.cloud.google.com/logs/query;query=resource.labels.job_id%3D%22${job_id}%22?project=polystar-cv"
echo "job:   https://console.cloud.google.com/ai-platform/jobs/${job_id}/charts/cpu?project=polystar-cv"

tensorboard --logdir="${job_dir}"
