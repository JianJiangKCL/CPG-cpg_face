#!/bin/bash
#this file created by Jian
cd ..
DATASETS=(
#    'None'     # dummy
#    'imagenet'
    'cubs_cropped'
    'stanford_cars_cropped'
    'flowers'
    'wikiart'
    'sketches'
)
NUM_CLASSES=(
#    0
#    1000
    200
    196
    102
    195
    250
)
INIT_LR=(
#    0
#    1e-2
    1e-2
    1e-2
    1e-2
    1e-2
    1e-2
)
GPU_ID=0
one_shot_prune_perc=0.6

arch='resnet18'
#finetune_epochs=100
#prune_epochs=30
finetune_epochs=1
prune_epochs=1

PATH_DATA='/vol/jj/dataset/KM_dataset'

for TASK_ID in `seq 0 4`; do
  if [ "$TASK_ID" != "0" ]
  then
      CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_imagenet_main.py \
          --arch $arch \
          --path $PATH_DATA \
          --dataset ${DATASETS[TASK_ID]} --num_classes ${NUM_CLASSES[TASK_ID]} \
        --lr ${INIT_LR[TASK_ID]} \
          --weight_decay 4e-5 \
          --save_folder checkpoints/PackNet/experiment2/$arch/${DATASETS[TASK_ID]}/scratch \
          --jsonfile logs/PackNet_imagenet.json \
          --load_folder checkpoints/PackNet/experiment2/$arch/${DATASETS[TASK_ID-1]}/one_shot_prune \
          --epochs $finetune_epochs \
          --mode finetune

  else
      CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_imagenet_main.py \
          --arch $arch \
          --path $PATH_DATA \
          --dataset ${DATASETS[TASK_ID]} --num_classes ${NUM_CLASSES[TASK_ID]} \
        --lr ${INIT_LR[TASK_ID]} \
          --weight_decay 4e-5 \
          --save_folder checkpoints/PackNet/experiment2/$arch/${DATASETS[TASK_ID]}/scratch \
          --jsonfile logs/PackNet_imagenet.json \
          --epochs $finetune_epochs \
          --use_imagenet_pretrained \
          --mode finetune

  fi

   #Prune tasks
  CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_imagenet_main.py \
      --arch $arch \
      --path $PATH_DATA \
      --dataset ${DATASETS[TASK_ID]} --num_classes ${NUM_CLASSES[TASK_ID]} \
      --lr ${INIT_LR[TASK_ID]} \
      --weight_decay 4e-5 \
      --save_folder checkpoints/PackNet/experiment2/$arch/${DATASETS[TASK_ID]}/one_shot_prune \
      --load_folder checkpoints/PackNet/experiment2/$arch/${DATASETS[TASK_ID]}/scratch \
      --jsonfile logs/PackNet_imagenet.json \
      --epochs $prune_epochs \
      --mode prune \
      --one_shot_prune_perc $one_shot_prune_perc
done


# Evaluate tasks
for history_id in `seq 0 4`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_imagenet_main.py \
        --arch $arch \
        --path $PATH_DATA \
        --dataset ${DATASETS[history_id]} --num_classes ${NUM_CLASSES[TASK_ID]} \
        --load_folder checkpoints/PackNet/experiment2/$arch/${DATASETS[history_id]}/one_shot_prune \
        --mode inference \
        --logfile logs/PackNet_imagenet.txt
done
