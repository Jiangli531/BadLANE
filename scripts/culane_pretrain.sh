#!/bin/bash
xhidden=64
xsize=128
yhidden=256
depth=8
level=3
batchsize=16
epochs=8
testgap=100000
loggap=10
savegap=1000
infergap=100000
lr=0.001
grad_clip=0
grad_norm=10
regularizer=0
adv_loss=False
learn_top=False
only=True
tanh=False
clamp=True
model_path=''
margin=5.0
linf=0.1

name='culane_train'
logroot="./save/laneatt/"
train_dataset_root="{your generated meta-tasks path}"
valid_dataset_root="{your generated meta-tasks path}"

cd ../

python train.py --dataset_name=tusimple \
  --train_dataset_root=${train_dataset_root} --valid_dataset_root=${valid_dataset_root} \
  --log_root=${logroot} --x_hidden_channels=${xhidden} --y_hidden_channels=${yhidden} \
  --x_hidden_size=${xsize} --flow_depth=${depth} --num_levels=${level} --num_epochs=${epochs} --batch_size=${batchsize} \
  --test_gap=${testgap} --log_gap=${loggap} --inference_gap=${infergap} --lr=${lr} --max_grad_clip=${grad_clip} \
  --max_grad_norm=${grad_norm} --save_gap=${savegap}  --regularizer=${regularizer} --adv_loss=${adv_loss} \
  --learn_top=${learn_top} --model_path=${model_path} --tanh=${tanh} --only=${only} --clamp=${clamp} \
  --name=${name} --down_sample_x 4 --down_sample_y 4 --linf=${linf}

