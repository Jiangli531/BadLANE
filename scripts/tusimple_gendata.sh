cd ../data

python data_prehandle.py \
  --mode train \
  --dataset tusimple \
  --dataroot ../TUSimple/train_set \
  --task_num 10 \
  --config_path ../models/laneatt/laneatt_tusimple_resnet34.yml \
  --checkpoint_path ../model_weights/laneatt_r34_tusimple/models/model_0100.pt \
  --shuffle \
  --attack_method pgd --batch_size 64 \
  --model laneatt --eps 0.2 --view