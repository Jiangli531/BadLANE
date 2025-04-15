cd ../data

python visualize_pois_img.py \
  --mode train \
  --dataset tusimple \
  --dataroot /mnt/cache/liyuxi/BadLANE/TUSimple/train_set \
  --task_num 10 \
  --shuffle \
  --batch_size 64 \
  --model laneatt --eps 0.2 --view