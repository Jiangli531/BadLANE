import pickle
import argparse
import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader
import sys
import cv2
import os
import copy

sys.path.append("../")
from data.datasets import tusimple, culane

import warnings
warnings.filterwarnings("ignore")


def vis_data():
    my_device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    print("Vis data.")
    model_name = args.model
    if args.dataset == 'tusimple':
        dataset = tusimple(root_dir=args.dataroot, split=args.mode, model_name=model_name, task_num=args.task_num)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    elif args.dataset == 'culane':
        if model_name == 'laneatt':
            dataset = culane(root_dir=args.dataroot, split=args.mode, model_name=model_name, task_num=args.task_num)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    else:
        raise NotImplementedError

    data_loader_length = len(data_loader)
    print("Batch size: {}, batch number: {}, total image number {}".format(args.batch_size, data_loader_length,
                                                                           args.batch_size * data_loader_length))
    
    bar = tqdm.tqdm(data_loader)
    for i, batch in enumerate(bar):
        image, pois_img, image_all, pois_img_all, pos, mask, all_mask = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4], batch[5].cuda(), batch[6].cuda()
    
        

        if args.view:
            w_img = (image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            w_img_pois = (pois_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            w_img_all = (image_all[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            w_img_pois_all = (pois_img_all[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            new_path = 'poi_visualize'
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            cv2.imwrite(new_path+ '/' + str(i) + "_cln.jpg", w_img)
            cv2.imwrite(new_path+ '/' + str(i) + "_pois.jpg", w_img_pois)

            cv2.imwrite(new_path+ '/' + str(i) + "_cln_all.jpg", w_img_all)
            cv2.imwrite(new_path+ '/' + str(i) + "_pois_all.jpg", w_img_pois_all)
        
        torch.cuda.empty_cache()

        if i == 15:
            print("Visualize 15 images, break")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Curriculum data pre-handling')
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--dataset", type=str, default="tusimple")
    parser.add_argument("--dataroot", type=str, default='/mnt/cache/liyuxi/BadLANE/TUSimple/train_set')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--model", type=str, default='laneatt')
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--view", action="store_true", default=True)
    parser.add_argument("--task_num", type=int, default=10)
    args = parser.parse_args()
    vis_data()
