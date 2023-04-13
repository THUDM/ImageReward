'''
@File       :   inference.py
@Time       :   2023/03/12 15:35:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   Inference reward model.
'''

import os
import torch
import json
from tqdm import tqdm
import ImageReward as RM
import argparse


def acc(score_sample, target_sample):
    
    tol_cnt = 0.
    true_cnt = 0.
    for idx in range(len(score_sample)):
        item_base = score_sample[idx]["ranking"]
        item = target_sample[idx]["rewards"]
        for i in range(len(item_base)):
            for j in range(i+1, len(item_base)):
                if item_base[i] > item_base[j]:
                    if item[i] >= item[j]:
                        tol_cnt += 1
                    elif item[i] < item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                elif item_base[i] < item_base[j]:
                    if item[i] > item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                    elif item[i] <= item[j]:
                        tol_cnt += 1
    
    return true_cnt / tol_cnt


def test(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    score_sample = []
    with open(args.source_path, "r") as f:
        score_sample = json.load(f)

    model_type_list = []
    if args.model_type == 'all':
        model_type_list = ['ImageReward-v1.0', 'CLIP', 'BLIP', 'Aesthetic']
    else:
        model_type_list = [args.model_type]

    for model_type in model_type_list:
        print(f"{model_type} Test begin: ")
        
        if model_type == 'ImageReward-v1.0':
            model = RM.load(name=model_type, device=device, download_root=args.rm_path)
        else:
            model = RM.load_score(name=model_type, device=device, download_root=args.rm_path)
        
        target_sample = []
        # bar = tqdm(range(len(score_sample)), desc=f'{model_type} ranking')
        with torch.no_grad():
            for item in score_sample:
                img_list = [os.path.join(args.img_prefix, img) for img in item["generations"]]
                ranking, rewards = model.inference_rank(item["prompt"], img_list)
                
                target_item = {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "ranking": ranking,
                    "rewards": rewards
                }
                target_sample.append(target_item)
                # bar.update(1)
                
        target_path = os.path.join(args.target_dir, f"test_{model_type}.json")
        with open(target_path, "w") as f:
            json.dump(target_sample, f, indent=4, ensure_ascii=False)
        
        test_acc = acc(score_sample, target_sample)
        print(f"{model_type:>16s} Test Acc: {100 * test_acc:.2f}%")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--source_path', default="data/test.json", type=str)
    parser.add_argument('--target_dir', default="data/", type=str)
    parser.add_argument('--img_prefix', default="data/test_images", type=str)
    
    parser.add_argument('--model_type', default="all", type=str, help="ImageReward-v1.0, CLIP, BLIP, Aesthetic or all")
    parser.add_argument('--rm_path', default="checkpoint/", type=str)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.rm_path):
        os.makedirs(args.rm_path)
        
    test(args)
    