import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils
import tqdm

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate



COS_SIM = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

@torch.no_grad()
def validate_meta(model, args, iters=24):
    model.eval()
    epe_list = []
    cos_sim_list = []


    val_dataset = datasets.MetaDataset(split='val', root=args.dataset)

    eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, 
    pin_memory=False, shuffle=False, num_workers=2, drop_last=True)

    for i_batch, data_blob in tqdm.tqdm(enumerate(eval_loader)):
    
        image1, image2, flow_gt, valid = [x.cuda() for x in data_blob]

    # for val_id in range(len(val_dataset)):
    #     image1, image2, flow_gt, _ = val_dataset[val_id]
    #     image1 = image1[None].cuda()
    #     image2 = image2[None].cuda()

        _, flow_pr = model(image2, image1, iters=iters, test_mode=True)
        # epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        
        epe = torch.sum((flow_pr - flow_gt)**2, dim=1).sqrt()

        cos_sim = COS_SIM(flow_pr, flow_gt)
        
        epe_list.append(epe.view(-1).cpu().numpy())
        cos_sim_list.append(cos_sim.view(-1).cpu().numpy())

    epe = np.mean(np.concatenate(epe_list))
    cos_sim = np.mean(np.concatenate(cos_sim_list))
    print(f"Validation meta EPE: {epe}, cos_sim: {cos_sim}")
    return {'val_epe': epe, 'val_cos_sim': cos_sim}




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--batch_size', type=int, default=2)
    
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    with torch.no_grad():
        validate_meta(model.module, args)


