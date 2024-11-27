
from myUtils import dispDepthMapper
mapper = dispDepthMapper('disp.txt')

import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    return flo
    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()

def disp_vis_cv2(disp, min_disp, max_disp):
	disp_vis = (disp - min_disp) / (max_disp - min_disp)
	disp_vis = np.clip(disp_vis, 0, 1)
	disp_vis *= 255.0
	disp_vis = disp_vis.astype("uint8")
	disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_VIRIDIS)
	return disp_vis

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()


    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)
            
            flow_viz = viz(image1, flow_up)

            flow_up = flow_up[0].permute(1,2,0).cpu().numpy()


            print(flow_up.shape)
            flow_shape = flow_up.shape
            flow_up = flow_up.reshape(-1, 2)
            
            print('matching depth..')
            flow_size = flow_up.shape[0]
            depth_estimation = []
            import tqdm
            batch_size = 512
            for i in tqdm.tqdm(range(batch_size)):
                id_start = i*flow_size//batch_size
                id_end = (i+1)*flow_size//batch_size
                id_end = min(id_end, flow_size)
                depth_estimation.append(mapper.disp2depth_batch(flow_up[id_start:id_end, :]))
            depth_estimation = np.stack(depth_estimation)
            depth_estimation = depth_estimation.reshape(flow_shape[0], flow_shape[1])
            print(depth_estimation.shape)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            print(flow_viz.shape, depth_estimation.shape)
            depth_estimation= disp_vis_cv2(depth_estimation, 0.5, 5.0)
            plt.imshow(np.hstack([flow_viz, depth_estimation]))  # You can use other colormaps like 'jet', 'plasma', etc.
            # plt.colorbar(label='Depth (units)')  # Add a colorbar to represent depth values
            # plt.title('Depth Map')
            # plt.xlabel('X-axis')
            # plt.ylabel('Y-axis')
            plt.show()


            # viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
