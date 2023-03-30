import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet
import cv2


# 抠图（原图，mask图，输出前景图）
def get_foreground(img_name, mask_name, output_name):
    img1 = cv2.imread(img_name)
    img2 = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
    h, w, c = img1.shape
    img3 = np.zeros((h, w, 4))
    img3[:, :, 0:3] = img1
    img3[:, :, 3] = img2
    cv2.imwrite(output_name, img3)


input_path = "data/input"
mask_path = "data/mask"
output_path = "data/output"
ckpt_name = "pretrained/modnet_photographic_portrait_matting.ckpt"

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='path of input images', default=input_path)
    parser.add_argument('--mask-path', type=str, help='path of mask images', default=mask_path)
    parser.add_argument('--output-path', type=str, help='path of output images', default=output_path)
    parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet', default=ckpt_name)
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.input_path):
        print('Cannot find input path: {0}'.format(args.input_path))
        exit()
    if not os.path.exists(args.mask_path):
        print('Cannot find mask path: {0}'.format(args.mask_path))
        exit()
    if not os.path.exists(args.output_path):
        print('Cannot find output path: {0}'.format(args.output_path))
        exit()
    if not os.path.exists(args.ckpt_path):
        print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
        exit()

    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(args.ckpt_path)
    else:
        weights = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()

    # inference images
    im_names = os.listdir(args.input_path)
    for im_name in im_names:
        print('Process image: {0}'.format(im_name))

        # read image
        im = Image.open(os.path.join(args.input_path, im_name))

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte_name = im_name.split('.')[0] + '.png'
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(args.mask_path, matte_name))

        get_foreground(
            os.path.join(args.input_path, im_name),
            os.path.join(args.mask_path, matte_name),
            os.path.join(args.output_path, matte_name)
        )
