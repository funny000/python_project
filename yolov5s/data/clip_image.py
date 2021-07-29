# -*- coding: utf-8 -*-
# author:liuxiaodong 
# contact: 2152550864@qq.com
# datetime:2021/7/8 14:31
# software: PyCharm
# python versions: Python3.7
# file: clip_image.py
# license: (C)Copyright 2019-2021 liuxiaodong
import cv2
import os
import PIL.Image as Image
from pathlib import Path

import torch

from models.experimental import attempt_load
from utils.general import non_max_suppression

# load model
weight_path = r"D:\WorkSpace\Aircarft_oiltank\best.pt"
model = attempt_load(weight_path)

# img_path = r"D:\ProgramFile0\feiqiu13\feiq\Recv Files\detect_img\beijing2.png"
img_path = r"D:\Program0\fqiu\feiq\Recv Files\beijing.tif"
img_data = cv2.imread(img_path)
height, width, channels =  img_data.shape
save_path = Path(img_path).parent

# image = Image.open(img_path)
# width, height = image.size
for i in range(0, width, 640):
    for j in range(0, height, 640):
        new_img2 = img_data[i:i + 640, j:j + 640, :]
        if i + 640 > width:
            # new_img2 = img_data.crop((width - 640, j, width, j + 640))
            new_img2 = img_data[width-640:width, j:j+640, :]
            # img_save_path2 = os.path.join(save_path, "{0}_{1}.png".format(width, height))
            # new_img2.save(img_save_path2)
        elif j + 640 > height:
            # new_img2 = img_data.crop((i, height - 640, i + 640, height))
            new_img2 = img_data[i:i+640, height-640:height, :]
        # new_img2 = img_data.crop((i, j, i + 640, j + 640))
        new_img2 = torch.transpose(torch.from_numpy(new_img2), dim0=2, dim1=0).unsqueeze(0)
        new_img2 = new_img2.float() / 255.0
        detect_output = model(new_img2, augment='store_true')[0]
        nms_detect_output = non_max_suppression(detect_output, conf_thres=0.25, iou_thres=0.45)[0]
        output = list(nms_detect_output.numpy())
        if not output:
            nms_detect_output = torch.transpose(new_img2.squeeze(0), dim0=0, dim1=2)
        img_save_path = os.path.join(save_path, "{0}_{1}.png".format(i, j))
        # cv2.imwrite(img_save_path, new_img2)
        cv2.imwrite(img_save_path, nms_detect_output.cpu().numpy())
print('---success---')



