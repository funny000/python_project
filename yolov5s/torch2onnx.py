# -*- coding: utf-8 -*-
# author:liuxiaodong 
# contact: 2152550864@qq.com
# datetime:2021/7/29 19:50
# software: PyCharm
# python versions: Python3.7
# file: yolov5s
# license: (C)Copyright 2019-2021 liuxiaodong
import os
import sys
import torch
from models.experimental import attempt_load

gpu_id = 0
batch_size = 2
input_shape = (3, 608, 608)

weight_file = r"D:\WorkSpace\Aircarft_oiltank\best.pt"
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu")
model = attempt_load(weights = weight_file)
model.eval()
x = torch.randn(batch_size, *input_shape).to(device)
onnx_file = "yolov5s.onnx"
torch.onnx.export(model, x, onnx_file, opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input':{0:"batch_size"},
                    "output":{0:"batch_size"}})
