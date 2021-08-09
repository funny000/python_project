# -*- coding: utf-8 -*-
# author:liuxiaodong 
# contact: 2152550864@qq.com
# datetime:2021/8/6 16:21
# software: PyCharm
# python versions: Python3.7
# file: network
# license: (C)Copyright 2019-2021 liuxiaodong
import os
import threading
import cv2
import sys
import shutil


basddir = r'/media/lxd/workspace/landslide_aricraft/yolov5s/data'
resultdir = r"/media/lxd/workspace/landslide_aricraft/landslide_aricraft/train/images"
newdir = "".join(["{}\\".format(content) for content in resultdir.split("\\")[:-1]])
otherdir = os.path.join(newdir, 'otherimg')
if not os.path.exists(otherdir):
    os.makedirs(otherdir)
files_list = os.listdir(basddir)
imgs_list = os.listdir(resultdir)
datas = list()
for fl in files_list:
    if fl.endswith(".txt"):
        fl_path = os.path.join(basddir, fl)
        with open(fl_path, 'r') as f:
            data = f.readlines()
            data = [d.strip('\n') for d in data]
            datas = data
        f.close()
for il in imgs_list:
    il_name = il.split('.')[0]
    if datas.__contains__(il_name):
        il_name_path = os.path.join(resultdir, "{}.txt".format(il_name))
        il_ohter_path = os.path.join(otherdir, '{}.txt'.format(il_name))
        shutil.copyfile(il_name_path, il_ohter_path)
        # il_data = cv2.imread(il_name_path)
        # cv2.imwrite(il_ohter_path, il_data)
        # new = cv2.imread(il_ohter_path)
        # os.remove(il_name_path)
        os.unlink(il_name_path)
        print("process success {}".format(il_name))


