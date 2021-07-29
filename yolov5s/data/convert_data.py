# -*- coding: utf-8 -*-
# author:liuxiaodong 
# contact: 2152550864@qq.com
# datetime:2021/7/7 14:26
# software: PyCharm
# python versions: Python3.7
# file: convert_data.py
# license: (C)Copyright 2019-2021 liuxiaodong
import os
import math
import yaml
from xml.dom.minidom import parse
try:
  import xml.etree.cElementTree as ET
except ImportError:
  import xml.etree.ElementTree as ET

# root_path = os.getcwd()
yaml_path = os.path.join(os.getcwd(), "data\coco128.yaml")
# yaml_path = "data\coco128.yaml"
# yaml_path = r"D:\WorkSpace\Aircarft_oiltank\yolov5s\data\coco128.yaml"
with open(yaml_path, 'r+') as f:
    yaml_data = yaml.load(f, Loader=yaml.SafeLoader)
    class_dict = dict()
    [class_dict.setdefault(iy, k) for k, iy in enumerate(yaml_data['names'])]

# Coordinate conversion method to YOLO network readable format
def coordinates_convert(img_size, xy_box):
    """
    :param img_size:[w, h]
    :param xy_box:[xmin, xmax, ymin, ymax]
    :return:
    """
    dw = 1. / img_size[0]
    dh = 1. / img_size[1]
    x = (xy_box[0] + xy_box[1]) / 2.0
    y = (xy_box[2] + xy_box[3]) / 2.0
    w = math.fabs(xy_box[1] - xy_box[0])
    h = math.fabs(xy_box[3] - xy_box[2])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]
def xml2txt(xml_path, txt_path):
    file_tree = ET.ElementTree(file=xml_path)
    object_width = file_tree.getiterator('width')
    width = [obs.text for obs in object_width]
    object_height = file_tree.getiterator('height')
    height = [obs.text for obs in object_height]
    # file_name = file_tree.getiterator("name")
    # fname = [obs.text for obs in file_name]
    # object_find = file_tree.find('object')
    object_iter = file_tree.getiterator('object')
    objects_list = list()
    for k, obi in enumerate(object_iter):
        name_obj = obi.iter("name")
        name = [n.text for n in name_obj]
        xy_obj = obi.iter("xmin")
        xmin = [x.text for x in xy_obj]
        xy_obj = obi.iter("xmax")
        xmax = [x.text for x in xy_obj]
        xy_obj = obi.iter("ymin")
        ymin = [y.text for y in xy_obj]
        xy_obj = obi.iter("ymax")
        ymax = [y.text for y in xy_obj]
        coors = [float(xmin[0]), float(xmax[0]), float(ymin[0]), float(ymax[0])]
        coorss = coordinates_convert(img_size=[float(width[0]), float(height[0])], xy_box=coors)
        objects_list.append(coorss)
    k = class_dict.get(name[0])
    l = list()
    for obc in objects_list:
        il = [str(k)]
        il.append(str(obc[0]))
        il.append(str(obc[1]))
        il.append(str(obc[2]))
        il.append(str(obc[3]))
        l.append(il)
    return l




