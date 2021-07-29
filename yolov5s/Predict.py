# -*- coding: utf-8 -*-
# author:liuxiaodong 
# contact: 2152550864@qq.com
# datetime:2021/7/15 9:09
# software: PyCharm
# python versions: Python3.7
# file: yolov5s
# license: (C)Copyright 2019-2021 liuxiaodong
# -------------------------------------#
#       创建YOLO类
# -------------------------------------#
import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from models.experimental import attempt_load
# from nets.yolo4 import YoloBody
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFont, ImageDraw
from torch.autograd import Variable
from aiUtils import aiUtils

from utils.general import non_max_suppression, bbox_iou, yolo_correct_boxes, DecodeBox
# from utils.utils import non_max_suppression, bbox_iou, DecodeBox, letterbox_image, yolo_correct_boxes
import warnings
from xml.etree import ElementTree as ET
from osgeo import gdal, gdalconst, osr, ogr
# import gdal, gdalconst, osr, ogr
import colorsys, time
import os, glob, sys
import json
import chardet
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings('ignore')
# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
# --------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_image_size": (608, 608, 3),
        "cuda": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        # classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    def _get_anchors(self):
        # anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):

        # self.net = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()

        # 加快模型训练的效率
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # state_dict = torch.load(weights_file, map_location=device)
        # self.net.load_state_dict(state_dict)

        self.net = attempt_load(weights_file)

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        print('Finished!')

        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], len(self.class_names), (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(weights_file))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------py-gpu-nms---------------------------#
    # 非极大值抑制的实现
    def py_cpu_nms(self, dets, thresh):

        if len(dets) == 0:
            return []
        """Pure Python NMS baseline."""
        dets = np.array(dets)
        # print(dets)
        x1 = dets[:, 2]
        y1 = dets[:, 3]
        x2 = dets[:, 4]
        y2 = dets[:, 5]
        scores = dets[:, 1]  # bbox打分
        boxes = []
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # 打分从大到小排列，取index
        order = scores.argsort()[::-1]
        # keep为最后保留的边框
        keep = []
        while order.size > 0:
            # order[0]是当前分数最大的窗口，肯定保留
            i = order[0]
            boxes.append(dets[i])
            keep.append(i)
            # 计算窗口i与其他所有窗口的交叠部分的面积
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # 交/并得到iou值
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
            inds = np.where(ovr <= thresh)[0]
            # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
            order = order[inds + 1]

        return boxes
        # ----------------------------------#

    # -------------------------------nums2nums  overlap ----------------------------------------
    def mat_inter(self, dets, area_th_ratio):

        # 判断两个矩形是否相交
        # box=(xA,yA,xB,yB)
        boxes_no = []
        boxes_over = []
        if len(dets) <= 1:
            return dets
        # 判断两个矩形是否相交
        else:
            for i in range(len(dets) - 1):

                x01 = dets[i][2]
                y01 = dets[i][3]
                x02 = dets[i][4]
                y02 = dets[i][5]

                for j in range(i + 1, len(dets)):

                    x11 = dets[j][2]  # 取所有行第一列的数据
                    y11 = dets[j][3]
                    x12 = dets[j][4]
                    y12 = dets[j][5]

                    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
                    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
                    sax = abs(x01 - x02)
                    sbx = abs(x11 - x12)
                    say = abs(y01 - y02)
                    sby = abs(y11 - y12)
                    # --------------------相交--------------------------
                    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:

                        col = min(x02, x12) - max(x01, x11)
                        row = min(y02, y12) - max(y01, y11)
                        intersection = col * row
                        area1 = (x02 - x01) * (y02 - y01)
                        area2 = (x12 - x11) * (y12 - y11)

                        area1_ratio = intersection / area1
                        area2_ratio = intersection / area2
                        if area1_ratio > area_th_ratio:
                            boxes_over.append(list(dets[i]))
                            # else:
                            # boxes_no.append(list(dets[i]))
                            continue

                        if area2_ratio > area_th_ratio:
                            boxes_over.append(list(dets[j]))

                    else:  # 不相交
                        continue

            a1 = np.asarray(dets)
            a2 = np.asarray(boxes_over)

            a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
            a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])

            return np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])

    def uint16to8(self, bands, lower_percent=0.001, higher_percent=99.999):
        out = np.zeros_like(bands).astype(np.uint8)  # .astype(np.float)
        n = bands.shape[2]
        for i in range(n):
            a = 0  # np.min(band)
            b = 255  # np.max(band)
            c = np.percentile(bands[:, :, i], lower_percent)
            d = np.percentile(bands[:, :, i], higher_percent)
            t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
            t[t < a] = a
            t[t > b] = b
            out[:, :, i] = t

        return out

    # ----------------------------------------------------------#

    def get_region_boxes(self, patch_box, x_start, y_start):
        image_box = []
        xmin = x_start + patch_box[2]
        ymin = y_start + patch_box[3]
        xmax = x_start + patch_box[4]
        ymax = y_start + patch_box[5]

        image_box = [patch_box[0], patch_box[1], xmin, ymin, xmax, ymax]
        return image_box

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, filename, output_path):
        # 为了支持中文路径，请添加下面这句代码
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
        # 为了使属性表字段支持中文，请添加下面这句
        gdal.SetConfigOption("SHAPE_ENCODING", "")
        print("数据：", filename)
        # 用GDAL打开文件
        dataset = gdal.Open(filename)
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        outbandsize = dataset.RasterCount
        im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
        im_proj = dataset.GetProjection()  # 获取投影信息
        datatype = dataset.GetRasterBand(1).DataType
        # 创建矢量文件用
        xoffset = im_geotrans[1]
        yoffset = im_geotrans[5]
        xbase = im_geotrans[0]
        ybase = im_geotrans[3]
        xscale = im_geotrans[2]
        yscale = im_geotrans[4]

        im_data = dataset.ReadAsArray(0, 0, width, height)  # .astype(np.float32)
        im_data = np.transpose(im_data, (1, 2, 0))  # 转换为WHC
        im_data_3Band = im_data[:, :, 0:3]  # 只取前三个波段

        strcoordinates = "POLYGON ("
        if (outbandsize > 3):  # 模型位三波段模型
            outbandsize = 3
        # 创建输出文件
        driver = gdal.GetDriverByName("GTiff")
        outfileName = filename.split('/')[(len(filename.split('/')) - 1)]
        outfileName = outfileName.split('\\')[(len(outfileName.split('\\')) - 1)]
        outfileName = outfileName.rsplit('.', 1)[0]
        # outdataset = driver.Create(output_path + "/"  + outfileName +'_mask' +".tif", width, height, outbandsize,
        #                            gdal.GDT_Byte)
        ShpFileName = output_path + "/" + "out_" + outfileName + ".shp"
        # ------------------------------------设置投影信息---------------------------
        srs = osr.SpatialReference()
        srs.ImportFromWkt(dataset.GetProjectionRef())
        # print("shp的投影信息", srs)
        prjFile = open(ShpFileName[:-4] + ".prj", 'w')
        # 转为字符
        srs.MorphToESRI()
        prjFile.write(srs.ExportToWkt())
        prjFile.close()
        # --------------------------------------------
        # 创建输出的xml文件
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img = im_data_3Band

        temp = 608

        x_idx = range(0, img.shape[1], temp - 208)
        y_idx = range(0, img.shape[0], temp - 208)
        rslt_mask = np.zeros((height, width, outbandsize), dtype=np.uint8)
        mask_temp = np.zeros((temp, temp, outbandsize), dtype=np.uint8)

        out_boxes = []
        all_boxes = []
        strcoordinates_box = []
        total_progress = len(x_idx) * len(y_idx)
        count = 0
        print("[AIProgress] {} 0".format(filename), flush=True)
        for x_start in x_idx:
            for y_start in y_idx:
                x_stop = x_start + temp
                if x_stop > img.shape[1]:
                    x_start = max(0, img.shape[1] - temp)
                    x_stop = img.shape[1]
                y_stop = y_start + temp
                if y_stop > img.shape[0]:
                    y_start = max(0, img.shape[0] - temp)
                    y_stop = img.shape[0]

                image = img[y_start:y_stop, x_start:x_stop, 0:3]
                mask_temp[0:(y_stop - y_start), 0:(x_stop - x_start), :] = image[:, :, :]

                image_shape = np.array(np.shape(mask_temp)[0:2])
                photo = np.array(mask_temp, dtype=np.float64)

                photo /= 255.0
                photo = np.transpose(photo, (2, 0, 1))
                photo = photo.astype(np.float32)
                images = []
                images.append(photo)
                images = np.asarray(images)

                with torch.no_grad():
                    images = torch.from_numpy(images)
                    if self.cuda:
                        images = images.cuda()
                    outputs = self.net(images)[0]

                # output_list = []
                # for i in range(3):
                #     output_list.append(self.yolo_decodes[i](outputs[i]))
                # output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(outputs)
                # batch_detections = non_max_suppression(output, len(self.class_names),
                #                                        conf_thres=score_threshold,
                #                                        nms_thres=0.3)
                if batch_detections[0] is None:
                    continue

                else:
                    batch_detections = batch_detections[0].cpu().numpy()
                    top_index = batch_detections[:, 4] * batch_detections[:, 5] > score_threshold
                    top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
                    top_label = np.array(batch_detections[top_index, -1], np.int32)
                    top_bboxes = np.array(batch_detections[top_index, :4])
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                        top_bboxes[:, 1],
                        -1), np.expand_dims(
                        top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

                    # 去掉灰条
                    boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                               np.array([self.model_image_size[0], self.model_image_size[1]]),
                                               image_shape)
                    # font = ImageFont.truetype(font='model_data/simhei.ttf',
                    #                           size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

                    # thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]
                    for i, c in enumerate(top_label):
                        predicted_class = self.class_names[c]
                        score = top_conf[i]

                        top, left, bottom, right = boxes[i]
                        top = top - 5
                        left = left - 5
                        bottom = bottom + 5
                        right = right + 5

                        patch_box = [int(c), score, left, top, right, bottom]
                        boxes_mc = self.get_region_boxes(patch_box, x_start, y_start)

                        all_boxes.append(boxes_mc)
                    count += 1
                    now_progress = int(100 * count / total_progress)
                    if now_progress < 100:
                        print("[AIProgress] {} {}".format(filename, now_progress), flush=True)

            out_boxes = self.py_cpu_nms(np.array(all_boxes), float(nms))
        print("[AIProgress] {} 100".format(filename), flush=True)

            # print(out_boxes)
        for k, out_box in enumerate(out_boxes):  # 对每个目标进行处理，按原始尺寸进行缩放
            classes = self.class_names[int(out_boxes[k][0])]

            xmin = xbase + out_boxes[k][2] * xoffset + out_boxes[k][3] * xscale
            ymin = ybase + out_boxes[k][3] * yoffset + out_boxes[k][2] * yscale
            xmax = xbase + out_boxes[k][4] * xoffset + out_boxes[k][5] * xscale
            ymax = ybase + out_boxes[k][5] * yoffset + out_boxes[k][4] * xscale

            strcoordinates = strcoordinates + '(%f1 %f2,%f3 %f4,%f5 %f6,%f7 %f8,%f9 %f10,%s)' % (
                xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin, classes)
            strcoordinates = strcoordinates + ','
            str = (xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin, classes)
            strcoordinates_box.append(str)
        n = len(strcoordinates)

        print('rect number:', n)
        strcoordinates = strcoordinates[0:n - 1] + ")"

        # 为了支持中文路径，请添加下面这句代码
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
        # 为了使属性表字段支持中文，请添加下面这句
        gdal.SetConfigOption("SHAPE_ENCODING", "")
        # 注册所有的驱动

        # 注册所有的驱动
        ogr.RegisterAll()

        # 创建数据，这里以创建ESRI的shp文件为例
        strDriverName = "ESRI Shapefile"
        oDriver = ogr.GetDriverByName(strDriverName)
        if oDriver == None:
            print("%s 驱动不可用！\n", strDriverName)
            return

        # 创建数据源
        oDS = oDriver.CreateDataSource(ShpFileName)
        if oDS == None:
            print("创建文件【%s】失败！", ShpFileName)
            return

        # 创建图层，创建一个多边形图层，这里没有指定空间参考，如果需要的话，需要在这里进行指定
        papszLCO = []
        oLayer = oDS.CreateLayer("TestPolygon", None, ogr.wkbPolygon, papszLCO)
        if oLayer == None:
            print("图层创建失败！\n")
            return
        # 下面创建属性表
        # 先创建一个叫FieldID的整型属性
        oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)
        oLayer.CreateField(oFieldID, 1)

        # 再创建一个叫FeatureName的字符型属性，字符长度为50
        oFieldName = ogr.FieldDefn("FieldName", ogr.OFTString)
        oFieldName.SetWidth(100)
        oLayer.CreateField(oFieldName, 1)

        oDefn = oLayer.GetLayerDefn()
        # 创建矩形要素
        for i in range(len(strcoordinates_box)):
            oFeatureRectangle = ogr.Feature(oDefn)
            oFeatureRectangle.SetField(0, 1)
            # print(strcoordinates_box[i])
            oFeatureRectangle.SetField(1, strcoordinates_box[i][10])
            geomRectangle = ogr.CreateGeometryFromWkt('POLYGON ((%f1 %f2,%f3 %f4,%f5 %f6,%f7 %f8,%f9 %f10))' %
                                                      (strcoordinates_box[i][0], strcoordinates_box[i][1],
                                                       strcoordinates_box[i][2], strcoordinates_box[i][3],
                                                       strcoordinates_box[i][4],
                                                       strcoordinates_box[i][5], strcoordinates_box[i][6],
                                                       strcoordinates_box[i][7], strcoordinates_box[i][8],
                                                       strcoordinates_box[i][9]))
            oFeatureRectangle.SetGeometry(geomRectangle)
            oLayer.CreateFeature(oFeatureRectangle)

        oDS.Destroy()
        del dataset


if __name__ == '__main__':
    t0 = time.time()  # 开始时间
    ogr.RegisterAll()
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "CP936")

    # config_path = input()
    # if len(sys.argv) < 2:
    #     config_path = sys.path[0] + "/cfg.json"
    # else:
    #     config_path = sys.argv[1]

    save_dir = os.getcwd()
    wdir = save_dir + '/' + 'weights'
    # wdir.mkdir(parents=True, exist_ok=True)  # make dir
    # last = wdir / 'last.pt'
    best = wdir + '/' + 'best.pt'


    # config_path = r"D:\WorkSpace\Aircarft_oiltank\config.json"
    # enc = chardet.detect(open(config_path, 'rb').read())['encoding']
    # with open(config_path, 'r', encoding=enc) as f:
    #     params = json.load(f)

    conf = input()
    print('conf:',conf)
    params = json.loads(conf)
    Input_imgpath = params['InputImgPath']
    classes_path = params['classes_path']
    anchors_path=params['anchors_path']
    nms= params['nms_thresh']
    score_threshold = float(params['score_threshold'])
    weights_file = params['WeightFile'] if params["WeightFile"] else best
    output_path = params['OutputPath']

    # classes_path = r"D:\WorkSpace\Aircarft_oiltank\yolov5s\two_classes.txt"
    # anchors_path = r"D:\WorkSpace\Aircarft_oiltank\yolov5s\yolo_anchors.txt"
    # Input_imgpath = r"D:\WorkSpace\Bijie_landslide_dataset\qianxi.tif"
    # nms = 0.5
    # score_threshold = 0.6
    # weights_file = r"D:\WorkSpace\Bijie_landslide_dataset\best.pt"
    # output_path = r"D:\WorkSpace\Bijie_landslide_dataset"

    yolo = YOLO()

    filelist = []
    if isinstance(Input_imgpath, list):
        filelist = Input_imgpath
        for i, idx_img in enumerate(filelist):
            if os.path.isdir(idx_img):
                img_list = [x for x in os.listdir(idx_img) if os.path.splitext(x)[1] in ['.tif', '.TIF','.tiff', '.jpg', '.png']]
                for i, idx_img_1 in enumerate(img_list):
                    print("[AIProgressFiles]: [{}/{}]".format(i + 1, len(img_list)))
                    img_path = os.path.join(idx_img, idx_img_1)
                    yolo.detect_image(img_path, output_path)
            elif os.path.isfile(idx_img):
                print("[AIProgressFiles]: [{}/{}]".format(1, 1))
                yolo.detect_image(idx_img, output_path)
            else:
                print("it is empty")
    elif Input_imgpath.endswith('.csv'):
        img_list_file = open(Input_imgpath, 'r', encoding='UTF-8')
        img_list = img_list_file.readlines()
        img_list_file.close()
        for i, idx_img in enumerate(img_list):
            print("[AIProgressFiles]: [{}/{}]".format(i + 1, len(img_list)))
            yolo.detect_image(Input_imgpath, output_path)
    else:
        if os.path.isdir(Input_imgpath):
            img_list = [x for x in os.listdir(Input_imgpath) if os.path.splitext(x)[1] in ['.tif', '.TIF','.tiff', '.jpg', '.png']]
            for i, idx_img in enumerate(img_list):
                print("[AIProgressFiles]: [{}/{}]".format(i + 1, len(img_list)))
                img_path = os.path.join(Input_imgpath, idx_img)
                yolo.detect_image(img_path, output_path)
        elif os.path.isfile(Input_imgpath):
            print("[AIProgressFiles]: [{}/{}]".format(1, 1))
            yolo.detect_image(Input_imgpath, output_path)
        else:
            print("it is empty")

