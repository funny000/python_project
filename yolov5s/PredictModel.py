# -*- coding: utf-8 -*-
# author:liuxiaodong 
# contact: 2152550864@qq.com
# datetime:2021/7/27 9:37
# software: PyCharm
# python versions: Python3.7
# file: yolov5s
# license: (C)Copyright 2019-2021 liuxiaodong
import datetime

import cv2
import numpy as np
import random
import colorsys
import os
import torch
import torch.nn as nn
from models.experimental import attempt_load
# from nets.yolo4 import YoloBody
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFont, ImageDraw
from torch.autograd import Variable
# from aiUtils import aiUtils
import codecs
import shapefile

from utils.plots import plot_one_box
from utils.general import non_max_suppression, bbox_iou, yolo_correct_boxes, DecodeBox, scale_coords
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

# set the input file and state dict file
# basedir = os.path.abspath(os.path.dirname(__file__))
# conf = basedir + 'traincfg.json'
# print('conf:', conf)
# params = json.loads(conf)
# Input_imgpath = params['InputImgPath']
# classes_path = params['classes_path']
# anchors_path = params['anchors_path']
# nms= params['nms_thresh']
# score_threshold = float(params['score_threshold'])
# weights_file = params['WeightFile']
weights_file = './output/yolov5s.pth'
output_path = './output/predict'
if not os.path.exists(output_path):
    os.makedirs(output_path)


# basedir = os.getcwd()
# Input_imgpath = params['InputImgPath']
# classes_path = basedir + "/class.txt"
# anchors_path = basedir + "/anchors.txt"
# nms = 0.45
# score_threshold = 0.4
# weights_file = basedir + "/*.th"
# output_path = basedir + "/output"

# --------------------------------------------#
#   ????????????????????????????????????????????????2?????????
#   model_path???classes_path??????????????????
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
    #   ?????????YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    # ---------------------------------------------------#
    #   ?????????????????????
    # ---------------------------------------------------#
    def _get_class(self):
        # classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   ????????????????????????
    # ---------------------------------------------------#
    def _get_anchors(self):
        # anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    # ---------------------------------------------------#
    #   ?????????????????????
    # ---------------------------------------------------#
    def generate(self):

        # self.net = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()

        # ???????????????????????????
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
        # ???????????????????????????
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------py-gpu-nms---------------------------#
    # ???????????????????????????
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
        scores = dets[:, 1]  # bbox??????
        boxes = []
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # ??????????????????????????????index
        order = scores.argsort()[::-1]
        # keep????????????????????????
        keep = []
        while order.size > 0:
            # order[0]?????????????????????????????????????????????
            i = order[0]
            boxes.append(dets[i])
            keep.append(i)
            # ????????????i?????????????????????????????????????????????
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # ???/?????????iou???
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # inds??????????????????i???iou?????????threshold???????????????index?????????????????????????????????i??????
            inds = np.where(ovr <= thresh)[0]
            # order????????????????????????i??????????????????threshold????????????????????????ovr?????????order?????????1(?????????i)?????????inds+1????????????????????????
            order = order[inds + 1]

        return boxes
        # ----------------------------------#

    # -------------------------------nums2nums  overlap ----------------------------------------
    def mat_inter(self, dets, area_th_ratio):

        # ??????????????????????????????
        # box=(xA,yA,xB,yB)
        boxes_no = []
        boxes_over = []
        if len(dets) <= 1:
            return dets
        # ??????????????????????????????
        else:
            for i in range(len(dets) - 1):
                x01 = dets[i][2]
                y01 = dets[i][3]
                x02 = dets[i][4]
                y02 = dets[i][5]
                for j in range(i + 1, len(dets)):
                    x11 = dets[j][2]  # ??????????????????????????????
                    y11 = dets[j][3]
                    x12 = dets[j][4]
                    y12 = dets[j][5]
                    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
                    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
                    sax = abs(x01 - x02)
                    sbx = abs(x11 - x12)
                    say = abs(y01 - y02)
                    sby = abs(y11 - y12)
                    # --------------------??????--------------------------
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

                    else:  # ?????????
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
    #   ????????????
    # ---------------------------------------------------#
    def detect_image(self, filename, output_path):
        # ??????????????????????????????????????????????????????
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
        # ????????????????????????????????????????????????????????????
        gdal.SetConfigOption("SHAPE_ENCODING", "")
        print("?????????", filename)
        # ???GDAL????????????
        dataset = gdal.Open(filename)
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        outbandsize = dataset.RasterCount
        im_geotrans = dataset.GetGeoTransform()  # ????????????????????????
        im_proj = dataset.GetProjection()  # ??????????????????
        datatype = dataset.GetRasterBand(1).DataType
        # ?????????????????????
        xoffset = im_geotrans[1]
        yoffset = im_geotrans[5]
        xbase = im_geotrans[0]
        ybase = im_geotrans[3]
        xscale = im_geotrans[2]
        yscale = im_geotrans[4]

        im_data = dataset.ReadAsArray(0, 0, width, height)  # .astype(np.float32)
        im_data = np.transpose(im_data, (1, 2, 0))  # ?????????WHC
        im_data_3Band = im_data[:, :, 0:3]  # ?????????????????????

        strcoordinates = "POLYGON ("
        if (outbandsize > 3):  # ????????????????????????
            outbandsize = 3
        # ??????????????????
        driver = gdal.GetDriverByName("GTiff")
        outfileName = filename.split('/')[(len(filename.split('/')) - 1)]
        outfileName = outfileName.split('\\')[(len(outfileName.split('\\')) - 1)]
        outfileName = outfileName.rsplit('.', 1)[0]
        # outdataset = driver.Create(output_path + "/"  + outfileName +'_mask' +".tif", width, height, outbandsize,
        #                            gdal.GDT_Byte)
        now_time = datetime.datetime.now()
        ShpFileName = output_path + "/" + "out_" + outfileName + "{}_.shp".format(now_time.strftime("%S"))
        # ------------------------------------??????????????????---------------------------
        srs = osr.SpatialReference()
        srs.ImportFromWkt(dataset.GetProjectionRef())
        # print("shp???????????????", srs)
        prjFile = open(ShpFileName[:-4] + ".prj", 'w')
        # ????????????
        srs.MorphToESRI()
        prjFile.write(srs.ExportToWkt())
        prjFile.close()
        # --------------------------------------------
        # ???????????????xml??????
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img = im_data_3Band
        temp = 508
        x_idx = range(0, img.shape[1], temp - 108)
        y_idx = range(0, img.shape[0], temp - 108)
        rslt_mask = np.zeros((height, width, outbandsize), dtype=np.uint8)
        mask_temp = np.zeros((temp, temp, outbandsize), dtype=np.uint8)
        out_boxes = []
        all_boxes = []
        strcoordinates_box = []
        total_progress = len(x_idx) * len(y_idx)
        count = 0
        print("[AIProgress] {} 0".format(filename), flush=True)
        # decide whether it's a big picture or a small picture
        if width > 450 and height > 450:
            # big picture
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
                    all_boxes = self.predict_big_image(img, mask_temp, y_start, y_stop, x_start, x_stop, all_boxes, count, total_progress, filename)
        else:
            # small picture
            self.predict_samll_image_new(img, weights_file)
        out_boxes = self.py_cpu_nms(np.array(all_boxes), float(nms))
        print("[AIProgress] {} 100".format(filename), flush=True)
        for k, out_box in enumerate(out_boxes):  # ?????????????????????????????????????????????????????????
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

        # ??????????????????????????????????????????????????????
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
        # ????????????????????????????????????????????????????????????
        gdal.SetConfigOption("SHAPE_ENCODING", "")
        # ?????????????????????

        # ?????????????????????
        ogr.RegisterAll()

        # ??????????????????????????????ESRI???shp????????????
        strDriverName = "ESRI Shapefile"
        oDriver = ogr.GetDriverByName(strDriverName)
        if oDriver == None:
            print("%s ??????????????????\n", strDriverName)
            return

        # ???????????????
        oDS = oDriver.CreateDataSource(ShpFileName)
        if oDS == None:
            print("???????????????%s????????????", ShpFileName)
            return

        # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        papszLCO = []
        oLayer = oDS.CreateLayer("TestPolygon", None, ogr.wkbPolygon, papszLCO)
        if oLayer == None:
            print("?????????????????????\n")
            return
        # ?????????????????????
        # ??????????????????FieldID???????????????
        oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)
        oLayer.CreateField(oFieldID, 1)

        # ??????????????????FeatureName????????????????????????????????????50
        oFieldName = ogr.FieldDefn("FieldName", ogr.OFTString)
        oFieldName.SetWidth(100)
        oLayer.CreateField(oFieldName, 1)

        oDefn = oLayer.GetLayerDefn()
        # ??????????????????
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
        self.shp2gjson(ShpFileName.split(".")[0])
        del dataset

    def predict_samll_image_new(self, img, weight_file, output_path=None):
        """
        new function to predict image
        :param img:
        :param weight_file:
        :param output_path:
        :return:
        """
        model = attempt_load(weight_file, map_location = 'cpu')
        output_path = os.path.join(os.getcwd(), "pred.jpg")
        names = self.net.module.names if hasattr(self.net, 'module') else self.net.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        img0 = img
        img = np.transpose(img, (2, 0, 1))
        channels, w, h = img.shape
        img = np.resize(img, (channels, 640, 640))
        img = torch.from_numpy(img).to("cpu")
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.20, 0.40, None, False)
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]}{conf:.2f}'
                    plot_one_box(xyxy, img0, label, colors[0])
            cv2.imwrite(output_path, img0)
        return pred

    def predict_small_image(self, img, mask_temp, width, height, all_boxes, count, total_progress, filename):
        """
        process small or ndarray image
        :param img:
        :param mask_temp:
        :param width:
        :param height:
        :param all_boxes:
        :param count:
        :param total_progress:
        :param filename:
        :return:
        """
        image = img[:, :, 0:3]
        mask_temp = np.zeros((width, height, 3), dtype=np.uint8)
        mask_temp[0:width, 0:height, :] = image[:, :, :]

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

        batch_detections = non_max_suppression(outputs)
        if batch_detections[0] is None:
            pass
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

            # ????????????
            boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                       np.array([self.model_image_size[0], self.model_image_size[1]]),
                                       image_shape)
            for i, c in enumerate(top_label):
                predicted_class = self.class_names[c]
                score = top_conf[i]
                top, left, bottom, right = boxes[i]
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5
                patch_box = [int(c), score, left, top, right, bottom]
                boxes_mc = self.get_region_boxes(patch_box, 0, 0)
                all_boxes.append(boxes_mc)
            count += 1
            now_progress = int(100 * count / total_progress)
            if now_progress < 100:
                print("[AIProgress] {} {}".format(filename, now_progress), flush=True)
            return all_boxes


    def predict_big_image(self, img, mask_temp, y_start, y_stop, x_start, x_stop, all_boxes, count, total_progress, filename):
        """
        process big image
        :param img:
        :param mask_temp:
        :param y_start:
        :param y_stop:
        :param x_start:
        :param x_stop:
        :param all_boxes:
        :param count:
        :param total_progress:
        :param filename:
        :return:
        """
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
        batch_detections = non_max_suppression(outputs)
        if batch_detections[0] is None:
            pass
        else:
            batch_detections = batch_detections[0].cpu().numpy()
            print(batch_detections[:, 4] * batch_detections[:, 5])
            top_index = batch_detections[:, 4] * batch_detections[:, 5] >= 0 # score_threshold
            top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
            top_label = np.array(batch_detections[top_index, -1], np.int32)
            top_bboxes = np.array(batch_detections[top_index, :4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                top_bboxes[:, 1],
                -1), np.expand_dims(
                top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)
            # ????????????
            boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                       np.array([self.model_image_size[0], self.model_image_size[1]]),
                                       image_shape)
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
            return all_boxes


    def shp2gjson(self, file_path, shp_encoding='utf-8',json_encoding='utf-8'):
        """
        shpfile transform to geojson
        :param file_path:
        :param shp_encoding:
        :param json_encoding:
        :return:
        """
        reader = shapefile.Reader(file_path, encoding=shp_encoding)
        fields = reader.fields[1:]
        field_names = [field[0] for field in fields]
        buffer = []
        for sr in tqdm(reader.shapeRecords()):
            record = sr.record
            record = [r.decode('gb2312', 'ignore') if isinstance(r, bytes)
                      else r for r in record]
            atr = dict(zip(field_names, record))
            geom = sr.shape.__geo_interface__
            buffer.append(dict(type="Feature",
                               geometry=geom,
                               properties=atr))
        geojson = codecs.open(file_path + "-geo.json", "w", encoding=json_encoding)
        geojson.write(json.dumps({"type": "FeatureCollection",
                                  "features": buffer}) + '\n')
        geojson.close()


class PredictModel():
    def __init__(self):
        super(PredictModel, self).__init__()
        self.yolov5 = YOLO()
    def predict(self, Input_imgpath, names = None, meta = None):
        """
        run the model to predict a image or a list images or a video
        :param X: array ??? file??? filelist
        :param names: None
        :param meta: None
        :return: the predict result
        """
        filelist = []
        if isinstance(Input_imgpath, list):
            filelist = Input_imgpath
            for i, idx_img in enumerate(filelist):
                if os.path.isdir(idx_img):
                    img_list = [x for x in os.listdir(idx_img) if
                                os.path.splitext(x)[1] in ['.tif', '.TIF', '.tiff', '.jpg', '.png']]
                    for i, idx_img_1 in enumerate(img_list):
                        print("[AIProgressFiles]: [{}/{}]".format(i + 1, len(img_list)))
                        img_path = os.path.join(idx_img, idx_img_1)
                        self.yolov5.detect_image(img_path, output_path)
                elif os.path.isfile(idx_img):
                    print("[AIProgressFiles]: [{}/{}]".format(1, 1))
                    self.yolov5.detect_image(idx_img, output_path)
                else:
                    print("it is empty")
        elif isinstance(Input_imgpath, str):
            if os.path.isdir(Input_imgpath):
                img_list = [x for x in os.listdir(Input_imgpath) if
                            os.path.splitext(x)[1] in ['.tif', '.TIF', '.tiff', '.jpg', '.png']]
                for i, idx_img in enumerate(img_list):
                    print("[AIProgressFiles]: [{}/{}]".format(i + 1, len(img_list)))
                    img_path = os.path.join(Input_imgpath, idx_img)
                    self.yolov5.detect_image(img_path, output_path)
            elif os.path.isfile(Input_imgpath):
                print("[AIProgressFiles]: [{}/{}]".format(1, 1))
                self.yolov5.detect_image(Input_imgpath, output_path)
        elif isinstance(Input_imgpath, np.ndarray):
            self.yolov5.predict_samll_image_new(Input_imgpath, weights_file)
            print("process success,{}".format(datetime.datetime.now().strftime("%Y-%m-%D %t-%m-%s")))
        else:
            print("the input data or file has to be follows format np.ndarray, file path, files path!")


if __name__ == '__main__':
    t0 = time.time()  # ????????????
    ogr.RegisterAll()
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "CP936")

    # config_path = input()
    # if len(sys.argv) < 2:
    #     config_path = sys.path[0] + "/cfg.json"
    # else:
    #     config_path = sys.argv[1]

    # save_dir = os.getcwd()
    # wdir = save_dir + '/' + 'weights'
    # # wdir.mkdir(parents=True, exist_ok=True)  # make dir
    # # last = wdir / 'last.pt'
    # best = wdir + '/' + 'best.pt'


    # config_path = r"D:\WorkSpace\Aircarft_oiltank\config.json"
    # enc = chardet.detect(open(config_path, 'rb').read())['encoding']
    # with open(config_path, 'r', encoding=enc) as f:
    #     params = json.load(f)

    # conf = input()
    # print('conf:',conf)
    # params = json.loads(conf)
    # Input_imgpath = params['InputImgPath']
    # classes_path = params['classes_path']
    # anchors_path=params['anchors_path']
    # nms= params['nms_thresh']
    # score_threshold = float(params['score_threshold'])
    # weights_file = params['WeightFile'] if params["WeightFile"] else best
    # output_path = params['OutputPath']

    classes_path = r"D:\WorkSpace\Aircarft_oiltank\yolov5s\two_classes.txt"
    anchors_path = r"D:\WorkSpace\Aircarft_oiltank\yolov5s\yolo_anchors.txt"
    Input_imgpath = r"D:\WorkSpace\Bijie_landslide_dataset\qianxi.tif"
    # Input_imgpath = r"D:\WorkSpace\Bijie_landslide_dataset\landslide_object\valid\images\qxg088.png"
    # Input_imgpath = r"D:\WorkSpace\Bijie_landslide_dataset\test_image\wn029.png"
    nms = 0.5
    score_threshold = 0.6
    weights_file = r"D:\WorkSpace\Bijie_landslide_dataset\best.pt" #D:\WorkSpace\Bijie_landslide_dataset\best.pt
    output_path = r"D:\WorkSpace\Bijie_landslide_dataset"
    # Input_imgpath = cv2.imread(Input_imgpath)
    model = PredictModel()
    print(model.predict(Input_imgpath))
