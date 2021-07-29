# -*- coding: utf-8 -*-
# author:liuxiaodong 
# contact: 2152550864@qq.com
# datetime:2021/7/16 10:55
# software: PyCharm
# python versions: Python3.7
# file: yolov5s
# license: (C)Copyright 2019-2021 liuxiaodong
import os
import matplotlib.pyplot as plt
from osgeo import ogr

def plot_point(point,symbol='ko',**kwargs):
    x,y=point.GetX(),point.GetY()
    plt.plot(x,y,symbol,**kwargs)

def plot_line(line,symbol='g-',**kwargs):
    x,y=zip(*line.GetPoints())
    plt.plot(x,y,symbol,**kwargs)

def plot_polygon(poly,symbol='r-',**kwargs):
    for i in range(poly.GetGeometryCount()):
        subgeom=poly.GetGeometryRef(i)
        x,y=zip(*subgeom.GetPoints())
        plt.plot(x,y,symbol,**kwargs)

def plot_layer(filename,symbol,layer_index=0,**kwargs):
    ds=ogr.Open(filename)
    for row in ds.GetLayer(layer_index):
        geom=row.geometry()
        geom_type=geom.GetGeometryType()

        if geom_type==ogr.wkbPoint:
            plot_point(geom,symbol,**kwargs)
        elif geom_type==ogr.wkbMultiPoint:
            for i in range(geom.GetGeometryCount()):
                subgeom=geom.GetGeometryRef(i)
                plot_point(subgeom,symbol,**kwargs)

        elif geom_type==ogr.wkbLineString:
            plot_line(geom,symbol,**kwargs)
        elif geom_type==ogr.wkbMultiLineString:
            for i in range(geom.GetGeometryCount()):
                subgeom=geom.GetGeometryRef(i)
                plot_line(subgeom,symbol,**kwargs)

        elif geom_type == ogr.wkbPolygon:
            plot_polygon(geom,symbol,**kwargs)
        elif geom_type==ogr.wkbMultiPolygon:
            for i in range(geom.GetGeometryCount()):
                subgeom=geom.GetGeometryRef(i)
                plot_polygon(subgeom,symbol,**kwargs)

os.chdir(r'D:\WorkSpace\Aircarft_oiltank')
#下面三个谁在上边就先显示谁，我就按照点，线，面来了
plot_layer('out_beijing.shp','ko',markersize=5)
# plot_layer('省界.shp','r-')
# plot_layer('中国地图_投影.shp','g-',markersize=20)
plt.axis('equal')
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.show()