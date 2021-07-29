# -*- coding: utf-8 -*-
# author:liuxiaodong 
# contact: 2152550864@qq.com
# datetime:2021/7/16 11:17
# software: PyCharm
# python versions: Python3.7
# file: yolov5s
# license: (C)Copyright 2019-2021 liuxiaodong
from osgeo import gdal
import numpy as np


def read_tiff(inpath):
    ds = gdal.Open(inpath)
    row = ds.RasterXSize
    col = ds.RasterYSize
    band = ds.RasterCount
    geoTransform = ds.GetTransform()
    proj = ds.GetTransform()
    data = np.zeros([row, col, band])
    img = ds.ReadAsArray(0, 0, col, row)
    # for i in range(band):
    #     dt = ds.GetRasterBand(1)
    #     data[:, :, i] = dt.ReadAsArray(0, 0, col, row)
    print(img)
    return img


# def array2raster(outpath, array, geoTransform, proj):
#     cols = array.shape[1]
#     rows = array.shape[0]
#     driver = gdal.GetDriverByName('Gtiff')
#     outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
#     outRaster.SetGeoTransform(geoTransform)  # 参数2,6为水平垂直分辨率，参数3,5表示图片是指北的
#     outband = outRaster.GetRasterBand(1)
#     outband.WriteArray(array)
#     outRaster.SetProjection(proj)  # 将几何对象的数据导出为wkt格式
#     outRaster.FlushCache()


if __name__ == "__main__":
    Input_imgpath = r"D:\Program0\fqiu\feiq\Recv Files\beijing.tif"
    # data, geoTransform, proj = read_tiff('d:/a.tif')
    img_data = read_tiff(Input_imgpath)

    # array2raster("d:/b.tif", np.zeros[2400, 2400], geoTransform, proj)
