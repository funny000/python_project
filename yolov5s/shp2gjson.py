# -*- coding: utf-8 -*-
# author:liuxiaodong 
# contact: 2152550864@qq.com
# datetime:2021/7/27 10:06
# software: PyCharm
# python versions: Python3.7
# file: yolov5s
# license: (C)Copyright 2019-2021 liuxiaodong

from tqdm import tqdm
import json
import codecs
import shapefile



def Shp2JSON(filename,shp_encoding='utf-8',json_encoding='utf-8'):
    '''
    这个函数用于将shp文件转换为GeoJSON文件
    :param filename: shp文件对应的文件名（去除文件拓展名）
    :return:
    '''
    '''创建shp IO连接'''
    reader = shapefile.Reader(filename, encoding=shp_encoding)
    '''提取所有field部分内容'''
    fields = reader.fields[1:]
    '''提取所有field的名称'''
    field_names = [field[0] for field in fields]
    '''初始化要素列表'''
    buffer = []
    for sr in tqdm(reader.shapeRecords()):
        '''提取每一个矢量对象对应的属性值'''
        record = sr.record
        '''属性转换为列表'''
        record = [r.decode('gb2312','ignore') if isinstance(r, bytes)
                  else r for r in record]
        '''对齐属性与对应数值的键值对'''
        atr = dict(zip(field_names, record))
        '''获取当前矢量对象的类型及矢量信息'''
        geom = sr.shape.__geo_interface__
        '''向要素列表追加新对象'''
        buffer.append(dict(type="Feature",
                           geometry=geom,
                           properties=atr))
    '''写出GeoJSON文件'''
    geojson = codecs.open(filename + "-geo.json","w", encoding=json_encoding)
    geojson.write(json.dumps({"type":"FeatureCollection",
                              "features":buffer}) + '\n')
    geojson.close()
    print('转换成功！')

if __name__ == '__main__':
    shpfile = r"D:\WorkSpace\Aircarft_oiltank\out_beijing"
    outfile = r"D:\WorkSpace\Aircarft_oiltank\out_beijing.geojeon"
    Shp2JSON(shpfile)