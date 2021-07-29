# -*- coding: utf-8 -*-
# author:liuxiaodong 
# contact: 2152550864@qq.com
# datetime:2021/7/16 15:07
# software: PyCharm
# python versions: Python3.7
# file: yolov5s
# license: (C)Copyright 2019-2021 liuxiaodong
from xml.dom.minidom import Document
import os
import cv2

def makexml(txtPath,xmlPath,picPath): #读取txt路径，xml保存路径，数据集图片所在路径
        files = os.listdir(txtPath)
        for i, name in enumerate(files):
            xmlBuilder = Document()
            annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
            xmlBuilder.appendChild(annotation)
            file_path = os.path.join(txtPath, name)
            txtFile=open(file_path)
            txtList = txtFile.readlines()[2:]
            # xmin, xmax, ymin, ymax
            xdata = list()
            ydata = list()
            for d in txtList:
                xdata.append(float(d.split(' ')[0]))
                ydata.append(float(d.split(' ')[1]))
            xdata = sorted(xdata)
            ydata = sorted(ydata)
            xydata = (xdata[0], xdata[-1], ydata[0], ydata[-1])
            img_name = name.replace(".txt", ".png")
            imgs_path = os.path.join(picPath, img_name)
            img = cv2.imread(imgs_path)
            Pheight,Pwidth,Pdepth=img.shape

            # for xmin, xmax, ymin, ymax in zip(*xydata):
            for i in range(1):
                xmin = xydata[0]
                ymin = xydata[2]
                xmax = xydata[1]
                ymax = xydata[3]

                # folder = xmlBuilder.createElement("folder")#folder标签
                # folderContent = xmlBuilder.createTextNode("landslide")
                # folder.appendChild(folderContent)
                # annotation.appendChild(folder)

                filename = xmlBuilder.createElement("filename")#filename标签
                filenameContent = xmlBuilder.createTextNode(img_name)
                filename.appendChild(filenameContent)
                annotation.appendChild(filename)

                source = xmlBuilder.createElement("source")
                sourceContent = xmlBuilder.createTextNode("whu")
                source.appendChild(sourceContent)
                annotation.appendChild(source)

                size = xmlBuilder.createElement("size")  # size标签
                width = xmlBuilder.createElement("width")  # size子标签width
                widthContent = xmlBuilder.createTextNode(str(Pwidth))
                width.appendChild(widthContent)
                size.appendChild(width)
                height = xmlBuilder.createElement("height")  # size子标签height
                heightContent = xmlBuilder.createTextNode(str(Pheight))
                height.appendChild(heightContent)
                size.appendChild(height)
                depth = xmlBuilder.createElement("depth")  # size子标签depth
                depthContent = xmlBuilder.createTextNode(str(Pdepth))
                depth.appendChild(depthContent)
                size.appendChild(depth)
                annotation.appendChild(size)

                segmented = xmlBuilder.createElement("segmented")
                segmentedContent = xmlBuilder.createTextNode("0")
                segmented.appendChild(segmentedContent)
                annotation.appendChild(segmented)

                object = xmlBuilder.createElement("object")
                picname = xmlBuilder.createElement("name")
                nameContent = xmlBuilder.createTextNode("landslide")
                picname.appendChild(nameContent)
                object.appendChild(picname)
                pose = xmlBuilder.createElement("pose")
                poseContent = xmlBuilder.createTextNode("Unspecified")
                pose.appendChild(poseContent)
                object.appendChild(pose)
                # truncated = xmlBuilder.createElement("truncated")
                # truncatedContent = xmlBuilder.createTextNode("0")
                # truncated.appendChild(truncatedContent)
                # object.appendChild(truncated)
                # difficult = xmlBuilder.createElement("difficult")
                # difficultContent = xmlBuilder.createTextNode("0")
                # difficult.appendChild(difficultContent)
                # object.appendChild(difficult)
                bndbox = xmlBuilder.createElement("bndbox")
                xxmin = xmlBuilder.createElement("xmin")
                # mathData=int(((float(oneline[0]))*Pwidth+1)-(float(oneline[1]))*0.5*Pwidth)
                # mathData = float(oneline[0])
                xminContent = xmlBuilder.createTextNode(str(xmin))
                xxmin.appendChild(xminContent)
                bndbox.appendChild(xxmin)
                yymin = xmlBuilder.createElement("ymin")
                # mathData = int(((float(oneline[2]))*Pheight+1)-(float(oneline[4]))*0.5*Pheight)
                yminContent = xmlBuilder.createTextNode(str(ymin))
                yymin.appendChild(yminContent)
                bndbox.appendChild(yymin)
                xxmax = xmlBuilder.createElement("xmax")
                # mathData = int(((float(oneline[1]))*Pwidth+1)+(float(oneline[3]))*0.5*Pwidth)
                xmaxContent = xmlBuilder.createTextNode(str(xmax))
                xxmax.appendChild(xmaxContent)
                bndbox.appendChild(xxmax)
                yymax = xmlBuilder.createElement("ymax")
                # mathData = int(((float(oneline[2]))*Pheight+1)+(float(oneline[4]))*0.5*Pheight)
                ymaxContent = xmlBuilder.createTextNode(str(ymax))
                yymax.appendChild(ymaxContent)
                bndbox.appendChild(yymax)
                object.appendChild(bndbox)
                annotation.appendChild(object)
            xml_file = os.path.join(xmlPath, img_name.replace(".png", ".xml"))
            f = open(xml_file, 'w')
            xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
            f.close()

txt_path = r'D:\WorkSpace\Bijie_landslide_dataset\Bijie-landslide-dataset\landslide\polygon_coordinate'
xml_path = r"D:\WorkSpace\Bijie_landslide_dataset\Bijie-landslide-dataset\landslide\polygon_coordinate_xml"
img_path = r"D:\WorkSpace\Bijie_landslide_dataset\Bijie-landslide-dataset\landslide\image"

if __name__ == '__main__':
    makexml(txt_path, xml_path, img_path)
    print("---success---")