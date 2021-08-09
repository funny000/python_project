import json
import os
import cv2
import sys
# os.environ['PROJ_LIB']='/home/zsc/.conda/envs/torchseg/lib/python3.6/site-packages/pyproj/proj_dir/share/proj'
from os import path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import rasterio as rio
from glob import glob
from raster import RasterSampleDataset
from segmentation_models_pytorch import DeepLabV3Plus
import segmentation_models_pytorch as smp
from transforms import get_testing_augmentation,get_preprocessing
# from raster2shp_mp import ras2shp, preprocess

basedir = os.path.abspath(os.path.dirname(__file__))
# conf = input()
# conf = basedir + 'traincfg.json'
# params = json.load(conf)
# weight_file = params["weightFile"]
# data_root = params["InputImgPath"]
# save_path = params["OutputPath"]
# GPU_ID = params["GpuId"]


# set paramters
Band_index = [3,2,1]
isflip = False
batchSize = 1
SUFFIXS = [".tif", ".tiff", ".img", "jpg", ".png"]
GPU_ID = 0
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else "cpu")
Num_classes = 2  # 类别+背景
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'


class PredictModel():
    def __init__(self, weight_file, data_root, save_path):
        """
        initialization model
        :param weight_file:
        :param data_root:
        :param save_path:
        """
        super(PredictModel, self).__init__()
        self.data_root = data_root
        self.save_path = save_path
        self.weight_file = weight_file
        self.model = DeepLabV3Plus(encoder_name=ENCODER,
            encoder_depth=5,
            encoder_weights=ENCODER_WEIGHTS,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            in_channels=3,
            classes=Num_classes,
            activation=None,
            upsampling=4,
            aux_params=None)
        self.preprocess_def = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        self.model.load_state_dict(torch.load(self.weight_file, map_location=device))
        self.model.eval()
        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.model = self.model.to(device)
    def check_inputdata(self, data_root=None):
        """
        check input data
        :param data_root:
        :return:
        """
        image_list = list()
        if osp.isfile(self.data_root):
            if osp.splitext(self.data_root)[1] in SUFFIXS:
                image_list.append(self.data_root)
        elif osp.isdir(self.data_root):
            image_files = glob(self.data_root + '/*')
            for i in image_files:
                if osp.splitext(i)[1] in SUFFIXS:
                    image_list.append(i)
        return image_list
    def predict(self, input_data, device = device):
        """
        return the predict result
        :param input_data:
        :param weight_file:
        :return:
        """
        pp = list()
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)
            pred = self.model(input_data)
            _, result = torch.max(pred, 0)
            result = result.cpu().numpy()
            cv2.imwrite(self.save_path, result)
        elif osp.isdir(input_data) or osp.isfile(input_data):
            image_list = self.check_inputdata(input_data)
            for files in sorted(image_list):
                dataset = RasterSampleDataset(files, win_size=512, step_size=512, pad_size=256,
                                              band_index=Band_index,
                                              to_type="uint8",
                                              transform=None, # get_testing_augmentation()
                                              preprocessing=None) # preprocessing=get_preprocessing(self.preprocess_def)
                dataLoader = DataLoader(dataset, batch_size=batchSize, num_workers=0, shuffle=False, drop_last=False)
                out_file = osp.join(self.save_path, osp.split(files)[1])
                with rio.open(files) as src:
                    kwds = src.profile
                kwds.update(dict(count = 1, compress = 'lzw', interleave = 'pixel'))
                kwds.update({'dtype': 'uint8'})
                kwds.update({'nodata': 0})
                kwds.update({"crs": '+proj=latlong'})
                kwds.update({"transform": (1.0, 0.0, 0.0,0.0, -1.0, 0.0)})
                out_raster = rio.open(out_file, 'w+', **kwds)
                pbar = tqdm(dataLoader, ascii=True)
                for img, x_offset, y_offset in pbar:
                    img = img.to(torch.float)
                    if img.shape[0] == 1 and img.shape[1] != 3:
                        img = torch.transpose(img, dim0=1, dim1=3)
                    pbar.set_description(osp.split(files)[1]) # files
                    with torch.no_grad():
                        x_tensor = img.to(device)
                        pr_mask = self.model(x_tensor)[-1]
                        _, result1 = torch.max(pr_mask, 0)
                        result1 = result1.data.cpu().numpy()
                    if isflip:
                        result2 = self.model(torch.flip(img, [-1]))
                        result2 = torch.flip(result2, [-1])
                        result3 = self.model(torch.flip(img, [-2]))
                        result3 = torch.flip(result3, [-2])
                        result4 = self.model(torch.flip(img, [-1, -2]))
                        result4 = torch.flip(result4, [-1, -2])
                        pred = torch.mean(result1 + result2 + result3 + result4, dim = 1, keepdim = True)
                    else:
                        pred = result1
                        pp.append(pred[..., 256:768, 256:768].astype(np.uint8))
                        pred_label = pred[..., 256:768, 256:768].astype(np.uint8)
                    if (pred_label == 0).all():
                        continue
                    center_xsize, center_ysize = 512, 512
                    write_window = rio.windows.Window(x_offset.item(), y_offset.item(), center_xsize, center_ysize)
                    test = pred_label
                    print(test)
                    out_raster.write(pred_label, indexes = 1, window = write_window)
                print(len(pp))
                out_raster.close()


if __name__ == "__main__":
    # model_file = sys.argv[1]
    # data_root = sys.argv[2]
    # result_root = sys.argv[3]
    # GPU_ID = sys.argv[4]

    # basedir = os.path.abspath(os.path.dirname(__file__))
    # conf = input()
    # params = json.load(conf)
    # weight_file = params["weightFile"]
    # data_root = params["InputImgPath"]
    # save_path = params["OutputPath"]

    data_root = r"E:\WorkSapce1\detect_water\upload_sarwater\sar_water\datas\trains\train\image"
    data_root = r"E:\WorkSapce1\detect_water\upload_sarwater\sar_water\datas\tests\rgb-compose21.tif"
    data_root = r"E:\WorkSapce1\detect_water\beijing.tif"
    data_root = r"E:\WorkSapce1\GF3_MYN_FSII_026031_E113.3_N34.7_20210720_L1A_HHHV_L10000000013\GF3_MYN_FSII_026031_E113.3_N34.7_20210720_L1A_HHHV_L10000000013\GF3_MYN_FSII_026031_E113.3_N34.7_20210720_L1A_HV_L10000000013.tiff"
    data_root = r"E:\WorkSapce1\GF3_MYN_FSII_026031_E113.3_N34.7_20210720_L1A_HHHV_L10000000013\GF3_MYN_FSII_026031_E113.3_N34.7_20210720_L1A_HHHV_L10000000013\output\10_6010.tif"
    data_root = r"E:\WorkSapce1\detect_water\upload_sarwater\sar_water\10_6010_new2.tiff"
    result_root = r"E:\WorkSapce1\detect_water\upload_sarwater\sar_water\datas\outs"
    weight_file = r"E:\WorkSapce1\detect_water\upload_sarwater\sar_water\weight\best_new.pth"
    # model_file = r"E:\WorkSapce1\resnet50-19c8e357.pth"
    GPU_ID = 0


    # create_img = r"./10_6010_new2.tiff"
    # data = rio.open(data_root)
    # print(data)
    # print(data.indexes)
    # nd_data = np.array(data.read(1), dtype=np.dtype(np.float))
    # kwgs = {
    #     "driver": "GTiff",
    #     "dtype": nd_data.dtype,
    #     "width": 6010,
    #     "height": 6010,
    #     "count": 3,
    #     "crs": "+proj=latlong",
    #     "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    # }
    # with rio.open(create_img, "w", **kwgs) as src:
    #     for i in range(0, 3, 1):
    #         src.write(nd_data, i + 1)
    # src.close()
    # band1 = data.read(1)
    # print(band1)
    print(1)

    deepnet = PredictModel(weight_file, data_root, result_root)
    deepnet.predict(data_root)

"""
    Batch_size = 1
    Num_classes = 2  # 类别+背景
    Band_index = [3,2,1]
    isflip = False

    os.makedirs(result_root) if not os.path.exists(result_root) else 0
    SUFFIXS = ['.tif', '.tiff', '.img']
    img_list = []
    if osp.isfile(data_root):
        if osp.splitext(data_root)[1] in SUFFIXS:
            img_list.append(data_root)
    elif osp.isdir(data_root):
        img_files = glob(data_root + '/*')
        for i in img_files:
            if osp.splitext(i)[1] in SUFFIXS:
                img_list.append(i)

    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    model = DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_depth=5, 
        encoder_weights=ENCODER_WEIGHTS, 
        encoder_output_stride=16, 
        decoder_channels=256, 
        decoder_atrous_rates=(12, 24, 36), 
        in_channels=3, 
        classes=Num_classes, 
        activation=None, 
        upsampling=4, 
        aux_params=None)


    # solution 1
    # model = torch.nn.DataParallel(model)
    # torch.backends.cudnn.benchmark = True

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    model.load_state_dict(torch.load(weight_file, map_location = 'cpu')) #, map_location=None

    # solution 2
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_file, map_location = "cpu").items()})

    for img_file in sorted(img_list):
        ds = RasterSampleDataset(img_file,
                                win_size=512,
                                step_size=512,
                                pad_size=256,
                                band_index=Band_index,
                                to_type="uint8",
                                transform=get_testing_augmentation(),
                                preprocessing=get_preprocessing(preprocessing_fn))
        data_loader = DataLoader(ds,
                            batch_size=Batch_size,
                            num_workers=0,
                            shuffle=False,
                            drop_last=False)

        model.eval()
        if torch.cuda.is_available():
            # model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model = model.to(device)

        out_file = osp.join(result_root, osp.split(img_file)[1])

        with rio.open(img_file) as src:
            kwds = src.profile
            transforms = src.transform
            bounds = src.bounds
        del kwds['crs']
        kwds.update(dict(count=1, compress='lzw', interleave='pixel'))
        kwds.update({'dtype': 'uint8'})
        kwds.update({'nodata': 0})
        kwds.update({"crs":"+proj=latlong"})
        out_raster = rio.open(out_file, 'w+', **kwds)

        pbar = tqdm(data_loader, ascii=True)
        for img, x_offset, y_offset in pbar:
            pbar.set_description(osp.split(img_file)[1])
            with torch.no_grad():
                x_tensor = img.to(device)
                pr_mask = model(x_tensor)[-1]
                _, result1 = torch.max(pr_mask, 0)
                result1 = result1.data.cpu().numpy()
            if isflip:
                result2 = model(torch.flip(img, [-1]))  # horizontal flip
                result2 = torch.flip(result2, [-1])

                result3 = model(torch.flip(img, [-2]))  # vertical flip
                result3 = torch.flip(result3, [-2])

                result4 = model(torch.flip(img, [-1, -2]))  # diagonal flip
                result4 = torch.flip(result4, [-1, -2])

                pred = torch.mean(result1 + result2 + result3 + result4,
                                    dim=1,
                                    keepdim=True)
            else:
                pred = result1
                pred_label = pred[..., 256:768,256:768].astype(np.uint8)

            if (pred_label == 0).all():
                continue
            center_xsize,center_ysize=512,512
            write_window = rio.windows.Window(x_offset.item(),
                                                y_offset.item(), center_xsize, # x_offset.item(), y_offset.item()
                                                center_ysize)
            # pred_label = np.fliplr(pred_label)
            pred_label = np.flipud(pred_label)
            out_raster.write(pred_label, indexes=1, window=write_window)
        out_raster.close()
        # from collections import Counter
        # datas = out_raster.read(1)
        # datas = preprocess(datas, min_spot=MIN_SPOT)
        # out_raster.write(datas, indexes=1)
        # datas = datas.flatten()
        # total_nums = datas.size*4/5
        # cloud_nums = Counter(datas)[1] + Counter(datas)[2]
        # txtfile = os.path.join(result_root, os.path.basename(out_file)[:-4] + '.txt')
        # f = open(txtfile, "w+")
        # f.write('cloud_percent: {:.2%}'.format(cloud_nums/total_nums) + "\n")
        # f.close()
        # torch.cuda.empty_cache()
        # rasfile = out_file
        # shpfile = os.path.join(result_root, os.path.basename(rasfile)[:-4] + '.shp')
        # ras2shp(rasfile, shpfile)
        # break
"""


