import json
import os
import numpy as np
import rasterio
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from segmentation_models_pytorch import DeepLabV3Plus
import segmentation_models_pytorch as smp
from datasets import TestDataset
from transforms import get_validation_augmentation,get_preprocessing

# set paramters
GPU_ID = 0
Batch_size = 1
Num_classes = 2  # 类别+背景
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

basedir = os.path.abspath(os.path.dirname(__file__))
# config = input()
config = basedir + 'traincfg.json'
params = json.load(config)
result_root = params["outputPath"]
model_file = params['weightFile']
data_root = params['inputData']


class Predict():
    def __init__(self):
        super(Predict, self).__init__()
        self.model =  DeepLabV3Plus(
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
        self.model.load_state_dict(torch.load(model_file, map_location=device))
        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.model = self.model.to(device)
        self.model.eval()
    def predict(self, input_data, result_root = None):
        """
        return predict result
        :param input_data:
        :param result_root:
        :return:
        """
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data).to(device)
            if len(input_data.shape) == 3:
                input_data = input_data.unsqueeze(0)
            input_data = input_data.float()
            output = self.model(input_data)[-1]
            _, output = torch.max(output, dim=0)
            output = output.data.cpu().numpy()
            if result_root == None:
                result_root = os.getcwd()
            result_root = os.path.join(result_root, "pred.jpg")
            cv2.imwrite(result_root, output)
            return result_root
        else:
            print("the input data is np.ndarray.")




if __name__ == "__main__":

    # data_root = r"/data1/zsc/features/1m_1/val/image"
    # result_root = r'/data1/zsc/features/result'
    # model_file = r"./models/best8.pth"
    # GPU_ID = 2

    data_root = r"E:\WorkSapce1\detect_water\sar_water\datas\trains\train\image\rgb-compose21_0_3584.tif"
    result_root = r"E:\WorkSapce1\detect_water\sar_water\datas\outs"
    model_file = r"E:\WorkSapce1\detect_water\sar_water\models\best_new.pth"
    # GPU_ID = 0
    # Batch_size = 1
    # Num_classes = 2  # 类别+背景
    data = cv2.imread(data_root)
    data = np.transpose(data, (2, 0, 1))
    model = Predict()
    model.predict(data, result_root)

"""
    os.makedirs(result_root) if not os.path.exists(result_root) else 0
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
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
        # aux_params={'classes':5,'dropout':0.1})

    # solution 1
    # model = torch.nn.DataParallel(model)
    # torch.backends.cudnn.benchmark = True
    # model.load_state_dict(torch.load(model_file, map_location='cpu'))

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    model.load_state_dict(torch.load(model_file, map_location='cpu'))

    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_file, map_location="cpu").items()})

    if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.to(device)

    test_dataset = TestDataset(data_root, transform=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn),img_suffix='tif')
    test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)


    model.eval()
    with torch.no_grad():
        for iter, (images, img_name) in enumerate(tqdm(test_loader, ascii=True)):
            images = images.to(device)
            outputs = model(images)[-1]
            _, outputs = torch.max(outputs, 0)
            outputs = outputs.data.cpu().numpy()
            # outputs = outputs.squeeze() * 255
            # print(outputs)
            # print(outputs.min())
            # print(outputs.max())

            # quit()
            # file_path = output_img + str(index_img).zfill(5)
            file_path = os.path.join(result_root, img_name[-1])
            cv2.imwrite(file_path, outputs)
            # break
"""