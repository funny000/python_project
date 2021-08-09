import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import FocalLoss
from segmentation_models_pytorch import DeepLabV3Plus
import torchmetrics
from transforms import get_training_augmentation, get_validation_augmentation,get_preprocessing
from datasets import DeeplabDataset
import sys
import os

if __name__ == "__main__":
    # model_file = sys.argv[1]
    # data_root = sys.argv[2]
    # Total_Epoch = int(sys.argv[3])
    # Batch_size = int(sys.argv[4])
    # GPU_ID = sys.argv[5]

    # model_file = r"/data1/zsc/water_sar/models/best2.pth"
    # data_root = r"/data1/zsc/water_sar/trains1" 
    # Total_Epoch = 100
    # Batch_size = 8
    # GPU_ID = 1

    # data_root = r"E:\WorkSapce1\sar_water\datas\tests\rgb-compose21.tif"
    data_root = r"E:\WorkSapce1\sar_water\datas\trains\train"
    result_root = r"E:\WorkSapce1\sar_water\datas\outs"
    model_file = r"E:\WorkSapce1\sar_water\models\best.pth"
    Total_Epoch = 10
    Batch_size = 8
    GPU_ID = 1

    lr = 1e-2
    min_lr = 1e-4
    num_classes = 2  # 类别+背景
    log_file = os.path.splitext(model_file)[0] + '_log.csv'

    
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
        classes=num_classes,
        activation=None,
        upsampling=4, 
        aux_params=None)
    # model = smp.PAN(
    #     encoder_name=ENCODER, 
    #     encoder_weights=ENCODER_WEIGHTS, 
    #     # encoder_output_stride=16, 
    #     decoder_channels=32, 
    #     in_channels=3, 
    #     classes=num_classes, 
    #     activation=None, 
    #     upsampling=4, 
    #     aux_params=None)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = DeeplabDataset(data_root, 'train', transform=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn),img_suffix='tif')
    val_dataset = DeeplabDataset(data_root, 'val', transform=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn),img_suffix='tif')
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    epoch_size_train = len(train_dataset) // Batch_size
    epoch_size_val = len(val_dataset) // Batch_size

    if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.to(device)

    criterion = FocalLoss(mode='multiclass', alpha=None, gamma=2.0, ignore_index=None, reduction='mean', normalized=False, reduced_threshold=None)

    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    max_iter = epoch_size_train*Total_Epoch
    lambda1 = lambda e:((lr - min_lr) * ((1 - (e+1) / max_iter)**0.9) + min_lr) / lr
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)

    log = pd.DataFrame()
    val_best_iou = 0
    miou = torchmetrics.IoU(num_classes=num_classes).to(device=device)
    for epoch in tqdm(range(Total_Epoch), ascii=True):
        train_total_loss = 0
        val_total_loss = 0
        total_acc = 0

        model.train()
        for iter, (images, labels) in enumerate(tqdm(train_loader, ascii=True)):
            images = images.to(device)
            labels = labels.long().to(device).squeeze()
            optimizer.zero_grad()
            outputs = model(images)
            train_loss = criterion(outputs, labels)
            train_total_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()

        model.eval()
        for iter, (images, labels) in enumerate(tqdm(val_loader, ascii=True)):
            with torch.no_grad():
                images = images.to(device)
                labels = labels.long().to(device).squeeze()
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_total_loss += val_loss.item()
                _, outputs = torch.max(outputs, 1)
                iou = miou(labels, outputs)

        lr_scheduler.step()
        iou = miou.compute().data.cpu().numpy()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('Epoch:' + str(epoch + 1) + '/' + str(Total_Epoch))
        print('Lr: %.4e || Train Loss: %.4f || Val Loss: %.4f || mIoU: %.4f ' % (lr, train_total_loss / (epoch_size_train + 1), val_total_loss / (epoch_size_val + 1), iou))

        if val_best_iou <= iou:
            torch.save(model.state_dict(), model_file)
            val_best_iou = iou
            print('save the best model !')

        miou.reset()
        tmp_lr = pd.DataFrame([lr], columns=['lr'])
        tmp_train = pd.DataFrame([train_total_loss / (epoch_size_train + 1)], columns=['train_loss'])
        tmp_val = pd.DataFrame([val_total_loss / (epoch_size_val + 1)], columns=['val_loss'])
        tmp_iou = pd.DataFrame([iou], columns=['miou'])
        tmp = pd.concat([tmp_lr, tmp_train, tmp_val, tmp_iou], axis=1)
        log = log.append(tmp, ignore_index=True)
        log.to_csv(log_file, index=True)
