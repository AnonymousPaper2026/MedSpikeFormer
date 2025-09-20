import numpy as np
from torch.utils.data import DataLoader
from utils.utils import print_and_save, shuffling,get_transform
from dataset.dataset import OCDC_Datasets
from torch.utils.data import Dataset, DataLoader
import cv2
from dataset.DATASET import DATASET

from utils.micro import ISIC,Kvasir,Monu,COVID,BUSI,TEST,TRAIN 


from dataset.dataset import *
def get_loader(datasets,batch_size,image_size,train_log_path):
    
    if datasets==Monu:
        train_data=Monu_Seg_Datasets(mode=TRAIN)
        test_data=Monu_Seg_Datasets(mode=TEST)
    
    (train_x, train_y) = train_data.get_data()
    (valid_x, valid_y) = test_data.get_data()
    
    train_x, train_y = shuffling(train_x, train_y)
    
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    
    print_and_save(train_log_path, data_str)

    train_dataset = DATASET(train_x, train_y, (image_size, image_size), transform=get_transform())
    valid_dataset = DATASET(valid_x, valid_y, (image_size, image_size), transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader,valid_loader


