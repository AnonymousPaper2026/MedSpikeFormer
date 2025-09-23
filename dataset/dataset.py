import random
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


from utils.micro import *


import tifffile as tiff
class Monu_Seg_Datasets(Dataset):
    def __init__(self,mode,proportion):
        super().__init__()
        cwd=os.getcwd()
        self.mode=mode
        middle_path=os.path.join('data','Monu_Seg','archive')
        
        gts_path=os.path.join(cwd,middle_path,'kmms_test','mask')
        images_path=os.path.join(cwd,middle_path,'kmms_test','image')

        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "png" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]

        self.data=[]
        for i in range(len(images_list)):
            image_path=os.path.join(images_path,images_list[i])
            mask_path=os.path.join(gts_path,gts_list[i])
            self.data.append([image_path, mask_path])
        
        gts_path_=os.path.join(cwd,middle_path,'kmms_training','mask')
        images_path_=os.path.join(cwd,middle_path,'kmms_training','image')
        
        images_list_=sorted(os.listdir(images_path_))
        images_list_ = [item for item in images_list_ if "tif" in item]
        gts_list_=sorted(os.listdir(gts_path_))
        gts_list_ = [item for item in gts_list_ if "png" in item]

        for i in range(len(images_list_)):
            image_path=os.path.join(images_path_,images_list_[i])
            mask_path=os.path.join(gts_path_,gts_list_[i])
            self.data.append([image_path, mask_path])
        
        random.shuffle(self.data)
        if mode==TRAIN:
            self.data=self.data[:int(proportion*len(self.data))]
        elif mode==TEST:
            self.data=self.data[int(proportion*len(self.data)):len(self.data)]

    def get_data(self):
        images = [item[0] for item in self.data]
        masks = [item[1] for item in self.data]
        return images, masks
