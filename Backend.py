from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from torchvision import transforms
import io
import numpy as np
from PIL import Image
import random
import os

#Set up dataset generator
class ImageRestorationDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path

        legal_files = ['.jpg','.JPG','png','PNG','tif']
        self.img_list = [os.path.join(root,i).split(root_path)[-1] for root, dirs, files in os.walk(root_path)
                         for i in files
                         if any(x in i for x in legal_files)
                         ]
        self.len = len(self.img_list)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        filepath = os.path.join(self.root_path,self.img_list[idx][1:])
        img = Image.open(filepath)
        #cvt to grayscale
        img = self.to_gray(img)
        #random rotation
        img = self.random_rotate(img)
        #random crop
        img = self.random_crop(img)
        #random mirror
        img = self.random_mirror(img)
        to_tensor = transforms.ToTensor()
        tensor = to_tensor(img)
        return tensor
    
    def to_gray(self, img):
        if not img.mode == 'L':
            img = img.convert('L')
        return img
    
    def random_mirror(self, img):
        if not random.randint(0,1):
            img = img.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        if not random.randint(0,1):
            img = img.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
        return img
    
    def random_rotate(self, img):
        randint = random.randint(0,3)
        if not randint == 0:
            img = img.rotate(90*randint)
        return img  

    def random_crop(self, img):
        wd, ht = img.size
        new_wd = random.randint(int(wd/6),wd)
        new_ht = random.randint(int(ht/6), ht)
        wd_start = random.randint(0, wd-new_wd)
        ht_start = random.randint(0, ht-new_ht)
        wd_end = wd_start+new_wd
        ht_end = ht_start+new_ht
        return img.crop(box=(wd_start, ht_start, wd_end, ht_end))
    
def compressor(tensor):
    to_tensor = transforms.ToTensor()
    to_PIL = transforms.ToPILImage()
    img = to_PIL(tensor)
    wd,ht = img.size
    img = img.resize((int(wd/2),int(ht/2)),Image.Resampling.BILINEAR)
    for i in range(int(max(0,np.random.normal(3,2)))):
        with  io.BytesIO() as buffer:
            img.save(buffer, format='JPEG', quality=20)
            img = Image.open(io.BytesIO(buffer.getvalue()))
    return to_tensor(img)