import math
import PIL
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

class patch:
    def __init__(self,img,size,padding):
        self.img = np.abs(img.squeeze().numpy() -1) # tensor to np then cvt to inverted float.
        self._size = size
        self.padding = padding
        self._row_size_max = math.ceil(self.img.shape[0]/math.ceil(self.img.shape[0]/self._size))
        self._col_size_max = math.ceil(self.img.shape[1]/math.ceil(self.img.shape[1]/self._size))
        self.num_rows = math.ceil(self.img.shape[0]/self._row_size_max)
        self.num_cols = math.ceil(self.img.shape[1]/self._col_size_max)
        self.patches = self.patch()
    
    def patch(self):
        patches = []
        to_tensor = transforms.ToTensor()
        #store data in row-major (c) order
        for i in np.arange(self.num_rows):
            for j in np.arange(self.num_cols):
                i_min = self.constrain(i*self._row_size_max, 0, self.img.shape[0]-1)
                j_min = self.constrain(j*self._col_size_max, 0, self.img.shape[1]-1)
                i_max = self.constrain(i*self._row_size_max+self._row_size_max+self.padding, 0, self.img.shape[0]-1)
                j_max = self.constrain(j*self._col_size_max+self._col_size_max+self.padding, 0, self.img.shape[1]-1)
                new_img = self.img[i_min:i_max,j_min:j_max]
                patches.append(to_tensor(new_img))
        return patches

    def constrain(self, value, min_val, max_val):
        return (min(max(min_val, value),max_val))


    def feather(self, new_patches, unrestored=False):
        if unrestored:
            padding = int(self.padding/2)
            typ_ht = int(self._row_size_max/2)
            typ_wd = int(self._col_size_max/2)
            canvas = np.full((int(self.img.shape[0]/2), int(self.img.shape[1]/2)),0.)
        else:
            padding = self.padding
            typ_ht = self._row_size_max
            typ_wd = self._col_size_max
            canvas = np.full((int(self.img.shape[0]), int(self.img.shape[1])),0.)

        #create default edge and corner masks:
        edge_feather = np.expand_dims(np.linspace(start=0,stop=1,num=padding), axis=1)
        corner_feather = np.full((padding,padding),0.25)

        for i in np.arange(self.num_rows):
            for j in np.arange(self.num_cols):
                upper_left=True
                upper_right=True
                lower_left=True
                lower_right=True
                #create mask. use linspace to feather edges. Corners are averaged.
                index = i*self.num_cols+j
        
                patch = new_patches[index].squeeze().cpu().detach().numpy()

                patch_mask = np.full((patch.shape[0],patch.shape[1]),1.)
    
                if not i == 0:   #top pad present
                    patch_mask[:padding,:] = np.tile(edge_feather,(1,patch.shape[1]))
                else:
                    upper_left = False
                    upper_right = False

                if not i == self.num_rows-1:    #lower pad present
                    patch_mask[-padding:,:] = np.tile(np.flip(edge_feather),(1,patch.shape[1]))
                else:
                    lower_left = False
                    lower_right = False

                if not j == 0:  #left present
                    patch_mask[:,:padding] = np.tile(np.transpose(edge_feather),(patch.shape[0],1))
                else:
                    upper_left = False
                    lower_left = False

                if not j == self.num_cols-1:    #right pad present
                    patch_mask[:,-padding:] = np.tile(np.transpose(np.flip(edge_feather)),(patch.shape[0],1))
                else:
                    upper_right = False
                    lower_right = False

                if upper_left:
                    patch_mask[0:padding,0:padding] = corner_feather
                if upper_right:
                    patch_mask[:padding,-padding:] = corner_feather
                if lower_left:
                    patch_mask[-padding:,:padding] = corner_feather
                if lower_right:
                    patch_mask[-padding:,-padding:] = corner_feather

                new_patch = patch*patch_mask

                i_min = self.constrain(i*typ_ht, 0, canvas.shape[0]-1)
                j_min = self.constrain(j*typ_wd, 0, canvas.shape[1]-1)
                i_max = self.constrain(i*typ_ht + patch.shape[0], 0, canvas.shape[0])
                j_max = self.constrain(j*typ_wd + patch.shape[1], 0, canvas.shape[1])

                #mask_canvas[i_min:i_max,j_min:j_max] += patch_mask
                canvas[i_min:i_max,j_min:j_max] += new_patch

        #restore to PIL
        image = (abs(canvas-1.)*255.).astype(np.uint8)
        return image
