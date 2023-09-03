import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler
from Backend import ImageRestorationDataset, compressor
from Model import ComNet
from Patch import patch
import random
import numpy as np
import os
from datetime import datetime
import tqdm
import matplotlib.pyplot as plt

# Set up CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Set up model:
model = ComNet()
model.to(device)

# Loss function:
def mse(img1, img2):
    return torch.mean((img1 - img2) ** 2)
criterion = mse

#Epochs for data loader
epochs = 100

#set up datasets
training_data = ImageRestorationDataset(r'F:\ImageRestoration\Data\Dataset_SomeArtifacts\training')
validation_data = ImageRestorationDataset(r'F:\ImageRestoration\Data\Dataset_SomeArtifacts\validation')

training_params = {
        'batch_size': 1,
        'sampler': None,
        'num_workers': 0
        }

validation_params = {
        'batch_size': 1,
        'sampler': None,
        'num_workers': 0
        }

# Helpers
to_tensor = transforms.ToTensor()
to_PIL = transforms.ToPILImage()

root_folder = r'F:\ImageRestoration\runs'
base_folder = 'RestoreNet'
dir = os.path.join(root_folder, base_folder+datetime.now().strftime("%Y%m%dT%H%M%S"))
os.mkdir(dir)

for epoch in tqdm.tqdm(range(epochs),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=False):
    total_loss = 0.
    print('\n')
    #Initialize random sampler for new batch
    training_params['sampler'] = RandomSampler(training_data, num_samples=int(training_data.len/5))
    validation_params['sampler'] = RandomSampler(validation_data, num_samples=int(validation_data.len/5))

    #Initialize dataloader for new batch
    training_generator = DataLoader(training_data, **training_params)
    training_generator = DataLoader(training_data, **validation_params)

    if epoch < 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # for sample in generator
    for index, tensor in enumerate(training_generator):
        #return patches from train_tensor
        train_patch = patch(img=tensor, size=100, padding=10)
        running_loss = 0
        new_patches = []
        loss_patches = []
        for i in train_patch.patches:
            #pass through model
            x = compressor(i)
            loss_patches.append(x)
            i = i.to(device)
            x = x.to(device)
            x = model(x)
            x=torch.nn.functional.interpolate(x.unsqueeze(dim=0),size=(i.size(1),i.size(2)),mode='bilinear').squeeze(dim=0)
            new_patches.append(x)
            loss = criterion(x,i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        running_loss = running_loss/len(train_patch.patches)
        print('loss: {:.4f}, sample: {}/{}'.format(running_loss, index, len(training_generator)), end='\r', flush=True)

    total_loss += running_loss
    
    #save fig
    restored = train_patch.feather(new_patches)
    unrestored = train_patch.feather(loss_patches, unrestored = True)
    original = abs(train_patch.img - 1)
    fig, ax = plt.subplots(1,3, figsize=(20,15))
    ax[0].imshow(unrestored, cmap='gray')
    ax[0].set_title("Unrestored")
    ax[1].imshow(restored, cmap='gray')
    ax[1].set_title("Restored")
    ax[2].imshow(original, cmap='gray')
    ax[2].set_title("Original")
    fig.tight_layout()


    fig_name = 'epoch_{}_loss_{:.4f}'.format(epoch, total_loss/(epoch+1))
    path = os.path.join(dir, fig_name+'.png')
    plt.savefig(path)  # Specify 'cmap' if needed for color mapping
    plt.close()
    

