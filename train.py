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
import Config

# Set up CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Get config params
config = Config.TrainConfig()
params = config.params

# Set up model:
model = ComNet()
model.to(device)

# Loss function:
def mse(img1, img2):
    return torch.mean((img1 - img2) ** 2)
criterion = mse

#Epochs for data loader
epochs = params['train']['epochs']

#set up datasets
training_data = ImageRestorationDataset(params['train']['train_dir'])
validation_data = ImageRestorationDataset(params['train']['train_dir'])

training_params = {
        'batch_size': params['train']['batch_size'],
        'sampler': params['train']['sampler'],
        'num_workers': params['train']['num_workers']
        }

validation_params = {
        'batch_size': params['train']['batch_size'],
        'sampler': params['train']['sampler'],
        'num_workers': params['train']['num_workers']
        }

# Helpers
to_tensor = transforms.ToTensor()
to_PIL = transforms.ToPILImage()

root_folder = params['out']['root_folder']  #Root folder for storing data
base_folder = params['out']['run_name'] #Run Name
root_dir = os.path.join(root_folder, base_folder+datetime.now().strftime("%Y%m%dT%H%M%S"))
img_dir = os.path.join(root_dir,'output','imgs')
val_dir = os.path.join(root_dir,'output','val')
model_dir = os.path.join(root_dir,'model')
print(img_dir)
os.makedirs(img_dir)
os.mkdir(val_dir)
os.mkdir(model_dir)

# Set up figure
training_loss = []
validation_loss = []
gen_fig, gen_ax = plt.subplots()
gen_ax.set_xlabel('Epoch')
gen_ax.set_ylabel('Loss')
gen_ax.set_title('Model Generalization')

#train model
for epoch in tqdm.tqdm(range(epochs),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=False):
    print('\nepoch: {}\n'.format(epoch))
    #Initialize random sampler for new batch
    training_params['sampler'] = RandomSampler(training_data, num_samples=int(training_data.len/200))
    validation_params['sampler'] = RandomSampler(validation_data, num_samples=int(validation_data.len/200))

    #Initialize dataloader for new batch
    training_generator = DataLoader(training_data, **training_params)
    validation_generator = DataLoader(training_data, **validation_params)

    if epoch < 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # train
    new_patches = []
    loss_patches = []
    train_patch = None
    model.train()
    for index, tensor in enumerate(training_generator):
        #return patches from train_tensor
        train_patch = patch(img=tensor, size=100, padding=10)
        running_loss = 0
        for i in train_patch.patches:
            #pass through model
            x = compressor(i)
            i = i.to(device)
            x = x.to(device)
            y = model(x)
            y=torch.nn.functional.interpolate(y.unsqueeze(dim=0),size=(i.size(1),i.size(2)),mode='bilinear').squeeze(dim=0)
            loss = criterion(y,i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()

            if (index == len(training_generator)-1):
                    loss_patches.append(x)
                    new_patches.append(y)

        running_loss = running_loss/len(train_patch.patches)
        print('training loss: {:.4f}, sample: {}/{}'.format(running_loss, index, len(training_generator)-1), end='\r', flush=True)

    training_loss.append(running_loss)
    print('\n')

    # validate
    model.eval()
    with torch.no_grad():
        for index, tensor in enumerate(validation_generator):
            val_patch = patch(img=tensor, size=100, padding=10)
            val_loss = 0
            for patch_num, i in enumerate(val_patch.patches):
                #pass through model
                x = compressor(i)
                i = i.to(device)
                x = x.to(device)
                y = model(x)
                y=torch.nn.functional.interpolate(y.unsqueeze(dim=0),size=(i.size(1),i.size(2)),mode='bilinear').squeeze(dim=0)
                loss = criterion(y,i)
                running_loss+=loss.item()

            running_loss = running_loss/len(val_patch.patches)
            print('validation loss: {:.4f}, sample: {}/{}'.format(running_loss, index, len(validation_generator)-1), end='\r', flush=True)

        validation_loss.append(running_loss)
        print('\n')
    
    #save img
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

    fig_name = 'epoch_{}_loss_{:.4f}'.format(epoch, training_loss[-1])
    path = os.path.join(img_dir, fig_name+'.png')
    plt.savefig(path)  # Specify 'cmap' if needed for color mapping
    plt.close(fig)

    #update generalization data
    gen_x = np.arange(0,len(training_loss),1)
    if (epoch == 0):
        #first epoch
        line1, = gen_ax.plot(gen_x, training_loss, label='train-loss', color='blue')
        line2, = gen_ax.plot(gen_x, validation_loss, label='val-loss', color='orange')
    else:
        #update data
        line1.set_data(gen_x, training_loss)
        
        line2.set_data(gen_x, validation_loss)
    gen_ax.legend()
    gen_ax.relim()
    gen_ax.autoscale_view()
    gen_fig.canvas.draw()
    plot_path = os.path.join(val_dir,'generalization.png')
    gen_fig.savefig(plot_path)
    #save model
    torch.save(model.state_dict(), os.path.join(model_dir,'model.pt'))
    

