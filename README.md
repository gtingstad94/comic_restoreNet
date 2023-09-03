# comic_restoreNet
Comic RestoreNET is a work-in-progress deep learning model for restoring compressed comics and generating higher quality images with super-resolution algorithms.

## Background:
Images come in many different formats, each tailored for specific goals. A few common image formats include PNG, TIFF, and JPEG. For applications which demand that source images maintain high-quality resolution, formats that support lossless compression, such as PNG, are used. Lossless formats, while nice to look at, may come at very high filesize costs. Since most images aren't examined at the pixel level, lossy compression formats, like JPEG, are often favored for their space-saving capabilities; in fact, most operating systems save images in this format by default.

Most observers will not notice a loss in detail from a single lossy compression of high-fidelity images, and such compression formats work very well when used in this manner. A common problem arises, however, as compressed images are copied, shared, or compressed with a high compression factor. As most of us have probably noticed, the more times a meme is screenshotted and shared, the more awful it begins to look. In the case of image files, particularly those containing text, even a few lossy compressions can destroy the fine features that make the text legible.

## The Project
There are a two major goals to RestoreNET. The priority of RestoreNET is to remove artifacts and recover destroyed details from repeated lossy compressions, with a particular focus on fine text details. A second goal of the project is develop a successful super-resolution method for upscaling previously resized images.

## Methodology
My current approach to these goals involves splitting an image into small overlapping patches and feeding them through a custom convolutional neural network one at a time. On the other end of the network, the image patches are feathered back into their original positions, with overlapping regions contributing a weighted average towards the corresponding pixel location in the final image. 

### Patch Processing
The rational for my patch processing method is threefold:
1. Feeding small image patches to the neural network allows for better memory management during training and inference, allowing for more complicated model architecture and enabling it to run on less powerful machines.
2. Chunking the image into smaller patches allows for batch processing. While it sounds unintuitive due to the overhead of splitting and reconstructing the image, this method can significantly speed up the model during both training and inference, as it allows the model to process images on multiple GPU or CPU cores.
3. **Most importantly**, the patch processing model enables us to achieve significantly better reconstruction performance over the alternative. I am currently designing an attention mechanisim through which the model will be able to dynamically adjust the weighted contribution of convolutional channels. This means that once the method is implemented, the network will be able to dynamically process each patch differently depending on the contents of the image. For example, the model may chose to employ one strategy over another depending on whether the current patch contains text on white background or artwork. In the end, I hope this methodology will make the model less static and improve reconstruction performance

### The Model
The base model is a convolutional neural network (covnet) based on UNet architecture. Using a combination of downsampling layers, transpose convolution layers, and skip connections, UNet has proven to be very good at learning from non-local features in images. Moreover, transpose convolutions have shown some promising results for image upscaling. 

My model uses a lightweight architecture inspired by UNet. It contains two Maxpool downsampling layers and subsequent transpose convolution layers with skip connections in between. Skip connections exist to ensure that information is not lost during downsampling. Currently, a very simple attention mechanism is employed that examines the channels after the final skip connection and adjusts the weighted contribution of each channel before a final transpose convolution is performed. This final upsampling layer is used to resize the image to a scale two times larger than the input. A final set of convolutional layers are used to fill in any gaps left by the final transpose convolution.

## Model Usage
**Note** that this project is still in progress, and therefore some features are unpolished. Use at your own peril.

At the time of writing, my code is only set up to train a model and output sample images and performance characteristics on each epoch of training. I do not yet have an inference script written, nor does my model save a a copy of itself once training completes. These, and other quality of life improvements, are things I intend to implement as I work on the project.

### Training
Trainig is 

## Note
This project is very much still in progress 
