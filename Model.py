import torch
import torch.nn as nn

if torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device('cpu')

# RestoreNet
class ComNet(nn.Module):
    def __init__(self):
        super(ComNet, self).__init__()   
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv1_1 = nn.Conv2d(1, 6, kernel_size=3, dilation=1, stride=1, padding=1, padding_mode='reflect')
        self.conv1_2 = nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=1, padding=2, padding_mode='reflect')
        self.conv1_3 = nn.Conv2d(1, 1, kernel_size=3, dilation=3, stride=1, padding=3, padding_mode='reflect')
        self.conv1_4 = nn.Conv2d(1, 1, kernel_size=3, dilation=4, stride=1, padding=4, padding_mode='reflect')
        self.conv1_5 = nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        #self.attnConv_1 = nn.Conv2d(9, 1, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(9, 18, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2_2 = nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        self.conv3_1 = nn.Conv2d(18, 36, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv3_2 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        self.upconv3 = nn.ConvTranspose2d(36, 18, kernel_size=2, stride=2, padding=0) #cat 2 conv3_2

        self.conv4_1 = nn.Conv2d(36, 18, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv4_2 = nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        self.upconv2 = nn.ConvTranspose2d(18, 9, kernel_size=2, stride=2, padding=0) # cat w conv2_2

        self.conv5_1 = nn.Conv2d(18, 9, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv5_2 = nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        self.upconv1 = nn.ConvTranspose2d(9, 3, kernel_size=2, stride=2, padding=0) # cat w conv1_4

        self.attnConv_2 = nn.Conv2d(18, 1, kernel_size=9, stride=1, padding=4, padding_mode='reflect')

        self.conv6_1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv6_2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv6_3 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)   

        # Initialize the weights of the model using He
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x=x.to(device)
        
        out = self.conv1_1(x)
        out1_1 = nn.LeakyReLU()(out)
        out = self.conv1_2(x)
        out1_2 = nn.LeakyReLU()(out)
        out = self.conv1_3(x)
        out1_3 = nn.LeakyReLU()(out)
        out = self.conv1_4(x)
        out1_4 = nn.LeakyReLU()(out)
        out = torch.cat((out1_1, out1_2, out1_3, out1_4), dim=0)
       
        '''attn = self.attnConv_1(out)
        attn = nn.Softmax(dim=1)(attn)
        out = out*attn'''

        out = self.conv1_5(out)
        out1 = nn.LeakyReLU()(out)

        out = self.maxpool(out1)

        out = self.conv2_1(out)
        out = nn.LeakyReLU()(out)
        out = self.conv2_2(out)
        out2 = nn.LeakyReLU()(out)
        out = self.maxpool(out2)

        out = self.conv3_1(out)
        out = nn.LeakyReLU()(out)
        out = self.conv3_2(out)
        out = nn.LeakyReLU()(out)
        
        out = self.upconv3(out)
        out = nn.LeakyReLU()(out)
        out=torch.nn.functional.interpolate(out.unsqueeze(dim=0),size=(out2.size(1),out2.size(2)),mode='bilinear').squeeze(dim=0)
        out = torch.cat((out2,out), dim=0)

        out = self.conv4_1(out)
        out = nn.LeakyReLU()(out)
        out = self.conv4_2(out)
        out = nn.LeakyReLU()(out)

        out = self.upconv2(out)
        out = nn.LeakyReLU()(out)
        out=torch.nn.functional.interpolate(out.unsqueeze(dim=0),size=(out1.size(1),out1.size(2)),mode='bilinear').squeeze(dim=0)
        out = torch.cat((out1,out), dim=0)

        '''attn = self.attnConv_2(out)
        attn = nn.Softmax(dim=1)(attn)
        out = out*attn'''

        out = self.conv5_1(out)
        out = nn.LeakyReLU()(out)
        out = self.conv5_2(out)
        out = nn.LeakyReLU()(out)

        out = self.upconv1(out)
        out = nn.LeakyReLU()(out)

        out = self.conv6_1(out)
        out = nn.LeakyReLU()(out)
        out = self.conv6_2(out)
        out = nn.LeakyReLU()(out)
        out = self.conv6_3(out)
        out = nn.ReLU()(out)
        return out
    