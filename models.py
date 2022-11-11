import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn, optim
import torch.nn.functional as F


# 掩码线性变换层
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
       super(MaskedLinear, self).__init__(in_features,out_features,bias)
       self.mask_flag = False
       self.mask = None

    # 设置掩码
    def set_mask(self,mask):
        weight_dev = self.weight.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = mask.data.cpu().numpy()
        # Apply new weight and mask
        self.weight.data = torch.from_numpy(tensor * mask).to(weight_dev)
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self,x):
        if self.mask_flag:
            weight  = self.weight * self.mask
            return F.linear(x,weight,self.bias)
        else:
            return F.linear(x,self.weight,self.bias)



# 四层简单线性网络
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet,self).__init__()
        self.fc1 = MaskedLinear(784, 256)
        self.fc2 = MaskedLinear(256, 128)
        self.fc3 = MaskedLinear(128, 64)
        self.fc4 = MaskedLinear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

    def prune(self,masks):
        self.fc1.set_mask(masks[0])
        self.fc2.set_mask(masks[1])
        self.fc3.set_mask(masks[2])
        self.fc4.set_mask(masks[3])