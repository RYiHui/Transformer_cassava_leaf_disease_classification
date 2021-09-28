import torch
import torch.nn as nn
import timm


class ViTBase16(nn.Module):
    def __init__(self,n_classes,pretrained=True):
        super(ViTBase16,self).__init__()
        self.model=timm.create_model("vit_base_patch16_224",pretrained=True)  #保存参数
        self.model.head=nn.Linear(self.model.head.in_features,n_classes)
        # 修改最后的全连接层  输入维度为768  输出维度自定

    def forward(self,x):
        x=self.model(x)
        return x

    def train_one_epoch(self, train_loader, criterion, optimizer, device):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        self.model.train()  #将BN和Dropout置为True
        for i,(data,target) in enumerate(train_loader):
            if device.type == 'cuda':
                data,target=data.cuda(),target.cuda()
            optimizer.zero_grad()
            output = self.forward(data)
            loss=criterion(output,target)
            loss.backward()
            accuracy = (output.argmax(dim=1)==target).float().mean()
            epoch_loss+=loss
            epoch_accuracy+=accuracy
            optimizer.step()
            print(f"\tBATCH{i+1}/{len(train_loader)} - LOSS:{loss}")
        return epoch_loss/len(train_loader),epoch_accuracy/len(train_loader)

    def valid_one_epoch(self,valid_loader,criterion,device):
        valid_loss = 0.0
        valid_accuracy = 0.0
        self.model.eval()  #将BN和Droout置为False
        for i,(data,target) in enumerate(valid_loader):
            if device.type == 'cuda':
                data,target = data.cuda(),target.cuda()

            with torch.no_grad(): # 被该语句wrap 起来的部分不会track梯度
                output = self.model(data)
                loss = criterion(output,target)

                accuracy = (output.argmax(dim=1) == target).float().mean()
                valid_loss+=loss
                valid_accuracy+=accuracy
        return valid_loss/len(valid_loader),valid_accuracy/len(valid_loader)
