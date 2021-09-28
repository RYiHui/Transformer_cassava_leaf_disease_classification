from torch import save

import Global_Variable as gl
import numpy as np
import gc


def fit_gpu(model, epochs, device, criterion, optimizer, train_loader, valid_loader=None):
    valid_loss_min = np.Inf
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    for epoch in  range(1,epochs+1):  #调用数据和模型进行训练-Log
        gc.collect()  #通过gc清理内存
        print(f"{'='*50}")
        print(f"EPOCH{epoch}-TRAINING...")

        train_loss,train_acc=model.train_one_epoch(train_loader,criterion,optimizer,device)
        print(f"\n\t[TRAIN] EPOCH{epoch}-LOSS:{train_loss},ACCURACY:{train_acc}\n")
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        gc.collect()

        #valid
        if valid_loader is not None:
            gc.collect()
            print(f"EPOCH{epoch}-VALIDATING...")
            valid_loss,valid_acc = model.valid_one_epoch(valid_loader,criterion,device)
            print(f"\t[VALID] LOSS:{valid_loss},ACCURACY:{valid_acc}\n")
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            gc.collect()
            #save !!!
            if valid_loss<=valid_loss_min and epoch!=1:
                print("Validation loss decreased ({:.4f} -->{:.4f}). Saving model...".format(valid_loss_min,valid_loss))
                save(model.state_dict(),'best_model.pth')
                valid_loss_min=valid_loss
    return {
        "train_loss":train_losses,
        "valid_losses":valid_losses,
        "train_acc":train_accs,
        "valid_acces":valid_accs,
    }