import os
import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from utils import OmniglotDataset, NWayOneShotEvalSet, split_drawers, split_alphabets
import matplotlib.pyplot as plt
from torchvision import transforms
from collections import defaultdict
from model import SiameseNN


random_seed = 0
torch.manual_seed(random_seed)

ROOT_DIR = './dataset/images_background/'
idx_drawers = split_drawers()
idx_alps = split_alphabets()


if torch.cuda.is_available():
    device = torch.device("cuda")  



    #for idx, (img1, img2, label) in enumerate(trainLoader):
    #    if idx <2:
    #        print(label[0])
    #        plt.subplot(2,2,1)
    #        plt.imshow(img1[0][0])
    #        plt.subplot(2,2,2)
    #        plt.imshow(img2[0][0])
    #        plt.show()
    #    else:
    #        break

    #evalset= NWayOneShotEvalSet(root_dir='./dataset/images_evaluation/', alphabets=idx_alps['valid'], drawers=idx_drawers['valid'], numWay=20, numTrials = 20, transform=None)

def oneshot_eval(eval_dataloader, model):
    #eval_dataloader  = DataLoader(eval_dataset, batch_size = 128)    

    total = len(eval_dataloader.dataset)
    TP = torch.tensor(0, dtype=torch.int64)
    epoch_iterator = tqdm(eval_dataloader, desc="Validation Iteration")
    for step, batch in enumerate(epoch_iterator):
        with torch.no_grad():
            label = batch[2].to(device)
            
            scores = torch.tensor([], dtype=torch.float32).to(device)
            for i in range(eval_dataset.numWay):
                outputs = model(batch[0].to(device), batch[1][i].to(device))
                scores = torch.cat((scores, 
                                    outputs.view(1, outputs.size()[0])))
            predicted = torch.argmax(scores, dim=0).type(label.dtype)
            matched = torch.sum(torch.eq(predicted, 
                                         label.view(label.size()[0])))
            TP += matched
    epoch_iterator.close()
    acc = int(TP) / total * 100

    print(f"\n \n Validation" + " Accuracy : {:.5} / 100.00 \n".format(acc))
    
    return acc

def train(model, train_loader, val_loader, num_epochs):
    log_step = 10
    train_losses = []
    val_losses = []
    best_val = {'epoch' : 0, 'acc' : 0.0}
    cur_step = 0
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 3e-2, betas = (.99, .999), weight_decay=.05)
    train_iterator = trange(0, int(num_epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        running_loss = 0.0
        model.train() #model.train 위치???
        print("Starting epoch " + str(epoch+1))
        for step, (img1, img2, labels) in enumerate(epoch_iterator):
            
            # Forward
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        if (step % log_step) == (log_step - 1):
            print(" - ")
            print('\n [epoch %d , iteration :%5d] Loss = %.5f' %
                        (epoch + 1, step + 1, running_loss / log_step))



        #check validation loss after every epoch
        val_acc = oneshot_eval(val_loader, model)
        # with torch.no_grad():
        #     model.eval()
        #     for img1, img2, labels in val_loader:
        #         img1 = img1.to(device)
        #         img2 = img2.to(device)
        #         labels = labels.to(device)
        #         outputs = model(img1, img2)
        #         loss = criterion(outputs, labels)
        #         val_running_loss += loss.item()
        # avg_val_loss = val_running_loss / len(val_loader)
        # val_losses.append(avg_val_loss)
        # print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
        #     .format(epoch+1, num_epochs, avg_train_loss, avg_val_loss))
    print("Finished Training")  
    #return train_losses, val_losses  


if __name__ == '__main__':
    #transformations = transforms.Compose([transforms.ToTensor()]) 
    trainset = OmniglotDataset(root_dir=ROOT_DIR, drawers= idx_drawers['eval'], size = 100, transform=transforms.ToTensor())
    trainLoader = DataLoader(trainset, batch_size = 32)
    valset = NWayOneShotEvalSet(root_dir='./dataset/images_evaluation/', idx_alps=idx_alps['valid'], \
                                idx_drawers=idx_drawers['valid'], numWay=20, numTrials = 20, transform=transforms.ToTensor())
    valLoader = DataLoader(valset, batch_size = 32)
    model = SiameseNN().to(device)
    model.apply(SiameseNN.init_weights)
    train(model, trainLoader, valLoader, 20)