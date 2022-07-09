import torch
import os
import pandas as pd
import torch.nn as nn
from data import MyDataSet
from model import LeNet
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def train(model, train_ds, eval_ds, optimizer, criterion, epoch):
    model.train()
    total_num = len(train_ds.dataset)
    train_loss = 0
    correct_num_train  = 0

    for image, label in train_ds:
        image = image.to(device)
        label = label.to(device)
        # Convert the tag from int32 type to long type, otherwise the calculation loss will report an error
        label = label.to(torch.long)

        output = model(image)
        loss = criterion(output, label)
        train_loss += loss.item() * label.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predict = torch.argmax(output, dim=-1)
        correct_num_train += label.eq(predict).sum()

    train_loss = train_loss / total_num
    train_acc = correct_num_train / total_num
    print('epoch: {} --> train_loss: {:.6f} - train_acc: {:.6f} - '.format(
        epoch, train_loss, train_acc), end='')


    model.eval()

    total_num = len(eval_ds.dataset)
    eval_loss = 0
    correct_num_valid = 0

    for image, label in eval_ds:
        image = image.to(device)
        label = label.to(device)
        label = label.to(torch.long)
        
        output = model(image)
        loss = criterion(output, label)
        eval_loss += loss.item() * label.size(0)

        predict = torch.argmax(output, dim=-1)
        correct_num_valid += label.eq(predict).sum()
    
    eval_loss = eval_loss / total_num
    eval_acc = correct_num_valid / total_num
    
    print('valid_loss: {:.6f} - valid_acc: {:.6f}'.format(
         eval_loss, eval_acc))

    torch.save(model.state_dict(), 'models/model.ckpt')


if __name__ == '__main__':


    path = '--filepath consisting train, test, and ** files--'
    test_path = path+'Test.csv'
    train_path = path+'Train.csv'
    image_path = path + 'Images'
    train_file = pd.read_csv(train_path)
    test_file = pd.read_csv(test_path)
    test_names = test_file['Image_id'].values
    train_names = dict(train_file.to_numpy())

    test_image_paths = []
    training_image_paths = []

    for path in os.listdir(image_path):
        if path in test_names:
            test_image_paths.append(os.path.join(image_path, path))
        else:
            training_image_paths.append(os.path.join(image_path, path))


    image_path = path+'Images/'
    epochs = 100
    batch_size = 32
    lr = 0.001
    image_size = 256

    model = LeNet()

    transformations = transforms.Compose(
        [
        transforms.Resize((image_size,image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
        ]
    )

    train_image_paths = training_image_paths[:1350]
    valid_image_paths = training_image_paths[1350:]


    train_ds  = MyDataSet(train_names, train_image_paths, transformations)
    valid_ds = MyDataSet(train_names, valid_image_paths, transformations)

    train_dataloader = DataLoader(train_ds, batch_size = batch_size)
    valid_dataloader = DataLoader(valid_ds, batch_size = batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    for i in range(epochs):
        train(model, train_dataloader, valid_dataloader, optimizer, criterion, i)