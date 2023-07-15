import torch, os, random, argparse
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
from dataset_load import create_dataLoader
from Swin_pytorch import SwinTransformer

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--epoch', type=int, help="training epoches")
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--test', type=bool, help="Run testdataset and eval")
    parser.add_argument('--type', type = str, help = "model type (Original / Custom)")
    parser.add_argument('--resume', help='resume from checkpoint')

    args, unparsed = parser.parse_known_args()
    return args
# Set Hyperparameters
args = parse_option()

batch_size = args.batch_size
epochs = args.epoch
lr = args.lr
TEST = args.test
gamma = 0.7
seed = 42

# Set dataLoader
image_size = 224
train_loader, valid_loader, test_loader = create_dataLoader('./dataset/train', './dataset/test', image_size = image_size, batch_size = batch_size)

# Set model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SwinTransformer(
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heads=(3, 6, 12, 24),
    channels=3,
    num_classes=2,
    head_dim=32,
    window_size=7,
    downscaling_factors=(4, 2, 2, 2),
    relative_pos_embedding=True
).to(device)

# train
seed_everything(seed)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

    if (epoch + 1) % 10 == 0 :
        torch.save(model, f'./Model_weight/SWIN_{epoch + 1}.pth')

# 
torch.save(model, f'./Model_weight/final_model.pth')


         