'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from models import *
from utils import progress_bar
try:
    import wandb
    wandb_installed = True
except ImportError:
    wandb_installed = False
import time, os
def check_injection_status():
    cuda_injection_path = os.getenv("CUDA_INJECTION64_PATH")
    if cuda_injection_path:
        return "INJECTION_ON"
    else:
        return "INJECTION_OFF"
#from rax_util import CudaProfiler
# Set a random seed for all devices (both CPU and CUDA)
torch.manual_seed(42)

# If you are working with CUDA and you want to ensure reproducibility
# across runs even on GPU, you might also want to set:
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU setups

# To ensure further that the same initialization occurs every time,
# you might want to disable the benchmark mode and enable deterministic algorithms:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Define the custom dataset
class RandomCIFAR10(torch.utils.data.Dataset):
    def __init__(self, num_samples, transform=None):
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random image (3 channels, 32x32)
        image = torch.rand(3, 32, 32)
        # Generate a random label (0 to 9)
        label = torch.randint(0, 10, (1,)).item()
        
        if self.transform:
            image = self.transform(image)

        return image, label

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch', default=256, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', '-e', default=100000, type=int, help='resume from checkpoint')
parser.add_argument("--iters", "-i", default=-1, type=int)
parser.add_argument('--gpu', type=int, default=0, help='GPU device index (default: 0)')
args = parser.parse_args()

# Set the GPU device
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    device = f'cuda:{args.gpu}'
else:
    device = 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
if wandb_installed:
    wandb.init(name=f"{args.batch}_{check_injection_status()}")
    wandb.config.update(args)
    wandb.config.update({"GROUP": f"{check_injection_status()}_{os.getenv('GPU', 'GPU')}_{args.batch}"})


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = RandomCIFAR10(num_samples=50000, transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=False)

testset = RandomCIFAR10(num_samples=10000, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet50()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    count = 0 

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    data_iter = iter(trainloader)  # Create an iterator from the data loader

    start.record()
    start_device = time.time()
    for batch_idx in range(len(trainloader)):
        if batch_idx > args.iters :
            break
        try:
            inputs, targets = next(data_iter)  # Get the next batch
        except StopIteration:
            data_iter = iter(trainloader)  # Reinitialize the iterator if we reach the end
            inputs, targets = next(data_iter)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(batch_idx)
    end_device = time.time()
    print(f"Epoch {epoch} time :  {end_device - start_device}")
    if wandb_installed:
        wandb.log({"epoch_time": end_device - start_device})
        #progress_bar(batch_idx, len(trainloader))
        #torch.cuda.synchronize()

for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
