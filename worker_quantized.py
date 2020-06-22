import sys
import argparse
import torch
import torchvision
import numpy as np
import time
import copy
from torchvision import datasets, transforms
from torch.utils import data
from torch.utils.data import ConcatDataset, Subset
from torchvision.models import resnet
import resnet_cifar
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils import data
from tensorboardX import SummaryWriter
from torchvision.datasets.folder import ImageFolder, default_loader, is_image_file
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
import os
import random
from valdataset import LabeledImageFolder

from quantization import *

import mpi4py
################################################################
#         Warning: You are about to approach something         #
#                  evil. Prepare youself as there's a          #
#                  good chance it will cause you a             #
#                  lot of frustration and annoyance.           #
################################################################
# This line is evil.............                               #
# Whenever you have multithreading or window-creation issues,  #
# try commenting/uncommenting this line to see if it fixes it. #
# I absolutely hate this line. On our local server commenting  #
# it would make it crash. On Daint, uncommenting it would make #
# it crash.                                                    #
#                    YOU HAVE BEEN WARNED!!!!                  #
################################################################


# mpi4py.rc.threads = False


################################################################
#                        End of warning.                       #
################################################################
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
group = comm.Get_group()


def log(x):
    print("[Process %s] %s" % (rank, x))
    sys.stdout.flush()


def optimizer_construct(model, lr, dataset_name, weight_decay=0.0):
    if dataset_name in ['cifar10', 'cifar100', 'imagenette', 'imagewoof', 'imagenet']:
        momentum = 0.9
    else:
        momentum = 0
    return SGD(model.parameters(),
               lr=lr, momentum=momentum, weight_decay=weight_decay)


def model_construct(dataset_name):
    if dataset_name == 'cifar10':
        return resnet_cifar.resnet20_cifar(), 'resnet20'
    elif dataset_name == 'imagenet':
        return resnet.resnet50(), 'resnet50'
    elif dataset_name == 'imagenette':
        return resnet.resnet18(), 'resnet18'
    elif dataset_name == 'imagewoof':
        return SmallCifar10Net(), 'SmallCifar10Net'
    elif dataset_name == 'cifar100':
        return SmallCifar100Net(), 'SmallCifar100Net'
    elif dataset_name == 'mnist':
        return MNISTNet(), 'MNISTNet'


# "Fast" schedule from Fast.AI
def fast_lr_schedule(epoch):
    if epoch < 4:
        return 1
    elif 4 <= epoch < 18:
        return 0.4
    elif 18 <= epoch < 21:
        return 0.1
    elif 21 <= epoch < 35:
        return 0.01
    elif 35 <= epoch < 43:
        return 0.001
    elif 43 <= epoch:
        return 0.0001


def scheduler_construct(optimizer, dataset_name, warmup_epochs):
    scheduler = None

    if dataset_name == "imagenet":
        steps = [5, 30, 60, 80]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fast_lr_schedule)
    elif dataset_name in ["cifar10", "cifar100"]:
        steps = [81, 122, 164]
    else:
        steps = []

    if not scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1)

    return scheduler


def loader_construct(dataset, batch_size=64, num_workers=0, sampler=None):
    if sampler is None:
        return data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)




class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SmallCifar10Net(nn.Module):
    def __init__(self):
        super(SmallCifar10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SmallCifar100Net(nn.Module):
    def __init__(self):
        super(SmallCifar100Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 120)
        self.fc3 = nn.Linear(120, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_class(model, loader, device):
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if (batch_idx + 1) % 100 == 0:
                log(batch_idx + 1)

            data, target = data.to(device), target.to(device)
            output = model(data)
            scalar = torch.tensor([0.5]).to(device)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    return correct


def test(epoch, model, device, writer, train_loader, test_loader, verbose=True):
    model.eval()
    loader = train_loader
#     train_correct = test_class(model, loader, device)
    train_correct = 0
    if rank == 0 and verbose:
        log('\nTrain set: Epoch: {} Accuracy: {:.6f}%'.format(epoch + 1, (train_correct / len(loader.dataset)) * 100))

    if writer:
        writer.add_scalar('Train accuracy', (train_correct / len(loader.dataset)) * 100, epoch + 1)

    loader = test_loader
    test_correct = test_class(model, loader, device)
    if rank == 0 and verbose:
        log('\nTest set: Epoch: {} Accuracy: {:.6f}%'.format(epoch + 1, (test_correct / len(loader.dataset)) * 100))
    model.train()

    if writer:
        writer.add_scalar('Test accuracy', (test_correct / len(loader.dataset)) * 100, epoch + 1)
    if writer:
        writer.flush()

    return train_correct, test_correct


def model_update(model, optimizer, epoch, data, target, criterion):
    optimizer.zero_grad()

    output = model(data)
    if dataset_name in ['cifar10', 'mnist', 'cifar100', 'imagenette', 'imagewoof', 'imagenet']:
        loss = criterion(output, target)
    else:
        loss = criterion(output, target.view(output.size()).float())
    loss.backward()

    optimizer.step()
    return loss

def copy_to_model(model, model_copy_tensor):
    counter = 0
    for param in model.parameters():
        t = param.data.view(-1)
        
        decompressed_t = decompressor.decompress(model_copy_tensor[counter: counter + t.nelement() * decompressor.length])
        decoded_t = decoder.decode(decompressed_t, t)
        
        t[:] = (t + decoded_t) / 2
        counter += t.nelement() * decompressor.length


def model_to_copy(model, model_copy_tensor):
    counter = 0
    for param in model.parameters():
        t = param.data.view(-1)
        
        encoded_t = encoder.encode(t)
        compressed_t = compressor.compress(encoded_t)
        
        
        model_copy_tensor[counter: counter + compressed_t.nelement()] = compressed_t.view(-1)
        counter += compressed_t.nelement()


def total_copy_to_model(model, model_copy_tensor):
    counter = 0
    for param in model.parameters():
        t = param.data
        t.view(-1)[:] = model_copy_tensor[counter: counter + t.nelement()]
        counter += t.nelement()
    for name, buf in model.named_buffers():
        t = buf.data
        t.view(-1)[:] = model_copy_tensor[counter: counter + t.nelement()]
        counter += t.nelement()


def total_model_to_copy(model, model_copy_tensor):
    counter = 0
    for param in model.parameters():
        t = param.data
        model_copy_tensor[counter: counter + t.nelement()] = t.view(-1)[:]
        counter += t.nelement()
    for name, buf in model.named_buffers():
        t = buf.data
        model_copy_tensor[counter: counter + t.nelement()] = t.view(-1)[:]
        counter += t.nelement()


##############################################################################################
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.01, metavar='M',
                    help='SGD momentum (default: 0.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--async', action='store_true', default=False,
                    help='Asynchronous version of popSGD')
# parser.add_argument("-p", type=int, default=8,
#                     help="Number of models")
# parser.add_argument("-k", type=int, default=2,
#                     help="Number of averaging models")
parser.add_argument("--num-processes", type=int, default=16,
                    help="Number of processes for multiprocessing")
parser.add_argument("--dataset-name", type=str, default='cpusmall',
                    help="Number of averaging models")

parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--everyone-save', action='store_true', default=False,
                    help='All processes save models')

parser.add_argument('--no-log', action='store_true', default=False,
                    help='For saving logs of the model')
parser.add_argument('--local-updates', type=int, default=1,
                    help='Number of local updates to perform before communicating')

parser.add_argument('--warmup-epochs', type=int, default=0,
                    help="Number of warmup epochs before communication begins")
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help="Amount of weight decay in SGD optimizer")

parser.add_argument('--average-epochs', type=int, default=10,
                    help='Average number of times each node goes through the entire dataset')
parser.add_argument('--virtual-epoch-num', type=int,
                    help='Number of times to consider checking accuracies (This will affect the scheduler steps)')

args = parser.parse_args()
dataset_name = args.dataset_name
batch_size = args.batch_size
test_batch_size = args.test_batch_size
warmup_epochs = args.warmup_epochs
save_model = args.save_model
everyone_save = args.everyone_save
save_log = not args.no_log
local_updates = args.local_updates
log_interval = args.log_interval
lr = args.lr
momentum = args.momentum
average_epochs = args.average_epochs
virtual_epoch_num = args.virtual_epoch_num
if not virtual_epoch_num:
    virtual_epoch_num = average_epochs

if rank == 0:
    print(args)

train_set = None
test_set = None
if args.dataset_name == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
elif args.dataset_name == 'mnist':
    train_set = datasets.MNIST('~/popsgd/data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
    test_set = datasets.MNIST('~/popsgd/data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
elif args.dataset_name == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
elif args.dataset_name == 'imagenet':
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_set = LabeledImageFolder('./',
                                   label_root='/scratch/snx3000/alistarh/mojtaba_stuff/imagenet_map/train_map.txt',
                                   transform=transform_train)
    test_set = LabeledImageFolder('./',
                                  label_root='/scratch/snx3000/alistarh/mojtaba_stuff/imagenet_map/val_map.txt',
                                  transform=transform_test)
else:
    print("No such dataset ", args.dataset_name)
    exit(1)

criterion = torch.nn.MSELoss()
if args.dataset_name in ['cifar10', 'cifar100', 'imagenet']:
    criterion = torch.nn.CrossEntropyLoss()
elif args.dataset_name == 'mnist':
    criterion = torch.nn.NLLLoss(reduction='mean')

steps_per_virtual_epoch = (np.ceil(len(train_set) / batch_size) * average_epochs - 3) / virtual_epoch_num

np.random.seed(args.seed)
torch.manual_seed(args.seed)

##############################################################################################

comm.Barrier()
log("Data getting complete")

sys.stdout.flush()
print(len(train_set))
print(len(test_set))

device = torch.device('cuda')
comm.Barrier()

# Create one model on the first process and send it to everyone so that everyone has the same starting point.
model = None
model_name = None
writer = None
if rank == 0:
    model, model_name = model_construct(dataset_name)

if size > 1:
    model = comm.bcast(model, root=0)
    model_name = comm.bcast(model_name, root=0)

# model.load_state_dict(torch.load('/scratch/snx3000/alistarh/mojtaba_stuff/popsgd/models/model_base.pt'))

# Create the summary writer on first process to write the logs.
# Also if specified, create the folder to save the model at the end
# of each epoch
warmup_included = ""
if warmup_epochs > 0:
    warmup_included = "_with_warmup"

filename = 'run_%s_%s_workers_%s_average_epochs_%s_%s_virtual_epochs' % (
    dataset_name, size, average_epochs, model_name, virtual_epoch_num) + \
           warmup_included + "_%s_local_updates" % local_updates  + "_with_chunking"


model_dir = 'models/' + filename
if rank == 0:
    if save_log:
        writer_dir = 'logs/' + filename
        dir_per_rank = writer_dir + '/model_%s' % rank
        writer = SummaryWriter(dir_per_rank)

    if save_model:
        import os

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

# Compute the size of the model (Buffers and all)
total_elements = 0
for param in list(model.parameters()):
    total_elements += param.data.nelement()
buffer_size = 0
for buf in list(model.buffers()):
    buffer_size += buf.data.nelement()
model_size = total_elements + buffer_size

# All the stuff needed for the quantization
encoder = SimpleLatticeEncoder(256, size)
decoder = SimpleLatticeDecoder(256, size)
compressor = ModuloCompressor(encoder.q)
decompressor = ModuloDecompressor(decoder.q)


# Allocate 2 sequential blocks in memory with the same size. (We will use these later)
model_copy = torch.empty(model_size * compressor.length, dtype=torch.uint8, device=device)
partner_model = torch.empty(model_size * compressor.length, dtype=torch.uint8, device=device)
partner_buf = MPI.memory.fromaddress(partner_model.data_ptr(), partner_model.nelement() * partner_model.element_size())

# Create dataloaders
num_proc = size // 2 if size > 1 else size
idx = rank % num_proc
sampler = torch.utils.data.DistributedSampler(train_set, num_proc, idx)

# sampler = None
# num_proc = 1
train_loader = loader_construct(train_set, batch_size=batch_size, sampler=sampler)
test_loader = loader_construct(test_set, batch_size=test_batch_size)

# Divide the train and test sets to chunks of almost equal size to parallelise
# finding accuracy on test and train.
train_per_rank_size = int(np.ceil(len(train_set) / size))
test_per_rank_size = int(np.ceil(len(test_set) / size))

# Find start and end indices of the datasets for this process
train_start = train_per_rank_size * rank
train_end = min(train_start + train_per_rank_size, len(train_set))
test_start = test_per_rank_size * rank
test_end = min(test_start + test_per_rank_size, len(test_set))

# Create a subset of the datasets using those start and end points
train_set_chunk = Subset(train_set, list(range(train_start, train_end)))
test_set_chunk = Subset(test_set, list(range(test_start, test_end)))

# Create a loader on each of those subsets
distributed_train_loader = loader_construct(train_set_chunk, batch_size=test_batch_size)
distributed_test_loader = loader_construct(test_set_chunk, batch_size=test_batch_size)

log("%s - %s" % (train_start, train_end))
log("%s - %s" % (test_start, test_end))

# Create dummy model for purposes of finding accuracy (We need it to be a module not a tensor to use it)
# test_model = copy.deepcopy(model)

# Move everything to the gpu and set it to train mode
model = model.to(device)
model.train()
#test_model = test_model.to(device)
#test_model.train()

# Create the optimizer
optimizer = optimizer_construct(model, lr, args.dataset_name, weight_decay=args.weight_decay)

# Create the scheduler (For decreasing the lr)
scheduler = scheduler_construct(optimizer, args.dataset_name, args.warmup_epochs)

# Create a window on the first sequential block of memory with the same size of the model
# (Instead of creating it on our own model, other processes will only see this copy of the
#  model and once the model has changed (SGD step has been performed), we will update this
#  copy)
buf = MPI.memory.fromaddress(model_copy.data_ptr(), model_copy.nelement() * model_copy.element_size())
win = MPI.Win.Create(buf, comm=comm)

# Put up a barrier to make sure everyone has reached this point before continuing
comm.Barrier()

# Start the clock!
start = time.time()
train_accuracies = []
test_accuracies = []

def save_checkpoint(epoch, counter):
    if not save_model:
        return 
    if everyone_save or (rank == 0):
        path = model_dir + "/" + "model_%s_epoch_%s.pt" % (rank, epoch)
        torch.save({
            'epoch': epoch,
            'counter': counter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, path)
        

def compute_accuracies():
    if rank == 0:
        log("Starting test")
    # We will accumulate the results in this in process 0
    final_corrects = np.zeros(2)

    if rank == 0:
        final_corrects[0], final_corrects[1] = test(epoch, model, device, None, train_loader, test_loader, verbose=False)
    
    comm.Barrier()

    return final_corrects


def random_partner():
    # Choose a random partner to do the averaging with
    partner_rank = np.random.randint(size)
    while partner_rank == rank:
        partner_rank = np.random.randint(size)
    return partner_rank


def communicate():
    partner_rank = random_partner()

    # Get a "shared" lock on your partner.
    # (Using shared, if multiple processes choose the same partner, they can read at the same time)
    win.Lock(partner_rank, lock_type=MPI.LOCK_SHARED)

    # Copy the model from your partner to the second sequential block we took in the memory
    # (Note that we can't just use the tensor here. We must use the buffer of it.
    win.Get((partner_buf, MPI.UINT8_T), target_rank=partner_rank)

    # Release lock
    win.Unlock(partner_rank)

    # Apply this new to the model (Here we only average parameters not buffers as it will mess things up)
    copy_to_model(model, partner_model)


def end_of_epoch(epoch, counter):

    # Make sure everyone has finished their epoch
    comm.Barrier()

    # This is only valid on the 0 process (rank = 0)
    final_corrects = compute_accuracies()

    # Print out and log the accuracies
    if rank == 0:
        train_accuracy = (final_corrects[0] / len(train_set)) * 100
        test_accuracy = (final_corrects[1] / len(test_set)) * 100
        log('\nTrain set: Epoch: {} Accuracy: {:.6f}%'.format(epoch, train_accuracy))
        log('\nTest set: Epoch: {} Accuracy: {:.6f}%'.format(epoch, test_accuracy))

        # train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        if writer:
            writer.add_scalar('Train accuracy', train_accuracy, epoch)
            writer.add_scalar('Test accuracy', test_accuracy, epoch)

    # If --save-model was given, save model of process 0 (We are at the end of an epoch)
    save_checkpoint(epoch, counter)
    

    # Tell the scheduler we have finished an epoch
    scheduler.step()

    # Wait for everyone to catch up to start the next epoch
    comm.Barrier()


try:
    # This counter will save the number of SGD steps performed on this process so far
    counter = 0
    virtual_epoch = 0
    batch_times = []
    if size > 1:
        # Warm up phase
        for epoch in range(warmup_epochs):
            for (data, target) in train_loader:
                # Move data to the gpu
                data, target = data.to(device), target.to(device)
                # Perform one SGD step and get the loss
                loss = model_update(model, optimizer, epoch, data, target, criterion)

        # If size > 1 we perform popsgd
        for epoch in range(average_epochs):
            for _ in range(num_proc):
                for (data, target) in train_loader:
                    
                    counter += 1
                    
                    batch_start = time.time()
                    
                    # Move data to the gpu
                    data, target = data.to(device), target.to(device)

                    # Perform one SGD step and get the loss
                    loss = model_update(model, optimizer, epoch, data, target, criterion)

                    # Lock your current window so no one can try to read from it while you are updating the values
                    win.Lock(rank, lock_type=MPI.LOCK_EXCLUSIVE)
                    # Copy all the values from model to the copy which is sequential and everyone can see
                    model_to_copy(model, model_copy)
                    # Release the lock or else no one else can use it
                    win.Unlock(rank)

                    if rank == 0 and counter % log_interval == 0:
                        log('Train: Epoch: {} Step:{} Error: {:.6f}'.format(epoch + 1, counter, loss.item()))
                        log("--------------------------------")
                        if writer:
                            # Add loss to the logs
                            writer.add_scalar('Train loss', loss.item(), counter)

                            if counter % (20 * log_interval):
                                # Write whatever is in the buffers to the file
                                writer.flush()
                    
                    if counter % local_updates == 0:
                    	communicate()
                    
                    
                    batch_end = time.time()
                    
                    batch_times.append(batch_end - batch_start)
                    
                    # Compute and log accuracies, update scheduler, and wait for everyone to catch up
                    # at the end of each virtual epoch
                    if counter // steps_per_virtual_epoch != virtual_epoch:
                        virtual_epoch += 1
                        if rank == 0 and len(batch_times) > 0:
                            log("Average time/batch is %.3f" % (np.mean(batch_times)))
                        batch_times.clear()
                        end_of_epoch(virtual_epoch, counter)
                    

    else:
        # If size=1 then we perform a simple SGD.
        log("Vanilla SGD running")
        counter = 0
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                counter += 1
                data, target = data.to(device), target.to(device)
                loss = model_update(model, optimizer, epoch, data, target, criterion)

                if batch_idx % log_interval == 0:
                    log('Train: Epoch: {} Step:{} Error: {:.6f}'.format(epoch + 1, batch_idx, loss.item()))
                    if writer:
                        writer.add_scalar('Train loss', loss.item(), counter)

            test(epoch, model, device, writer, train_loader, test_loader)
            scheduler.step()

    log("WOOOOOOOOOOOOOOOO")
    if rank == 0 :
        # log("Best train accuracy: %s" % np.max(train_accuracies))
        if len(test_accuracies) > 0:
            log("Best test accuracy: %s" % np.max(test_accuracies))

    # Wait for everyone to finish
    comm.Barrier()

    # Stop the clock
    end = time.time()

    if rank == 0:
        log(end - start)
    
    if len(batch_times) > 0 :
        log("Mean batch time %.3f, Std %.3f" % (np.mean(batch_times), np.std(batch_times)))
        log("Min batch time %.3f, Max batch time %.3f"  % (np.min(batch_times), np.max(batch_times)))
        log(batch_times)
        
    
    # Deallocate the window
    win.Free()
    if rank == 0:
        if writer:
            writer.close()


except Exception as err:
    import traceback

    traceback.print_exc()
    print(err)
    sys.stdout.flush()
    sys.exit()
