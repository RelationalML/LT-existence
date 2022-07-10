import torch
import numpy as np
from torchvision import datasets, transforms
#import datagen

#data loading
def device(gpu):
    use_cuda = torch.cuda.is_available()
    return torch.device(("cuda:" + str(gpu)) if use_cuda else "cpu")

def dimension(dataset):
    if dataset == 'mnist':
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == 'cifar10':
        input_shape, num_classes = (3, 32, 32), 10
    if dataset == 'cifar100':
        input_shape, num_classes = (3, 32, 32), 100
    if dataset == 'tiny-imagenet':
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == 'imagenet':
        input_shape, num_classes = (3, 224, 224), 1000
    return input_shape, num_classes

def get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)

def dataloader(dataset, batch_size, train, workers, noise=None, length=None):
    # Dataset
    if dataset == 'mnist':
        mean, std = (0.1307,), (0.3081,)
        transform = get_transform(size=28, padding=0, mean=mean, std=std, preprocess=False)
        dataset = datasets.MNIST('Data', train=train, download=True, transform=transform)
    if dataset == 'cifar10':
        if train:
            transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])        
            #mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
            #transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        else:
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        dataset = datasets.CIFAR10('Data', train=train, download=True, transform=transform) 
    if dataset == 'cifar100':
        mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = datasets.CIFAR100('Data', train=train, download=True, transform=transform)
    if dataset == 'tiny-imagenet':
        mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
        transform = get_transform(size=64, padding=4, mean=mean, std=std, preprocess=train)
        dataset = custom_datasets.TINYIMAGENET('Data', train=train, download=True, transform=transform)
    if dataset == 'imagenet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2,1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        folder = 'Data/imagenet_raw/{}'.format('train' if train else 'val')
        dataset = datasets.ImageFolder(folder, transform=transform)
    
    # Dataloader
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    shuffle = train is True
    if length is not None:
        indices = torch.randperm(len(dataset))[:length]
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle, 
                                             **kwargs)

    return dataloader, dataset


#model evaluation
def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), torch.squeeze(target).to(device=device, dtype=torch.long)
            output = model(data)
            ## rescale output
            # scale = tickets.rescale_output_torch(target, output)
            # output = scale*output
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(1, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1
