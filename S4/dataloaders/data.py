import torch
import torchvision
import torchvision.transforms as transforms


def MNIST(batch_size=128, num_val=10E3):
    print("[*] Generating MNIST Classification Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 10, 1
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(IN_DIM, SEQ_LENGTH).t()),
        ]
    )
    # Note: Add data augmentation here if needed

    train = torchvision.datasets.MNIST(root="../data", train=True, download=True, transform=tf)
    test = torchvision.datasets.MNIST(root="../data", train=False, download=True, transform=tf)

    # Split the train set into validation and train sets
    split = int(num_val) # Take samples for validation
    indices = torch.randperm(len(train)).tolist()
    train_inds, val_inds = indices[split:], indices[:split]

    val_subset = torch.utils.data.Subset(train, val_inds)
    train_subset = torch.utils.data.Subset(train, train_inds)

    print(f"[*] Train set size: {len(train_subset)}")
    print(f"[*] Validation set size: {len(val_subset)}")
    print(f"[*] Test set size: {len(test)}")

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    constants = {
        "N_CLASSES": N_CLASSES,
        "SEQ_LENGTH": SEQ_LENGTH,
        "IN_DIM": IN_DIM,
    }

    return trainloader, valloader, testloader, constants


class Datasets():
    mnist = MNIST