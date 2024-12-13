import os
import torch
from torchvision import datasets, transforms
from PIL import Image
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import tiny_iid
from sampling import sample_dirichlet
from torchvision.transforms import ToTensor
from config import config
from torchvision.datasets.folder import default_loader



def prepare_val_folder(dataset_path):
    """
    Split validation images into separate class-specific sub folders. Like this the
    validation dataset can be loaded as an ImageFolder.
    """
    val_dir = os.path.join(dataset_path, 'val')
    img_dir = os.path.join(val_dir, 'images')

    # read csv file that associates each image with a class
    annotations_file = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = annotations_file.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    annotations_file.close()

    # create class folder if not present and move image into it
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(os.path.dirname(img_dir), folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

    # remove old image path
    if os.path.exists(img_dir):
        os.rmdir(img_dir)


class TINY_idx(datasets.ImageFolder):
    def __init__(self, root: str,  transform = None, target_transform = None,):
        super(TINY_idx, self).__init__(root, loader=default_loader,
        transform = transform,
        target_transform = target_transform,
        is_valid_file = None)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index



#    def __getitem__(self, index: int):
#        img, target = self.data[index], self.targets[index]
#
#        img = Image.fromarray(img)
#
#        if self.transform is not None:
#            img = self.transform(img)
#
#        if self.target_transform is not None:
#            target = self.target_transform(target)
#
#        return img, target, index 


class CIFAR10_idx(datasets.CIFAR10):
    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, download: bool = False,):
        super(CIFAR10_idx, self).__init__(root, train=train, transform=transform,
                                      target_transform=target_transform, download=download)
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index 
    
class MINIST_idx(datasets.MNIST):
    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, download: bool = False,):
        super(MINIST_idx, self).__init__(root, train=train, transform=transform,
                                      target_transform=target_transform, download=download)
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # pdb.set_trace()
        img = img.numpy()
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index 

def get_dataset():
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if config.dataset == 'tiny':
        # dwonload dataset with
        # wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
        prepare_val_folder(config.datadir+'tiny-imagenet-200')
        data_dir=config.datadir+'tiny-imagenet-200/train'
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
        #train_dataset = datasets.ImageFolder(
        #root=config.datadir+'/tiny-imagenet-200/train',
        #transform=transforms.Compose([
        #        transforms.ToTensor(),
        #        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        #    ])
        #)

        train_dataset = TINY_idx(data_dir, transform=train_transforms)

        test_dataset = datasets.ImageFolder(
            root=config.datadir+'tiny-imagenet-200/val',
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
            ])
        )

        if config.iid:
            # Sample IID user data from Tiny 
            user_groups = tiny_iid(train_dataset, config.n_parties)
        elif config.dirichlet:
            tr_dataset = datasets.ImageFolder(root=data_dir, 
                                                transform=train_transforms)
            user_groups = sample_dirichlet(tr_dataset, config.n_parties, config.dirichlet_deg)
        else:
            print("TinyImagenet sorted non-iid not implemented.")
            exit()

    elif config.dataset == 'cifar10':
        data_dir = config.datadir+'cifar/'
        train_transforms = transforms.Compose(
            [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ]
        )
        test_transforms = transforms.Compose(
            [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        )

        train_dataset = CIFAR10_idx(data_dir, train=True, download=True,
                                       transform=train_transforms)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=test_transforms)

        # sample training data amongst users
        if config.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, config.n_parties)
        elif config.dirichlet:
            tr_dataset = datasets.CIFAR10(data_dir, train=True, download=True, 
                                          transform=train_transforms)
            user_groups = sample_dirichlet(tr_dataset, config.n_parties, config.dirichlet_deg)
        else:
            # Sample Non-IID user data from Mnist
            if config.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, config.n_parties)

    elif config.dataset == 'mnist':
        data_dir = config.datadir+'mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = MINIST_idx(data_dir, train=True, download=True,
                                       transform=apply_transform)
        #train_dataset = MINIST_idx(data_dir, train=True, download=True,
        #                               transform=ToTensor())
        # train_dataset = datasets.MNIST(data_dir, train=True, download=True,
        #                                transform=apply_transform)
        
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        #test_dataset = datasets.MNIST(data_dir, train=False, download=True,
        #                              transform=ToTensor())
        

        # sample training data amongst users
        if config.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, config.n_parties)
        elif config.dirichlet:
            tr_dataset = datasets.MNIST(data_dir, train=True, download=True, 
                                        transform=apply_transform)
            user_groups = sample_dirichlet(tr_dataset, config.n_parties, config.dirichlet_deg)
        else:
            # Sample Non-IID user data from Mnist
            if config.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, config.n_parties)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, config.n_parties)

    return train_dataset, test_dataset, user_groups

