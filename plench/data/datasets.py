import sys
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets as dsets
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
import pickle
import numpy as np
from scipy.special import comb
from .randaugment import RandomAugment
from .cutout import Cutout
from scipy.io import loadmat


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    "PLCIFAR10_Aggregate",
    "PLCIFAR10_Vaguest",
    "Lost",
    "MSRCv2",
    "Mirflickr",
    "Birdsong",
    "Malagasy",
    "SoccerPlayer",
    "Italian",
    "YahooNews",
    "English"
]

IMAGE_DATASETS = [
    "PLCIFAR10_Aggregate",
    "PLCIFAR10_Vaguest"
]

TABULAR_DATASETS = [
    "Lost",
    "MSRCv2",
    "Mirflickr",
    "Birdsong",
    "Malagasy",
    "SoccerPlayer",
    "Italian",
    "YahooNews",
    "English"
]

class gen_index_test_tabular_dataset(Dataset):
    def __init__(self, images, true_labels):
        self.data = images
        self.data = torch.from_numpy(self.data).to(torch.float32)
        self.ord_labels = true_labels
        
    def __len__(self):
        return len(self.ord_labels)
        
    def __getitem__(self, index):
        each_image = self.data[index]
        each_true_label = self.ord_labels[index]
        
        return each_image, each_true_label

class gen_index_train_tabular_dataset(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.data = images
        self.data = torch.from_numpy(self.data).to(torch.float32)
        self.partial_targets = given_label_matrix
        self.ord_labels = true_labels
        self.input_dim = images.shape[1]
        self.num_classes = given_label_matrix.shape[1]
        
    def __len__(self):
        return len(self.ord_labels)
        
    def __getitem__(self, index):
        each_image = self.data[index]
        each_label = self.partial_targets[index]
        each_true_label = self.ord_labels[index]
        
        return each_image, each_image, each_image, each_image, each_label, each_true_label

def tabular_train_test_dataset_gen(root, seed, args=None):
    dataset_path = os.path.join(root, (args.dataset+".mat"))
    total_data = loadmat(dataset_path)
    data, ord_labels_mat, partial_targets = total_data['data'], total_data['target'], total_data['partial_target']
    ord_labels_mat = ord_labels_mat.transpose()
    partial_targets = partial_targets.transpose()
    if type(ord_labels_mat) != np.ndarray:
        ord_labels_mat = ord_labels_mat.toarray()
        partial_targets = partial_targets.toarray()
    if data.shape[0] != ord_labels_mat.shape[0] or data.shape[0] != partial_targets.shape[0]:
        raise RuntimeError('The shape of data and labels does not match!')
    if ord_labels_mat.sum() != len(data):
        raise RuntimeError('Data may have more than one label!')
    _, ord_labels = np.where(ord_labels_mat == 1)
    data = (data - data.mean(axis=0, keepdims=True))/(data.std(axis=0, keepdims=True)+1e-6)
    data = data.astype(float)
    total_size = data.shape[0]
    train_size = int(total_size * (1 - args.tabular_test_fraction))
    keys = list(range(total_size))
    np.random.RandomState(seed).shuffle(keys)
    train_idx = keys[:train_size]
    test_idx = keys[train_size:]
    train_set = gen_index_train_tabular_dataset(data[train_idx], partial_targets[train_idx], ord_labels[train_idx])
    test_set = gen_index_test_tabular_dataset(data[test_idx], ord_labels[test_idx])
    return train_set, test_set





class Tabular_Dataset(Dataset):
    def __init__(self, root, args=None):
        self.dataset_path = root
        self.total_data = loadmat(mat_path)

        self.data, self.ord_labels_mat, self.partial_targets = self.total_data['data'], self.total_data['target'], self.total_data['partial_target']
        self.ord_labels_mat = self.ord_labels_mat.transpose()
        self.partial_targets = self.partial_targets.transpose()
        if self.data.shape[0] != self.ord_labels_mat.shape[0] or self.data.shape[0] != self.partial_targets.shape[0]:
            raise RuntimeError('The shape of data and labels does not match!')
        if self.ord_labels_mat.sum() != len(self.data):
            raise RuntimeError('Data may have more than one label!')
        _, self.ord_labels = np.where(self.ord_labels_mat == 1)
        self.data = (self.data - self.data.mean(axis=0, keepdims=True))/self.data.std(axis=0, keepdims=True)
        self.input_dim = self.data.shape[1]
        self.num_classes = self.ord_labels_mat.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        original_image = image
        weak_image = image
        strong_image = image
        return original_image, weak_image, strong_image, self.partial_targets[index], self.ord_labels[index]


class PLCIFAR10_Aggregate(Dataset):
    def __init__(self, root, args=None):


        dataset_path = os.path.join(root, 'plcifar10', f"plcifar10.pkl")
        partial_label_all = pickle.load(open(dataset_path, "rb"))

        self.transform = transforms.Compose(
            [transforms.ToTensor(),  
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(32,4),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605))])
        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, None),
            transforms.ToTensor(),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605)),
        ])
        self.distill_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605))
        ])
        self.test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605))])
        self.input_dim = 32 * 32 * 3
        self.num_classes = 10

        original_dataset = dsets.CIFAR10(root=root, train=True, download=True)
        self.data = original_dataset.data

        self.partial_targets = np.zeros((len(self.data), self.num_classes))


        for key, value in partial_label_all.items():
            for candidate_label_set in value:
                for label in candidate_label_set:
                    self.partial_targets[key, label] = 1


        self.ord_labels = original_dataset.targets
        self.ord_labels = np.array(self.ord_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        original_image = self.test_transform(image)
        weak_image = self.transform(image)
        strong_image=self.strong_transform(image)
        distill_image=self.distill_transform(image)
        return original_image, weak_image, strong_image, distill_image, self.partial_targets[index], self.ord_labels[index]


class PLCIFAR10_Vaguest(Dataset):
    def __init__(self, root, args=None):
        dataset_path = os.path.join(root, 'plcifar10', f"plcifar10.pkl")
        partial_label_all = pickle.load(open(dataset_path, "rb"))
        self.transform = transforms.Compose(
            [transforms.ToTensor(),  
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(32,4),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605))])
        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, None),
            transforms.ToTensor(),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605)),
        ])
        self.distill_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605))
        ])
        self.test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605))])
        self.input_dim = 32 * 32 * 3
        self.num_classes = 10

        original_dataset = dsets.CIFAR10(root=root, train=True, download=True)
        self.data = original_dataset.data

        self.partial_targets = np.zeros((len(self.data), self.num_classes))


        for key, value in partial_label_all.items():
            vaguest_candidate_label_set = []
            largest_num = 0
            for candidate_label_set in value:
                if len(candidate_label_set) > largest_num:
                    vaguest_candidate_label_set = candidate_label_set
                    largest_num = len(candidate_label_set)
            for label in vaguest_candidate_label_set:
                self.partial_targets[key, label] = 1            

        self.ord_labels = original_dataset.targets
        self.ord_labels = np.array(self.ord_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        original_image = self.test_transform(image)
        weak_image = self.transform(image)
        strong_image=self.strong_transform(image)
        distill_image = self.distill_transform(image)
        return original_image, weak_image, strong_image, distill_image, self.partial_targets[index], self.ord_labels[index]


def test_dataset_gen(root, args=None):
    if args.dataset == "PLCIFAR10_Aggregate" or args.dataset == "PLCIFAR10_Vaguest":
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605))])
        test_dataset = dsets.CIFAR10(root=root, train=False, transform=test_transform)
    return test_dataset
