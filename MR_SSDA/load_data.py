import torch.utils.data as data
import torchvision
from torchvision import datasets, transforms
import torch.utils.data
from builtins import object


def loader_train(path, img_size, batch_size, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize(256),
                                 # transforms.Scale(img_size),
                                 transforms.CenterCrop(img_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )


def loader_test(path, img_size, batch_size, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize(256),
                                 # transforms.Scale(img_size),
                                 transforms.CenterCrop(img_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )


class PairedData(object):
    def __init__(self, data_loader_B, max_dataset_size):
        self.data_loader_B = data_loader_B
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_B = False
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        B, B_paths = None, None

        try:
            B, B_paths = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        if self.stop_B or self.iter > self.max_dataset_size:
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'data': B, 'label': B_paths}


class UnalignedDataLoader():
    def initialize(self, T_img, batch_size, shuffle, num_workers=4, pin_memory=True):
        img_target = torchvision.datasets.ImageFolder(T_img,
                                                      transform=transforms.Compose([
                                                          transforms.Resize(256),
                                                          transforms.CenterCrop(224),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                               [0.229, 0.224, 0.225])]))

        target_loader = torch.utils.data.DataLoader(img_target,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory,
                                                    drop_last=True)

        self.dataset_t = img_target
        self.paired_data = PairedData(target_loader, float("inf"))
        return self.paired_data, len(target_loader.dataset)

    def name(self):
        return 'UnalignedDataLoader'


if __name__ == '__main__':

    code = 0
