import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from dataloader import ChestXrayDataset, ChestXrayTestDataset

if __name__ == "__main__":
    # dataset = ChestXrayDataset(phase="train", crop=True, transfom=transforms.ToTensor())
    # print(dataset[0])
    test_dataset = ChestXrayTestDataset(crop=True, n_per_image=10)
    print(test_dataset[0])
