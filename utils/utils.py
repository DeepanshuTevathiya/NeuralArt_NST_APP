from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

class FolderImageDataset(Dataset):
    def __init__(self, root, transformation=None):
        self.root = root,
        self.transformation = transformation
        self.files = list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith((".png",".jpg",".jpeg"))]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.files[idx])
        image = Image.open(file_path)

        if self.transformation:
            image = self.transformation(image)
        
        return image
    
def get_transforms(size, crop, final_size):
    transforms_list = []
    if size>0:
        transforms_list.append(transforms.Resize(size))
    if crop:
        transforms_list.append(transforms.RandomCrop(final_size))
    else:
        transforms_list.append(transforms.Resize(final_size))

    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)
