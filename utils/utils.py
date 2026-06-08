from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

class FolderImageDataset(Dataset):
    def __init__(self, root, transformation=None):
        self.root = root
        self.transformation = transformation
        self.files = list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith((".png",".jpg",".jpeg"))]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.files[idx])
        image = Image.open(file_path).convert('RGB')

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

def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    content_mean, content_std = get_mean_std(content_feat)
    style_mean, style_std = get_mean_std(style_feat)
    
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)      # (x - mean)/ sigma
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)

def get_mean_std(feat, eps=1e-5):
    size = feat.size()  # [batch size, n channel, h, w]
    assert (len(size)==4)
    batch_size, channels = size[:2]
                                # 3 dim                         # back to 4 dim
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std
