import argparse
import torch
from pathlib import Path
from utils.utils import *
from torch.utils.data import DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--content_dir", type=str, default="E:/PROJECTS/MAJOR/Dataset/train",
                        help="Location of style dir.")
    parser.add_argument("--style_dir", type=str, default="E:/PROJECTS/MAJOR/Dataset/train_2",
                        help="Location of style dir.")
    parser.add_argument("--vgg", type=str, default="",  #!!
                        help="Location of vgg model.")
    parser.add_argument("--experiment", type=str, default="experiment1",
                        help="Name of experiment.")
    parser.add_argument("--final_size", type=int, default=512,
                        help="Size of final image")
    parser.add_argument("--content_size", type=int, default=256,
                        help="Size of content image")
    parser.add_argument("--style_size", type=int, default=256,
                        help="Size of style image")
    parser.add_argument("--crop", action="store_true", default=True,
                        help='Crop image')
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    device = torch.device("cude" if torch.cuda.is_available() else "cpu")

    save_dir = Path("experiment") / args.experiment  #dir to save experiment
    save_dir.mkdir(exist_ok=True, parents=True)
    # saving args values
    with open(save_dir / "args.txt", "w") as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')

    # creating Datasets
    content_transforms = get_transforms(args.content_size, args.crop, args.final_size)
    style_transforms = get_transforms(args.style_size, args.crop, args.final_size)

    content_dataset = FolderImageDataset(args.content_dir, content_transforms)
    style_dataset = FolderImageDataset(args.style_dir, style_transforms)


    content_loader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,  # fast conversion CPU=>GPU
        drop_last=True
    )

    style_loader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,  # fast conversion CPU=>GPU
        drop_last=True    # drop last batch
    )












if __name__ == "__main__":
    main()