import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from utils.utils import *
from utils.model import *
from pathlib import Path
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--content_dir", type=str, default="E:/PROJECTS/MAJOR/NST_APP/content_img" ,
                        help="Location of style dir.")
    parser.add_argument("--style_dir", type=str, default="E:/PROJECTS/MAJOR/NST_APP/style_img" ,
                        help="Location of style dir.")
    parser.add_argument("--vgg", type=str, default="E:/PROJECTS/MAJOR/NST_APP/vgg_normalised.pth" ,  #!!
                        help="Location of vgg model.")
    parser.add_argument("--experiment", type=str, default="experiment1",
                        help="Name of experiment.")
    
    parser.add_argument("--final_size", type=int, default=256,
                        help="Size of final image")
    parser.add_argument("--content_size", type=int, default=512,
                        help="Size of content image")
    parser.add_argument("--style_size", type=int, default=512,
                        help="Size of style image")
    parser.add_argument("--crop", action="store_true", default=True,
                        help='Crop image')
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default='1e-4',
                        help='Learning Rate')
    parser.add_argument('--lr_decay', type=float, default='5e-5',
                        help='Learning Rate Decay')
    
    parser.add_argument('--epochs', type=float, default=1,
                        help='Epochs')
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='content_weight')
    
    parser.add_argument('--style_weight', type=float, default=10.0,
                        help='style_weight')
    
    parser.add_argument('--log_interval', type=float, default=1,
                        help='Log Interval')

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

    content_dataset = FolderImageDataset(args.content_dir , content_transforms)
    style_dataset = FolderImageDataset(args.style_dir , style_transforms)


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

    print('Number of batches in content dataset:', len(content_loader))
    print('Number of batches in style dataset:', len(style_loader))
    
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda epoch : 1.0 / (1.0 + args.lr_decay * epoch)  # return multiplier for lr
    )

    criterion = nn.MSELoss()

    running_loss=None
    running_closs=None
    running_sloss=None

    encoder.eval()

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_loader, style_loader),  # to track the progress
                            total = min(len(content_loader), len(style_loader)))
        
        running_loss=0
        running_closs=0
        running_sloss=0
        for content_batch, style_batch in progress_bar:

            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            c_feats = encoder(content_batch)      # content feature
            s_feats = encoder(style_batch)
 
            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1]) # t = features of c and s

            g = decoder(t)  # o/p img from content img and style img features

            g_feats = encoder(g) # features of op img

            c_loss = criterion(g_feats[-1], t) * args.content_weight

            s_loss = 0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = get_mean_std(g_f)
                s_mean, s_std = get_mean_std(s_f)

                s_loss += criterion(g_mean, s_mean) + criterion(g_std, s_std)

            s_loss = s_loss * args.style_weight

            loss = c_loss + s_loss  #loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Loss:{loss.item():4f}, C_Loss:{c_loss.item():4f}, S_Loss:{c_loss.item():4f}')

            running_loss += loss.item() 
            running_closs += c_loss.item()
            running_sloss += s_loss.item()

        scheduler.step()

        running_loss /= len(content_loader)
        running_closs /= len(content_loader)
        running_sloss /= len(content_loader)
                
        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter: {epoch+1}  | Loss: {running_loss:4f} | C_loss: {running_closs:4f} | S_loss: {running_sloss:4f}')









if __name__ == "__main__":
    main()