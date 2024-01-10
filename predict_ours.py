import argparse
import logging
import os
import glob
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import h5py
from unet import UNet
from utils.utils import plot_img_and_mask
import matplotlib.pyplot as plt
from tqdm import tqdm

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(np.asarray(full_img)[np.newaxis, ...])
    # print(img.shape)
    img = torch.div(img.type(torch.FloatTensor), 255)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()
        # print(full_mask.shape)
    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', help="pls enter a string", type=str)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images',
                        default=os.getcwd())
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    parser.add_argument('--predict', type=str, default='1')

    parser.add_argument('--data_dir', type=str, default=os.getcwd())
    parser.add_argument('--whicheye', type=str, default="right")
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the program on (cpu / cuda)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = args.device
    net = UNet(n_channels=1, n_classes=2, bilinear=args.bilinear)

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    net.to(device=device)
    model_dir = 'test.pth'
    net.load_state_dict(torch.load(model_dir, map_location=device))

    logging.info('Model loaded!')

    datasets = glob.glob("eye_dataset/gt_data/ellipse_masks_*.h5")
    train_data = []
    train_label = []
    for d in datasets:
        with h5py.File(d,'r') as hf:
            data_grp = hf["data"]
            label_grp = hf["label"]
            for frame_name in data_grp:
                img_data = data_grp[frame_name][()]
                mask_data = label_grp[frame_name][()]
                train_data.append(img_data)
                train_label.append(mask_data)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_data = (train_data.T).reshape(-1, 1, 260, 346)
    train_label = (train_label.T).reshape(-1, 260, 346)
    print("train_data.shape", train_data.shape)
    print("train_label.shape", train_label.shape)
    combined_dataset = torch.utils.data.TensorDataset((torch.from_numpy(train_data).type(torch.FloatTensor) / 255),
                                        torch.div(torch.from_numpy(train_label).type(torch.LongTensor), 1))

    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        # pin_memory=True
    )
    frames = glob.glob(f"eye_dataset/gt_data/saved_frame_*.png")

    for i, filename in enumerate(tqdm(frames)):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)
        mask = predict_img(net=net,
                            full_img=img,  #
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)

        result = (np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8)
        img = np.asarray(img)
        img = img + result
        plt.imshow(img)
        plt.show()
