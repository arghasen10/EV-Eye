import argparse
import logging
import sys
import os
import sys
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import torch.utils.data as Data
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
import numpy as np
import glob

def train_net(
        device,
        epochs: int = 5,
        batch_size: int = 1,
        amp: bool = False
):
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
    for_train_data = train_data[:int(len(train_data)*0.7)]
    for_train_label = train_label[:int(len(train_data)*0.7)]
    test_data = train_data[int(len(train_data)*0.7):]
    test_label = train_label[int(len(train_data)*0.7):]

    trainDataset = Data.TensorDataset((torch.from_numpy(for_train_data).type(torch.FloatTensor) / 255),
                                        torch.div(torch.from_numpy(for_train_label).type(torch.LongTensor), 1))

    n_train = len(trainDataset)
    print(n_train)
    train_loader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        # pin_memory=True
    )

    testDataset = Data.TensorDataset((torch.from_numpy(test_data).type(torch.FloatTensor) / 255),
                                        torch.div(torch.from_numpy(test_label).type(torch.LongTensor), 1))
    n_val = len(testDataset)
    val_loader = torch.utils.data.DataLoader(
        testDataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        # pin_memory=True
    )

    net = UNet(n_channels=1, n_classes=2, bilinear=False)
    net.to(device=device)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch
                torch.set_printoptions(profile="full")

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    # print(masks_pred.shape)
                    loss = criterion(masks_pred, true_masks) \
                            + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                        multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')

                        val_score, miou = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
    
    os.makedirs("checkpoints")
    with open(os.path.join("checkpoints/", "result.txt"), 'w') as outfiletotal:
        outfiletotal.write("dice_score:" + str(val_score.cpu().numpy()) + ",miou_score:" + str(miou) + "\n")
        sys.stdout.flush()
        outfiletotal.flush()
        torch.save(net.state_dict(), os.path.join(str("checkpoints"), "test.pth"))
        logging.info(f'Checkpoint {epoch} saved!')

    
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    try:
        train_net(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            amp=args.amp,
        )
    except KeyboardInterrupt:
        logging.info('Interrupt')
        raise
