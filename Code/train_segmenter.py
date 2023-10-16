import numpy as np
from custom_dataset import CustomDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tvt
import segmentation_models_pytorch as smp
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
import sys
from tqdm import tqdm
from time import sleep


def compute_iou(outputs, labels):
    """
    Function to compute the iou score
    Args:
        outputs (tensor): predicted masks
        labels (tensor): ground-truth masks

    Returns:
        IoU score (float)
    """

    # True if value > 0.5, False otherwise
    outputs, labels = torch.sigmoid(outputs) > 0.5, labels > 0.5
    smooth = 1e-6

    # batch x num_classes x height x weight
    b, n, h, w = outputs.shape

    # Iterates over the tensor channels, except channel 0 (background)
    iou_list = []

    try:
        for i in range(1, n):
            # Extracts a mask per channel
            _out, _labs = outputs[:, i, :, :], labels[:, i, :, :]

            # Computes the intersection over axis 1 and 2
            intersection = (_out & _labs).float().sum(axis=(1, 2))

            # Computes the union over axis 1 and 2
            union = (_out | _labs).float().sum(axis=(1, 2))

            # Computes the IoU score and inserts the value into a list
            iou = (intersection + smooth) / (union + smooth)
            iou_list.append(iou.mean().item())
    except Exception as e:
        print('Exception: %e' % str(e))

    # Returns the mean value of the IoU scores
    return np.mean(iou_list)


def fit(model, device, dataloader, epochs=100, lr=1e-3):
    """
    Function to train the model and save the best and the last weights under a selected metric
    Args:
        model (torch model): NN model
        device (torch device): device (CPU or GPU)
        dataloader (torch dataloader): loads images and labels
        epochs (int): number of epochs for training
        lr (float): learning rate

    Returns:
        a dictionary with the history training
    """

    # Loss functions
    criterion1 = smp.losses.DiceLoss('binary', from_logits=True)
    criterion2 = smp.losses.SoftBCEWithLogitsLoss()

    # Optimizer
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )

    # Puts the model in a device
    model.to(device)

    # Training history dictionary
    hist = {'loss': [], 'iou': [], 'val_loss': [], 'val_iou': []}

    # Initial validation loss reference
    val_loss_ref = 1.0

    try:
        # Training loop
        for epoch in range(1, epochs + 1):
            # Loads images and labels
            bar = tqdm(dataloader['train'])

            # Initializes the loss and IoU values for training
            train_loss, train_iou = [], []

            # Sets the model to train mode
            model.train()

            for images, masks in bar:
                # Puts the images and the labels in the device
                images, masks = images.to(device), masks.to(device)

                # Sets the gradients to None
                optimizer.zero_grad()

                # Model predictions
                y_hat = model(images)

                # Computes the training loss
                loss = criterion1(y_hat, masks.float()) + criterion2(y_hat, masks.float())
                # loss = criterion1(y_hat, masks.float())

                # Performs the network backpropagation
                loss.backward()

                # Performs the parameter update
                optimizer.step()

                # Computes the training parameters for the current batch
                iou = compute_iou(y_hat, masks)
                train_loss.append(loss.item())
                train_iou.append(iou)
                bar.set_description("loss=%.5f iou=%.5f" % (float(np.mean(train_loss)), float(np.mean(train_iou))))
                sleep(0.1)

            # Stores the training parameters for the current epoch
            hist['loss'].append(np.mean(train_loss))
            hist['iou'].append(np.mean(train_iou))
            bar = tqdm(dataloader['val'])

            # Initializes the loss and IoU values for validation
            val_loss, val_iou = [], []

            # Sets the model to evaluation mode
            model.eval()

            # Validation loop
            with torch.no_grad():
                for images, masks in bar:
                    # Puts the images and the labels in the device
                    images, masks = images.to(device), masks.to(device)

                    # Model predictions
                    y_hat = model(images)

                    # Computes the validation loss
                    loss = criterion1(y_hat, masks.float()) + criterion2(y_hat, masks.float())
                    # loss = criterion1(y_hat, masks.float())

                    # Stores the validation parameters for the current batch
                    iou = compute_iou(y_hat, masks)
                    val_loss.append(loss.item())
                    val_iou.append(iou)
                    bar.set_description("val_loss=%.5f val_iou=%.5f" %
                                        (float(np.mean(val_loss)), float(np.mean(val_iou))))
                    sleep(0.1)

            # Stores the validation parameters for the current epoch
            hist['val_loss'].append(np.mean(val_loss))
            hist['val_iou'].append(np.mean(val_iou))

            # Prints the results at ending the epoch
            print('Epoch %d/%d loss=%.5f iou=%.5f val_loss=%.5f val_iou=%.5f' %
                  (epoch,
                   epochs,
                   float(np.mean(train_loss)),
                   float(np.mean(train_iou)),
                   float(np.mean(val_loss)),
                   float(np.mean(val_iou))))

            # Saves the best model
            if np.mean(val_loss) < val_loss_ref:
                print('Saving model...')
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, BEST_WEIGHT_FILE))
                val_loss_ref = np.mean(val_loss)

            # Applies the learning rate scheduler policy
            scheduler.step(np.mean(val_loss))
            print()
            sleep(0.1)

        # Saves the final model
        print('Saving model...')
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, LAST_WEIGHT_FILE))
    except Exception as e:
        print('Exception: %s' % e)

    return hist


def main():
    """
    Classes:
        0- Background
        1- Cytoplasm
        2- Perivitelline area
        3- Zona pellucida
        4- Polar body
    """

    # Creates a directory for the current experiment (if the directory exists, the experiment is aborted)
    try:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        else:
            sys.exit('Path to %s already exists. Please, assign another directory for the experiment results' %
                     RESULTS_DIR)
    except Exception as e:
        sys.exit('Exception: %s' % str(e))

    # Composes the image transformations
    transform_img = tvt.Compose([tvt.ToPILImage(),
                                 tvt.Resize((HEIGHT, WIDTH),
                                            interpolation=tvt.InterpolationMode.BILINEAR),
                                 # tvt.RandomRotation(ANGLE),
                                 tvt.RandomAffine(degrees=(-15, 15),
                                                  shear=(-15, 15),
                                                  interpolation=tvt.InterpolationMode.BILINEAR),
                                 # tvt.RandomVerticalFlip(p=0.5),
                                 tvt.PILToTensor()
                                 ])

    # Prepares the image transformations in a pipeline
    transform_mask = tvt.Compose([tvt.ToTensor(),
                                  tvt.Resize((HEIGHT, WIDTH), tvt.InterpolationMode.NEAREST),
                                  # tvt.RandomRotation(ANGLE),
                                  tvt.RandomAffine(degrees=(-15, 15),
                                                   shear=(-15, 15),
                                                   interpolation=tvt.InterpolationMode.NEAREST),
                                  # tvt.RandomVerticalFlip(p=0.5)
                                  ])

    # Gets the image/mask names
    img_names = sorted(glob(os.path.join(DATA_DIR, IMG_DIR, '*.jpg')))
    mask_names = sorted(glob(os.path.join(DATA_DIR, MASK_DIR, '*.npy')))

    # Gets the number of images and masks
    n_img = len(img_names)
    n_mask = len(mask_names)

    # Compares the images/masks sizes
    if n_img != n_mask:
        sys.exit('Error: the number of inputs and targets are different')

    # Sets the train data size
    train_size = round(n_img * TRAIN_DATA_FACTOR)

    # Gets the train/validation datasets
    dataset = {
        'train': CustomDataset(img_names[:train_size],
                               mask_names[:train_size],
                               N_CLASSES,
                               transform_img,
                               transform_mask),
        'val': CustomDataset(img_names[train_size:],
                             mask_names[train_size:],
                             N_CLASSES,
                             transform_img,
                             transform_mask)
    }

    # Configures the train/validation dataloaders
    dataloader = {
        'train': DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True, pin_memory=True),
        'val': DataLoader(dataset['val'], batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    }

    # Sets the device (CPU or GPU)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Creates the NN model
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights='imagenet',
        in_channels=N_CHANNELS,
        classes=N_CLASSES,
        # activation='sigmoid'
    )

    # Trains the model
    try:
        hist = fit(model, device, dataloader, epochs=N_EPOCHS, lr=LR)
        df = pd.DataFrame(hist)
        df.to_csv(os.path.join(RESULTS_DIR, TRAIN_CSV_FILE), sep=',', index=False, encoding='latin1')
        title = 'Model: %s  Encoder: %s  Loss function: %s\n Input image format: %s' % \
                (MODEL_NAME, ENCODER_NAME, LOSS_FUNCTION_NAME, IMG_FORMAT)
        plt.tight_layout()
        df.plot(grid=True, title=title, xlabel='Epoch', ylabel='Loss / IoU score', figsize=(10, 8))
        # plt.show()
        plt.savefig(os.path.join(RESULTS_DIR, TRAIN_IMG_FILE))
    except Exception as e:
        sys.exit('Exception: %s' % str(e))


if __name__ == '__main__':
    # Main parameters
    MODEL_NAME = 'Unet'
    ENCODER_NAME = 'resnet101'
    LOSS_FUNCTION_NAME = 'Dice + SoftBCEWithLogits'
    IMG_FORMAT = 'grayscale'
    DATA_DIR = 'data'
    IMG_DIR = 'images'
    MASK_DIR = 'masks'
    WEIGHT_DIR = 'weights'
    BEST_WEIGHT_FILE = 'best.pt'
    LAST_WEIGHT_FILE = 'last.pt'
    RUN_DIR = 'runs'
    TRAIN_DIR = 'train'
    TEST_DIR = 'test'
    TRAIN_CSV_FILE = 'train.csv'
    TRAIN_IMG_FILE = 'train.jpg'
    WIDTH = 256  # image width
    HEIGHT = 256  # image height
    LR = 1e-3
    BATCH_SIZE = 10
    N_EPOCHS = 50
    N_CLASSES = 5
    N_CHANNELS = 1
    ANGLE = 15
    N_EXP = 1
    TRAIN_DATA_FACTOR = 0.8
    RESULTS_DIR = os.path.join(RUN_DIR, TRAIN_DIR, str(N_EXP))
    # SEED = 42
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    # Main function call
    main()
