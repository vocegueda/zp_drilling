import numpy as np
import cv2
from torch.utils.data import Dataset
from torch import get_rng_state, set_rng_state
from torchvision import transforms as tvt


class CustomDataset(Dataset):
    """
    This class is used to process oocytes images and masks
    """

    def __init__(self, x, y, n_classes=1,
                 transform_img=tvt.Compose([tvt.ToPILImage(),
                                            tvt.Resize((256, 256)),
                                            tvt.PILToTensor()]),
                 transform_mask=tvt.Compose([tvt.ToTensor(),
                                             tvt.Resize((256, 256), tvt.InterpolationMode.NEAREST)
                                             ])):
        self.x = x
        self.y = y
        self.n_classes = n_classes
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ix):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = cv2.imread(self.x[ix], cv2.IMREAD_GRAYSCALE)
        img = clahe.apply(img).astype(np.float32)

        state = get_rng_state()
        img = self.transform_img(img)

        mask = np.load(self.y[ix])
        mask = (np.arange(self.n_classes) == mask[..., None]).astype(np.float32)
        set_rng_state(state)
        mask = self.transform_mask(mask)

        return img, mask
