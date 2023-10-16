import numpy as np
import cv2
from skimage import measure
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms as tvt
from PIL import Image
import argparse
import os
import sys
import json


def main():
    """
    Classes:
        0- Background
        1- Cytoplasm
        2- Perivitelline area
        3- Zona pellucida
        4- Polar body
    """

    # Sets the paths to detection and segmentation models
    det_weights_path = os.path.join(WEIGHT_DIR, DETECTION_WEIGHTS)
    seg_weights_path = os.path.join(WEIGHT_DIR, SEGMENTATION_WEIGHTS)

    # Gets the user parameters
    parser = argparse.ArgumentParser(description='script to compute some points of interest from a segmented oocyte')
    parser.add_argument('--input', type=str, required=True, help='path/to/image')
    parser.add_argument('--output', type=str, required=True, help='path/to/json_file')

    # Parses the user parameters
    args = parser.parse_args()
    img_path = args.input
    json_path = args.output

    dst_path = os.path.dirname(json_path)

    if dst_path.strip() == "":
        dst_path = './'

    # Performs some path verifications
    if not os.path.exists(det_weights_path):
        sys.exit('Error: path to "%s" does not exist' % det_weights_path)

    if not os.path.exists(seg_weights_path):
        sys.exit('Error: path to "%s" does not exist' % seg_weights_path)

    if not os.path.exists(img_path):
        sys.exit('Error: path to "%s" does not exist' % img_path)

    if not os.path.exists(dst_path):
        sys.exit('Error: directory "%s" does not exist' % dst_path)

    # Sets other parameters
    json_file = os.path.basename(json_path)
    base_name = json_file.split('.')[0]

    if json_file.strip() == "":
        sys.exit('Error: missing JSON filename')

    # Prepares the image transformations in a pipeline
    transform_img = tvt.Compose([tvt.ToPILImage(),
                                 tvt.Resize((HEIGHT, WIDTH), interpolation=tvt.InterpolationMode.BILINEAR),
                                 tvt.PILToTensor()])

    try:
        # Sets the device (CPU or GPU)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Creates the detection model
        print('Loading the detection model...')
        det_model = torch.hub.load('.',
                                   'custom',
                                   path=det_weights_path,
                                   force_reload=True,
                                   source='local')

        # Initializes the detection model
        det_model.iou = 0.6  # IoU threshold
        det_model.conf = 0.6  # Confidence threshold
        det_model.max_det = 1  # Max number of detections allowed per frame
        print('Done\n')
        
        # Detection model predictions
        results = det_model(img_path)
        detections = results.xyxy[0].detach().cpu().numpy()

        # Checks if there are oocyte predictions
        if len(detections) > 0:
            # Extracts the bounding box (we are considering that only one object is detected)
            box = detections[0][:4]

            # Extracts the upper-left corner, width, and height from the bounding box
            x, y, w, h = round(box[0]), round(box[1]), round(box[2] - box[0]), round(box[3] - box[1])
        else:
            sys.exit('No oocyte detection')

        # Creates the segmentation model
        print('Loading the segmentation model...')

        seg_model = smp.Unet(
                encoder_name="resnet101",
                encoder_weights='imagenet',
                in_channels=N_CHANNELS,
                classes=N_CLASSES
                )

        # Puts the model in the device (CPU or GPU)
        seg_model = seg_model.to(device)

        # Loads the trained weights
        seg_model.load_state_dict(torch.load(seg_weights_path))

        # Sets the model to evaluation mode
        seg_model.eval()
        print('Done\n')

        # Reads the image
        color_img = cv2.imread(img_path)

        # Converts the image to grayscale
        img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        # Crops the image using the predicted bounding box
        crop_img = img[y:y + h, x:x + w]
        crop_height = crop_img.shape[0]
        crop_width = crop_img.shape[1]

        # Applies the CLAHE algorithm to the crop
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        crop_img = clahe.apply(crop_img).astype(np.float32)

        # Applies the transformations to the crop
        crop_img = transform_img(crop_img)

        # Puts the transformed tensor in the device (CPU or GPU)
        crop_img = torch.autograd.Variable(crop_img, requires_grad=False).to(device).unsqueeze(0)

        # Segmentation model predictions
        with torch.no_grad():
            prd = seg_model(crop_img)

        # Resizes the predicted mask
        prd = tvt.Resize((crop_height, crop_width), tvt.InterpolationMode.NEAREST)(prd[0])

        # Loads the predicted mask in the CPU as a numpy array
        seg = prd.data.cpu().detach().numpy()

        # Gets a binary mask for each channel
        seg = (seg > 0.5).astype(np.uint8)

        print('Calculating the points of interest on the oocyte and saving the results...')
        # Get the cytoplasm mask (channel 1)
        cy_mask = seg.copy()[1, :, :]
        cy_mask[cy_mask != 1] = 0
        cy_mask = cy_mask.astype(np.uint8)
        cy_size = np.count_nonzero(cy_mask)
        if cy_size == 0:
            sys.exit("Error: the class 'cytoplasm' was not found in the predicted mask")

        # Gets the cytoplasm centroid
        # A point of interest in the input image (not the crop)
        m = measure.moments(cy_mask)
        cx = round(m[0, 1] / m[0, 0])
        cy = round(m[1, 0] / m[0, 0])
        c = [x + cx, y + cy]

        # Gets the zp mask (channel 3)
        zp_mask = seg.copy()[3, :, :]
        zp_mask[zp_mask != 1] = 0
        zp_mask = zp_mask.astype(np.uint8)
        zp_size = np.count_nonzero(zp_mask)
        if zp_size == 0:
            sys.exit("Error: the class 'zona pellucida' was not found in the predicted mask")

        # Gets the coordinates of the zp mask
        coord = cv2.findNonZero(zp_mask)
        coord = np.squeeze(coord, axis=1)
        coord = coord[(coord[:, 0] > cx) & (coord[:, 1] == cy)]

        # Gets the zp mask inner point
        # A point of interest in the input image (not the crop)
        p1 = coord[0].tolist()
        p1[0] += x
        p1[1] += y

        # Gets the zp mask outer point
        # A point of interest in the input image (not the crop)
        p2 = coord[-1].tolist()
        p2[0] += x
        p2[1] += y

        print('cytoplasm_centroid =', c)
        print('zp_inner_point =', p1)
        print('zp_outer_point =', p2)

        # Initializes some data structures to store the results
        data = []
        results = {'data': []}

        d = {'cytoplasm_centroid': c,
             'zp_inner_point': p1,
             'zp_outer_point': p2,
             }

        data.append(d)
        results['data'] = data

        # Dumps the results to a JSON file
        with open(os.path.join(os.path.join(dst_path, base_name + '.json')), 'w') as fp:
            json.dump(results, fp, indent=4)

        # Graph parameters
        '''
        color = np.array([255, 0, 0], dtype='uint8')
        alpha = 0.4
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        crop_img = color_img[y:y + h, x:x + w]

        masked_img = np.where(zp_mask[..., None], color, crop_img)
        out = cv2.addWeighted(color_img[y:y + h, x:x + w], 1 - alpha, masked_img, alpha, 0)
        cv2.circle(out, (c[0] - x, c[1] - y), 5, (0, 255, 0), -1)
        cv2.circle(out, (p1[0] - x, p1[1] - y), 5, (0, 255, 0), -1)
        cv2.circle(out, (p2[0] - x, p2[1] - y), 5, (0, 255, 0), -1)
        out = Image.fromarray(out).convert('RGB')

        crop_img = Image.fromarray(crop_img).convert('RGB')
        '''

        # Plots the image with the computed points of interest
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        cv2.circle(color_img, (c[0], c[1]), 5, (0, 255, 0), -1)
        cv2.circle(color_img, (p1[0], p1[1]), 5, (0, 255, 0), -1)
        cv2.circle(color_img, (p2[0], p2[1]), 5, (0, 255, 0), -1)
        color_img = Image.fromarray(color_img)
        color_img.save(os.path.join(dst_path, base_name + '.jpg'))
        print('Done')

    except Exception as e:
        print('Exception: %s' % str(e))


if __name__ == '__main__':
    # Main parameters
    WIDTH = 256
    HEIGHT = 256
    N_CHANNELS = 1
    N_CLASSES = 5
    WEIGHT_DIR = 'weights'
    DETECTION_WEIGHTS = 'oocyte_det.pt'
    SEGMENTATION_WEIGHTS = 'oocyte_seg.pt'

    # Main function call
    main()
