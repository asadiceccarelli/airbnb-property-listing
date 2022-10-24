import logging
import glob
import math
import cv2
import os
import pandas as pd
from torchvision import datasets, transforms

logging.basicConfig(level=logging.INFO)

def resize_images():
    """Resizes each image to the identical dimensions. The height of the
    smallest image is used and the rest are resized to this height. Black
    padding is added to the width of photos to ensure aspect ratio is
    maintained. Discards any images not in RGB format.
    """
    logging.info('Calculating smallest image height...')
    all_paths = glob.glob('project/data/unstructured/images/*/*')
    h_min = math.inf
    for path in all_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        if h < h_min:
            h_min = h
    logging.info(f'Smallest image height: {h_min}')
    logging.info('Resizing images and adding padding...')
    listing_paths = glob.glob('project/data/unstructured/images/*')
    for listing_path in listing_paths:
        listing_id = listing_path.split('/')[-1]
        folder_name = f'project/data/unstructured/processed_images/{listing_id}'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        image_paths = glob.glob(f'{listing_path}/*')
        for image_path in image_paths:
            photo_id = image_path.split('/')[-1]
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3:
                h, w = img.shape[:2]
                scale = h_min / h
                dim = (int(w*scale), h_min)
                scaled_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                pad = w - dim[0]
                if pad % 2 != 0:
                    pad_left = int(pad // 2)
                    pad_right = int((pad // 2) + 1)
                else:
                    pad_left = int(pad / 2)
                    pad_right = int(pad / 2)
                padded_img = cv2.copyMakeBorder(scaled_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])
                print(f'padded_dim = {padded_img.shape}\n')
                cv2.imwrite(f'project/data/unstructured/processed_images/{listing_id}/{photo_id}', padded_img)


def get_image_data():
    """Creates a dataframe with the ID and five photos as the columns.
    Iterates throught the images of each listing and upserts the ID and
    image RGB tensors to the dataframe, before exporting as a .csv file.
    """
    df = pd.DataFrame(columns=['ID', 'Photo 1', 'Photo 2', 'Photo 3', 'Photo 4', 'Photo 5'])
    listing_paths = glob.glob('project/data/unstructured/processed_images/*')
    for listing_path in listing_paths:
        id = listing_path.split('/')[-1]
        listing_tensors = [id]
        image_paths = glob.glob(f'{listing_path}/*')
        for image_path in image_paths:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            img_tensor = transforms.ToTensor()(img)
            listing_tensors.append(img_tensor)
        df = pd.concat([pd.DataFrame([listing_tensors], columns=df.columns), df], ignore_index=True)
    df.to_csv('project/data/unstructured/listing_df.csv')


if __name__ == '__main__':
    # resize_images()
    get_image_data()
