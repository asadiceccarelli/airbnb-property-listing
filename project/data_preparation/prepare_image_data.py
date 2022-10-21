import logging
import glob
import cv2
import os

logging.basicConfig(level=logging.INFO)

def resize_images():
    """Resizes each image to the smallest
    """
    all_paths = glob.glob('project/data/unstructured/images/*/*')
    heights = []
    for path in all_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        h, w, c = img.shape[:2]
        heights.append(h)
    min_height = min(heights)
    listing_paths = glob.glob('project/data/unstructured/images/*')
    for listing_path in listing_paths:
        listing_id = listing_path.split('/')[-1]
        os.mkdir(f'project/data/unstructured/processed_images/{listing_id}')
        image_paths = glob.glob(f'{listing_path}/*')
        for image_path in image_paths:
            photo_id = image_path.split('/')[-1]
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            h, w, c = img.shape
            if c == 3:
                scale = min_height / h
                dim = (int(w*scale), min_height)
                resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(f'project/data/unstructured/processed_images/{listing_id}/{photo_id}', resized_img)
        return


if __name__ == '__main__':
    resize_images()
