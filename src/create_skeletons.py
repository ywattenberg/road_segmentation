import os
import sys
import argparse
from PIL import Image
import numpy as np
from multiprocessing import Pool
from Loss.cldice import opencv_skelitonize

def skeletonize_image(image_path, skeleton_dir):
    image = np.array(Image.open(image_path))
    image = image[:,:,0]
    skeleton = opencv_skelitonize(image)
    skeleton = skeleton.astype(np.uint8)
    skeleton = Image.fromarray(skeleton)
    skeleton.save(os.path.join(skeleton_dir, image_path.split('/')[-1]))
    print(f'Skeletonized {image_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skeletonize images')
    parser.add_argument('--mask_dir', type=str, help='Path to mask folder')
    parser.add_argument('--skeleton_dir', type=str, help='Path to skeleton folder')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers to use')
    args = parser.parse_args()

    mask_dir = os.path.abspath(args.mask_dir)
    skeleton_dir = os.path.abspath(args.skeleton_dir)
    num_workers = args.num_workers

    if not os.path.exists(skeleton_dir):
        os.makedirs(skeleton_dir)
    
    print(f'Skeletonizing images in {mask_dir}')
    print(f'Saving to {skeleton_dir}')

    mask_list = os.listdir(mask_dir)
    mask_list = [os.path.join(mask_dir, mask) for mask in mask_list]
    with Pool(num_workers) as p:
        p.starmap(skeletonize_image, zip(mask_list, [skeleton_dir]*len(mask_list)))


