import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def opencv_skelitonize(img):
    skel = np.zeros(img.shape, np.uint8)
    img = img.astype(np.uint8)
    size = np.size(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel


if __name__ == "__main__":
    path = "submission/masks"
    files = os.listdir(path)
    mask = cv2.imread(os.path.join(path, files[0]), cv2.IMREAD_GRAYSCALE)
    skel = opencv_skelitonize(np.array(mask))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.array(mask))
    ax[1].imshow(skel)
    fig.savefig("post.png", dpi=300)
