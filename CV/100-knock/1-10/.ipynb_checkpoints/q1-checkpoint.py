import cv2
import numpy as np
import matplotlib.pyplot as plt

# チャネル入れ替え


def BGR2RGB(img):
    blue = img[:, :, 0].copy()
    green = img[:, :, 1].copy()
    red = img[:, :, 2].copy()

    return [red, blue, green]


if __name__ == '__main__':
    img = cv2.imread("imori.jpg")
    print("jjjjhj")
    plt.imshow(img)
