import cv2
import numpy as np
import os
from skimage.morphology import skeletonize
from copy import deepcopy
import edgeTraceBifurcation

class color_seg:
    def black_seg(self, image):
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 46])
        return cv2.inRange(image, lower, upper)

    def gray_seg(self, image):
        lower = np.array([0, 0, 46])
        upper = np.array([180, 43, 220])
        return cv2.inRange(image, lower, upper)

    def white_seg(self, image):
        lower = np.array([0, 0, 221])
        upper = np.array([180, 30, 255])
        return cv2.inRange(image, lower, upper)

    def red_seg(self, image):
        lower_1 = np.array([0, 43, 46])
        upper_1 = np.array([10, 255, 255])
        red_1 = cv2.inRange(image, lower_1, upper_1)
        lower_2 = np.array([156, 43, 46])
        upper_2 = np.array([180, 255, 255])
        red_2 = cv2.inRange(image, lower_2, upper_2)
        return cv2.add(red_1, red_2)

    def orange_seg(self, image):
        lower = np.array([11, 43, 46])
        upper = np.array([25, 255, 255])
        return cv2.inRange(image, lower, upper)

    def yellow_seg(self, image):
        lower = np.array([26, 43, 46])
        upper = np.array([34, 255, 255])
        return cv2.inRange(image, lower, upper)

    def green_seg(self, image):
        lower = np.array([35, 43, 46])
        upper = np.array([77, 255, 255])
        return cv2.inRange(image, lower, upper)

    def cyan_seg(self, image):
        lower = np.array([78, 43, 46])
        upper = np.array([99, 255, 255])
        return cv2.inRange(image, lower, upper)

    def blue_seg(self, image):
        lower = np.array([100, 43, 46])
        upper = np.array([124, 255, 255])
        return cv2.inRange(image, lower, upper)

    def purple_seg(self, image):
        lower = np.array([125, 43, 46])
        upper = np.array([155, 255, 255])
        return cv2.inRange(image, lower, upper)

    def color_segmentation(self, image):
        black_area = self.black_seg(image)
        #black_area = cv2.bitwise_and(image, image, mask=black_area)
        black_area = cv2.bitwise_and(np.ones_like(image)*255, np.ones_like(image)*255, mask=black_area)
        #black_area = cv2.cvtColor(black_area, cv2.COLOR_HSV2BGR)
        cv2.imwrite("./Color_block/black.png", black_area)
        gray_area = self.gray_seg(image)
        gray_area = cv2.bitwise_and(image, image, mask=gray_area)
        gray_area = cv2.cvtColor(gray_area, cv2.COLOR_HSV2BGR)
        cv2.imwrite("./Color_block/gray.png", gray_area)
        white_area = self.white_seg(image)
        white_area = cv2.bitwise_and(image, image, mask=white_area)
        white_area = cv2.cvtColor(white_area, cv2.COLOR_HSV2BGR)
        cv2.imwrite("./Color_block/white.png", white_area)
        red_area = self.red_seg(image)
        red_area = cv2.bitwise_and(image, image, mask=red_area)
        red_area = cv2.cvtColor(red_area, cv2.COLOR_HSV2BGR)
        cv2.imwrite("./Color_block/red.png", red_area)
        orange_area = self.orange_seg(image)
        orange_area = cv2.bitwise_and(image, image, mask=orange_area)
        orange_area = cv2.cvtColor(orange_area, cv2.COLOR_HSV2BGR)
        cv2.imwrite("./Color_block/orange.png", orange_area)
        yellow_area = self.yellow_seg(image)
        yellow_area = cv2.bitwise_and(image, image, mask=yellow_area)
        yellow_area = cv2.cvtColor(yellow_area, cv2.COLOR_HSV2BGR)
        cv2.imwrite("./Color_block/yellow.png", yellow_area)
        green_area = self.green_seg(image)
        green_area = cv2.bitwise_and(image, image, mask=green_area)
        green_area = cv2.cvtColor(green_area, cv2.COLOR_HSV2BGR)
        cv2.imwrite("./Color_block/green.png", green_area)
        cyan_area = self.cyan_seg(image)
        cyan_area = cv2.bitwise_and(image, image, mask=cyan_area)
        cyan_area = cv2.cvtColor(cyan_area, cv2.COLOR_HSV2BGR)
        cv2.imwrite("./Color_block/cyan.png", cyan_area)
        blue_area = self.blue_seg(image)
        blue_area = cv2.bitwise_and(image, image, mask=blue_area)
        blue_area = cv2.cvtColor(blue_area, cv2.COLOR_HSV2BGR)
        cv2.imwrite("./Color_block/blue.png", blue_area)
        purple_area = self.purple_seg(image)
        purple_area = cv2.bitwise_and(image, image, mask=purple_area)
        purple_area = cv2.cvtColor(purple_area, cv2.COLOR_HSV2BGR)
        cv2.imwrite("./Color_block/purple.png", purple_area)

    def seg(self, image):
        img = deepcopy(image)
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.color_segmentation(img_HSV)


def seg_s(max_hw, colors):
    sigma = 0.33
    c = 20
    k_size = int(max_hw/c) if int(max_hw/c)%2==1 else int(max_hw/c) +1
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    for color in colors:
        tmp = cv2.imread("./Color_block/" + color + ".png", 1)
        gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        v = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        retval, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        eroded = deepcopy(binimg)
        i = 0
        while True:
            contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                break
            edge = cv2.Canny(eroded, 50, 80)
            edge_skel = skeletonize(edge, method="zhang")
            edge_skel = np.array(edge_skel, dtype=np.int8) * 255
            if edgeTraceBifurcation.trace(edge_skel, i, color) == False:
                break
            cv2.imwrite(os.path.join("Color_block_layers", color, str(i) + ".png"), edge_skel)
            eroded = cv2.erode(eroded, erode_kernel, iterations=1)
            i += 1


def mkdir(colors):
    if os.path.exists('./Color_block_layers') is False:
        os.mkdir('./Color_block_layers')
        for f in colors:
            os.mkdir(os.path.join("./Color_block_layers",f))
    elif len(os.listdir('./Color_block_layers')) >= 0:
        folder_list = os.listdir('./Color_block_layers')
        for f in colors:
            if f in folder_list:
                file_list = os.listdir(os.path.join("./Color_block_layers", f))
                for ff in file_list:
                    os.unlink(os.path.join('./Color_block_layers',f,ff))
            else:
                os.mkdir(os.path.join("./Color_block_layers",f))


def main():
    colors = ["black", "blue", "cyan", "green", "orange", "purple", "red", "yellow"]
    img = cv2.imread("src.png", 1)
    max_hw = max(img.shape)
    mkdir(colors)
    color_seg().seg(img)
    seg_s(max_hw, colors)


if __name__ == "__main__":
    main()
