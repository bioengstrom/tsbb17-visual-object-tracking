#!/usr/bin/env python3

import cv2

import numpy as np

from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers import NCCTracker
from cvl.trackers import MOSSE_DCF
from matplotlib import pyplot as plt
import matplotlib.colors

dataset_path = "/courses/TSBB17/otb_mini"

SHOW_TRACKING = True
SEQUENCE_IDX = 3
mode = "HSV"

if __name__ == "__main__":

    dataset = OnlineTrackingBenchmark(dataset_path)

    a_seq = dataset[SEQUENCE_IDX]

    if SHOW_TRACKING:
        cv2.namedWindow("tracker")

    tracker = MOSSE_DCF()

    for frame_idx, frame in enumerate(a_seq):
        print(f"{frame_idx} / {len(a_seq)}")
        image_color = frame['image']
        image_grayscale = np.sum(image_color, 2) / 3
        image_hsv = matplotlib.colors.rgb_to_hsv(image_color/255)


        if mode == "GRAY":
            image = image_grayscale
        elif mode == "COLOR":
            image = image_color
        elif mode == "GRADIENTS":
            #laplacian = np.expand_dims(cv2.Laplacian(image_grayscale, cv2.CV_64F), 2)
            sobelx = np.expand_dims(cv2.Sobel(image_grayscale, cv2.CV_64F, 1, 0, ksize=5), 2)
            sobely = np.expand_dims(cv2.Sobel(image_grayscale, cv2.CV_64F, 0, 1, ksize=5), 2)
            canny = cv2.Canny(np.uint8(image_grayscale), 100, 300)
            #plt.imshow(canny, cmap="gray")
            #plt.imshow(image_grayscale)
            #plt.show()
            image = np.concatenate((sobely, sobelx), axis=2)
            #image = canny
        elif mode == "HSV":
            image = image_hsv



        if frame_idx == 0:
            bbox = frame['bounding_box']
            if bbox.width % 2 == 0:
                bbox.width += 1

            if bbox.height % 2 == 0:
                bbox.height += 1

            current_position = bbox
            tracker.start(image, bbox)
        else:
            tracker.update(image)

        if SHOW_TRACKING:
            bbox = tracker.region
            pt0 = (bbox.xpos, bbox.ypos)
            pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
            image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
            cv2.imshow("tracker", image_color)
            cv2.waitKey(0)
