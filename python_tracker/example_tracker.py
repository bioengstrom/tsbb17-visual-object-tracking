#!/usr/bin/env python3

import cv2

import numpy as np

from cvl.dataset import OnlineTrackingBenchmark
from cvl.dataset import BoundingBox
from cvl.trackers import NCCTracker
from cvl.trackers import MOSSE_DCF
from matplotlib import pyplot as plt
import matplotlib.colors

dataset_path = "/courses/TSBB17/otb_mini"

SHOW_BOUNDING_BOX = True
SHOW_SEARCH_REGION = False
SEQUENCE_IDXS = [1, 2]
mode = "COLOR"

if __name__ == "__main__":

    dataset = OnlineTrackingBenchmark(dataset_path)
    
    # For evaluation
    per_seq_performance = [ [] for _ in range(len(dataset.sequences))]

    for seq_idx in SEQUENCE_IDXS:
        tracker = MOSSE_DCF()
        a_seq = dataset[seq_idx]
        for frame_idx, frame in enumerate(a_seq):
            print(f"{frame_idx} / {len(a_seq)}", end='\r')
            image_color = frame['image']

            if mode == "GRAY":
                image_grayscale = np.sum(image_color, 2) / 3
                image = image_grayscale
            elif mode == "COLOR":
                image = image_color
            elif mode == "GRADIENTS":
                image_grayscale = np.sum(image_color, 2) / 3
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
                image = matplotlib.colors.rgb_to_hsv(image_color/255)


            #searchRegion = None
            if frame_idx == 0:
                bbox = frame['bounding_box']
                if bbox.width % 2 == 0:
                    bbox.width += 1

                if bbox.height % 2 == 0:
                    bbox.height += 1

                per_seq_performance[seq_idx].append(bbox)

                # Implementing search region
                
                # Changing the parameters individually will grow the search region in a specific direction
                additionalPixelsX = 50 # Number of additional pixels per side in the x direction
                additionalPixelsY = 30 # Number of additional pixels per side in the y direction
                searchRegionPosX = bbox.xpos - additionalPixelsX
                searchRegionPosY = bbox.ypos - additionalPixelsY
                searchRegionWidth = bbox.width + 2*additionalPixelsX
                searchRegionHeight = bbox.height + 2*additionalPixelsY

                searchRegion = BoundingBox('tl-size', searchRegionPosX, searchRegionPosY, searchRegionWidth, searchRegionHeight)

                tracker.start(image, bbox, searchRegion)
            else:
                per_seq_performance[seq_idx].append(tracker.update(image))

            if SHOW_BOUNDING_BOX and not SHOW_SEARCH_REGION:
                bbox = tracker.boundingBox
                pt0 = (bbox.xpos, bbox.ypos)
                pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
                image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
                cv2.imshow("Bounding box", image_color)
                cv2.waitKey(0)

            if SHOW_SEARCH_REGION and not SHOW_BOUNDING_BOX:
                search_region = tracker.searchRegion
                pt0search = (searchRegion.xpos, searchRegion.ypos)
                pt1search = (searchRegion.xpos + searchRegion.width, searchRegion.ypos + searchRegion.height)
                image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_color, pt0search, pt1search, color=(0, 0, 255), thickness=3)
                cv2.imshow("Search region", image_color)

                cv2.waitKey(0)

            if SHOW_SEARCH_REGION and SHOW_BOUNDING_BOX:
                image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)

                bbox = tracker.boundingBox
                pt0 = (bbox.xpos, bbox.ypos)
                pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
                cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
                
                search_region = tracker.searchRegion
                pt0search = (searchRegion.xpos, searchRegion.ypos)
                pt1search = (searchRegion.xpos + searchRegion.width, searchRegion.ypos + searchRegion.height)
                cv2.rectangle(image_color, pt0search, pt1search, color=(0, 0, 255), thickness=3)

                cv2.imshow("Search region and bounding box", image_color)
                cv2.waitKey(0)

    # Tracking complete, calculate performance
    per_seq_auc = dataset.calculate_performance(per_seq_performance, SEQUENCE_IDXS)



            
