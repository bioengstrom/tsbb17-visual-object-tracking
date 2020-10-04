#!/usr/bin/env python3

import cv2

import numpy as np

from cvl.dataset import OnlineTrackingBenchmark
from cvl.dataset import BoundingBox
from cvl.trackers import NCCTracker
from cvl.trackers import MOSSE_DCF
from cvl.features import colornames_image
from matplotlib import pyplot as plt
import matplotlib.colors
import copy

dataset_path = "/courses/TSBB17/otb_mini"

SHOW_BOUNDING_BOX = True
SHOW_SEARCH_REGION = False

SEQUENCE_IDXS = [23]

TRACKERS = [MOSSE_DCF, NCCTracker]
Legends = ["MOSSE_DCF", "NCCTracker"] # Used for legends

mode = "COLORNAMES"

if __name__ == "__main__":

    dataset = OnlineTrackingBenchmark(dataset_path)

    #per_seq_performance = [ [] for _ in range(len(dataset.sequences))]
    #seq_performance = [ [] for _ in range(len(dataset.sequences))]
    
    tracker_seq_performance = [ [] for _ in range(len(TRACKERS))]
    for i in range(len(TRACKERS)):
        for _ in range(len(dataset.sequences)):
            tracker_seq_performance[i] = [ [] for _ in range(len(dataset.sequences))]

    
    per_tracker_performance = [ [] for _ in range(len(TRACKERS))]

    for seq_idx in SEQUENCE_IDXS:
        bboxStartX = dataset[seq_idx][1]['bounding_box'].xpos
        bboxStartY = dataset[seq_idx][1]['bounding_box'].ypos

        bboxStartW = dataset[seq_idx][1]['bounding_box'].width
        bboxStartH = dataset[seq_idx][1]['bounding_box'].height

        bboxStart = BoundingBox('tl-size', bboxStartX, bboxStartY, bboxStartW, bboxStartH)

        for seqTracker in range(len(TRACKERS)):
            tracker = TRACKERS[seqTracker]()
            a_seq = dataset[seq_idx]
            for frame_idx, frame in enumerate(a_seq):
                
                print(f"{frame_idx} / {len(a_seq)}", end='\r')
                image_color = frame['image']
                image_grayscale = np.sum(image_color, 2) / 3

                if mode == "GRAY" or isinstance(tracker, NCCTracker):
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
                    image = matplotlib.colors.rgb_to_hsv(image_color/255)

                elif mode == "COLORNAMES":
                    image_colornames = colornames_image(image_color, mode='probability')
                    image = np.concatenate((np.expand_dims(image_grayscale, 2), image_colornames), axis=2)


                #searchRegion = None
                #print(frame['bounding_box'])
                if frame_idx == 0:
                    bbox = copy.deepcopy(bboxStart)
                    if bbox.width % 2 == 0:
                        bbox.width += 1

                    if bbox.height % 2 == 0:
                        bbox.height += 1

                    tracker_seq_performance[seqTracker][seq_idx].append(bbox)

                    # Implementing search region
                    
                    # Changing the parameters individually will grow the search region in a specific direction
                    if isinstance(tracker, MOSSE_DCF):
                        additionalPixelsX = 20 # Number of additional pixels per side in the x direction
                        additionalPixelsY = 20 # Number of additional pixels per side in the y direction
                        searchRegionPosX = bbox.xpos - additionalPixelsX
                        searchRegionPosY = bbox.ypos - additionalPixelsY
                        searchRegionWidth = bbox.width + 2*additionalPixelsX
                        searchRegionHeight = bbox.height + 2*additionalPixelsY

                        searchRegion = BoundingBox('tl-size', searchRegionPosX, searchRegionPosY, searchRegionWidth, searchRegionHeight)

                        tracker.start(image, bbox, searchRegion)
                    else:
                        tracker.start(image, bbox)
                else:
                    tracker_seq_performance[seqTracker][seq_idx].append(tracker.update(image))

                if SHOW_BOUNDING_BOX and not SHOW_SEARCH_REGION:
                    bbox = tracker.boundingBox
                    pt0 = (bbox.xpos, bbox.ypos)
                    pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
                    image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
                    cv2.imshow("Bounding box", image_color)
                    cv2.waitKey(20)

                elif SHOW_SEARCH_REGION and not SHOW_BOUNDING_BOX:
                    search_region = tracker.searchRegion
                    pt0search = (searchRegion.xpos, searchRegion.ypos)
                    pt1search = (searchRegion.xpos + searchRegion.width, searchRegion.ypos + searchRegion.height)
                    image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(image_color, pt0search, pt1search, color=(0, 0, 255), thickness=3)
                    cv2.imshow("Search region", image_color)

                    cv2.waitKey(20)

                elif SHOW_SEARCH_REGION and SHOW_BOUNDING_BOX:
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
                    cv2.waitKey(20)

            # Tracking complete, calculate performance
            per_tracker_performance[seqTracker] = dataset.calculate_performance(tracker_seq_performance[seqTracker], SEQUENCE_IDXS)

        #print(per_tracker_performance)
        #print(per_tracker_performance[0])
        print("\n")
        for t in range(len(per_tracker_performance)):
            for s in range(len(SEQUENCE_IDXS)):
                plt.plot(per_tracker_performance[t][s], label=Legends[t])
                plt.legend()
        
        plt.show()
        #plt.plot(per_tracker_performance[0])
        #for tracker in per_tracker_performance:
            #print(tracker)
            #plt.plot(tracker)

        #plt.show()



            
