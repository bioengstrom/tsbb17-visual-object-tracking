#!/usr/bin/env python3

import cv2

import numpy as np

from cvl.dataset import OnlineTrackingBenchmark
from cvl.dataset import BoundingBox
from cvl.trackers import NCCTracker
from cvl.trackers import MOSSE_DCF
from cvl.trackers import MOSSE_DEEP
from cvl.features import alexnetFeatures
from matplotlib import pyplot as plt
import matplotlib.colors

import torch
from PIL import Image
from torchvision import transforms

dataset_path = "/courses/TSBB17/otb_mini"

SHOW_BOUNDING_BOX = False
SHOW_SEARCH_REGION = False
SEQUENCE_IDXS = [7]
mode = "COLOR"

if __name__ == "__main__":
    model = alexnetFeatures(pretrained=True)
    #First conv and RELU
    first_layer = model.features[0:2]
    dataset = OnlineTrackingBenchmark(dataset_path)
    
    # For evaluation
    per_seq_performance = [ [] for _ in range(len(dataset.sequences))]

    for seq_idx in SEQUENCE_IDXS:
        tracker = MOSSE_DEEP()
        a_seq = dataset[seq_idx]
        for frame_idx, frame in enumerate(a_seq):
            print(f"{frame_idx} / {len(a_seq)}", end='\r')
            image = frame['image']

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
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image, pt0, pt1, color=(0, 255, 0), thickness=3)
                cv2.imshow("Bounding box", image)
                cv2.waitKey(0)

            if SHOW_SEARCH_REGION and not SHOW_BOUNDING_BOX:
                search_region = tracker.searchRegion
                pt0search = (searchRegion.xpos, searchRegion.ypos)
                pt1search = (searchRegion.xpos + searchRegion.width, searchRegion.ypos + searchRegion.height)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image, pt0search, pt1search, color=(0, 0, 255), thickness=3)
                cv2.imshow("Search region", image)

                cv2.waitKey(0)

            if SHOW_SEARCH_REGION and SHOW_BOUNDING_BOX:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                bbox = tracker.boundingBox
                pt0 = (bbox.xpos, bbox.ypos)
                pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
                cv2.rectangle(image, pt0, pt1, color=(0, 255, 0), thickness=3)
                
                search_region = tracker.searchRegion
                pt0search = (searchRegion.xpos, searchRegion.ypos)
                pt1search = (searchRegion.xpos + searchRegion.width, searchRegion.ypos + searchRegion.height)
                cv2.rectangle(image, pt0search, pt1search, color=(0, 0, 255), thickness=3)

                cv2.imshow("Search region and bounding box", image)
                cv2.waitKey(0)

    # Tracking complete, calculate performance
    per_seq_auc = dataset.calculate_performance(per_seq_performance, SEQUENCE_IDXS)
    print(per_seq_auc)



            
