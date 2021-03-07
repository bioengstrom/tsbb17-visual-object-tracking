#!/usr/bin/env python3

import cv2

import numpy as np

from cvl.dataset import OnlineTrackingBenchmark
from cvl.dataset import BoundingBox
from cvl.trackers import NCCTracker
from cvl.trackers import MOSSE_SCALE
from cvl.trackers import MOSSE_DCF
from cvl.features import colornames_image
from matplotlib import pyplot as plt
import matplotlib.colors
import copy

dataset_path = "/courses/TSBB17/otb_mini"

SHOW_BOUNDING_BOX = False
SHOW_SEARCH_REGION = False

SEQUENCE_IDXS = range(30)

TRACKERS = [MOSSE_DCF, MOSSE_DCF]#, MOSSE_DCF]#, MOSSE_DCF, MOSSE_DCF, NCCTracker]
Legends = ["MOSSE_DCF_GREY+COLORNAMES", "MOSSE_DCF_GREY+COLOR"]#, "MOSSE_DCF_HSV_GRAY"]#, "MOSSE_DCF_COLOR", "MOSSE_DCF_COLORNAMES", "MOSSE_DCF_GRADIENTS", "MOSSE_DCF_HSV", "NCCTracker"] # Used for legends
MODES = ["COLORNAMES", "COLOR", "GRADIENTS", "HSV", "GRAY"]


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
        print("Sequence idx: ", seq_idx)
        bboxStartX = dataset[seq_idx][1]['bounding_box'].xpos
        bboxStartY = dataset[seq_idx][1]['bounding_box'].ypos

        bboxStartW = dataset[seq_idx][1]['bounding_box'].width
        bboxStartH = dataset[seq_idx][1]['bounding_box'].height

        bboxStart = BoundingBox('tl-size', bboxStartX, bboxStartY, bboxStartW, bboxStartH)

        for seqTracker in range(len(TRACKERS)):
            mode = MODES[seqTracker]
            print("Tracker: ", mode)
            tracker = TRACKERS[seqTracker]()
            a_seq = dataset[seq_idx]
            for frame_idx, frame in enumerate(a_seq):
                
                #print(f"{frame_idx} / {len(a_seq)}")#, end='\r')
                image_color = frame['image']
                image_grayscale = np.sum(image_color, 2) / 3
                image_colornames = colornames_image(image_color, mode='probability')
                image_hsv = matplotlib.colors.rgb_to_hsv(image_color/255)
                sobelx = np.expand_dims(cv2.Sobel(image_grayscale, cv2.CV_64F, 1, 0, ksize=5), 2)
                sobely = np.expand_dims(cv2.Sobel(image_grayscale, cv2.CV_64F, 0, 1, ksize=5), 2)
                image_gradients = np.concatenate((sobely, sobelx), axis=2)

                if mode == "GRAY" or isinstance(tracker, NCCTracker):
                    image = image_grayscale
                elif mode == "COLOR":
                    image = np.concatenate((image_color, np.expand_dims(image_grayscale, 2)), axis=2)
                    #image = image_color
                elif mode == "GRADIENTS":
                    #laplacian = np.expand_dims(cv2.Laplacian(image_grayscale, cv2.CV_64F), 2)
                    #plt.imshow(canny, cmap="gray")
                    #plt.imshow(image_grayscale)
                    #plt.show()
                    canny = cv2.Canny(np.uint8(image_grayscale), 100, 300)
                    #image = canny
                    image = image_gradients
                elif mode == "HSV":
                    image = np.concatenate((image_hsv, np.expand_dims(image_grayscale, 2)), axis=2)

                elif mode == "COLORNAMES":
                    image = np.concatenate((image_colornames, np.expand_dims(image_grayscale, 2)), axis=2)
                    #image = np.concatenate((np.expand_dims(image_grayscale, 2), image_colornames), axis=2)

                elif mode == "ALL":
                    image = np.concatenate((image_color, np.expand_dims(image_grayscale, 2), image_colornames, image_hsv, image_gradients), axis=2)


                #searchRegion = None
                #print(frame['bounding_box'])
                region = 0
                if frame_idx == 0:
                    bbox = copy.deepcopy(bboxStart)
                    region = bbox
                    if bbox.width % 2 == 0:
                        bbox.width += 1

                    if bbox.height % 2 == 0:
                        bbox.height += 1

                    tracker_seq_performance[seqTracker][seq_idx].append(bbox)

                    # Implementing search region

                    # Changing the parameters individually will grow the search region in a specific direction
                    if not isinstance(tracker, NCCTracker):
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
                elif image.shape[1] > tracker.boundingBox.xpos > 0    \
                        and image.shape[0] > tracker.boundingBox.ypos > 0: #crash prevention
                    #print(tracker.boundingBox.xpos, tracker.boundingBox.ypos)
                    #print(-tracker.boundingBox.width, -tracker.boundingBox.height)
                    #print(image.shape)
                    region = tracker.update(image)
                    tracker_seq_performance[seqTracker][seq_idx].append(region)

                else:
                    region = BoundingBox('tl-size', 0, 0, 0, 0)
                    tracker_seq_performance[seqTracker][seq_idx].append(region)

                if SHOW_BOUNDING_BOX and not SHOW_SEARCH_REGION:
                    bbox = tracker.boundingBox
                    pt0 = (bbox.xpos, bbox.ypos)
                    pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
                    image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
                    cv2.imshow("Bounding box", image_color)
                    cv2.waitKey(0)

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

                    if not isinstance(tracker, NCCTracker):
                        search_region = tracker.searchRegion
                        pt0search = (search_region.xpos, search_region.ypos)
                        pt1search = (search_region.xpos + search_region.width, search_region.ypos + search_region.height)
                        cv2.rectangle(image_color, pt0search, pt1search, color=(0, 0, 255), thickness=3)

                        scaleRegion = region
                        pts0search = (scaleRegion.xpos, scaleRegion.ypos)
                        pts1search = (scaleRegion.xpos + scaleRegion.width, scaleRegion.ypos + scaleRegion.height)
                        cv2.rectangle(image_color, pts0search, pts1search, color=(255, 0, 0), thickness=3)

                    cv2.imshow("Search region and bounding box", image_color)
                    cv2.waitKey(20)

    # Tracking complete, calculate performance
    for seqTracker in range(len(TRACKERS)):
        per_tracker_performance[seqTracker] = dataset.calculate_performance(tracker_seq_performance[seqTracker], SEQUENCE_IDXS)

    #print(per_tracker_performance)
    #print(per_tracker_performance[0])
    per_tracker_total_performance = [[0 for _ in range(len(SEQUENCE_IDXS))] for _ in range(len(TRACKERS))]
    print("\n")
    for s in range(len(SEQUENCE_IDXS)):
        for t in range(len(per_tracker_performance)):
            per_tracker_total_performance[t][s] += (per_tracker_performance[t][s][-1]/dataset.sequences[s].num_frames)
            #print(dataset.sequences[s].num_frames)
            plt.plot(per_tracker_performance[t][s], label=Legends[t])
            plt.legend()
        plt.xlabel("Frames")
        plt.ylabel("AUC")
        plt.title("Sequence %s" % s)
        plt.savefig("Sequence%s_comb2.png" % s)
        plt.show()

    per_tracker_mean_performance = np.mean(per_tracker_total_performance, 1)
    np.save('per_tracker_mean_performance_all.npy', per_tracker_mean_performance)
    print(per_tracker_total_performance)
    print('DONE')
    #plt.plot(per_tracker_performance[0])
    #for tracker in per_tracker_performance:
        #print(tracker)
        #plt.plot(tracker)

    #plt.show()



            
