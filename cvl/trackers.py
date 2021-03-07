import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import shift
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from .image_io import crop_patch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from cvl.dataset import BoundingBox
from math import floor
import sys
import torch
import torch.nn.functional as F

from cvl.features import alexnetFeatures
import torch
from PIL import Image
from torchvision import transforms

def hanning(patch):
    hanning_w = np.hanning(patch.shape[0])
    hanning_h = np.hanning(patch.shape[1])
    hanning_2d = np.sqrt(np.outer(hanning_w, hanning_h))
    return patch*hanning_2d

# kernel becomes kernelLength by kernelLength
def fftGuassianKernel(kernelHeight, kernelWidth, peakRow, peakColumn, sigma=2.0): 
    x = np.zeros([kernelHeight, kernelWidth])
    x[peakRow][peakColumn] = 1
    return fft2(gaussian_filter(x, sigma))

def hanning(patch):
    hanning_w = np.hanning(patch.shape[0])
    hanning_h = np.hanning(patch.shape[1])
    hanning_2d = np.sqrt(np.outer(hanning_w, hanning_h))
    return patch*hanning_2d

class NCCTracker:

    def __init__(self, learning_rate=0.1):
        self.template = None 
        self.last_response = None
        self.boundingBox = None
        self.boundingBoxShape = None
        self.region_center = None
        self.learning_rate = learning_rate

    def crop_patch(self, image):
        boundingBox = self.boundingBox
        return crop_patch(image, boundingBox)

    def start(self, image, boundingBox):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        self.boundingBox = boundingBox
        self.boundingBoxShape = (boundingBox.height, boundingBox.width)
        self.region_center = (boundingBox.height // 2, boundingBox.width // 2)

        patch = self.crop_patch(image)
        patch = patch/255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)

        self.template = fft2(patch)

    def detect(self, image):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)

        responsef = self.template * np.conj(patchf)
        response = ifft2(responsef)
        r, c = np.unravel_index(np.argmax(response), response.shape)

        # Keep for visualisation
        self.last_response = response

        r_offset = np.mod(r + self.region_center[0], self.boundingBox.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.boundingBox.width) - self.region_center[1]

        self.boundingBox.xpos += c_offset
        self.boundingBox.ypos += r_offset

        return self.boundingBox

    def update(self, image, lr=0.1):
        self.detect(image)
        
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)
        self.template = self.template * (1 - lr) + patchf * lr

        return copy.deepcopy(self.boundingBox)

class MOSSE_DCF:
    def __init__(self):
        self.boundingBox = None
        self.searchRegion = None
        self.boundingBoxShape = None
        self.boundingBoxCenter = None

        self.gaussianScore = None
        self.A = []
        self.B = 0
        self.M = []
        self.P = None
        self.dims = 1

        self.forgettingFactor = 0.125
        self.sigma = 2.0
        self.regularization = 0.1 # lambda

    #
    def crop_patch(self, image):
        #searchRegion = self.searchRegion
        return crop_patch(image, self.searchRegion)

    def start(self, image, boundingBox, searchRegion):
        if len(image.shape) > 2:
            self.dims = image.shape[2]
        
        self.M = [0 for _ in range(self.dims)]
        self.A = [0 for _ in range(self.dims)]
        self.P = [0 for _ in range(self.dims)]
        
        self.searchRegion = searchRegion
        self.boundingBox = boundingBox

        self.boundingBoxShape = (boundingBox.height, boundingBox.width)
        self.boundingBoxCenter = (boundingBox.height // 2, boundingBox.width // 2)

        self.searchRegionShape = (searchRegion.height, searchRegion.width)
        self.searchRegionCenter = (searchRegion.height // 2, searchRegion.width // 2)

        self.gaussianScore = fftGuassianKernel(searchRegion.height, searchRegion.width, self.searchRegionCenter[0], self.searchRegionCenter[1], self.sigma)
        #plt.imshow(abs(ifft2(self.gaussianScore)))
        #plt.show()

        # Vi fixar f????rsta framen (frame 0), via start
        self.updateFirstFrame(image)

    def updateFirstFrame(self, image):
        for dim in range(self.dims):
            self.P[dim] = self.getFFTPatch(image, dim)

            self.A[dim] = self.P[dim] * np.conj(self.gaussianScore)
            self.B += self.P[dim] * np.conj(self.P[dim])
       
        for dim in range(self.dims):
            self.M[dim] = self.A[dim] / (self.regularization + self.B)

    def detect(self, image):
        sum_M = 0
        for dim in range(self.dims):
            self.P[dim] = self.getFFTPatch(image, dim)
            # FFT reponse between patch and our learned filter
            fftresponse = np.conj(self.M[dim]) * self.P[dim]    
            sum_M += fftresponse


        response = (ifft2(sum_M))

        # Find maximum response
        r, c = np.unravel_index(np.argmax(response), response.shape)
        
        # Move kernel to new peak
        self.gaussianScore = fftGuassianKernel(self.searchRegion.height, self.searchRegion.width, r, c, self.sigma)
        #plt.imshow(abs(ifft2(self.gaussianScore)))
        #plt.show()
        r_offset = r - self.searchRegionCenter[0]
        c_offset = c - self.searchRegionCenter[1]

        # Update pos
        #if self.boundingBox.xpos + c_offset < image.shape[1] and self.boundingBox.xpos + c_offset > self.boundingBox.width and self.boundingBox.ypos + r_offset < image.shape[0] and self.boundingBox.ypos + c_offset > self.boundingBox.height:

        self.boundingBox.xpos += c_offset
        self.boundingBox.ypos += r_offset

        self.searchRegion.xpos += c_offset
        self.searchRegion.ypos += r_offset

        return self.boundingBox

    def update(self, image):
        self.detect(image) # Detect before update

        B = 0
        for dim in range(self.dims):
            # Vi beräknar A
            self.A[dim] = (self.forgettingFactor * self.P[dim] * np.conj(self.gaussianScore)) + (1 - self.forgettingFactor) * self.A[dim]
           
            B += self.forgettingFactor * (self.P[dim] * np.conj(self.P[dim]))

        # Vi ber????knar B
        self.B = B + (1 - self.forgettingFactor) * self.B

        # Vi ber????knar M
        for dim in range(self.dims):
            self.M[dim] = self.A[dim] / (self.regularization + self.B)

        return copy.deepcopy(self.boundingBox)

    # Returnerar normaliserad FFT av input image (patch)
    def getFFTPatch(self, image, dim=0):
        if self.dims > 1:
            patch = self.crop_patch(image[:, :, dim])
        else:
            patch = self.crop_patch(image)

        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patch = hanning(patch)

        return fft2(patch)

class MOSSE_SCALE:
    def __init__(self):
        self.boundingBox = None
        self.searchRegion = None
        self.boundingBoxShape = None
        self.boundingBoxCenter = None

        self.gaussianScore = None
        self.A = []
        self.B = 0
        self.M = []

        self.MScale = []
        self.AScale = []
        self.BScale = []
        self.gaussianScoreScale = []

        self.P = []
        #self.PScale = []
        self.dims = 1

        self.numberOfScales = 11
        self.scaleFactor = 1.02
        self.scaleCoef = 1.0
        self.scales = []
        self.scaleP = []
        self.scaleF = []
        self.meanScale = []

        self.forgettingFactor = 0.025
        self.sigma = 2.0
        self.regularization = 0.1 # lambda

    #
    def crop_patch(self, image):
        #searchRegion = self.searchRegion
        return crop_patch(image, self.searchRegion)

    def start(self, image, boundingBox, searchRegion):
        if len(image.shape) > 2:
            print("colorimage")
            self.dims = image.shape[2]
        
        self.M = [0 for _ in range(self.dims)]
        self.A = [0 for _ in range(self.dims)]
        self.P = [0 for _ in range(self.dims)]
        
        self.searchRegion = searchRegion
        self.boundingBox = boundingBox

        self.boundingBoxShape = (boundingBox.height, boundingBox.width)
        self.boundingBoxCenter = (boundingBox.height // 2, boundingBox.width // 2)

        self.searchRegionShape = (searchRegion.height, searchRegion.width)
        self.searchRegionCenter = (searchRegion.height // 2, searchRegion.width // 2)
        
        self.gaussianScore = fftGuassianKernel(searchRegion.height, searchRegion.width, self.searchRegionCenter[0], self.searchRegionCenter[1], self.sigma)

        # New part
        self.scaleP = [[] for _ in range(self.dims)]
        self.AScale = [[] for _ in range(self.dims)]
        self.BScale = [0 for _ in range(self.numberOfScales)]
        self.MScale = [[] for _ in range(self.dims)]
        self.scaleRegions = [0 for _ in range(self.numberOfScales)]
        self.gaussianScoreScale = [[] for _ in range(self.numberOfScales)]

        for dim in range(self.dims):
            self.scaleP[dim] = [[] for _ in range(self.numberOfScales - 1)]
            self.AScale[dim] = [[] for _ in range(self.numberOfScales - 1)]
            self.MScale[dim] = [[] for _ in range(self.numberOfScales - 1)]

        scaleBound = (self.numberOfScales - 1) // 2
        self.scales = list(range(-scaleBound, scaleBound))

        #print(self.scales)
        for scale in range(len(self.scales)):
            self.scales[scale] = pow(self.scaleFactor, self.scales[scale])
        
        for dim in range(self.dims):
            for scale in range(len(self.scales)):
                tempRegion = copy.deepcopy(self.boundingBox)

                # The actual length increse needed for the
                heightIncrese = floor(self.scales[scale] * tempRegion.height) - tempRegion.height
                widthIncrese = floor(self.scales[scale] * tempRegion.width) - tempRegion.width

                tempRegion.height = floor(self.scales[scale] * self.scaleCoef * tempRegion.height)
                tempRegion.width = floor(self.scales[scale] * self.scaleCoef * tempRegion.width)

                # Move half of the new length
                tempRegion.xpos = tempRegion.xpos - (widthIncrese // 2)
                tempRegion.ypos = tempRegion.ypos - (heightIncrese // 2)

                #tempRegionMiddleHeight = tempRegion.height // 2
                #tempRegionMiddleWidth = tempRegion.width // 2

                self.scaleRegions[scale] = tempRegion
                #self.gaussianScoreScale[scale] = fftGuassianKernel(tempRegion.height, tempRegion.width, tempRegionMiddleHeight, tempRegionMiddleWidth, self.sigma)
                
                if (self.dims > 1):
                    self.scaleP[dim][scale] = crop_patch(image[:, :, dim], tempRegion)
                else:
                    self.scaleP[dim][scale] = crop_patch(image, tempRegion)

                # Normalize
                self.scaleP[dim][scale] = self.scaleP[dim][scale] / 255
                self.scaleP[dim][scale] = self.scaleP[dim][scale] - np.mean(self.scaleP[dim][scale])
                self.scaleP[dim][scale] = self.scaleP[dim][scale] / np.std(self.scaleP[dim][scale])
                self.scaleP[dim][scale] = hanning(self.scaleP[dim][scale])

                # Pscale
                #self.scaleP[dim][scale] = fft2(self.scaleP[dim][scale])

        self.scaleP = self.resizePatches()
        self.fftPDims()

        # Vi fixar f????rsta framen (frame 0), via start
        self.updateFirstFrame(image)

    def updateFirstFrame(self, image):
        for dim in range(self.dims):
            self.P[dim] = self.getFFTPatch(image, dim)

            self.A[dim] = self.P[dim] * np.conj(self.gaussianScore)
            self.B += self.P[dim] * np.conj(self.P[dim])
       
        for dim in range(self.dims):
            self.M[dim] = self.A[dim] / (self.regularization + self.B)

        # Scale
        for dim in range(self.dims):
            for scale in range(self.numberOfScales - 1):            
                self.AScale[dim][scale] = self.scaleP[dim][scale] * np.conj(self.gaussianScore) # One gaussian? Or one for each patch?

                self.BScale[scale] += self.scaleP[dim][scale] * np.conj(self.scaleP[dim][scale])
                
        for dim in range(self.dims):
            for scale in range(self.numberOfScales - 1):
                self.MScale[dim][scale] = self.AScale[dim][scale] / (self.regularization + self.BScale[scale])

    def resizePatches(self):
        rescaled = [[] for _ in range(self.dims)]
        for dim in range(self.dims):
            for scale in self.scaleP[dim]:
                temp = cv2.resize(scale, (self.searchRegionShape[1], self.searchRegionShape[0]), interpolation = cv2.INTER_NEAREST )
                rescaled[dim].append(temp)
                #plt.imshow(temp)
                #plt.show()

        return rescaled

    def fftPDims(self):
        for dim in range(self.dims):
            for scale in range(len(self.scaleP[dim])):
                self.scaleP[dim][scale] = fft2(self.scaleP[dim][scale])


    def detect(self, image):
        sum_M = 0
        for dim in range(self.dims):
            self.P[dim] = self.getFFTPatch(image, dim)
            # FFT reponse between patch and our learned filter
            self.M[dim].resize(self.P[dim].shape)
            fftresponse = np.conj(self.M[dim]) * self.P[dim]   
            sum_M += fftresponse

        response = (ifft2(sum_M))

        # Find maximum response
        r, c = np.unravel_index(np.argmax(response), response.shape)
        
        # Move kernel to new peak
        self.gaussianScore = fftGuassianKernel(self.searchRegion.height, self.searchRegion.width, r, c, self.sigma)

        r_offset = r - self.searchRegionCenter[0]
        c_offset = c - self.searchRegionCenter[1]

        # Update pos
        if self.boundingBox.xpos + c_offset > image.shape[1] or self.boundingBox.xpos + c_offset < 0 or self.boundingBox.ypos + r_offset > image.shape[0] or self.boundingBox.ypos + c_offset < 0:
            self.boundingBox.xpos += c_offset
            self.boundingBox.ypos += r_offset

            self.searchRegion.xpos += c_offset
            self.searchRegion.ypos += r_offset

        # Scaling, we get the newly moved patched P for all scale factors
        for dim in range(self.dims):
            for scale in range(len(self.scales)):
                tempRegion = copy.deepcopy(self.boundingBox)

                # The actual length increse needed for the
                heightIncrese = floor(self.scales[scale] * self.scaleCoef * tempRegion.height) - tempRegion.height
                widthIncrese = floor(self.scales[scale] * self.scaleCoef * tempRegion.width) - tempRegion.width

                tempRegion.height = floor(self.scales[scale] * self.scaleCoef * tempRegion.height)
                tempRegion.width = floor(self.scales[scale] * self.scaleCoef * tempRegion.width)

                # Move half of the new length
                tempRegion.xpos = tempRegion.xpos - (widthIncrese // 2)
                tempRegion.ypos = tempRegion.ypos - (heightIncrese // 2)

                #tempRegionMiddleHeight = tempRegion.height // 2
                #tempRegionMiddleWidth = tempRegion.width // 2

                self.scaleRegions[scale] = tempRegion
                #self.gaussianScoreScale[scale] = fftGuassianKernel(tempRegion.height, tempRegion.width, tempRegionMiddleHeight, tempRegionMiddleWidth)
                #plt.imshow(abs(ifft2(self.gaussianScoreScale[scale])))
                #plt.show()
                
                if (self.dims > 1):
                    self.scaleP[dim][scale] = crop_patch(image[:, :, dim], tempRegion)
                else:
                    self.scaleP[dim][scale] = crop_patch(image, tempRegion)

                # Normalize
                self.scaleP[dim][scale] = self.scaleP[dim][scale] / 255
                self.scaleP[dim][scale] = self.scaleP[dim][scale] - np.mean(self.scaleP[dim][scale])
                self.scaleP[dim][scale] = self.scaleP[dim][scale] / np.std(self.scaleP[dim][scale])
                self.scaleP[dim][scale] = hanning(self.scaleP[dim][scale])


                # Pscale              
                #self.scaleP[dim][scale] = fft2(self.scaleP[dim][scale])

        self.scaleP = self.resizePatches()
        self.fftPDims()

        #Maximize the scale filter, calculate  numberOfScales different fourier responses and inverse them and look for the highest value in each spatial response
        sumMDims = [0 for _ in range(self.numberOfScales - 1)]
        for dim in range(self.dims):
            for scale in range(self.numberOfScales - 1):
                fftresponse = np.conj(self.MScale[dim][scale]) * self.scaleP[dim][scale] 
                sumMDims[scale] += fftresponse
        
        response = [0 for _ in range(self.numberOfScales - 1)]
        for scale in range(self.numberOfScales - 1):
            responseIFFT2 = ifft2(sumMDims[scale])
            response[scale] = responseIFFT2

        bestValRCandScale = []
        for scale in range(self.numberOfScales - 1):
            rs, cs = np.unravel_index(np.argmax(response[scale].real), response[scale].shape) #  Finds higest peak in each spatial response
            bestValRCandScale.append((response[scale][rs][cs], rs, cs, scale))

        # For plotting purposes, shows the highest peak and the corresponding scale factor
        peakPerScale = []
        for scale in range(self.numberOfScales - 1):
            peakPerScale.append(bestValRCandScale[scale][0].real)
        
        bestMatch = (0., 0, 0, 0)
        for scale in range(self.numberOfScales - 1):
            #print(bestValRCandScale[scale])
            if bestValRCandScale[scale][0].real > bestMatch[0].real:
                #print("The one above me is new best!")
                bestMatch = bestValRCandScale[scale]

        self.meanScale.append(self.scales[bestMatch[3]])

        #if abs(self.scales[bestMatch[3]] - self.scaleCoef) > 0.2:           
            #plt.plot(self.scales, peakPerScale)       
            #plt.show()
        self.scaleCoef *= self.scales[bestMatch[3]]

        #print("Current scale mean:", np.average(self.meanScale))
        bestRegion = self.scaleRegions[bestMatch[3]]
        
        return bestRegion

    def update(self, image):
        returnRegion = self.detect(image) # Detect before update

        B = 0
        for dim in range(self.dims):
            # Vi ber????knar A
            self.A[dim] = (self.forgettingFactor * self.P[dim] * np.conj(self.gaussianScore)) + (1 - self.forgettingFactor) * self.A[dim]
           
            B += self.forgettingFactor * (self.P[dim] * np.conj(self.P[dim]))

        # Vi ber????knar B
        self.B = B + (1 - self.forgettingFactor) * self.B

        # Vi ber????knar M
        for dim in range(self.dims):
            self.M[dim] = self.A[dim] / (self.regularization + self.B)

        # Scale
        BScale = [0 for _ in range(self.numberOfScales)]
        for dim in range(self.dims):
            for scale in range(self.numberOfScales - 1):                    
                self.AScale[dim][scale] = (self.forgettingFactor * self.scaleP[dim][scale] * np.conj(self.gaussianScore)) + (1 - self.forgettingFactor) * self.AScale[dim][scale]

                BScale[scale] += self.forgettingFactor * (self.scaleP[dim][scale] * np.conj(self.scaleP[dim][scale]))
        
        for scale in range(self.numberOfScales - 1):
            self.BScale[scale] = BScale[scale] + (1 - self.forgettingFactor) * self.BScale[scale]
        
        for dim in range(self.dims):
            for scale in range(self.numberOfScales - 1):
                self.MScale[dim][scale] = self.AScale[dim][scale] / (self.regularization + self.BScale[scale])

        return copy.deepcopy(returnRegion)

    # Returnerar normaliserad FFT av input image (patch)
    def getFFTPatch(self, image, dim=0):
        if self.dims > 1:
            patch = self.crop_patch(image[:, :, dim])
        else:
            patch = self.crop_patch(image)
        
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patch = hanning(patch)

        return fft2(patch)

class MOSSE_DEEP:
    def __init__(self):
        self.model = alexnetFeatures(pretrained=True)
        self.first_layer = self.model.features[0:2]

        self.boundingBox = None
        self.searchRegion = None
        self.boundingBoxShape = None
        self.boundingBoxCenter = None

        # Changed every conv test
        self.convOutShape = 55
        self.searchConvRatioY = 0
        self.searchConvRatioX = 0

        self.gaussianScore = None
        self.A = []
        self.B = 0
        self.M = []
        self.P = None
        self.dims = 64

        self.forgettingFactor = 0.125
        self.sigma = 2.0
        self.regularization = 0.1 # lambda

        # https://pytorch.org/hub/pytorch_vision_alexnet/
        self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def crop_patch(self, image):
        return crop_patch(image, self.searchRegion)

    def start(self, image, boundingBox, searchRegion):
        self.M = [0 for _ in range(self.dims)]
        self.A = [0 for _ in range(self.dims)]
        self.P = [0 for _ in range(self.dims)]
        
        self.searchRegion = searchRegion
        self.boundingBox = boundingBox

        self.boundingBoxShape = (boundingBox.height, boundingBox.width)
        self.boundingBoxCenter = (boundingBox.height // 2, boundingBox.width // 2)

        self.searchRegionShape = (searchRegion.height, searchRegion.width)
        self.searchRegionCenter = (searchRegion.height // 2, searchRegion.width // 2)

        #### CONV FIX
        self.gaussianScore = fftGuassianKernel(self.convOutShape, self.convOutShape, self.convOutShape//2, self.convOutShape//2, self.sigma)
        self.searchConvRatioY = self.searchRegion.height / self.convOutShape
        self.searchConvRatioX = self.searchRegion.width / self.convOutShape
        
        # Vi fixar första framen (frame 0), via start
        self.updateFirstFrame(image)

    def updateFirstFrame(self, image):
        patches = self.getPatchesDeep(image)
        for dim in range(self.dims):
            patch = patches[dim]
            #patch = self.normalize(patch)
            patch = hanning(patch)
            self.P[dim] = fft2(patch)
            self.A[dim] = self.P[dim] * np.conj(self.gaussianScore)
            self.B += self.P[dim] * np.conj(self.P[dim])
       
        for dim in range(self.dims):
            self.M[dim] = self.A[dim] / (self.regularization + self.B)

    def detect(self, image):
        sum_M = 0
        patches = self.getPatchesDeep(image)
        for dim in range(self.dims):
            patch = patches[dim]
            #patch = self.normalize(patch)
            patch = hanning(patch)
            self.P[dim] = fft2(patch)
            # FFT reponse between patch and our learned filter
            fftresponse = np.conj(self.M[dim]) * self.P[dim]
            sum_M += fftresponse


        response = (ifft2(sum_M))
        # Find maximum response
        r, c = np.unravel_index(np.argmax(response), response.shape)
        
        # Move kernel to new peak
        self.gaussianScore = fftGuassianKernel(self.convOutShape, self.convOutShape, r, c, self.sigma)

        #### CONV FIX
        r_offset = r - self.convOutShape // 2
        c_offset = c - self.convOutShape // 2

        print("searchRegionShape", self.searchRegionShape)
        print("r_offset", r_offset)
        print("c_offset", c_offset)
        print("gaussianScoreShape", self.gaussianScore.shape)
        print("searchConvRatioX", self.searchConvRatioX)
        print("searchConvRatioY", self.searchConvRatioY)

        # Update pos
        self.boundingBox.xpos += int(c_offset*self.searchConvRatioX)
        self.boundingBox.ypos += int(r_offset*self.searchConvRatioY)

        self.searchRegion.xpos += int(c_offset*self.searchConvRatioX)
        self.searchRegion.ypos += int(r_offset*self.searchConvRatioY)

        return self.boundingBox

    def update(self, image):
        self.detect(image) # Detect before update

        B = 0
        for dim in range(self.dims):
            # Vi beräknar A
            self.A[dim] = (self.forgettingFactor * self.P[dim] * np.conj(self.gaussianScore)) + (1 - self.forgettingFactor) * self.A[dim]
           
            B += self.forgettingFactor * (self.P[dim] * np.conj(self.P[dim]))

        # Vi beräknar B
        self.B = B + (1 - self.forgettingFactor) * self.B

        # Vi beräknar M
        for dim in range(self.dims):
            self.M[dim] = self.A[dim] / (self.regularization + self.B)

        return copy.deepcopy(self.boundingBox)

    def normalize(self, patch):
        # patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        return patch

    def getPatchesDeep(self, image):
        
        r = self.crop_patch(image[:, :, 0])
        g = self.crop_patch(image[:, :, 1])
        b = self.crop_patch(image[:, :, 2])
        
        rgb_patch = np.dstack((r,g,b))
        PIL_patch = Image.fromarray(rgb_patch)
        input_tensor = self.preprocess(PIL_patch)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        res = self.first_layer(input_batch)
        #res = torch.nn.functional.interpolate(res, self.searchRegionShape) # resize images to searchRegion
        res = res.detach().numpy()
        res = res.squeeze() #remove first uneccesary dim
        #res = np.moveaxis(res, 0, -1) # (dims, w, h) => (w, h, dims)

        return res