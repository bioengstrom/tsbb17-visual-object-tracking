import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import shift
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from .image_io import crop_patch
import cv2
import matplotlib.pyplot as plt
import copy

# kernel becomes kernelLength by kernelLength
def fftGuassianKernel(kernelHeight, kernelWidth, peakRow, peakColumn, sigma=2.0): 
    x = np.zeros([kernelHeight, kernelWidth])
    x[peakRow][peakColumn] = 1
    return fft2(gaussian_filter(x, sigma))

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

        self.forgettingFactor = 0.2175
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
        
        self.searchRegion = searchRegion
        self.boundingBox = boundingBox

        self.boundingBoxShape = (boundingBox.height, boundingBox.width)
        self.boundingBoxCenter = (boundingBox.height // 2, boundingBox.width // 2)

        self.searchRegionShape = (searchRegion.height, searchRegion.width)
        self.searchRegionCenter = (searchRegion.height // 2, searchRegion.width // 2)

        self.gaussianScore = fftGuassianKernel(searchRegion.height, searchRegion.width, self.searchRegionCenter[0], self.searchRegionCenter[1], self.sigma)
        #plt.imshow(abs(ifft2(self.gaussianScore)))
        #plt.show()

        # Vi fixar första framen (frame 0), via start
        self.updateFirstFrame(image)

    def updateFirstFrame(self, image):
        for dim in range(self.dims):
            self.P = self.getFFTPatch(image, dim)

            self.A[dim] = self.P * np.conj(self.gaussianScore)
            self.B += self.P * np.conj(self.P)
       
        for dim in range(self.dims):
            self.M[dim] = self.A[dim] / (self.regularization + self.B)

    def detect(self, image):
        sum_M = 0
        for dim in range(self.dims):
            self.P = self.getFFTPatch(image, dim)
            # FFT reponse between patch and our learned filter
            fftresponse = np.conj(self.M[dim]) * self.P    
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
            self.A[dim] = (self.forgettingFactor * self.P * np.conj(self.gaussianScore)) + (1 - self.forgettingFactor) * self.A[dim]
           
            B += self.forgettingFactor * (self.P * np.conj(self.P))

        # Vi beräknar B
        self.B = B + (1 - self.forgettingFactor) * self.B

        # Vi beräknar M
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
        self.P = None
        self.dims = 1

        self.forgettingFactor = 0.2175
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
        
        self.searchRegion = searchRegion
        self.boundingBox = boundingBox

        self.boundingBoxShape = (boundingBox.height, boundingBox.width)
        self.boundingBoxCenter = (boundingBox.height // 2, boundingBox.width // 2)

        self.searchRegionShape = (searchRegion.height, searchRegion.width)
        self.searchRegionCenter = (searchRegion.height // 2, searchRegion.width // 2)

        self.gaussianScore = fftGuassianKernel(searchRegion.height, searchRegion.width, self.searchRegionCenter[0], self.searchRegionCenter[1], self.sigma)
        #plt.imshow(abs(ifft2(self.gaussianScore)))
        #plt.show()

        # Vi fixar första framen (frame 0), via start
        self.updateFirstFrame(image)

    def updateFirstFrame(self, image):
        for dim in range(self.dims):
            self.P = self.getFFTPatch(image, dim)

            self.A[dim] = self.P * np.conj(self.gaussianScore)
            self.B += self.P * np.conj(self.P)
       
        for dim in range(self.dims):
            self.M[dim] = self.A[dim] / (self.regularization + self.B)

    def detect(self, image):
        sum_M = 0
        for dim in range(self.dims):
            self.P = self.getFFTPatch(image, dim)
            # FFT reponse between patch and our learned filter
            fftresponse = np.conj(self.M[dim]) * self.P    
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
            self.A[dim] = (self.forgettingFactor * self.P * np.conj(self.gaussianScore)) + (1 - self.forgettingFactor) * self.A[dim]
           
            B += self.forgettingFactor * (self.P * np.conj(self.P))

        # Vi beräknar B
        self.B = B + (1 - self.forgettingFactor) * self.B

        # Vi beräknar M
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

        return fft2(patch)