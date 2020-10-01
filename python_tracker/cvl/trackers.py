import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import shift
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from .image_io import crop_patch
import cv2
import matplotlib.pyplot as plt

# kernel becomes kernelLength by kernelLength
def guassianKernel(kernelHeight, kernelWidth, sigma=2.0):  
    gaussianKernel1DH = signal.gaussian(kernelHeight, std=sigma).reshape(kernelHeight, 1)
    gaussianKernel1DW = signal.gaussian(kernelWidth, std=sigma).reshape(kernelWidth, 1)
    gaussianKernel2D = np.outer(gaussianKernel1DH, gaussianKernel1DW)
    return gaussianKernel2D

class NCCTracker:

    def __init__(self, learning_rate=0.1):
        self.template = None 
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate

    def crop_patch(self, image):
        region = self.region
        return crop_patch(image, region)

    def start(self, image, region):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)

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

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.region

    def update(self, image, lr=0.1):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)
        self.template = self.template * (1 - lr) + patchf * lr

class MOSSE:
    def __init__(self, learning_rate=0.1):
        self.template = None # P in lecture 8, NCC
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate


        self.gaussianScore = None
        self.A = 0
        self.B = 0
        self.M = 0

    #
    def crop_patch(self, image):
        region = self.region
        return crop_patch(image, region)

    # Inits region
    # Grön ruta i NCC
    #
    def start(self, image, region):
        assert len(image.shape) == 2, "MOSSE is only defined for grayscale images"
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)


        self.gaussianScore = guassianKernel(region.height, region.width)
        #self.gaussianScore[region.height // 2, region.width // 2] = 1
        #self.gaussianScore = cv2.GaussianBlur(self.gaussianScore, (0, 0), 2.0) # 2.0 std in x and y led
        self.gaussianScore = fft2(self.gaussianScore)


    #copy paste from https://github.com/opencv/opencv/blob/master/samples/python/mosse.py
    def divSpec(self, A, B):
        Ar, Ai = A[...,0], A[...,1]
        Br, Bi = B[...,0], B[...,1]
        C = (Ar+1j*Ai)/(Br+1j*Bi)
        C = np.dstack([np.real(C), np.imag(C)]).copy()
        return C

    def updateFrame1(self, image):
        assert len(image.shape) == 2, "MOSSE is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = cv2.dft(patch, flags=cv2.DFT_COMPLEX_OUTPUT)

        self.A = cv2.mulSpectrums(patchf, self.gaussianScore, 0, conjB=True)
        self.B = cv2.mulSpectrums(patchf, patchf, 0, conjB=True)

        self.M = self.divSpec(self.A, self.B)

        m = cv2.idft(self.M, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        r, c = np.unravel_index(np.argmax(m), m.shape)

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]
        print("r offset: ({} + {} mod {}) - {} = {}".format(r, self.region_center[0], self.region.height, self.region_center[0], r_offset))
        print("c offset: ({} + {} mod {}) - {} = {}".format(c, self.region_center[1], self.region.width, self.region_center[1], c_offset))

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.region

    def update(self, image, ff=0.425):
        assert len(image.shape) == 2, "MOSSE is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = cv2.dft(patch, flags=cv2.DFT_COMPLEX_OUTPUT)

        A = cv2.mulSpectrums(patchf, self.gaussianScore, 0, conjB=True)
        B = cv2.mulSpectrums(patchf, patchf, 0, conjB=True)
        self.A = ff*A + (1-ff)*self.A
        self.B = ff*B + (1-ff)*self.B
        self.M = self.divSpec(self.A, self.B)

        m = cv2.idft(self.M, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        r, c = np.unravel_index(np.argmax(m), m.shape)

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]
        print("r offset: ({} + {} mod {}) - {} = {}".format(r, self.region_center[0], self.region.height, self.region_center[0], r_offset))
        print("c offset: ({} + {} mod {}) - {} = {}".format(c, self.region_center[1], self.region.width, self.region_center[1], c_offset))

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.region

class MOSSE_DCF:
    #Y : Gaussian filter (C)
    #X : Input (P)
    #
    def __init__(self):
        self.region = None
        self.region_shape = None
        self.region_center = None

        self.gaussianScore = None
        self.A = []
        self.B = 0
        self.M = []
        self.dims = 1

        self.forgettingFactor = 0.625
        self.sigma = 4.0
        self.regularization = 0.1

    #
    def crop_patch(self, image):
        region = self.region
        return crop_patch(image, region)

    def start(self, image, region):
        if len(image.shape) > 2:
            self.dims = image.shape[2]
        
        
        self.M = [0 for _ in range(self.dims)]
        self.A = [0 for _ in range(self.dims)]
        
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)

        self.gaussianScore = guassianKernel(region.height, region.width, self.sigma)
        self.gaussianScore = fft2(self.gaussianScore)

        # Vi fixar första framen (frame 0), via start
        self.updateFirstFrame(image)

    def updateFirstFrame(self, image):
        for dim in range(self.dims):
            patchf = self.getFFTPatch(image, dim)

            self.A[dim] = patchf * np.conj(self.gaussianScore)
            self.B += patchf * np.conj(patchf)
       
        return self.update_region(image)

    def update(self, image):
        B = 0
        for dim in range(self.dims):
            patchf = self.getFFTPatch(image, dim)

            # Vi beräknar A
            self.A[dim] = (self.forgettingFactor * patchf * np.conj(self.gaussianScore)) + (1 - self.forgettingFactor) * self.A[dim]

            B += self.forgettingFactor * patchf * np.conj(patchf)

        # Vi beräknar B
        self.B = B + (1 - self.forgettingFactor) * self.B

        # Uppdaterar våran region
        return self.update_region(image)

    def update_region(self, image):

        # Fourier sum, sum() did not properly deal with complex coordinates
        for dim in range(self.dims):
            self.M[dim] = self.A[dim] / (self.regularization + self.B)

        mArray = np.array(self.M)
        mSum = np.zeros(self.region_shape, dtype="complex_")

        for row in range(mArray.shape[1]):
            for column in range(mArray.shape[2]):
                for dim in range(mArray.shape[0]):
                    mSum[row][column] += mArray[dim][row][column]
        
        mSumSpatial = ifft2(mSum)

        plt.imshow(abs(mSumSpatial))
        plt.show()

        # Är lite osäker på denna biten tbh
        r, c = np.unravel_index(np.argmax(mSumSpatial), mSumSpatial.shape)

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.region

    def getFFTPatch(self, image, dim=0):
        if self.dims > 1:
            patch = self.crop_patch(image[:, :, dim])
        else:
            patch = self.crop_patch(image)
        
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)

        return fft2(patch)

