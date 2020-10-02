import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import shift
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from .image_io import crop_patch
import cv2
import matplotlib.pyplot as plt

# kernel becomes kernelLength by kernelLength
def fftGuassianKernel(kernelHeight, kernelWidth, peakRow, peakColumn, sigma=2.0): 
    x = np.zeros([kernelHeight, kernelWidth])
    x[peakRow][peakColumn] = 1
    return fft2(gaussian_filter(x, sigma=3.0))

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
    def __init__(self):
        self.region = None
        self.region_shape = None
        self.region_center = None

        self.gaussianScore = None
        self.A = []
        self.B = 0
        self.M = []
        self.P = None
        self.dims = 1

        self.forgettingFactor = 0.125
        self.sigma = 2.0
        self.regularization = 0.1

    #
    def crop_patch(self, image):
        region = self.region
        return crop_patch(image, region)

    def start(self, image, region):
        if len(image.shape) > 2:
            self.dims = image.shape[2]

        #plt.imshow(image, cmap="gray")
        #plt.show()
        
        self.M = [0 for _ in range(self.dims)]
        self.A = [0 for _ in range(self.dims)]
        
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)

        self.gaussianScore = fftGuassianKernel(region.height, region.width, self.region_center[0], self.region_center[1], self.sigma)

        # Vi fixar första framen (frame 0), via start
        self.updateFirstFrame(image)

    def updateFirstFrame(self, image):
        for dim in range(self.dims):
            self.P = self.getFFTPatch(image, dim)

            self.A[dim] = self.P * np.conj(self.gaussianScore)
            self.B += self.P * np.conj(self.P)
       
        for dim in range(self.dims):
            self.M[dim] = self.A[dim] / (self.regularization + self.B)

        return self.region

    def detect(self, image):
        for dim in range(self.dims):
            self.P = self.getFFTPatch(image, dim)

        # FFT reponse between patch and our learned filter
        fftresponse = np.conj(self.M[0]) * self.P
        response = (ifft2(fftresponse))

        #plt.imshow(response.real)
        #plt.show()

        # Find maximum response
        r, c = np.unravel_index(np.argmax(response), response.shape)
        
        # Move kernel to new peak
        self.gaussianScore = fftGuassianKernel(self.region.height, self.region.width, r, c, self.sigma)
        r_offset = r - self.region_center[0]
        c_offset = c - self.region_center[1]

        # Update pos
        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.region

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

        return self.region

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

    

