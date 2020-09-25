import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from .image_io import crop_patch
import cv2


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
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)


        self.gaussianScore = np.zeros((region.height, region.width))
        self.gaussianScore[region.height // 2, region.width // 2] = 1
        self.gaussianScore = cv2.GaussianBlur(self.gaussianScore, (0, 0), 2.0) # 2.0 std in x and y led
        self.gaussianScore = fft2(self.gaussianScore)
        
        patch = self.crop_patch(image)

        patch = patch/255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)

        self.template = fft2(patch)

    #copy paste from https://github.com/opencv/opencv/blob/master/samples/python/mosse.py
    def divSpec(self, A, B):
        Ar, Ai = A[...,0], A[...,1]
        Br, Bi = B[...,0], B[...,1]
        C = (Ar+1j*Ai)/(Br+1j*Bi)
        C = np.dstack([np.real(C), np.imag(C)]).copy()
        return C

    def updateFrame1(self, image):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)

        self.A = cv2.mulSpectrums(patchf, self.gaussianScore, 0, conjB=True)
        self.B = cv2.mulSpectrums(patchf, patchf, 0, conjB=True)
        self.M = self.divSpec(self.A, self.B)

        r, c = np.unravel_index(np.argmin(self.M), self.M.shape)

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]
        print("r offset: ({} + {} mod {}) - {} = {}".format(r, self.region_center[0], self.region.height, self.region_center[0], r_offset))
        print("c offset: ({} + {} mod {}) - {} = {}".format(c, self.region_center[1], self.region.width, self.region_center[1], c_offset))

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.region

    def update(self, image, ff=0.9):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)

        A = cv2.mulSpectrums(patchf, self.gaussianScore, 0, conjB=True)
        B = cv2.mulSpectrums(patchf, patchf, 0, conjB=True)
        self.A = ff*A + (1-ff)*self.A
        self.B = ff*B + (1-ff)*self.B
        self.M = self.divSpec(self.A, self.B)

        r, c = np.unravel_index(np.argmin(self.M), self.M.shape)

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]
        print("r offset: ({} + {} mod {}) - {} = {}".format(r, self.region_center[0], self.region.height, self.region_center[0], r_offset))
        print("c offset: ({} + {} mod {}) - {} = {}".format(c, self.region_center[1], self.region.width, self.region_center[1], c_offset))

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.region
