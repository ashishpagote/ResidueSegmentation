import numpy as np
import cv2
from skimage.util import random_noise

class CustomRandomHorizontalFlip(object):
    def __call__(self, sample):
        image,mask = sample
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return image, mask

class CustomRandomVerticalFlip(object):
    def __call__(self, sample):
        image, mask = sample
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        return image, mask

class CustomRandomRotation90(object):
    def __call__(self, sample):
        image, mask = sample
        if np.random.rand() < 0.5:
            image = np.rot90(image)
            mask = np.rot90(mask)
        return image, mask

class CustomContrastEnhancement(object):
    def __call__(self, sample):
        image, mask = sample
        if np.random.rand() < 0.5:
            contrast=np.random.uniform(0.5,1.5)
            image = cv2.addWeighted(image,contrast,np.zeros(image.shape,image.dtype),0,0)
        return image, mask

class CustomBrightnessEnhancement(object):
    def __call__(self, sample):
        image, mask = sample
        if np.random.rand() < 0.5:
            bright=np.random.randint(low=-50,high=50)
            image = cv2.addWeighted(image,1,np.zeros(image.shape,image.dtype),0,bright)
        return image, mask

class CustomSaltPepperNoise(object):
    def __call__(self, sample):
        image, mask = sample
        if np.random.rand() < 0.5:
            noise_amount=np.random.uniform(0.25,0.5)
            image = random_noise(image, mode='s&p', amount=noise_amount)
            image = np.array(255 * image, dtype=np.uint8)

        return image, mask



class CustomRandomCrop(object):
    def __init__(self, size=(256,256)):
        self.crop_size = size
    
    def __call__(self, sample):
        # Randomly crop the image and mask
        """
        Randomly crop a given image and mask
        # Arguments
            image: The image to be cropped
            mask: The mask to be cropped
            crop_size: The size of the cropping
        # Returns
            A tuple of the cropped image and mask
        """
        image, mask = sample
        # If the image is too small, return it as is
        if image.shape[0] < self.crop_size[0] or image.shape[1] < self.crop_size[1]:
            return image, mask

        heterogenous = False
        while not heterogenous:
            # Randomly crop the image and mask
            x_offset = np.random.randint(0, image.shape[1] - self.crop_size[1])
            y_offset = np.random.randint(0, image.shape[0] - self.crop_size[0])
            img = image[y_offset:y_offset + self.crop_size[0], x_offset:x_offset + self.crop_size[1], :]
            mk = mask[y_offset:y_offset + self.crop_size[0], x_offset:x_offset + self.crop_size[1]]

            mask_size = 1
            for i in mk.shape:
                mask_size *=i    
            val = (np.count_nonzero(mk) / mask_size)
            heterogenous =  0.1 < val < 0.9

        return img, mk