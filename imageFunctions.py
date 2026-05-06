import cv2
import numpy as np
from pathlib import Path
from fileFunctions import *
from scipy import ndimage as ndi

def bwImg(imagePath : str):
    '''Converts an image to black and white (binary) format.
    
    Parameters
    ----------
    imagePath : str
        The file path to the image to be converted.
    
    Returns
    -------
    numpy.ndarray
        A binary (black and white) image.
    '''
    
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bwImg = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return bwImg

def thresholdImg(img):
    '''Applies adaptive thresholding to an image to enhance feature extraction.
    
    Parameters
    ----------
    img : numpy.ndarray
        The input image to be thresholded.
    
    Returns
    -------
    numpy.ndarray
        A thresholded binary image with inverted colors.
    '''
    
    thresholdImg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresholdImg

def crop(img):
    '''Crops an image to its content bounds, removing excess whitespace.
    
    Parameters
    ----------
    img : numpy.ndarray
        The input image to be cropped.
    
    Returns
    -------
    tuple
        A tuple containing:
        - croppedImg (numpy.ndarray): The cropped image.
        - w (int): The width of the cropped region.
        - h (int): The height of the cropped region.  
    '''
    
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    croppedImg = img[y:y+h, x:x+w]
    return croppedImg, w, h

def img128(img : str, w : int, h : int):
    '''Resizes and centers an image to fit within a 128x128 canvas.
    
    Parameters
    ----------
    img : str
        The image to be resized.
    w : int
        The width of the input image.
    h : int
        The height of the input image.
    
    Returns
    -------
    numpy.ndarray
        A 128x128 binary image centered on a canvas.
    '''
    
    canvasSize = 128
    margin = 20
    scale = (canvasSize - margin) / max(w, h) #scaling fit
    nh, nw = int(h * scale), int(w * scale)
    resizedImg = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    _, resizedImg = cv2.threshold(resizedImg, 50, 255, cv2.THRESH_BINARY)
    canvas = np.zeros((canvasSize, canvasSize), dtype=np.uint8)
    ox, oy = (canvasSize - nw) // 2, (canvasSize - nh) // 2
    canvas[oy:oy+nh, ox:ox+nw] = resizedImg
    return canvas

def stage1(imagePath : str, rootPath : str):
    '''Applies the complete stage 1 image processing pipeline.
    This gets the input image to the same state as the first image as described in the emnist pipeline.
    
    This function performs the following operations in sequence:
    1. Converts image to black and white
    2. Applies adaptive thresholding
    3. Crops to content bounds
    4. Resizes to 128x128 canvas
    
    Parameters
    ----------
    imagePath : str
        The file path to the input image.
    rootPath : str
        The directory path where processed images will be saved.
    '''
    
    bw = bwImg(imagePath)
    saveImage(bw, f'{rootPath}/bwImg.png')
    thresh = thresholdImg(bw)
    saveImage(thresh, f'{rootPath}/thresholdImg.png')
    cropped, w, h = crop(thresh)
    saveImage(cropped, f'{rootPath}/croppedImg.png')
    img = img128(cropped, w, h)
    saveImage(img, f'{rootPath}/img128.png')
    
def gaussianBlur(img):
    '''Applies Gaussian blur to an image and converts it to binary.
    
    Parameters
    ----------
    img : numpy.ndarray
        The input image to be blurred.
    
    Returns
    -------
    numpy.ndarray
        A binary image after Gaussian filtering.  
    '''
    
    blurredImg = ndi.gaussian_filter(img, sigma=1)
    blurredImg = np.where(blurredImg > 0.1, 255, 0).astype(np.uint8)
    return blurredImg

def img62(img, w : int, h : int):
    '''Resizes and centers an image to fit within a 62x62 canvas.
    
    Parameters
    ----------
    img : numpy.ndarray
        The image to be resized.
    w : int
        The width of the input image.
    h : int
        The height of the input image.
    
    Returns
    -------
    numpy.ndarray
        A 62x62 binary image centered on a canvas.
    '''
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canvasSize = 62
    margin = 4
    scale = (canvasSize - margin) / max(w, h)
    nh, nw = int(h * scale), int(w * scale)
    resizedImg = cv2.resize(img, (nw, nh))
    canvas = np.zeros((canvasSize, canvasSize), dtype=np.uint8)
    ox, oy = (canvasSize - nw) // 2, (canvasSize - nh) // 2
    canvas[oy:oy+nh, ox:ox+nw] = resizedImg
    return canvas

def stage2(rootPath : str):
    '''Applies the complete stage 2 image processing pipeline.
    This is vaguely the process referenced in the emnist paper.
    
    This function performs the following operations in sequence:
    1. Applies Gaussian blur to the 128x128 image
    2. Resizes to 62x62 canvas
    3. Downsamples to 28x28
    4. Standardises the image for neural network input
    
    Parameters
    ----------
    rootPath : str
        The directory path containing the processed images from stage1.
    
    Returns
    -------
    numpy.ndarray
        A standardised 784-element (28x28 flattened) image array.  
    '''
    
    img = cv2.imread(rootPath + '/img128.png')
    blurred = gaussianBlur(img)
    saveImage(blurred, f'{rootPath}/blurredImg.png')
    path = cv2.imread(f'{rootPath}/blurredImg.png')
    img62x62 = img62(path, 1, 1)
    saveImage(img62x62, f'{rootPath}/img62x62.png')
    downsampled = downsample(img62x62)
    saveImage(downsampled, f'{rootPath}/downsampledImg.png')
    standardised = standardise(downsampled)
    saveImage(standardised, f'{rootPath}/standardisedImg.png')
    return standardised.astype(np.uint8)

def downsample(img):
    return cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

def standardise(img):
    ''' This function does the opposite of generating an image from matlab therefore formating the data to be the same as the matlab data.
    
    Parameters
    ----------
    img : numpy.ndarray
        28x28 image array
        
    Returns
    -------
    img : numpy.ndarray
        1x784 image array
    '''
    
    img = img.T
    img = img.reshape(1,784)
    return img

def prepareImage(imagePath, rootPath):
    stage1(imagePath, rootPath)
    return stage2(rootPath)
