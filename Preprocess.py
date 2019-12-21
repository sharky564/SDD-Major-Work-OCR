import cv2
import random
import numpy as np

def preprocess(image, image_size, randomiser=False):
    '''Change the size of images to required dimensions whilst maintaining the colours, and transposing for TensorFlow'''
    (width, height) = image_size
    # in case of damaged files
    if image is None:
        image = np.zeros([image_size[1], image_size[0]])

    # create more data
    if randomiser:
        factor = random.random() + 0.5
        new_width = max(int(image.shape[1] * factor), 1)
        image = cv2.resize(image, (new_width, image.shape[0]))

    # creating required image
    (h, w) = image.shape
    
    scale_width = w / width
    scale_height = h / height
    scale = max(scale_width, scale_height)
    
    new_width = max(1, min(width, int(w / scale)))
    new_height = max(1, min(height, int(h / scale)))
    image = cv2.resize(image, (new_width, new_height))

    canvas = np.ones([height, width]) * 255
    canvas[0: new_height, 0: new_width] = image

    # TensorFlow modification
    image = cv2.transpose(canvas)

    # fix colours
    (mean, stddev) = cv2.meanStdDev(image)
    mean = mean[0][0]
    stddev = stddev[0][0]
    image = image - mean
    if stddev > 0:
        image = image / stddev
    else:
        image = image
        
    return image
