import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import warnings
warnings.filterwarnings('ignore')

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(idx1, idx2):
    plt.figure(figsize=(10,5))
    plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()))
    
def extract_cloth(image, bbox):
    max_area = 0
    image_np = load_image_into_numpy_array(image)
    h,w,_ = image_np.shape
    if(bbox.shape[0]==0):
        return np.array([])
    for i in range(bbox.shape[0]):
        ymin, xmin, ymax, xmax = bbox[i]
        ymin = int(ymin*h)
        ymax = int(ymax*h)
        xmin = int(xmin*w)
        xmax = int(xmax*w)
        area = (ymax-ymin)*(xmax-xmin)
        if area>max_area:
            image_ = image_np[ymin:ymax,xmin:xmax]
    return image_

def extract_clothes(image_np, bbox):
    faces = []
    h,w,_ = image_np.shape
    for i in range(bbox.shape[0]):
        ymin, xmin, ymax, xmax = bbox[i]
        ymin = int(ymin*h)
        ymax = int(ymax*h)
        xmin = int(xmin*w)
        xmax = int(xmax*w)
        image_ = image_np[ymin:ymax,xmin:xmax]
        faces.append(image_)
    return faces

def load_image(path):
    img = cv2.imread(path, 1)
    return img[...,::-1]


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

class IdentityMetadata():
    def __init__(self, name, file):
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.name, self.file) 
    
def load_metadata(path):
    metadata = []
    for f in os.listdir(path):
        # Check file extension. Allow only jpg/jpeg' files.
        ext = os.path.splitext(f)[1]
        if ext == '.jpg' or ext == '.jpeg':
            metadata.append(IdentityMetadata(path,f))
    return np.array(metadata)














