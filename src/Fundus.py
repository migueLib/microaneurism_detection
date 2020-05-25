# Built-In libraries
import os
import copy
import itertools

# External libraries
import PIL
import torch
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import scipy.cluster.hierarchy as shc
import sklearn
from sklearn.cluster import AgglomerativeClustering

# Local libraries
from src.plots import plot_color_bar


class Fundus():
    def __init__(self, source=False, **kwargs):
        
        # Constructors
        if isinstance(source, str):
            self.im = self._image_from_file(source)

        if isinstance(source, np.ndarray):
            self.im = self._image_from_pixels(source, **kwargs)
            
        if isinstance(source, PIL.Image.Image):
            self.im = source
            
        if isinstance(source, torch.Tensor):
            self.im = source.to("cpu").numpy().astype(np.uint8)

        # Attributes
        self._pixels = self._get_pixels()
        
        self._palette = self._get_palette()
                
        self.w, self.h = self.im.size

    # Constructors
    @staticmethod
    def _image_from_file(path):
        return Image.open(path, mode="r")

    @staticmethod
    def _image_from_pixels(pixels, **kwargs):
        arr = np.resize(pixels, (kwargs["w"], kwargs["h"], 3)).astype(np.uint8)
        im = Image.fromarray(arr)
        im = im.rotate(90, expand=True)
        im = ImageOps.flip(im)
        return im

    # Access attributes
    @property
    def palette(self):
        return self._palette
    
    @property
    def pixels(self):
        return self._pixels

    # Get attributes
    def _get_pixels(self):
        r, g, b = np.asarray(self.im).T
        r, g, b = r.flatten(), g.flatten(), b.flatten()
        return np.asarray([r, g, b]).T
    
    def _get_palette(self):
        r, g, b = np.asarray(self.im).T
        r, g, b = r.flatten(), g.flatten(), b.flatten()
        pre_palette = zip(r, g, b)
        return np.asarray(list(set(pre_palette)))

    # VISUALIZATION    
    def plot_palette(self):
        plot_color_bar(sorted(self._palette))

    # MODIFICATION FILTERING
    def mask(self, colors, replacement=None, inplace=False, inverse=False):
        """
        Replaces a list of pixels for a given value
        :param colors: 2-D array of the RGB pixel values for the image.
        :param replacement: 1-D [0-255] RGB array of the color to replace with
        :return: modified 2-D array
        """
        # Empty black canvas if inverse else image
        canvas = np.zeros(self._pixels.shape, dtype=np.uint8) if inverse else self._pixels
        
        # Mask pixels
        for c in colors:
            canvas[(self._pixels == c).all(axis=1)] = replacement if replacement is not None else [0, 255, 0]

        # Output in place
        self.im = self._image_from_pixels(canvas, w=self.w, h=self.h) if inplace else self.im
        
        
        return canvas
    
    

 
