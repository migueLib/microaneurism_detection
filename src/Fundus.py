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
            self.im = Image.fromarray(source.to("cpu").numpy().astype(np.uint8))

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
    
    
    def mask_torch(self, colors, replacement=None, inplace=False, inverse=False, in_cpu=True):
        """
        Replaces pixels 
        """

        # Emtpy black canvas if inverse else use the image
        pixels_tensor = torch.cuda.ByteTensor(self.pixels)

        if inverse:
            canvas = torch.cuda.ByteTensor(pixels_tensor.shape).fill_(0)
        else:
            canvas = torch.cuda.ByteTensor(self.pixels)

        # Create tensors in gpu directly
        replacement_tensor = torch.cuda.ByteTensor([[0,255,0]]) if replacement is None else torch.cuda.ByteTensor(replacement)
        colors_tensor = torch.cuda.ByteTensor(colors)

        # Compare pixels to list of colors
        for c in colors_tensor:

            # Create flag tensor for pixels to change
            ispixel = (pixels_tensor==c).all(dim=1).view(1, len(pixels_tensor))
            ispixel = torch.transpose(torch.cat((ispixel, ispixel, ispixel), dim=0), 0, 1)

            # Replace pixels
            canvas = torch.where(ispixel, replacement_tensor, canvas)

        # Output in place
        #canvas = canvas.to("cpu").numpy().astype(np.uint8)
        #self.im = self._image_from_pixels(canvas, w=self.w, h=self.h) if inplace else self.im
        canvas = canvas.to("cpu").numpy().astype(np.uint8) if in_cpu else canvas
        
        return canvas
    
    
    def cluster_pixels(self, n):
        pal = self.palette

        cluster = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')
        clustered = cluster.fit_predict(pal)

        a = np.zeros(self.pixels.shape, dtype=np.uint8)
        for c in tqdm(np.unique(clustered)):
            a+=self.mask(pal[clustered == c], inverse=True, replacement=pal[clustered == c][0])

        return a
    
    def cluster_pixels_torch(self, n):
        pal = self.palette

        cluster = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')
        clustered = cluster.fit_predict(pal)

        a = torch.cuda.ByteTensor(self.pixels.shape[0],self.pixels.shape[1]).fill_(0)
        for c in tqdm(np.unique(clustered)):
            #plot_color_bar(pal[clustered == c][0])
            a += self.mask_torch(pal[clustered == c], replacement=pal[clustered == c][0], inverse=True, in_cpu=False)

        return Fundus(a.to("cpu").numpy(), w=self.w, h=self.h)

    
    

 
