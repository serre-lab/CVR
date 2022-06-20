import copy

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
import cv2
import time

gap_max = 0.006

nb_max_pixels = 50*50

margin = 1


class Shape(object):

    def __init__(self, gap_max=0.007, radius = 0.5, hole_radius = 0.05, randomize=True, sym_flip=False, sym_rot=1):
        self.gap_max = gap_max
        self.radius = radius
        self.hole_radius = hole_radius
        self.nb_pixels = 0
        self.x_pixels = []
        self.y_pixels = []
        self.transformations = []

        self.sym_flip = sym_flip
        self.sym_rot = sym_rot
        
        if randomize:
            self.randomize()
        

    def generate_part_part(self, radius, hole_radius, x1, y1, x2, y2): 
    
        if abs(x1 - x2) > self.gap_max or abs(y1 - y2) > self.gap_max:

            d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)/5
            
            dx = (2 * np.random.rand() - 1) * d
            dy = (2 * np.random.rand() - 1) * d
            while dx**2 + dy**2 > d**2:
                dx = (2 * np.random.rand() - 1) * d
                dy = (2 * np.random.rand() - 1) * d
            x3 = (x1 + x2) / 2 + dx
            y3 = (y1 + y2) / 2 + dy

            while x3**2 + y3**2 > radius**2:
                dx = (2 * np.random.rand() - 1) * d
                dy = (2 * np.random.rand() - 1) * d
                while dx**2 + dy**2 > d**2:
                    dx = (2 * np.random.rand() - 1) * d
                    dy = (2 * np.random.rand() - 1) * d
                x3 = (x1 + x2) / 2 + dx
                y3 = (y1 + y2) / 2 + dy
                

            if self.generate_part_part(radius, hole_radius, x1, y1, x3, y3):
                return True

            if self.generate_part_part(radius, hole_radius, x3, y3, x2, y2):
                return True
            
        else:

            if x1**2 + y1**2 >= radius**2 or x1**2 + y1**2 < hole_radius**2:
                return True
            
            self.x_pixels.append(x1)
            self.y_pixels.append(y1)

        return False

    def generate_part(self, radius, hole_radius):

        err1, err2, err3, err4 = True, True, True, True
        
        while err1 or err2 or err3 or err4:
            nb_pixels = 0
            
            self.x_pixels = []
            self.y_pixels = []

            x1 = np.random.rand() * radius
            y1 = np.random.rand() * radius
            while x1**2 + y1**2 > radius**2 or x1**2 + y1**2 < hole_radius**2:
                x1 = np.random.rand() * radius
                y1 = np.random.rand() * radius

            x2 = -np.random.rand() * radius
            y2 = np.random.rand() * radius
            while x2**2 + y2**2 > radius**2 or x2**2 + y2**2 < hole_radius**2:
                x2 = -np.random.rand() * radius
                y2 = np.random.rand() * radius

            x3 = -np.random.rand() * radius
            y3 = -np.random.rand() * radius
            while x3**2 + y3**2 > radius**2 or x3**2 + y3**2 < hole_radius**2:
                x3 = -np.random.rand() * radius
                y3 = -np.random.rand() * radius

            x4 = np.random.rand() * radius
            y4 = -np.random.rand() * radius
            while x4**2 + y4**2 > radius**2 or x4**2 + y4**2 < hole_radius**2:
                x4 = np.random.rand() * radius
                y4 = -np.random.rand() * radius

            self.n_pixels1 = len(self.x_pixels)
            err1 = self.generate_part_part(radius, hole_radius, x1, y1, x2, y2)
            self.n_pixels2 = len(self.x_pixels)
            err2 = self.generate_part_part(radius, hole_radius, x2, y2, x3, y3)
            self.n_pixels3 = len(self.x_pixels)
            err3 = self.generate_part_part(radius, hole_radius, x3, y3, x4, y4)
            self.n_pixels4 = len(self.x_pixels)
            err4 = self.generate_part_part(radius, hole_radius, x4, y4, x1, y1)


    def randomize(self):
        self.x_pixels = []
        self.y_pixels = []

        # self.nb_pixels = 0
        
        self.generate_part(self.radius, self.hole_radius)

        self.nb_pixels = len(self.x_pixels)

        self.x_pixels = np.array(self.x_pixels)
        self.y_pixels = np.array(self.y_pixels)

        ### random rotation
        self.rotate(np.random.rand() * np.pi * 2)

        if self.sym_rot > 1:
            self.rot_symmetrize(self.sym_rot)
        if self.sym_flip:
            self.flip_symmetrize()
        
        self.x_pixels = self.x_pixels - (self.x_pixels.max() + self.x_pixels.min())/2
        self.y_pixels = self.y_pixels - (self.y_pixels.max() + self.y_pixels.min())/2

        ### resets to a squre
        self.x_pixels = self.x_pixels/(self.x_pixels.max() - self.x_pixels.min())
        self.y_pixels = self.y_pixels/(self.y_pixels.max() - self.y_pixels.min())

        w = self.x_pixels.max() - self.x_pixels.min() # = w
        h = self.y_pixels.max() - self.y_pixels.min() # = h
        
        self.wh = (1, 1)

        self.transformations = []

    def flip_diag(self):
        self.x_pixels, self.y_pixels = self.y_pixels, self.x_pixels

    def get_contour(self):
        p1 = np.stack([self.x_pixels, self.y_pixels], 1)
        p1 = np.concatenate([p1, p1[:1]], 0)
        return p1

    def rotate(self, alpha):
        ux, uy = np.cos(alpha), -np.sin(alpha)
        vx, vy = np.sin(alpha), np.cos(alpha)
        
        x = self.x_pixels * ux + self.y_pixels * uy
        y = self.x_pixels * vx + self.y_pixels * vy

        self.x_pixels = x
        self.y_pixels = y
        
        self.transformations.append(('r', alpha))
        
        w = self.x_pixels.max() - self.x_pixels.min() # = w
        h = self.y_pixels.max() - self.y_pixels.min() # = h
        
        temp_size = np.sqrt(w * h)
        self.bb = (w, h)
        self.wh = (w/temp_size, h/temp_size)

    
    def get_bb(self):
        w = self.x_pixels.max() - self.x_pixels.min()
        h = self.y_pixels.max() - self.y_pixels.min()
        return (w,h)

    def scale(self, s): 
        if isinstance(s, tuple):
            self.x_pixels = self.x_pixels*s[0]
            self.y_pixels = self.y_pixels*s[1]
        else:
            self.x_pixels = self.x_pixels*s
            self.y_pixels = self.y_pixels*s
        
        self.transformations.append(('s', s))

        w = self.x_pixels.max() - self.x_pixels.min() # = w
        h = self.y_pixels.max() - self.y_pixels.min() # = h
        
        temp_size = np.sqrt(w * h)
        # temp_size = w * h
        self.bb = (w, h)
        self.wh = (w/temp_size, h/temp_size)

    def set_wh(self, wh):
        current_w = self.x_pixels.max() - self.x_pixels.min()
        current_h = self.y_pixels.max() - self.y_pixels.min()
        
        scale = (wh[0]/current_w, wh[1]/current_h) 
        self.scale(scale)

    def set_size(self, s):
        current_size = np.sqrt(self.bb[0]*self.bb[1])
        self.scale(s/current_size)

    def clone(self):
        s = self.__class__(randomize=False)
        s.nb_pixels = copy.deepcopy(self.nb_pixels)
        s.x_pixels = np.copy(self.x_pixels)
        s.y_pixels = np.copy(self.y_pixels)
        s.n_pixels1 = copy.deepcopy(self.n_pixels1)
        s.n_pixels2 = copy.deepcopy(self.n_pixels2)
        s.n_pixels3 = copy.deepcopy(self.n_pixels3)
        s.n_pixels4 = copy.deepcopy(self.n_pixels4)
        s.wh = copy.deepcopy(self.wh)
        
        s.transformations = copy.deepcopy(self.transformations)
        
        return s
    
    def flip(self):
        self.x_pixels = - self.x_pixels

        self.transformations.append(('f', None))
    
    def subsample(self):
        # sample many intermediate points
        N = max(500//len(self.x_pixels), 1)
        xy = np.stack([self.x_pixels, self.y_pixels], 1)

        if N>1:
            a = np.linspace(0,1,N)
            xy = xy[:-1,None,:] + (xy[1:,None,:]-xy[:-1,None,:])*a[None,:,None]
            xy = xy.reshape([-1, 2])

        for i in range(1,len(xy)-1):
            xy[i] = (xy[i+1] + xy[i-1])/2 + (np.random.rand()-0.5)*(xy[i+1] - xy[i-1])[::-1] * np.array([-1,1]) /np.linalg.norm(xy[i+1] - xy[i-1]) * 0.07

        self.x_pixels = xy[:,0]
        self.y_pixels = xy[:,1]

    def smooth(self, window=10):
        xy = np.stack([self.x_pixels, self.y_pixels], 1)
        xy = np.concatenate([xy, xy[1:window]], 0)
        xy = xy.cumsum(0)
        xy = (xy[window:] - xy[:-window])/window

        self.x_pixels = xy[:,0]
        self.y_pixels = xy[:,1]
    
    def reset(self):
        
        for t, v in reversed(self.transformations):
            if t == 'r':
                self.rotate(-v)
            if t == 's':
                if isinstance(v, tuple):
                    v = (1/v[0], 1/v[1])
                else:
                    v = 1/v
                self.scale(v)
            if t == 'f':
                self.flip()
        
        self.x_pixels = self.x_pixels - (self.x_pixels.max() + self.x_pixels.min())/2
        self.y_pixels = self.y_pixels - (self.y_pixels.max() + self.y_pixels.min())/2

        self.transformations = []
        
        w = self.x_pixels.max() - self.x_pixels.min() # = w
        h = self.y_pixels.max() - self.y_pixels.min() # = h
        
        temp_size = np.sqrt(w * h)

        self.bb = (w, h)
        self.wh = (w/temp_size, h/temp_size)

    def get_hole_radius(self):
        
        img = (np.zeros((100,100)) * 255).astype(np.uint8)
        x_pixels = np.concatenate([self.x_pixels, self.x_pixels[0:1]]) 
        y_pixels = np.concatenate([self.y_pixels, self.y_pixels[0:1]]) 
        
        contours = np.stack([x_pixels,y_pixels], 1) * 70 + 50
        contours = contours.astype(np.int32)
        contours[:10,:10]
        cv2.fillPoly(img, pts =[contours], color=(255,255,255))

        dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)

        idx = dist.argmax()
        rad = dist[idx//100, idx%100]
        point = np.array([idx//100, idx%100])

        point1 = (point - 50)/70
        rad1 = rad/70

        self.circle_center = point1
        self.max_radius = rad1


class ShapeCurl(Shape):

    def randomize(self):
        self.x_pixels = []
        self.y_pixels = []

        self.generate_part(self.radius, self.hole_radius)

        self.nb_pixels = len(self.x_pixels)

        self.x_pixels = np.array(self.x_pixels)
        self.y_pixels = np.array(self.y_pixels)

        ### random rotation
        self.rotate(np.random.rand() * np.pi * 2)

        self.subsample()
        
        if self.sym_rot > 1:
            self.rot_symmetrize(self.sym_rot)
        if self.sym_flip:
            self.flip_symmetrize()

        self.x_pixels = self.x_pixels - (self.x_pixels.max() + self.x_pixels.min())/2
        self.y_pixels = self.y_pixels - (self.y_pixels.max() + self.y_pixels.min())/2

        ### resets to a squre
        self.x_pixels = self.x_pixels/(self.x_pixels.max() - self.x_pixels.min())
        self.y_pixels = self.y_pixels/(self.y_pixels.max() - self.y_pixels.min())

        w = self.x_pixels.max() - self.x_pixels.min() # = w
        h = self.y_pixels.max() - self.y_pixels.min() # = h
        
        self.wh = (1, 1)


        self.transformations = []
