
import os
import copy
import pickle
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image

import itertools
import math

import cv2

from data_generation.shape import Shape
from data_generation.utils import *

render = render_cv


def sample_random_colors(n_samples):
    h = np.random.rand(n_samples)
    s = np.random.rand(n_samples) * 0.5
    v = np.random.rand(n_samples) * 0.8

    color = np.stack([h,s,v],1)
    return color

#################################################################################
# shape 

def task_shape(condition='xysc'): 
     
    n_samples = 4

    max_size = 0.8
    min_size = max_size/2

    s = Shape()
    s_odd = Shape()
    shape = [[s.clone()] for i in range(n_samples-1)] + [[s_odd]]

    if 's' in condition:
        size = np.random.rand(n_samples) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size*np.ones(n_samples)

    if 'x' in condition:
        x = np.random.rand(n_samples) * (1-size) + size/2
    else:
        x = np.random.rand() * (1-size.max()) + size.max()/2
        x = x*np.ones(n_samples)

    if 'y' in condition:
        y = np.random.rand(n_samples) * (1-size) + size/2
    else:
        y = np.random.rand() * (1-size.max()) + size.max()/2
        y = y*np.ones(n_samples)
    
    xy = np.stack([x,y], 1)[:,None,:]
    size = size[:,None]

    if 'r' in condition:
        angles = np.random.rand(n_samples-1) * 2 * np.pi
        for i in range(n_samples-1):
            shape[i][0].rotate(angles[i])

    if 'f' in condition:
        flips = np.array([0,1,0,1])[np.random.permutation(4)]
        for i in range(n_samples-1):
            if flips[i] == 1:
                shape[i][0].flip()

    if 'c' in condition:
        color = sample_random_colors(n_samples)
        color = [color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# position

def task_pos(condition='xycsid'):

    n_samples = 4

    max_size = 0.6
    min_size = max_size/2

    if 's' in condition:
        size = np.random.rand(n_samples) * (max_size-min_size) + min_size
        size_max = size.max()
    else:
        size = np.random.rand() * (max_size-min_size) + min_size

    xy = np.random.rand(2) * (1-size_max) + size_max/2
    if 'xy' in condition:
        xy_odd = np.random.rand(2) * (1-size_max) + size_max/2
    elif 'x' in condition:
        xy_odd = np.random.rand(2) * (1-size_max) + size_max/2
        xy_odd[0] = xy[0]
    elif 'y' in condition:
        xy_odd = np.random.rand(2) * (1-size_max) + size_max/2
        xy_odd[1] = xy[1]
    
    while np.linalg.norm(xy_odd - xy) < 0.2:
        # xy_odd = np.random.rand(2)

        xy = np.random.rand(2) * (1-size_max) + size_max/2
        if 'xy' in condition:
            xy_odd = np.random.rand(2) * (1-size_max) + size_max/2
        elif 'x' in condition:
            xy_odd = np.random.rand(2) * (1-size_max) + size_max/2
            xy_odd[0] = xy[0]
        elif 'y' in condition:
            xy_odd = np.random.rand(2) * (1-size_max) + size_max/2
            xy_odd[1] = xy[1]

    if 'c' in condition:
        c = sample_random_colors(n_samples)
        color = c[:,None,:]
    else:
        c = sample_random_colors(1)
        color = c[None, :,:] * np.ones([n_samples,1,3])

    if 'id' in condition:
        shapes = [[Shape()] for i in range(n_samples)]
    else:
        shape = Shape()
        shapes = [[shape.clone()] for i in range(n_samples)]
    
    xy = np.ones([n_samples, 1, 2]) * xy[None,None,:]
    xy[-1,0] = xy_odd
    
    size = np.ones([n_samples, 1]) * size
    
    return xy, size, shapes, color

#################################################################################
# size

def task_size(condition='xyidc'):

    n_samples = 4

    # image and object parameters
    internal_frame = 0.8
    pad = (1-internal_frame)/2
    
    max_size = 0.6
    min_size = max_size/2
    
    min_diff = (max_size - min_size)/3
    
    size = np.random.rand() * (max_size-min_size) + min_size
    min_prop, max_prop = 1.5/5, 1/2
    prop = np.random.rand() * (max_prop - min_prop) + min_prop
    
    if np.random.randint(2)==0:
        prop = 1/prop

    size_odd = size * prop
    if size + size_odd > 0.9:
        size, size_odd = size/(size + size_odd)*0.9, size_odd/(size + size_odd)*0.9
    # size_odd = np.random.rand(20) * (max_size-min_size) + min_size
    # size_odd = [s for s in size_odd if np.abs(s-size)>min_diff][0]
    
    size = np.ones(n_samples)*size
    size[-1] = size_odd

    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            s = Shape()
            shape.append([s])
    else:
        shape = Shape()
        shape = [[shape.clone()] for _ in range(n_samples)]

    if 'x' in condition:
        x = np.random.rand(n_samples) * (1-size) + size/2
    else:
        x = np.random.rand() * (1-size.max()) + size.max()/2
        x = x * np.ones(n_samples)


    if 'y' in condition:
        y = np.random.rand(n_samples) * (1-size) + size/2
    else:
        y = np.random.rand() * (1-size.max()) + size.max()/2
        y = y * np.ones(n_samples)

    if 'c' in condition:
        c = sample_random_colors(n_samples)
        color = c[:,None,:]
    else:
        c = sample_random_colors(1)
        color = c[None, :,:] * np.ones([n_samples,1,3])

    xy = np.stack([x,y], 1)[:,None,:]
    size = size[:,None]
    
    return xy, size, shape, color

#################################################################################
# color

def task_color(condition='xysid'):
     
    n_samples = 4
    
    max_size = 0.6
    min_size = max_size/4

    h = np.random.rand()
    s = np.random.rand() * 0.4 + 0.1
    v = np.random.rand() * 0.5 + 0.4
    min_diff = 0.15
    h_odd = (h + (np.random.rand() * (1-min_diff*2) + min_diff) * np.random.choice([-1,1]))%1

    color = [[[h,s,v]] for i in range(n_samples-1)] + [[[h_odd, s, v]]]

    if 'id' in condition:
        shape = [[Shape()] for i in range(n_samples)]
    else:
        s = Shape()
        shape = [[s.clone()] for i in range(n_samples)]

    if 's' in condition:
        size = np.random.rand(n_samples) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size*np.ones(n_samples)

    if 'x' in condition:
        x = np.random.rand(n_samples) * (1-size) + size/2
    else:
        x = np.random.rand() * (1-size.max()) + size.max()/2
        x = x*np.ones(n_samples)

    if 'y' in condition:
        y = np.random.rand(n_samples) * (1-size) + size/2
    else:
        y = np.random.rand() * (1-size.max()) + size.max()/2
        y = y*np.ones(n_samples)
    
    xy = np.stack([x,y], 1)[:,None,:]
    size = size[:,None]

    if 'r' in condition:
        angles = np.random.rand(n_samples-1) * 2 * np.pi
        for i in range(n_samples-1):
            shape[i][0].rotate(angles[i])

    return xy, size, shape, color

#################################################################################
# rot

task_rot = lambda: task_shape(condition='xyscr')

#################################################################################
# flip

task_flip = lambda: task_shape(condition='xyscf')

#################################################################################
# inside

def task_inside(condition='c'):
    
    n_samples = 4
    
    max_size = 0.6
    min_size = max_size/2

    size_a = np.random.rand(n_samples) * (max_size - min_size) + min_size 
    size_b = np.random.rand(n_samples) * (size_a/2.5 - size_a/4) + size_a/4

    done = False
    max_attempts = 10 

    range_1 = 1 - size_a[:,None]
    starting_1 = size_a[:,None]/2

    xy1 = np.random.rand(n_samples,2) * range_1 + starting_1 

    xy2 = []
    shapes = []
    
    for i in range(n_samples-1):
        done = False

        s1 = Shape(gap_max=0.07, hole_radius=0.2)
        s2 = Shape(gap_max=0.01)
        for _ in range(max_attempts):
            
            samples = sample_position_inside_1(s1, s2, size_b[i]/size_a[i])
            if len(samples)>0:
                done = True
            
            if done:
                break
            else:
                s1.randomize()
                s2.randomize()

        if not done:
            return np.zeros([100,100])

        xy2.append(samples[0])
        shapes.append([s1, s2])

    range_2 = 1 - size_b[-1]
    starting_2 = size_b[-1]/2
    xy2_odd = np.random.rand(100,2) * range_2 + starting_2 
    xy1_odd = xy1[-1:]

    xy2_odd = xy2_odd[(np.abs(xy2_odd - xy1_odd) > (size_a[-1] + size_b[-1])/2).any(1)]
    xy2_odd = xy2_odd[0:1]

    s1 = Shape(gap_max=0.01, hole_radius=0.2)
    s2 = Shape(gap_max=0.01)
    shapes.append([s1,s2])
    xy2 = np.concatenate([np.array(xy2)*size_a[:-1,None] + xy1[:-1],xy2_odd], 0)
    
    xy = np.stack([xy1, xy2], axis=1)
    size = np.stack([size_a, size_b], axis=1)

    if 'c' in condition:
        color = sample_random_colors(n_samples)
        color = [np.ones([2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([2, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, size, shapes, color

#################################################################################
# contact

def task_contact(condition='sc'):
        
    n_samples = 4
    n_objects = 2

    max_size = 0.2
    min_size = max_size/2

    size = np.random.rand(n_samples, 2) * (max_size - min_size) + min_size

    shape = []
    xy = []
    for i in range(n_samples):
            
        s1 = Shape()
        s1.randomize()
        s2 = Shape()
        s2.randomize()

        if i == n_samples-1:
            xy_ = np.random.rand(2,2) * (1-size[i,:,None]) + size[i,:,None]/2
            while not (np.abs(xy_[0] - xy_[1]) - size[i].sum()/2 > 0).any():
                xy_ = np.random.rand(2,2) * (1-size[i,:,None]) + size[i,:,None]/2
            
            xy.append(xy_)
            
        else:
            positions, clump_size = sample_contact_many([s1, s2], size[i])
        
            xy0 = np.random.rand(2) * (1-clump_size) + clump_size/2
            xy_ = positions + xy0[None,:]

            xy.append(xy_)

        shape.append([s1,s2])

    xy = np.stack(xy, 0)
    

    if 'c' in condition:
        color = sample_random_colors(n_samples)
        color = [np.ones([2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([2, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# count

def task_count(condition='xysidc', condition_arr='xysid'):
        
    n_samples = 4
    n_objects = np.random.randint(4,9)
    n_objects_odd = [n for n in np.random.randint(4,9, size=20) if n!=n_objects][0]

    n_objs = np.ones(n_samples, dtype=int)*n_objects
    n_objs[-1] = n_objects_odd


    max_n_objs = max(n_objects, n_objects_odd)
    
    max_size = 0.9/np.sqrt(max_n_objs*4)
    min_size = max_size/2

    shape = Shape()

    if 'id' not in condition_arr and 'id' not in condition:
        id_idx = np.zeros([n_samples, max_n_objs])
    else:
        if np.random.rand()<0.33:
            id_idx = np.stack([np.arange(n_samples)]*max_n_objs, axis=1)
        elif np.random.rand()<0.5:
            id_idx = np.stack([np.arange(max_n_objs)]*n_samples, axis=0)
        else:
            id_idx = np.arange(max_n_objs*n_samples).reshape([n_samples, max_n_objs])

    if 's' not in condition_arr and 's' not in condition:
        s_idx = np.zeros([n_samples, max_n_objs])
    else:
        if np.random.rand()<0.33:
            s_idx = np.stack([np.arange(n_samples)]*max_n_objs, axis=1)
        elif np.random.rand()<0.5:
            s_idx = np.stack([np.arange(max_n_objs)]*n_samples, axis=0)
        else:
            s_idx = np.arange(max_n_objs*n_samples).reshape([n_samples, max_n_objs])

    unique_s_idx = np.unique(s_idx)
    
    unique_sizes = np.random.rand(len(unique_s_idx)) * (max_size - min_size) + min_size
    sizes = unique_sizes[s_idx.flatten().astype(int)].reshape(s_idx.shape)

    if 'xy' in condition:
        # x = np.random.rand(n_samples)
        
        xy = []
        for i in range(n_samples):
            xy_ = sample_positions(sizes[i:i+1,:n_objs[i]], n_sample_min=1)
            xy.append(xy_[0])

        # xy = np.random.rand(n_samples, max_n_objs, 2) * (1-sizes[:,:,None]) + sizes[:,:,None]/2
        # def stop_cond(xy, sizes):
        #     dists = np.abs(xy[:,:,None,:] - xy[:,None,:,:])
        #     max_dists = sizes[:,:,None,None]/2 + sizes[:,None,:,None]/2
        #     all_d = (dists > max_dists).any(3)
        #     return all([all_d[i][np.triu_indices(max_n_objs,1)].all() for i in range(n_samples)])

        # while not stop_cond(xy, sizes):
        #     xy = np.random.rand(n_samples, max_n_objs, 2) * (1-sizes[:,:,None]) + sizes[:,:,None]/2

    else:
        xy = []
        for i in range(n_samples):
            xy_ = sample_positions(sizes[i:i+1,:n_objs[i]], n_sample_min=1)
            xy.append(xy_[0])

        # xy = np.random.rand(max_n_objs,2) * (1-sizes.max(0)[:,None]) + sizes.max(0)[:,None]/2
        # def stop_cond(xy, sizes):
        #     dists = np.abs(xy[None,:,None,:] - xy[None,None,:,:])
        #     max_dists = sizes[:,:,None,None]/2 + sizes[:,None,:,None]/2
        #     all_d = (dists > max_dists).any(3)
        #     return all([all_d[i][np.triu_indices(max_n_objs,1)].all() for i in range(n_samples)])

        # while not stop_cond(xy, sizes):
        #     xy = np.random.rand(max_n_objs,2) * (1-sizes.max(0)[:,None]) + sizes.max(0)[:,None]/2

        # xy = np.stack([xy]*n_samples, 0)

    unique_id = np.unique(id_idx)
    all_shapes = []
    for i in range(len(unique_id)):
        shape = Shape()
        all_shapes.append(shape)

    shapes = [[all_shapes[k].clone() for k in id_idx[i][:n_objects].astype(int)] for i in range(n_samples-1)]
    shapes = shapes + [[all_shapes[k].clone() for k in id_idx[-1][:n_objects_odd].astype(int)]]


        
    if 'c' in condition:
        color = [sample_random_colors(n_objs[i]) for i in range(n_samples)]
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objs[i], 3]) * color[i:i+1] for i in range(n_samples)]

    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objs[i], 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    xy = [xy[i][:n_objs[i]] for i in range(n_samples)]
    size = [sizes[i][:n_objs[i]] for i in range(n_samples)]
    shape = shapes

    return xy, size, shape, color


#################################################################################
#################################################################################
#################################################################################


#################################################################################
# position + position

def task_pos_pos_1(condition='xycsid'):
    
    n_samples = 4
    
    n_objects = np.random.randint(3,7)
    
    max_size_sc = 0.7
    min_size_sc = max_size_sc/2
    
    if 's' in condition:
        size_sc = np.random.rand(n_samples) * (max_size_sc - min_size_sc) + min_size_sc
    else:
        size_sc = np.random.rand() * (max_size_sc - min_size_sc) + min_size_sc
        size_sc = size_sc * np.ones([n_samples])
    
    size_sc_min = size_sc.min()

    max_size = size_sc_min/(n_objects)*1.3
    min_size = max_size/2
    
    # size = np.random.rand(n_objects) * (max_size - min_size) + min_size
    size = np.random.rand() * (max_size - min_size) + min_size
    
    triu_idx = np.triu_indices(n_objects, k=1)
    triu_idx = triu_idx[0]*n_objects + triu_idx[1]

    n_samples_over = 100
    spatial_config = np.random.rand(n_samples_over, n_objects, 2) * (size_sc_min - size)

    valid = (np.abs(spatial_config[:,:,None,:] - spatial_config[:,None,:,:]) - size > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
    while valid.sum() < 2:
        spatial_config = np.random.rand(n_samples_over, n_objects, 2) * (size_sc_min - size)
        valid = (np.abs(spatial_config[:,:,None,:] - spatial_config[:,None,:,:]) - size > 0).any(3).reshape([n_samples_over, n_objects**2])[:, triu_idx].all(1)
    
    spatial_config = spatial_config[valid]
    spatial_config = spatial_config - spatial_config.mean(1)[:,None,:]
    
    sc = spatial_config[0]
    sc_odd = spatial_config[1]
    sc = np.stack([sc]*3 + [sc_odd], 0)

    sc = sc / (size_sc_min - size) * (size_sc[:,None,None] - size)

    if 'xy' in condition:
        xy_sc = np.random.rand(n_samples,2) * (1 - size_sc[:,None]) + size_sc[:,None]/2
    else:
        xy_sc = np.ones([n_samples, 2]) * np.random.rand(1,2) * (1 - size_sc.max()) + size_sc.max()/2

    xy = xy_sc[:,None,:] + sc
    # xy[-1] = xy_sc[-1:,:] + sc_odd

    size = size * np.ones([n_samples, n_objects])

    if 'c' in condition:
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # c = sample_random_colors(n_samples)
        # color = c[:,None,:] * np.ones([n_samples, n_objects, 3])
    else:
        c = sample_random_colors(1)
        color = c[None, :,:] * np.ones([n_samples,n_objects, 3])

    if 'id' in condition:
        shapes = [[Shape() for _ in range(n_objects)] for _ in range(n_samples)]
    else:
        s = Shape()
        shapes = [[s.clone() for _ in range(n_objects)] for _ in range(n_samples)]

    return xy, size, shapes, color

def task_pos_pos_2(condition='xycsid'):
        
    n_samples = 4
    
    n_objects = np.random.randint(3,6)
    
    max_size_sc = 0.9/2 # n_forms
    min_size_sc = max_size_sc/2
    
    if 's' in condition:
        size_sc = np.random.rand(n_samples, 2) * (max_size_sc - min_size_sc) + min_size_sc
    else:
        size_sc = np.random.rand() * (max_size_sc - min_size_sc) + min_size_sc
        size_sc = size_sc * np.ones([n_samples, 2])
    
    size_sc_min = size_sc.min()

    max_size = size_sc_min/np.sqrt(n_objects*3)
    min_size = max_size*2/3
    
    # size = np.random.rand(n_objects) * (max_size - min_size) + min_size
    size = np.random.rand() * (max_size - min_size) + min_size
    
    triu_idx = np.triu_indices(n_objects, k=1)
    triu_idx = triu_idx[0]*n_objects + triu_idx[1]

    n_samples_over = 100

    n_unique_sc = n_samples + 1
    n_unique_sc_left = n_unique_sc
    unique_sc = []

    while n_unique_sc_left>0:
        spatial_config = np.random.rand(n_samples_over, n_objects, 2) * (size_sc_min - size)
        valid = (np.abs(spatial_config[:,:,None,:] - spatial_config[:,None,:,:]) - size > 0).any(3).reshape([n_samples_over, n_objects**2])[:, triu_idx].all(1)
        if valid.sum()>0:
            unique_sc.append(spatial_config[valid][:n_unique_sc_left])
            n_unique_sc_left = n_unique_sc_left - valid.sum()

    unique_sc = np.concatenate(unique_sc, axis=0)
    unique_sc = unique_sc - unique_sc.mean(1)[:,None,:]

    co1 = sample_over_range_t(n_samples, np.array([0,1]), size_sc)
    co2 = np.random.rand(n_samples, 2) * (1-size_sc) + size_sc/2
    xy_sc = [co1, co2] if np.random.randint(2)==0 else [co2, co1]
    xy_sc = np.stack(xy_sc, axis=2)


    # unique_sc
    sc = np.stack([unique_sc[:-1], np.concatenate([unique_sc[:-2], unique_sc[-1:]],0)], 1)
    # sc = sc.reshape([n_samples, 2, n_objects, 2])    
    # sc = np.concatenate([unique_sc[:-1], unique_sc[:-2], unique_sc[-1:]], 0).reshape([n_samples, 2, n_objects, 2])
    sc = sc / (size_sc_min - size) * (size_sc[:,:,None,None] - size)

    xy = xy_sc[:,:,None,:] + sc

    xy = xy.reshape([n_samples, n_objects*2, 2])
    size = size * np.ones([n_samples, n_objects*2])
    
    if 'c' in condition:
        c = sample_random_colors(n_samples*n_objects*2)
        color = c.reshape([n_samples, n_objects*2, 3])
        # c = sample_random_colors(n_samples)
        # color = c[:,None,:] * np.ones([n_samples, n_objects*2, 3])
    else:
        c = sample_random_colors(1)
        color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    if 'id' in condition:
        shapes = [[Shape() for _ in range(n_objects*2)] for _ in range(n_samples)]
    else:
        s = Shape()
        shapes = [[s.clone() for _ in range(n_objects*2)] for _ in range(n_samples)]

        # s1 = Shape()
        # s2 = Shape()
        # shapes = [[s1.clone() for _ in range(n_objects)] + [s2.clone() for _ in range(n_objects)] for _ in range(n_samples)]

    return xy, size, shapes, color

def task_pos_pos_3(condition='s_id_c'):
    
    n_samples = 4
    n_objects = 4
    
    index = np.random.randint(n_objects)
    
    max_size = 0.8/3
    min_size = max_size/2
    
    if 's' in condition:
        size = np.random.rand(n_samples, 4) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = np.ones([n_samples, 4]) * size
    
    if 'id' in condition:
        shapes = [[Shape() for _ in range(4)] for _ in range(n_samples)]
    else:
        s = Shape()
        shapes = [[s.clone() for _ in range(4)] for _ in range(n_samples)]

    range_d = (1-size.reshape([n_samples, 2, 2]).sum(2).max(1))
    starting_d = size.reshape([n_samples, 2, 2]).sum(2).max(1)/2
    dist = np.random.rand(n_samples, 2) * range_d[:,None] + starting_d[:,None]
    idx_change = dist[:,0]>dist[:,1]
    dist[idx_change] = dist[idx_change, ::-1]
    
    dist[-1] = dist[-1, ::-1]

    y = sample_over_range_t(n_samples, np.array([0,1]), size.reshape([n_samples, 2, 2]).max(2))
    
    range_x = (1 - dist - size.reshape([n_samples, 2, 2]).sum(2)/2)
    starting_x = dist/2 + size.reshape([n_samples, 2, 2])[:,:,0]/2
    xc = np.random.rand(n_samples, 2) * range_x + starting_x
    
    x = np.stack([xc[:,0] - dist[:,0]/2, xc[:,0] + dist[:,0]/2, xc[:,1] - dist[:,1]/2, xc[:,1] + dist[:,1]/2], 1)
    y = np.stack([y[:,0], y[:,0], y[:,1], y[:,1]], 1)

    ## gen change
    # xy = np.stack([x,y], 2)
    xy = np.stack([y,x], 2)
    
    if 'c' in condition:
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, size, shapes, color

def task_pos_pos_4(condition='cid'):

    a = np.random.choice(3)
    if a==0:
        condition+='min'
    elif a==1: 
        condition+='avg'
    else: 
        condition+='max'

    a = np.random.choice(2)
    if a==0:
        axis = 'y'
    else:
        axis = 'x'

    n_samples = 4

    n_objects = np.random.randint(3, 5)
    
    index = np.random.randint(n_objects)
    
    max_size = 0.8/n_objects
    min_size = max_size/2
    
    k = np.random.rand()

    size = np.random.rand(n_samples) * (max_size - min_size) + min_size

    size = np.ones([n_samples, n_objects]) * size[:,None]

    x = sample_over_range_t(n_samples, np.array([0,1]), size)    
    y = sample_over_range_t(n_samples, np.array([0.1,0.9]), size/3)    

    if np.math.factorial(n_objects-1) < n_samples:
        perms = np.array([np.random.permutation(n_objects-1)]*n_samples)
    else:
        perms = np.array([np.random.permutation(n_objects-1) for i in range(n_samples)])
        diffs = np.abs(perms[:,None,:] - perms[None,:,:]).sum(-1)[np.triu_indices(n_samples,1)]
        while (diffs == 0).any():
            perms = np.array([np.random.permutation(n_objects-1) for i in range(n_samples)])
            diffs = np.abs(perms[:,None,:] - perms[None,:,:]).sum(-1)[np.triu_indices(n_samples,1)]

    if 'max' in condition:
        y_all, y_max = y[:,:-1], y[:,-1]
    elif 'min' in condition:
        y_all, y_max = y[:,1:], y[:,0]
    elif 'avg' in condition:
        idx = n_objects//2
        y_max = y[:,idx]
        y_all = np.concatenate([y[:,:idx], y[:,idx+1:]], axis=1)

    for i in range(n_samples-1):
        y_all[i] = y_all[i][perms[i]]
    
    # y_reg = np.concatenate(y_all[:-1], y_max[:-1, None]
     
    # get the highest ys
    y_reg = np.insert(y_all[:-1], index, y_max[:-1], axis=1)
    # sample  index other than index
    index_odd = [i for i in np.random.randint(n_objects, size=10) if i!=index][0] 
    y_odd = np.insert(y_all[-1], index_odd, y_max[-1])
    
    y = np.concatenate([y_reg, y_odd[None,:]], 0)

    # if axis=='x':
    #     xy = np.stack([y,x], 2)
    # else:
    #     xy = np.stack([x,y], 2)
    xy = np.stack([x,y], 2)

    
    if 'id' in condition:
        s = Shape()
        shapes = [[s.clone() for _ in range(n_objects)] for _ in range(n_samples)]
    else:
        shapes = [[Shape() for _ in range(n_objects)] for _ in range(n_samples)]
    
    if 'c' in condition:
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    shape = shapes

    if axis=='x':
        for i in range(n_samples):
            xy[i], shape[i] = flip_diag_scene(xy[i], shape[i])

    return xy, size, shape, color


#################################################################################
# position + size

def task_pos_size_1(condition='cid'):
        
    n_samples = 4
    n_objects = 2
    
    max_size = 0.3
    min_size = max_size/2

    # max_r, min_r = 2/3, 1/5
    max_r, min_r = 4/5, 1/2

    r = np.random.rand(n_samples) * (max_r-min_r) + min_r
    size = np.random.rand(n_samples) * (max_size-min_size) + min_size
    size = np.stack([size, size*r], 1)
    non_valid = size.sum(1)>0.9
    if non_valid.any():
        size[non_valid] = size[non_valid] / size[non_valid].sum(1)[:,None] * 0.9
    # odd one out
    size[-1] = size[-1,::-1]

    co1 = sample_over_range_t(n_samples, np.array([0,1]), size)
    co2 = np.random.rand(n_samples, 2) * (1-size) + size/2
    xy = [co1, co2] if np.random.randint(2)==0 else [co2, co1]
    xy = np.stack(xy, axis=2)


    if 'c' in condition:
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([2, 3]) * color for i in range(n_samples)]

    if 'id' in condition:
        shape = [[Shape(), Shape()] for i in range(n_samples)]
    else:
        s = Shape()
        shape = [[s.clone(), s.clone()] for _ in range(n_samples)]

    if 'r' in condition:
        angle = np.random.rand(n_samples, 2) * 2 * np.pi
        for i in range(n_samples-1):
            shape[i][0].rotate(angle[i,0])
            shape[i][1].rotate(angle[i,1])

    if 'f' in condition:
        for i in range(n_samples-1):
            shape[i][1].flip()

    return xy, size, shape, color

def task_pos_size_2(condition='cid'):
    
    n_samples = 4
    
    n_objects = np.random.randint(4,7)
    
    max_size_sc = 0.9*2/3 # n_forms
    min_size_sc = max_size_sc*2/3
    max_r, min_r = 2/3, 1/2
    r = np.random.rand(n_samples) * (max_r-min_r) + min_r
    
    size_sc = np.random.rand(n_samples) * (max_size_sc - min_size_sc) + min_size_sc

    size_sc = np.stack([size_sc, size_sc*r], 1)
    non_valid = size_sc.sum(1)>0.9
    if non_valid.any():
        size_sc[non_valid] = size_sc[non_valid] / size_sc[non_valid].sum(1)[:,None] * 0.9

    # odd one out
    size_sc[-1] = size_sc[-1,::-1]

    max_size = size_sc.min(0)/np.sqrt(n_objects*3)
    min_size = max_size #*2/3
    
    # size = np.random.rand(n_objects) * (max_size - min_size) + min_size
    size = np.random.rand(2) * (max_size - min_size) + min_size

    triu_idx = np.triu_indices(n_objects, k=1)
    triu_idx = triu_idx[0]*n_objects + triu_idx[1]

    n_samples_over = 100

    # n_unique_sc = n_samples + 1
    n_unique_sc = 2
    n_unique_sc_left = n_unique_sc
    unique_sc = []
    i = 0
    while n_unique_sc_left>0:
        spatial_config = np.random.rand(n_samples_over, n_objects, 2) * (size_sc.min(0)[i] - size[i])
        valid = (np.abs(spatial_config[:,:,None,:] - spatial_config[:,None,:,:]) - size[i] > 0).any(3).reshape([n_samples_over, n_objects**2])[:, triu_idx].all(1)
        if valid.sum()>0:
            unique_sc.append(spatial_config[valid][:1])
            n_unique_sc_left = n_unique_sc_left - 1
            i += 1

            # unique_sc.append(spatial_config[valid][:n_unique_sc_left])
            # n_unique_sc_left = n_unique_sc_left - valid.sum()

    unique_sc = np.concatenate(unique_sc, axis=0)
    unique_sc = unique_sc - unique_sc.mean(1)[:,None,:]
    unique_sc = unique_sc

    sc = ((size_sc - size[None,:])/ (size_sc.min(0)[None,:] - size[None,:]))[:, :, None, None] * unique_sc[None, :, :, :]
    # sc = unique_sc[0]
    # sc_odd = 

    xy_sc_ = []
    n_unique_xy_sc_left = n_samples
    i = 0
    while n_unique_xy_sc_left>0:
        xy_sc = np.random.rand(n_samples_over,2,2) * (1 - size_sc[i:i+1] - size[None,:]) + (size_sc[i:i+1]  + size[None,:])/2
        valid = (np.abs(xy_sc[:,0,:] - xy_sc[:,1,:]) - size_sc[i:i+1] > 0).any(1)
        if valid.any():
            xy_sc_.append(xy_sc[valid][0])
            n_unique_xy_sc_left = n_unique_xy_sc_left - 1
            i += 1

    xy_sc = np.stack(xy_sc_, 0)    
    
    xy = xy_sc[:,:,None,:] + sc

    xy = xy.reshape([n_samples, n_objects*2, 2])
    size = size[None,:,None] * np.ones([n_samples, 2, n_objects])
    size = size.reshape([n_samples, n_objects*2])
    
    if 'c' in condition:
        c = sample_random_colors(n_samples*n_objects*2)
        color = c.reshape([n_samples, n_objects*2, 3])
        # c = sample_random_colors(n_samples)
        # color = c[:,None,:] * np.ones([n_samples, n_objects*2, 3])
    else:
        c = sample_random_colors(1)
        color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    if 'id' in condition:
        shapes = [[Shape() for _ in range(n_objects*2)] for _ in range(n_samples)]
    else:
        s = Shape()
        shapes = [[s.clone() for _ in range(n_objects*2)] for _ in range(n_samples)]

        s1 = Shape()
        s2 = Shape()
        shapes = [[s1.clone() for _ in range(n_objects)]+[s2.clone() for _ in range(n_objects)] for _ in range(n_samples)]

    return xy, size, shapes, color

#################################################################################
# position + shape

def task_pos_shape_1(condition='idcs', task='>'):

    a = np.random.choice(3)
    if a==0:
        task = '='
    elif a==1:
        task = '!='
    else:
        task = '>'

    n_samples = 4
    
    max_size = 0.5
    min_size = max_size/4

    # odd_condition = np.random.randint(2)
    odd_condition = 0

    if 's' in condition:
        size = np.random.rand(n_samples, 2) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = np.ones([n_samples, 2]) * size

    if task in ['=', '!=']:

        co1 = sample_over_range_t(n_samples, np.array([0,1]), size)
        co2 = np.random.rand(n_samples) * (1-size.max(1)) + size.max(1)/2
        co2 = co2[:,None] * np.ones([n_samples, 2])
        if odd_condition == 0:
            if task == '=':
                # co2[-1] = np.random.rand(2) * (1-size[-1]) + size[-1]/2
                co2[-1] = sample_over_range_t(1, np.array([0,1]), size[-1:])
            else:
                co2[:-1] = np.random.rand(n_samples-1, 2) * (1-size[:-1]) + size[:-1]/2

        xy = [co1, co2] if np.random.randint(2)==0 else [co2, co1]
        
        xy = np.stack(xy, axis=2)

    elif task in ['>', '<']:
        co1 = sample_over_range_t(n_samples, np.array([0,1]), size)
        co2 = np.random.rand(n_samples, 2) * (1-size) + size/2
        
        if odd_condition == 0:
            co1[-1] = co1[-1, ::-1]
        xy = [co1, co2] if np.random.randint(2)==0 else [co2, co1]
        
        xy = np.stack(xy, axis=2)

    s1 = Shape()
    s2 = Shape()
    shapes = []
    for i in range(n_samples):
        shapes += [[s1.clone(), s2.clone()]]

    # if odd_condition == 1:
        # shapes[-1][1] = Shape()

    if 'c' in condition:
        c = sample_random_colors(n_samples*2)
        color = c.reshape([n_samples, 2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([2, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, size, shapes, color

def task_pos_shape_2(condition='cs'):

    n_samples = 4

    n_object_pairs = np.random.randint(2, 5) # always <5

    n_objects = n_object_pairs*2
    
    reg_pairs = np.arange(n_objects).reshape([n_object_pairs, 2])
    odd_pairs = np.copy(reg_pairs)
    
    # exchange 1 with 3
    odd_pairs[0,1] = 3
    odd_pairs[1,1] = 1

    shapes = []
    for i in range(n_objects):
        shapes.append(Shape())
    
    # max_size = 0.9 / np.sqrt(n_objects*4)
    max_size = 0.9 / 4
    min_size = max_size/2

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size - min_size) + min_size 
    else:
        size = np.random.rand() * (max_size - min_size) + min_size 
        size = size * np.ones([n_samples, n_objects])

    # xy = np.stack([xy-size/2, xy+size/2], 1)
    # links = np.array([[0,1,0,-1], [1,0,-1,0], [1,1,-1,-1], [1,-1,-1,1]])
    links = np.array([[np.cos(a),np.sin(a),np.cos(a+np.pi),np.sin(a+np.pi)] for a in np.random.rand(4)*2*np.pi])
    # links = np.random.choice(links, size=n_object_pairs)
    links = links.reshape(4,2,2)

    if np.random.randint(2)==0:
        links[:,:,0] = np.sign(links[:,:,0])
    else:
        links[:,:,1] = np.sign(links[:,:,1])


    links_ = links[np.random.randint(4,size=n_object_pairs)]

    pos = np.array([[0,0], [0,1], [1,0], [1,1]])
    # xy = [pos[np.random.permutation(4)[:n_object_pairs]] * 0.5 + 0.25 for i in range(n_samples)]
    xy = pos[np.random.permutation(4)[:n_object_pairs]] * 0.5 + 0.25
    
    xy = np.stack([xy[None,:,:] + links_[:,0,:]*size.reshape([n_samples,n_object_pairs,2,1])[:,:,0]/2, xy[None,:,:] + links_[:,1,:]*size.reshape([n_samples,n_object_pairs,2,1])[:,:,1]/2], 2)

    xy = xy.reshape(n_samples, n_objects,2)
    # xy = xy[None,:,:] * np.ones(n_samples)[:,None,None]

    all_shapes = []
    shapes_reg = []
    for i in range(n_samples-1):
        perm = np.random.permutation(n_object_pairs)
        shape_idx = reg_pairs[perm].flatten()
        shapes_reg = [shapes[idx].clone() for idx in shape_idx]
        all_shapes.append(shapes_reg)

    shape_idx = odd_pairs[perm].flatten()
    shapes_odd = [shapes[idx] for idx in shape_idx]
    all_shapes.append(shapes_odd)

    size = np.ones([n_samples, n_objects]) * size

    if 'c' in condition:
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, all_shapes, color

#################################################################################
# position + rotation

def task_pos_rot_3(condition='csid'):

    axis = 'y' if np.random.choice(2)==0 else 'x'

    n_samples = 4
    
    n_objects = 3
    
    max_size = 0.9/np.sqrt(n_objects*2)
    min_size = max_size/2

    min_diff = 1/6

    angles0 = (np.random.rand(n_samples) * 1/2 - 1/4) * np.pi
    angles = (np.random.rand(n_samples) * (1 - min_diff*2) + min_diff) * np.pi
    # angles = (np.random.rand(n_samples, 3)*(2*np.pi - min_diff*2)+min_diff)

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])
    
    non_valid = (size).sum(1)>0.9/np.sqrt(2)
    if non_valid.any():
        size[non_valid] = size[non_valid] / size[non_valid].sum(1)[:,None] / np.sqrt(2) * 0.9


    shape = []
    if 'id' not in condition:
        s = Shape()

    for i in range(n_samples):
        if 'id' in condition:
            s = Shape()

        s1 = s.clone()
        s2 = s.clone()
        s3 = s.clone()

        s1.rotate(angles0[i] - angles[i])
        s2.rotate(angles0[i])
        s3.rotate(angles0[i] + angles[i])
        
        if i == n_samples-1:
            l = np.random.randint(3)
            if l == 0:
                shape.append([s3, s2, s1])
            elif l == 1:
                shape.append([s2, s1, s3])
            else:
                shape.append([s1, s3, s2])
        
        else:
            shape.append([s1, s2, s3])


    co1 = sample_over_range_t(n_samples, np.array([0,1]), size)
    co2 = np.random.rand(n_samples, n_objects) * (1-size) + size/2
    # xy = [co1, co2] if np.random.randint(2)==0 else [co2, co1]
    if axis=='y':
        xy = [co1, co2]
    else:
        xy = [co2, co1]
    
    xy = np.stack(xy, axis=2)

    if 'c' in condition:
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

def task_pos_rot_1(condition='xycsid'):
    
    n_samples = 4
    
    n_objects = np.random.randint(3,6)
    
    max_size_sc = 0.7
    min_size_sc = max_size_sc/2

    # size_sc = np.random.rand() * (max_size_sc - min_size_sc) + min_size_sc

    if 's' in condition:
        size_sc = np.random.rand(n_samples) * (max_size_sc - min_size_sc) + min_size_sc
    else:
        size_sc = np.random.rand() * (max_size_sc - min_size_sc) + min_size_sc
        size_sc = size_sc * np.ones([n_samples])
    
    size_sc_min = size_sc.min()

    max_size = size_sc_min/(n_objects)*1.3
    min_size = max_size/2

    angles = np.random.rand(n_objects) * 2 * np.pi
    angles_odd = np.random.rand(n_objects) * 2 * np.pi

    angles = np.ones([n_samples, n_objects]) * angles[None,:]
    angles[-1] = angles_odd

    # size = np.random.rand(n_objects) * (max_size - min_size) + min_size
    size = np.random.rand() * (max_size - min_size) + min_size

    triu_idx = np.triu_indices(n_objects, k=1)
    triu_idx = triu_idx[0]*n_objects + triu_idx[1]

    n_samples_over = 100
    spatial_config = np.random.rand(n_samples_over, n_objects, 2) * (size_sc_min - size)

    valid = (np.abs(spatial_config[:,:,None,:] - spatial_config[:,None,:,:]) - size > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
    while valid.sum() < 2:
        spatial_config = np.random.rand(n_samples_over, n_objects, 2) * (size_sc_min - size)
        valid = (np.abs(spatial_config[:,:,None,:] - spatial_config[:,None,:,:]) - size > 0).any(3).reshape([n_samples_over, n_objects**2])[:, triu_idx].all(1)

    spatial_config = spatial_config[valid]
    spatial_config = spatial_config - spatial_config.mean(1)[:,None,:]

    sc = spatial_config[0]
    sc = np.stack([sc]*n_samples, 0)

    sc = sc / (size_sc_min - size) * (size_sc[:,None,None] - size)

    # sc = spatial_config[0]
    
    if 'xy' in condition:
        xy_sc = np.random.rand(n_samples,2) * (1 - size_sc[:,None]) + size_sc[:,None]/2
    else:
        xy_sc = np.ones([n_samples, 2]) * np.random.rand(1,2) * (1 - size_sc.max()) + size_sc.max()/2

    # xy = xy_sc[:,None,:] + sc[None,:,:]
    xy = xy_sc[:,None,:] + sc

    size = size * np.ones([n_samples, n_objects])
    
    if 'c' in condition:
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # c = sample_random_colors(n_samples)
        # color = c[:,None,:] * np.ones([n_samples, n_objects, 3])
    else:
        c = sample_random_colors(1)
        color = c[None, :,:] * np.ones([n_samples,n_objects, 3])

    if 'id' in condition:
        shapes = [[s.clone() for j in range(n_objects)] for s in [Shape() for i in range(n_samples)]]
    else:
        shape = Shape()
        shapes = [[shape.clone() for j in range(n_objects)] for i in range(n_samples)]

    for i in range(n_samples):        
        for j in range(n_objects):
            shapes[i][j].rotate(angles[i,j])

    return xy, size, shapes, color

def task_pos_rot_2(condition='xyc'):
    n_samples = 4
    
    n_objects = np.random.randint(4,6)
    
    max_size_sc = 0.9/2 # n_forms
    min_size_sc = max_size_sc/2
    
    if 's' in condition:
        size_sc = np.random.rand(n_samples, 2) * (max_size_sc - min_size_sc) + min_size_sc
    else:
        size_sc = np.random.rand() * (max_size_sc - min_size_sc) + min_size_sc
        size_sc = size_sc * np.ones([n_samples, 2])
    
    size_sc_min = size_sc.min()

    max_size = size_sc_min/(n_objects)*1.1
    min_size = max_size/2
    
    # size = np.random.rand(n_objects) * (max_size - min_size) + min_size
    size = np.random.rand() * (max_size - min_size) + min_size
    
    angles = np.random.choice([-np.pi/2, np.pi/2, np.pi], size=n_samples)

    triu_idx = np.triu_indices(n_objects, k=1)
    triu_idx = triu_idx[0]*n_objects + triu_idx[1]

    n_samples_over = 100

    n_unique_sc = n_samples + 1
    n_unique_sc_left = n_unique_sc
    unique_sc = []

    while n_unique_sc_left>0:
        spatial_config = np.random.rand(n_samples_over, n_objects, 2) * (size_sc_min - size)
        valid = (np.abs(spatial_config[:,:,None,:] - spatial_config[:,None,:,:]) - size > 0).any(3).reshape([n_samples_over, n_objects**2])[:, triu_idx].all(1)
        if valid.sum()>0:
            unique_sc.append(spatial_config[valid][:n_unique_sc_left])
            n_unique_sc_left = n_unique_sc_left - valid.sum()

    unique_sc = np.concatenate(unique_sc, axis=0)
    unique_sc = unique_sc - unique_sc.mean(1)[:,None,:]


    sc = unique_sc[:4]
    
    ####
    rot_matrix = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)], 1)
    sc_rot = (np.concatenate([sc, sc],2) * rot_matrix[:,None,:]).reshape([n_samples, n_objects, 2, 2]).sum(-1)

    ####
    sc = np.stack([sc, sc_rot], 1) # samples, sc, objs, xy
    sc[-1, 1] = unique_sc[-1]

    sc = sc / (size_sc_min - size) * (size_sc[:,:,None,None] - size)

    co1 = sample_over_range_t(n_samples, np.array([0,1]), size_sc)
    co2 = np.random.rand(n_samples, 2) * (1-size_sc) + size_sc/2
    xy_sc = [co1, co2] if np.random.randint(2)==0 else [co2, co1]
    xy_sc = np.stack(xy_sc, axis=2)


    xy = xy_sc[:,:,None,:] + sc

    xy = xy.reshape([n_samples, n_objects*2, 2])
    size = size * np.ones([n_samples, n_objects*2])
    
    if 'c' in condition:
        c = sample_random_colors(n_samples*n_objects*2)
        color = c.reshape([n_samples, n_objects*2, 3])
        # c = sample_random_colors(n_samples)
        # color = c[:,None,:] * np.ones([n_samples, n_objects*2, 3])
    else:
        c = sample_random_colors(1)
        color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    if 'id' in condition:        
        shapes = [[Shape() for j in range(n_objects)] for i in range(n_samples)]
        for i in range(n_samples):
            a = angles[i]
            for j in range(n_objects):
                shapes[i].append(shapes[i][j].clone())  
            for j in range(n_objects, n_objects*2):
                shapes[i][j].rotate(a)

    else:
        shape1 = Shape()
        shapes = [[shape1.clone() for j in range(n_objects*2)] for i in range(n_samples)]
        for i in range(n_samples):
            a = angles[i]
            for j in range(n_objects, n_objects*2):
                shapes[i][j].rotate(a)

    return xy, size, shapes, color

#################################################################################
# position + flip

def task_pos_flip_1(condition='scid'):

    n_samples = 4

    n_objects = 3
    
    max_size = 0.9/np.sqrt(n_objects*4)
    min_size = max_size/2

    angles = (np.random.rand(n_samples, n_objects) * 2/3 - 1/3) * np.pi
    # angles = (np.random.rand(n_samples, 3)*(2*np.pi - min_diff*2)+min_diff)

    odd_condition = np.random.randint(4)

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])
    
    flips = np.random.randint(3, size=n_objects-1)

    odd_flips = np.random.randint(3, size=n_objects-1)
    # odd_flips = np.random.randint(2, size=n_objects)
    while (flips==odd_flips).all():
        odd_flips = np.random.randint(3, size=n_objects-1)

    flips = np.stack([flips]*(n_samples-1) + [odd_flips], 0)

    shape = []
    if 'id' not in condition:
        s = Shape()

    for i in range(n_samples):
        if 'id' in condition:
            s = Shape()

        s1 = s.clone()
        s2 = s.clone()
        s3 = s.clone()
        
        if flips[i,0]==1:
            s1.flip()
        elif flips[i,0]==2:
            s1.flip()
            s1.rotate(np.pi)

        if flips[i,1]==1:
            s3.flip()
        elif flips[i,0]==2:
            s3.flip()
            s3.rotate(np.pi)

        # s1.flip()

        # s1.rotate(angles[i,0])
        # s2.rotate(angles[i,1])
        # s3.rotate(angles[i,2])
        
        shape.append([s1, s2, s3])


    co1 = sample_over_range_t(n_samples, np.array([0,1]), size)
    co2 = np.random.rand(n_samples, n_objects) * (1-size) + size/2
    # gen
    xy = [co1, co2] if np.random.randint(2)==0 else [co2, co1]
    # xy = [co1, co2]
    xy = np.stack(xy, axis=2)


    if 'c' in condition:
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

def task_pos_flip_2(condition='csid'):

    n_samples = 4

    n_objects = 2

    # odd conditions: shape1 != shape2, size1 != size2, flip1 == flip2, position doesn't fit, incorrect angle
    # odd_condition = np.random.randint(1, 5) # 5
    odd_condition = np.random.randint(2) # 5

    max_size = 0.7/n_objects
    min_size = max_size/2
    
    if 's' in condition:
        size = np.random.rand(n_samples, 2) * (max_size - min_size) + min_size
    else:
        size = np.random.rand() * np.ones([n_samples, 2])
    
    # if odd_condition == 1:
    #     idx = np.random.randint(2)
    #     size[-1,idx] = size[-1,idx] * (np.random.rand()* (2/3 - 1/3) + 1/3)

    a = (np.random.rand(n_samples) * 1 - 1/2) * np.pi 
    r = np.random.rand(n_samples) * (0.4-size.max(1)) + size.max(1)/2

    a_g = a + np.pi/2
    # a1, a2 = a - np.pi/2, a + np.pi/2

    range_ = 1 - np.sqrt(2) * size.max(1)[:,None] - 2 * np.abs(np.stack([r * np.cos(a), r * np.sin(a)], 1))
    starting_ = np.sqrt(2) * size.max(1)[:,None]/2 + np.abs(np.stack([r * np.cos(a), r * np.sin(a)], 1))
    
    xy0 = np.random.rand(n_samples, n_objects) * range_ + starting_

    xy = xy0[:,None,:] + np.stack([r * np.cos(a_g), r * np.sin(a_g), r * np.cos(a_g + np.pi), r * np.sin(a_g + np.pi)], 1).reshape([n_samples, 2, 2])

    if 'id' not in condition:
        s = Shape()
    shape = []
    for i in range(n_samples):
        if 'id' in condition:
            s = Shape()
        s1 = s.clone()
        s2 = s.clone()
        a_ = a[i]

        if odd_condition == 1 and i == n_samples-1:
            a_ = a_ + np.random.choice([-1,1]) * np.pi/4
            s1.rotate(a_)
        else:
            s1.rotate(a_)

        if not (odd_condition == 0 and i == n_samples-1):
            s2.flip()
        
        s2.rotate(a_)

        shape.append([s1, s2])

    if 'c' in condition:
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# position + color
def task_pos_col_1(condition='xyscidv'):
        
    n_samples = 4
    n_objects = 2
    
    max_size = 0.7/n_objects
    min_size = max_size/4

    if 's' in condition:
        size = np.random.rand(n_samples, 2) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones(n_samples, 2)

    co1 = sample_over_range_t(n_samples, np.array([0,1]), size)
    co2 = np.random.rand(n_samples, 2) * (1-size) + size/2
    xy = [co1, co2] if np.random.randint(2)==0 else [co2, co1]
    xy = np.stack(xy, axis=2)
    
    h1 = np.random.rand()
    min_diff = 0.15
    h2 = (h1 + (np.random.rand() * (1-min_diff*2) + min_diff) * np.random.choice([-1,1]))%1
    h = np.array([h1, h2])[None,:] * np.ones([n_samples, 2])
    h[-1] = h[-1,::-1]
    
    # if 'h' in condition:
    #     h = np.random.rand(n_samples, 2)
    # else:
    #     h = np.random.rand()
    #     h = h * np.ones([n_samples, 2])

    if 's' in condition:
        s = (np.random.rand(n_samples, 2) * 0.6 + 0.4) * np.ones([n_samples, 2])
    else:
        s = np.random.rand() * 0.6 + 0.4
        s = s * np.ones([n_samples, 2])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples, 2) * 0.5 + 0.5)
    else:
        v = np.random.rand() * 0.5 + 0.5
        v = v * np.ones([n_samples, 2])
 
    color = np.stack([h,s,v], 2)

    if 'id' in condition:
        shape = [[Shape(), Shape()] for i in range(n_samples)]
    else:
        s = Shape()
        shape = [[s.clone(), s.clone()] for _ in range(n_samples)]

    if 'r' in condition:
        angle = np.random.rand(n_samples, 2) * 2 * np.pi
        for i in range(n_samples-1):
            shape[i][0].rotate(angle[i,0])
            shape[i][1].rotate(angle[i,1])

    if 'f' in condition:
        for i in range(n_samples-1):
            shape[i][1].flip()

    return xy, size, shape, color

def task_pos_col_2(condition='xychsid'):
    
    n_samples = 4
    
    n_objects = np.random.randint(4,7)
    
    max_size_sc = 0.7
    min_size_sc = max_size_sc/2

    if 's' in condition:
        size_sc = np.random.rand(n_samples) * (max_size_sc - min_size_sc) + min_size_sc
    else:
        size_sc = np.random.rand() * (max_size_sc - min_size_sc) + min_size_sc

    size_sc_min = size_sc.min()

    max_size = size_sc_min/np.sqrt(n_objects*3)
    min_size = max_size*2/3
    
    unique_colors = np.random.randint(2, n_objects)
    color_idx = []
    n_cols = []
    n_cols_left = n_objects
    for i in range(unique_colors-1):
        max_possible = n_cols_left - max(unique_colors-1-i, 0)
        # sample an int = n_cols_left - unique_colors-1-i
        sampled_n = np.random.randint(1, max_possible)

        n_cols.append(sampled_n)
        color_idx += [i]*sampled_n
        n_cols_left = n_cols_left - sampled_n
    
    n_cols.append(n_cols_left)
    color_idx += [unique_colors-1]*n_cols_left

    color_idx = np.array(color_idx)
    color_idx_odd = color_idx[np.random.permutation(len(color_idx))]
    while (color_idx == color_idx_odd).all():
        color_idx_odd = color_idx[np.random.permutation(len(color_idx))]

    if 'h' in condition:
        h_unique = np.arange(unique_colors) / unique_colors
        h = h_unique[color_idx][None,:] * np.ones([n_samples, n_objects]) + np.random.rand(n_samples)[:,None]
        h = np.mod(h, 1)
    else:
        h_unique = np.mod(np.arange(unique_colors) / unique_colors + np.random.rand(), 1)
        h = h_unique[color_idx][None,:] * np.ones([n_samples, n_objects])
    
    h[-1] = h_unique[color_idx_odd]
    s = np.random.rand() * 0.6 + 0.4
    s = s * np.ones([n_samples, n_objects])
    v = np.random.rand() * 0.5 + 0.25
    v = v * np.ones([n_samples, n_objects])
 
    color = np.stack([h,s,v], 2)

    size = np.random.rand() * (max_size - min_size) + min_size
    
    triu_idx = np.triu_indices(n_objects, k=1)
    triu_idx = triu_idx[0]*n_objects + triu_idx[1]

    n_samples_over = 100
    spatial_config = np.random.rand(n_samples_over, n_objects, 2) * (size_sc_min - size)

    valid = (np.abs(spatial_config[:,:,None,:] - spatial_config[:,None,:,:]) - size > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
    while valid.sum() < 2:
        spatial_config = np.random.rand(n_samples_over, n_objects, 2) * (size_sc_min - size)
        valid = (np.abs(spatial_config[:,:,None,:] - spatial_config[:,None,:,:]) - size > 0).any(3).reshape([n_samples_over, n_objects**2])[:, triu_idx].all(1)
    
    spatial_config = spatial_config[valid]
    spatial_config = spatial_config - spatial_config.mean(1)[:,None,:]

    sc = spatial_config[0]
    sc = sc / (size_sc_min - size) *  (size_sc[:,None,None] - size) 
    # sc_odd = spatial_config[1]
    
    if 'xy' in condition:
        xy_sc = np.random.rand(n_samples,2) * (1 - size_sc[:,None]) + size_sc[:,None]/2
    else:
        xy_sc = np.ones([n_samples, 2]) * np.random.rand(1,2) * (1 - size_sc.max()) + size_sc.max()/2

    xy = xy_sc[:,None,:] + sc

    size = size * np.ones([n_samples, n_objects])
    
    if 'id' in condition:
        shape = [[Shape() for _ in range(n_objects)] for _ in range(n_samples)]
    else:
        s = Shape()
        shape = [[s.clone() for _ in range(n_objects)] for _ in range(n_samples)]

    return xy, size, shape, color


#################################################################################
# position + inside # set 1

def task_pos_inside_1(condition='c'):
    
    axis = 'x' if np.random.randint(2)==0 else 'y'
    condition += 'max' if np.random.randint(2)==0 else 'min'
    
    n_samples = 4
    
    max_size = 0.7
    min_size = max_size/2

    size_a = np.random.rand(n_samples) * (max_size - min_size) + min_size 
    size_b = np.random.rand(n_samples) * (size_a/3.5 - size_a/5) + size_a/5

    done = False
    max_attempts = 10 
    
    range_1 = 1 - size_a[:,None]
    starting_1 = size_a[:,None]/2

    xy1 = np.random.rand(n_samples,2) * range_1 + starting_1 

    xy2 = []
    shapes = []
    
    for i in range(n_samples):
        done = False

        s1 = Shape(gap_max=0.07, hole_radius=0.25)
        s2 = Shape()
        for _ in range(max_attempts):

            samples = sample_position_inside_1(s1, s2, size_b[i]/size_a[i])
            if len(samples)>0:
                done = True
            
            if done:
                break
            else:
                s1.randomize()
                s2.randomize()

        if not done:
            return np.zeros([100,100])

        if axis == 'x':
            comp_samples = samples[:,0]
        elif axis == 'y':
            comp_samples = samples[:,1]
        
        if 'min' in condition:
            if i == n_samples-1:
                idx = np.argmax(comp_samples)
            else:
                idx = np.argmin(comp_samples)

        elif 'max' in condition:
            
            if i == n_samples-1:
                idx = np.argmin(comp_samples)
            else:
                idx = np.argmax(comp_samples)
        
        xy2.append(samples[idx])
        shapes.append([s1, s2])

    xy2 = np.array(xy2)*size_a[:,None] + xy1
    
    xy = np.stack([xy1, xy2], axis=1)
    size = np.stack([size_a, size_b], axis=1)

    if 'c' in condition:
        c = sample_random_colors(n_samples*3)
        color = c.reshape([n_samples, 3, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([3, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([3, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, size, shapes, color

def task_pos_inside_2(condition='c'):
    a = np.random.randint(3)
    if a==0:
        axis = 'x' 
    elif a==1:
        axis = 'y' 
    else:
        axis = 'xy' 
    
    n_samples = 4
    
    max_size = 0.7
    min_size = max_size/2

    size_a = np.random.rand(n_samples) * (max_size - min_size) + min_size 
    size_b = np.random.rand(n_samples) * (size_a/4 - size_a/5) + size_a/5
    size_c = np.random.rand(n_samples) * (size_a/4 - size_a/5) + size_a/5

    done = False
    max_attempts = 10 

    range_1 = 1 - size_a[:,None]
    starting_1 = size_a[:,None]/2

    xy1 = np.random.rand(n_samples,2) * range_1 + starting_1 

    xy2 = []
    shapes = []

    # positive: 
    # a(b,c) and x_b > x_c

    # negative:
    # 1- a(b),c or a(c),b or a,b,c
    # or
    # 2- x_b < x_c <- focus on this

    cond = np.random.randint(2)==0

    for i in range(n_samples):
        done = False

        s1 = Shape(gap_max=0.05, hole_radius=0.25)
        s2 = Shape()
        s3 = Shape()

        for _ in range(max_attempts):

            samples = sample_position_inside_many(s1, [s2,s3], [size_b[i]/size_a[i], size_c[i]/size_a[i]])
            if len(samples)>0:
                
                # diagonal condition
                if len(axis)>1:
                
                    comp_samples1 = np.abs((samples[:,0,0]+samples[:,0,1]) - (samples[:,1,0]+samples[:,1,1]))
                    comp_samples2 = np.abs((-samples[:,0,0]+samples[:,0,1]) - (-samples[:,1,0]+samples[:,1,1]))
                    if cond == 0:
                        if i == n_samples-1:
                            comp_samples = comp_samples1
                        else:
                            comp_samples = comp_samples2
                    else:
                        if i == n_samples-1:
                            comp_samples = comp_samples2
                        else:
                            comp_samples = comp_samples1
                    if (comp_samples>0).sum():
                        done = True

                else:
                    comp_samples = np.abs(samples[:,0,0] - samples[:,1,0]) - np.abs(samples[:,0,1] - samples[:,1,1])
                    if axis == 'y':
                        comp_samples = -comp_samples # np.logical_not(comp_samples)        
                    if i == n_samples-1:
                        comp_samples = -comp_samples # np.logical_not(comp_samples)

                    if (comp_samples>0).sum():
                        done = True

                if done:
                    sample = samples[np.argmax(comp_samples)]


            if done:
                break
            else:
                s1.randomize()
                s2.randomize()
                s3.randomize()

        if not done:
            return np.zeros([100,100])

        xy2.append(sample)
        shapes.append([s1, s2, s3])


    xy2 = np.stack(xy2, 0)*size_a[:,None,None] + xy1[:,None,:]
    
    xy = np.concatenate([xy1[:,None], xy2], axis=1)
    size = np.stack([size_a, size_b, size_c], axis=1)

    if 'c' in condition:
        c = sample_random_colors(n_samples*3)
        color = c.reshape([n_samples, 3, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([3, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([3, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, size, shapes, color

def task_pos_inside_3(condition='c'):
    a = np.random.randint(2)==0
    if a==0:
        axis = 'x' 
    else:
        axis = 'y' 
    
    a = np.random.randint(2)==0
    if a==0:
        condition += 'min' 
    else:
        condition += 'max' 
    

    max_attempts = 10
    
    n_samples = 4
    n_samples_over = 100
    
    n_objects_samples = np.random.randint(low=2, high=6, size=n_samples)
    # n_objects = n_objects_samples.max()
    # min_size_obj = 0.2
    min_btw_size = 0.3

    
    all_xy = []
    all_size = []
    all_shape = []
    
    n_objects_all = n_objects_samples.copy()

    for i in range(n_samples):
        # xy_i = xy[i,:n_objects_samples[i]]
        xy_ = []
        size_ = []
        shape_ = []

        done = False
        n_objects = n_objects_samples[i]

        min_size_obj = min(0.9/np.sqrt(n_objects*3), 0.3)
        min_btw_size = min_size_obj
        
        for _ in range(max_attempts):
            xy = np.random.rand(n_samples_over, n_objects, 2) * (1-min_size_obj) + min_size_obj/2 
            xy = xy.reshape([n_samples_over*n_objects, 2])

            change_xy = xy[np.logical_and(xy[:,0]>0.45, xy[:,0]<0.55), 0]
            change_xy = change_xy + np.random.choice([-1,1], size=change_xy.shape[0])*0.2
            xy[np.logical_and(xy[:,0]>0.45, xy[:,0]<0.55), 0] = change_xy

            xy = xy.reshape([n_samples_over, n_objects, 2])

            triu_idx = np.triu_indices(n_objects, k=1)[0]*n_objects + np.triu_indices(n_objects, k=1)[1]
            no_overlap = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - min_btw_size>0).any(3).reshape([-1, n_objects*n_objects])[:,triu_idx].all(1)
            if no_overlap.sum()>0:
                done = True
                break
            
        if not done:
            print('')

        xy = xy[no_overlap][0]

        non_diag = np.where(~np.eye(n_objects,dtype=bool))
        non_diag = non_diag[0]*n_objects + non_diag[1]
        dists_obj = np.abs(xy[:,None,:] - xy[None,:,:]).max(2).reshape([n_objects**2])[non_diag].reshape([n_objects, n_objects-1]).min(1)
        dists_edge = np.stack([xy, 1-xy],2).min(1).min(1)*2
        max_size = np.stack([dists_edge, dists_obj], 1).min(1)

        # max_size = np.stack([xy, 1-xy],2).min(2).min(2)
        min_size = max_size/2

        size = np.random.rand(n_objects) * (max_size - min_size) + min_size 
        size_in = np.random.rand(n_objects) * (size/2.5 - size/4) + size/4

        object_in = xy[:,0]<0.5

        # if axis == 'x':
        #     object_in = xy[:,0]<0.5
        # elif axis == 'y':
        #     object_in = xy[:,1]<0.5

        # if condition == 'min':
        if condition == 'max':
            object_in = np.logical_not(object_in)

        # odd one out
        if i == n_samples -1:
            object_in = np.logical_not(object_in)
        
        n_objects_all[i] = n_objects_all[i] + object_in.sum()

        for j in range(n_objects_samples[i]):
            if object_in[j]:
                    
                done = False                
                s1 = Shape(gap_max=0.08, hole_radius=0.2)
                s2 = Shape()
                for _ in range(max_attempts):

                    samples = sample_position_inside_1(s1, s2, size_in[j]/size[j])
                    if len(samples)>0:
                        done = True
                    
                    if done:
                        break
                    else:
                        s1.randomize()
                        s2.randomize()
                
                if done:
                    xy_in = samples[0]

                    xy_.append(xy[j])
                    shape_.append(s1)
                    size_.append(size[j])
                    
                    xy_.append(xy_in * size[j] + xy[j])
                    shape_.append(s2)
                    size_.append(size_in[j])
            
            else:
                if np.random.randint(2) == 0:
                    s1 = Shape(gap_max=0.01)
                    size_.append(size[j])
                else:
                    s1 = Shape(gap_max=0.08, hole_radius=0.2)
                    size_.append(size_in[j])
                
                xy_.append(xy[j])
                shape_.append(s1)

        all_xy.append(xy_)
        all_size.append(size_)
        all_shape.append(shape_)


    if 'c' in condition:
        color = [sample_random_colors(n_objects_all[i]) for i in range(n_samples)]
        # c = sample_random_colors(n_samples*n_objects_all[i])
        # color = c.reshape([n_samples, n_objects_all[i], 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects_all[i], 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects_all[i], 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])


    xy, size, shape = all_xy, all_size, all_shape

    if axis=='x':
        for i in range(n_samples):
            xy[i], shape[i] = flip_diag_scene(xy[i], shape[i])

    return xy, size, shape, color

def task_pos_inside_4(condition='s_id_c'):
    
    variable='x' if np.random.randint(2)==0 else 'y'

    n_samples = 4
    
    max_size = 0.3
    min_size = max_size/2

    if 's' in condition:
        size = np.random.rand(n_samples, 2) * (max_size-min_size) + min_size
        size_in = np.random.rand(n_samples, 2) * (size/3 - size/4) + size/4
        size = np.concatenate([size, size_in[:, ::-1]], 1)
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = np.ones([n_samples, 2]) * size
        size_in = np.random.rand(n_samples, 2) * (size/3 - size/4) + size/4
        size = np.concatenate([size, size_in[:, ::-1]], 1)

    # x_dists
    shuffle_ = np.random.randint(2, size=n_samples)
    # idx = np.arange(3)[None,:] * np.ones([n_samples, 3])
    size_shuffle = size.copy()
    
    for i in range(n_samples):
        if shuffle_[i] == 1:
            size_shuffle[i] = size_shuffle[i][np.array([0,2,1,3])]

    x = sample_over_range_t(n_samples, np.array([0,1]), size_shuffle[:,:3])
    
    for i in range(n_samples):
        if shuffle_[i] == 1:
            x[i] = x[i][np.array([0,2,1])]

    # odd one out
    x[-1,:] = 1 - x[-1,:]
    if np.random.randint(2)==1:
        x = 1-x

    y = np.random.rand(n_samples, 3) * (1-size[:,:3]) + size[:,:3]/2
        
    xy123 = np.stack([x,y], axis=2)

    max_attempts = 10
    shapes = []
    xy4 = []
    for i in range(n_samples):
    
        done = False

        s1 = Shape(gap_max=0.07, hole_radius=0.2)
        s2 = Shape(gap_max=0.07, hole_radius=0.2)
        s3 = Shape()
        s4 = Shape()
        
        for _ in range(max_attempts):
    
            samples = sample_position_inside_1(s1, s4, size[i,3]/size[i,0])
            if len(samples)>0:
                done = True
            
            if done:
                break
            else:
                s1.randomize()
                s4.randomize()

        if done:
            xy4_ = np.array(samples[0]) * size[i,0] + xy123[i,0]
        else:
            xy4_ = xy123[i,0]
              
        xy4.append(xy4_)

        shapes.append([s1, s2, s3, s4])

    xy = np.concatenate([xy123,np.array(xy4)[:,None]], axis=1) 


    if 'c' in condition:
        c = sample_random_colors(n_samples*4)
        color = c.reshape([n_samples, 4, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([4, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([4, 3]) * color for i in range(n_samples)]

    shape = shapes

    if variable=='x':
        for i in range(n_samples):
            xy[i], shape[i] = flip_diag_scene(xy[i], shape[i])

    return xy, size, shape, color

#################################################################################
# position + contact

def task_pos_contact(condition='c'):
        
    n_samples = 4
    n_objects = 2
    
    max_size = 0.3
    min_size = max_size/2

    size = np.random.rand(n_samples, 2) * (max_size - min_size) + min_size

    angle1 = np.random.rand() * 2 * np.pi
    angle2 = angle1 + np.pi / 2 * np.random.choice([-1, 1])
    angle = np.ones(n_samples) * angle1
    angle[-1] = angle2
    
    shape = []
    xy = []
    for i in range(n_samples):
            
        s1 = Shape()
        s2 = Shape()

        positions, clump_size = sample_contact_many([s1, s2], size[i], angle[i])
        
        xy0 = np.random.rand(2) * (1-clump_size) + clump_size/2
        xy_ = positions + xy0[None,:]

        xy.append(xy_)

        shape.append([s1,s2])

    xy = np.stack(xy, 0)
    
    if 'c' in condition:
        c = sample_random_colors(n_samples*2)
        color = c.reshape([n_samples, 2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([2, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# position + count

def task_pos_count_1(condition='cid'):
    
    variable = 'x' if np.random.randint(2)==0 else 'y'
    n_samples = 4
    
    n_objects = [np.random.choice(np.arange(6,11), size=2, replace=False) for _ in range(n_samples)]
    n_objects = np.sort(n_objects, axis=1)
    # n_objs_odd = n_objects[-1]
    # n_objects[-1] = np.array([n_objs_odd[1], n_objs_odd[0]])

    n_objects = n_objects.flatten()

    max_size = 0.9/np.sqrt(n_objects*5)
    max_size[::2] = max_size[1::2]
    min_size = max_size/2

    size = np.random.rand(len(n_objects), max(n_objects)) * (max_size[:,None] - min_size[:,None]) + min_size[:,None]
    # size = np.random.rand(len(n_objects)) * (max_size - min_size) + min_size
    # size = size[:, None]* np.ones([len(n_objects), max(n_objects)]) 

    n_samples_over = 100

    xy = []
    for i in range(n_samples*2):
        triu_idx = np.triu_indices(n_objects[i], k=1)
        triu_idx = triu_idx[0]*n_objects[i] + triu_idx[1]

        xy_ = np.random.rand(n_samples_over, n_objects[i], 2) * (np.array([0.4,1])[None,None,:] - size[i,:n_objects[i]][None,:,None]) + size[i,:n_objects[i]][None,:,None]/2
        valid = (np.abs(xy_[:,:,None,:] - xy_[:,None,:,:]) - (size[i,:n_objects[i]][None,:,None,None]+size[i,:n_objects[i]][None,None,:,None])/2 > 0).any(3).reshape([n_samples_over, n_objects[i]**2])[:, triu_idx].all(1)
        while not valid.any():
            xy_ = np.random.rand(n_samples_over, n_objects[i], 2) * (np.array([0.4,1])[None,None,:] - size[i,:n_objects[i]][None,:,None]) + size[i,:n_objects[i]][None,:,None]/2
            valid = (np.abs(xy_[:,:,None,:] - xy_[:,None,:,:]) - (size[i,:n_objects[i]][None,:,None,None]+size[i,:n_objects[i]][None,None,:,None])/2 > 0).any(3).reshape([n_samples_over, n_objects[i]**2])[:, triu_idx].all(1)
        xy_ = xy_[valid][0]

        xy.append(xy_)
    
    if 'c' in condition:
        color = [sample_random_colors(n_objects[i*2] + n_objects[i*2 + 1]) for i in range(n_samples)]
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects[i*2] + n_objects[i*2 + 1], 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects[i*2] + n_objects[i*2 + 1], 3]) * color for i in range(n_samples)]

    shape = Shape()
    xyr = []
    sizer = []
    shapes = []
    for i in range(n_samples):
        if i == n_samples-1:
            left = xy[i*2+1]
            right = xy[i*2]        
        else:
            left = xy[i*2]
            right = xy[i*2+1]
        
        right[:,0] = right[:,0] + 0.6
        xy_ = np.concatenate([left, right], 0)
        
        size_ = np.concatenate([size[i*2,:n_objects[i*2]], size[i*2+1,:n_objects[i*2+1]]])
        shapes_ = []
        for j in range(n_objects[i*2] + n_objects[i*2 + 1]):
            if 'id' in condition:
                shape.randomize()
            shapes_.append(shape.clone())
        
        xyr.append(xy_)
        sizer.append(size_)
        shapes.append(shapes_)

    xy = xyr
    size = sizer
    shape = shapes
    if variable=='x':
        for i in range(n_samples):
            xy[i], shape[i] = flip_diag_scene(xy[i], shape[i])

    return xy, size, shape, color

def task_pos_count_2(condition='cid'):
    
    n_samples = 4

    n_objects_aligned_v = np.random.choice(np.arange(5,9), size=2, replace=False)

    n_objects_aligned = np.ones(n_samples) * n_objects_aligned_v[0]
    n_objects_aligned[-1] = n_objects_aligned_v[1]

    n_objects_other = np.random.choice(np.arange(4,6), size=n_samples)
    n_objects = n_objects_aligned + n_objects_other
    
    n_objects_aligned = n_objects_aligned.astype(int)
    n_objects_other = n_objects_other.astype(int)
    n_objects = n_objects.astype(int)

    max_size = 0.9/np.sqrt(n_objects*4)
    min_size = max_size/2

    # size = np.random.rand(n_samples, max(n_objects)) * (max_size[:,None] - min_size[:,None]) + min_size[:,None]
    size = np.random.rand(n_samples) * (max_size - min_size) + min_size
    size = size[:, None]* np.ones([n_samples, max(n_objects)]) 

    n_samples_over = 100

    xy = []
    for i in range(n_samples):
                
        dists = np.random.rand(n_objects_aligned[i])
        dists = dists / dists.sum() * (1-size[i,:n_objects_aligned[i]].sum())
        dists[0] = dists[0] * np.random.rand()
        dists = dists + size[i,:n_objects_aligned[i]]
        xy_al = np.cumsum(dists)
        xy_al = xy_al - size[i,:n_objects_aligned[i]]/2
        
        # apply segment
        seg = np.random.rand(2)
        seg = seg/seg.max()
        idx = np.argmin(seg)
        
        xy_al = xy_al[:,None] * seg[None,:]
        xy_al[:,idx] = xy_al[:,idx] + np.random.rand() * (1 - seg[idx] - size[i,0]/2 - size[i,n_objects_aligned[i]-1]/2) + size[i,0]/2 

        if np.random.rand()>0.5:
            xy_al[:,idx] = 1 - xy_al[:,idx]

        triu_idx = np.triu_indices(n_objects_other[i], k=1)
        triu_idx = triu_idx[0]*n_objects_other[i] + triu_idx[1]

        xy_ot = np.random.rand(n_samples_over, n_objects_other[i], 2) * (1-size[i:i+1,-n_objects_other[i]:,None]) + size[i:i+1,-n_objects_other[i]:,None]/2

        valid_al = (np.abs(xy_ot[:,:,None,:] - xy_al[None,None,:,:]) - (size[i,-n_objects_other[i]:][None,:,None,None]+size[i,:n_objects_aligned[i]][None,None,:,None])/2 > 0).any(3).all(2).all(1)
        valid_ot = (np.abs(xy_ot[:,:,None,:] - xy_ot[:,None,:,:]) - (size[i,-n_objects_other[i]:][None,:,None,None]+size[i, -n_objects_other[i]:][None,None,:,None])/2 > 0).any(3).reshape([n_samples_over, n_objects_other[i]**2])[:, triu_idx].all(1)
        valid = np.logical_and(valid_al, valid_ot)
        while not valid.any():
            xy_ot = np.random.rand(n_samples_over, n_objects_other[i], 2) * (1-size[i:i+1,-n_objects_other[i]:,None]) + size[i:i+1,-n_objects_other[i]:,None]/2
            valid_al = (np.abs(xy_ot[:,:,None,:] - xy_al[None,None,:,:]) - (size[i,-n_objects_other[i]:][None,:,None,None]+size[i,:n_objects_aligned[i]][None,None,:,None])/2 > 0).any(3).all(2).all(1)
            valid_ot = (np.abs(xy_ot[:,:,None,:] - xy_ot[:,None,:,:]) - (size[i,-n_objects_other[i]:][None,:,None,None]+size[i, -n_objects_other[i]:][None,None,:,None])/2 > 0).any(3).reshape([n_samples_over, n_objects_other[i]**2])[:, triu_idx].all(1)
            valid = np.logical_and(valid_al, valid_ot)
        
        xy_ot = xy_ot[valid][0]

        xy.append(np.concatenate([xy_al, xy_ot],0))
    
    if 'c' in condition:
        color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        # c = sample_random_colors(n_objects.sum())
        # color = c.reshape([n_samples, 2, 3])
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    if 'id' in condition:
        shapes = [[Shape() for _ in range(n_objects[i])] for i in range(n_samples)]
    else:
        s = Shape()
        shapes = [[s.clone() for _ in range(n_objects[i])] for i in range(n_samples)]

    shape = shapes
    size = [size[i,:n_objects[i]] for i in range(n_samples)]

    return xy, size, shape, color

def task_pos_count_3(condition='cid'):
    
    variable = 'x' if np.random.randint(2)==0 else 'y'

    n_samples = 4
    
    n_lines_v = np.random.choice(np.arange(6,11), size=2, replace=False)

    n_lines = np.ones(n_samples) * n_lines_v[0]
    n_lines[-1] = n_lines_v[1]
    n_lines = n_lines.astype(int)
    
    n_lines_objects = np.random.randint(2,5, size=[n_samples, n_lines_v.max()])
    n_lines_objects = n_lines_objects.astype(int)

    n_objects = [n_lines_objects[i,:n_lines[i]].sum().astype(int) for i in range(n_samples)] 

    max_size = 0.6/np.stack([n_lines, n_lines_objects.max(1)], 1).max(1)
    min_size = max_size/2

    # size = np.random.rand(n_samples, max(n_objects)) * (max_size[:,None] - min_size[:,None]) + min_size[:,None]
    size = np.random.rand(n_samples) * (max_size - min_size) + min_size
    size = size[:, None]* np.ones([n_samples, max(n_objects)]) 

    xy = []
    
    for i in range(n_samples):
        dists = np.random.rand(n_lines[i])
        dists = dists / dists.sum() * (1 - size[i,0]*n_lines[i])
        dists[0] = dists[0] * np.random.rand()
        dists = dists + size[i,0]
        x_lines = np.cumsum(dists)
        x_lines = x_lines - size[i,0]/2
        
        y_lines_ = []
        x_lines_ = []
        
        for l in range(n_lines[i]):
            dists = np.random.rand(n_lines_objects[i,l])
            dists = dists / dists.sum() * (1 - size[i,0]*n_lines_objects[i,l])
            dists[0] = dists[0] * np.random.rand()
            dists = dists + size[i,0]
            y_lines = np.cumsum(dists)
            y_lines = y_lines - size[i,0]/2
            y_lines_.append(y_lines)
            x_lines_.append([x_lines[l]]*n_lines_objects[i,l])
            
        x_lines = np.concatenate(x_lines_)
        y_lines = np.concatenate(y_lines_)
        
        # # apply segment
        # seg = np.random.rand(2)
        # seg = seg/seg.max()
        # idx = np.argmin(seg)
        
        # xy_al = xy_al[:,None] * seg[None,:]
        # xy_al[:,idx] = xy_al[:,idx] + np.random.rand() * (1 - seg[idx] - size[i,0]/2 - size[i,n_objects_aligned[i]-1]/2) + size[i,0]/2 

        # if np.random.rand()>0.5:
        #     xy_al[:,idx] = 1 - xy_al[:,idx]


        xy.append(np.stack([x_lines, y_lines],1))
    
    if 'c' in condition:
        color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        # c = sample_random_colors(n_objects.sum())
        # color = c.reshape([n_samples, 2, 3])
        # color = [np.ones([n_objects[i], 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    if 'id' in condition:
        shapes = [[Shape() for _ in range(n_objects[i])] for i in range(n_samples)]
    else:
        s = Shape()
        shapes = [[s.clone() for _ in range(n_objects[i])] for i in range(n_samples)]

    shape = shapes
    size = [size[i,:n_objects[i]] for i in range(n_samples)]

    if variable=='x':
        for i in range(n_samples):
            xy[i], shape[i] = flip_diag_scene(xy[i], shape[i])

    return xy, size, shape, color
        
def task_pos_count_4(condition='cid'):
    
    variable = 'x' if np.random.randint(2)==0 else 'y'

    n_samples = 4
    
    n_objects_1 = np.random.randint(3, 7, size=n_samples)
    n_objects_2 = np.random.randint(3, 7, size=n_samples)
    n_objects_3 = n_objects_1 + n_objects_2

    n_objects_3[-1] = max(n_objects_3[-1] + np.random.choice([-1, 1])*np.random.randint(1,3),1)
    

    n_objects = n_objects_1 + n_objects_2 + n_objects_3
    
    max_n_objs = n_objects.max()
    
    # image and object parameters    
    max_size = 0.95/(n_objects_3)
    min_size = max_size/2

    s_idx = np.zeros([n_samples, max_n_objs])
    
    size = np.random.rand(n_samples) * (max_size - min_size) + min_size

    # perms = [[0, 1, 2], [2, 0, 1], [0, 2, 1]]
    a = np.random.randint(3) 
    # perm = perms[a]
    # n_objs = [p]

    if a==0:
        n_objs = [n_objects_1, n_objects_2, n_objects_3]
    elif a==1:
        n_objs = [n_objects_3, n_objects_1, n_objects_2]
    elif a==2:
        n_objs = [n_objects_1, n_objects_3, n_objects_2]

    x_idx = [[0]*n_objs[0][i] + [1]*n_objs[1][i] + [2]*n_objs[2][i] for i in range(n_samples)]

    triu_idx = np.triu_indices(3,k=1)
    triu_idx = triu_idx[0]*3 + triu_idx[1]

    dists_x = np.random.rand(n_samples, 3)
    dists_x = dists_x/dists_x.sum(1)[:,None] * (1 - size[:,None]*3)
    dists_x[:, 0] = dists_x[:,0] * np.random.rand(n_samples)
    dists_x = dists_x + size[:,None]
    dists_x[:,0] = dists_x[:,0] - size/2
    x = np.cumsum(dists_x, axis=1)
    
    xs = [x[i][x_idx[i]].tolist() for i in range(n_samples)]

    ys = []
    shape = []
    size_ = []
    for i in range(n_samples):
        ys_ = []
        for n_o in [n_objs[0], n_objs[1], n_objs[2]]:
            dists_y = np.random.rand(n_o[i])# * (1 - size[i]*n_o[i])
            dists_y = dists_y/dists_y.sum() * (1 - size[i]*n_o[i])
            dists_y[0] = dists_y[0]/2
            dists_y = dists_y + size[i]
            dists_y[0] = dists_y[0] - size[i]/2
            y = np.cumsum(dists_y)
            ys_ = ys_ + y.tolist()
        ys.append(ys_)

        # shape.append([unique_shapes[idx].clone() for idx in id_idx[i]])
        size_.append([size[i]]*(n_objects[i]))
    
    if 'id' in condition:
        shape = [[Shape() for _ in range(n_objects[i])] for i in range(n_samples)]
    else:
        s = Shape()
        shape = [[s.clone() for _ in range(n_objects[i])] for i in range(n_samples)]
        
    size = size_
    
    xy = [np.stack([np.array(xs[i]), np.array(ys[i])], axis=1) for i in range(n_samples)]
    size = [np.array(size[i]) for i in range(n_samples)]

    if 'c' in condition:
        color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        # c = sample_random_colors(n_objects.sum())
        # color = c.reshape([n_samples, 2, 3])
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    if variable=='x':
        for i in range(n_samples):
            xy[i], shape[i] = flip_diag_scene(xy[i], shape[i])

    return xy, size, shape, color

#################################################################################
#################################################################################
#################################################################################

#################################################################################
# size + size

def task_size_size_1(condition='c', reference='id'):
    
    n_samples = 4
    
    max_size = 0.6
    min_size = max_size/4
    
    min_diff = (max_size - min_size)/2
    
    size = np.random.rand(n_samples) * (max_size - min_size - min_diff) + min_size
    diff = np.random.rand(n_samples) * (max_size - size - min_diff) + min_diff
    size = np.stack([size, size + diff], axis=1)

    size[-1] = [size[-1,1], size[-1,0]]

    if 'id' in reference:
        shape = []
        s1 = Shape()
        s2 = Shape()
            
        for i in range(n_samples):
            shape.append([s1.clone(), s2.clone()])

    elif 'id' in condition:
        shape = []
        for i in range(n_samples):
            s1 = Shape()
            s2 = Shape()
            shape.append([s1.clone(), s2.clone()])
            
    else:
        s = Shape()
        shape = []
        for i in range(n_samples):
            shape.append([s.clone(), s.clone()])

    co1 = sample_over_range_t(n_samples, np.array([0,1]), size)
    co2 = np.random.rand(n_samples, 2) * (1-size) + size/2
    xy = [co1, co2] if np.random.randint(2)==0 else [co2, co1]
    xy = np.stack(xy, axis=2)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*2)
        color = c.reshape([n_samples, 2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([2, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

def task_size_size_2(condition='xycid'):

    n_samples = 4

    n_objects = 3
    
    max_size = 0.9/2
    min_size = max_size/2

    size_1 = np.random.rand(n_samples-1) * (max_size - min_size) + min_size
    r_prop = np.random.rand(n_samples-1) * (2/3 - 1/3) + 1/3
    idx = np.random.choice(np.arange(n_samples-1), size=np.random.randint(2,n_samples-1), replace=False)
    r_prop[idx] = 1/r_prop[idx]

    size_2 = r_prop * size_1

    s = (2*size_1 + size_2)
    non_valid = s>0.9
    if non_valid.any():
        size_1[non_valid] = size_1[non_valid] / s[non_valid] * 0.9
        size_2[non_valid] = size_2[non_valid] / s[non_valid] * 0.9

    size_reg = np.stack([size_1, size_1, size_2], axis=1)
    
    odd_condition = np.random.randint(2)

    if odd_condition == 0:
        # odd_n_unique_sizes = 1
        size_ = np.random.rand() * (max_size - min_size) + min_size
        if size_*3 >0.9:
            size_ = size_*0.9/3
        size_ = np.ones([3])*size_
    else: 
        # odd_n_unique_sizes = 3 
        size_ = np.random.rand() * (max_size - min_size) + min_size
        r_prop = np.random.rand(2) * (2/3 - 1/3) + 1/3
        size_ = np.array([size_, size_*r_prop[0], size_*r_prop[0]*r_prop[1]])
        # (s + s*r_prop[0] + s*r_prop[0]*r_prop[1])*r < 0.8
        r = 0.9 / size_.sum()
        size_ = size_ * r


    size = np.concatenate([size_reg, size_[None,:]], axis=0)

    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            s1 = Shape()
            s2 = Shape()
            s3 = Shape()
            shape.append([s1, s2, s3])
            
    else:
        s = Shape()
        shape = []
        for i in range(n_samples):
            shape.append([s.clone(), s.clone(), s.clone()])


    if 'xy' in condition:
        
        triu_idx = np.triu_indices(n_objects, k=1)
        triu_idx = triu_idx[0]*n_objects + triu_idx[1]

        # n_samples_over = 100
        xy = np.random.rand(n_samples, n_objects, 2) * (1-size[:,:,None]) + size[:,:,None]/2
        valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,None]+size[:,None,:,None])/2 > 0).any(3).reshape([n_samples, n_objects**2])[:,triu_idx].all(1)
        while not valid.all():
            xy = np.random.rand(n_samples, n_objects, 2) * (1-size[:,:,None]) + size[:,:,None]/2
            valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,None]+size[:,None,:,None])/2 > 0).any(3).reshape([n_samples, n_objects**2])[:,triu_idx].all(1)


    else:
        triu_idx = np.triu_indices(n_objects, k=1)
        triu_idx = triu_idx[0]*n_objects + triu_idx[1]

        size_max = size.max(0)

        n_samples_over = 100
        xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size_max[None,:,None]) + size_max[None,:,None]/2
        valid = (np.abs(xy[:,None,:,None,:] - xy[:,None,None,:,:]) - (size[None,:,:,None,None]+size[None,:,None,:,None])/2 > 0).any(4).reshape([n_samples_over, n_samples, n_objects**2])[:,:,triu_idx].all(2).all(1)
        while not valid.any():
            xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size_max[None,:,None]) + size_max[None,:,None]/2
            valid = (np.abs(xy[:,None,:,None,:] - xy[:,None,None,:,:]) - (size[None,:,:,None,None]+size[None,:,None,:,None])/2 > 0).any(4).reshape([n_samples_over, n_samples, n_objects**2])[:,:,triu_idx].all(2).all(1)
            # valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[None,:,:,None,None]+size[None,:,None,:,None])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)

        xy = xy[valid][0]
        xy = np.stack([xy]*n_samples, 0)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

def task_size_size_3(condition='xycid'):
    
    n_samples = 4

    n_objects = 3

    max_size = 0.9/2
    min_size = max_size/2

    # odd_n_unique_sizes = 3 
    size_ = np.random.rand(n_samples-1) * (max_size - min_size) + min_size
    r_prop = np.random.rand(n_samples-1, 2) * (2/3 - 1/3) + 1/3
    size_ = np.stack([size_, size_*r_prop[:,0], size_*r_prop[:,0]*r_prop[:,1]],1)
    # (s + s*r_prop[0] + s*r_prop[0]*r_prop[1])*r < 0.8

    r = 0.9 / size_.sum(1)
    size_ = size_ * r[:,None]
    size_reg = size_
    
    odd_condition = np.random.randint(2)

    if odd_condition==0:
        # odd_n_unique_sizes = 1
        size_ = np.random.rand() * (max_size - min_size) + min_size
        if size_*3 >0.9:
            size_ = 0.9/3
        size_ = np.ones([3])*size_
    else: 
        size_1 = np.random.rand() * (max_size - min_size) + min_size
        r_prop = np.random.rand() * (2/3 - 1/3) + 1/3
        if np.random.randint(2)==1:
            r_prop = 1/r_prop

        size_2 = r_prop * size_1

        # # r st. 2*s1*r + s2*r < 0.8 
        r = 0.8/(2*size_1 + size_2)
        size_1 = size_1*r
        size_2 = size_2*r

        size_ = np.array([size_1, size_1, size_2])

    size = np.concatenate([size_reg, size_[None,:]], axis=0)

    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            s1 = Shape()
            s2 = Shape()
            s3 = Shape()
            shape.append([s1, s2, s3])
            
    else:
        s = Shape()
        shape = []
        for i in range(n_samples):
            shape.append([s.clone(), s.clone(), s.clone()])


    if 'xy' in condition:
        
        triu_idx = np.triu_indices(n_objects, k=1)
        triu_idx = triu_idx[0]*n_objects + triu_idx[1]

        # n_samples_over = 100
        xy = np.random.rand(n_samples, n_objects, 2) * (1-size[:,:,None]) + size[:,:,None]/2
        valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,None]+size[:,None,:,None])/2 > 0).any(3).reshape([n_samples, n_objects**2])[:,triu_idx].all(1)
        while not valid.all():
            xy = np.random.rand(n_samples, n_objects, 2) * (1-size[:,:,None]) + size[:,:,None]/2
            valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,None]+size[:,None,:,None])/2 > 0).any(3).reshape([n_samples, n_objects**2])[:,triu_idx].all(1)


    else:
        triu_idx = np.triu_indices(n_objects, k=1)
        triu_idx = triu_idx[0]*n_objects + triu_idx[1]

        size_max = size.max(0)

        n_samples_over = 100
        xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size_max[None,:,None]) + size_max[None,:,None]/2
        valid = (np.abs(xy[:,None,:,None,:] - xy[:,None,None,:,:]) - (size[None,:,:,None,None]+size[None,:,None,:,None])/2 > 0).any(4).reshape([n_samples_over, n_samples, n_objects**2])[:,:,triu_idx].all(2).all(1)
        while not valid.any():
            xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size_max[None,:,None]) + size_max[None,:,None]/2
            valid = (np.abs(xy[:,None,:,None,:] - xy[:,None,None,:,:]) - (size[None,:,:,None,None]+size[None,:,None,:,None])/2 > 0).any(4).reshape([n_samples_over, n_samples, n_objects**2])[:,:,triu_idx].all(2).all(1)
            # valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[None,:,:,None,None]+size[None,:,None,:,None])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)

        xy = xy[valid][0]
        xy = np.stack([xy]*n_samples, 0)


    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

def task_size_size_4(condition='xycid'):
    
    n_samples = 4

    n_objects = 2

    max_size = 0.6
    min_size = max_size/2
        
    size = np.random.rand(n_samples) * (max_size - min_size) + min_size
    size = np.stack([size, size/2], axis=1)

    prop_odd = np.random.choice([4/5, 2/3, 1/4, 1/5])

    size[-1,1] = size[-1,0] * prop_odd

    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            shape.append([Shape(), Shape()])
            # s = Shape()
            # shape.append([s.clone(), s.clone()])
            
    else:
        s = Shape()
        shape = []
        for i in range(n_samples):
            shape.append([s.clone(), s.clone()])

    n_objects = 2

    if 'xy' in condition:
        
        xy = []
        for i in range(n_samples):
            xy_ = sample_positions(size[i:i+1], n_sample_min=1)
            xy.append(xy_[0])
        
    else:
        xy = []
        for i in range(n_samples):
            xy_ = sample_positions(size[i:i+1], n_sample_min=1)
            xy.append(xy_[0])

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

def task_size_size_5(condition='cid'):
    reference=''

    n_samples = 4
    n_objects = 3

    max_size = 0.9/n_objects
    min_size = max_size/5
    
    size = sample_over_range_t(n_samples, np.array([min_size, max_size]), np.ones([n_samples,n_objects])*(max_size - min_size)/4)
    y = sample_over_range_t(n_samples, np.array([0.1, 0.9]), np.ones([n_samples,n_objects])*size.max(1)[:,None])
    x = np.random.rand(n_samples, n_objects) * (0.5 - size) + size/2

    xy_shuf_perm = np.stack([sample_shuffle_unshuffle_indices(n_objects)[0] for i in range(n_samples)], 0)

    shuf_perm1, _ = sample_shuffle_unshuffle_indices(n_objects)
    shuf_perm, _ = sample_shuffle_unshuffle_indices(n_objects)
    shuf_perm_odd, _ = sample_shuffle_unshuffle_indices(n_objects)
    while np.abs(shuf_perm_odd-shuf_perm).sum()==0:
        shuf_perm_odd, _ = sample_shuffle_unshuffle_indices(n_objects)

    size2 = size.copy()
    
    perms1 = np.stack([shuf_perm1]*(n_samples), 0)
    perms2 = np.stack([shuf_perm]*(n_samples-1)+ [shuf_perm_odd], 0)

    shuffle_t(size, perms1)
    shuffle_t(size2, perms2)

    shuffle_t(y, xy_shuf_perm)
    
    size = np.concatenate([size,size2],1)
    y = np.concatenate([y,y],1)
    x = np.concatenate([x,x+0.5],1)
    xy = np.stack([x,y],2)

    if 'id' in reference:
        shape = []
        for i in range(n_samples):
            shape_1, shape_2 = [], []
            for i in range(n_objects):
                s = Shape()
                shape_1.append(s)
                shape_2.append(s.clone())

            shape.append(shape_1 + shape_2)

    elif 'id' in condition:
        shape = [[Shape() for _ in range(n_objects*2)] for i in range(n_samples)]
    
    else:
        s = Shape()
        shape = [[s.clone() for _ in range(n_objects*2)] for i in range(n_samples)]

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects*2)
        color = c.reshape([n_samples, n_objects*2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects*2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects*2, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# size + shape

def task_size_shape_1(condition='xyscid'):
        
    n_samples = 4

    n_objects = 4
    n_objects_max = n_objects
 
    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]

    if 'id' in condition: 
        shape = [[s1.clone(), s1.clone(), s2.clone(), s2.clone()] for s1,s2 in [(Shape(), Shape()) for i in range(n_samples)]]
    else:
        s1 = Shape()
        s2 = Shape()    
        shape = [[s1.clone(), s1.clone(), s2.clone(), s2.clone()] for i in range(n_samples)]

    if 'r' in condition:
        angle = np.random.rand(n_samples, n_objects_max) * 2 * np.pi
        for i in range(n_samples-1):
            for j in range(n_objects):
                shape[i][j].rotate(angle[i,j])
                # shape[i][1].rotate(angle[i,1])

    if 'f' in condition:
        for i in range(n_samples-1):
            for j in range(n_objects):
                if np.random.rand()>0.5:
                    shape[i][j].flip()

    max_size = 0.9/np.sqrt(n_objects*2)
    min_size = max_size/2

    max_r, min_r = 2/3, 1/3
    r = np.random.rand(n_samples) * (max_r-min_r) + min_r
    size = np.random.rand(n_samples) * (max_size-min_size) + min_size

    if np.random.randint(2)==0:
        size = np.stack([size, size, size*r, size*r], 1)
        non_valid = size.sum(1)>0.9
        if non_valid.any():
            size[non_valid] = size[non_valid] / size[non_valid].sum(1)[:,None] * 0.9

        # odd one out
        size[-1] = np.array([size[-1,0], size[-1,2], size[-1,1], size[-1,3]])
    else:
        size = np.stack([size, size*r, size, size*r], 1)
        non_valid = size.sum(1)>0.9
        if non_valid.any():
            size[non_valid] = size[non_valid] / size[non_valid].sum(1)[:,None] * 0.9
        # odd one out
        size[-1] = np.array([size[-1,0], size[-1,2], size[-1,1], size[-1,3]])

    if 'xy' in condition:        
        xy = []
        for i in range(n_samples):
            xy_ = sample_positions(size[i:i+1], n_sample_min=1)
            xy.append(xy_[0])
        
    else:
        xy = sample_positions(size.max(0)[None,:], n_sample_min=n_samples)

    return xy, size, shape, color

def task_size_shape_2(condition='xyc', reference='id'):

    n_samples = 4

    n_objects = 4
    
    # odd_n_unique_sizes = 3 
    # size_ = np.random.rand(n_samples-1)
    # size = np.random.rand()

    max_size = 0.9/np.sqrt(n_objects*2)
    min_size = max_size/2
    # min_diff = (max_size - min_size)/2
    
    # size = np.random.rand(n_samples, 2) * (max_size - min_size - min_diff) + min_size
    # diff = np.random.rand(n_samples, 2) * (max_size - size - min_diff) + min_diff
    # size = np.concatenate([size, size + diff], axis=1)

    max_ratio, min_ratio = 2/3, 1/3
    ratio = np.random.rand(n_samples, 2)*(max_ratio - min_ratio) + min_ratio

    size = np.random.rand(n_samples, 2) * (max_size - min_size) + min_size

    size = np.concatenate([size, size*ratio], axis=1)
    
    for i in range(n_samples):
        if np.random.randint(2)==0:
            size[i] = size[i, [2,3,0,1]]

    # a>b <=> b>c

    odd_condition = np.random.randint(2)
    
    size_switch = (size[-1,1], size[-1,3])
    size[-1:,1] = size_switch[1]
    size[-1:,3] = size_switch[0]


    if 'id' in reference:
        shape = []
        s1 = Shape()
        s2 = Shape()
        s3 = Shape()
        s4 = Shape()
        for i in range(n_samples):
            shape.append([s1.clone(), s2.clone(), s3.clone(), s4.clone()])

    elif 'id' in condition:
        shape = []
        for i in range(n_samples):
            s1 = Shape()
            s2 = Shape()
            s3 = Shape()
            s4 = Shape()
            shape.append([s1, s2, s3, s4])
            
    else:
        s = Shape()
        shape = []
        for i in range(n_samples):
            shape.append([s.clone(), s.clone(), s.clone(), s.clone()])


    if 'xy' in reference or 'xy' not in condition:
        triu_idx = np.triu_indices(n_objects, k=1)
        triu_idx = triu_idx[0]*n_objects + triu_idx[1]

        size_max = size.max(0)

        n_samples_over = 100
        xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size_max[None,:,None]) + size_max[None,:,None]/2
        valid = (np.abs(xy[:,None,:,None,:] - xy[:,None,None,:,:]) - (size[None,:,:,None,None]+size[None,:,None,:,None])/2 > 0).any(4).reshape([n_samples_over, n_samples, n_objects**2])[:,:,triu_idx].all(2).all(1)
        while not valid.any():
            xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size_max[None,:,None]) + size_max[None,:,None]/2
            valid = (np.abs(xy[:,None,:,None,:] - xy[:,None,None,:,:]) - (size[None,:,:,None,None]+size[None,:,None,:,None])/2 > 0).any(4).reshape([n_samples_over, n_samples, n_objects**2])[:,:,triu_idx].all(2).all(1)
            # valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[None,:,:,None,None]+size[None,:,None,:,None])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)

        xy = xy[valid][0]
        xy = np.stack([xy]*n_samples, 0)

    elif 'xy' in condition:
        
        triu_idx = np.triu_indices(n_objects, k=1)
        triu_idx = triu_idx[0]*n_objects + triu_idx[1]

        # n_samples_over = 100
        xy = np.random.rand(n_samples, n_objects, 2) * (1-size[:,:,None]) + size[:,:,None]/2
        valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,None]+size[:,None,:,None])/2 > 0).any(3).reshape([n_samples, n_objects**2])[:,triu_idx].all(1)
        while not valid.all():
            xy = np.random.rand(n_samples, n_objects, 2) * (1-size[:,:,None]) + size[:,:,None]/2
            valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,None]+size[:,None,:,None])/2 > 0).any(3).reshape([n_samples, n_objects**2])[:,triu_idx].all(1)


    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# size + color

def task_size_color_1(condition='xycid_hv'):
        
    n_samples = 4
    n_objects = 2
    
    max_size = 0.6
    min_size = max_size/2

    max_r, min_r = 2/3, 2/5
    r = np.random.rand(n_samples) * (max_r-min_r) + min_r
    size = np.random.rand(n_samples) * (max_size-min_size) + min_size
    size = np.stack([size, size*r], 1)
    non_valid = size.sum(1)>0.9
    if non_valid.any():
        size[non_valid] = size[non_valid] / size[non_valid].sum(1)[:,None] * 0.9
    # odd one out
    size[-1] = size[-1,::-1]

    if 'xy' in condition:
            
        co1 = sample_over_range_t(n_samples, np.array([0,1]), size*np.sqrt(2))
        co2 = np.random.rand(n_samples, 2) * (1-size*np.sqrt(2)) + size*np.sqrt(2)/2
        xy = [co1, co2]# if np.random.randint(2)==0 else [co2, co1]
        xy = np.stack(xy, axis=2)
        perm = np.random.randint(2, size=n_samples)
        xy[perm==0] = xy[perm==0, :, ::-1]
       
    else:
        x = np.ones(n_samples, 2) * 0.25
        x[:,1] = 1 - x[:,1]
        y = np.ones(n_samples, 2) * 0.5
        xy = np.stack([x,y], axis=2)

    h1 = np.random.rand()
    min_diff = 0.15
    
    h2 = (h1 + (np.random.rand(n_samples) * (1-min_diff*2) + min_diff) * np.random.choice([-1,1]))%1
    h = np.stack([h1 * np.ones([n_samples]),h2], 1) 
    
    # if 'h' in condition:
    #     h = np.random.rand(n_samples, 2)
    # else:
    #     h = np.random.rand()
    #     h = h * np.ones([n_samples, 2])

    if 's' in condition:
        s = (np.random.rand(n_samples, 2) * 0.6 + 0.4) * np.ones([n_samples, 2])
    else:
        s = np.random.rand() * 0.6 + 0.4
        s = s * np.ones([n_samples, 2])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples, 2) * 0.5 + 0.35)
    else:
        v = np.random.rand() * 0.5 + 0.35
        v = v * np.ones([n_samples, 2])
 
    color = np.stack([h,s,v], 2)

    if 'id' in condition:
        shape = [[Shape(), Shape()] for i in range(n_samples)]

    else:
        s = Shape()
        shape = [[s.clone(), s.clone()] for _ in range(n_samples)]

    if 'r' in condition:
        angle = np.random.rand(n_samples, 2) * 2 * np.pi
        for i in range(n_samples-1):
            shape[i][0].rotate(angle[i,0])
            shape[i][1].rotate(angle[i,1])

    if 'f' in condition:
        for i in range(n_samples-1):
            shape[i][1].flip()

    return xy, size, shape, color

def task_size_color_2(condition='idv'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.choice([2,3,4,5], replace=False)
    # n_unique_sizes = n_groups
    n_objects_samples = np.random.randint(2, 4, size=n_groups)
    n_objects = n_objects_samples.sum()
    
    size_idx = [[i]*n_objects_samples[i] for i in range(n_groups)]
    size_idx = np.array(cat_lists(size_idx)) 
    
    max_size = 0.9/(np.sqrt(n_objects*2))
    min_size = max_size*3/4

    max_r, min_r = 2/3, 2/5
    
    min_diff = (max_r - min_r)/4
    r = sample_over_range_t(1, np.array([min_r - min_diff/2, max_r + min_diff/2]), np.ones([1, n_groups])* min_diff)
    u_size = np.random.rand(n_samples) * (max_size - min_size) + min_size
    u_size = np.concatenate([u_size[:,None]*r, u_size[:,None]], 1)

    size = [u_size[i, size_idx] for i in range(n_samples)] # + [u_size_odd[0, size_idx_odd]]
    size = np.stack(size, 0)
    
    non_valid = size.__pow__(2).sum(1).__pow__(1/2)>0.9
    if non_valid.any():
        size[non_valid] = size[non_valid] / size[non_valid].__pow__(2).sum(1).__pow__(1/2)[:,None] * 0.8

    #########
    color_idx = [cat_lists([[j]*n_objects_samples[j] for j in range(n_groups)]) for i in range(n_samples)]

    color_idx[-1] = np.roll(color_idx[-1], 1).tolist()

    max_size = 0.9/n_objects
    min_size = max_size/2

    min_diff = 0.15

    h_u = sample_over_range_t(n_samples, np.array([0,1]), np.ones([n_samples, n_groups])*min_diff)
    # h_u_odd = sample_over_range_t(1, np.array([0,1]), np.ones([1, n_groups_odd])*min_diff)

    if 's' in condition:
        s = (np.random.rand(n_samples, n_groups) * 0.6 + 0.4)
    else:
        s = np.random.rand() * 0.6 + 0.4
        s = s * np.ones([n_samples, n_groups])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples, n_groups) * 0.5 + 0.35)
    else:
        v = np.random.rand() * 0.5 + 0.35
        v = v * np.ones([n_samples, n_groups])

    color_u = np.stack([h_u,s[:n_samples, :n_groups],v[:n_samples, :n_groups]], 2)
    # color_u_odd = np.stack([h_u_odd,s[-1:, :n_groups_odd],v[-1:, :n_groups_odd]], 2)

    color = [color_u[i, color_idx[i]] for i in range(n_samples)]
    
    #########

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
        
    if 'id' in condition:
        shape = [[Shape() for i in range(n_objects)] for i in range(n_samples)]

        # shape = []
        # for i in range(n_samples):
        #     s = Shape()
        #     shape += [[s.clone() for i in range(n_objects)]]
    else:
        s = Shape()
        shape = [[s.clone() for i in range(n_objects)] for _ in range(n_samples)]

    return xy, size, shape, color


#################################################################################
# size + rotation

def task_size_rot(condition='cid'):
        
    n_samples = 4
    n_objects = 3
    
    max_size = 0.9/(np.sqrt(n_objects*2))
    min_size = max_size*2/3

    max_r, min_r = 3/4, 2/3
    r = np.random.rand(n_samples, 2) * (max_r-min_r) + min_r
    size = np.random.rand(n_samples) * (max_size-min_size) + min_size
    size = np.stack([size, size*r[:,0], size*r[:,0]*r[:,1]], 1)
    
    non_valid = size.sum(1)>0.9
    
    if non_valid.any():
        size[non_valid] = size[non_valid] / size[non_valid].sum(1)[:,None] * 0.9
    
    # odd one out
    size[-1] = size[-1, ::-1]

    angles = (np.random.rand(n_samples) * (1/2 - 1/6) + 1/6) * np.pi
    angles1 = (np.random.rand(n_samples) * (1/2 - 1/6) + 1/6) * np.pi

    angles = angles[:, None] + np.stack([angles1, np.zeros(n_samples), -angles1], 1)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]

    if 'id' in condition:
        shape = [[s.clone(), s.clone(), s.clone()] for s in [Shape() for _ in range(n_samples)]]
    else:
        s = Shape()
        shape = [[s.clone(), s.clone(), s.clone()] for _ in range(n_samples)]

    for i in range(n_samples):
        shape[i][0].rotate(angles[i,0])
        shape[i][1].rotate(angles[i,1])
        shape[i][2].rotate(angles[i,2])
    
    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])

    return xy, size, shape, color

#################################################################################
# size + flip

def task_size_flip_1(condition='cid'):
        
    n_samples = 4
    n_objects = 3
    
    max_size = 0.9/(np.sqrt(n_objects*2))
    min_size = max_size*2/3

    max_r, min_r = 3/4, 2/3
    r = np.random.rand(n_samples, 2) * (max_r-min_r) + min_r
    size = np.random.rand(n_samples) * (max_size-min_size) + min_size
    size = np.stack([size, size*r[:,0], size*r[:,0]*r[:,1]], 1)
    
    non_valid = size.sum(1)>0.9
    
    if non_valid.any():
        size[non_valid] = size[non_valid] / size[non_valid].sum(1)[:,None] * 0.9
    
    # odd one out
    size[-1] = size[-1, ::-1]

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]

    if 'id' in condition:
        shape = [[s.clone(), s.clone(), s.clone()] for s in [Shape() for _ in range(n_samples)]]
    else:
        s = Shape()
        shape = [[s.clone(), s.clone(), s.clone()] for _ in range(n_samples)]

    for i in range(n_samples):
        shape[i][0].flip()
        # shape[i][1].rotate(angles[i,1])
        shape[i][2].flip()
        shape[i][2].rotate(np.pi)
    
    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])

    return xy, size, shape, color


#################################################################################
# size + inside

def task_size_inside_1(condition='xyc'):
        
    n_samples = 4
    n_objects = 2
    
    max_attempts = 10

    max_size = 0.9/np.sqrt(n_objects*2)
    min_size = max_size/2

    max_r, min_r = 3/4, 1/2
    r = np.random.rand(n_samples) * (max_r-min_r) + min_r
    size = np.random.rand(n_samples) * (max_size-min_size) + min_size
    if np.random.randint(2)==0:
        size = np.stack([size, size*r], 1)
    else:
        size = np.stack([size*r, size], 1)

    non_valid = size.sum(1)>0.9
    if non_valid.any():
        size[non_valid] = size[non_valid] / size[non_valid].sum(1)[:,None] * 0.9

    # odd one out
    size[-1] = size[-1, ::-1]

    # size = np.random.rand(n_objects) * (max_size - min_size) + min_size 
    size_in = np.random.rand(n_samples) * (size[:,0]/2.5 - size[:,0]/4) + size[:,0]/4
    size = np.concatenate([size, size_in[:,None]], 1)

    if 'xy' in condition:
            
        co1 = sample_over_range_t(n_samples, np.array([0,1]), size[:,:2])
        co2 = np.random.rand(n_samples, 2) * (1-size[:,:2]) + size[:,:2]/2
        xy = [co1, co2] if np.random.randint(2)==0 else [co2, co1]
        xy = np.stack(xy, axis=2)

        perm = np.random.randint(2, size=n_samples)
        xy[perm==0] = xy[perm==0, :, ::-1]
       
    else:
        x = np.ones(n_samples, 2) * 0.25
        x[:,1] = 1 - x[:,1]
        y = np.ones(n_samples, 2) * 0.5
        xy = np.stack([x,y], axis=2)


    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*(n_objects+1))
        color = c.reshape([n_samples, n_objects+1, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects+1, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects+1, 3]) * color for i in range(n_samples)]

    shape = []
    xy_in = []
    # size_in = []

    for i in range(n_samples):
        s1 = Shape(gap_max=0.06, hole_radius=0.2)
        s2 = Shape()
        done = False
        for _ in range(max_attempts):

            samples = sample_position_inside_1(s1, s2, size_in[i]/size[i,0])
            if len(samples)>0:
                done = True
            
            if done:
                break
            else:
                s1.randomize()
                s2.randomize()

        if done:
            xy_ = samples[0]
            xy_in.append(xy_ * size[i,0] + xy[i,0])
    
        shape.append([s1, Shape(gap_max=0.06, hole_radius=0.2), s2])
    
    xy_in = np.array(xy_in)
    xy = np.concatenate([xy, xy_in[:, None, :]], 1)

    return xy, size, shape, color

def task_size_inside_2(condition='xysc'):
        
    n_samples = 4
    n_objects = 2
    
    max_attempts = 10

    max_size = 0.9/n_objects
    min_size = max_size/2

    max_r, min_r = 2/5, 1/5
    min_diff_r = 0.5
    r1 = np.random.rand(n_samples)
    r2 = np.mod(r1 + min_diff_r, 1)
    r = np.stack([r1, r2], 1) * (max_r-min_r) + min_r
    # odd one out
    r[-1,1] = r[-1,0]

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    size_in = size * r
    size = np.concatenate([size, size_in], 1)

    if 'xy' in condition:
            
        co1 = sample_over_range_t(n_samples, np.array([0,1]), size[:,:2])
        co2 = np.random.rand(n_samples, 2) * (1-size[:,:2]) + size[:,:2]/2
        xy = [co1, co2]# if np.random.randint(2)==0 else [co2, co1]
        xy = np.stack(xy, axis=2)

        perm = np.random.randint(2, size=n_samples)
        xy[perm==0] = xy[perm==0, :, ::-1]
       
    else:
        x = np.ones(n_samples, 2) * 0.25
        x[:,1] = 1 - x[:,1]
        y = np.ones(n_samples, 2) * 0.5
        xy = np.stack([x,y], axis=2)


    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects*2)
        color = c.reshape([n_samples, n_objects*2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects*2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects*2, 3]) * color for i in range(n_samples)]

    shape = []
    xy_in = []
    # size_in = []

    for i in range(n_samples):
        s1 = Shape(gap_max=0.08, hole_radius=0.2)
        s2 = Shape(gap_max=0.08, hole_radius=0.2)
        s3 = Shape()
        s4 = Shape()

        done1, done2 = False, False
        for _ in range(max_attempts):

            samples = sample_position_inside_1(s1, s3, size_in[i,0]/size[i,0])
            if len(samples)>0:
                done1 = True
            
            if done1:
                break
            else:
                s1.randomize()
                s3.randomize()
        if done1:
            xy_ = samples[0]
            xy_in.append(xy_ * size[i,0] + xy[i,0])

        for _ in range(max_attempts):
            samples = sample_position_inside_1(s2, s4, size_in[i,1]/size[i,1])
            if len(samples)>0:
                done2 = True
            
            if done2:
                break
            else:
                s2.randomize()
                s4.randomize()

        if done2:
            xy_ = samples[0]
            xy_in.append(xy_ * size[i,1] + xy[i,1])
    
        shape.append([s1, s2, s3, s4])
    
    xy_in = np.stack(xy_in, 0).reshape([n_samples, 2, 2])
    xy = np.concatenate([xy, xy_in], 1)

    # if 'f' in condition:
    #     for i in range(n_samples-1):
    #         shape[i][1].flip()

    return xy, size, shape, color

#################################################################################
# size + contact

def task_size_contact(condition='sid'):
        
    n_samples = 4
    n_objects = 4
    
    max_size = 0.9/np.sqrt(n_objects*2)
    min_size = max_size*3/4

    # a=b c=d a=c 1
    # a=b c=d a<c 2
    # a>b c>d a=c b=d 2
    # a=b c>d a>c 3
    # a=b c>d a<d 3
    # a=b c>d c<a<d 3
    # a<b c<d b<c 4 smaller-small big-bigger
    # a<c c<b b<d 4 smaller-big small-bigger
    # a<c c<d d<b 4 smaller-bigger small-big

    perm = np.array([    
        [0,0,0,0],
        [0,0,1,1],
        [0,1,0,1],
        [2,2,1,0],
        [0,0,2,1],
        [1,1,2,0],
        [0,1,2,3],
        [0,2,1,3],
        [0,3,1,2],
    ])

    all_conditions = [0,1,2,3,4,5,6,7,8]
    cond_reg, cond_odd = np.random.choice(all_conditions, replace=False, size=2)
    
    n_unique_sizes = [1,2,2,3,3,3,4,4,4]
    n_uniques_sizes_reg = n_unique_sizes[cond_reg]
    n_uniques_sizes_odd = n_unique_sizes[cond_odd]

    max_r, min_r = 2/3, 2/5

    if n_uniques_sizes_reg>1:
        min_diff = (max_r - min_r)/4
        if n_uniques_sizes_reg>2:
            r = sample_over_range_t(n_samples-1, np.array([min_r - min_diff/2, max_r + min_diff/2]), np.ones([n_samples-1, n_uniques_sizes_reg-1])* min_diff)
        else:
            r = np.random.rand(n_samples-1) * (max_r-min_r) + min_r
        u_size = np.random.rand(n_samples-1) * (max_size - min_size) + min_size

        u_size_reg = np.concatenate([u_size[:,None]*r, u_size[:,None]], 1)
    else:
        u_size_reg = np.random.rand(n_samples-1)[:,None] * (max_size - min_size) + min_size

    max_r, min_r = 2/3, 2/5

    if n_uniques_sizes_odd>1:
        # max_r, min_r = 2/3, 1/5
        min_diff = (max_r - min_r)/4
        if n_uniques_sizes_odd>2:
            r = sample_over_range_t(1, np.array([min_r - min_diff/2, max_r + min_diff/2]), np.ones([1, n_uniques_sizes_odd-1])* min_diff)
        else:
            r = np.random.rand(1) * (max_r-min_r) + min_r
        u_size = np.random.rand(1) * (max_size - min_size) + min_size

        u_size_odd = np.concatenate([u_size[:,None]*r, u_size[:,None]], 1)
    else:
        u_size_odd = np.random.rand(1) * (max_size - min_size) + min_size
        u_size_odd = np.ones([1,1]) * u_size_odd
    
    perm_reg = perm[cond_reg]
    perm_odd = perm[cond_odd]

    size = [u_size_reg[i, perm_reg] for i in range(n_samples-1)]
    size += [u_size_odd[0, perm_odd]]

    size = np.stack(size, 0)
    
    non_valid = size.sum(1)>0.9
    if non_valid.any():
        size[non_valid] = size[non_valid] / size[non_valid].sum(1)[:,None] * 0.9


    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            # s = Shape()
            # shape += [[s.clone() for i in range(n_objects)]]
            shape += [[Shape() for i in range(n_objects)]]

    else:
        s = Shape()
        shape = [[s.clone() for i in range(n_objects)] for _ in range(n_samples)]

    if 'r' in condition:
        angle = np.random.rand(n_samples, n_objects) * 2 * np.pi
        for i in range(n_samples-1):
            for j in range(n_objects):
                shape[i][0].rotate(angle[i,j])

    if 'f' in condition:
        for i in range(n_samples):
            for j in range(n_objects):
                if np.random.randint(2) == 0:
                    shape[i][j].flip()

    # shape = []
    xy = []
    for i in range(n_samples):
            
        s1 = shape[i][0]
        s2 = shape[i][1]
        s3 = shape[i][2]
        s4 = shape[i][2]

        positions1, clump_size1 = sample_contact_many([s1, s2], size[i,:2])
        positions2, clump_size2 = sample_contact_many([s3, s4], size[i,2:])
        # positions1 = positions1*0.9
        # positions2 = positions2*0.9
        c_size = np.stack([clump_size1, clump_size2], 0)
        co1 = sample_over_range_t(1, np.array([0,1]), c_size[None,:,0])
        co2 = np.random.rand(1, 2) * (1-c_size[None,:,1]) + c_size[None,:,1]/2
        xy_ = [co1, co2]# if np.random.randint(2)==0 else [co2, co1]
        xy_ = np.stack(xy_, axis=2)[0]
        
        xy_1 = positions1 + xy_[0:1,:]
        xy_2 = positions2 + xy_[1:,:]
        xy_ = np.concatenate([xy_1, xy_2], 0)
        xy.append(xy_)

    xy = np.stack(xy, 0)
    
    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# size + count

def task_size_count_1(condition='idc'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.randint(2, 4)
    # n_unique_sizes = np.min(np.random.randint(2, 5), n_groups)
    n_unique_sizes = n_groups
    n_objects_samples = np.random.randint(2, 4, size=n_groups)
    n_objects = n_objects_samples.sum()

    n_objects_samples_odd = sample_int_sum_n(n_groups, n_objects, min_v=1).astype(int)
    while (n_objects_samples_odd == n_objects_samples).all():
        n_objects_samples_odd = sample_int_sum_n(n_groups, n_objects, min_v=1).astype(int)

    size_idx = [[i]*n_objects_samples[i] for i in range(n_groups)]
    size_idx = np.array(cat_lists(size_idx)) 
    size_idx_odd = [[i]*n_objects_samples_odd[i] for i in range(n_groups)]
    size_idx_odd = np.array(cat_lists(size_idx_odd)) 
    
    # n_objects_samples = np.stack([n_objects_samples]*n_samples, 0)
    # n_objects_samples[-1] = n_objects_samples_odd

    max_size = 0.9/(np.sqrt(n_objects))
    min_size = max_size/2

    max_r, min_r = 2/3, 1/4
    
    min_diff = (max_r - min_r)/4
    r = sample_over_range_t(1, np.array([min_r - min_diff/2, max_r + min_diff/2]), np.ones([1, n_unique_sizes])* min_diff)
    u_size = np.random.rand(n_samples-1) * (max_size - min_size) + min_size
    u_size = np.concatenate([u_size[:,None]*r, u_size[:,None]], 1)

    size = [u_size[i, size_idx] for i in range(n_samples-1)] + [u_size[-1, size_idx_odd]]
    size = np.stack(size, 0)
    
    non_valid = size.__pow__(2).sum(1).__pow__(1/2)>0.9
    if non_valid.any():
        size[non_valid] = size[non_valid] / size[non_valid].__pow__(2).sum(1).__pow__(1/2)[:,None] * 0.8

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])

    if 'id' in condition:
        shape = [[Shape() for i in range(n_objects)] for i in range(n_samples)]
    else:
        s = Shape()
        shape = [[s.clone() for i in range(n_objects)] for _ in range(n_samples)]

    
    if 'c' in condition:
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])

    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, size, shape, color
    
def task_size_count_2(condition='idc'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.choice([2,3,4], size=2, replace=False)
    # n_unique_sizes = n_groups
    n_objects_samples = np.random.randint(2, 4, size=n_groups[0])
    n_objects = n_objects_samples.sum()
    n_objects_samples_odd = sample_int_sum_n(n_groups[1], n_objects, min_v=1)

    size_idx = [[i]*n_objects_samples[i] for i in range(n_groups[0])]
    size_idx = np.array(cat_lists(size_idx)) 
    size_idx_odd = [[i]*n_objects_samples_odd[i] for i in range(n_groups[1])]
    size_idx_odd = np.array(cat_lists(size_idx_odd)) 

    max_size = 0.9/(np.sqrt(n_objects))
    min_size = max_size/2

    max_r, min_r = 2/3, 2/5
    
    min_diff = (max_r - min_r)/4
    r = sample_over_range_t(1, np.array([min_r - min_diff/2, max_r + min_diff/2]), np.ones([1, n_groups[0]])* min_diff)
    u_size = np.random.rand(n_samples-1) * (max_size - min_size) + min_size
    u_size = np.concatenate([u_size[:,None]*r, u_size[:,None]], 1)

    min_diff = (max_r - min_r)/4
    r = sample_over_range_t(1, np.array([min_r - min_diff/2, max_r + min_diff/2]), np.ones([1, n_groups[1]])* min_diff)
    u_size_odd = np.random.rand(1) * (max_size - min_size) + min_size
    u_size_odd = np.concatenate([u_size_odd[:,None]*r, u_size_odd[:,None]], 1)

    size = [u_size[i, size_idx] for i in range(n_samples-1)] + [u_size_odd[0, size_idx_odd]]
    size = np.stack(size, 0)
    
    non_valid = size.__pow__(2).sum(1).__pow__(1/2)>0.9
    if non_valid.any():
        size[non_valid] = size[non_valid] / size[non_valid].__pow__(2).sum(1).__pow__(1/2)[:,None] * 0.8

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
        
    if 'id' in condition:
        shape = [[Shape() for i in range(n_objects)] for i in range(n_samples)]
        # shape = []
        # for i in range(n_samples):
        #     s = Shape()
        #     shape += [[s.clone() for i in range(n_objects)]]
    else:
        s = Shape()
        shape = [[s.clone() for i in range(n_objects)] for _ in range(n_samples)]

    
    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]

    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, size, shape, color

#################################################################################
# size + sym # set 2

def task_size_sym_1(condition='xyc'):

    # mir sym:
    # (x0,y0), r, a
    # o_1 = (0, a - pi/2) 
    # pos_1 = r * (cos(a1),sin(a1)) + (x0,y0)
    # o_2 = (1, a + pi/2)
    # pos_2 = r * (cos(a2),sin(a2)) + (x0,y0)
    # and

    n_samples = 4

    n_objects = 5
    
    # odd: distance not correlated to size, no sym for n_objects
    odd_condition = np.random.randint(2)
    # odd_condition = 0
    
    max_size = 0.25
    min_size = max_size/5
    
    size = np.random.rand(n_samples, n_objects) * (max_size - min_size) + min_size
    non_valid = size.sum(1) > 0.9
    if non_valid.any():
        size[non_valid] = size[non_valid]/size[non_valid].sum(1)[:,None] * 0.9

    margin_r = 1.3

    x_range = (0.5 - (size.max(1) + size.min(1))/2*margin_r)
    x_starting = size.min(1)/2*margin_r
    
    x1 = (size - size.min(1)[:,None]) / (size.max(1) - size.min(1))[:,None] * x_range[:,None] + x_starting[:,None]
    x1 = 0.5 - x1

    if odd_condition == 0:
        x1[-1] = np.random.rand(n_objects) * (0.5 - size[-1]) + size[-1]/2
        
    x2 = 1 - x1

    y = sample_over_range_t(n_samples, np.array([0,1]), size)

    size = np.concatenate([size]*2, axis=1)

    x = np.concatenate([x1,x2], 1)
    y = np.concatenate([y,y], 1)
    xy = np.stack([x,y], 2)
    # choose size
    # choose x based on size
    # choose y st there's no overlap

    
    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
            
            if odd_condition in [1,2] and i==n_samples-1:
                odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects-1), replace=False)
            
            for j in range(n_objects):
                s = Shape()
                s1 = s.clone()
                s2 = s.clone()

                # a_ = a[i]

                # if odd_condition == 4 and i == n_samples-1:
                #     a_ = a_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi
                
                # s1.rotate(a_)

                if not (odd_condition == 2 and i == n_samples-1 and j in odd_samples):
                    s2.flip()
                
                # s2.rotate(a_)

                if odd_condition == 1 and i == n_samples-1 and j in odd_samples:
                    s_ = Shape()
                    if np.random.rand()>0.5:
                        s1 = s_
                    else:
                        s2 = s_

                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append(n_objs + n_objs_f)
            
    else:
        shape = []
        s = Shape()
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
            
            if odd_condition in [1,2] and i==n_samples-1:
                odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects-1), replace=False)
            
            for j in range(n_objects):

                s1 = s.clone()
                s2 = s.clone()

                # a_ = a[i]
                
                # if odd_condition == 4 and i == n_samples-1:
                #     a_ = a_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi

                # s1.rotate(a_)
                
                if not (odd_condition == 2 and i == n_samples-1 and j in odd_samples):
                    s2.flip()
                
                # s2.rotate(a_)

                if odd_condition == 1 and i == n_samples-1 and j in odd_samples:
                    s_ = Shape()
                    if np.random.rand()>0.5:
                        s1 = s_
                    else:
                        s2 = s_

                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append(n_objs + n_objs_f)

    if 'c' in condition:
        color = sample_random_colors(n_samples)
        color = [np.ones([n_objects*2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects*2, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

def task_size_sym_2(condition='xyc'):

    n_samples = 4

    n_objects = 3

    xy0 = np.ones([n_samples, 1, 2]) * 0.5
    
    # odd: distance not correlated to size, no sym for n_objects
    odd_condition = np.random.randint(2)
    # odd_condition = 0
    
    max_size = 0.20
    min_size = max_size/5
    
    size = np.random.rand(n_samples, n_objects) * (max_size - min_size) + min_size
    size = np.concatenate([size]*2, 1)
    size_margin = 0.02
    size = (size-size.min(1)[:,None])/(size.max(1)-size.min(1))[:,None] * (max_size - min_size - size_margin*2) + min_size + size_margin
    # rad = np.random.rand(n_samples, n_objects) * (0.5 - size[:,:n_objects]*np.sqrt(2)) + size[:,:n_objects]*np.sqrt(2)/2
    # rad = np.concatenate([rad]*2, 1)
    
    # non_valid = np.pi - (size[:,:n_objects]*np.sqrt(2)/rad[:,:n_objects]).sum(1) < 0
    # if non_valid.any():
    #     size[non_valid] = size[non_valid] / (size[non_valid,:n_objects]*np.sqrt(2)/rad[non_valid,:n_objects]).sum(1) * np.pi * 0.9

    # keep rad or size

    # np.stack(np.random.permutation(n_objects*2)
    shuffle_perm, unshuffle_perm = zip(*[sample_shuffle_unshuffle_indices(n_objects*2) for _ in range(n_samples)])
    shuffle_perm, unshuffle_perm = np.stack(shuffle_perm, 0), np.stack(unshuffle_perm, 0)

    
    n_samples_over = 100

    angles_ = []
    xy_ = []
    rad_ = []
    triu_idx = np.triu_indices(n_objects*2, k=1)
    triu_idx = triu_idx[0]*n_objects*2 + triu_idx[1]

    for i in range(n_samples):
        shuf_size = size.copy()
        # shuffle_t(shuf_size, shuffle_perm)

        if i == n_samples-1:
            rad = np.random.rand(n_samples_over, n_objects) * (0.5 - shuf_size[i:i+1,:n_objects]*np.sqrt(2)) + shuf_size[i:i+1,:n_objects]*np.sqrt(2)/2
            rad = np.concatenate([rad]*2, 1)
        else:
            margin_r = 1.2
            rad = (shuf_size[i:i+1] - shuf_size[i:i+1].min()) / (shuf_size[i:i+1].max() - shuf_size[i:i+1].min()) * (0.5 - (shuf_size[i:i+1].max() + shuf_size[i:i+1].min())*margin_r*np.sqrt(2)/2 ) + shuf_size[i:i+1].min()*margin_r
            rad = rad * np.ones([n_samples_over, n_objects*2])


        # angles = sample_over_range_t(n_samples_over, np.array([0,2*np.pi]), shuf_size[i:i+1]*np.sqrt(2)/rad)
        angles = np.random.rand(n_samples_over, n_objects*2) * np.pi * 2

        # shuffle_t(angles, unshuffle_perm)
        xy = xy0[0:1,:,:] + rad[:,:,None] * np.stack([np.cos(angles), np.sin(angles)], 2)[:,:,:]
        valid = (np.linalg.norm(xy[:,:,None,:] - xy[:,None,:,:], axis=-1) - (shuf_size[None,i,:,None]+shuf_size[None,i,None,:])*np.sqrt(2)/2 > 0).reshape([n_samples_over, (n_objects*2)**2])[:,triu_idx].all(1)
        while not valid.any():    

            if i == n_samples-1:
                rad = np.random.rand(n_samples_over, n_objects) * (0.5 - shuf_size[i:i+1,:n_objects]*np.sqrt(2)) + shuf_size[i:i+1,:n_objects]*np.sqrt(2)/2
                rad = np.concatenate([rad]*2, 1)

            # angles = sample_over_range_t(n_samples_over, np.array([0,2*np.pi]), shuf_size[i:i+1]*np.sqrt(2)/rad)
            angles = np.random.rand(n_samples_over, n_objects*2) * np.pi * 2

            xy = xy0[0:1,:,:] + rad[i:i+1,:,None] * np.stack([np.cos(angles), np.sin(angles)], 2)[:,:,:]
            valid = (np.linalg.norm(xy[:,:,None,:] - xy[:,None,:,:], axis=-1) - (shuf_size[None,i,:,None]+shuf_size[None,i,None,:])*np.sqrt(2)/2 > 0).reshape([n_samples_over, (n_objects*2)**2])[:,triu_idx].all(1)
        angles = angles[valid][0]
        xy = xy[valid][0]
        rad = rad[valid][0]
        xy_.append(xy)
        rad_.append(rad)
        angles_.append(angles)
    rad = np.stack(rad_, 0)
    xy = np.stack(xy_, 0)
    angles = np.stack(angles_, 0)

    size = np.concatenate([np.ones([n_samples, 1])*0.03, size], 1)
    xy = np.concatenate([xy0, xy], 1) 

    s0 = Shape()

    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
            
            if odd_condition in [1,2] and i==n_samples-1:
                odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects), replace=False)
            
            for j in range(n_objects):
                s = Shape()
                s1 = s.clone()
                s2 = s.clone()

                s1.rotate(angles[i,j])
                s2.rotate(angles[i,n_objects+j])

                # if odd_condition == 1 and i == n_samples-1 and j in odd_samples:
                #     s_ = Shape()
                #     if np.random.rand()>0.5:
                #         s1 = s_
                #     else:
                #         s2 = s_

                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append([s0.clone()] + n_objs + n_objs_f)
            
    else:
        shape = []
        s = Shape()
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
            
            if odd_condition in [1,2] and i==n_samples-1:
                odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects-1), replace=False)
            
            for j in range(n_objects):

                s1 = s.clone()
                s2 = s.clone()

                s1.rotate(angles[i,j])
                s2.rotate(angles[i,n_objects+j])

                # if odd_condition == 1 and i == n_samples-1 and j in odd_samples:
                #     s_ = Shape()
                #     if np.random.rand()>0.5:
                #         s1 = s_
                #     else:
                #         s2 = s_


                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append([s0.clone()] + n_objs + n_objs_f)

    if 'c' in condition:
        color = sample_random_colors(n_samples)
        color = [np.ones([n_objects*2+1, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects*2+1, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
#################################################################################
#################################################################################

#################################################################################
# shape + shape

def task_shape_shape(condition='xyscid'):
        
    n_samples = 4

    n_objects = 2
    
    # image and object parameters

    max_size = 0.9/2
    min_size = max_size/3


    if 'id' in condition:
        shape = []
        for i in range(n_samples-1):
            s = Shape()
            shape += [[s.clone(), s.clone()]]
        shape += [[Shape(), Shape()]]

    else:
        s = Shape()
        shape = [[s.clone(), s.clone()] for _ in range(n_samples-1)] + [[s.clone(), Shape()]]

    if 'r' in condition:
        angle = np.random.rand(n_samples, 2) * 2 * np.pi
        for i in range(n_samples-1):
            shape[i][0].rotate(angle[i,0])
            shape[i][1].rotate(angle[i,1])

    if 'f' in condition:
        for i in range(n_samples-1):
            shape[i][1].flip()

    if 's' in condition:
        size = np.random.rand(n_samples, 2) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones(n_samples, 2)

    if 'xy' in condition:
        co1 = sample_over_range_t(n_samples, np.array([0,1]), size)
        co2 = np.random.rand(n_samples, 2) * (1-size) + size/2
        xy = [co1, co2] if np.random.randint(2)==0 else [co2, co1]
        xy = np.stack(xy, axis=2)

    else:
        x = np.ones(n_samples, 2) * 0.25
        x[:,1] = 1 - x[:,1]
        y = np.ones(n_samples, 2) * 0.5
        xy = np.stack([x,y], axis=2)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*2)
        color = c.reshape([n_samples, 2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([2, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# shape + color

def task_shape_color(condition='scid_hs_v'):
        
    n_samples = 4
    n_objects = 2
    
    max_size = 0.9/2
    min_size = max_size/4

    if 's' in condition:
        size = np.random.rand(n_samples, 2) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = np.ones([n_samples, 2]) * size

    co1 = sample_over_range_t(n_samples, np.array([0,1]), size)
    co2 = np.random.rand(n_samples, 2) * (1-size) + size/2
    xy = [co1, co2] # if np.random.randint(2)==0 else [co2, co1]
    xy = np.stack(xy, axis=2)
    perm = np.random.randint(2, size=n_samples)
    xy[perm==0] = xy[perm==0, :, ::-1]

    h1 = np.random.rand()
    min_diff = 0.15
    h2 = (h1 + (np.random.rand() * (1-min_diff*2) + min_diff) * np.random.choice([-1,1]))%1
    h = np.array([h1, h2])[None,:] * np.ones([n_samples, 2])
    h[-1] = h[-1,::-1]

    if 's_' in condition:
        s = (np.random.rand(n_samples, 2) * 0.6 + 0.4) * np.ones([n_samples, 2])
    else:
        s = np.random.rand() * 0.6 + 0.4
        s = s * np.ones([n_samples, 2])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples, 2) * 0.5 + 0.35)
    else:
        v = np.random.rand() * 0.5 + 0.35
        v = v * np.ones([n_samples, 2])
 
    color = np.stack([h,s,v], 2)

    s1, s2 = Shape(), Shape()
    shape = [[s1.clone(), s2.clone()] for _ in range(n_samples)]

    if 'r' in condition:
        angle = np.random.rand(n_samples, 2) * 2 * np.pi
        for i in range(n_samples-1):
            shape[i][0].rotate(angle[i,0])
            shape[i][1].rotate(angle[i,1])

    if 'f' in condition:
        for i in range(n_samples-1):
            shape[i][1].flip()

    return xy, size, shape, color

def task_shape_color_2(condition='idcs'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.randint(2, 4)
    
    n_objects_samples = [np.random.randint(3, 5, size=n_groups) for i in range(n_samples)]
    n_objects = np.array([n.sum() for n in n_objects_samples])

    n_objects_max = n_objects.max()
    color_idx = [cat_lists([[j]*n_objects_samples[i][j] for j in range(n_groups)])  for i in range(n_samples)]
    
    # make sure shuffling creates an odd one out
    # np.random.shuffle(color_idx[-1])
    color_idx[-1] = np.roll(color_idx[-1], 1)

    max_size = 0.9/np.sqrt(n_objects*4)
    min_size = max_size*2/3
    
    if 's' in condition:
        size = np.random.rand(n_samples, n_objects_max) * (max_size[:,None]-min_size[:,None]) + min_size[:,None]
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = np.ones([n_samples, n_objects_max]) * size[:,None]

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1, :n_objects[i]], n_sample_min=1)
        xy.append(xy_[0])
        
    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            shape_ = []
            for j in range(n_groups):
                s = Shape()
                shape_ += [s.clone() for _ in range(n_objects_samples[i][j])]
            shape.append(shape_)
    else:
        shape = []
        s = [Shape() for i in range(n_groups)]
        for i in range(n_samples):
            shape_ = []
            for j in range(n_groups):
                shape_ += [s[j].clone() for _ in range(n_objects_samples[i][j])]
            shape.append(shape_)

    if 'c' in condition:

        min_diff = 0.15
        # max_idx = max([max(color_idx[i])for i in range(n_samples)])+1
        # h_u = sample_over_range_t(n_samples, np.array([0,1]), min_diff*np.ones([n_samples, max_idx]))
        h_u = sample_over_range_t(n_samples, np.array([0,1]), min_diff*np.ones([n_samples, n_groups]))

        if 's' in condition:
            s = (np.random.rand(n_samples, n_groups) * 0.6 + 0.4)
        else:
            s = np.random.rand() * 0.6 + 0.4
            s = s * np.ones([n_samples, n_groups])
            
        if 'v' in condition:
            v = (np.random.rand(n_samples, n_groups) * 0.5 + 0.35)
        else:
            v = np.random.rand() * 0.5 + 0.35
            v = v * np.ones([n_samples, n_groups])

        color = np.stack([h_u,s,v], 2)
        color = [color[i][color_idx[i]] for i in range(n_samples)]


    return xy, size, shape, color

def task_shape_color_3(condition='idcs'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.randint(2, 4)
    
    n_objects_samples = [np.random.randint(3, 5, size=n_groups) for i in range(n_samples)]
    n_objects = np.array([n.sum() for n in n_objects_samples])

    n_objects_max = n_objects.max()

    color_idx = [cat_lists([list(range(n_objects_samples[i][j])) for j in range(n_groups)])  for i in range(n_samples)]

    # make sure shuffling creates an odd one out
    # np.random.shuffle(color_idx[-1])
    # color_idx[-1] = np.roll(color_idx[-1], 1)
    color_idx[-1][1] = 0
    color_idx[-1][n_objects_samples[-1][0]] = 1

    max_size = 0.9/np.sqrt(n_objects*4)
    min_size = max_size*2/3
    
    if 's' in condition:
        size = np.random.rand(n_samples, n_objects_max) * (max_size[:,None]-min_size[:,None]) + min_size[:,None]
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = np.ones([n_samples, n_objects_max]) * size[:,None]

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
        
    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            shape_ = []
            for j in range(n_groups):
                s = Shape()
                shape_ += [s.clone() for _ in range(n_objects_samples[i][j])]
            shape.append(shape_)
    else:
        shape = []
        s = [Shape() for i in range(n_groups)]
        for i in range(n_samples):
            shape_ = []
            for j in range(n_groups):
                shape_ += [s[j].clone() for _ in range(n_objects_samples[i][j])]
            shape.append(shape_)

    if 'c' in condition:

        min_diff = 0.15
        h_u = sample_over_range_t(n_samples, np.array([0,1]), min_diff*np.ones([n_samples, n_objects.max()]))
        for i in range(n_samples):
            np.random.shuffle(h_u[i])

        if 's' in condition:
            s = (np.random.rand(n_samples, n_objects.max()) * 0.4 + 0.6)
        else:
            s = np.random.rand() * 0.4 + 0.6
            s = s * np.ones([n_samples, n_objects.max()])
            
        if 'v' in condition:
            v = (np.random.rand(n_samples, n_objects.max()) * 0.5 + 0.5)
        else:
            v = np.random.rand() * 0.5 + 0.5
            v = v * np.ones([n_samples, n_objects.max()])

        color = np.stack([h_u,s,v], 2)
        color = [color[i][color_idx[i]] for i in range(n_samples)]

    return xy, size, shape, color

#################################################################################
# shape + rot

def task_shape_rot_1(condition='cs'):

    n_samples = 4

    n_objects = 3
    
    max_size = 0.9/n_objects
    min_size = max_size/2

    min_diff = 1/12

    angles = (np.random.rand(n_samples, 3)*(1 - min_diff*2)+min_diff) * 2*np.pi

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    shape = []
    s1 = Shape()
    s3 = Shape()
    for i in range(n_samples):
        s1_ = s1.clone()
        s3_ = s3.clone()
        if i == n_samples-1:
            s2 = s3.clone()
        else:
            s2 = s1.clone()
        s1_.rotate(angles[i,0])
        s2.rotate(angles[i,1])
        s3_.rotate(angles[i,2])
            
        shape.append([s1_, s2, s3_])

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1]*np.sqrt(2), n_sample_min=1)
        xy.append(xy_[0])
        

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# shape + flip

def task_shape_flip_1(condition='cs'):

    n_samples = 4

    n_objects = 3
    
    max_size = 0.9/n_objects
    min_size = max_size/2

    min_diff = 1/12

    angles = (np.random.rand(n_samples, 3)*(1 - min_diff*2)+min_diff) * 2*np.pi

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    shape = []
    s1 = Shape()
    s3 = Shape()
    for i in range(n_samples):
        s1_ = s1.clone()
        s3_ = s3.clone()
        if i == n_samples-1:
            s2 = s3.clone()
        else:
            s2 = s1.clone()
        # s1_.rotate(angles[i,0])
        s2.flip()
        if np.random.randint(2)==0:
            s2.rotate(np.pi)

        # s2.rotate(angles[i,1])
        
        # s3_.rotate(angles[i,2])
            
        shape.append([s1_, s2, s3_])

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1]*np.sqrt(2), n_sample_min=1)
        xy.append(xy_[0])
        

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# shape + inside

def task_shape_inside(condition='c'):
    
    n_samples = 4
    
    max_size = 0.7
    min_size = max_size/3

    size_a = np.random.rand(n_samples) * (max_size - min_size) + min_size 
    size_b = np.random.rand(n_samples) * (size_a/2.5 - size_a/4) + size_a/4

    done = False
    max_attempts = 10 
    
    range_1 = 1 - size_a[:,None]
    starting_1 = size_a[:,None]/2

    xy1 = np.random.rand(n_samples,2) * range_1 + starting_1 

    s1 = Shape(gap_max=0.06, hole_radius=0.2)
    s2 = Shape(gap_max=0.06, hole_radius=0.2)

    done = False

    for _ in range(max_attempts):
        s1.randomize()
        s2.randomize()
        samples = []
        for i in range(n_samples-1):
            
            samples_i = sample_position_inside_1(s1, s2, size_b[i]/size_a[i])
            samples.append(samples_i)
    
        samples_i = sample_position_inside_1(s2, s1, size_b[-1]/size_a[-1])
        samples.append(samples_i)
            
        if all([len(s)>0 for s in samples]):
            done = True
    
        if done:
            break
    
    if not done:
        return np.zeros([100,100])

    shapes = []
    for i in range(n_samples-1):
        shapes.append([s1.clone(), s2.clone()])
    shapes.append([s2.clone(), s1.clone()])

    shape = shapes
    xy2 = [s[0] for s in samples]
    xy2 = np.array(xy2)*size_a[:,None] + xy1
    
    xy = np.stack([xy1, xy2], axis=1)
    size = np.stack([size_a, size_b], axis=1)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*2)
        color = c.reshape([n_samples, 2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([2, 3]) * color for i in range(n_samples)]

    return xy, size, shape, color

def task_shape_inside_1(condition='sc'):
        
    n_samples = 4
    n_objects = np.random.randint(2,4)
    n_objects_in = np.random.randint(1, n_objects, size=n_samples)
    max_attempts = 20

    max_size = 0.9/np.sqrt(n_objects*2)
    min_size = max_size*2/3

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
    xy = np.stack(xy, 0)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects*2)
        color = c.reshape([n_samples, n_objects*2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects*2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects*2, 3]) * color for i in range(n_samples)]

    shape = []
    xy_in = []
    # size_in = []

    size_in = np.random.rand(n_samples, n_objects) * (size/2 - size/3) + size/3

    for i in range(n_samples):
        shape_ = []
        shape_in = []
        # xy_ = []
        xy_in_ = []
        for j in range(n_objects_in[i]):
            
            if i == n_samples-1:
        
                s1 = Shape(gap_max=0.06, hole_radius=0.2)
                s2 = Shape(gap_max=0.06, hole_radius=0.2)
                
                xy_in_.append(xy[i,j])
            
                shape_.append(s1)
                shape_in.append(s2)
        
            else:
                s1 = Shape(gap_max=0.06, hole_radius=0.2)
            
                xy_in_.append(xy[i,j])
        
                shape_.append(s1)
                shape_in.append(s1.clone())
        
        shape_ += [Shape() for j in range(n_objects_in[i], n_objects)]

        shape.append(shape_ + shape_in)
        xy_in.append(np.array(xy_in_))

    size = np.concatenate([size, size_in], 1)
    xy = [np.concatenate([xy[i], xy_in[i]], 0) for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# shape + contact

def task_shape_contact_2(condition='s'):
        
    n_samples = 4
    # n_objects = 3
    n_objects = 3 * np.ones(n_samples).astype(int)

    max_size = 0.9/np.sqrt(n_objects*2)
    min_size = max_size/3.5

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects[0]) * (max_size[:,None] - min_size[:,None]) + min_size[:,None]
    else:
        size = np.random.rand() * (max_size[0] - min_size[0]) + min_size[0]
        if size*3>0.9:
            size = 0.9/3
        size = size * np.ones([n_samples, n_objects[0]])

    n_obj_connected = n_objects - 1

    shape = []
    for i in range(n_samples):
        s = Shape()
        shapes_ = [s.clone(), s.clone(), Shape()]
        if 'f' in condition:
            shapes_[1].flip()

        if 'r' in condition:
            # shapes_[1].rotate(np.random.rand()*2*np.pi)
            shapes_[1].rotate( (np.random.rand() * (1 - 1/6) + 1/12) * 2*np.pi)

        shape.append(shapes_)

    shape[-1] = [shape[-1][0], shape[-1][2], shape[-1][1]]
    xy = []

    for i in range(n_samples):

        positions, clump_size = sample_contact_many(shape[i][:n_obj_connected[i]], size[i,:n_obj_connected[i]])

        n_free_objs = n_objects[i] - n_obj_connected[i] + 1

        triu_idx = np.triu_indices(n_free_objs, k=1)
        triu_idx = triu_idx[0]*n_free_objs + triu_idx[1]

        size_ = np.ones([n_free_objs, 2])
        size_[0] = clump_size
        size_[1:] = size_[1:] * size[i, n_obj_connected[i]:n_objects[i], None]
        
        n_samples_over = 100
        xy_ = np.random.rand(n_samples_over, n_free_objs, 2) * (1 - size_[None,:,:]) + size_[None,:,:]/2
        valid = (np.abs(xy_[:,:,None,:] - xy_[:,None,:,:]) - (size_[None,:,None,:] + size_[None,None,:,:])/2 > 0).any(3).reshape([n_samples_over, n_free_objs**2])[:, triu_idx].all(1)
        while not valid.any():
            xy_ = np.random.rand(n_samples_over, n_free_objs, 2) * (1 - size_[None,:,:]) + size_[None,:,:]/2
            valid = (np.abs(xy_[:,:,None,:] - xy_[:,None,:,:]) - (size_[None,:,None,:] + size_[None,None,:,:])/2 > 0).any(3).reshape([n_samples_over, n_free_objs**2])[:, triu_idx].all(1)
        xy_ = xy_[valid][0]
        xy0 = xy_[0]
        xy_ = np.concatenate([positions + xy0[None,:], xy_[1:]], 0)

        xy.append(xy_)    

    if 'c' in condition:
        color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        # c = sample_random_colors(n_samples*n_objects)
        # color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects[i], 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects[i], 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

task_shape_contact_3 = lambda: task_shape_contact_2('sc')
task_shape_contact_4 = lambda: task_shape_contact_2('c')

def task_shape_contact_5(condition='sc'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.choice([3,4], replace=False)

    n_objects = np.random.randint(n_groups+1, n_groups*2)

    n_objects_samples = []
    for i in range(n_samples):
        n_objects_samples.append(sample_int_sum_n(n_groups, n_objects, min_v=1))    
    # n_objects_samples.append(sample_int_sum_n(n_groups_odd, n_objects, min_v=1))
    
    max_size = 0.9/np.sqrt(n_objects*4)
    min_size = max_size/2

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]

    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    shape = []

    xy = []
    for i in range(n_samples):
        n_obj_s = n_objects_samples[i]
        # n_g = n_groups_odd if i == n_samples-1 else n_groups
        n_g = n_groups
        xy_ = []
        n = 0
        positions_ = [] 
        clump_size_ = []

        shape_ = []
        for j in range(n_g):
            s = Shape()

            shape_j = [s.clone() for k in range(n_obj_s[j])]

            if 'f' in condition:
                for k in range(n_obj_s[j]):
                    if np.random.randint(2) == 0:
                        shape_j[k].flip()

            if 'r' in condition:
                for k in range(n_obj_s[j]):
                    shape_j[k].rotate(np.random.rand()*2*np.pi)


            # shape_ += shape_j
            shape_.append(shape_j)

        if i == n_samples - 1:
            idx = [k for k in range(n_g) if len(shape_[k])>1][0]
            idx2 = [k for k in range(n_g) if k!=idx][0]
            shape_ex = shape_[idx][0]
            shape_[idx][0] = shape_[idx2][0]
            shape_[idx2][0] = shape_ex


        for j in range(n_g):
        
            # positions, clump_size = sample_contact_many(shape[i][n:n + n_obj_s[j]], size[i, n:n + n_obj_s[j]])
            positions, clump_size = sample_contact_many(shape_[j], size[i, n:n + n_obj_s[j]])
            n = n + n_obj_s[j]
            
            positions_.append(positions)
            clump_size_.append(clump_size)
        
        clump_size = np.stack(clump_size_, 0)*1.1
        xy0 = sample_positions_bb(clump_size[None,:,:], n_sample_min=1)
        xy0 = xy0[0]
        
        for j in range(n_g):
            xy_.append(positions_[j] + xy0[j:j+1,:])

        xy.append(np.concatenate(xy_, 0))
        shape_ = cat_lists(shape_)
        shape.append(shape_)

    return xy, size, shape, color

#################################################################################
# shape + count

def task_shape_count_1(condition='idcs'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.randint(2, 4)

    n_objects_samples = np.random.randint(3, 5, size=n_groups)
    n_objects = n_objects_samples.sum()

    n_objects_samples_odd = sample_int_sum_n(n_groups, n_objects, min_v=1)
    while (np.sort(n_objects_samples_odd) == np.sort(n_objects_samples)).all():
        n_objects_samples_odd = sample_int_sum_n(n_groups, n_objects, min_v=1)

    shape_idx = [[i]*n_objects_samples[i] for i in range(n_groups)]
    shape_idx = np.array(cat_lists(shape_idx)) 
    shape_idx_odd = [[i]*n_objects_samples_odd[i] for i in range(n_groups)]
    shape_idx_odd = np.array(cat_lists(shape_idx_odd)) 
    
    # TODO make sure the intersection of shape_idx and shape_idx_odd is empty 

    max_size = 0.9/np.sqrt(n_objects*4)
    min_size = max_size/2

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            unique_shape = [Shape() for j in range(n_groups)]
            if i == n_samples-1:
                shape.append([unique_shape[idx].clone() for idx in shape_idx_odd])
            else:
                shape.append([unique_shape[idx].clone() for idx in shape_idx])
    else:
        shape = []
        unique_shape = [Shape() for j in range(n_groups)]
        for i in range(n_samples):
            if i == n_samples-1:
                shape.append([unique_shape[idx].clone() for idx in shape_idx_odd])
            else:
                shape.append([unique_shape[idx].clone() for idx in shape_idx])

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
        
    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]

    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, size, shape, color

def task_shape_count_2(condition='idcs'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.choice([2,3,4], replace=False, size=2)

    n_objects = np.random.randint(n_groups.max(), n_groups.max()*2)

    n_groups_odd = n_groups[1]
    n_groups = n_groups[0]

    # n_objects_samples = np.random.randint(2, 4, size=n_groups)
    # n_objects = n_objects_samples.sum()
    n_objects_samples = []
    for i in range(n_samples):
        n_objects_samples.append(sample_int_sum_n(n_groups, n_objects, min_v=1))    
    n_objects_samples.append(sample_int_sum_n(n_groups_odd, n_objects, min_v=1))

    shape_idx = [cat_lists([[j]*n_objects_samples[i][j] for j in range(n_groups)]) for i in range(n_samples-1)]
    shape_idx += [cat_lists([[j]*n_objects_samples[-1][j] for j in range(n_groups_odd)])]
    # shape_idx = cat_lists(shape_idx) 
    
    max_size = 0.9/np.sqrt(n_objects*4)
    min_size = max_size/2

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            if i == n_samples-1:
                unique_shape = [Shape() for j in range(n_groups_odd)]
            else:
                unique_shape = [Shape() for j in range(n_groups)]

            shape.append([unique_shape[idx].clone() for idx in shape_idx[i]])
    else:
        shape = []
        unique_shape = [Shape() for j in range(max(n_groups,n_groups_odd))]
        for i in range(n_samples):
            shape.append([unique_shape[idx].clone() for idx in shape_idx[i]])

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
        
    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]

    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, size, shape, color

#################################################################################
#################################################################################
#################################################################################

#################################################################################
# rotation + rotation

task_rot_rot_1 = lambda: task_shape_shape(condition='xyscidr')

def task_rot_rot_3(condition='xyidsc'):
        
    n_samples = 4

    n_objects = 2
    
    # image and object parameters
    internal_frame = 0.8
    pad = (1-internal_frame)/2
    
    max_size = 0.4
    min_size = max_size/2

    angle_diff_diff = (np.random.rand() * (1-2/3) + 1/3)* np.pi

    angle_diff = (np.random.rand() * (1-1/6) + 1/6) * np.pi
    angle_diff_odd = np.mod(angle_diff + angle_diff_diff * np.random.choice([-1,1]), np.pi)
    angle_diffs = np.ones(n_samples) * angle_diff
    angle_diffs[-1] = angle_diff_odd
    
    angle = np.random.rand(n_samples) * 2 * np.pi
    angle = np.stack([angle, angle + angle_diffs], 1)

    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            s = Shape()
            shape += [[s.clone(), s.clone()]]
    else:
        s = Shape()
        shape = [[s.clone(), s.clone()] for i in range(n_samples)]

    for i in range(n_samples):
        shape[i][0].rotate(angle[i,0])
        shape[i][1].rotate(angle[i,1])

    if 's' in condition:
        size = np.random.rand(n_samples, 2) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones(n_samples, 2)

    if 'xy' in condition:
            
        co1 = sample_over_range_t(n_samples, np.array([0,1]), size)
        co2 = np.random.rand(n_samples, 2) * (1-size) + size/2
        xy = [co1, co2] if np.random.randint(2)==0 else [co2, co1]
        xy = np.stack(xy, axis=2)

    else:
        x = np.ones(n_samples, 2) * 0.25
        x[:,1] = 1 - x[:,1]
        y = np.ones(n_samples, 2) * 0.5
        xy = np.stack([x,y], axis=2)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([2, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# rotation + flip

def task_rot_flip_1(condition='idcs'):

    n_samples = 4

    n_objects = 4
    
    max_size = 0.9/2
    min_size = max_size/2

    min_diff = 1/6

    angles = sample_over_range_t(n_samples, np.array([0,1]), min_diff*np.ones([n_samples,2]) )
    angles = (np.random.rand(n_samples, 2)*(1 - min_diff*2)+min_diff) * 2*np.pi

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    shape = []
    if 'id' not in condition:
        s = Shape()
    # s3 = Shape()
    for i in range(n_samples):
        if 'id' in condition:
            s = Shape()

        s1 = s.clone()
        s2 = s.clone()
        s3 = s.clone()
        s4 = s.clone()

        if i == n_samples-1:
            l = np.random.randint(2)
            if l == 0:
                s3.flip()
                s4.flip()
                
                s2.rotate(angles[i,0])        
                s4.rotate(angles[i,1])
            
            elif l == 1:
                s3.flip()
                # s4.flip()

                s2.rotate(angles[i,0])        
                s4.rotate(angles[i,0])

        else:

            s3.flip()
            s4.flip()
            
            s2.rotate(angles[i,0])        
            s4.rotate(angles[i,0])
                
        shape.append([s1, s2, s3, s4])

    # xy = []
    # for i in range(n_samples):
    #     xy_ = sample_positions(size[i:i+1]*np.sqrt(2), n_sample_min=1)
    #     xy.append(xy_[0])

    xy = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])        
    xy = np.stack([xy]*n_samples, 0)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# rotation + color

def task_rot_color(condition='idcsv'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.randint(2, 4)
    
    n_objects_samples = [np.random.randint(3, 5, size=n_groups) for i in range(n_samples)]
    n_objects = np.array([n.sum() for n in n_objects_samples])

    color_idx = [cat_lists([[j]*n_objects_samples[i][j] for j in range(n_groups)])  for i in range(n_samples)]
    # TODO make sure shuffling creates an odd one out
    np.random.shuffle(color_idx[-1])

    max_size = 0.9/np.sqrt(n_objects*4)
    min_size = max_size*2/3
    
    if 's' in condition:
        size = np.random.rand(n_samples, n_objects.max()) * (max_size[:,None]-min_size[:,None]) + min_size[:,None]
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = np.ones([n_samples, n_objects.max()]) * size[:,None]

    xy = []
    for i in range(n_samples):
        # xy_ = sample_positions(size[i:i+1] * np.sqrt(2), n_sample_min=1)
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
        
    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            shape_ = []
            for j in range(n_groups):
                s = Shape()
                shape_ += [s.clone() for _ in range(n_objects_samples[i][j])]
            shape.append(shape_)
    else:
        shape = []
        s = [Shape() for i in range(n_groups)]
        for i in range(n_samples):
            shape_ = []
            for j in range(n_groups):
                shape += [s[j].clone() for _ in range(n_objects_samples[i][j])]
            shape.append(shape_)
    # add rotations
    for i in range(n_samples):
        for j in range(n_objects[i]):
            shape[i][j].rotate(np.random.choice([0, 1/4, 2/4, 3/4]) * 2 * np.pi)

    # if 'c' in condition:

    min_diff = 0.15
    # max_idx = max([max(color_idx[i])for i in range(n_samples)])
    h_u = sample_over_range_t(n_samples, np.array([0,1]), min_diff*np.ones([n_samples, n_groups]))

    if 's_' in condition:
        s = (np.random.rand(n_samples, n_groups) * 0.6 + 0.4)
    else:
        s = np.random.rand() * 0.6 + 0.4
        s = s * np.ones([n_samples, n_groups])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples, n_groups) * 0.5 + 0.35)
    else:
        v = np.random.rand() * 0.5 + 0.35
        v = v * np.ones([n_samples, n_groups])

    color = np.stack([h_u,s,v], 2)
    color = [color[i][color_idx[i]] for i in range(n_samples)]

    return xy, size, shape, color

#################################################################################
# rotation + inside

def task_rot_inside_1(condition='xysc'):
        
    n_samples = 4
    n_objects = 2
    
    max_attempts = 10

    max_size = 0.9/n_objects
    min_size = max_size/2

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    size_in = np.random.rand(n_samples, 2) * (size/2.5 - size/4) + size/4
    size = np.concatenate([size, size_in], 1)

    if 'xy' in condition:
            
        co1 = sample_over_range_t(n_samples, np.array([0,1]), size[:,:2]*np.sqrt(2))
        co2 = np.random.rand(n_samples, 2) * (1-size[:,:2]*np.sqrt(2)) + size[:,:2]*np.sqrt(2)/2
        xy = [co1, co2]# if np.random.randint(2)==0 else [co2, co1]
        xy = np.stack(xy, axis=2)

        perm = np.random.randint(2, size=n_samples)
        xy[perm==0] = xy[perm==0, :, ::-1]
       
    else:
        x = np.ones(n_samples, 2) * 0.25
        x[:,1] = 1 - x[:,1]
        y = np.ones(n_samples, 2) * 0.5
        xy = np.stack([x,y], axis=2)


    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects*2)
        color = c.reshape([n_samples, n_objects*2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects*2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects*2, 3]) * color for i in range(n_samples)]

    shape = []
    xy_in = []
    # size_in = []

    # angles = np.random.rand(n_samples) * 2 * np.pi
    angles = np.random.choice([0, 1/4, 2/4, 3/4], size=n_samples) * 2 * np.pi
    angle_odd = angles[-1] + (np.random.rand() * (1-0.5) + 0.5) * np.pi

    for i in range(n_samples):
        s1 = Shape(gap_max=0.08, hole_radius=0.2)
        # s2 = Shape(gap_max=0.08, hole_radius=0.2)
        s3 = Shape()
        # s4 = Shape()

        done1, done2 = False, False
        for _ in range(max_attempts):

            samples = sample_position_inside_1(s1, s3, size_in[i,0]/size[i,0])
            if len(samples)>0:
                done1 = True
            
            if done1:
                break
            else:
                s1.randomize()
                s3.randomize()

        if done1:
            xy_ = samples[0]
            xy_in.append(xy_ * size[i,0] + xy[i,0])

            rot_matrix = np.array([np.cos(angles[i]), -np.sin(angles[i]), np.sin(angles[i]), np.cos(angles[i])])

            xy_r = (np.concatenate([xy_,xy_], 0) * rot_matrix).reshape(2,2).sum(-1)
            xy_in.append(xy_r * size[i,1] + xy[i,1])

            s2 = s1.clone()
            s4 = s3.clone()
            
            if i == n_samples-1:
                odd_cond = np.random.choice([0,1])
                if odd_cond == 0:
                    s2.rotate(angles[i])
                    s4.rotate(angle_odd)
                else:
                    s2.rotate(angle_odd)
                    s4.rotate(angles[i])
            else:
                s2.rotate(angles[i])
                s4.rotate(angles[i])

        shape.append([s1, s2, s3, s4])
    
    xy_in = np.stack(xy_in, 0).reshape([n_samples, 2, 2])
    xy = np.concatenate([xy, xy_in], 1)

    return xy, size, shape, color

def task_rot_inside_2(condition='sc'):
    n_samples = 4
    n_objects = np.random.randint(2,4)
    n_objects_in = np.random.randint(1, n_objects, size=n_samples)
    max_attempts = 20

    max_size = 0.9/np.sqrt(n_objects*2)
    min_size = max_size*2/3

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
    xy = np.stack(xy, 0)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects*2)
        color = c.reshape([n_samples, n_objects*2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects*2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects*2, 3]) * color for i in range(n_samples)]

    shape = []
    xy_in = []
    # size_in = []
        
    # angles = np.random.rand(n_samples, n_objects) * 2 * np.pi 
    angles = np.random.choice([0,1/4,2/4,3/4], size=n_samples * n_objects).reshape([n_samples, n_objects]) * 2 * np.pi 
    # angles = np.random.rand(n_samples, n_objects) * 2 * np.pi 

    size_in = np.random.rand(n_samples, n_objects) * (size/2 - size/3) + size/3

    for i in range(n_samples):
        shape_ = []
        shape_in = []
        # xy_ = []
        xy_in_ = []
        for j in range(n_objects_in[i]):
            
            if i == n_samples-1:
        
                s1 = Shape(gap_max=0.06, hole_radius=0.2)
                s2 = Shape(gap_max=0.06, hole_radius=0.2)
            
                xy_in_.append(xy[i,j])
            
                shape_.append(s1)
                shape_in.append(s2)
        
            else:
                s1 = Shape(gap_max=0.06, hole_radius=0.2)
                sr = s1.clone()
                sr.rotate(angles[i,j])
            
                xy_in_.append(xy[i,j])
                
                shape_.append(s1)
                shape_in.append(sr)
        
        shape_ += [Shape() for j in range(n_objects_in[i], n_objects)]

        shape.append(shape_ + shape_in)
        xy_in.append(np.array(xy_in_))

    size = np.concatenate([size, size_in], 1)
    xy = [np.concatenate([xy[i], xy_in[i]], 0) for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# rotation + contact # set 3

task_rot_contact_1 = lambda: task_shape_contact_2('src')
task_rot_contact_2 = lambda: task_shape_contact_5('cr')

#################################################################################
# rotation + count

def task_rot_count_1(condition='idcs'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_objects = np.random.randint(5, 9)

    n_groups = np.random.randint(2, min(n_objects//2+1, 4))

    # n_objects_samples = np.random.randint(2, 4, size=n_groups)
    # n_objects = n_objects_samples.sum()
    n_objects_samples = sample_int_sum_n(n_groups, n_objects, min_v=1)

    n_objects_samples_odd = sample_int_sum_n(n_groups, n_objects, min_v=1)
    while (np.sort(n_objects_samples_odd) == np.sort(n_objects_samples)).all():
        n_objects_samples_odd = sample_int_sum_n(n_groups, n_objects, min_v=1)

    shape_idx = [[i]*n_objects_samples[i] for i in range(n_groups)]
    shape_idx = np.array(cat_lists(shape_idx)) 
    shape_idx_odd = [[i]*n_objects_samples_odd[i] for i in range(n_groups)]
    shape_idx_odd = np.array(cat_lists(shape_idx_odd)) 
    
    # TODO make sure the intersection of shape_idx and shape_idx_odd is empty 

    max_size = 0.9/(np.sqrt(n_objects)*1.5)
    min_size = max_size*2/3

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            unique_shape = [Shape() for j in range(n_groups)]
            if i == n_samples-1:
                shape.append([unique_shape[idx].clone() for idx in shape_idx_odd])
            else:
                shape.append([unique_shape[idx].clone() for idx in shape_idx])
    else:
        shape = []
        unique_shape = [Shape() for j in range(n_groups)]
        for i in range(n_samples):
            if i == n_samples-1:
                shape.append([unique_shape[idx].clone() for idx in shape_idx_odd])
            else:
                shape.append([unique_shape[idx].clone() for idx in shape_idx])

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
    
    angles = np.random.rand(n_samples, n_objects) * 2 * np.pi 
    
    for i in range(n_samples):
        for j in range(n_objects):
            shape[i][j].rotate(angles[i,j])

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]

    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]

    return xy, size, shape, color

#################################################################################
#################################################################################
#################################################################################

#################################################################################
# flip + flip
task_flip_flip_1 = lambda: task_shape_shape(condition='xyscidf')

#################################################################################
# flip + color

def task_flip_color_1(condition='csidv'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.randint(2, 4)
    
    n_objects_samples = [np.random.randint(3, 5, size=n_groups) for i in range(n_samples)]
    n_objects = np.array([n.sum() for n in n_objects_samples])

    color_idx = [cat_lists([[j]*n_objects_samples[i][j] for j in range(n_groups)])  for i in range(n_samples)]
    # TODO make sure shuffling creates an odd one out
    np.random.shuffle(color_idx[-1])

    flips = [np.random.randint(2, size=[n_objects[i]]) for i in range(n_samples)]

    max_size = 0.9/np.sqrt(n_objects*4)
    min_size = max_size*2/3
    
    if 's' in condition:
        size = np.random.rand(n_samples, n_objects.max()) * (max_size[:,None]-min_size[:,None]) + min_size[:,None]
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = np.ones([n_samples, n_objects.max()]) * size[:,None]

    xy = []
    for i in range(n_samples):
        # xy_ = sample_positions(size[i:i+1] * np.sqrt(2), n_sample_min=1)
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
        
    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            shape_ = []
            for j in range(n_groups):
                s = Shape()
                shape_ += [s.clone() for _ in range(n_objects_samples[i][j])]
            shape.append(shape_)
    else:
        shape = []
        s = [Shape() for i in range(n_groups)]
        for i in range(n_samples):
            shape_ = []
            for j in range(n_groups):
                shape += [s[j].clone() for _ in range(n_objects_samples[i][j])]
            shape.append(shape_)

    # add rotations
    for i in range(n_samples):
        for j in range(n_objects[i]):
            if flips[i][j]==1:
                shape[i][j].flip()

    min_diff = 0.15
    # max_idx = max([max(color_idx[i])for i in range(n_samples)])
    h_u = sample_over_range_t(n_samples, np.array([0,1]), min_diff*np.ones([n_samples, n_groups]))

    if 's_' in condition:
        s = (np.random.rand(n_samples, n_groups) * 0.6 + 0.4)
    else:
        s = np.random.rand() * 0.6 + 0.4
        s = s * np.ones([n_samples, n_groups])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples, n_groups) * 0.5 + 0.35)
    else:
        v = np.random.rand() * 0.5 + 0.35
        v = v * np.ones([n_samples, n_groups])

    color = np.stack([h_u,s,v], 2)
    color = [color[i][color_idx[i]] for i in range(n_samples)]

    return xy, size, shape, color


#################################################################################
# flip + inside

def task_flip_inside_1(condition='xysc'):
    
    n_samples = 4
    n_objects = 2
    
    max_attempts = 10

    max_size = 0.7/n_objects
    min_size = max_size/2

    if 's' in condition:
        size = np.random.rand(n_samples, 1) * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    size_in = np.random.rand(n_samples, 1) * (size/2.5 - size/4) + size/4
    # size_in = np.random.rand(n_samples, 2) * (size/2.5 - size/4) + size/4
    size_in = size_in * np.ones([n_samples, n_objects])
    size = np.concatenate([size, size_in], 1)

    if 'xy' in condition:
            
        co1 = sample_over_range_t(n_samples, np.array([0,1]), size[:,:2]*np.sqrt(2))
        co2 = np.random.rand(n_samples, 2) * (1-size[:,:2]*np.sqrt(2)) + size[:,:2]*np.sqrt(2)/2
        xy = [co1, co2]# if np.random.randint(2)==0 else [co2, co1]
        xy = np.stack(xy, axis=2)

        perm = np.random.randint(2, size=n_samples)
        xy[perm==0] = xy[perm==0, :, ::-1]
       
    else:
        x = np.ones(n_samples, 2) * 0.25
        x[:,1] = 1 - x[:,1]
        y = np.ones(n_samples, 2) * 0.5
        xy = np.stack([x,y], axis=2)


    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects*2)
        color = c.reshape([n_samples, n_objects*2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects*2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects*2, 3]) * color for i in range(n_samples)]

    shape = []
    xy_in = []
    # size_in = []

    angles = np.random.rand(n_samples) * np.pi/3 - np.pi/6
    # angles = np.random.choice([0, 1/4, 2/4, 3/4], size=n_samples) * 2 * np.pi
    # angle_odd = angles[-1] + (np.random.rand() * (1-0.5) + 0.5) * np.pi

    for i in range(n_samples):
        s1 = Shape(gap_max=0.08, hole_radius=0.2)
        # s2 = Shape(gap_max=0.08, hole_radius=0.2)
        s3 = Shape()
        # s4 = Shape()

        done1, done2 = False, False
        for _ in range(max_attempts):

            samples = sample_position_inside_1(s1, s3, size_in[i,0]/size[i,0])
            if len(samples)>0:
                done1 = True
            
            if done1:
                break
            else:
                s1.randomize()
                s3.randomize()

        if done1:
            s2 = s1.clone()
            s4 = s3.clone()
            
            if i == n_samples-1:
                xy_ = samples[0]
                xy_in.append(xy_ * size[i,0] + xy[i,0])

                odd_cond = np.random.choice([0,1])

                if odd_cond == 0:
                    rot_matrix = np.array([np.cos(angles[i]), -np.sin(angles[i]), np.sin(angles[i]), np.cos(angles[i])])

                    xy_r = (np.concatenate([xy_,xy_], 0) * rot_matrix).reshape(2,2).sum(-1)
                    
                    xy_in.append(xy_r * size[i,1] + xy[i,1])

                    s2.rotate(angles[i])
                    s4.flip()

                else:
                    xy_r = np.copy(xy_)
                    # xy_r[0] = 1-xy_r[0]
                    xy_r[0] = - xy_r[0]
                    
                    xy_in.append(xy_r * size[i,1] + xy[i,1])

                    s4.rotate(angles[i])
                    s2.flip() 
                    # s4.flip()

                    
            else:
                xy_ = samples[0]
                xy_in.append(xy_ * size[i,0] + xy[i,0])

                xy_r = np.copy(xy_)
                xy_r[0] = -xy_r[0]
                
                xy_in.append(xy_r * size[i,1] + xy[i,1])

                s2.flip() 
                s4.flip()
                
        shape.append([s1, s2, s3, s4])
    
    xy_in = np.stack(xy_in, 0).reshape([n_samples, 2, 2])
    xy = np.concatenate([xy, xy_in], 1)

    return xy, size, shape, color

def task_flip_inside_2(condition='cs'):
    n_samples = 4
    n_objects = np.random.randint(3,5)
    n_objects_in = np.random.randint(1, n_objects, size=n_samples)
    max_attempts = 20

    max_size = 0.9/np.sqrt(n_objects*2)
    min_size = max_size*2/3

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
    xy = np.stack(xy, 0)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects*2)
        color = c.reshape([n_samples, n_objects*2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects*2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects*2, 3]) * color for i in range(n_samples)]

    shape = []
    xy_in = []

    angles = np.random.rand(n_samples, n_objects) * np.pi/3 - np.pi/6

    size_in = np.random.rand(n_samples, n_objects) * (size/2 - size/3) + size/3

    for i in range(n_samples):
        shape_ = []
        shape_in = []
        # xy_ = []
        xy_in_ = []
        for j in range(n_objects_in[i]):
            
            if i == n_samples-1:
        
                s1 = Shape(gap_max=0.06, hole_radius=0.2)
                sr = s1.clone()
                sr.rotate(angles[i,j])
                # s2 = Shape(gap_max=0.06, hole_radius=0.2)
                
                xy_in_.append(xy[i,j])
            
                shape_.append(s1)
                shape_in.append(sr)
        
            else:
                s1 = Shape(gap_max=0.06, hole_radius=0.2)
                sf = s1.clone()
                sf.flip()
                
                xy_in_.append(xy[i,j])
                
                shape_.append(s1)
                shape_in.append(sf)
        
        shape_ += [Shape() for j in range(n_objects_in[i], n_objects)]

        shape.append(shape_ + shape_in)
        xy_in.append(np.array(xy_in_))

    size = np.concatenate([size, size_in], 1)
    xy = [np.concatenate([xy[i], xy_in[i]], 0) for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# flip + contact

task_flip_contact_1 = lambda: task_shape_contact_2('sfc')
task_flip_contact_2 = lambda: task_shape_contact_5('cf')

#################################################################################
# flip + count

def task_flip_count_1(condition='csid'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_objects = np.random.randint(5, 9)

    n_groups = np.random.randint(2, min(n_objects//2+1, 4))

    # n_objects_samples = np.random.randint(2, 4, size=n_groups)
    # n_objects = n_objects_samples.sum()
    n_objects_samples = sample_int_sum_n(n_groups, n_objects, min_v=1)

    n_objects_samples_odd = sample_int_sum_n(n_groups, n_objects, min_v=1)
    while (np.sort(n_objects_samples_odd) == np.sort(n_objects_samples)).all():
        n_objects_samples_odd = sample_int_sum_n(n_groups, n_objects, min_v=1)

    shape_idx = [[i]*n_objects_samples[i] for i in range(n_groups)]
    shape_idx = np.array(cat_lists(shape_idx)) 
    shape_idx_odd = [[i]*n_objects_samples_odd[i] for i in range(n_groups)]
    shape_idx_odd = np.array(cat_lists(shape_idx_odd)) 
    
    # TODO make sure the intersection of shape_idx and shape_idx_odd is empty 

    max_size = 0.9/(np.sqrt(n_objects)*2)
    min_size = max_size*2/3

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            unique_shape = [Shape() for j in range(n_groups)]
            if i == n_samples-1:
                shape.append([unique_shape[idx].clone() for idx in shape_idx_odd])
            else:
                shape.append([unique_shape[idx].clone() for idx in shape_idx])
    else:
        shape = []
        unique_shape = [Shape() for j in range(n_groups)]
        for i in range(n_samples):
            if i == n_samples-1:
                shape.append([unique_shape[idx].clone() for idx in shape_idx_odd])
            else:
                shape.append([unique_shape[idx].clone() for idx in shape_idx])

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
        
    # angles = np.random.rand(n_samples, n_objects) * 2 * np.pi 
    flips = np.random.randint(2, size=[n_samples, n_objects]) 
    
    for i in range(n_samples):
        for j in range(n_objects):
            if flips[i,j] == 0:
                shape[i][j].flip()

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]

    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, size, shape, color

#################################################################################
#################################################################################
#################################################################################

#################################################################################
# color + color # set 2

def task_color_color_1(condition='xysid_hs-v'):
    a = np.random.randint(3)
    variable = 'h'
    # if a==0:
        # variable = 'h'
    # elif a==1:
    #     variable = 's'
    # else:
    #     variable = 'v'


    n_samples = 4

    # image and object parameters

    max_size = 0.9/2
    min_size = max_size/5

    if variable == 'h':
        h = np.random.rand(n_samples)[:,None] * np.ones([n_samples, 2])
        min_diff = 0.15
        h[-1, 1] = (h[-1, 1] + (np.random.rand() * (1-min_diff*2) + min_diff) * np.random.choice([-1,1]))%1
    elif 'h' in condition:
        h = np.random.rand(n_samples, 2)
    else:
        h = np.random.rand()
        h = h * np.ones([n_samples, 2])

    if variable == 's':
        s = (np.random.rand(n_samples)[:,None] * 0.5 + 0.5) * np.ones([n_samples, 2])
        min_diff = 0.15
        s[-1, 1] = (s[-1, 1] + (np.random.rand() * (1-min_diff*2) + min_diff + 0.2) * np.random.choice([-1,1]))%1
    elif 's-' in condition:
        s = (np.random.rand(n_samples, 2) * 0.5 + 0.5) * np.ones([n_samples, 2])
    else:
        s = np.random.rand() * 0.5 + 0.5
        s = s * np.ones([n_samples, 2])
        
    if variable == 'v':
        v = (np.random.rand(n_samples)[:,None] * 0.5 + 0.35) * np.ones([n_samples, 2])
        min_diff = 0.15
        v[-1, 1] = (v[-1, 1] + (np.random.rand() * (0.5-min_diff*2) + min_diff + 0.5) * np.random.choice([-1,1]))%1
    elif 'v' in condition:
        v = (np.random.rand(n_samples, 2) * 0.5 + 0.35)
    else:
        v = np.random.rand() * 0.5 + 0.35
        v = v * np.ones([n_samples, 2])
 

    color = np.stack([h,s,v], 2)

    if 'id' in condition:
        shape = [[Shape(), Shape()] for i in range(n_samples)]

    else:
        s = Shape()
        shape = [[s.clone(), s.clone()] for _ in range(n_samples)]

    if 'r' in condition:
        angle = np.random.rand(n_samples, 2) * 2 * np.pi
        for i in range(n_samples-1):
            shape[i][0].rotate(angle[i,0])
            shape[i][1].rotate(angle[i,1])

    if 'f' in condition:
        for i in range(n_samples-1):
            shape[i][1].flip()

    if 's' in condition:
        size = np.random.rand(n_samples, 2) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones(n_samples, 2)

    if 'xy' in condition:
            
        co1 = sample_over_range_t(n_samples, np.array([0,1]), size)
        co2 = np.random.rand(n_samples, 2) * (1-size) + size/2
        xy = [co1, co2] if np.random.randint(2)==0 else [co2, co1]
        xy = np.stack(xy, axis=2)
        # perm = np.stack([np.arange(n_samples), np.random.randint(2,size=n_samples)], 1)
        # perm = np.random.randint(2, size=n_samples)
        # xy[perm==0] = xy[perm==0, ::-1]
                
    else:
        x = np.ones(n_samples, 2) * 0.25
        x[:,1] = 1 - x[:,1]
        y = np.ones(n_samples, 2) * 0.5
        xy = np.stack([x,y], axis=2)

    return xy, size, shape, color

def task_color_color_2(condition='nxysid_hs-v'):
    a = np.random.randint(3)
    variable = 'h'
    # if a==0:
    #     variable = 'h'
    # elif a==1:
    #     variable = 's'
    # else:
    #     variable = 'v'
 
    n_samples = 4
    
    if 'n' in condition:
        n_objects = np.random.randint(5, 9, size=n_samples)
    else:
        n_objects = np.random.randint(5, 9)
        n_objects = np.ones(n_samples).astype(int) * n_objects
    
    n_objects_max = n_objects.max()

    if variable == 'h':
        h = np.random.rand(n_samples)[:,None] * np.ones([n_samples, n_objects_max])
        n_objects_odd = np.random.randint(1, n_objects[-1])
        min_diff = 0.15
        h[-1, :n_objects_odd] = (h[-1, :n_objects_odd] + (np.random.rand(n_objects_odd) * (1 - min_diff*2) + min_diff) * np.random.choice([-1,1], size=n_objects_odd))%1

    elif 'h' in condition:
        h = np.random.rand(n_samples, n_objects_max)
    else:
        h = np.random.rand()
        h = h * np.ones([n_samples, n_objects_max])

    if variable == 's':
        s = (np.random.rand(n_samples)[:,None] * 0.7 + 0.3) * np.ones([n_samples, n_objects_max])
        n_objects_odd = np.random.randint(1, n_objects[-1])
        min_diff = 0.15
        s[-1, :n_objects_odd] = (s[-1, :n_objects_odd] + (np.random.rand(n_objects_odd) * (0.7-min_diff*2) + min_diff + 0.3) * np.random.choice([-1,1], size=n_objects_odd))%1
        
    elif 's-' in condition:
        s = (np.random.rand(n_samples, n_objects_max) * 0.7 + 0.3) * np.ones([n_samples, n_objects_max])
    else:
        s = np.random.rand() * 0.7 + 0.3
        s = s * np.ones([n_samples, n_objects_max])
        
    if variable == 'v':
        v = (np.random.rand(n_samples)[:,None] * 0.5 + 0.35) * np.ones([n_samples, n_objects_max])
        n_objects_odd = np.random.randint(1, n_objects[-1])
        min_diff = 0.15
        v[-1, :n_objects_odd] = (v[-1, :n_objects_odd] + (np.random.rand(n_objects_odd) * (0.5-min_diff*2) + min_diff + 0.5) * np.random.choice([-1,1], size=n_objects_odd))%1

    elif 'v' in condition:
        v = (np.random.rand(n_samples, n_objects_max) * 0.5 + 0.35)
    else:
        v = np.random.rand() * 0.5 + 0.35
        v = v * np.ones([n_samples, n_objects_max])
 

    color = np.stack([h,s,v], 2)

    if 'id' in condition:
        shape = [[Shape() for j in range(n_objects[i])] for i in range(n_samples)]

    else:
        s = Shape()
        shape = [[s.clone() for j in range(n_objects[i])] for i in range(n_samples)]

    if 'r' in condition:
        angle = np.random.rand(n_samples, n_objects_max) * 2 * np.pi
        for i in range(n_samples-1):
            for j in range(n_objects[i]):
                shape[i][j].rotate(angle[i,j])
                # shape[i][1].rotate(angle[i,1])

    if 'f' in condition:
        for i in range(n_samples-1):
            for j in range(n_objects[j]):
                if np.random.rand()>0.5:
                    shape[i][j].flip()
    
    if 's' in condition:
        max_size = 0.9/np.sqrt(n_objects*2)
        min_size = max_size/3

        size = (np.random.rand(n_samples, n_objects_max) * (max_size[:,None] - min_size[:,None]) + min_size[:,None]) * np.ones([n_samples, n_objects_max])
    else:
        max_size = 0.9/np.sqrt(n_objects_max*2)
        min_size = max_size/3

        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects_max])
    
    if 'xy' in condition:        
        xy = []
        for i in range(n_samples):
            xy_ = sample_positions(size[i:i+1, :n_objects[i]], n_sample_min=1)
            xy.append(xy_[0])
        
    else:
        xy = sample_positions(size.max(0)[None,:], n_sample_min=n_samples)

    xy = [xy[i][:n_objects[i]] for i in range(n_samples)]
    size = [size[i][:n_objects[i]] for i in range(n_samples)]
    shape = [shape[i][:n_objects[i]] for i in range(n_samples)]
    color = [color[i][:n_objects[i]] for i in range(n_samples)]

    return xy, size, shape, color


#################################################################################
# color + inside

def task_color_inside_1(condition='xysid_hs-v'):
    
    n_samples = 4
    n_objects = 1
    
    max_attempts = 10

    max_size = 0.7/n_objects
    min_size = max_size/2

    if 's' in condition:
        size = np.random.rand(n_samples) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones(n_samples)

    if 'xy' in condition:
        xy = np.random.rand(n_samples, 2) * (1 - size[:, None]) + size[:, None]/2
       
    else:
        xy = np.random.rand(2) * (1 - size[:, None]) + size[:, None]/2
        xy = np.ones([n_samples, 2]) * xy[None,:] 

    h1 = np.random.rand()
    min_diff = 0.15
    h2 = (h1 + (np.random.rand() * (1-min_diff*2) + min_diff) * np.random.choice([-1,1]))%1
    h = np.array([h1, h2])[None,:] * np.ones([n_samples, 2])
    h[-1] = h[-1,::-1]

    if 's-' in condition:
        s = (np.random.rand(n_samples, 2) * 0.6 + 0.4) * np.ones([n_samples, 2])
    else:
        s = np.random.rand() * 0.6 + 0.4
        s = s * np.ones([n_samples, 2])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples, 2) * 0.5 + 0.35)
    else:
        v = np.random.rand() * 0.5 + 0.35
        v = v * np.ones([n_samples, 2])
 
    color = np.stack([h,s,v], 2)


    size_in = np.random.rand(n_samples) * (size/2.5 - size/4) + size/4
    size = np.stack([size, size_in], 1)

    shape = []
    xy_in = []
    # size_in = []

    for i in range(n_samples):
        s1 = Shape(gap_max=0.08, hole_radius=0.2)
        s2 = Shape()
        done = False
        for _ in range(max_attempts):

            samples = sample_position_inside_1(s1, s2, size[i,1]/size[i,0])
            if len(samples)>0:
                done = True
            
            if done:
                break
            else:
                s1.randomize()
                s2.randomize()

        if done:
            xy_ = samples[0]
            xy_in.append(xy_ * size[i,0] + xy[i])
    
        shape.append([s1, s2])
    
    xy_in = np.array(xy_in)
    xy = np.stack([xy, xy_in], 1)

    return xy, size, shape, color

def task_color_inside_2(condition='xysid_hs-v'):
    
    n_samples = 4
    n_objects = np.random.randint(4,9)
    n_objects_in = np.random.randint(1, n_objects, size=n_samples)
    max_attempts = 10

    max_size = 0.9/np.sqrt(n_objects*2)
    min_size = max_size/2

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
    xy = np.stack(xy, 0)
    
    if 'h' in condition:
        h = np.random.rand(n_samples+1, n_objects)
    else:
        h = np.random.rand()
        h = h * np.ones([n_samples+1, n_objects])

    if 's' in condition:
        s = (np.random.rand(n_samples+1, n_objects) * 0.6 + 0.4)
    else:
        s = np.random.rand() * 0.6 + 0.4
        s = s * np.ones([n_samples+1, n_objects])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples+1, n_objects) * 0.5 + 0.35)
    else:
        v = np.random.rand() * 0.5 + 0.35
        v = v * np.ones([n_samples+1, n_objects])
    
    # might create some incorrect examples
    color = np.stack([h,s,v], 2)
    color = np.concatenate([color]*2, 1)
    color[-2, n_objects:] = color[-1, n_objects:]
    color = color[:-1]

    shape = []
    xy_in = []
    # size_in = []

    size_in = np.random.rand(n_samples, n_objects) * (size/2.5 - size/4) + size/4
    for i in range(n_samples):
        shape_ = []
        shape_in = []
        # xy_ = []
        xy_in_ = []
        for j in range(n_objects_in[i]):
        
            s1 = Shape(gap_max=0.06, hole_radius=0.2)
            s2 = Shape()
            done = False
            for _ in range(max_attempts):
                samples = sample_position_inside_1(s1, s2, size_in[i,j]/size[i,j])
                if len(samples)>0:
                    done = True
                if done:
                    break
                else:
                    s1.randomize()

            if done:
                xy_ = samples[0]
                xy_in_.append(xy_ * size[i,j] + xy[i,j])
        
            shape_.append(s1)
            shape_in.append(s2)
    
        shape_ += [Shape() for i in range(n_objects_in[i], n_objects)]
        shape.append(shape_ + shape_in)
        xy_in.append(np.array(xy_in_))

    size = np.concatenate([size, size_in], 1)
    xy = [np.concatenate([xy[i], xy_in[i]], 0) for i in range(n_samples)]

    return xy, size, shape, color

#################################################################################
# color + contact

def task_color_contact(condition='sids-v'):
        
    n_samples = 4
    n_objects = 4
    
    max_size = 0.9/np.sqrt(n_objects*2)
    min_size = max_size/2

    # a=b c=d a=c 1
    # a=b c=d a!c 2
    # a!b c!d a=c b=d 2
    # a=b c!d a!c 3
    # a!b c!d a=d 3
    # a!c c!d d!b 4 smaller-bigger small-big

    perm = np.array([    
        [0,0,0,0],
        [0,0,1,1],
        [0,1,0,1],
        [2,2,1,0],
        [0,2,2,1],
        [0,1,2,3],
    ])

    all_conditions = [0,1,2,3,4,5]
    cond_reg, cond_odd = np.random.choice(all_conditions, replace=False, size=2)
    
    n_unique_colors = [1,2,2,3,3,4]
    n_uniques_colors_reg = n_unique_colors[cond_reg]
    n_uniques_colors_odd = n_unique_colors[cond_odd]


    if n_uniques_colors_reg>1:
        min_diff = 0.15
        u_h_reg = sample_over_range_t(n_samples-1, np.array([0,1]), min_diff*np.ones([n_samples-1, n_uniques_colors_reg]))
    else:
        u_h_reg = np.random.rand(n_samples-1)[:,None]

    if n_uniques_colors_odd>1:
        min_diff = 0.15
        u_h_odd = sample_over_range_t(1, np.array([0,1]), min_diff*np.ones([1, n_uniques_colors_odd]))
    else:
        u_h_odd = np.random.rand(1) * (max_size - min_size) + min_size
        u_h_odd = np.ones([1,1]) * u_h_odd

    perm_reg = perm[cond_reg]
    perm_odd = perm[cond_odd]

    h = [u_h_reg[i, perm_reg] for i in range(n_samples-1)]
    h += [u_h_odd[0, perm_odd]]
    h = np.stack(h, 0)

    if 's-' in condition:
        s = (np.random.rand(n_samples, n_objects) * 0.4 + 0.6)
    else:
        s = np.random.rand() * 0.4 + 0.6
        s = s * np.ones([n_samples, n_objects])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples, n_objects) * 0.5 + 0.5)
    else:
        v = np.random.rand() * 0.5 + 0.5
        v = v * np.ones([n_samples, n_objects])
    
    color = np.stack([h,s,v], 2)
    

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])


    size = np.stack(size, 0)

    if 'id' in condition:
        shape = [[s.clone() for i in range(n_objects)] for s in [Shape() for i in range(n_samples)]]

    else:
        s = Shape()
        shape = [[s.clone() for i in range(n_objects)] for _ in range(n_samples)]

    if 'r' in condition:
        angle = np.random.rand(n_samples, n_objects) * 2 * np.pi
        for i in range(n_samples-1):
            for j in range(n_objects):
                shape[i][j].rotate(angle[i,j])

    if 'f' in condition:
        for i in range(n_samples):
            for j in range(n_objects):
                if np.random.randint(2) == 0:
                    shape[i][j].flip()

    xy = []
    for i in range(n_samples):
            
        s1 = shape[i][0]
        s2 = shape[i][1]
        s3 = shape[i][2]
        s4 = shape[i][3]

        positions1, clump_size1 = sample_contact_many([s1, s2], size[i,:2])
        positions2, clump_size2 = sample_contact_many([s3, s4], size[i,2:])
        c_size = np.stack([clump_size1, clump_size2], 0)
        co1 = sample_over_range_t(1, np.array([0,1]), c_size[None,:,0])
        co2 = np.random.rand(1, 2) * (1-c_size[None,:,1]) + c_size[None,:,1]/2
        xy_ = [co1, co2]# if np.random.randint(2)==0 else [co2, co1]
        xy_ = np.stack(xy_, axis=2)[0]

        xy_1 = positions1 + xy_[0:1,:]
        xy_2 = positions2 + xy_[1:,:]
        xy_ = np.concatenate([xy_1, xy_2], 0)
        xy.append(xy_)

    xy = np.stack(xy, 0)

    return xy, size, shape, color

#################################################################################
# color + count

def task_color_count_1(condition='csids-v'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.randint(2, 5)

    n_objects_samples = np.random.randint(3, 5, size=n_groups)
    n_objects = n_objects_samples.sum()

    n_objects_samples_odd = sample_int_sum_n(n_groups, n_objects, min_v=1)
    while (n_objects_samples_odd == n_objects_samples).all():
        n_objects_samples_odd = sample_int_sum_n(n_groups, n_objects, min_v=1)

    color_idx = [[i]*n_objects_samples[i] for i in range(n_groups)]
    color_idx = np.array(cat_lists(color_idx)) 
    color_idx_odd = [[i]*n_objects_samples_odd[i] for i in range(n_groups)]
    color_idx_odd = np.array(cat_lists(color_idx_odd)) 
    
    
    max_size = 0.9/np.sqrt(n_objects*4)
    min_size = max_size/2
    
    min_diff = 0.15

    h_u = sample_over_range_t(n_samples, np.array([0,1]), np.ones([n_samples, n_groups])*min_diff)

    if 's-' in condition:
        s = (np.random.rand(n_samples, n_groups) * 0.4 + 0.6)
    else:
        s = np.random.rand() * 0.4 + 0.6
        s = s * np.ones([n_samples, n_groups])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples, n_groups) * 0.5 + 0.5)
    else:
        v = np.random.rand() * 0.5 + 0.5
        v = v * np.ones([n_samples, n_groups])

    color_u = np.stack([h_u,s,v], 2)

    color = [color_u[i,color_idx] for i in range(n_samples-1)] + [color_u[-1,color_idx_odd]]

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    if 'id' in condition:
        shape = [[Shape() for j in range(n_objects)] for i in range(n_samples)]

    else:
        s = Shape()
        shape = [[s.clone() for j in range(n_objects)] for i in range(n_samples)]

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
        
    return xy, size, shape, color

def task_color_count_2(condition='csids-v'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.choice([2,3,4,5], replace=False, size=2)

    n_objects = np.random.randint(n_groups.max(), n_groups.max()*2)

    n_groups_odd = n_groups[1]
    n_groups = n_groups[0]

    # n_objects_samples = np.random.randint(2, 4, size=n_groups)
    # n_objects = n_objects_samples.sum()
    n_objects_samples = []
    for i in range(n_samples-1):
        n_objects_samples.append(sample_int_sum_n(n_groups, n_objects, min_v=1))    
    n_objects_samples.append(sample_int_sum_n(n_groups_odd, n_objects, min_v=1))

    color_idx = [cat_lists([[j]*n_objects_samples[i][j] for j in range(n_groups)]) for i in range(n_samples-1)]
    color_idx += [cat_lists([[j]*n_objects_samples[-1][j] for j in range(n_groups_odd)])]
    # shape_idx = cat_lists(shape_idx) 
    
    max_size = 0.9/np.sqrt(n_objects*3)
    min_size = max_size/2

    min_diff = 0.15

    h_u = sample_over_range_t(n_samples-1, np.array([0,1]), np.ones([n_samples-1, n_groups])*min_diff)
    h_u_odd = sample_over_range_t(1, np.array([0,1]), np.ones([1, n_groups_odd])*min_diff)

    if 's-' in condition:
        s = (np.random.rand(n_samples, max(n_groups, n_groups_odd)) * 0.4 + 0.6)
    else:
        s = np.random.rand() * 0.4 + 0.6
        s = s * np.ones([n_samples, max(n_groups, n_groups_odd)])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples, max(n_groups, n_groups_odd)) * 0.5 + 0.5)
    else:
        v = np.random.rand() * 0.5 + 0.5
        v = v * np.ones([n_samples, max(n_groups, n_groups_odd)])

    color_u = np.stack([h_u,s[:n_samples-1, :n_groups],v[:n_samples-1, :n_groups]], 2)
    color_u_odd = np.stack([h_u_odd,s[-1:, :n_groups_odd],v[-1:, :n_groups_odd]], 2)

    color = [color_u[i, color_idx[i]] for i in range(n_samples-1)] + [color_u_odd[0, color_idx[-1]]]

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    xy = []
    for i in range(n_samples):
        xy_ = sample_positions(size[i:i+1], n_sample_min=1)
        xy.append(xy_[0])
        
    if 'id' in condition:
        shape = [[Shape() for j in range(n_objects)] for i in range(n_samples)]

    else:
        s = Shape()
        shape = [[s.clone() for j in range(n_objects)] for i in range(n_samples)]

    return xy, size, shape, color

#################################################################################
#################################################################################
#################################################################################

#################################################################################
# inside + inside # set 1

def task_inside_inside_1(condition='c'):
       
    n_samples = 4

    max_attempts = 10 

    xy = []
    shapes = []
    sizes = []
    conditions_r = [1,2]
    conditions_o = [3,4,5,6]
    sampled_conditions = np.random.choice(conditions_r, size=n_samples)
    sampled_conditions[-1] = np.random.choice(conditions_o) 

    for i in range(n_samples):
        done = False

        # conditions:
        # (+)
        # 1- a(b(c))
        # 2- a(b,c)
        # (-)
        # 1- a, b(c)
        # 2- a(b),c
        # 3- a(c),b  
        # 4- a,b,c  
                
        for _ in range(max_attempts):
            
            # a(b(c))
            if sampled_conditions[i] == 1:
                max_size = 0.7
                min_size = max_size/2

                size_a = np.random.rand() * (max_size - min_size) + min_size 
                size_b = np.random.rand() * (size_a/2.5 - size_a/4) + size_a/4
                size_c = np.random.rand() * (size_b/2.5 - size_b/4) + size_b/4

                sa = Shape(gap_max=0.08, hole_radius=0.2)
                sb = Shape(gap_max=0.08, hole_radius=0.2)
                sc = Shape()

                samples_c = sample_position_inside_many(sb, [sc], [size_c/size_b])
                while len(samples_c)==0:
                    sb.randomize()
                    samples_c = sample_position_inside_many(sb, [sc], [size_c/size_b])

                samples_b = sample_position_inside_many(sa, [sb], [size_b/size_a])
                while len(samples_b)==0:
                    sa.randomize()
                    samples_b = sample_position_inside_many(sa, [sb], [size_b/size_a])
                
                done = True

                xy_b = samples_b[0,0]
                xy_c = samples_c[0,0]

                range_1 = 1 - size_a
                starting_1 = size_a/2
                xy_a = np.random.rand(2) * range_1 + starting_1
                xy_b = xy_b * size_a + xy_a
                xy_c = xy_c * size_b + xy_b

            # a(b,c)
            elif sampled_conditions[i] == 2:
                max_size = 0.7
                min_size = max_size/2

                size_a = np.random.rand() * (max_size - min_size) + min_size 
                size_b = np.random.rand() * (size_a/2.5 - size_a/4) + size_a/4
                size_c = np.random.rand() * (size_b/2.5 - size_b/4) + size_b/4

                sa = Shape(gap_max=0.08, hole_radius=0.2)
                sb = Shape()
                sc = Shape()
                
                samples_bc = sample_position_inside_many(sa, [sb, sc], [size_b/size_a, size_c/size_a])
                while len(samples_bc)==0:
                    sa.randomize()
                    sb.randomize()
                    sc.randomize()
                    samples_bc = sample_position_inside_many(sa, [sb, sc], [size_b/size_a, size_c/size_a])
                
                done = True

                xy_b = samples_bc[0,0]
                xy_c = samples_bc[0,1]

                range_1 = 1 - size_a
                starting_1 = size_a/2
                xy_a = np.random.rand(2) * range_1 + starting_1
                xy_b = xy_b * size_a + xy_a
                xy_c = xy_c * size_a + xy_a

            # a, b(c)
            elif sampled_conditions[i] == 3:
                max_size = 0.7
                min_size = max_size/2

                size_a = np.random.rand() * (max_size - min_size) + min_size 
                size_b = np.random.rand() * (size_a/2.5 - size_a/4) + size_a/4
                size_c = np.random.rand() * (size_b/2.5 - size_b/4) + size_b/4

                sa = Shape(gap_max=0.08, hole_radius=0.2)
                sb = Shape(gap_max=0.08, hole_radius=0.2)
                sc = Shape()
                
                samples_c = sample_position_inside_many(sb, [sc], [size_c/size_b])
                while len(samples_c)==0:
                    sb.randomize()
                    sc.randomize()
                    samples_c = sample_position_inside_many(sb, [sc], [size_c/size_b])
                
                done = True

                xy_c = samples_c[0,0]

                range_ab = 1 - np.array([size_a, size_b])
                starting_ab = np.array([size_a, size_b])/2

                xy_ab = np.random.rand(10, 2, 2) * range_ab[None,:,None] + starting_ab[None,:,None]
                no_overlap = (np.abs(xy_ab[:,0]-xy_ab[:,1]) - (size_a + size_b)/2 > 0).any(1)
                while not no_overlap.any(0):
                    xy_ab = np.random.rand(10, 2, 2) * range_ab[None,:,None] + starting_ab[None,:,None]
                    no_overlap = (np.abs(xy_ab[:,0]-xy_ab[:,1]) - (size_a + size_b)/2 > 0).any(1)

                xy_ab = xy_ab[no_overlap]

                xy_a, xy_b = xy_ab[0, 0], xy_ab[0, 1]
                xy_c = xy_c * size_b + xy_b

            # a(b),c
            elif sampled_conditions[i] == 4:
                max_size = 0.7
                min_size = max_size/2

                size_a = np.random.rand() * (max_size - min_size) + min_size 
                size_b = np.random.rand() * (size_a/2.5 - size_a/4) + size_a/4
                size_c = np.random.rand() * (size_b/2.5 - size_b/4) + size_b/4

                sa = Shape(gap_max=0.08, hole_radius=0.2)
                sb = Shape(gap_max=0.08, hole_radius=0.2)
                sc = Shape()

                samples_b = sample_position_inside_many(sa, [sb], [size_b/size_a])
                while len(samples_b)==0:
                    sa.randomize()
                    sb.randomize()
                    samples_b = sample_position_inside_many(sa, [sb], [size_b/size_a])
                
                done = True

                xy_b = samples_b[0,0]

                range_ac = 1 - np.array([size_a, size_c])
                starting_ac = np.array([size_a, size_c])/2

                xy_ac = np.random.rand(10, 2, 2) * range_ac[None,:,None] + starting_ac[None,:,None]
                no_overlap = (np.abs(xy_ac[:,0] - xy_ac[:,1]) - (size_a + size_c)/2 > 0).any(1)
                while not no_overlap.any(0):
                    xy_ac = np.random.rand(10, 2, 2) * range_ac[None,:,None] + starting_ac[None,:,None]
                    no_overlap = (np.abs(xy_ac[:,0] - xy_ac[:,1]) - (size_a + size_c)/2 > 0).any(1)

                xy_ac = xy_ac[no_overlap]

                xy_a, xy_c = xy_ac[0, 0], xy_ac[0, 1]
                xy_b = xy_b * size_a + xy_a

            # a(c),b
            elif sampled_conditions[i] == 5:
                max_size = 0.7
                min_size = max_size/2

                size_a = np.random.rand() * (max_size - min_size) + min_size 
                size_b = np.random.rand() * (size_a/2.5 - size_a/4) + size_a/4
                size_c = np.random.rand() * (size_b/2.5 - size_b/4) + size_b/4

                sa = Shape(gap_max=0.08, hole_radius=0.2)
                sb = Shape(gap_max=0.08, hole_radius=0.2)
                sc = Shape()

                samples_c = sample_position_inside_many(sa, [sc], [size_c/size_a])
                while len(samples_c)==0:
                    sa.randomize()
                    sc.randomize()
                    samples_c = sample_position_inside_many(sa, [sc], [size_c/size_a])
                
                done = True

                xy_c = samples_c[0,0]

                range_ab = 1 - np.array([size_a, size_b])
                starting_ab = np.array([size_a, size_b])/2

                xy_ab = np.random.rand(10, 2, 2) * range_ab[None,:,None] + starting_ab[None,:,None]
                no_overlap = (np.abs(xy_ab[:,0]-xy_ab[:,1]) - (size_a + size_b)/2 > 0).any(1)
                while not no_overlap.any(0):
                    xy_ab = np.random.rand(10, 2, 2) * range_ab[None,:,None] + starting_ab[None,:,None]
                    no_overlap = (np.abs(xy_ab[:,0]-xy_ab[:,1]) - (size_a + size_b)/2 > 0).any(1)

                xy_ab = xy_ab[no_overlap]

                xy_a, xy_b = xy_ab[0, 0], xy_ab[0, 1]
                xy_c = xy_c * size_a + xy_a

            # a,b,c
            elif sampled_conditions[i] == 6:
                max_size = 0.7
                min_size = max_size/2

                size_a = np.random.rand() * (max_size - min_size) + min_size 
                size_b = np.random.rand() * (size_a/2.5 - size_a/4) + size_a/4
                size_c = np.random.rand() * (size_b/2.5 - size_b/4) + size_b/4

                sa = Shape(gap_max=0.08, hole_radius=0.2)
                sb = Shape(gap_max=0.08, hole_radius=0.2)
                sc = Shape()
                
                size = np.array([size_a, size_b, size_c])
                
                range_abc = 1 - size
                starting_abc = size/2

                triu_idx = np.triu_indices(3, k=1)[0]*3 + np.triu_indices(3, k=1)[1]

                xy_abc = np.random.rand(100, 3, 2) * range_abc[None,:,None] + starting_abc[None,:,None]                
                dists = np.abs(xy_abc[:,:,None,:] - xy_abc[:,None,:,:]) - (size[None,:,None,None] + size[None,None,:,None])/2
                no_overlap = dists.any(3).reshape(100, 3*3)[:, triu_idx].all(1)

                while not no_overlap.any():
                    
                    xy_abc = np.random.rand(100, 3, 2) * range_abc[None,:,None] + starting_abc[None,:,None]                
                    dists = np.abs(xy_abc[:,:,None,:] - xy_abc[:,None,:,:]) - (size[None,:,None,None] + size[None,None,:,None])/2
                    no_overlap = dists.any(3).reshape(100, 3*3)[:, triu_idx].all(1)
                
                done = True

                xy_abc = xy_abc[no_overlap]
                xy_a, xy_b, xy_c = xy_abc[0, 0], xy_abc[0, 1], xy_abc[0, 2]
                
            if done:
                break

        xy.append(np.array([xy_a, xy_b, xy_c]))
        sizes.append([size_a, size_b, size_c])
        shapes.append([sa, sb, sc])
        
        if not done:
            return np.zeros([256,256])

    xy = np.stack(xy, axis=0)
    size = np.stack(sizes, axis=0)
    
    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*3)
        color = c.reshape([n_samples, 3, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([3, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([3, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, size, shapes, color

def task_inside_inside_2(condition='c'):

    n_samples = 4

    max_attempts = 10 

    xy = []
    shapes = []
    sizes = []

    n_objects = np.random.randint(4, 9, size=[n_samples])

    n_objects_out = np.ones_like(n_objects)
    n_objects_out[-1] = np.random.randint(2, n_objects[-1])
    n_objects_in = n_objects - n_objects_out

    for i in range(n_samples):
        done = False

        for _ in range(max_attempts):
            
            # a(b(c))
            if n_objects_out[i] == 1:
                max_size = 0.7
                min_size = max_size*2/3

                size = np.random.rand() * (max_size - min_size) + min_size 
                size_in = np.random.rand() * (size/n_objects_in[i]/2.5 - size/n_objects_in[i]/3.5) + size/n_objects_in[i]/3.5 

                sa = Shape(gap_max=0.07, hole_radius=0.2)

                shapes_in = []
                for _ in range(n_objects_in[i]):
                    s_in = Shape()
                    shapes_in.append(s_in)

                samples_in = sample_position_inside_many(sa, shapes_in, [size_in/size]*n_objects_in[i])

                range_1 = 1 - size
                starting_1 = size/2
                xy_out = np.random.rand(2) * range_1 + starting_1

                if len(samples_in)>0:
                    done = True
                    
                    xy_in = samples_in[0]
                    xy_in = xy_in * size + xy_out[None,:]
                
                else:
                    
                    xy_in = xy_out[None,:]*np.ones([n_objects_in[i],1])
                
                if done:

                    xy.append(np.concatenate([xy_out[None,:], xy_in],0))
                    sizes.append(np.array([size] + [size_in]*n_objects_in[i]))
                    shapes.append([sa]+shapes_in)

            # a(b,c)
            else:
                max_size = 0.7
                min_size = max_size*2/3

                size = np.random.rand() * (max_size - min_size) + min_size 
                size_in = np.random.rand() * (size/n_objects_in.mean()/2.5 - size/n_objects_in.mean()/3.5) + size/n_objects_in.mean()/3.5

                sa = Shape(gap_max=0.07, hole_radius=0.2)

                shapes_in = []
                for _ in range(n_objects_in[i]+n_objects_out[i]-1):
                    s_in = Shape()
                    shapes_in.append(s_in)
                
                shapes_out = shapes_in[n_objects_in[i]:]
                shapes_in = shapes_in[:n_objects_in[i]]

                samples_in = sample_position_inside_many(sa, shapes_in, [size_in/size]*n_objects_in[i])

                size_ = np.array([size]+[size_in]*(n_objects_out[i]-1))

                range_1 = 1 - size_
                starting_1 = size_/2
                xy_out_samples = np.random.rand(200,n_objects_out[i],2) * range_1[None,:,None] + starting_1[None,:,None]

                dists = np.abs(xy_out_samples[:,:,None,:] - xy_out_samples[:,None,:,:]) - (size_[None,:,None,None] + size_[None,None,:,None])/2 > 0
                triu_idx = np.triu_indices(n_objects_out[i], k=1)[0]*n_objects_out[i] + np.triu_indices(n_objects_out[i], k=1)[1]
                no_overlap = dists.any(3).reshape(200, n_objects_out[i]*n_objects_out[i])[:, triu_idx].all(1)
                
                if no_overlap.sum(0)>0:
                    xy_out = xy_out_samples[no_overlap][0]
                else:
                    xy_out = [[0.5,0.5]]*n_objects_out[i]

                if len(samples_in)>0:
                    done = True
                    
                    xy_in = samples_in[0]
                    xy_in = xy_in * size + xy_out[0]
                
                else:
                    xy_in = xy_in * size + xy_out[0]
    
                if done:
                    xy.append(np.concatenate([xy_out, xy_in], 0))
                    sizes.append(np.array([size]+[size_in]*(n_objects_in[i]+n_objects_out[i]-1)))
                    shapes.append([sa]+shapes_out+shapes_in)

            if done:
                break

        if not done:
            return np.zeros([256,256])

    if 'c' in condition:
        color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        # c = sample_random_colors(n_samples*3)
        # color = c.reshape([n_samples, 3, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects[i], 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects[i], 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return xy, sizes, shapes, color

#################################################################################
# inside + count # set 1

def task_inside_count_1(condition='c'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_objects_inside = np.random.randint(low=2, high=6)
    n_objects_inside = np.ones(n_samples) * n_objects_inside
    n_objects_inside[-1] = n_objects_inside[-1] + np.random.choice([-1,1]) 
    n_objects_inside = n_objects_inside.astype(int)
    n_objects_samples = n_objects_inside + np.random.randint(low=2, high=6, size=n_samples)    
    n_objects_samples[n_objects_samples>6] = 6
    n_objects_samples = n_objects_samples.astype(int)

    # n_objects = n_objects_samples.max()
    # min_size_obj = 0.2
    min_btw_size = 0.3

    
    all_xy = []
    all_size = []
    all_shape = []
    
    for i in range(n_samples):
        # xy_i = xy[i,:n_objects_samples[i]]
        xy_ = []
        size_ = []
        shape_ = []

        done = False
        n_objects = n_objects_samples[i]

        min_size_obj = min(0.9/np.sqrt(n_objects*3), 0.3)
        min_btw_size = min_size_obj

        triu_idx = np.triu_indices(n_objects, k=1)
        triu_idx = triu_idx[0]*n_objects + triu_idx[1]        
        
        for _ in range(max_attempts):
            xy = np.random.rand(n_samples_over, n_objects, 2) * (1-min_size_obj) + min_size_obj/2 
            no_overlap = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - min_btw_size>0).any(3).reshape([-1, n_objects*n_objects])[:,triu_idx].all(1)
            if no_overlap.sum()>0:
                done = True
                break
            
        if not done:
            print('')

        xy = xy[no_overlap][0]

        if n_objects >1:
            non_diag = np.where(~np.eye(n_objects,dtype=bool))
            non_diag = non_diag[0]*n_objects + non_diag[1]
            dists_obj = np.abs(xy[:,None,:] - xy[None,:,:]).max(2).reshape([n_objects**2])[non_diag].reshape([n_objects, n_objects-1]).min(1)
            dists_edge = np.stack([xy, 1-xy],2).min(1).min(1)*2
            max_size = np.stack([dists_edge, dists_obj], 1).min(1)
        else:
            max_size = 0.6
        # max_size = np.stack([xy, 1-xy],2).min(2).min(2)
        min_size = max_size*2/3

        size = np.random.rand(n_objects) * (max_size - min_size) + min_size 
        size_in = np.random.rand(n_objects) * (size/2.5 - size/4) + size/4

        object_in = np.arange(n_objects) < n_objects_inside[i]

        for j in range(n_objects_samples[i]):
            if object_in[j]:
                    
                done = False                
                s1 = Shape(gap_max=0.07, hole_radius=0.2)
                s2 = Shape()
                for _ in range(max_attempts):

                    samples = sample_position_inside_1(s1, s2, size_in[j]/size[j])
                    if len(samples)>0:
                        done = True
                    
                    if done:
                        break
                    else:
                        s1.randomize()
                        s2.randomize()

                if done:
                    xy_in = samples[0]

                    xy_.append(xy[j])
                    shape_.append(s1)
                    size_.append(size[j])
                    
                    xy_.append(xy_in * size[j] + xy[j])
                    shape_.append(s2)
                    size_.append(size_in[j])
            
            else:
                if np.random.randint(2) == 0:
                    s1 = Shape()
                    size_.append(size[j])
                else:
                    s1 = Shape(gap_max=0.07, hole_radius=0.2)
                    size_.append(size_in[j])
                                
                xy_.append(xy[j])
                shape_.append(s1)

        all_xy.append(xy_)
        all_size.append(size_)
        all_shape.append(shape_)

    if 'c' in condition:
        color = [sample_random_colors(n_objects_samples[i] + n_objects_inside[i]) for i in range(n_samples)]
        # c = sample_random_colors(n_samples*3)
        # color = c.reshape([n_samples, 3, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects_samples[i] + n_objects_inside[i], 3]) * color[i:i+1] for i in range(n_samples)]

    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects_samples[i] + n_objects_inside[i], 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    return all_xy, all_size, all_shape, color

#################################################################################
# inside + contact

def task_inside_contact(condition='sc'):
    
    max_attempts = 10

    n_samples = 4
    n_objects = 4
    
    max_size = 0.9/np.sqrt(n_objects*2)
    min_size = max_size/2
    
    cond = np.random.randint(2)
    # a()-c() d-f
    # a()-c   d()-f

    cond = [cond]*(n_samples-1) + [1-cond]

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])
    
    size_in = np.random.rand(n_samples, 2) * (size[:,:2]/2.5 - size[:,:2]/4) + size[:,:2]/4
    shape = []
    xy_in = []
    for i in range(n_samples):
        shape_out = []
        shape_in = []
        # xy_ = []
        xy_in_ = []
        for j in range(2):
        
            s1 = Shape(gap_max=0.06, hole_radius=0.2)
            s2 = Shape(gap_max=0.06, hole_radius=0.2)
            done = False
            for _ in range(max_attempts):
                samples = sample_position_inside_1(s1, s2, size_in[i,j]/size[i,j])
                if len(samples)>0:
                    done = True
                if done:
                    break
                else:
                    s1.randomize()
                    s2.randomize()

            if done:
                xy_in_.append(samples[0])
                # xy_in_.append(xy_ * size[i,j] + xy[i,j])
        
            shape_out.append(s1)
            shape_in.append(s2)
        
        shape.append(shape_out + [Shape(), Shape()] + shape_in)
        xy_in.append(xy_in_)

    xy_in = np.array(xy_in)

    xy = []
    for i in range(n_samples):
        if cond[i] == 0:
            idx1, idx2, idx3, idx4 = 0, 1, 2, 3
        else:
            idx1, idx2, idx3, idx4 = 0, 2, 1, 3
        
        s1 = shape[i][idx1]
        s2 = shape[i][idx2]
        s3 = shape[i][idx3]
        s4 = shape[i][idx4]
    
        positions1, clump_size1 = sample_contact_many([s1, s2], np.array([size[i,idx1], size[i,idx2]]))
        positions2, clump_size2 = sample_contact_many([s3, s4], np.array([size[i,idx3], size[i,idx4]]))
        
        c_size = np.stack([clump_size1, clump_size2], 0)
        co1 = sample_over_range_t(1, np.array([0,1]), c_size[None,:,0])
        co2 = np.random.rand(1, 2) * (1-c_size[None,:,1]) + c_size[None,:,1]/2
        xy_ = [co1, co2]# if np.random.randint(2)==0 else [co2, co1]
        xy_ = np.stack(xy_, axis=2)[0]

        xy_1 = positions1 + xy_[0:1,:]
        xy_2 = positions2 + xy_[1:,:]
        xy_ = np.concatenate([xy_1, xy_2], 0)
        xy.append(xy_[[idx1, idx2, idx3, idx4]])

    size = np.concatenate([size, size_in], 1)
    xy = np.stack(xy, 0)
    xy_in = xy_in * size[:,:2, None] + xy[:,:2]
    xy = np.concatenate([xy, xy_in], 1)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects_samples[i] + n_objects_inside[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*(n_objects+2))
        color = c.reshape([n_samples, n_objects+2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects+2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects+2, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

#################################################################################
# inside + sym

def task_inside_sym_mir(condition='xyc'):

    variable = 'y' if np.random.randint(2) == 0 else 'x'

    n_samples = 4

    n_objects = np.random.randint(2,5)
    
    n_objects_in = np.random.randint(1,n_objects)

    # odd: distance not correlated to size, no sym for n_objects
    # odd_condition = np.random.randint(2)
    # odd_condition = -1

    # odd_condition = 0
    
    max_size = 0.9/n_objects
    min_size = max_size*2/3
    
    size = np.random.rand(n_samples, n_objects) * (max_size - min_size) + min_size

    size_in = np.random.rand(n_samples, n_objects_in) * (size[:,:n_objects_in]/2.5 - size[:,:n_objects_in]/4) + size[:,:n_objects_in]/4


    non_valid = size.sum(1) > 0.9
    if non_valid.any():
        size[non_valid] = size[non_valid]/size[non_valid].sum(1)[:,None] * 0.9

    margin_r = 1.3

    x_range = (0.5 - (size.max(1) + size.min(1))/2*margin_r)
    x_starting = size.min(1)/2*margin_r
    
    # x1 = (size - size.min(1)[:,None]) / (size.max(1) - size.min(1))[:,None] * x_range[:,None] + x_starting[:,None]
    # x1 = 0.5 - x1

    # if odd_condition == 0:
    x1 = np.random.rand(n_samples, n_objects) * (0.5 - size) + size/2
        
    x2 = 1 - x1

    y = sample_over_range_t(n_samples, np.array([0,1]), size)

    size = np.concatenate([size]*2, axis=1)
    size_in = np.concatenate([size_in]*2, axis=1)

    size = np.concatenate([size, size_in], 1)

    x = np.concatenate([x1,x2], 1)
    y = np.concatenate([y,y], 1)
    xy = np.stack([x,y], 2)
    # choose size
    # choose x based on size
    # choose y st there's no overlap

    max_attempts = 10

    shape = []
    xy_in = []
    if 'id' not in condition:
        s = Shape(gap_max=0.06, hole_radius=0.2)
    for i in range(n_samples):
        n_objs = []
        n_objs_f = []
        shape_out = []
        shape_out_f = []
        shape_in = []
        shape_in_f = []
        xy_in_ = []
        xy_in_f = []
        # if odd_condition in [1,2] and i==n_samples-1:
        #     odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects-1), replace=False)
        
        for j in range(n_objects):

            
            if j<n_objects_in:

                if 'id' in condition:
                    s = Shape(gap_max=0.06, hole_radius=0.2)
    
                so = s.clone()
                si = Shape()

                done = False
                for _ in range(max_attempts):
                    samples = sample_position_inside_1(so, si, size_in[i,j]/size[i,j])
                    if len(samples)>0:
                        done = True
                    if done:
                        break
                    else:
                        # so.randomize()
                        si.randomize()

                if done:
                    xy_ = samples[0]

                    xy_in_.append(xy_ * size[i,j] + xy[i,j])
                else:
                    xy_in_.append(xy[i,j])


                # so1 = s.clone()
                so2 = so.clone()
                si2 = si.clone()

                # if not (odd_condition == 2 and i == n_samples-1 and j in odd_samples):
                if i != n_samples-1:
                    so2.flip()
                    si2.flip()
                    xy__ = np.copy(xy_in_[-1])
                    xy__[0] = 1 - xy__[0]

                    xy_in_f.append(xy__)
                
                else:

                    xy_in_f.append(xy_in_[-1] - xy[i,j] + xy[i,j+n_objects])

                shape_out.append(so)
                shape_out_f.append(so2)
                shape_in.append(si)
                shape_in_f.append(si2)


            else:

                odd_condition = np.random.randint(2)+1

                if 'id' in condition:
                    s = Shape()

                s1 = s.clone()
                s2 = s.clone()
                
                if i == n_samples-1:
                    s2.flip()
                else:
                    if not (odd_condition == 2):
                        s2.flip()

                    if odd_condition == 1:
                        s_ = Shape()
                        if np.random.rand()>0.5:
                            s1 = s_
                        else:
                            s2 = s_

                shape_out.append(s1)
                shape_out_f.append(s2)
        
        shape.append(shape_out + shape_out_f + shape_in + shape_in_f)
        xy_in.append(xy_in_ + xy_in_f)

    xy_in = np.array(xy_in)
    xy = np.concatenate([xy, xy_in], 1)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects_samples[i] + n_objects_inside[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*(n_objects+n_objects_in)*2)
        color = c.reshape([n_samples, (n_objects+n_objects_in)*2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([(n_objects+n_objects_in)*2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([(n_objects+n_objects_in)*2, 3]) * color for i in range(n_samples)]

    if variable=='x':
        for i in range(n_samples):
            xy[i], shape[i] = flip_diag_scene(xy[i], shape[i])

    return xy, size, shape, color

#################################################################################
#################################################################################
#################################################################################

#################################################################################
# contact + contact # set 3

def task_contact_contact_1(condition='cid'):
        
    n_samples = 4
    # n_objects = 3
    n_objects = np.random.randint(4, 9, size=n_samples)

    max_size = 0.9/np.sqrt(n_objects*3)
    min_size = max_size/2

    size = np.random.rand(n_samples, n_objects.max()) * (max_size[:,None] - min_size[:,None]) + min_size[:,None]


    n_obj_connected = n_objects.copy()

    n_obj_connected[-1] = np.random.randint(n_obj_connected[-1])

    if 'id' not in condition:
        s = Shape()
    
    shape = []
    xy = []

    for i in range(n_samples):
        
        if 'id' in condition:
            shapes_ = [Shape() for i in range(n_objects[i])]
        else:
            shapes_ = [s.clone() for i in range(n_objects[i])]

        # s1 = Shape()
        # s2 = Shape()
        # s3 = Shape()

        # dir_ = dirs[i] 
        if n_obj_connected[i] == n_objects[i]:
            positions, clump_size = sample_contact_many(shapes_, size[i])
            xy0 = np.random.rand(2) * (1-clump_size) + clump_size/2
            xy_ = positions + xy0[None,:]
        
        elif n_obj_connected[i] > 0:
            positions, clump_size = sample_contact_many(shapes_[:n_obj_connected[i]], size[i,:n_obj_connected[i]])

            n_free_objs = n_objects[i] - n_obj_connected[i] + 1

            triu_idx = np.triu_indices(n_free_objs, k=1)
            triu_idx = triu_idx[0]*n_free_objs + triu_idx[1]

            size_ = np.ones([n_free_objs, 2])
            size_[0] = clump_size
            size_[1:] = size_[1:] * size[i, n_obj_connected[i]:n_objects[i], None]
            
            n_samples_over = 100
            xy_ = np.random.rand(n_samples_over, n_free_objs, 2) * (1 - size_[None,:,:]) + size_[None,:,:]/2
            valid = (np.abs(xy_[:,:,None,:] - xy_[:,None,:,:]) - (size_[None,:,None,:] + size_[None,None,:,:])/2 > 0).any(3).reshape([n_samples_over, n_free_objs**2])[:, triu_idx].all(1)
            while not valid.any():
                xy_ = np.random.rand(n_samples_over, n_free_objs, 2) * (1 - size_[None,:,:]) + size_[None,:,:]/2
                valid = (np.abs(xy_[:,:,None,:] - xy_[:,None,:,:]) - (size_[None,:,None,:] + size_[None,None,:,:])/2 > 0).any(3).reshape([n_samples_over, n_free_objs**2])[:, triu_idx].all(1)
            xy_ = xy_[valid][0]
            xy0 = xy_[0]
            xy_ = np.concatenate([positions + xy0[None,:], xy_[1:]], 0)

        else:
            n_samples_over = 100
            triu_idx = np.triu_indices(n_objects[i], k=1)
            triu_idx = triu_idx[0]*n_objects[i] + triu_idx[1]
            
            xy_ = np.random.rand(n_samples_over, n_objects[i], 2) * (1 - size[i:i+1,:n_objects[i],None]) + size[i:i+1,:n_objects[i],None]/2
            valid = (np.abs(xy_[:,:,None,:] - xy_[:,None,:,:]) - (size[i:i+1,:n_objects[i],None,None] + size[i:i+1,None,:n_objects[i],None])/2 > 0).any(3).reshape([n_samples_over, n_objects[i]**2])[:, triu_idx].all(1)
            while not valid.any():
                xy_ = np.random.rand(n_samples_over, n_objects[i], 2) * (1 - size[i:i+1,:n_objects[i],None]) + size[i:i+1,:n_objects[i],None]/2
                valid = (np.abs(xy_[:,:,None,:] - xy_[:,None,:,:]) - (size[i:i+1,:n_objects[i],None,None] + size[i:i+1,None,:n_objects[i],None])/2 > 0).any(3).reshape([n_samples_over, n_objects[i]**2])[:, triu_idx].all(1)
            xy_ = xy_[valid][0]

        xy.append(xy_)
        shape.append(shapes_)    

    size = [size[i,:n_objects[i]] for i in range(n_samples)]

    if 'c' in condition:
        color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        # c = sample_random_colors(n_samples*(n_objects+n_objects_in)*2)
        # color = c.reshape([n_samples, (n_objects+n_objects_in)*2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects[i], 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects[i], 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

def task_contact_contact_2(condition='csid'):
        
    n_samples = 4
    
    n_groups = np.random.randint(3, 6)

    n_objects_samples = np.random.randint(2, 4, size=n_groups)
    n_objects_samples = [np.random.randint(2, 4, size=n_groups) for i in range(n_samples)]

    n_objects_samples_ = []

    for i in range(n_samples):
        n_objects_samples = np.random.randint(2, 4, size=n_groups)
        # n_objects.append(n_objects_samples.sum())
        n_objects_samples_.append(n_objects_samples)

    n_objects_samples = n_objects_samples_

    n_objects_samples[-1][0] = 1

    n_objects = [n_.sum() for n_ in n_objects_samples]

    contact_idx_ = []
    for j in range(n_samples):
            
        contact_idx = [[i]*n_objects_samples[j][i] for i in range(n_groups)]
        contact_idx = np.array(cat_lists(contact_idx)) 
        contact_idx_.append(contact_idx)

    contact_idx = contact_idx_

    max_size = 0.9/np.sqrt(np.array(n_objects)*4)
    min_size = max_size/2
    
    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*max(n_objects))
        color = c.reshape([n_samples, max(n_objects), 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([max(n_objects), 3]) * color[i:i+1] for i in range(n_samples)]

    else:
        color = sample_random_colors(1)
        color = [np.ones([max(n_objects), 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    if 's' in condition:
        size = np.random.rand(n_samples, max(n_objects)) * (max_size[:,None] - min_size[:,None]) + min_size[:,None]
    else:
        size = np.random.rand() * (max_size[:,None] - min_size[:,None]) + min_size[:,None]
        size = size * np.ones([n_samples, max(n_objects)])

    if 'id' in condition:
        shape = [[Shape() for j in range(n_objects[i])] for i in range(n_samples)]

    else:
        s = Shape()
        shape = [[s.clone() for j in range(n_objects[i])] for i in range(n_samples)]

    xy = []
    for i in range(n_samples):
        n_obj_s = n_objects_samples[i]
        xy_ = []
        if i == n_samples-1:
            n = 1
            positions_ = [] 
            clump_size_ = []
            for j in range(1, n_groups):
                positions, clump_size = sample_contact_many(shape[i][n:n + n_obj_s[j]], size[i, n:n + n_obj_s[j]])
                n = n + n_obj_s[j]
                
                positions_.append(positions)
                clump_size_.append(clump_size)
            
            clump_size_ = [np.array([size[i,0], size[i,0]])]+clump_size_

            clump_size = np.stack(clump_size_, 0)*1.1
            xy0 = sample_positions_bb(clump_size[None,:,:], n_sample_min=1)
            xy0 = xy0[0]
            xy_.append(xy0[0:1,:])
            for j in range(1, n_groups):
                xy_.append(positions_[j-1] + xy0[j:j+1,:])

        else:
            n = 0
            positions_ = [] 
            clump_size_ = []
            for j in range(n_groups):
                positions, clump_size = sample_contact_many(shape[i][n:n + n_obj_s[j]], size[i, n:n + n_obj_s[j]])
                n = n + n_obj_s[j]
                
                positions_.append(positions)
                clump_size_.append(clump_size)
            
            if i == n_samples-1:
                clump_size = [np.array([size[i,0], size[i,0]])]+clump_size

            clump_size = np.stack(clump_size_, 0)*1.1
            xy0 = sample_positions_bb(clump_size[None,:,:], n_sample_min=1)
            xy0 = xy0[0]
            for j in range(n_groups):
                xy_.append(positions_[j] + xy0[j:j+1,:])

        xy.append(np.concatenate(xy_, 0))

    return xy, size, shape, color

#################################################################################
# contact + count

def task_contact_count_1(condition='csid'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.randint(2, 6)

    n_objects_samples = np.random.randint(2, 4, size=n_groups)
    n_objects = n_objects_samples.sum()

    n_objects_samples_odd = sample_int_sum_n(n_groups, n_objects, min_v=1)
    while (np.sort(n_objects_samples_odd) == np.sort(n_objects_samples)).all():
        n_objects_samples_odd = sample_int_sum_n(n_groups, n_objects, min_v=1)

    contact_idx = [[i]*n_objects_samples[i] for i in range(n_groups)]
    contact_idx = np.array(cat_lists(contact_idx)) 
    contact_idx_odd = [[i]*n_objects_samples_odd[i] for i in range(n_groups)]
    contact_idx_odd = np.array(cat_lists(contact_idx_odd)) 
    
    max_size = 0.9/np.sqrt(n_objects*4)
    min_size = max_size/2
    
    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])

    if 'id' in condition:
        shape = [[Shape() for j in range(n_objects)] for i in range(n_samples)]
    else:
        s = Shape()
        shape = [[s.clone() for j in range(n_objects)] for i in range(n_samples)]


    xy = []
    for i in range(n_samples):
        n_obj_s = n_objects_samples_odd if i == n_samples-1 else n_objects_samples
        xy_ = []
        n = 0
        positions_ = [] 
        clump_size_ = []
        for j in range(n_groups):
            positions, clump_size = sample_contact_many(shape[i][n:n + n_obj_s[j]], size[i, n:n + n_obj_s[j]])
            n = n + n_obj_s[j]
            
            positions_.append(positions)
            clump_size_.append(clump_size)
        
        clump_size = np.stack(clump_size_, 0)*1.1
        xy0 = sample_positions_bb(clump_size[None,:,:], n_sample_min=1)
        xy0 = xy0[0]
        for j in range(n_groups):
            xy_.append(positions_[j] + xy0[j:j+1,:])

        xy.append(np.concatenate(xy_, 0))

    return xy, size, shape, color

def task_contact_count_2(condition='csid'):

    max_attempts = 20
    
    n_samples = 4
    n_samples_over = 100
    
    n_groups = np.random.choice([2,3,4], replace=False, size=2)

    n_objects = np.random.randint(n_groups.max(), n_groups.max()*2)

    n_groups_odd = n_groups[1]
    n_groups = n_groups[0]

    # n_objects_samples = np.random.randint(2, 4, size=n_groups)
    # n_objects = n_objects_samples.sum()
    n_objects_samples = []
    for i in range(n_samples-1):
        n_objects_samples.append(sample_int_sum_n(n_groups, n_objects, min_v=1))    
    n_objects_samples.append(sample_int_sum_n(n_groups_odd, n_objects, min_v=1))
    
    max_size = 0.9/np.sqrt(n_objects*4)
    min_size = max_size/2

    if 's' in condition:
        size = np.random.rand(n_samples, n_objects) * (max_size-min_size) + min_size
    else:
        size = np.random.rand() * (max_size-min_size) + min_size
        size = size * np.ones([n_samples, n_objects])
        
    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            s = Shape()
            shape += [[s.clone() for i in range(n_objects)]]
    else:
        s = Shape()
        shape = [[s.clone() for i in range(n_objects)] for _ in range(n_samples)]

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]

    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])


    xy = []
    for i in range(n_samples):
        n_obj_s = n_objects_samples[i]
        n_g = n_groups_odd if i == n_samples-1 else n_groups
        xy_ = []
        n = 0
        positions_ = [] 
        clump_size_ = []
        for j in range(n_g):
            positions, clump_size = sample_contact_many(shape[i][n:n + n_obj_s[j]], size[i, n:n + n_obj_s[j]])
            n = n + n_obj_s[j]
            
            positions_.append(positions)
            clump_size_.append(clump_size)
        
        clump_size = np.stack(clump_size_, 0)*1.1
        xy0 = sample_positions_bb(clump_size[None,:,:], n_sample_min=1)
        xy0 = xy0[0]
        
        for j in range(n_g):
            xy_.append(positions_[j] + xy0[j:j+1,:])

        xy.append(np.concatenate(xy_, 0))

    return xy, size, shape, color

#################################################################################
#################################################################################
#################################################################################

#################################################################################
# count + count

def task_count_count(condition='xyscid'):

    n_samples = 4
    n_objects = np.random.choice(np.arange(3, 7), size=[n_samples], replace=False)*2
    if np.random.randint(2)==0:
        n_objects[:-1] = n_objects[:-1] + np.random.choice([-1,1])
    else:
        n_objects[-1] = n_objects[-1] + np.random.choice([-1,1])
    
    max_n_objs = n_objects.max()
        
    max_size = 0.9/np.sqrt(max_n_objs*3)
    min_size = max_size/2

    shape = Shape()

    if 'id' in condition:
        id_idx = np.arange(max_n_objs*n_samples).reshape([n_samples, max_n_objs])
    elif 'id' not in condition:
        id_idx = np.zeros([n_samples, max_n_objs])

    if 's' in condition:
        s_idx = np.arange(max_n_objs*n_samples).reshape([n_samples, max_n_objs])
    elif 's' not in condition:
        s_idx = np.zeros([n_samples, max_n_objs])

    unique_s_idx = np.unique(s_idx)
    
    unique_sizes = np.random.rand(len(unique_s_idx)) * (max_size - min_size) + min_size
    sizes = unique_sizes[s_idx.flatten().astype(int)].reshape(s_idx.shape)

    if 'xy' in condition:

        xy = []
        for i in range(n_samples):
            xy_ = sample_positions(sizes[i:i+1,:n_objects[i]], n_sample_min=1)
            xy.append(xy_[0])
            
    else:
        xy = []
        for i in range(n_samples):
            xy_ = sample_positions(sizes[i:i+1,:n_objects[i]], n_sample_min=1)
            xy.append(xy_[0])
                
    unique_id = np.unique(id_idx)
    all_shapes = []
    for i in range(len(unique_id)):
        shape = Shape()
        all_shapes.append(shape)
            
    if 'c' in condition:
        color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]

    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects[i], 3]) * color for i in range(n_samples)]
        # color = c[:, None,:] * np.ones([n_samples, n_objects*2, 3])

    xy = [xy[i][:n_objects[i]] for i in range(n_samples)]
    size = [sizes[i][:n_objects[i]] for i in range(n_samples)]
    shape = [[all_shapes[k].clone() for k in id_idx[i][:n_objects[i]].astype(int)] for i in range(n_samples)]

    return xy, size, shape, color

############################

#################################################################################
#################################################################################
#################################################################################

##########################
# sym

def task_sym_mir(condition='cid'):
    # variable = 'x' if np.randon.randint(2)==0 else 'y'
    # mir sym:
    # (x0,y0), r, a
    # o_1 = (0, a - pi/2) 
    # pos_1 = r * (cos(a1),sin(a1)) + (x0,y0)
    # o_2 = (1, a + pi/2)
    # pos_2 = r * (cos(a2),sin(a2)) + (x0,y0)
    # and

    n_samples = 4

    n_objects = 2

    # odd conditions: shape1 != shape2, size1 != size2, flip1 == flip2, position doesn't fit, incorrect angle
    odd_condition = np.random.randint(5) # 5

    max_size = 0.4
    min_size = max_size/4
    
    size = np.random.rand(n_samples) * (max_size - min_size) + min_size
    size = np.stack([size]*n_objects, axis=1)
    
    if odd_condition == 1:
        idx = np.random.randint(2)
        size[-1,idx] = size[-1,idx] * (np.random.rand()* (2/3 - 1/3) + 1/3)

    a = (np.random.rand(n_samples) * 1 - 1/2) * np.pi 
    r = np.random.rand(n_samples) * (0.4-size.max(1)) + size.max(1)/2

    # a1, a2 = a - np.pi/2, a + np.pi/2

    range_ = 1 - size[:,0:1] - 2 * np.abs(np.stack([r * np.cos(a), r * np.sin(a)], 1))
    starting_ = size[:,0:1]/2 + np.abs(np.stack([r * np.cos(a), r * np.sin(a)], 1))
    
    xy0 = np.random.rand(n_samples, n_objects) * range_ + starting_

    xy = xy0[:,None,:] + np.stack([r * np.cos(a), r * np.sin(a), r * np.cos(a + np.pi), r * np.sin(a + np.pi)], 1).reshape([n_samples, 2, 2])

    if odd_condition == 3:
        xy[-1] = sample_positions(size[-1:])
    
    
    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            s = Shape()
            s1 = s.clone()
            s2 = s.clone()
            a_ = a[i]

            if odd_condition == 4 and i == n_samples-1:
                a_ = a_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi
            
            s1.rotate(a_)

            if not (odd_condition == 2 and i == n_samples-1):
                s2.flip()
            
            s2.rotate(a_)

            if odd_condition == 0 and i == n_samples-1:
                s_ = Shape()
                if np.random.rand()>0.5:
                    s1 = s_
                else:
                    s2 = s_

            shape.append([s1, s2])
            
    else:
        shape = []
        s = Shape()
        for i in range(n_samples):
            s1 = s.clone()
            s2 = s.clone()

            a_ = a[i]
            
            if odd_condition == 4 and i == n_samples-1:
                a_ = a_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi

            s1.rotate(a_)
            
            if not (odd_condition == 2 and i == n_samples-1):
                s2.flip()
            
            s2.rotate(a_)

            if odd_condition == 0 and i == n_samples-1:
                s_ = Shape()
                if np.random.rand()>0.5:
                    s1 = s_
                else:
                    s2 = s_

            shape.append([s1, s2])

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

def task_sym_rot(condition='cid'):
    
    # rot sym:
    # r
    # o_1 = (0, a1)
    # pos_1 = r * (cos(a1),sin(a1)) + (x0,y0)
    # o_2 = (0, a2)
    # pos_2 = r * (cos(a2),sin(a2)) + (x0,y0)
    # and

    n_samples = 4

    n_objects = 3

    # odd conditions: shape1 != shape2, size1 != size2, position doesn't fit, incorrect angle
    odd_condition = np.random.randint(4) # 5

    max_size = 0.4
    min_size = max_size/4
    
    size = np.random.rand(n_samples) * (max_size - min_size) + min_size
    # size = np.stack([np.ones(n_samples)*0.05]+[size]*2, axis=1)
    size = np.stack([size]*2, axis=1)
    
    ################ can be random
    # a1 = (np.random.rand(n_samples)) * np.pi
    # a2 = (np.random.rand(n_samples) * 2/3 + 1/3) * np.pi +  a1
    ################ or specific values
    a1 = (np.random.rand(n_samples)) * np.pi * 2
    # a2 = np.random.choice([1/2, 1]) * np.pi +  a1
    a2 = np.pi/2 +  a1

    wh = (np.max([np.zeros(n_samples), np.cos(a1), np.cos(a2)], 0) - np.min([np.zeros(n_samples), np.cos(a1), np.cos(a2)], 0), 
        np.max([np.zeros(n_samples), np.sin(a1), np.sin(a2)], 0) - np.min([np.zeros(n_samples), np.sin(a1), np.sin(a2)], 0))
    # max(wh) * r + max_size < 0.8
    max_r = (1 - size[:,0] * np.sqrt(2)) / np.max(wh, 0)
    min_r = (size[:,0] * np.sqrt(2))/(2*np.sin((a2-a1)/2))
    r = np.random.rand(n_samples) * (max_r - min_r) + min_r

    if odd_condition == 1:
        idx = np.random.randint(2)
        size[-1,idx] = size[-1,idx] * (np.random.rand()* (2/3 - 1/3) + 1/3)

    # xy0 * 3 - (x1+x2)
    # x0, x1, x2 
    # y0, y1, y2

    # x = np.stack([0, r*np.cos(a1), r*np.cos(a2)], 1)
    # y = np.stack([0, r*np.sin(a1), r*np.sin(a2)], 1)
    max_x_pos = np.max([np.zeros(n_samples), r*np.cos(a1) + np.sqrt(2) * size[:,0]/2, r*np.cos(a1) - np.sqrt(2) * size[:,0]/2, r*np.cos(a2) + np.sqrt(2) * size[:,1]/2, r*np.cos(a2) - np.sqrt(2) * size[:,1]/2], 0)
    max_y_pos = np.max([np.zeros(n_samples), r*np.sin(a1) + np.sqrt(2) * size[:,0]/2, r*np.sin(a1) - np.sqrt(2) * size[:,0]/2, r*np.sin(a2) + np.sqrt(2) * size[:,1]/2, r*np.sin(a2) - np.sqrt(2) * size[:,1]/2], 0)

    wh = (  np.max([np.zeros(n_samples), r*np.cos(a1) + np.sqrt(2) * size[:,0]/2, r*np.cos(a1) - np.sqrt(2) * size[:,0]/2, r*np.cos(a2) + np.sqrt(2) * size[:,1]/2, r*np.cos(a2) - np.sqrt(2) * size[:,1]/2], 0) - 
            np.min([np.zeros(n_samples), r*np.cos(a1) + np.sqrt(2) * size[:,0]/2, r*np.cos(a1) - np.sqrt(2) * size[:,0]/2, r*np.cos(a2) + np.sqrt(2) * size[:,1]/2, r*np.cos(a2) - np.sqrt(2) * size[:,1]/2], 0),
            np.max([np.zeros(n_samples), r*np.sin(a1) + np.sqrt(2) * size[:,0]/2, r*np.sin(a1) - np.sqrt(2) * size[:,0]/2, r*np.sin(a2) + np.sqrt(2) * size[:,1]/2, r*np.sin(a2) - np.sqrt(2) * size[:,1]/2], 0) -
            np.min([np.zeros(n_samples), r*np.sin(a1) + np.sqrt(2) * size[:,0]/2, r*np.sin(a1) - np.sqrt(2) * size[:,0]/2, r*np.sin(a2) + np.sqrt(2) * size[:,1]/2, r*np.sin(a2) - np.sqrt(2) * size[:,1]/2], 0))

    wh = np.stack(wh, 1)
    range_ = 1 - wh
    starting_ = wh/2
    
    xy0 = np.random.rand(n_samples, 2) * range_ + starting_

    xy0[:,0] = xy0[:,0] + wh[:,0]/2 - max_x_pos
    xy0[:,1] = xy0[:,1] + wh[:,1]/2 - max_y_pos

    xy = xy0[:,None,:] + r[:,None,None] * np.stack([np.cos(a1),np.sin(a1),np.cos(a2),np.sin(a2)], 1).reshape([n_samples, 2, 2])

    xy = np.concatenate([np.ones([n_samples,1,2])*xy0[:,None,:], xy], 1) 
    size = np.concatenate([np.ones([n_samples,1])*0.05, size], 1)

    if odd_condition == 2:
        xy[-1] = sample_positions(size[-1:])

    if 'id' in condition:
        shape = []
        s0 = Shape()
        
        for i in range(n_samples):
            s = Shape()
            s1 = s.clone()
            s2 = s.clone()
            a1_ = a1[i]
            a2_ = a2[i]

            if odd_condition == 3 and i == n_samples-1:
                a1_ = a1_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi
                a2_ = a2_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi
            
            s1.rotate(a1_)            
            s2.rotate(a2_)

            if odd_condition == 0 and i == n_samples-1:
                s_ = Shape()
                if np.random.rand()>0.5:
                    s1 = s_
                else:
                    s2 = s_

            shape.append([s0.clone(), s1, s2])
            
    else:
        shape = []
        s0 = Shape()
        
        s = Shape()
        for i in range(n_samples):
            a1_ = a1[i]
            a2_ = a2[i]
            
            if odd_condition == 3 and i == n_samples-1:
                if np.random.rand() > 0.5:
                    a1_ = a1_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi
                else:
                    a2_ = a2_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi

            s1 = s.clone()
            s2 = s.clone()

            s1.rotate(a1_)            
            s2.rotate(a2_)

            if odd_condition == 0 and i == n_samples-1:
                s_ = Shape()
                if np.random.rand()>0.5:
                    s1 = s_
                else:
                    s2 = s_

            shape.append([s0.clone(), s1, s2])

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects)
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

def task_size_sym_1(condition='cid'):

    variable = 'x' if np.random.randint(2)==0 else 'y'

    # mir sym:
    # (x0,y0), r, a
    # o_1 = (0, a - pi/2) 
    # pos_1 = r * (cos(a1),sin(a1)) + (x0,y0)
    # o_2 = (1, a + pi/2)
    # pos_2 = r * (cos(a2),sin(a2)) + (x0,y0)
    # and

    n_samples = 4

    n_objects = 6
    
    # odd: distance not correlated to size, no sym for n_objects
    odd_condition = np.random.randint(2)
    # odd_condition = 0
    
    max_size = 0.2
    min_size = max_size/5
    
    size = np.random.rand(n_samples, n_objects) * (max_size - min_size) + min_size
    non_valid = size.sum(1) > 0.9
    if non_valid.any():
        size[non_valid] = size[non_valid]/size[non_valid].sum(1)[:,None] * 0.9

    margin_r = 1.3

    x_range = (0.5 - (size.max(1) + size.min(1))/2*margin_r)
    x_starting = size.min(1)/2*margin_r
    
    x1 = (size - size.min(1)[:,None]) / (size.max(1) - size.min(1))[:,None] * x_range[:,None] + x_starting[:,None]
    x1 = 0.5 - x1

    if odd_condition == 0:
        x1[-1] = np.random.rand(n_objects) * (0.5 - size[-1]) + size[-1]/2
        
    x2 = 1 - x1

    y = sample_over_range_t(n_samples, np.array([0,1]), size)

    size = np.concatenate([size]*2, axis=1)

    x = np.concatenate([x1,x2], 1)
    y = np.concatenate([y,y], 1)
    xy = np.stack([x,y], 2)
    # choose size
    # choose x based on size
    # choose y st there's no overlap

    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
            
            if odd_condition in [1,2] and i==n_samples-1:
                odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects-1), replace=False)
            
            for j in range(n_objects):
                s = Shape()
                s1 = s.clone()
                s2 = s.clone()

                # a_ = a[i]

                # if odd_condition == 4 and i == n_samples-1:
                #     a_ = a_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi
                
                # s1.rotate(a_)

                if not (odd_condition == 2 and i == n_samples-1 and j in odd_samples):
                    s2.flip()
                
                # s2.rotate(a_)

                if odd_condition == 1 and i == n_samples-1 and j in odd_samples:
                    s_ = Shape()
                    if np.random.rand()>0.5:
                        s1 = s_
                    else:
                        s2 = s_

                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append(n_objs + n_objs_f)
            
    else:
        shape = []
        s = Shape()
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
            
            if odd_condition in [1,2] and i==n_samples-1:
                odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects-1), replace=False)
            
            for j in range(n_objects):

                s1 = s.clone()
                s2 = s.clone()

                # a_ = a[i]
                
                # if odd_condition == 4 and i == n_samples-1:
                #     a_ = a_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi

                # s1.rotate(a_)
                
                if not (odd_condition == 2 and i == n_samples-1 and j in odd_samples):
                    s2.flip()
                
                # s2.rotate(a_)

                if odd_condition == 1 and i == n_samples-1 and j in odd_samples:
                    s_ = Shape()
                    if np.random.rand()>0.5:
                        s1 = s_
                    else:
                        s2 = s_

                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append(n_objs + n_objs_f)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*n_objects*2)
        color = c.reshape([n_samples, n_objects*2, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects*2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects*2, 3]) * color for i in range(n_samples)]

    if variable=='x':
        for i in range(n_samples):
            xy[i], shape[i] = flip_diag_scene(xy[i], shape[i])
    
    return xy, size, shape, color

def task_size_sym_2(condition='cid'):

    n_samples = 4

    n_objects = 3

    xy0 = np.ones([n_samples, 1, 2]) * 0.5
    
    # odd: distance not correlated to size, no sym for n_objects
    odd_condition = np.random.randint(2)
    # odd_condition = 0
    
    max_size = 0.25
    min_size = max_size/5
    
    size = np.random.rand(n_samples, n_objects) * (max_size - min_size) + min_size
    size = np.concatenate([size]*2, 1)
    size_margin = 0.02
    size = (size-size.min(1)[:,None])/(size.max(1)-size.min(1))[:,None] * (max_size - min_size - size_margin*2) + min_size + size_margin
    # rad = np.random.rand(n_samples, n_objects) * (0.5 - size[:,:n_objects]*np.sqrt(2)) + size[:,:n_objects]*np.sqrt(2)/2
    # rad = np.concatenate([rad]*2, 1)
    
    # non_valid = np.pi - (size[:,:n_objects]*np.sqrt(2)/rad[:,:n_objects]).sum(1) < 0
    # if non_valid.any():
    #     size[non_valid] = size[non_valid] / (size[non_valid,:n_objects]*np.sqrt(2)/rad[non_valid,:n_objects]).sum(1) * np.pi * 0.9

    # keep rad or size

    # np.stack(np.random.permutation(n_objects*2)
    shuffle_perm, unshuffle_perm = zip(*[sample_shuffle_unshuffle_indices(n_objects*2) for _ in range(n_samples)])
    shuffle_perm, unshuffle_perm = np.stack(shuffle_perm, 0), np.stack(unshuffle_perm, 0)

    
    n_samples_over = 100

    angles_ = []
    xy_ = []
    rad_ = []
    triu_idx = np.triu_indices(n_objects*2, k=1)
    triu_idx = triu_idx[0]*n_objects*2 + triu_idx[1]

    for i in range(n_samples):
        shuf_size = size.copy()
        # shuffle_t(shuf_size, shuffle_perm)

        if i == n_samples-1 and odd_condition==0:
            # np.argmin(shuf_size)
            rad = np.random.rand(n_samples_over, n_objects) * (0.5 - shuf_size[i:i+1,:n_objects]*np.sqrt(2)) + shuf_size[i:i+1,:n_objects]*np.sqrt(2)/2
            rad = np.concatenate([rad]*2, 1)
        else:
            margin_r = 1.2
            rad = (shuf_size[i:i+1] - shuf_size[i:i+1].min()) / (shuf_size[i:i+1].max() - shuf_size[i:i+1].min()) * (0.5 - (shuf_size[i:i+1].max() + shuf_size[i:i+1].min())*margin_r*np.sqrt(2)/2 ) + shuf_size[i:i+1].min()*margin_r
            rad = rad * np.ones([n_samples_over, n_objects*2])


        # angles = sample_over_range_t(n_samples_over, np.array([0,2*np.pi]), shuf_size[i:i+1]*np.sqrt(2)/rad)
        # angles = np.random.choice([-3/4, -1/2, -1/4, 1/4, 1/2, 3/4, 1], size=[n_samples_over, n_objects*2]) * np.pi
        angles = np.random.rand(n_samples_over, n_objects) * np.pi * 2
        angles_diffs = np.random.choice([-1/2, 1/2, 1], size=[n_samples_over, n_objects]) * np.pi
        angles = np.concatenate([angles, angles+angles_diffs], 1)

        # shuffle_t(angles, unshuffle_perm)
        xy = xy0[0:1,:,:] + rad[:,:,None] * np.stack([np.cos(angles), np.sin(angles)], 2)[:,:,:]
        valid = (np.linalg.norm(xy[:,:,None,:] - xy[:,None,:,:], axis=-1) - (shuf_size[None,i,:,None]+shuf_size[None,i,None,:])*np.sqrt(2)/2 > 0).reshape([n_samples_over, (n_objects*2)**2])[:,triu_idx].all(1)
        while not valid.any():    

            if i == n_samples-1:
                rad = np.random.rand(n_samples_over, n_objects) * (0.5 - shuf_size[i:i+1,:n_objects]*np.sqrt(2)) + shuf_size[i:i+1,:n_objects]*np.sqrt(2)/2
                rad = np.concatenate([rad]*2, 1)

            # angles = sample_over_range_t(n_samples_over, np.array([0,2*np.pi]), shuf_size[i:i+1]*np.sqrt(2)/rad)
            # angles = np.random.rand(n_samples_over, n_objects*2) * np.pi * 2
            angles = np.random.rand(n_samples_over, n_objects) * np.pi * 2
            angles_diffs = np.random.choice([-1/2, 1/2, 1], size=[n_samples_over, n_objects]) * np.pi
            angles = np.concatenate([angles, angles+angles_diffs], 1)

            xy = xy0[0:1,:,:] + rad[i:i+1,:,None] * np.stack([np.cos(angles), np.sin(angles)], 2)[:,:,:]
            valid = (np.linalg.norm(xy[:,:,None,:] - xy[:,None,:,:], axis=-1) - (shuf_size[None,i,:,None]+shuf_size[None,i,None,:])*np.sqrt(2)/2 > 0).reshape([n_samples_over, (n_objects*2)**2])[:,triu_idx].all(1)

        angles = angles[valid][0]
        xy = xy[valid][0]
        rad = rad[valid][0]
        xy_.append(xy)
        rad_.append(rad)
        angles_.append(angles)

    rad = np.stack(rad_, 0)
    xy = np.stack(xy_, 0)
    angles = np.stack(angles_, 0)

    size = np.concatenate([np.ones([n_samples, 1])*0.03, size], 1)
    xy = np.concatenate([xy0, xy], 1) 

    s0 = Shape()

    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
            
            if odd_condition in [1,2] and i==n_samples-1:
                odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects), replace=False)
            
            for j in range(n_objects):
                s = Shape()
                s1 = s.clone()
                s2 = s.clone()

                s1.rotate(angles[i,j])
                s2.rotate(angles[i,n_objects+j])

                if odd_condition == 1 and i == n_samples-1 and j in odd_samples:
                    s_ = Shape()
                    if np.random.rand()>0.5:
                        s1 = s_
                    else:
                        s2 = s_

                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append([s0.clone()] + n_objs + n_objs_f)
            
    else:
        shape = []
        s = Shape()
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
            
            if odd_condition in [1,2] and i==n_samples-1:
                odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects-1), replace=False)
            
            for j in range(n_objects):

                s1 = s.clone()
                s2 = s.clone()

                s1.rotate(angles[i,j])
                s2.rotate(angles[i,n_objects+j])

                if odd_condition == 1 and i == n_samples-1 and j in odd_samples:
                    s_ = Shape()
                    if np.random.rand()>0.5:
                        s1 = s_
                    else:
                        s2 = s_


                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append([s0.clone()] + n_objs + n_objs_f)

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*(n_objects*2+1))
        color = c.reshape([n_samples, n_objects*2+1, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects*2+1, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects*2+1, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

def task_color_sym_1(condition='cidv'):

    n_samples = 4

    n_objects = np.random.randint(2, 5)

    xy0 = np.ones([n_samples, 1, 2]) * 0.5
    
    # odd: distance not correlated to size, no sym for n_objects
    odd_condition = np.random.randint(2)
    # odd_condition = 0
    
    max_size = 0.5/n_objects
    min_size = max_size/3
    
    size = np.random.rand(n_samples, n_objects) * (max_size - min_size) + min_size
    size = np.concatenate([size]*2, 1)
    size_margin = 0.02
    size = (size-size.min(1)[:,None])/(size.max(1)-size.min(1))[:,None] * (max_size - min_size - size_margin*2) + min_size + size_margin

    # np.stack(np.random.permutation(n_objects*2)
    shuffle_perm, unshuffle_perm = zip(*[sample_shuffle_unshuffle_indices(n_objects*2) for _ in range(n_samples)])
    shuffle_perm, unshuffle_perm = np.stack(shuffle_perm, 0), np.stack(unshuffle_perm, 0)
    
    n_samples_over = 100

    angles_ = []
    xy_ = []
    rad_ = []
    triu_idx = np.triu_indices(n_objects*2, k=1)
    triu_idx = triu_idx[0]*n_objects*2 + triu_idx[1]

    size = []
    for i in range(n_samples):
        # shuf_size = size.copy()
        # shuf_size = shuf_size

        size_ = np.random.rand(1, n_objects) * (max_size - min_size) + min_size
        size_ = np.concatenate([size_]*2, 1)
        size_margin = 0.02
        size_ = (size_-size_.min(1)[:,None])/(size_.max(1)-size_.min(1))[:,None] * (max_size - min_size - size_margin*2) + min_size + size_margin

        shuf_size = size_

        rad = np.random.rand(n_samples_over, n_objects) * (0.5 - shuf_size[:,:n_objects]*np.sqrt(2)) + shuf_size[:,:n_objects]*np.sqrt(2)/2
        rad = np.concatenate([rad]*2, 1)

        # angles = sample_over_range_t(n_samples_over, np.array([0,2*np.pi]), shuf_size[i:i+1]*np.sqrt(2)/rad)
        angles = np.random.rand(n_samples_over, n_objects*2) * np.pi * 2

        # shuffle_t(angles, unshuffle_perm)
        xy = xy0[0:1,:,:] + rad[:,:,None] * np.stack([np.cos(angles), np.sin(angles)], 2)[:,:,:]
        valid = (np.linalg.norm(xy[:,:,None,:] - xy[:,None,:,:], axis=-1) - (shuf_size[None,:,:,None]+shuf_size[None,:,None,:])*np.sqrt(2)/2 > 0).reshape([n_samples_over, (n_objects*2)**2])[:,triu_idx].all(1)
        while not valid.any():    

            size_ = np.random.rand(1, n_objects) * (max_size - min_size) + min_size
            size_ = np.concatenate([size_]*2, 1)
            size_margin = 0.02
            size_ = (size_-size_.min(1)[:,None])/(size_.max(1)-size_.min(1))[:,None] * (max_size - min_size - size_margin*2) + min_size + size_margin

            shuf_size = size_

            rad = np.random.rand(n_samples_over, n_objects) * (0.5 - shuf_size[:,:n_objects]*np.sqrt(2)) + shuf_size[:,:n_objects]*np.sqrt(2)/2
            rad = np.concatenate([rad]*2, 1)

            # angles = sample_over_range_t(n_samples_over, np.array([0,2*np.pi]), shuf_size[i:i+1]*np.sqrt(2)/rad)
            angles = np.random.rand(n_samples_over, n_objects*2) * np.pi * 2

            xy = xy0[0:1,:,:] + rad[i:i+1,:,None] * np.stack([np.cos(angles), np.sin(angles)], 2)[:,:,:]
            valid = (np.linalg.norm(xy[:,:,None,:] - xy[:,None,:,:], axis=-1) - (shuf_size[None,:,:,None]+shuf_size[None,:,None,:])*np.sqrt(2)/2 > 0).reshape([n_samples_over, (n_objects*2)**2])[:,triu_idx].all(1)
    
        angles = angles[valid][0]
        xy = xy[valid][0]
        rad = rad[valid][0]
        xy_.append(xy)
        rad_.append(rad)
        angles_.append(angles)
        size.append(size_)
    
    size = np.concatenate(size, 0)
    
    rad = np.stack(rad_, 0)
    xy = np.stack(xy_, 0)
    angles = np.stack(angles_, 0)

    size = np.concatenate([np.ones([n_samples, 1])*0.03, size], 1)
    xy = np.concatenate([xy0, xy], 1) 

    s0 = Shape()

    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
                        
            for j in range(n_objects):
                s = Shape()
                s1 = s.clone()
                s2 = s.clone()

                s1.rotate(angles[i,j])
                s2.rotate(angles[i,n_objects+j])

                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append([s0.clone()] + n_objs + n_objs_f)
            
    else:
        shape = []
        s = Shape()
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
            
            for j in range(n_objects):

                s1 = s.clone()
                s2 = s.clone()

                s1.rotate(angles[i,j])
                s2.rotate(angles[i,n_objects+j])

                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append([s0.clone()] + n_objs + n_objs_f)


    min_diff = 0.15

    h_u = sample_over_range_t(n_samples, np.array([0,1]), np.ones([n_samples, n_objects])*min_diff)

    if 's' in condition:
        s = (np.random.rand(n_samples, n_objects) * 0.6 + 0.4) * np.ones([n_samples, n_objects])
    else:
        s = np.random.rand() * 0.6 + 0.4
        s = s * np.ones([n_samples, n_objects])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples, n_objects) * 0.5 + 0.35)
    else:
        v = np.random.rand() * 0.5 + 0.35
        v = v * np.ones([n_samples, n_objects])
 
    color = np.stack([h_u,s,v], 2)
    
    color = np.concatenate([color[:,0:1]*0, color, color], 1)
    
    # odd condition
    h_ = color[-1,1,0]
    color[-1,1,0] = color[-1,2,0]
    color[-1,2,0] = h_

    return xy, size, shape, color

def task_color_sym_2(condition='cidv'):

    variable = 'x' if np.randon.randint(2)==0 else 'y'

    n_samples = 4

    n_objects = 5
    
    # odd: distance not correlated to size, no sym for n_objects
    # odd_condition = np.random.randint(2)
    odd_condition = -1

    # odd_condition = 0
    
    max_size = 0.25
    min_size = max_size/5
    
    size = np.random.rand(n_samples, n_objects) * (max_size - min_size) + min_size
    non_valid = size.sum(1) > 0.9
    if non_valid.any():
        size[non_valid] = size[non_valid]/size[non_valid].sum(1)[:,None] * 0.9

    margin_r = 1.3

    x_range = (0.5 - (size.max(1) + size.min(1))/2*margin_r)
    x_starting = size.min(1)/2*margin_r
    
    x1 = (size - size.min(1)[:,None]) / (size.max(1) - size.min(1))[:,None] * x_range[:,None] + x_starting[:,None]
    x1 = 0.5 - x1

    if odd_condition == 0:
        x1[-1] = np.random.rand(n_objects) * (0.5 - size[-1]) + size[-1]/2
        
    x2 = 1 - x1

    y = sample_over_range_t(n_samples, np.array([0,1]), size)

    size = np.concatenate([size]*2, axis=1)

    x = np.concatenate([x1,x2], 1)
    y = np.concatenate([y,y], 1)
    xy = np.stack([x,y], 2)
    # choose size
    # choose x based on size
    # choose y st there's no overlap

    
    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
            
            if odd_condition in [1,2] and i==n_samples-1:
                odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects-1), replace=False)
            
            for j in range(n_objects):
                s = Shape()
                s1 = s.clone()
                s2 = s.clone()


                if not (odd_condition == 2 and i == n_samples-1 and j in odd_samples):
                    s2.flip()
                
                # s2.rotate(a_)

                if odd_condition == 1 and i == n_samples-1 and j in odd_samples:
                    s_ = Shape()
                    if np.random.rand()>0.5:
                        s1 = s_
                    else:
                        s2 = s_

                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append(n_objs + n_objs_f)
            
    else:
        shape = []
        s = Shape()
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
            
            if odd_condition in [1,2] and i==n_samples-1:
                odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects-1), replace=False)
            
            for j in range(n_objects):

                s1 = s.clone()
                s2 = s.clone()

                # a_ = a[i]
                
                # if odd_condition == 4 and i == n_samples-1:
                #     a_ = a_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi

                # s1.rotate(a_)
                
                if not (odd_condition == 2 and i == n_samples-1 and j in odd_samples):
                    s2.flip()
                
                # s2.rotate(a_)

                if odd_condition == 1 and i == n_samples-1 and j in odd_samples:
                    s_ = Shape()
                    if np.random.rand()>0.5:
                        s1 = s_
                    else:
                        s2 = s_

                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append(n_objs + n_objs_f)


    min_diff = 0.15

    h_u = sample_over_range_t(n_samples, np.array([0,1]), np.ones([n_samples, n_objects])*min_diff)

    if 's' in condition:
        s = (np.random.rand(n_samples, n_objects) * 0.6 + 0.4) * np.ones([n_samples, n_objects])
    else:
        s = np.random.rand() * 0.6 + 0.4
        s = s * np.ones([n_samples, n_objects])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples, n_objects) * 0.5 + 0.35)
    else:
        v = np.random.rand() * 0.5 + 0.35
        v = v * np.ones([n_samples, n_objects])
 
    color = np.stack([h_u,s,v], 2)

    color = np.concatenate([color, color], 1)
    
    # odd condition
    h_ = color[-1,1,0]
    color[-1,1,0] = color[-1,2,0]
    color[-1,2,0] = h_

    if variable=='x':
        for i in range(n_samples):
            xy[i], shape[i] = flip_diag_scene(xy[i], shape[i])
    
    return xy, size, shape, color

def task_sym_sym_1(condition='cid'):
    
    # mir sym:
    # (x0,y0), r, a
    # o_1 = (0, a - pi/2) 
    # pos_1 = r * (cos(a1),sin(a1)) + (x0,y0)
    # o_2 = (1, a + pi/2)
    # pos_2 = r * (cos(a2),sin(a2)) + (x0,y0)
    # and

    n_samples = 4

    n_objects = 2

    # odd conditions: shape1 != shape2, size1 != size2, flip1 == flip2, position doesn't fit, incorrect angle, different ax
    # odd_condition = np.random.randint(6) # 5
    odd_condition = 5 # 5

    max_size = 0.4
    min_size = max_size/4
    
    size = np.random.rand(n_samples) * (max_size - min_size) + min_size
    size = np.stack([size]*2, axis=1)
    
    if odd_condition == 1:
        idx = np.random.randint(2)
        size[-1,idx] = size[-1,idx] * (np.random.rand()* (2/3 - 1/3) + 1/3)

    # a = (np.random.rand(n_samples) * 1 - 1/2) * np.pi 
    a = (np.random.rand() * 1 - 1/2) * np.pi 
    a = np.ones(n_samples) * a
    
    if odd_condition == 5:

        a_odd = (np.random.rand(100) * 1 - 1/2) * np.pi
        a_odd = a_odd[np.abs(a[0] - a_odd)>np.pi/3][0]

        a[-1] = a_odd

    r = np.random.rand(n_samples) * (0.4-size.max(1)) + size.max(1)/2

    # a1, a2 = a - np.pi/2, a + np.pi/2

    range_ = 1 - size[:,0:1] - 2 * np.abs(np.stack([r * np.cos(a), r * np.sin(a)], 1))
    starting_ = size[:,0:1]/2 + np.abs(np.stack([r * np.cos(a), r * np.sin(a)], 1))
    
    xy0 = np.random.rand(n_samples, 2) * range_ + starting_

    xy = xy0[:,None,:] + np.stack([r * np.cos(a), r * np.sin(a), r * np.cos(a + np.pi), r * np.sin(a + np.pi)], 1).reshape([n_samples, 2, 2])

    if odd_condition == 3:
        xy[-1] = sample_positions(size[-1:])
    
    
    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            s = Shape()
            s1 = s.clone()
            s2 = s.clone()
            a_ = a[i]

            if odd_condition == 4 and i == n_samples-1:
                a_ = a_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi
            
            s1.rotate(a_)

            if not (odd_condition == 2 and i == n_samples-1):
                s2.flip()
            
            s2.rotate(a_)

            if odd_condition == 0 and i == n_samples-1:
                s_ = Shape()
                if np.random.rand()>0.5:
                    s1 = s_
                else:
                    s2 = s_

            shape.append([s1, s2])
            
    else:
        shape = []
        s = Shape()
        for i in range(n_samples):
            s1 = s.clone()
            s2 = s.clone()

            a_ = a[i]
            
            if odd_condition == 4 and i == n_samples-1:
                a_ = a_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi

            s1.rotate(a_)
            
            if not (odd_condition == 2 and i == n_samples-1):
                s2.flip()
            
            s2.rotate(a_)

            if odd_condition == 0 and i == n_samples-1:
                s_ = Shape()
                if np.random.rand()>0.5:
                    s1 = s_
                else:
                    s2 = s_

            shape.append([s1, s2])

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*(n_objects))
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

def task_sym_sym_2(condition='cid'):
    
    # rot sym:
    # r
    # o_1 = (0, a1)
    # pos_1 = r * (cos(a1),sin(a1)) + (x0,y0)
    # o_2 = (0, a2)
    # pos_2 = r * (cos(a2),sin(a2)) + (x0,y0)
    # and


    n_samples = 4

    n_objects = 3

    # odd conditions: shape1 != shape2, size1 != size2, position doesn't fit, incorrect angle, different angle diff
    odd_condition = np.random.randint(5) # 5
    odd_condition = 4

    max_size = 0.4
    min_size = max_size/4
    
    size = np.random.rand(n_samples) * (max_size - min_size) + min_size
    # size = np.stack([np.ones(n_samples)*0.05]+[size]*2, axis=1)
    size = np.stack([size]*2, axis=1)
    
    a1 = (np.random.rand(n_samples) * 1 - 1/2) * np.pi

    min_diff = 1/4

    if odd_condition == 4:
        # diff = (np.random.rand() * 2/3 + 1/3)
        
        # odd_diff = (np.random.rand(100) * 2/3 + 1/3)
        # odd_diff = odd_diff[np.abs(odd_diff-diff[0]) >  np.pi/3][0]
        

        diffs = sample_over_range(np.array([min_diff/2, 1+min_diff/2]), np.ones(2)*min_diff) * np.pi

        if np.random.rand() >0.5:
            diffs = diffs[::-1]

        diff = np.ones(n_samples) * diffs[0]
        
        diff[-1] = diffs[1]

    else:
        diff = (np.random.rand() * (1-min_diff) + min_diff) * np.pi * np.ones(n_samples)

    a2 = diff + a1
    # a2 = (np.random.rand(n_samples) * 2/3 + 1/3) * np.pi +  a1

    # r = np.random.rand(n_samples) * (0.4-size.max(1)) + size.max(1)/2
    # 2*np.sin(a/2)*r > min_size
    wh = (np.max([np.zeros(n_samples), np.cos(a1), np.cos(a2)], 0) - np.min([np.zeros(n_samples), np.cos(a1), np.cos(a2)], 0), 
        np.max([np.zeros(n_samples), np.sin(a1), np.sin(a2)], 0) - np.min([np.zeros(n_samples), np.sin(a1), np.sin(a2)], 0))
    # max(wh) * r + max_size < 0.8
    max_r = (1 - size[:,0] * np.sqrt(2)) / np.max(wh, 0)
    min_r = (size[:,0] * np.sqrt(2))/(2*np.sin((a2-a1)/2))
    r = np.random.rand(n_samples) * (max_r - min_r) + min_r
    
    # max_size = np.min([max_size*np.ones(n_samples), r], 0) 
    

    if odd_condition == 1:
        idx = np.random.randint(2)
        size[-1,idx] = size[-1,idx] * (np.random.rand()* (2/3 - 1/3) + 1/3)

    # xy0 * 3 - (x1+x2)
    # x0, x1, x2 
    # y0, y1, y2

    # x = np.stack([0, r*np.cos(a1), r*np.cos(a2)], 1)
    # y = np.stack([0, r*np.sin(a1), r*np.sin(a2)], 1)
    max_x_pos = np.max([np.zeros(n_samples), r*np.cos(a1) + np.sqrt(2) * size[:,0]/2, r*np.cos(a1) - np.sqrt(2) * size[:,0]/2, r*np.cos(a2) + np.sqrt(2) * size[:,1]/2, r*np.cos(a2) - np.sqrt(2) * size[:,1]/2], 0)
    max_y_pos = np.max([np.zeros(n_samples), r*np.sin(a1) + np.sqrt(2) * size[:,0]/2, r*np.sin(a1) - np.sqrt(2) * size[:,0]/2, r*np.sin(a2) + np.sqrt(2) * size[:,1]/2, r*np.sin(a2) - np.sqrt(2) * size[:,1]/2], 0)

    wh = (  np.max([np.zeros(n_samples), r*np.cos(a1) + np.sqrt(2) * size[:,0]/2, r*np.cos(a1) - np.sqrt(2) * size[:,0]/2, r*np.cos(a2) + np.sqrt(2) * size[:,1]/2, r*np.cos(a2) - np.sqrt(2) * size[:,1]/2], 0) - 
            np.min([np.zeros(n_samples), r*np.cos(a1) + np.sqrt(2) * size[:,0]/2, r*np.cos(a1) - np.sqrt(2) * size[:,0]/2, r*np.cos(a2) + np.sqrt(2) * size[:,1]/2, r*np.cos(a2) - np.sqrt(2) * size[:,1]/2], 0),
            np.max([np.zeros(n_samples), r*np.sin(a1) + np.sqrt(2) * size[:,0]/2, r*np.sin(a1) - np.sqrt(2) * size[:,0]/2, r*np.sin(a2) + np.sqrt(2) * size[:,1]/2, r*np.sin(a2) - np.sqrt(2) * size[:,1]/2], 0) -
            np.min([np.zeros(n_samples), r*np.sin(a1) + np.sqrt(2) * size[:,0]/2, r*np.sin(a1) - np.sqrt(2) * size[:,0]/2, r*np.sin(a2) + np.sqrt(2) * size[:,1]/2, r*np.sin(a2) - np.sqrt(2) * size[:,1]/2], 0))

    wh = np.stack(wh, 1)
    range_ = 1 - wh
    starting_ = wh/2
    
    xy0 = np.random.rand(n_samples, 2) * range_ + starting_

    xy0[:,0] = xy0[:,0] + wh[:,0]/2 - max_x_pos
    xy0[:,1] = xy0[:,1] + wh[:,1]/2 - max_y_pos


    xy = xy0[:,None,:] + r[:,None,None] * np.stack([np.cos(a1),np.sin(a1),np.cos(a2),np.sin(a2)], 1).reshape([n_samples, 2, 2])

    xy = np.concatenate([np.ones([n_samples,1,2])*xy0[:,None,:], xy], 1) 
    size = np.concatenate([np.ones([n_samples,1])*0.05, size], 1)

    if odd_condition == 2:
        xy[-1] = sample_positions(size[-1:])
    
    

    if 'id' in condition:
        shape = []
        s0 = Shape()
        
        for i in range(n_samples):
            s = Shape()
            s1 = s.clone()
            s2 = s.clone()
            a1_ = a1[i]
            a2_ = a2[i]

            if odd_condition == 3 and i == n_samples-1:
                a1_ = a1_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi
                a2_ = a2_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi
            
            s1.rotate(a1_)            
            s2.rotate(a2_)

            if odd_condition == 0 and i == n_samples-1:
                s_ = Shape()
                if np.random.rand()>0.5:
                    s1 = s_
                else:
                    s2 = s_

            shape.append([s0.clone(), s1, s2])
            
    else:
        shape = []
        s0 = Shape()
        
        s = Shape()
        for i in range(n_samples):
            a1_ = a1[i]
            a2_ = a2[i]
            
            if odd_condition == 3 and i == n_samples-1:
                if np.random.rand() > 0.5:
                    a1_ = a1_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi
                else:
                    a2_ = a2_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi

            s1 = s.clone()
            s2 = s.clone()

            s1.rotate(a1_)            
            s2.rotate(a2_)

            if odd_condition == 0 and i == n_samples-1:
                s_ = Shape()
                if np.random.rand()>0.5:
                    s1 = s_
                else:
                    s2 = s_

            shape.append([s0.clone(), s1, s2])

    if 'c' in condition:
        # color = [sample_random_colors(n_objects[i]) for i in range(n_samples)]
        c = sample_random_colors(n_samples*(n_objects))
        color = c.reshape([n_samples, n_objects, 3])
        # color = sample_random_colors(n_samples)
        # color = [np.ones([n_objects, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([n_objects, 3]) * color for i in range(n_samples)]
    
    return xy, size, shape, color

def task_inside_sym_1(condition='xyc'):

    n_samples = 4

    n_objects = np.random.randint(2,5)
    
    n_objects_in = np.random.randint(1,n_objects)

    # odd: distance not correlated to size, no sym for n_objects
    # odd_condition = np.random.randint(2)
    odd_condition = -1

    # odd_condition = 0
    
    max_size = 0.9/n_objects
    min_size = max_size*2/3
    
    size = np.random.rand(n_samples, n_objects) * (max_size - min_size) + min_size

    size_in = np.random.rand(n_samples, n_objects_in) * (size[:,:n_objects_in]/2.5 - size[:,:n_objects_in]/4) + size[:,:n_objects_in]/4


    non_valid = size.sum(1) > 0.9
    if non_valid.any():
        size[non_valid] = size[non_valid]/size[non_valid].sum(1)[:,None] * 0.9

    margin_r = 1.3

    x_range = (0.5 - (size.max(1) + size.min(1))/2*margin_r)
    x_starting = size.min(1)/2*margin_r
    
    x1 = (size - size.min(1)[:,None]) / (size.max(1) - size.min(1))[:,None] * x_range[:,None] + x_starting[:,None]
    x1 = 0.5 - x1

    if odd_condition == 0:
        x1[-1] = np.random.rand(n_objects) * (0.5 - size[-1]) + size[-1]/2
        
    x2 = 1 - x1

    y = sample_over_range_t(n_samples, np.array([0,1]), size)

    size = np.concatenate([size]*2, axis=1)
    size_in = np.concatenate([size_in]*2, axis=1)

    size = np.concatenate([size, size_in], 1)

    x = np.concatenate([x1,x2], 1)
    y = np.concatenate([y,y], 1)
    xy = np.stack([x,y], 2)
    # choose size
    # choose x based on size
    # choose y st there's no overlap

    max_attempts = 10

    shape = []
    xy_in = []
    if 'id' not in condition:
        s = Shape(gap_max=0.06, hole_radius=0.2)
    for i in range(n_samples):
        n_objs = []
        n_objs_f = []
        shape_out = []
        shape_out_f = []
        shape_in = []
        shape_in_f = []
        xy_in_ = []
        xy_in_f = []
        # if odd_condition in [1,2] and i==n_samples-1:
        #     odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects-1), replace=False)
        
        for j in range(n_objects):

            
            if j<n_objects_in:

                if 'id' in condition:
                    s = Shape(gap_max=0.06, hole_radius=0.2)
    
                so = s.clone()
                si = Shape()

                done = False
                for _ in range(max_attempts):
                    samples = sample_position_inside_1(so, si, size_in[i,j]/size[i,j])
                    if len(samples)>0:
                        done = True
                    if done:
                        break
                    else:
                        # so.randomize()
                        si.randomize()

                if done:
                    xy_ = samples[0]

                    xy_in_.append(xy_ * size[i,j] + xy[i,j])
                else:
                    xy_in_.append(xy[i,j])


                # so1 = s.clone()
                so2 = so.clone()
                si2 = si.clone()

                # if not (odd_condition == 2 and i == n_samples-1 and j in odd_samples):
                if i != n_samples-1:
                    so2.flip()
                    si2.flip()
                    xy__ = np.copy(xy_in_[-1])
                    xy__[0] = 1 - xy__[0]

                    xy_in_f.append(xy__)
                
                else:

                    xy_in_f.append(xy_in_[-1] - xy[i,j] + xy[i,j+n_objects])

                shape_out.append(so)
                shape_out_f.append(so2)
                shape_in.append(si)
                shape_in_f.append(si2)


            else:

                odd_condition = np.random.randint(2)+1

                if 'id' in condition:
                    s = Shape()

                s1 = s.clone()
                s2 = s.clone()
                
                if i == n_samples-1:
                    s2.flip()
                else:
                    if not (odd_condition == 2):
                        s2.flip()

                    if odd_condition == 1:
                        s_ = Shape()
                        if np.random.rand()>0.5:
                            s1 = s_
                        else:
                            s2 = s_

                shape_out.append(s1)
                shape_out_f.append(s2)
        
        shape.append(shape_out + shape_out_f + shape_in + shape_in_f)
        xy_in.append(xy_in_ + xy_in_f)

    xy_in = np.array(xy_in)
    xy = np.concatenate([xy, xy_in], 1)

    if 'c' in condition:
        color = sample_random_colors(n_samples)
        color = [np.ones([(n_objects+n_objects_in)*2, 3]) * color[i:i+1] for i in range(n_samples)]
    else:
        color = sample_random_colors(1)
        color = [np.ones([(n_objects+n_objects_in)*2, 3]) * color for i in range(n_samples)]

    return xy, size, shape, color

def task_color_sym_2(condition='cidv'):

    variable = 'x' if np.random.randint(2)==0 else 'y'

    n_samples = 4

    n_objects = 6
    
    # odd: distance not correlated to size, no sym for n_objects
    # odd_condition = np.random.randint(2)
    odd_condition = -1

    # odd_condition = 0
    
    max_size = 0.20
    min_size = max_size/5
    
    size = np.random.rand(n_samples, n_objects) * (max_size - min_size) + min_size
    non_valid = size.sum(1) > 0.9
    if non_valid.any():
        size[non_valid] = size[non_valid]/size[non_valid].sum(1)[:,None] * 0.9

    margin_r = 1.3

    x_range = (0.5 - (size.max(1) + size.min(1))/2*margin_r)
    x_starting = size.min(1)/2*margin_r
    
    x1 = (size - size.min(1)[:,None]) / (size.max(1) - size.min(1))[:,None] * x_range[:,None] + x_starting[:,None]
    x1 = 0.5 - x1

    if odd_condition == 0:
        x1[-1] = np.random.rand(n_objects) * (0.5 - size[-1]) + size[-1]/2
        
    x2 = 1 - x1

    y = sample_over_range_t(n_samples, np.array([0,1]), size)

    size = np.concatenate([size]*2, axis=1)

    x = np.concatenate([x1,x2], 1)
    y = np.concatenate([y,y], 1)
    xy = np.stack([x,y], 2)
    # choose size
    # choose x based on size
    # choose y st there's no overlap

    
    if 'id' in condition:
        shape = []
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
            
            if odd_condition in [1,2] and i==n_samples-1:
                odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects-1), replace=False)
            
            for j in range(n_objects):
                s = Shape()
                s1 = s.clone()
                s2 = s.clone()


                if not (odd_condition == 2 and i == n_samples-1 and j in odd_samples):
                    s2.flip()
                
                # s2.rotate(a_)

                if odd_condition == 1 and i == n_samples-1 and j in odd_samples:
                    s_ = Shape()
                    if np.random.rand()>0.5:
                        s1 = s_
                    else:
                        s2 = s_

                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append(n_objs + n_objs_f)
            
    else:
        shape = []
        s = Shape()
        for i in range(n_samples):
            n_objs = []
            n_objs_f = []
            
            if odd_condition in [1,2] and i==n_samples-1:
                odd_samples = np.random.choice(n_objects, size=np.random.randint(1,n_objects-1), replace=False)
            
            for j in range(n_objects):

                s1 = s.clone()
                s2 = s.clone()

                # a_ = a[i]
                
                # if odd_condition == 4 and i == n_samples-1:
                #     a_ = a_ + np.random.choice([-1,1]) * (np.random.rand()* (1-1/2)+1/2)*np.pi

                # s1.rotate(a_)
                
                if not (odd_condition == 2 and i == n_samples-1 and j in odd_samples):
                    s2.flip()
                
                # s2.rotate(a_)

                if odd_condition == 1 and i == n_samples-1 and j in odd_samples:
                    s_ = Shape()
                    if np.random.rand()>0.5:
                        s1 = s_
                    else:
                        s2 = s_

                n_objs.append(s1)
                n_objs_f.append(s2)
            
            shape.append(n_objs + n_objs_f)


    min_diff = 0.15

    h_u = sample_over_range_t(n_samples, np.array([0,1]), np.ones([n_samples, n_objects])*min_diff)

    if 's' in condition:
        s = (np.random.rand(n_samples, n_objects) * 0.6 + 0.4) * np.ones([n_samples, n_objects])
    else:
        s = np.random.rand() * 0.6 + 0.4
        s = s * np.ones([n_samples, n_objects])
        
    if 'v' in condition:
        v = (np.random.rand(n_samples, n_objects) * 0.5 + 0.35)
    else:
        v = np.random.rand() * 0.5 + 0.35
        v = v * np.ones([n_samples, n_objects])
 
    color = np.stack([h_u,s,v], 2)

    color = np.array([color[i, np.random.permutation(n_objects)] for i in range(n_samples)])

    color = np.concatenate([color, color], 1)
    
    # odd condition
    
    h_ = color[-1,1,0]
    color[-1,1,0] = color[-1,2,0]
    color[-1,2,0] = h_

    if variable=='x':
        for i in range(n_samples):
            xy[i], shape[i] = flip_diag_scene(xy[i], shape[i])
    
    return xy, size, shape, color


TASKS=[
    ["task_shape", task_shape, "The images contain objects of the same shape."],
    ["task_pos", task_pos, "The images contain objects in the same position."],
    ["task_size", task_size, "The images contain objects of the same size."],
    ["task_color", task_color, "The images contain objects of the same color."],
    ["task_rot", task_rot, "The images contain objects of the same shape with different rotations."],
    ["task_flip", task_flip, "The images contain objects of the same shape with different flips."],
    ["task_count", task_count, "The images contain the same number of objects."],
    ["task_inside", task_inside, "The images contain an object inside another object."],
    ["task_contact", task_contact, "The images contain an object in contact with another object."],
    ["task_sym_rot", task_sym_rot, "The images contain 2 objects in a rotational symmetry around a center indicated by a third small object."],
    ["task_sym_mir", task_sym_mir, "The images contain 2 objects in a mirror symmetry."],
    ["task_pos_pos_1", task_pos_pos_1, "The images contain a set of objects that have the same spatial configuration."],
    ["task_pos_pos_2", task_pos_pos_2, "Each image contains 2 sets of objects that have the same spatial configuration."],
    ["task_pos_count_2", task_pos_count_2, "The images contain a sets of objects that are aligned. The number of objects in the set is the same."],
    ["task_pos_count_1", task_pos_count_1, "In each image, the relationship between the number of objects on one side and the number of objects on the other side is the same. For example, if the number of objects on the left is always bigger than the number of objects on the right. "],
    ["task_pos_pos_4", task_pos_pos_4, "The images contain the same number of objects. One of the objects will maintain the same order over both spatial dimensions across images."],
    ["task_pos_count_3", task_pos_count_3, "The images contain sets of aligned objects. The number of sets is the same in all images."],
    ["task_inside_count_1", task_inside_count_1, "The images contain the same number of objects that contain an object."],
    ["task_count_count", task_count_count, "The number of objects is either odd or even in all images."],
    ["task_shape_shape", task_shape_shape, "Each image contain 2 similarly shaped objects."],
    ["task_shape_contact_2", task_shape_contact_2, "Each image contains 3 objects among which 2 have the same shape and 2 are in contact. The objects with similar shapes are in contact in all images."],
    ["task_contact_contact_1", task_contact_contact_1, "In each image, there is one group of connected objects."],
    ["task_inside_inside_1", task_inside_inside_1, "Each image contains 3 objects. In all images, 1 object contains the 2 other objects."],
    ["task_inside_inside_2", task_inside_inside_2, "In each image, one object contains all the other objects."],
    ["task_pos_inside_3", task_pos_inside_3, "Objects that contain objects are always on the same side of the image."],
    ["task_pos_inside_1", task_pos_inside_1, "In each image, an object contains another object. The inner object is always on the same side of the outer object. "],
    ["task_pos_inside_2", task_pos_inside_2, "In each image, an object contains 2 objects. The 2 inner objects are always positioned similarly within the outer object.  "],
    ["task_pos_inside_4", task_pos_inside_4, "In each image, the object that contains another object is always positioned similarly with respect to the other objects across the same dimension."],
    ["task_rot_rot_1", task_rot_rot_1, "Each image contain 2 similarly shaped objects with different rotations."],
    ["task_flip_flip_1", task_flip_flip_1, "Each image contain 2 similarly shaped objects flipped differently."],
    ["task_rot_rot_3", task_rot_rot_3, "Each image contain 2 similarly shaped objects, the difference between rotation of the 2 objects is the same in all images."],
    ["task_pos_pos_3", task_pos_pos_3, "Each image contains 2 pairs of objects. Objects of each pair have the same position in one spatial dimension (for example the x axis) but different positions in the other dimension (for example the y axis). The distance between 2 objects of a pair along the second dimension and the position of the pair along the first dimension maintain the same order across images. "],
    ["task_pos_count_4", task_pos_count_4, "The images contain 3 sets of aligned objects. The sum of the numbers of objects in 2 sets equals the number of objects in the third set."],
    ["task_size_size_1", task_size_size_1, "Each image contains 2 objects. The first object is smaller than the second object in all images."],
    ["task_size_size_2", task_size_size_2, "Each image contains 3 objects. 2 objects of each image have similar sizes and the other object has a different size."],
    ["task_size_size_3", task_size_size_3, "Each image contains 3 objects. The 3 objects of each image have different sizes."],
    ["task_size_size_4", task_size_size_4, "Each image contains 2 objects. In each image, one object's size is half the other object's size."],
    ["task_size_size_5", task_size_size_5, "Each image contains 3 pairs of aligned objects. The object sizes in each pair are maintained in all images."],
    ["task_size_sym_1", task_size_sym_1, "Each image contains objects in a mirror symmetry with respect to the same axis. The distance to the axis is correlated with the size for each object."],
    ["task_size_sym_2", task_size_sym_2, "Each image contains objects in a rotational symmetry with respect to the center of the image. The distance to the center is correlated with the size for each object."],
    ["task_color_color_1", task_color_color_1, "Each image contains 2 objects with similar color hues."],
    ["task_color_color_2", task_color_color_2, "All objects in an image have the same hue."],
    ["task_sym_sym_1", task_sym_sym_1, "Each image contains 2 objects that have a mirror symmetry. The axis of symmetry is the same in all images."],
    ["task_sym_sym_2", task_sym_sym_2, "Each image contains 2 objects that have a rotational symmetry. The angle of symmetry is the same in all images."],
    ["task_shape_contact_3", task_shape_contact_3, "Each image contains 3 objects among which 2 have the same shape and 2 are in contact. The objects with similar shapes are in contact in all images. The objects have different sizes and colors."],
    ["task_shape_contact_4", task_shape_contact_4, "Each image contains 3 objects among which 2 have the same shape and 2 are in contact. The objects with similar shapes are in contact in all images. The objects have different colors."],
    ["task_contact_contact_2", task_contact_contact_2, "In each image, each object is in contact with another object."],
    ["task_pos_size_1", task_pos_size_1, "In each image, the large object is always positioned similarly with respect to the small object along the same dimension."],
    ["task_pos_size_2", task_pos_size_2, "In each image, the large object is always positioned similarly with respect to the small object along the same dimension."],
    ["task_pos_shape_1", task_pos_shape_1, "In each image, the object with the first shape is always positioned similarly with respect to the other object along the same dimension."],
    ["task_pos_shape_2", task_pos_shape_2, "Each image contains many pairs of close objects. The pairs, identified by the object shapes, are similar across images."],
    ["task_pos_rot_1", task_pos_rot_1, "The images contain a set of objects that have the same spatial configuration. The spatial configurations are rotated and objects in the configuration are rotated with the same angle."],
    ["task_pos_rot_2", task_pos_rot_2, "Each image contains 2 sets of objects that have the same spatial configuration. One of the spatial configurations is rotated and objects in the configuration are rotated with the same angle."],
    ["task_pos_col_1", task_pos_col_1, "In each image, the objects are always positioned similarly along the same dimension with respect to their colors."],
    ["task_pos_col_2", task_pos_col_2, "The images contain a set of objects that have the same spatial configuration and color configuration. Both configurations are maintained in all images."],
    ["task_pos_contact", task_pos_contact, "The images contain two objects in contact along the same direction."],
    ["task_size_shape_1", task_size_shape_1, "The images contain two pairs of objects. Each pair of objects have the same size and shape."],
    ["task_size_shape_2", task_size_shape_2, "The images contain 4 objects with 4 different shapes (a,b,c,d). When a is bigger than b, c is bigger d. Conversly, when a is smaller than b, c is smaller than d. "],
    ["task_size_rot", task_size_rot, "Each image contain 3 objects with the same shape. The smallest is rotated to the left and the largest is rotated to the right with respect to the average one."],
    ["task_size_inside_1", task_size_inside_1, "In each image, one of the objects contains an object. The object that contains an object is either always the larger one or always the smaller one."],
    ["task_size_contact", task_size_contact, "The images contain 4 objects a,b,c and d. The contact and size configurations are constant across images. For example, a contact configuration is A being in contact with B and C being in contact with D. A size configuration is A and B having the same size which is bigger than C and D."],
    ["task_size_count_1", task_size_count_1, "Each image contains sets of objects. Within each set, objects have the same size. All images have the same number of objects in each set. "],
    ["task_size_count_2", task_size_count_2, "Each image contains sets of objects. Within each set, objects have the same size. All images have the same number of sets. "],
    ["task_shape_color", task_shape_color, "Each image contains 2 objects. The association between shapes and colors is fixed across images. "],
    ["task_shape_color_2", task_shape_color_2, "In each image, objects with the same shapes have the same color. "],
    ["task_shape_color_3", task_shape_color_3, "In each image, objects with the same shapes have different colors. "],
    ["task_shape_inside", task_shape_inside, "Each image contains 2 objects, A and B, whose shapes don't change across images. In each image, A is inside B."],
    ["task_shape_inside_1", task_shape_inside_1, "Each image contains several objects. In each image, if an object is inside another object, they share the same shape."],
    ["task_shape_count_1", task_shape_count_1, "Each image contains sets of objects. Within each set, objects have the same shape. The number of objects in each set is constant across images."],
    ["task_shape_count_2", task_shape_count_2, "Each image contains sets of objects. Within each set, objects have the same shape. The number of sets is constant across images."],
    ["task_rot_color", task_rot_color, "In each image objects with the same shapes have the same color. All objects are randomly rotated. "],
    ["task_rot_inside_1", task_rot_inside_1, "Each image contains 2 pairs of objects. In each pair, one of the objects is inside the other. One of the pairs is a rotation of the other pair."],
    ["task_rot_inside_2", task_rot_inside_2, "In each image, if an object contains on object, they share the same shape. The objects are randomly rotated."],
    ["task_rot_count_1", task_rot_count_1, "Each image contains sets of objects. Within each set objects have the same shape and are randomly rotated. The number of objects in each set is constant across images."],
    ["task_color_inside_1", task_color_inside_1, "Each image contains 2 objects, A and B, whose colors don't change across images. In each image, A is inside B."],
    ["task_color_inside_2", task_color_inside_2, "Each image contains several objects. In each image, if an object is inside another object, they share the same color."],
    ["task_color_contact", task_color_contact, "The images contain 4 objects A,B,C and D. The contact and color configurations are constant across images. For example, a contact configuration is A being in contact with B and C being in contact with D. A color configuration is A and B having the same color which is different from the colors of C and D."],
    ["task_color_count_1", task_color_count_1, "Each image contains sets of objects. Within each set, objects have the same color. The number of objects in each set is constant across images."],
    ["task_color_count_2", task_color_count_2, "Each image contains sets of objects. Within each set, objects have the same color. The number of sets is constant across images."],
    ["task_inside_contact", task_inside_contact, "The images contain 4 objects A,B,C and D, 2 of which contain an object. The insideness and contact configurations are constant across images. For example, a contact configuration is A being in contact with B and C being in contact with D. An insideness configuration is A and B containing an object and C and D containing no objects."],
    ["task_contact_count_1", task_contact_count_1, "Each image contains sets of objects. Within each set, objects are in contact. The number of objects in each set is constant across images."],
    ["task_contact_count_2", task_contact_count_2, "Each image contains sets of objects. Within each set, objects are in contact. The number of sets is constant across images."],
    ["task_size_color_1", task_size_color_1, "Each image contains 2 objects. The larger object always has the same color."],
    ["task_size_color_2", task_size_color_2, "In each image, objects with the same size have the same color."],
    ["task_color_sym_1", task_color_sym_1, "In each image, each pair of objects are in a rotational symmetry around the center of the image. The objects of each pair have the same color."],
    ["task_color_sym_2", task_color_sym_2, "In each image, each pair of objects are in a mirror symmetry around the same axis. The objects of each pair have the same color."],
    ["task_shape_rot_1", task_shape_rot_1, "Each image contains 2 objects that have similar shapes and a third differently shaped object. The similarly shaped objects are randomly rotated. The shapes are the same across images. "],
    ["task_shape_contact_5", task_shape_contact_5, "In each image, similarly shaped objects are in contact."],
    ["task_rot_contact_1", task_rot_contact_1, "Each image contains 3 objects among which 2 have the same shape and 2 are in contact. The objects with similar shapes are in contact and randomly rotated in all images."],
    ["task_rot_contact_2", task_rot_contact_2, "In each image, similarly shaped objects are in contact. All objects are randomly rotated."],
    ["task_inside_sym_mir", task_inside_sym_mir, "In each image, only objects that contain other objects are in mirror symmetry over the same axis."],
    ["task_flip_count_1", task_flip_count_1, "Each image contains sets of objects. Within each set objects have the same shape and are randomly flipped. The number of objects in each set is constant across images."],
    ["task_flip_inside_1", task_flip_inside_1, "Each image contains 2 pairs of objects. In each pair, one of the objects is inside the other. One of the pairs is a flip of the other pair."],
    ["task_flip_inside_2", task_flip_inside_2, "In each image, if an object contains on object, they share the same shape. The objects are randomly flipped."],
    ["task_flip_color_1", task_flip_color_1, "In each image objects with the same shapes have the same color. All objects are randomly flipped. "],
    ["task_shape_flip_1", task_shape_flip_1, "Each image contains 2 objects that have similar shapes and a third differently shaped object. The similarly shaped objects are randomly flipped. The shapes are the same across images. "],
    ["task_rot_flip_1", task_rot_flip_1, "Each image contains 4 objects. All 4 objects have the same shape and are placed in the same 4 locations. Objects aligned vertically have a the same difference in rotation angle and objects aligned vertically are flipped differently."],
    ["task_size_flip_1", task_size_flip_1, "Each image contain 3 objects with the same shape. The smallest object is vertically flipped and the largest object is horizantally flipped with respect to the average one."],
    ["task_pos_rot_3", task_pos_rot_3, "Each image contain 3 objects with the with successive positions along a spatial dimension. The rotation angles remain the same for each position across images."],
    ["task_pos_flip_1", task_pos_flip_1, "Each image contain 3 objects with the with successive positions along a spatial dimension. The flips remain the same for each position across images."],
    ["task_pos_flip_2", task_pos_flip_2, "Each image contains 2 objects with similar shapes. One of the objects is flipped along the axis formed by the two objects."],
    ["task_flip_contact_1", task_flip_contact_1, "Each image contains 3 objects among which 2 have the same shape and 2 are in contact. The objects with similar shapes are in contact and randomly flipped in all images."],
    ["task_flip_contact_2", task_flip_contact_2, "In each image, similarly shaped objects are in contact. All objects are randomly flipped."],
]



# if __name__ == '__main__':
    
#     seed = 0

#     base_path = '../cvrt_images_gen/'
#     os.makedirs(base_path, exist_ok=True)

#     n_samples = 1
    
#     for tn, tfn in TASKS:
#         np.random.seed(seed)
#         # images = np.concatenate([tfn() for i in range(n_samples)], 0)
#         images = np.concatenate([tfn() for i in range(n_samples)], 0)
#         ims = np.split(images, 4, axis=1)
#         ims[-1][:2,:,0] = 255
#         ims[-1][:,:2,0] = 255
#         ims[-1][-2:,:,0] = 255
#         ims[-1][:,-2:,0] = 255
#         ims = np.concatenate([np.concatenate([ims[0], ims[1]], 0), np.concatenate([ims[2], ims[3]], 0)], 1)
#         save_image(ims, base_path, tn)
    
    