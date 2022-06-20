import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import cv2

import matplotlib
matplotlib.use('Agg')

def cat_lists(lists):
    o = []
    for l in lists:
        o += l
    return o


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h*6.0) # XXX assume int() truncates!
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    if i == 0: 
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

# helper functions
def sample_position_inside_1(s1, s2, scale):
    c1 = s1.get_contour()
    c2 = s2.get_contour()
    
    c2 = c2 * scale
    bb_2 = c2.max(0) - c2.min(0)

    # sampling points
    range_ = (c1.max(0) - c1.min(0) - bb_2)
    starting = (c1.min(0) + bb_2/2)
    samples = np.random.rand(100, 2) * range_[None,:] + starting[None,:]

    p1c = np.concatenate([c1[:-1], c1[1:]], 1)[None,:,:]
    samples = samples[:,None,:]
    res = np.logical_and(
        np.logical_or(
            p1c[:,:,0:1] < samples[:,:,0:1], 
            p1c[:,:,2:3] < samples[:,:,0:1]), 
        np.logical_xor(
            p1c[:,:,1:2] <= samples[:,:,1:2], 
            p1c[:,:,3:4] <= samples[:,:,1:2])
        )[:,:,0]
    res1 = (res.sum(1)%2==1)
    res2 = (np.abs(samples - c1) > bb_2[None,None,:]/2).any(2).all(1)

    res = np.logical_and(res1, res2)

    samples = samples[res,0]
    # if len(samples)==0:
    #     print('')
        
    return samples

def sample_position_inside_many(s1, shapes, scales):
    c1 = s1.get_contour()
    c2s = [s2.get_contour() for s2 in shapes]

    # c2s = [c2 * scale for c2,scale in zip(c2s, scales)]
    
    # c2s = c2s * np.array(scales)[:,None,None]
    # bbs_2 = c2s.max(1) - c2s.min(1) # [n_shapes, 2]
    
    bbs_2 = np.array([c2.max(0) - c2.min(0) for c2 in c2s]) * np.array(scales)[:,None]
    # bbs_2 = bbs_2 * np.array(scales)[:,None,None]
    
    n_shapes = len(shapes)

    # sampling points
    ranges_ = (c1.max(0)[None,:] - c1.min(0)[None,:] - bbs_2)
    starting = (c1.min(0)[None,:] + bbs_2/2)
    samples = np.random.rand(500, n_shapes, 2) * ranges_[None, :, :] + starting[None, :, :]

    dists = np.abs(samples[:,:,None,:] - samples[:,None,:,:]) - (bbs_2[None,:,None,:] + bbs_2[None,None,:,:])/2 > 0
    triu_idx = np.triu_indices(n_shapes, k=1)[0]*n_shapes + np.triu_indices(n_shapes, k=1)[1]
    no_overlap = dists.any(3).reshape(500, n_shapes*n_shapes)[:, triu_idx].all(1)
    
    samples = samples[no_overlap]

    n_samples_left = len(samples)
    bb_2_ = np.concatenate([bbs_2]*n_samples_left, 0)
    
    samples = samples.reshape([n_samples_left*n_shapes, 2])
    
    p1c = np.concatenate([c1[:-1], c1[1:]], 1)[None,:,:]
    samples = samples[:,None,:]
    res = np.logical_and(
        np.logical_or(
            p1c[:,:,0:1] < samples[:,:,0:1], 
            p1c[:,:,2:3] < samples[:,:,0:1]), 
        np.logical_xor(
            p1c[:,:,1:2] <= samples[:,:,1:2], 
            p1c[:,:,3:4] <= samples[:,:,1:2])
        )[:,:,0]
    res1 = (res.sum(1)%2==1)
    res2 = (np.abs(samples - c1[None,:,:]) > bb_2_[:,None,:]/2).any(2).all(1)
    
    res = np.logical_and(res1, res2)
    res = res.reshape([-1, n_shapes]).all(1)
    samples = samples.reshape([-1, n_shapes, 2])

    # samples = samples[res,0]
    samples = samples[res]
        
    return samples


def sample_int_sum_n(n_numbers, s, min_v=0):
    samples = np.random.rand(n_numbers)
    samples = samples/samples.sum()*s
    samples = np.ceil(samples).astype(int)
    samples[samples<min_v] = min_v
    
    while samples.sum()>s:
        diff = samples.sum() - s    
        idx = np.where(samples>min_v)[0]
        if diff<len(idx):
            idx = np.random.choice(idx, size=diff, replace=False)
        samples[idx] -=1
    return samples    


# different n values that cover a range without overlapping with minimum distances between them
def sample_over_range(range_, min_dists):
    n_values = len(min_dists)
    
    dists = np.random.rand(n_values)
    dists = dists / dists.sum() * (range_[1] - range_[0] - min_dists.sum())
    dists[0] = dists[0] * np.random.rand()
    dists = dists + min_dists
    v = np.cumsum(dists)
    v = v - min_dists[0]/2 + range_[0]

    return v

def sample_over_range_t(n_samples, range_, min_dists):
    if len(range_.shape) == 1:
        range_ = range_[None,:]
    if len(min_dists.shape) == 1:
        min_dists = min_dists[None,:]

    n_values = min_dists.shape[1]
    
    dists = np.random.rand(n_samples, n_values)
    dists = dists / dists.sum(1)[:,None] * (range_[:,1] - range_[:,0] - min_dists.sum(1)[:,None])
    dists[:,0] = dists[:,0] * np.random.rand(n_samples)
    dists = dists + min_dists
    v = np.cumsum(dists, 1)
    v = v - min_dists/2 + range_[:,0:1]

    return v


def sample_positions(size, n_sample_min=1, max_tries=10, n_samples_over=100):
    max_tries = 10
    i = 0
    n_samples_over = 100

    n_objects = size.shape[1]

    triu_idx = np.triu_indices(n_objects, k=1)
    triu_idx = triu_idx[0]*n_objects + triu_idx[1]
    xy_ = []
    xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size[:,:,None]) + size[:,:,None]/2
    valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,None]+size[:,None,:,None])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
    if valid.any():
        xy_ = xy[valid][:n_sample_min]
    # while  not valid.any() and i<max_tries:
    while  len(xy_) < n_sample_min and i<max_tries:
        xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size[:,:,None]) + size[:,:,None]/2
        valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,None]+size[:,None,:,None])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
        if valid.any():
            if len(xy_)==0:
                xy_ = xy[valid][:n_sample_min-len(xy_)]
            else:
                xy_ = np.concatenate([xy_, xy[valid][:n_sample_min-len(xy_)]], 0)

        i+=1

    if len(xy_) == 0:
        xy_ = xy[:n_sample_min]
    elif len(xy_) < n_sample_min:
        xy_ = np.concatenate([xy_, xy[valid][:n_sample_min-len(xy_)]], 0)
        
    return xy_



def sample_positions_bb(size, n_sample_min=1, max_tries=10, n_samples_over=100):
    max_tries = 10
    i = 0
    n_samples_over = 100

    n_objects = size.shape[1]

    triu_idx = np.triu_indices(n_objects, k=1)
    triu_idx = triu_idx[0]*n_objects + triu_idx[1]
    xy_ = []
    xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size) + size/2
    valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,:]+size[:,None,:,:])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
    if valid.any():
        xy_ = xy[valid][:n_sample_min]
    # while  not valid.any() and i<max_tries:
    while  len(xy_) < n_sample_min and i<max_tries:
        xy = np.random.rand(n_samples_over, n_objects, 2) * (1-size) + size/2
        valid = (np.abs(xy[:,:,None,:] - xy[:,None,:,:]) - (size[:,:,None,:]+size[:,None,:,:])/2 > 0).any(3).reshape([n_samples_over, n_objects**2])[:,triu_idx].all(1)
        if valid.any():
            if len(xy_)==0:
                xy_ = xy[valid][:n_sample_min-len(xy_)]
            else:
                xy_ = np.concatenate([xy_, xy[valid][:n_sample_min-len(xy_)]], 0)

        i+=1

    if len(xy_) == 0:
        xy_ = xy[:n_sample_min]
    elif len(xy_) < n_sample_min:
        xy_ = np.concatenate([xy_, xy[valid][:n_sample_min-len(xy_)]], 0)
        
    return xy_


def sample_random_colors(n_samples):
    h = np.random.rand(n_samples)
    s = np.random.rand(n_samples) * 0.5 + 0.5
    v = np.random.rand(n_samples) * 1
    # v = np.random.rand(n_samples) * 0.3 + 0.7

    color = np.stack([h,s,v],1)
    return color


def sample_shuffle_unshuffle_indices(n):
    perm = np.random.permutation(n)
    indices_input = np.arange(n)
    indices_output = indices_input[perm]
    rev_perm = (indices_output[:, None] == indices_input).argmax(axis=0)
    return perm, rev_perm


def shuffle_t(t, perms):
    # t.reshape()
    for i in range(t.shape[0]):
        t[i] = t[i, perms[i]]


def sample_contact(s1, s2, scale, direction=0):
    c1 = s1.get_contour()
    c2 = s2.get_contour()
    
    c2 = c2 * scale
    
    if direction==0:
        p1 = np.argmax(c1[:,0]) 
        p2 = np.argmin(c2[:,0]) 
    elif direction==1:
        p1 = np.argmin(c1[:,0]) 
        p2 = np.argmax(c2[:,0]) 
    elif direction==2:
        p1 = np.argmax(c1[:,1]) 
        p2 = np.argmin(c2[:,1]) 
    elif direction==3:
        p1 = np.argmin(c1[:,1]) 
        p2 = np.argmax(c2[:,1]) 

    xy2 = (c2.max(0) + c2.min(0))/2 - c2[p2] + c1[p1]
    xy1 = np.zeros(2)

    
    return xy2


def sample_contact_many(shapes, sizes, a=None):
    n_objects = len(shapes)
    contours = [shapes[i].get_contour() * sizes[i] for i in range(n_objects)]

    # intialize clump as the first object
    clump = contours[0]
    positions = np.zeros([1,2])
    clump_size = np.ones(2) * sizes[0]
    for i in range(1, n_objects):
        # sample direction
        if a is None:
            angle = np.random.rand() * 2 * np.pi
        elif isinstance(a, float):
            angle = a
        else:
            angle = a[i]
            
        pos2 = (sizes[i]+clump_size) * np.array([np.cos(angle), np.sin(angle)])[None,:]
        
        idx_p_contact_clump = (clump * (np.cos(angle),np.sin(angle))).sum(-1) > 0
        idx_p_contact_object = (contours[i] * (np.cos(angle),np.sin(angle))).sum(-1) < 0
        
        # move object in direction
        c = contours[i] + pos2
        
        idx_min = np.linalg.norm(clump[idx_p_contact_clump][:,None,:] - c[idx_p_contact_object][None,:,:], axis=2).argmin()
        s_ = idx_p_contact_object.sum()
        idx_min_clump, idx_min_object = idx_min // s_, idx_min % s_
        p_clump = clump[idx_p_contact_clump][idx_min_clump]
        p_obj = contours[i][idx_p_contact_object][idx_min_object]
        new_pos = (p_clump - p_obj)*(1-4/128)
        
        clump = np.concatenate([clump, contours[i]+new_pos[None,:]], 0)
        bb = clump.min(0), clump.max(0)
        
        clump = clump - (bb[1] + bb[0])/2
        clump_size = bb[1] - bb[0]

        positions = np.concatenate([positions,new_pos[None,:]], 0)
        positions = positions - (bb[1] + bb[0])/2

    return positions, clump_size

def flip_diag_scene(xys, shapes):

    # for s_ in shapes:
    #     for s in s_:
    #         s.flip_diag()

    # for i, xy_ in enumerate(xys):
    #     for j, xy in enumerate(xy_):
    #         xys[i][j] = xy[::-1]

    for s in shapes:
        s.flip_diag()

    for i, xy in enumerate(xys):
        xys[i] = xy[::-1]

    return xys, shapes

# def render_cv(xy, size, shapes, color=None, axis='y', image_size=128):
def render_cv(xy, size, shapes, color=None, image_size=128):
    # # image_size = 128

    # if color is None:
    #     color = np.ones([len(shapes), 3])
    #     color[:,0] = np.random.rand()               # h
    #     color[:,1] = np.random.rand() * 0.5 + 0.5   # s
    #     color[:,2] = 0.7                            # v

        
    color = [hsv_to_rgb(c[0], c[1], c[2]) for c in color]

    image = (np.ones([image_size,image_size, 3]) * 255).astype(np.uint8)

    # contours = []
    for i in range(len(shapes)):
        size_ = size[i]
        s_ = shapes[i]
        s_.scale(size_)
        xy_ = xy[i]

        c = s_.get_contour()
        
        c = (c*image_size).astype(int)

        c_ = np.concatenate([c,c[0:1]],0)
        dist = np.abs(c_[1:] - c_[:-1]) 
        c = c[(dist>0).any(1)]

        c = c + (xy_[None,:] * image_size).astype(int)

        # if axis=='x':
        #     c = c[:,::-1]

        # contours.append(c)
        col_ = (np.array(color[i])*255).tolist()
        cv2.drawContours(image, [c], -1, col_, 1)
        
    return image


def render_ooo(xy, size, shape, color, image_size=128):

    images = []
    for i in range(len(shape)):
        im = render_cv(xy[i], size[i], shape[i], color[i], image_size=128)
        im = np.pad(im, [[4,4], [4,4], [0,0]], constant_values=0)
        images.append(im)

    images = np.concatenate(images, axis=1)
    
    return images


def save_image_human_exp(images, meta, base_path):
    im_shape = images.shape
    
    dim_0 = im_shape[1]//4

    pad = (dim_0-128)//2

    images = images.reshape([im_shape[0]//dim_0, dim_0, 4, dim_0, 3]).transpose([0,2,1,3,4]).reshape([-1, dim_0, dim_0, 3])
    for i in range(len(images)):
        idx1, idx2 = i//4, i%4 
        save_path = os.path.join(base_path, '{:02d}_{}.png'.format(idx1, idx2))
        img = Image.fromarray(images[i,pad:dim_0-pad, pad:dim_0-pad]).convert('RGB')
        img.save(save_path)


def save_image_bin(images, base_path, task_name):
    save_path = base_path + '{}.png'.format(task_name)
    if images.dtype != np.uint8:
        images = (images*255).astype(np.uint8)
    img = Image.fromarray(images).convert('1')
    img.save(save_path)



def save_image(images, base_path, task_name):
    save_path = base_path + '{}.png'.format(task_name)
    # if images.dtype != np.uint8:
    #     images = (images*255).astype(np.uint8)
    img = Image.fromarray(images).convert('RGB')
    img.save(save_path)


def generate_dataset(task_name, task_fn, data_path='/media/data_cifs_lrs/projects/prj_visreason/cvrt_data/', seed=0, train_size=10000, val_size=500,  test_size=1000, test_gen_size=1000):
    # data_path = '/home/aimen/projects/cvrt_git/algs_images/'
    # data_path = '/media/data_cifs_lrs/projects/prj_visreason/cvrt_data/'

    task_path = os.path.join(data_path, task_name)
    
    n_train_samples_0 = 0
    n_train_samples_1 = train_size

    n_val_samples_0 = 0
    n_val_samples_1 = val_size

    n_test_samples_0 = 0
    n_test_samples_1 = test_size


    os.makedirs(task_path, exist_ok=True)
    os.makedirs(os.path.join(task_path,'train'), exist_ok=True)
    os.makedirs(os.path.join(task_path,'val'), exist_ok=True)
    os.makedirs(os.path.join(task_path,'test'), exist_ok=True)
    os.makedirs(os.path.join(task_path,'test_gen'), exist_ok=True)

    np.random.seed(seed)
    split = 'train'
    for i in range(n_train_samples_0, n_train_samples_1):
        # images, metadata = task()
        images = task_fn()
        # save_path = os.path.join(task_path, split, '{:05d}.bmp'.format(i))
        # img = Image.fromarray(images).convert('1')

        save_path = os.path.join(task_path, split, '{:05d}.png'.format(i))
        img = Image.fromarray(images).convert('RGB')
        img.save(save_path)

    np.random.seed(seed+1)
    split = 'val'
    for i in range(n_val_samples_0, n_val_samples_1):
        # images, metadata = task()
        images = task_fn()
        # save_path = os.path.join(task_path, split, '{:05d}.bmp'.format(i))
        # img = Image.fromarray(images).convert('1')

        save_path = os.path.join(task_path, split, '{:05d}.png'.format(i))
        img = Image.fromarray(images).convert('RGB')
        img.save(save_path)

    np.random.seed(seed+2)
    split = 'test'
    for i in range(n_test_samples_0, n_test_samples_1):
        # images, metadata = task()
        images = task_fn()
        # save_path = os.path.join(task_path, split, '{:05d}.bmp'.format(i))
        # img = Image.fromarray(images).convert('1')

        save_path = os.path.join(task_path, split, '{:05d}.png'.format(i))
        img = Image.fromarray(images).convert('RGB')
        img.save(save_path)

    np.random.seed(seed+2)
    split = 'test_gen'
    for i in range(n_test_samples_0, n_test_samples_1):
        # images, metadata = task()
        images = task_fn()
        # save_path = os.path.join(task_path, split, '{:05d}.bmp'.format(i))
        # img = Image.fromarray(images).convert('1')

        save_path = os.path.join(task_path, split, '{:05d}.png'.format(i))
        img = Image.fromarray(images).convert('RGB')
        img.save(save_path)


def experiment_details():
    
    # curriculum condition

    n_groups = 1

    n_subjects_group = 20

    n_elementary_tasks = 3
    
    # combinations of 2
    n_combinations = n_elementary_tasks ** 2

    # n_training_tasks_group = n_elementary_tasks + n_combinations
    
    n_training_tasks_group = 68

    n_evaluation_tasks_group = 0

    n_trials_per_task = 20

    time_per_trial = 10

    base_hourly_fee = 8.5
    full_hourly_fee = 8.5 * 1.33 # prolific fees


    ####

    n_tasks = (n_training_tasks_group + n_evaluation_tasks_group) * n_groups

    n_tasks_group = n_training_tasks_group + n_evaluation_tasks_group * n_groups
    n_trials_subjects = n_trials_per_task * n_tasks_group
    n_trials_group = n_trials_subjects * n_subjects_group
    n_trials = n_trials_group * n_groups
    n_subjects = n_subjects_group * n_groups

    time_subject = (n_trials_per_task* (time_per_trial+2) + 5) * n_tasks_group / 3600 

    fee_subject = time_subject * full_hourly_fee 
    
    fee_total = fee_subject * n_subjects
    
    out_dict = [
        ["n_combinations", n_combinations],
        ["n_training_tasks_group", n_training_tasks_group],
        ["n_tasks_all", n_tasks],
        ["n_tasks_group", n_tasks_group],
        ["n_trials_subjects", n_trials_subjects],
        ["n_trials_group", n_trials_group],
        ["n_trials", n_trials],
        ["n_subjects", n_subjects],
        ["time_subject", time_subject],
        ["fee_subject", fee_subject],    
        ["fee_total", fee_total],    
    ]

    out = '{}: {}\n'* (len(out_dict))
    out_strs = []
    for s in out_dict:
        out_strs += s
    out = out.format(*out_strs)
    print(out)
    
