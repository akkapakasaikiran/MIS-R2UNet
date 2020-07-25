
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import os
import pickle
from tqdm import tqdm

AUTOTUNE = tf.data.experimental.AUTOTUNE
print(tf.test.is_gpu_available())

absp = lambda x : os.path.abspath(x) 
absjoin = lambda x, y : absp(os.path.join(x,y))
abs_train_img = './rbv/patches128/'
abs_train_mask  = './rbv/patches128/'

def find_mean_tf(img_path, shape):
    if type(img_path) == str: imgs = sorted([absjoin(img_path, i) for i in os.listdir(img_path)])
    else: imgs = img_path
    mean = tf.zeros(shape)
    for i in tqdm(imgs):
        mean = mean + get_patch(i)
    mean = mean / len(imgs)
    return mean

def find_std_dev_tf(img_path, shape, mean):
    if type(img_path) == str : sorted([absjoin(img_path, i) for i in os.listdir(img_path)])
    else: imgs = img_path

    std_dev = tf.zeros(shape)

    for i in tqdm(imgs):
        std_dev = std_dev + tf.square(get_patch(i, False, mean))
    std_dev = std_dev / len(imgs)
    return tf.sqrt(std_dev)

def get_patch(pth, img_wise_norm= False, mean = None, std_dev = None):
    img = tf.io.read_file(pth)
    img = tf.io.decode_image(img, dtype = tf.dtypes.float32)
    if not (mean is None): img = img - mean
    if not (std_dev is None): img = img / std_dev
    if img_wise_norm: img = tf.image.per_image_standardization(img)
    return img

def process_training_data(data_point, shape, img_wise_norm = False, mean = None, std_dev = None):
    return tf.reshape(get_patch(data_point[0], img_wise_norm, mean, std_dev), shape) , tf.reshape(get_patch(data_point[1], False, None, None), (shape[0], shape[1]))

def tf_dataset(img_path = abs_train_img + 'patch_img', mask_path = abs_train_mask + 'patch_gt', batch_trn = 1, batch_val = 1, shape = (128,128,3), split = 0.8, cache = False, random_state = 7, img_wise_norm = False, center = False, mean = None, std_dev = None):
    
    random.seed(random_state)
    imgs = sorted([absjoin(img_path, i) for i in os.listdir(img_path)])
    masks = sorted([absjoin(mask_path, i) for i in os.listdir(mask_path)])

    data = [(i,j) for i,j in  zip(imgs, masks)]
    random.shuffle(data)
    
    n = len(data)
    n_train = int(n*split)
    n_val = n - n_train

    data_train = data[:n_train]
    data_val = data[n_train:]

    list_ds_trn = tf.data.Dataset.from_tensor_slices(data_train)
    list_ds_val = tf.data.Dataset.from_tensor_slices(data_val)

    if img_wise_norm:
        center = False
        mean = None
        std_dev = None

    if center:
        if (mean is None): mean = find_mean_tf([i for i,j in data_train], shape)
        if (std_dev is None): std_dev = find_std_dev_tf([i for i,j in data_train], shape, mean)

    ds_trn = list_ds_trn.map(lambda x : process_training_data(x , shape, img_wise_norm, mean, std_dev), num_parallel_calls = AUTOTUNE)
    ds_val = list_ds_val.map(lambda x : process_training_data(x , shape, img_wise_norm, mean, std_dev), num_parallel_calls = AUTOTUNE)
    
    if type(cache) == str:
        ds_trn, ds_val = ds_trn.cache(cache + 'trn'), ds_val.cache(cache + 'val')
    elif cache:
        ds_trn = ds_trn.cache()

    ds_trn, ds_val = ds_trn.repeat().batch(batch_trn).prefetch(buffer_size = AUTOTUNE), ds_val.repeat().batch(batch_val).prefetch(buffer_size = AUTOTUNE)
    
    return ds_trn, ds_val

def tf_dataset_real_time(data_path, batch_trn = 1, batch_val = 1, shape = (64,64,3), split = 0.8, random_state = 7, center = False, ret_norm_factors = False, stride = 10, rate = 1, img_wise_norm = False):
    
    absp = lambda x , y : os.path.abspath(os.path.join(x,y))
    imgp  = lambda x : os.path.abspath(os.path.join(data_path, 'img256', x))
    gtp = lambda x : os.path.abspath(os.path.join(data_path, 'gt256', x))

    list_img = sorted([i for i in os.listdir(os.path.join(data_path, 'img256'))])
    list_gt = sorted([i for i in os.listdir(os.path.join(data_path, 'gt256'))])
    n = len(list_img)
    imgs = np.zeros((n, 256, 256, shape[2]), dtype = "float32")
    gts = np.zeros((n, 256, 256, 1), dtype = "float32")

    print('reading images...')
    for j,i in enumerate(list_img):
        im = np.asarray(Image.open(imgp(i))) if shape[2] == 3 else np.asarray(Image.open(imgp(i)).convert('L'))[..., np.newaxis]
        imgs[j] = im

    for j,i in enumerate(list_gt):
        gt = np.asarray(Image.open(gtp(i)))
        gts[j] = gt[... , np.newaxis]

    imgs = imgs / 255.
    gts = (gts / 255.)

    if ret_norm_factors:
        return np.mean(imgs, axis = 0), np.sqrt(np.mean(np.square(imgs - np.mean(imgs, axis = 0)), axis = 0))
    
    if img_wise_norm:
        center = False
        imgs = tf.map_fn(tf.image.per_image_standardization, imgs)
    
    if center:
        print('calculating mean...')
        mean = np.mean(imgs, axis = 0)
        print('calculating std dev...')
        std_dev =  np.sqrt(np.mean(np.square(imgs - mean), axis = 0))
        print('normalizing images')
        imgs = (imgs - mean) / std_dev
    
    print('getting patches ...')
    patch_imgs = tf.image.extract_patches(imgs, sizes = [1, shape[0], shape[1], 1], rates=[1, rate, rate, 1], strides = [1,stride,stride,1], padding = 'VALID')
    patch_gts = tf.image.extract_patches(gts, sizes = [1, shape[0], shape[1], 1], rates=[1, rate, rate, 1], strides = [1,stride,stride,1], padding = 'VALID')
    sh = patch_imgs.shape

    final_shape = (sh[0]*sh[1]*sh[2], shape[0], shape[1], shape[2])
    print('init shape:', sh, 'reshaping into:', final_shape)

    print('reshping tensors ...')
    patch_imgs = tf.reshape(patch_imgs, final_shape)
    patch_gts = tf.reshape(patch_gts, (sh[0]*sh[1]*sh[2], shape[0], shape[1]))
    print('shape after:', patch_imgs.shape)
    
    total = sh[0]*sh[1]*sh[2]
    print('total ptaches:', total)

    trn_n = int(split*total)
    print('trn_patches:', trn_n)

    print('preparing datasets...')
    ds1 = tf.data.Dataset.from_tensor_slices(patch_imgs[:trn_n])
    ds2 = tf.data.Dataset.from_tensor_slices(patch_gts[:trn_n])
    ds3 = tf.data.Dataset.from_tensor_slices(patch_imgs[trn_n:])
    ds4 = tf.data.Dataset.from_tensor_slices(patch_gts[trn_n:])
    
    ds1, ds2 = ds1.repeat().batch(batch_trn).prefetch(buffer_size = AUTOTUNE), ds2.repeat().batch(batch_trn).prefetch(buffer_size = AUTOTUNE)
    ds3, ds4 = ds3.repeat().batch(batch_val).prefetch(buffer_size = AUTOTUNE), ds4.repeat().batch(batch_val).prefetch(buffer_size = AUTOTUNE)

    ds_trn = zip(ds1, ds2)
    ds_val = zip(ds3, ds4)
    print('complete...')
    return ds_trn, ds_val
#################### Pure Tensorflow Data-Generators Tensorflow ##########################

# def find_mean_tf(img_path = abs_train_img):
#     imgs = sorted([absp(os.path.join(img_path, i)) for i in os.listdir(img_path) if not i.startswith('.')])
#     mean = tf.zeros((256,256,3))
#     for i in tqdm(imgs):
#         mean = mean + get_img(i)
#     mean = mean / len(imgs)
#     return mean

# def find_std_dev_tf(img_pth = abs_train_img, mean = None):
#     if mean is None:
#         return
#     std_dev = tf.zeros((256,256,3))
#     imgs = sorted([absp(os.path.join(img_path, i)) for i in os.listdir(img_path) if not i.startswith('.')])
#     for i in tqdm(imgs):
#         std_dev = std_dev + tf.square(get_img(i) - mean)
#     std_dev = std_dev / len(imgs)
#     return std_dev

# def get_mask(mask_pth):
#     mask = tf.io.read_file(mask_pth)
#     mask = tf.reshape(tf.io.decode_image(mask, dtype = tf.dtypes.float32), (256,256))
#     return mask

# def get_img(img_path, mean = None, std_dev = None):
#     img = tf.io.read_file(img_path)
#     img = tf.io.decode_image(img, dtype = tf.dtypes.float32)
#     if mean is None: return img
#     img = img - mean
#     if std_dev is None: return img
#     img = img / std_dev
#     return tf.reshape(img, (256,256,3))

# def process(file_ds, mean = None, std_dev = None):
#     return tf.reshape(get_img(file_ds[0], mean, std_dev),(256,256,3)), tf.reshape(get_mask(file_ds[1]), (256,256))

# def tf_dataset(img_path = abs_train_img, mask_path = abs_train_mask, batch_size = 10, cache = False, random_state = 7, center = False, mean = None, std_dev = None):
    
#     imgs = sorted([absp(os.path.join(img_path, i)) for i in os.listdir(img_path) if not i.startswith('.')])
#     masks = sorted([absp(os.path.join(mask_path, i)) for i in os.listdir(mask_path) if not i.startswith('.')])
#     random.seed(random_state)

#     data = [(i,j) for i,j in zip(imgs, masks)]

#     if center:
#         if mean is None: mean = find_mean_tf()
#         if std_dev is None: std_dev = find_std_dev(abs_train_img, mean)
#     else:
#         mean = None
#         std_dev = None

#     random.shuffle(data)
#     list_ds = tf.data.Dataset.from_tensor_slices(data)
#     ds = list_ds.map(lambda x : process(x , mean, std_dev), num_parallel_calls = AUTOTUNE)
#     if type(cache) == str:
#         ds = ds.cache(cache)
#     elif cache:
#         ds = ds.cache()
    
#     ds = ds.repeat()
#     ds = ds.batch(batch_size)
#     ds = ds.prefetch(buffer_size = AUTOTUNE)

#     return ds