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

abs_train_img = './isic-challenge-2017/ISIC-2017_Training_Data'
abs_train_mask = './isic-challenge-2017/ISIC-2017_Training_GroundTruth'

abs_test_img = './isic-challenge-2017/ISIC-2017_Test_Data'
abs_test_mask = './isic-challenge-2017/ISIC-2017_Test_GroundTruth'

abs_val_img = './isic-challenge-2017/ISIC-2017_Validation_Data'
abs_val_mask = './isic-challenge-2017/ISIC-2017_Validation_GroundTruth'

def find_mean(img_path):
    imgs = sorted([absp(os.path.join(img_path, i)) for i in os.listdir(img_path) if not i.startswith('.')])
    mean = np.zeros((256,256,3))
    for i in tqdm(imgs):
        imgs_np = np.asarray(Image.open(i)) / 255.
        mean = mean + imgs_np
    
    mean = mean / len(imgs)

    return mean

def find_std_dev(img_path, mean):
    imgs = sorted([absp(os.path.join(img_path, i)) for i in os.listdir(img_path) if not i.startswith('.')])
    err = np.zeros((256,256,3))
    for i in imgs:
        imgs_np = np.asarray(Image.open(i)) / 255.
        err = err + np.square(imgs_np - mean)

    err = err / len(imgs)
    return np.sqrt(err)

def data_gen(img_path = abs_train_img, mask_path = abs_train_mask, batch_size = 1, random_state = 7, center = False, norm = False, mean = None, std_dev = None, sample_wise_normalize = False):
    imgs = sorted([absp(os.path.join(img_path, i)) for i in os.listdir(img_path) if not i.startswith('.')])
    masks = sorted([absp(os.path.join(mask_path, i)) for i in os.listdir(mask_path) if not i.startswith('.')])
    
    if norm: center = True
    if center: 
        if (mean is None): mean = find_mean(img_path)

    if norm:
        if (std_dev is None): std_dev = find_std_dev(img_path, mean)

    if len(imgs) != len(masks):
        print("img-mask mismatch")
        return

    data = [(i,j) for i,j in zip(imgs, masks)]
    
    random.seed(random_state)
    random.shuffle(data)
    
    n = len(data)
    imgs_np = np.zeros((batch_size, 256, 256, 3))
    masks_np = np.zeros((batch_size,256,256))
    
    i = 0

    while True:
        for c in range(batch_size):
            imgs_np[c] = np.asarray(Image.open(data[i][0])) / 255.
            if sample_wise_normalize : imgs_np[c] = tf.image.per_image_standardization(imgs_np[c])
            masks_np[c] = np.asarray(Image.open(data[i][1])) / 255.
            i+=1
            i%=n 
        if center: imgs_np = imgs_np - mean
        if norm: imgs_np = imgs_np / std_dev
        yield imgs_np, masks_np

#################### Pure Tensorflow Data-Generators Tensorflow ##########################

def find_mean_tf(img_path = abs_train_img):
    imgs = sorted([absp(os.path.join(img_path, i)) for i in os.listdir(img_path) if not i.startswith('.')])
    mean = tf.zeros((256,256,3))
    for i in tqdm(imgs):
        mean = mean + get_img(i)
    mean = mean / len(imgs)
    return mean

def find_std_dev_tf(img_path = abs_train_img, mean = None):
    if mean is None:
        return
    std_dev = tf.zeros((256,256,3))
    imgs = sorted([absp(os.path.join(img_path, i)) for i in os.listdir(img_path) if not i.startswith('.')])
    for i in tqdm(imgs):
        std_dev = std_dev + tf.square(get_img(i) - mean)
    std_dev = std_dev / len(imgs)
    return tf.sqrt(std_dev)

def get_mask(mask_pth):
    mask = tf.io.read_file(mask_pth)
    mask = tf.reshape(tf.io.decode_image(mask, dtype = tf.dtypes.float32), (256,256))
    return mask

def get_img(img_path, mean = None, std_dev = None):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img, dtype = tf.dtypes.float32)
    if mean is None: return img
    img = img - mean
    if std_dev is None: return img
    img = img / std_dev
    return img



def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def augment_util(img, mask):
  img = img.numpy()
  mask = mask.numpy()
  im_merge = np.concatenate((img[...,None], mask[...,None]), axis=3)
  im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
  return im_merge_t


def process(file_ds, mean = None, std_dev = None):

    img = tf.reshape(get_img(file_ds[0], mean, std_dev),(256,256,3))
    mask = tf.reshape(get_mask(file_ds[1]), (256,256,1))

    img = tf.convert_to_tensor(img)
    mask = tf.convert_to_tensor(mask)

    val = tf.random.uniform(shape=[], minval=0, maxval=1)
    if val>0.5:
      img = tf.image.flip_left_right(img)
      mask = tf.image.flip_left_right(mask)
    
    val = tf.random.uniform(shape=[], minval=0, maxval=1)
    if val>0.5:
      img = tf.image.flip_up_down(img)
      mask = tf.image.flip_up_down(mask)
  
    val = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, val)
    mask = tf.image.rot90(mask,val)

    mask = tf.reshape(mask,(256,256))

    return (img, mask)

def tf_dataset(img_path = abs_train_img, mask_path = abs_train_mask, batch_size = 10, cache = False, random_state = 7, center = False, mean = None, std_dev = None):
    
    imgs = sorted([absp(os.path.join(img_path, i)) for i in os.listdir(img_path) if not i.startswith('.')])
    masks = sorted([absp(os.path.join(mask_path, i)) for i in os.listdir(mask_path) if not i.startswith('.')])
    random.seed(random_state)

    data = [(i,j) for i,j in zip(imgs, masks)]

    if center:
        if mean is None: mean = find_mean_tf()
        if std_dev is None: std_dev = find_std_dev_tf(abs_train_img, mean)
    else:
        mean = None
        std_dev = None

    random.shuffle(data)
    list_ds = tf.data.Dataset.from_tensor_slices(data)
    ds = list_ds.map(lambda x : process(x , mean, std_dev), num_parallel_calls = AUTOTUNE)
    if type(cache) == str:
        ds = ds.cache(cache)
    elif cache:
        ds = ds.cache()
    
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size = AUTOTUNE)

    return ds