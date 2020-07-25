import tensorflow as tf
from PIL import Image
import numpy as np
import random
import os
import pickle
from tqdm import tqdm
import sys
import elasticdeform
import cv2
from scipy.ndimage.filters import gaussian_filter
import tifffile as tiff
from scipy.ndimage.interpolation import map_coordinates
sys.path.append('/content/drive/My Drive/data')
path = '/content/drive/My Drive/data/isbi/files'

files = [os.path.abspath(os.path.join(path, i)) for i in os.listdir(path)]

AUTOTUNE = tf.data.experimental.AUTOTUNE
print(tf.test.is_gpu_available())

img_list = tiff.imread(files[2])
label_list = tiff.imread(files[1])

print(tf.__version__)
def find_mean_tf(imgs = img_list):
  mean = tf.zeros((512,512))
  for i in tqdm(imgs):
    mean += get_img(i)
  mean /= len(imgs)
  return mean

def find_std_dev(imgs = img_list, mean = None):
  if mean == None:
    return
  std_dev = tf.zeros((512,512), dtype = tf.float32)
  for i in tqdm(imgs):
    std_dev += tf.square(get_img(i)-mean)
  std_dev = std_dev / len(imgs)
  std_dev = tf.math.sqrt(std_dev)
  return std_dev


def get_mask(masks = label_list, mask_no = 0):
    mask = masks[mask_no]
    print(mask.dtype)
    return mask

def get_img(img, mean = None, std_dev = None):
    if mean is None: return img
    img = img - mean
    if std_dev is None: return img
    img = img / std_dev
    return tf.reshape(img, (512, 512))

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
  im_merge = np.concatenate((img[...,None], mask[...,None]), axis=2)
  im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
  return im_merge_t

def process(pair, mean = None, std_dev = None):
    

    img = get_img(pair[0], mean, std_dev)
    mask = pair[1]

    img = tf.convert_to_tensor(img)
    mask = tf.convert_to_tensor(mask)
    
    print(img.dtype)
    print(mask.dtype)

    img = tf.reshape(img,(512,512,1))
    mask = tf.reshape(mask, (512,512,1))

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
  
    img1 = tf.reshape(img,(512,512))
    mask1 = tf.reshape(mask, (512,512))
 
    im_merge_t = tf.py_function(augment_util, inp=[img1,mask1], Tout = tf.float32)
    
    img = im_merge_t[...,0]
    mask = im_merge_t[...,1]
  

    img = tf.reshape(img,(512,512,1))
    mask = tf.reshape(mask, (512,512))
    #plt.imshow(im_t)

    print('hi')
    return (img, mask)

  
#process((img_list[0], label_list[0]))



def tf_dataset(imgs = img_list, masks = label_list, batch_size = 5, cache = False, random_state = 7, center = False, mean = None, std_dev = None):
    
    imgs = tf.convert_to_tensor(imgs, dtype=tf.float32)/255.0
    masks = tf.convert_to_tensor(masks, dtype=tf.float32)/255.0
    random.seed(random_state)

    data = [(i,j) for i,j in zip(imgs, masks)]

    if center:
        if mean is None: mean = find_mean_tf()
        if std_dev is None: std_dev = find_std_dev(imgs, mean)
    else:
        mean = None
        std_dev = None

   #random.shuffle(data)
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



ds = tf_dataset()

'''train_it = iter(ds)
for i in range(35):
  k = next(train_it)
  img = k[0]
  mask = k[1]
  img1 = tf.reshape(img[0],(512,512))
  mask1 = tf.reshape(mask[0], (512,512))
  fig = plt.figure(i+1)
  fig.suptitle(i)
  plt.subplot(1,2,1)
  plt.imshow(img1)
  plt.subplot(1,2,2)
  plt.imshow(mask1)'''