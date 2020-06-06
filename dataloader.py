import numpy as np
import tensorflow as tf
from glob import glob
from conf import *


TRAIN_IMAGES = sorted(glob(config_map["data_root"]+"/train/image/*"))
TRAIN_MASKS  = sorted(glob(config_map["data_root"]+"/train/mask/*"))
VAL_IMAGES  = sorted(glob(config_map["data_root"]+"/train/image/*"))
VAL_MASKS   = sorted(glob(config_map["data_root"]+"/train/mask/*"))


H,W = config_map["dim"],config_map["dim"]

def read_images_mask(image,mask):
	im = tf.io.read_file(image)
	im = tf.io.decode_image(im,channels=3)
	ma = tf.io.read_file(mask)
	ma = tf.io.decode_image(ma,channels=1)
	#ma = tf.image.rgb_to_grayscale(ma)
	im = tf.cast(im,tf.float32)/255.
	ma = tf.cast(ma,tf.float32)/255.
	return im,ma



def random_rotate(image,mask):
    prob = np.random.randint(0,10,1)[0]
    if prob > 6:
        rotates = np.random.randint(0,15,1)[0]
        if rotates > 10:
            im = tf.image.rot90(image,k=1)
            ma = tf.image.rot90(mask,k=1)
        else:
            im = tf.image.rot90(image,k=2)
            ma = tf.image.rot90(mask,k=2)
    else:
        im = image
        ma = mask
        
    return im,ma      



def change_il(image):
    if np.random.randint(0,16,1)[0] > 13:
        im_proc = tf.image.random_contrast(image,lower=0.3,upper=0.6)
    if np.random.randint(0,16,1)[0] > 13:
        im_proc = tf.image.random_saturation(image,lower=0.3,upper=0.6)
    else:
        im_proc = image
    return im_proc 


def resize_pad(image,mask,H,W):
    m = 26
    n = 52
    o = 104
    p = 416
    im = tf.image.resize_with_pad(image,H,W)
    ma1 = tf.image.resize_with_pad(mask,m,m)
    ma2 = tf.image.resize_with_pad(mask,n,n)
    ma3 = tf.image.resize_with_pad(mask,o,o)
    ma4 = tf.image.resize_with_pad(mask,p,p)
    return im,(ma1,ma2,ma3,ma4)



def central_crop(image,mask):
    if tf.random.uniform((),minval=1,maxval=14) > 10:
        im = tf.image.central_crop(image,0.9)
        ma = tf.image.central_crop(mask,0.9)
    if tf.random.uniform((),minval=1,maxval=14) > 13:
        im = tf.image.central_crop(image,0.6)
        ma = tf.image.central_crop(mask,0.6)
    else:
        im = image
        ma = mask
    return im,ma
	
@tf.function
def traingen(ima,mas):
    i,m = read_images_mask(ima,mas)
    i   = change_il(i)
    i,m = random_rotate(i,m)
    #i,m = central_crop(i,m)
    i,(m1,m2,m3,m4) = resize_pad(i,m,H,W)
    
    return i,(m1,m2,m3,m4)

@tf.function
def valgen(ima,mas):
    i,m = read_images_mask(ima,mas)
    i,(m1,m2,m3,m4) = resize_pad(i,m,H,W)
    return i,(m1,m2,m3,m4)

BUFFER = len(TRAIN_IMAGES)
val_list = len(VAL_IMAGES)
BATCH_SIZE=config_map["batch"]
TRAIN = tf.data.Dataset.from_tensor_slices((TRAIN_IMAGES,TRAIN_MASKS))
VAL = tf.data.Dataset.from_tensor_slices((VAL_IMAGES,VAL_MASKS))

TRAIN = TRAIN.map(traingen,num_parallel_calls=tf.data.experimental.AUTOTUNE)
VAL = VAL.map(valgen,num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ = TRAIN.batch(BATCH_SIZE,drop_remainder=True).shuffle(40).repeat()
val_ = VAL.batch(BATCH_SIZE,drop_remainder=True).shuffle(40).repeat()
ts = BUFFER//BATCH_SIZE
vs = val_list//BATCH_SIZE
