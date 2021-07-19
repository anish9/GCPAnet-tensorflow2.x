import numpy as np
import tensorflow as tf
from glob import glob
from conf import *


TRAIN_IMAGES = sorted(glob(config_map["data_root"]+"/train/image/*"))
TRAIN_MASKS  = sorted(glob(config_map["data_root"]+"/train/mask/*"))
VAL_IMAGES  = sorted(glob(config_map["data_root"]+"/val/image/*"))
VAL_MASKS   = sorted(glob(config_map["data_root"]+"/val/mask/*"))


H,W = config_map["dim"],config_map["dim"]

x_limit,y_limit = config_map["crop_patch_interval"]

def read_images_mask_train(image,mask):
    im = tf.io.read_file(image)
    im = tf.io.decode_png(im,channels=3)
    pr1 = np.random.randint(1,15,1)[0]
    pr2 = np.random.randint(1,15,1)[0]
    if pr1 > 14:
        jpeg_quality_ = np.random.randint(80,95,1)[0]
        im = tf.image.adjust_jpeg_quality(im, jpeg_quality=jpeg_quality_)
    if pr2 > 10:
        gamma_ = (np.random.randint(800,1330,1)[0])/1000
        im =  tf.image.adjust_gamma(im, gamma=gamma_, gain=1)

    ma = tf.io.read_file(mask)
    ma = tf.io.decode_png(ma,channels=1)
    im = tf.cast(im,tf.float32)/255.
    ma = tf.cast(ma,tf.float32)/255.
    
    pr3 = np.random.randint(1,14,1)[0]
    if pr3 > 8:
        im = tf.image.resize_with_pad(im,target_height=H,target_width=W)
        ma = tf.image.resize_with_pad(ma,target_height=H,target_width=W)
        concs = tf.concat((im,ma),axis=-1)
        crop_patch = np.random.randint(x_limit,y_limit,1)[0]

        concs = tf.image.random_crop(concs, (crop_patch,crop_patch,4))
        im,ma = concs[:,:,:-1],concs[:,:,-1]
        ma = tf.expand_dims(ma,axis=-1)
    
    return im,ma

def read_images_mask_val(image,mask):
    im = tf.io.read_file(image)
    im = tf.io.decode_png(im,channels=3)
    ma = tf.io.read_file(mask)
    ma = tf.io.decode_png(ma,channels=1)
    im = tf.cast(im,tf.float32)/255.
    ma = tf.cast(ma,tf.float32)/255.
    return im,ma



def random_flip(image,mask):
    merge = tf.concat((image,mask),axis=-1)
    rot   = tf.image.flip_left_right(merge)
    im,ma = rot[:,:,:3],rot[:,:,3:]
    return im,ma



def resize_pad(image,mask,H,W,qs1=H,qs2=W):
    m1 = qs1//16
    n1 = qs1//8
    o1 = qs1//4
    p1 = qs1
    
    m2 = qs2//16
    n2 = qs2//8
    o2 = qs2//4
    p2 = qs2
    
    im  = tf.image.resize_with_pad(image,target_height=H,target_width=W)
    ma1 = tf.image.resize_with_pad(mask,target_height=m1,target_width=m2)
    ma2 = tf.image.resize_with_pad(mask,target_height=n1,target_width=n2)
    ma3 = tf.image.resize_with_pad(mask,target_height=o1,target_width=o2)
    ma4 = tf.image.resize_with_pad(mask,target_height=p1,target_width=p2)
    return im,(ma1,ma2,ma3,ma4)




def traingen(ima,mas):
    i,m = read_images_mask_train(ima,mas)
    k1 = np.random.randint(1,15,1)[0]
    if k1>12:
        i,m = random_flip(i,m)
    i,(m1,m2,m3,m4) = resize_pad(i,m,H,W)
    return i,(m1,m2,m3,m4)

def valgen(ima,mas):
    i,m = read_images_mask_val(ima,mas)
    i,(m1,m2,m3,m4) = resize_pad(i,m,H,W)
    return i,(m1,m2,m3,m4)

train_list     = len(TRAIN_IMAGES)
val_list       = len(VAL_IMAGES)
BATCH_SIZE     = config_map["batch"]


ts = train_list//BATCH_SIZE
vs = val_list//BATCH_SIZE


TRAIN      = tf.data.Dataset.from_tensor_slices((TRAIN_IMAGES,TRAIN_MASKS))
VAL        = tf.data.Dataset.from_tensor_slices((VAL_IMAGES,VAL_MASKS))

TRAIN  = TRAIN.map(traingen,num_parallel_calls=tf.data.experimental.AUTOTUNE)
VAL    = VAL.map(valgen,num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ = TRAIN.batch(BATCH_SIZE,drop_remainder=True).shuffle(100).repeat()
val_   = VAL.batch(BATCH_SIZE,drop_remainder=True).shuffle(100).repeat()

