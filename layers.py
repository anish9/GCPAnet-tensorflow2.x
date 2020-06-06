import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from conf import *


def self_refinement(inps,d):
    x1 = Conv2D(256//d,3,padding="same")(inps)
    x1 = BatchNormalization(axis=-1)(x1)
    x1 = Activation("relu")(x1)
    x2 = Conv2D(512//d,3,padding="same")(x1)
    w,b = tf.split(x2,2,axis=-1)
    out = Multiply()([x1,w])
    out = Add()([out,b])
    out = Activation("relu")(out)
    return out

def head_attention(inps):
	x1 = Conv2D(256,3,padding="same")(inps)
	x1 = BatchNormalization(axis=-1)(x1)
	x1 = Activation("relu")(x1)
	x2 = Conv2D(512,3,padding="same")(x1)
	w,b = tf.split(x2,2,axis=-1)
	out = Multiply()([x1,w])
	out = Add()([out,b])
	F   = Activation("relu")(out)
	ave = tf.math.reduce_mean(F,axis=-1)
	ave = tf.expand_dims(ave,axis=-1)
	
	ave = Conv2D(256,1,padding="same",activation="relu")(ave)
	Fn  = Conv2D(256,1,padding="same",activation="sigmoid")(ave)
	out = Multiply()([F,Fn])
	return out


def GCF(inps):
	x1 = Conv2D(256,3,padding="same")(inps)
	x1 = BatchNormalization(axis=-1)(x1)
	x1 = Activation("relu")(x1)
	
	gap = tf.math.reduce_mean(x1,axis=-1)
	gap = tf.expand_dims(gap,axis=-1)
	gap = Conv2D(256,1,padding="same",activation="relu")(gap)
	gap = Conv2D(256,1,padding="same",activation="sigmoid")(gap)

	out = Multiply()([gap,x1])
	return out


def FIA(hin,gin,tin,ft):
    fl = Conv2D(ft,1,padding="same")(tin)
    fl = BatchNormalization(axis=-1)(fl)
    fl = Activation("relu")(fl)
    
    fh = Conv2D(ft,3,padding="same")(hin)
    fh = Activation("relu")(fh)
    fh = UpSampling2D(interpolation="bilinear")(fh)
    fhl  = Multiply()([fl,fh]) 
    
    fdh = UpSampling2D()(hin)  
    flh = Conv2D(ft,3,padding="same")(fl)
    flh = Activation("relu")(flh)
    flh = Multiply()([fdh,flh]) 
    
    fg = Conv2D(ft,3,padding="same")(gin)
    fg = Activation("relu")(fg)
    fg = UpSampling2D(interpolation="bilinear")(fg)
    fgl = Multiply()([fg,fl])
    
    conc = Concatenate()([fhl,flh,fgl])
    conc = Conv2D(ft,3,padding="same")(conc)
    conc = BatchNormalization(axis=-1)(conc)
    conc = Activation("relu")(conc)

    return conc


def GCPA():
    ENCODER_BASE = tf.keras.applications.ResNet50(include_top=False,input_shape=(config_map["dim"],config_map["dim"],3))
    for layer in ENCODER_BASE.layers:
        layer.traiable = True

    f0 = ENCODER_BASE.get_layer("conv1_relu").output      

    f1 = ENCODER_BASE.get_layer("conv2_block2_out").output   
    f2 = ENCODER_BASE.get_layer("conv3_block4_out").output   
    f3 = ENCODER_BASE.get_layer("conv4_block6_out").output  
    f4 = ENCODER_BASE.get_layer("conv5_block3_out").output  
    
    aux1 = head_attention(f4)
    aux1 = self_refinement(aux1,1) 
    gux1 = GCF(f4)
    aux1 = FIA(aux1,gux1,f3,256)
    aux1_out = Conv2D(1,3,padding="same")(aux1)
    aux1_out = Activation("sigmoid",name="aux1")(aux1_out)
    
    aux2 = self_refinement(aux1,1)
    gux2 = GCF(f4)
    gux2 = UpSampling2D(interpolation="bilinear")(gux2)
    aux2 = FIA(aux2,gux2,f2,256)
    aux2_out = Conv2D(1,3,padding="same")(aux2)
    aux2_out = Activation("sigmoid",name="aux2")(aux2_out)
    
    aux3 = self_refinement(aux2,1)
    gux3 = GCF(f4)
    gux3 = UpSampling2D(interpolation="bilinear")(gux3)
    gux3 = UpSampling2D(interpolation="bilinear")(gux3)
    aux3 = FIA(aux3,gux3,f1,256)
    aux3_out = Conv2D(1,3,padding="same")(aux3)
    aux3_out = Activation("sigmoid",name="aux3")(aux3_out)

    dom0 = self_refinement(aux3,1)
    dom0 = UpSampling2D(interpolation="bilinear")(dom0)
    dom0 = Conv2D(64,3,padding="same")(dom0)
    dom0 = BatchNormalization(axis=-1)(dom0)
    dom0 = Activation("relu")(dom0)
    dom0 = UpSampling2D(interpolation="bilinear")(dom0)

    out  = Conv2D(1,3,padding="same")(dom0)
    out  = Activation("sigmoid",name="dom")(out)
    
    MODEL  = Model(ENCODER_BASE.input,[aux1_out,aux2_out,aux3_out,out])
    return MODEL
