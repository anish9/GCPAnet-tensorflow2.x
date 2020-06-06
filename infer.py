from layers import *
import numpy as np
import cv2
import sys

img_rgb  = sys.argv[1]
model = GCPA()
model.load_weights("dut_omron.h5")


def predict(rgb):
	image      = cv2.imread(rgb)
	image      = tf.image.resize_with_pad(image,416,416)
	image      = np.expand_dims(image,axis=0)/ 255.
	pred_chart = model.predict(image)
	prediction = pred_chart[3][0,:,:,0]
	cv2.imshow(prediction)
	cv2.waitKey(0)


predict(img_rgb)