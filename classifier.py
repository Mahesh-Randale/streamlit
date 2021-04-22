import cv2
from PIL import Image, ImageDraw
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
from string import ascii_uppercase
import operator
import time
from PIL import Image, ImageOps

def sign(img , weightsfile):
	json_file = open("model-bw.json", "r")
	model_json =json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	model.load_weights("model-bw.h5")

	"""
	image_x = image_y = 128
	img2 = img.copy()
	img2 = cv2.resize(img2, (image_x, image_y))
	img2 = np.array(img2, dtype=np.float32)
	img2 = np.reshape(img2, (1, image_x, image_y, 1))
"""
	data = np.ndarray(shape=(1, 128, 128, 1), dtype=np.float32)
	image = img
	size = (128, 128)
	image = ImageOps.fit(image, size, Image.ANTIALIAS)

	image_array = np.asarray(image)
	
	img1 = np.reshape(image_array, (1, 128, 128, 1))
	data[0] = img1
	
	result = model.predict(data)
	prediction = {}
	prediction['blank'] = result[0][0]
	inde = 1
	for i in ascii_uppercase:
		prediction[i] = result[0][inde]
		inde +=1
	prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
	return prediction[0][0]


