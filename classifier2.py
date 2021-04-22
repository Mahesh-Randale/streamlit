import cv2
from keras.models import model_from_json
from string import ascii_uppercase
import numpy as np
import operator

def sign(img , weightsfile):
	json_file = open("model-bw.json", "r")
	model_json =json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	model.load_weights("model-bw.h5")

	image_x = image_y = 128
	img2 = img.copy()
	img2 = cv2.resize(img2, (image_x, image_y))
	img2 = np.array(img2, dtype=np.float32)
	img2 = np.reshape(img2, (1, image_x, image_y, 1))

	result = model.predict(img2)
	prediction = {}
	prediction['blank'] = result[0][0]
	inde = 1
	for i in ascii_uppercase:
		prediction[i] = result[0][inde]
		inde +=1
	prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
	return prediction[0][0]




