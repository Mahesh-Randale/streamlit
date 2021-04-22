import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import streamlit as st
from classifier2 import sign

class VideoTransformer(VideoTransformerBase):
	def __init__(self):
		
		
		self.threshold2 = 200
	def transform(self, frame):
		img = frame.to_ndarray(format="bgr24")
		img = cv2.flip(img , 1)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_crop = gray[50:250 ,400:600]
		blur = cv2.GaussianBlur(gray_crop,(5,5),2)
		th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
		ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		#img = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
		label = sign(res, "model-bw.h5")
		cv2.rectangle(img , (400 , 50) , (600 , 250) , ( 0 , 0,255),5)
		cv2.putText(img, label, (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
		#print(label)
		return img

st.title("Sign Language Recognition using CNN")
#st.write(label)
ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
