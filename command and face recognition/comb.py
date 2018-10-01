#!/usr/bin/python
from os import environ, path
import pyaudio
from pocketsphinx import pocketsphinx as ps
from sphinxbase import sphinxbase 
#import pocketsphinx.pocketsphinx as ps
#import sphinxbase.sphinxbase
import sys
import dlib
from skimage import io
from scipy.spatial import distance
import pickle
import cv2
import csv

FILENAME = 'users.csv' 
MODELDIR = "../../../model"
config = ps.Decoder.default_config()
config.set_string('-hmm', '/home/anna/diplom/test_speech/zero_ru_cont_8k_v3/zero_ru.cd_cont_4000/')
config.set_string('-dict', '/home/anna/diplom/comb_1/speech/vocabular.dict')
config.set_string('-jsgf', '/home/anna/diplom/comb_1/speech/sp.jsgf')
config.set_string('-logfn', '/dev/null')
#config.set_string('-lm', '/home/anna/diplom/test_speech/zero_ru_cont_8k_v3/ru.lm')
config.set_int('-nfft', 512)
config.set_float('-samprate', 8000.0) 
decoder = ps.Decoder(config)


#create models for looking face
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

users = {}
with open(FILENAME, "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            name = row[0]
            with open(name+'.pickle', 'rb') as f:
                face_descriptor = pickle.load(f)
                users[face_descriptor] = name

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=8000, input=True, frames_per_buffer=1024)
stream.start_stream() 

in_speech_bf = False
decoder.start_utt()
shape = None

cap = cv2.VideoCapture(-1)
win1 = dlib.image_window()

while True:
    voice_command = ""
    ret, frame = cap.read()
    buf = stream.read(1024)
    if buf:
        decoder.process_raw(buf, False, False)
        if decoder.get_in_speech() != in_speech_bf:
            in_speech_bf = decoder.get_in_speech()
            if not in_speech_bf:
                decoder.end_utt()
                try:
                    voice_command = decoder.hyp().hypstr
                except:
                    voice_command = ""
                decoder.start_utt()
    else:
        break
    if voice_command != "":
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        win1.clear_overlay()
        win1.set_image(img)
        #find face on photo
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = sp(img, d)
            win1.add_overlay(d)
            win1.add_overlay(shape)
            print(shape)
            
        if shape == None:
            continue
        face_descriptor2 = facerec.compute_face_descriptor(img, shape)
        min = 5
        user_name = ""
        for key in users:
            a = distance.euclidean(key, face_descriptor2)
            if a < min and a<0.6:
                min = a
                user_name = users[key]
        if user_name == "":
            print("Доступ отклонен")
        else:
            print(user_name)
        print(voice_command)
decoder.end_utt()
