import dlib
import pickle
import cv2
import csv
import os.path


FILENAME = "users.csv"

def descriptor_count(name):
    cap = cv2.VideoCapture(-1)
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    win2 = dlib.image_window()
    #win2.clear_overlay()
    win2.set_image(img)
    dets_webcam = detector(img, 1)

    for k, d in enumerate(dets_webcam):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        print(d)
        shape = sp(img, d)
        win2.clear_overlay()
        win2.add_overlay(d)
        win2.add_overlay(shape)
    face_descriptor = facerec.compute_face_descriptor(img, shape)

    with open(name+'.pickle', 'wb') as f:
        pickle.dump(face_descriptor, f)
        
def write_in_file(name):
    with open(FILENAME, "a", newline="") as file:
        writer = csv.writer(file)
        user = [name]
        writer.writerow(user)
    


#create models for looking face
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

print('Write person name')
name = input()
k = True

if os.path.exists(FILENAME):
    with open(FILENAME, "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == name:
                print('This name already exist. Rewrite?')
                answer = input()
                if (answer == 'yes') or (answer == 'y'):
                    descriptor_count(name)
                k = False
                break
        if k:
            write_in_file(name)
            descriptor_count(name)
else:
    write_in_file(name)
    descriptor_count(name)
                
    
             

