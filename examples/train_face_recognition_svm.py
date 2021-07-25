# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

"""
Structure:
        <test_image>.jpg
        <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
"""

import face_recognition
from sklearn import svm
import os
import pickle

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
train_dir = os.listdir('./train_dir/')
train_dir_path = './train_dir/'

import os

count = 0
for subdir, dirs, files in os.walk(train_dir_path):
    for file in files:
        image_path = os.path.join(subdir, file)
        if (image_path.find(".jpg") == -1):
            continue
        path_cmp =  image_path.split("/")
        person = path_cmp[len(path_cmp) - 2]
        #print(image_path)
        #print(person)
    # Loop through each training image for the current person
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file(image_path)
        face_bounding_boxes = face_recognition.face_locations(face)
        print("processing " + str(count) + " " + image_path)

        #If training image contains exactly one face
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
            print("name : " + person)
        else:
            print(person + "/" + person + " was skipped and can't be used for training")
        count = count + 1

# Create and train the SVC classifier
print("training model")
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)
print("training done")
# save the model to disk
print("saving model to disk")
modelfilename = 'finalized_model.sav'
pickle.dump(clf, open(modelfilename, 'wb'))
print("saved model to disk in file " + modelfilename)
# save the model

