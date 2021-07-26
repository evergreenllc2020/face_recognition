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

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names


# Test directory
test_dir_path = './test_dir/'
output_dir = "./out_dir/"

import pickle
from PIL import Image, ImageDraw
import numpy as np



def draw_name_on_face(image_path, person):
    known_image = face_recognition.load_image_file(image_path)
    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(known_image)
    face_encodings = face_recognition.face_encodings(known_image, face_locations)
    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(known_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)
    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        #matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = person[0].replace("pins_","")
        print(name)

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        #face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        #best_match_index = np.argmin(face_distances)
        #if matches[best_match_index]:
         #   name = known_face_names[best_match_index]

        left = left -10
        right = right + 10
        bottom = bottom + 25
        top = top - 20
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()

    # You can also save a copy of the new image to disk if you want by uncommenting this line
    #get image name
    image_file_array = image_path.split("/")
    image_file_name = image_file_array[len(image_file_array) - 1]

    pil_image.save(output_dir + image_file_name)



modelfilename = 'finalized_model.sav'
clf = pickle.load(open(modelfilename, 'rb'))
# Load the test image with unknown faces into a numpy array

for subdir, dirs, files in os.walk(test_dir_path):
    for file in files:
        image_path = os.path.join(subdir, file)
        if (image_path.find(".jpg") == -1):
            continue
        test_image = face_recognition.load_image_file(image_path)
        # Find all the faces in the test image using the default HOG-based model
        face_locations = face_recognition.face_locations(test_image)
        no = len(face_locations)
        print("Number of faces detected: ", no)
        # Predict all the faces in the test image using the trained classifier
        print("Found faces in image " + image_path)
        for i in range(no):
            test_image_enc = face_recognition.face_encodings(test_image)[i]
            name = clf.predict([test_image_enc])
            draw_name_on_face(image_path, name)
            #print(*name)
