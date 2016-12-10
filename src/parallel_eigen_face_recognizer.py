# Parallel implementation of the Eigen Face Recognizer

import cv2, os
import numpy as np
from PIL import Image
import timeit
from multiprocessing import Pool

start_time = timeit.default_timer()

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will use the Eigen Face Recognizer
recognizer = cv2.createEigenFaceRecognizer()

def face_recognition(image_path):

	predict_image_pil = Image.open(image_path).convert('L')
	predict_image = np.array(predict_image_pil, 'uint8')
	faces = faceCascade.detectMultiScale(predict_image)
	# print("Before 2nd")
	for (x, y, w, h) in faces:
		raw_image = predict_image[y: y + h, x: x + w]
		size = 92, 112
		resized_image = cv2.resize(raw_image, size)
		nbr_predicted, conf = recognizer.predict(resized_image)
		nbr_actual = int((os.path.split(image_path))[1].split('.')[0])
		# print(nbr_predicted, nbr_actual)
		# print("Inside")
		if nbr_actual == nbr_predicted:
			# print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
			print("{} is Correctly Recognized".format(nbr_actual))
		else:
			print "{} is Incorrectly Recognized as {}".format(nbr_actual, nbr_predicted)
		cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
		# cv2.waitKey(1000)

def populate_images_and_labels(image_path):

	images = []
	labels = []
	# Read the image and convert to grayscale
	image_pil = Image.open(image_path).convert('L')
	# Convert the image format into numpy array
	image = np.array(image_pil, 'uint8')
	# Get the label of the image
	nbr = int((os.path.split(os.path.split(image_path)[0])[1]).split('s')[1])
	# Detect the face in the image
	faces = faceCascade.detectMultiScale(image)
	# If face is detected, append the face to images and the label to labels
	for (x, y, w, h) in faces:
		raw_image = image[y: y + h, x: x + w]
		size = 92, 112
		resized_image = cv2.resize(raw_image, size)
		images.append(resized_image)
		labels.append(nbr)
		cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
		# cv2.waitKey(50)
	obj = zip(images, labels)
	return obj



def get_images_and_labels(path, image_paths):
	# images will contains face images
	images = []
	# labels will contains the label that is assigned to the image
	labels = []

	pool = Pool()
	image_label_pairs = pool.map(populate_images_and_labels, image_paths)
	
	# print(image_label_pairs[4][0][1])

	len_tuple = len(image_label_pairs)
	
	images = [image_label_pairs[i][0][0]  for i in range(0, len_tuple) if image_label_pairs[i] ]
	labels = [image_label_pairs[i][0][1] for i in range(0, len_tuple) if image_label_pairs[i] ]

	return images, labels

# Path to the Yale Dataset
path = 'att_faces'
# Call the get_images_and_labels function and get the face images and the 
# corresponding labels
adding_start_time = timeit.default_timer()

image_paths = []

for dirname in os.listdir(path):
	for f in os.listdir(path + '/' + dirname):
		image_paths.append('{2}/{0}/{1}'.format(dirname,f,path))
print "\nNo. of training images: ",len(image_paths)

images, labels = get_images_and_labels(path, image_paths)

print("No. of faces detected: " + str(len(images)))
print("No. of labels detected: " + str(len(labels)))

elapsed = timeit.default_timer() - adding_start_time
print("\n***\nTime taken for adding faces to training set = {0}s.\n***".format(round(elapsed,5)))   

cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

# Append the images with the extension .sad into image_paths
image_paths = [str('test/' + f) for f in os.listdir('test')]

print ('\n\n-----------------------------------------------------------------------\nRecognizing Faces:\n')
# print(image_paths)

rec_start_time = timeit.default_timer()

pool = Pool()
pool.map(face_recognition, image_paths)

end_time = timeit.default_timer()

rec_time_elapsed = end_time - rec_start_time

total_time_elapsed = end_time - start_time

print("***\nTime taken for face recognition = {0}s.\n***".format(round(rec_time_elapsed,5)))
print("***\nTotal execution time = {0}s.\n***".format(round(total_time_elapsed,5)))