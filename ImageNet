from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
import numpy as np

# Load pre-trained model
model = VGG16(weights='imagenet')

# Load an image to predict
image = load_img('path_to_your_image.jpg', target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

# Predict the image
predictions = model.predict(image)
print(decode_predictions(predictions, top=3)[0])
