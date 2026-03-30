import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# class labels (must match training folder names)
classes = ['bruise', 'burn', 'cut', 'infection']

# load trained model
model = tf.keras.models.load_model("vgg_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    print("Prediction:", classes[class_index])
    print("Confidence:", confidence)

# test image path
predict_image("test.jpg")
