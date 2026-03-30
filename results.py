import tensorflow as tf
import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

IMG_SIZE = 224
BATCH_SIZE = 16

# Load dataset
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    "dataset/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

models = {
    "EfficientNet": "efficientnet_model.h5",
    "MobileNet": "mobilenet_model.h5",
    "VGG": "vgg_model.h5"
}

for name, path in models.items():

    print("\n==============================")
    print("Model:", name)

    try:
        model = tf.keras.models.load_model(
    path,
    compile=False,
    custom_objects={"MobileNetV2": MobileNetV2}
)

        # parameters
        params = model.count_params()
        print("Parameters:", params)

        # testing accuracy
        loss, acc = model.evaluate(test_data)
        print("Test Accuracy:", acc)

        # execution time
        sample = next(test_data)[0][0]
        sample = np.expand_dims(sample, axis=0)

        start = time.time()
        model.predict(sample)
        end = time.time()

        print("Execution Time:", end-start)

    except Exception as e:
        print("Error loading", name, "model:", e)