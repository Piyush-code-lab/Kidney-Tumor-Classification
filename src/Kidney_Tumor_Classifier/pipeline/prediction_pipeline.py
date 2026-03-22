import numpy as np
import tensorflow as tf
from pathlib import Path


class PredictionPipeline:
    def __init__(self, model_path: str = "saved_models/model.h5"):
        self.model_path = model_path
        self.model = None
        self.class_names = ["Cyst", "Normal", "Stone", "Tumor"]

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

    def preprocess_image(self, image_path: str):
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path: str):
        if self.model is None:
            self.load_model()

        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_index]
        confidence = round(float(predictions[0][predicted_class_index]) * 100, 2)

        all_probabilities = {
            self.class_names[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(self.class_names))
        }

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }