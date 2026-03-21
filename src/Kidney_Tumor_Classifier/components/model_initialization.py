import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from Kidney_Tumor_Classifier.entity.config_entity import BaseModelConfig


class modelInitialization:
    def __init__(self, config: BaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        for layer in model.layers:
            layer.trainable = False

        for layer in model.layers[-4:]:
            layer.trainable = True

        x = tf.keras.layers.Flatten()(model.output)

        x = tf.keras.layers.Dense(units=256, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(units=128, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(x)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)