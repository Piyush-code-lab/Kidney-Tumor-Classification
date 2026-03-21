import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from Kidney_Tumor_Classifier.entity.config_entity import EvaluationConfig
from Kidney_Tumor_Classifier.utils.common import save_json
import os


class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        data_dir = str(self.config.training_data)

        datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.20
        )

        self.valid_generator = datagenerator.flow_from_directory(
            directory=data_dir,
            subset="validation",
            shuffle=False,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical"
        )

    def load_model(self):
        self.model = tf.keras.models.load_model(self.config.model_path)

    def evaluation(self):
        self._valid_generator()

        predictions = self.model.predict(self.valid_generator)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = self.valid_generator.classes
        class_labels = list(self.valid_generator.class_indices.keys())

        accuracy = accuracy_score(true_classes, pred_classes)
        cm = confusion_matrix(true_classes, pred_classes)
        report = classification_report(
            true_classes,
            pred_classes,
            target_names=class_labels,
            output_dict=True
        )

        results = {
            "accuracy": round(accuracy * 100, 2),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "class_labels": class_labels
        }

        save_json(
            path=self.config.eval_results_path,
            data=results
        )

        print(f"\nAccuracy: {accuracy * 100:.2f}%")
        print(f"\nClassification Report:")
        print(classification_report(true_classes, pred_classes, target_names=class_labels))
        print(f"\nConfusion Matrix:")
        print(cm)

        return results