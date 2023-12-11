import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from tensorflow.keras.applications import MobileNetV2
import kerastuner as kt
from tensorflow.keras.optimizers import Adam

from .constants import IMG_SIZE, BATCH_SIZE, AUTO, SEED, LIST_SEED, PATH_DATA, classes



# Load data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = shuffle(data).reset_index(drop=True)
    data['full_link'] = PATH_DATA + data['image:FILE']
    return data

# Data preprocessing functions
def img_preprocessing(image, label):
    img = tf.io.read_file(image)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size=IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augmentation(image, label):
    img = tf.image.random_flip_left_right(image)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return img, label

def create_dataset_loader(data, batch_size, preprocessing_functions=[]):
    loader = tf.data.Dataset.from_tensor_slices((data['full_link'], data['category']))

    for func in preprocessing_functions:
        loader = loader.map(func, num_parallel_calls=AUTO)

    loader = loader.batch(batch_size).prefetch(AUTO)

    return loader


def create_custom_model(input_shape, num_classes):
    # Load pre-trained MobileNetV2 model with ImageNet weights
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Create a new model by adding custom layers on top of the pre-trained base
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(3456, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Explicitly create an instance of the optimizer outside the compile method
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(model, train_dataset, valid_dataset, name='best_model.h5', epochs=10):
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(name, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

    # Training
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=valid_dataset,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    return history


def evaluate_best_model(best_model_path, test_data, test_dataset):
    # Load the best model
    best_model = tf.keras.models.load_model(best_model_path)

    # Display model summary
    best_model.summary()

    # Evaluate on test set
    test_eval_final = best_model.evaluate(test_dataset)
    print('--' * 50)
    print('Test Loss: {0:.3f}'.format(test_eval_final[0]))
    print('Test Accuracy: {0:.2f} %'.format(test_eval_final[1] * 100))

    # Test set prediction using the best model
    pred_best_model = best_model.predict(test_dataset)
    pred_best_model = np.argmax(pred_best_model, axis=1)

    # Classification report
    clf_best_model = classification_report(test_data['category'], pred_best_model, target_names=list(classes.values()))
    print(clf_best_model)
    
    
