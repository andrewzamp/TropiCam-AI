# libraries
import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.applications import ConvNeXtBase  
from tensorflow.keras.applications.convnext import preprocess_input  
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout  
from tensorflow.keras.optimizers import Adam, RMSprop, SGD  
from tensorflow.keras.callbacks import EarlyStopping  
from tensorflow.keras.metrics import Recall  

# Check if TensorFlow is using GPU
gpus = tf.config.list_physical_devices('GPU')
num_gpus = len(gpus)
if num_gpus > 0:
    print(f"Num GPUs Available: {num_gpus}")
else:
    print("No GPUs available. Please check your TensorFlow installation.")

# Multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Set working directory
base_dir = '~/data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Define image size and batch size
img_height, img_width = 224, 224
batch_size = 32 * num_gpus

# Generator for training images with various data augmentation techniques
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    channel_shift_range=0.1,
    fill_mode='nearest'
)

# No data augmentation for validation
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Initialize generators
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical', shuffle=True)
validation_generator = val_datagen.flow_from_directory(
    validation_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical', shuffle=True)

# Calculate steps per epoch
steps_train = train_generator.samples // batch_size + (1 if train_generator.samples % batch_size != 0 else 0)
steps_validation = validation_generator.samples // batch_size + (1 if validation_generator.samples % batch_size != 0 else 0)

# Define Bayesian finetuning module
def build_model(hp):
    with strategy.scope():
        # Load pretrained ConvNeXt-Base architecture
        conv_base = ConvNeXtBase(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        conv_base.trainable = False 
        
        # Tunable hyperparameters
        model = Sequential([
            conv_base,
            GlobalAveragePooling2D(),
            Dense(hp.Choice('dense_units', [128, 256, 512, 1024, 2048]), activation='relu'),
            Dropout(hp.Float('dropout_rate', 0.3, 0.7, step=0.05)),
            Dense(84, activation='softmax')
        ])
        
        optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        learning_rate = hp.Float('learning_rate', 1e-6, 1e-3, sampling='log')
        optimizer = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD}[optimizer_name](learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', Recall(name='recall')])
        return model

# Initialize the Bayesian tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective('val_recall', direction='max'),
    max_trials=50,  
    executions_per_trial=1,
    directory='~/results/tuning',
    project_name='TropiCam_AI_tuning'
)

# Define function to compute class weights
def get_class_weights(hp):
    weighting_scheme = hp.Choice('class_weighting', ['none', 'inverse', 'sqrt_inverse', 'log_inverse'])
    if weighting_scheme == 'none':
        return None
    
    class_counts = np.sum(train_generator.classes == np.arange(train_generator.num_classes)[:, None], axis=1)
    weights = {'inverse': 1.0 / class_counts,
               'sqrt_inverse': 1.0 / np.sqrt(class_counts),
               'log_inverse': 1.0 / np.log1p(class_counts)}.get(weighting_scheme)
    
    return dict(enumerate(weights / np.sum(weights) * len(class_counts)))

# Run optimization
tuner.search(
    train_generator,
    steps_per_epoch=steps_train,
    validation_data=validation_generator,
    validation_steps=steps_validation,
    epochs=100,
    class_weight=get_class_weights(tuner.oracle.hyperparameters),
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
)

# Retrieve best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")