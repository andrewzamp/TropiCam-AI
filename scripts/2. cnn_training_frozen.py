# Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtBase 
from tensorflow.keras.applications.convnext import preprocess_input 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout 
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback 
import multiprocessing

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
test_dir = os.path.join(base_dir, 'test')

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

# No data augmentation for the validation and testing
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Initialize generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Calculate steps per epoch
steps_train = train_generator.samples // batch_size + (1 if train_generator.samples % batch_size != 0 else 0)
steps_validation = validation_generator.samples // batch_size + (1 if validation_generator.samples % batch_size != 0 else 0)
steps_test = test_generator.samples // batch_size + (1 if test_generator.samples % batch_size != 0 else 0)


# Build the model
with strategy.scope():
    # Load pretrained ConvNeXt-Base architecture
    conv_base = ConvNeXtBase(weights='imagenet',
                            include_top=False,
                            input_shape=(img_height, img_width, 3))

    # Now we stack on it our label-specific densely connected layers
    model = Sequential([
        conv_base,
        GlobalAveragePooling2D(),
        Dense(2048, activation='relu'),
        Dropout(0.45),
        Dense(84, activation='softmax') 
    ])

    # We need to freeze the weight of ConvNeXtBase to avoid modifications
    conv_base.trainable = False
    model.summary() 

    # Compile the model
    target_lr=6.6754e-05
    model.compile(optimizer=Adam(learning_rate=target_lr),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Callbacks for early stopping and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint('~/models/best_model', monitor='val_loss', save_best_only=True)

# Callback for reducing learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)

# Calculate the class weights
class_counts = np.sum(train_generator.classes == np.arange(train_generator.num_classes)[:, None], axis=1)
class_weights = 1.0 / np.sqrt(class_counts)
class_weights = class_weights / np.sum(class_weights) * len(class_counts)
class_weight_dict = dict(enumerate(class_weights))

# Calculate the steps per epoch based on training generator
steps_per_epoch = steps_train

# Define callbacks
callbacks = [early_stopping, model_checkpoint, reduce_lr]

# Determine the number of available CPU cores and use half of them
num_cores = multiprocessing.cpu_count()
workers = num_cores // 2

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_train,
    validation_data=validation_generator,
    validation_steps=steps_validation,
    epochs=100,
    callbacks=callbacks,
    workers=workers,
    use_multiprocessing=True,
    class_weight=class_weight_dict)

# Check if early stopping was triggered
if early_stopping.stopped_epoch > 0:
    # Early stopping was triggered, load and rename the best model
    best_model = load_model('~/results/models/best_model')
    best_model.save('~/models/TropiCam_AI_tuned')
else:
    # Save the final model
    model.save('~/models/TropiCam_AI_tuned')

# Save training history
np.save('~/results/models/TropiCam_AI_tuned.npy', history.history)

# Preview the accuracy of the model
print('Training accuracy:', round(history.history['accuracy'][len(history.history['accuracy'])-1]*100, 2), end='%\n')
print('Validation accuracy:', round(history.history['val_accuracy'][len(history.history['val_accuracy'])-1]*100, 2), end='%\n')

# Extract the accuracy and loss data from history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Plotting training and validation accuracy and loss
plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()
plt.close() 

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Cross entropy')
plt.show()
plt.close()