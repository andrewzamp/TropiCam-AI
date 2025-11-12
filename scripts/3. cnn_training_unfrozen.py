# Libraries
import os
import numpy as np
import multiprocessing
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.applications.convnext import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check if TensorFlow is using GPU
gpus = tf.config.list_physical_devices('GPU')
num_gpus = len(gpus)
if num_gpus > 0:
    print(f"Num GPUs Available: {num_gpus}")
else:
    print("No GPUs available. Please check your TensorFlow installation.")

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Set working directory
base_dir = '~/data/full_dataset'
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
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Initialize generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Calculate steps per epoch
steps_train = train_generator.samples // batch_size + (1 if train_generator.samples % batch_size != 0 else 0)
steps_validation = validation_generator.samples // batch_size + (1 if validation_generator.samples % batch_size != 0 else 0)
steps_test = test_generator.samples // batch_size + (1 if test_generator.samples % batch_size != 0 else 0)

# Calculate class weights
class_counts = np.sum(train_generator.classes == np.arange(train_generator.num_classes)[:, None], axis=1)
class_weights = 1.0 / np.log1p(class_counts)
class_weights = class_weights / np.sum(class_weights) * len(class_counts)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# Define the percentage of layers to unfreeze
unfreeze_percentage = 0.8

# Build the model
with strategy.scope():
    best_model_path = '~/results/models/TropiCam_AI_tuned'
    model = load_model(best_model_path)
    model.summary()
    
    conv_base = model.layers[0]
    conv_base.trainable = True
    
    total_layers = len(conv_base.layers)
    fine_tune_from = int(total_layers * unfreeze_percentage)
    print(f"Total layers in conv_base: {total_layers}")
    print(f"Unfreezing layers from index {fine_tune_from} onward.")
    for i, layer in enumerate(conv_base.layers):
        layer.trainable = (i >= fine_tune_from)
    
    print("Trainable layers in conv_base:")
    for i, layer in enumerate(conv_base.layers):
        if layer.trainable:
            print(f"  {i}: {layer.name}")
    
    target_lr = 6.6754e-05
    model.compile(optimizer=Adam(learning_rate=target_lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

# Callbacks for early stopping and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(
    '~/results/models/best_model_ft',
    monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
callbacks = [early_stopping, model_checkpoint, reduce_lr]

num_cores = multiprocessing.cpu_count()
workers = num_cores // 2

# Fine-tune the model
fine_tune_epochs = 100
history_ft = model.fit(
    train_generator,
    steps_per_epoch=steps_train,
    validation_data=validation_generator,
    validation_steps=steps_validation,
    epochs=fine_tune_epochs,
    callbacks=callbacks,
    workers=workers,
    use_multiprocessing=True,
    class_weight=class_weight_dict,
    initial_epoch=59
)

print("Training accuracy: {:.4f}".format(history_ft.history['accuracy'][-1]))
print("Validation accuracy: {:.4f}".format(history_ft.history['val_accuracy'][-1]))

test_loss, test_acc = model.evaluate(test_generator, steps=steps_test)
print("Test accuracy: {:.4f}".format(test_acc))

# Save the fine-tuned model
final_model_path = '~/results/models/TropiCam_AI_unfrozen'
model.save(final_model_path)
np.save(final_model_path + '.npy', history_ft.history)
os.remove('~/results/models/best_model_ft')
print("Fine-tuned model saved.")