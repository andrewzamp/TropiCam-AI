##################################################################################################
######################################## Import libraries ########################################
##################################################################################################

# Standard libraries
import os

# Numerical and Data Handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib
matplotlib.use('Agg')  # Set the non-interactive backend
import matplotlib.pyplot as plt 
import seaborn as sns

# Machine Learning and Deep Learning
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# TensorFlow/Keras Modules
from tensorflow.keras.applications import ConvNeXtBase # type: ignore
from tensorflow.keras.applications.convnext import preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import tensorflow.keras.backend as K # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback # type: ignore
from tensorflow.keras.mixed_precision import experimental as mixed_precision # type: ignore

# Check if TensorFlow is using GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
print(f"Num GPUs Available: {num_gpus}")



##################################################################################################
##################################### Preparing the dataset ######################################
##################################################################################################

base_dir = r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\training\.FULL"

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')



##################################################################################################
##################################### Initializing generators ####################################
##################################################################################################

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

# The generator loads images in batches directly from the pertaining folder
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# Same for validation and testing
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

steps_train = train_generator.samples // batch_size + (1 if train_generator.samples % batch_size != 0 else 0)
steps_validation = validation_generator.samples // batch_size + (1 if validation_generator.samples % batch_size != 0 else 0)
steps_test = test_generator.samples // batch_size + (1 if test_generator.samples % batch_size != 0 else 0)



##################################################################################################
######################################## Building the CNN ########################################
##################################################################################################

# First load ConvNeXtBase model for transfer learning
conv_base = ConvNeXtBase(weights='imagenet',
                        include_top=False,
                        input_shape=(img_height, img_width, 3))

# Now we stack on it our label-specific densely connected layers
# Define the CNN model
model = Sequential([
    conv_base,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(63, activation='softmax') 
])

# We need to freeze the weight of ConvNeXtBase to avoid modifications
conv_base.trainable = False
model.summary() # We see now that the number of trainable parameters has decreased

# Compile the model
target_lr=1e-3
model.compile(optimizer=Adam(learning_rate=target_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Callbacks for early stopping and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\models\best_model", monitor='val_loss', save_best_only=True)

# Callback for reducing learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

# Custom learning rate scheduler for warmup

class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, target_lr, warmup_epochs, steps_per_epoch):
        super(WarmUpLearningRateScheduler, self).__init__()
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch

    def on_epoch_begin(self, epoch, logs=None):
        # Only adjust the learning rate during the warm-up phase
        if epoch < self.warmup_epochs:
            warmup_lr = (self.target_lr / 100) + ((self.target_lr - (self.target_lr / 100)) * (epoch / self.warmup_epochs))
            print(f"Epoch {epoch + 1}: Warm-up learning rate for this epoch is {warmup_lr:.8f}")
            K.set_value(self.model.optimizer.lr, warmup_lr)
        elif epoch == self.warmup_epochs:
            # Set the learning rate to the target after warm-up and stop further modifications
            print(f"Epoch {epoch + 1}: Warm-up phase complete. Learning rate is now stable at {self.target_lr:.8f}")
            K.set_value(self.model.optimizer.lr, self.target_lr)

    def on_epoch_end(self, epoch, logs=None):
        # Stop modifying the learning rate after the warm-up phase
        if epoch >= self.warmup_epochs:
            self.model.stop_training = False


##################################################################################################
####################################### Training the model #######################################
##################################################################################################

# Define the number of training epochs
warmup_epochs = 5

# Calculate the steps per epoch based on your training generator
steps_per_epoch = steps_train

# Instantiate the warmup scheduler
warmup_lr_scheduler = WarmUpLearningRateScheduler(target_lr=target_lr, warmup_epochs=warmup_epochs, steps_per_epoch=steps_per_epoch)

# Add the warmup scheduler to your callbacks
callbacks = [early_stopping, model_checkpoint, reduce_lr, warmup_lr_scheduler]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_train,
    validation_data=validation_generator,
    validation_steps=steps_validation,
    epochs=100,
    callbacks=callbacks
)

# Check if early stopping was triggered
if early_stopping.stopped_epoch > 0:
    # Early stopping was triggered, load and rename the best model
    best_model = load_model(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\models\best_model")
    best_model.save(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\models\FULL_ConvNeXtBase")
else:
    # Save the final model as it is the best one
    model.save(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\models\FULL_ConvNeXtBase")

# Remove the best model file
#os.remove(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\models\best_model")

# Save training history
np.save(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\models\FULL_ConvNeXtBase.npy", history.history)

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
plt.savefig(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\images\FULL_ConvNeXtBase_train_val_accuracy.png")
plt.close()  # Close the figure to free memory

# Plot and save the loss figure
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Cross entropy')
plt.savefig(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\images\FULL_ConvNeXtBase_train_val_loss.png")
plt.close()


##################################################################################################
######################################## Testing the model #######################################
##################################################################################################

# Load the model
model = load_model(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\models\FULL_ConvNeXtBase")

# Load taxonomy mapping
taxonomy_df = pd.read_csv(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\training\taxonomy\taxonomy_mapping.csv")


#------------------------------------------------------------------------------------------------------------------------

# Predict on the test dataset: SPECIES level
taxonomic_level = 'species'  # Change to 'species', 'genus', 'family', 'order', or 'class' as needed

print('Predicting at:', taxonomic_level, 'level')
test_generator.reset()
predictions = model.predict(test_generator, steps=steps_test)

# Function to aggregate predictions to a higher taxonomic level
def aggregate_predictions(predicted_probs, taxonomic_level=taxonomic_level):
    unique_labels = sorted(taxonomy_df[taxonomic_level].unique())
    aggregated_predictions = np.zeros((predicted_probs.shape[0], len(unique_labels)))

    for idx, row in taxonomy_df.iterrows():
        species = row['species']
        higher_level = row[taxonomic_level]
        
        species_index = list(test_generator.class_indices.keys()).index(species)
        higher_level_index = unique_labels.index(higher_level)
        
        aggregated_predictions[:, higher_level_index] += predicted_probs[:, species_index]
    
    return aggregated_predictions, unique_labels

# Function to get true labels at a higher taxonomic level
def get_true_labels(true_labels, taxonomic_level=taxonomic_level):
    higher_level_labels = []
    for label in true_labels:
        species = list(test_generator.class_indices.keys())[label]
        higher_level = taxonomy_df[taxonomy_df['species'] == species][taxonomic_level].values[0]
        higher_level_labels.append(higher_level)
    
    unique_labels = sorted(taxonomy_df[taxonomic_level].unique())
    higher_level_indices = [unique_labels.index(label) for label in higher_level_labels]
    return np.array(higher_level_indices), unique_labels

# Aggregate predictions to the specified taxonomic level
aggregated_predictions, aggregated_class_labels = aggregate_predictions(predictions, taxonomic_level=taxonomic_level)

# Get true labels at the specified taxonomic level
true_labels = test_generator.classes
aggregated_true_labels, aggregated_class_labels = get_true_labels(true_labels, taxonomic_level=taxonomic_level)

# Calculate top-3 and top-5 accuracies based on the aggregated predictions
def top_k_accuracy(y_true, y_pred_probs, k=1):
    top_k = np.argsort(y_pred_probs, axis=1)[:, -k:]
    true_in_top_k = np.any(top_k == y_true.reshape(-1, 1), axis=1)
    top_k_accuracy = np.mean(true_in_top_k)
    return top_k_accuracy

top1_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=1) * 100
top3_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=3) * 100
top5_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=5) * 100

# Print accuracies
print(f'Top-1 test accuracy: {top1_accuracy:.2f}%')
print(f'Top-3 test accuracy: {top3_accuracy:.2f}%')
print(f'Top-5 test accuracy: {top5_accuracy:.2f}%\n')

# Generate classification report
predicted_labels = np.argmax(aggregated_predictions, axis=1)
report = classification_report(aggregated_true_labels, predicted_labels, target_names=aggregated_class_labels, zero_division=0)
print(report)

# Compute confusion matrix and normalize over rows
cm = confusion_matrix(aggregated_true_labels, predicted_labels, normalize='true')

# Plot confusion matrix
plt.figure(figsize=(20, 15))  # Larger figure for more room
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False, 
            xticklabels=aggregated_class_labels, yticklabels=aggregated_class_labels, 
            vmin=0, vmax=1, annot_kws={'size': 6})

# Adjust font sizes
title_font = {'fontsize': 18, 'fontweight': 'bold'}
label_font = {'fontsize': 16, 'color': 'black'}
tick_label_font = {'fontsize': 9, 'color': 'grey'}  # Increase font size for better visibility

# Labels and title
plt.xlabel(f'Predicted {taxonomic_level}', fontdict=label_font)
plt.ylabel(f'True {taxonomic_level}', fontdict=label_font)

# Dynamic title based on taxonomic level
if taxonomic_level == 'class':
    plt.title(f'Accuracy = {top1_accuracy:.2f}% (Top-1)', fontdict=title_font)
elif taxonomic_level == 'order':
    plt.title(f'Accuracy = {top1_accuracy:.2f}% (Top-1), {top3_accuracy:.2f}% (Top-3)', fontdict=title_font)
else:
    plt.title(f'Accuracy = {top1_accuracy:.2f}% (Top-1), {top3_accuracy:.2f}% (Top-3), {top5_accuracy:.2f}% (Top-5)', fontdict=title_font)

# Rotate x labels for better fit
plt.xticks(rotation=45, ha='right', **tick_label_font)  # Increase rotation
plt.yticks(rotation=0, **tick_label_font)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\images\FULL_ConvNeXtBase_conf_matrix_species.png")
plt.close()


#------------------------------------------------------------------------------------------------------------------------

# Predict on the test dataset: GENUS level
taxonomic_level = 'genus'  # Change to 'species', 'genus', 'family', 'order', or 'class' as needed

print('Predicting at:', taxonomic_level, 'level')
test_generator.reset()
predictions = model.predict(test_generator, steps=steps_test)

# Aggregate predictions to the specified taxonomic level
aggregated_predictions, aggregated_class_labels = aggregate_predictions(predictions, taxonomic_level=taxonomic_level)

# Get true labels at the specified taxonomic level
true_labels = test_generator.classes
aggregated_true_labels, aggregated_class_labels = get_true_labels(true_labels, taxonomic_level=taxonomic_level)

top1_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=1) * 100
top3_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=3) * 100
top5_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=5) * 100

# Print accuracies
print(f'Top-1 test accuracy: {top1_accuracy:.2f}%')
print(f'Top-3 test accuracy: {top3_accuracy:.2f}%')
print(f'Top-5 test accuracy: {top5_accuracy:.2f}%\n')

# Generate classification report
predicted_labels = np.argmax(aggregated_predictions, axis=1)
report = classification_report(aggregated_true_labels, predicted_labels, target_names=aggregated_class_labels, zero_division=0)
print(report)

# Compute confusion matrix and normalize over rows
cm = confusion_matrix(aggregated_true_labels, predicted_labels, normalize='true')

# Plot confusion matrix
plt.figure(figsize=(20, 15))  # Larger figure for more room
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False, 
            xticklabels=aggregated_class_labels, yticklabels=aggregated_class_labels, 
            vmin=0, vmax=1, annot_kws={'size': 8})

# Adjust font sizes
title_font = {'fontsize': 18, 'fontweight': 'bold'}
label_font = {'fontsize': 16, 'color': 'black'}
tick_label_font = {'fontsize': 10, 'color': 'grey'}  # Increase font size for better visibility

# Labels and title
plt.xlabel(f'Predicted {taxonomic_level}', fontdict=label_font)
plt.ylabel(f'True {taxonomic_level}', fontdict=label_font)

# Dynamic title based on taxonomic level
if taxonomic_level == 'class':
    plt.title(f'Accuracy = {top1_accuracy:.2f}% (Top-1)', fontdict=title_font)
else:
    plt.title(f'Accuracy = {top1_accuracy:.2f}% (Top-1), {top3_accuracy:.2f}% (Top-3), {top5_accuracy:.2f}% (Top-5)', fontdict=title_font)

# Rotate x labels for better fit
plt.xticks(rotation=45, ha='right', **tick_label_font)  # Increase rotation
plt.yticks(rotation=0, **tick_label_font)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\images\FULL_ConvNeXtBase_conf_matrix_genus.png")
plt.close()


#------------------------------------------------------------------------------------------------------------------------

# Predict on the test dataset: FAMILY level
taxonomic_level = 'family'  # Change to 'species', 'genus', 'family', 'order', or 'class' as needed

print('Predicting at:', taxonomic_level, 'level')
test_generator.reset()
predictions = model.predict(test_generator, steps=steps_test)

# Aggregate predictions to the specified taxonomic level
aggregated_predictions, aggregated_class_labels = aggregate_predictions(predictions, taxonomic_level=taxonomic_level)

# Get true labels at the specified taxonomic level
true_labels = test_generator.classes
aggregated_true_labels, aggregated_class_labels = get_true_labels(true_labels, taxonomic_level=taxonomic_level)

top1_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=1) * 100
top3_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=3) * 100
top5_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=5) * 100

# Print accuracies
print(f'Top-1 test accuracy: {top1_accuracy:.2f}%')
print(f'Top-3 test accuracy: {top3_accuracy:.2f}%')
print(f'Top-5 test accuracy: {top5_accuracy:.2f}%\n')

# Generate classification report
predicted_labels = np.argmax(aggregated_predictions, axis=1)
report = classification_report(aggregated_true_labels, predicted_labels, target_names=aggregated_class_labels, zero_division=0)
print(report)

# Compute confusion matrix and normalize over rows
cm = confusion_matrix(aggregated_true_labels, predicted_labels, normalize='true')

# Plot confusion matrix
plt.figure(figsize=(20, 15))  # Larger figure for more room
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False, 
            xticklabels=aggregated_class_labels, yticklabels=aggregated_class_labels, 
            vmin=0, vmax=1, annot_kws={'size': 16})

# Adjust font sizes
title_font = {'fontsize': 18, 'fontweight': 'bold'}
label_font = {'fontsize': 16, 'color': 'black'}
tick_label_font = {'fontsize': 14, 'color': 'grey'}  # Increase font size for better visibility

# Labels and title
plt.xlabel(f'Predicted {taxonomic_level}', fontdict=label_font)
plt.ylabel(f'True {taxonomic_level}', fontdict=label_font)

# Dynamic title based on taxonomic level
if taxonomic_level == 'class':
    plt.title(f'Accuracy = {top1_accuracy:.2f}% (Top-1)', fontdict=title_font)
else:
    plt.title(f'Accuracy = {top1_accuracy:.2f}% (Top-1), {top3_accuracy:.2f}% (Top-3), {top5_accuracy:.2f}% (Top-5)', fontdict=title_font)

# Rotate x labels for better fit
plt.xticks(rotation=45, ha='right', **tick_label_font)  # Increase rotation
plt.yticks(rotation=0, **tick_label_font)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\images\FULL_ConvNeXtBase_conf_matrix_family.png")
plt.close()


#------------------------------------------------------------------------------------------------------------------------

# Predict on the test dataset: ORDER level
taxonomic_level = 'order'  # Change to 'species', 'genus', 'family', 'order', or 'class' as needed

print('Predicting at:', taxonomic_level, 'level')
test_generator.reset()
predictions = model.predict(test_generator, steps=steps_test)

# Aggregate predictions to the specified taxonomic level
aggregated_predictions, aggregated_class_labels = aggregate_predictions(predictions, taxonomic_level=taxonomic_level)

# Get true labels at the specified taxonomic level
true_labels = test_generator.classes
aggregated_true_labels, aggregated_class_labels = get_true_labels(true_labels, taxonomic_level=taxonomic_level)

top1_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=1) * 100
top3_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=3) * 100
top5_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=5) * 100

# Print accuracies
print(f'Top-1 test accuracy: {top1_accuracy:.2f}%')
print(f'Top-3 test accuracy: {top3_accuracy:.2f}%')
print(f'Top-5 test accuracy: {top5_accuracy:.2f}%\n')

# Generate classification report
predicted_labels = np.argmax(aggregated_predictions, axis=1)
report = classification_report(aggregated_true_labels, predicted_labels, target_names=aggregated_class_labels, zero_division=0)
print(report)

# Compute confusion matrix and normalize over rows
cm = confusion_matrix(aggregated_true_labels, predicted_labels, normalize='true')

# Plot confusion matrix
plt.figure(figsize=(20, 15))  # Larger figure for more room
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False, 
            xticklabels=aggregated_class_labels, yticklabels=aggregated_class_labels, 
            vmin=0, vmax=1, annot_kws={'size': 20})

# Adjust font sizes
title_font = {'fontsize': 18, 'fontweight': 'bold'}
label_font = {'fontsize': 16, 'color': 'black'}
tick_label_font = {'fontsize': 14, 'color': 'grey'}  # Increase font size for better visibility

# Labels and title
plt.xlabel(f'Predicted {taxonomic_level}', fontdict=label_font)
plt.ylabel(f'True {taxonomic_level}', fontdict=label_font)

# Dynamic title based on taxonomic level
if taxonomic_level == 'class':
    plt.title(f'Accuracy = {top1_accuracy:.2f}% (Top-1)', fontdict=title_font)
else:
    plt.title(f'Accuracy = {top1_accuracy:.2f}% (Top-1), {top3_accuracy:.2f}% (Top-3), {top5_accuracy:.2f}% (Top-5)', fontdict=title_font)

# Rotate x labels for better fit
plt.xticks(rotation=45, ha='right', **tick_label_font)  # Increase rotation
plt.yticks(rotation=0, **tick_label_font)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\images\FULL_ConvNeXtBase_conf_matrix_order.png")
plt.close()


#------------------------------------------------------------------------------------------------------------------------

# Predict on the test dataset: CLASS level
taxonomic_level = 'class'  # Change to 'species', 'genus', 'family', 'order', or 'class' as needed

print('Predicting at:', taxonomic_level, 'level')
test_generator.reset()
predictions = model.predict(test_generator, steps=steps_test)

# Aggregate predictions to the specified taxonomic level
aggregated_predictions, aggregated_class_labels = aggregate_predictions(predictions, taxonomic_level=taxonomic_level)

# Get true labels at the specified taxonomic level
true_labels = test_generator.classes
aggregated_true_labels, aggregated_class_labels = get_true_labels(true_labels, taxonomic_level=taxonomic_level)

top1_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=1) * 100
top3_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=3) * 100
top5_accuracy = top_k_accuracy(aggregated_true_labels, aggregated_predictions, k=5) * 100

# Print accuracies
print(f'Top-1 test accuracy: {top1_accuracy:.2f}%')
print(f'Top-3 test accuracy: {top3_accuracy:.2f}%')
print(f'Top-5 test accuracy: {top5_accuracy:.2f}%\n')

# Generate classification report
predicted_labels = np.argmax(aggregated_predictions, axis=1)
report = classification_report(aggregated_true_labels, predicted_labels, target_names=aggregated_class_labels, zero_division=0)
print(report)

# Compute confusion matrix and normalize over rows
cm = confusion_matrix(aggregated_true_labels, predicted_labels, normalize='true')

# Plot confusion matrix
plt.figure(figsize=(20, 15))  # Larger figure for more room
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False, 
            xticklabels=aggregated_class_labels, yticklabels=aggregated_class_labels, 
            vmin=0, vmax=1, annot_kws={'size': 20})

# Adjust font sizes
title_font = {'fontsize': 18, 'fontweight': 'bold'}
label_font = {'fontsize': 16, 'color': 'black'}
tick_label_font = {'fontsize': 14, 'color': 'grey'}  # Increase font size for better visibility

# Labels and title
plt.xlabel(f'Predicted {taxonomic_level}', fontdict=label_font)
plt.ylabel(f'True {taxonomic_level}', fontdict=label_font)

# Dynamic title based on taxonomic level
if taxonomic_level == 'class':
    plt.title(f'Accuracy = {top1_accuracy:.2f}% (Top-1)', fontdict=title_font)
else:
    plt.title(f'Accuracy = {top1_accuracy:.2f}% (Top-1), {top3_accuracy:.2f}% (Top-3), {top5_accuracy:.2f}% (Top-5)', fontdict=title_font)

# Rotate x labels for better fit
plt.xticks(rotation=45, ha='right', **tick_label_font)  # Increase rotation
plt.yticks(rotation=0, **tick_label_font)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(r"C:\Users\andre\Documents\PhD\I_year\Amazon_CNN\results\images\FULL_ConvNeXtBase_conf_matrix_class.png")
plt.close()