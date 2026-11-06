import shutil
import os
import subprocess
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
except ImportError:
    import keras as tf
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications import MobileNetV2
    from keras.models import Sequential
    from keras.layers import Dense, GlobalAveragePooling2D, Dropout



#--------------------  CLONING THE OFFICIAL REPOSITORY  -------------

# Check if 'solar_data' already exists to avoid permission errors and unnecessary downloads
if not os.path.exists('solar_data'):
    print("Cloning the official Raptor Maps repository directly into 'solar_data'...")
    subprocess.run(
        ["git", "clone", "https://github.com/RaptorMaps/InfraredSolarModules.git", "solar_data"],
        check=True,
    )
    print("¡Listo! Dataset descargado con éxito en la carpeta ./solar_data")
else:
    print("La carpeta 'solar_data' ya existe localmente. Omitiendo clonación.")


#-----------------------  DATASET DESCOMPRESSION  ------------------

# We check the main folders within solar_data
print(os.listdir('./solar_data'))


# ----------------- INSPECTION OF THE FILE STRUCTURE ---------------

# Path to the zip file within the solar_data folder
zip_path = './solar_data/2020-02-14_InfraredSolarModules.zip'

# Extract into the same solar_data folder
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('./solar_data')

print("¡Archive successfully decompressed!")

# We check what folders or files we have now within solar_data
print("New content:", os.listdir('./solar_data'))

# Path to the unzipped folder
file_route = './solar_data/InfraredSolarModules'

# See what files are inside the InfraredSolarModules folder  
print("Content of InfraredSolarModules:", os.listdir(file_route))


# ------------------ LOADING AND TRANSPOSITION OF THE METADATA FILE -----------------

# Path to the metadata file
ruta_json = './solar_data/InfraredSolarModules/module_metadata.json'

# Load the file into a DataFrame
df = pd.read_json(ruta_json)

# Display the first rows to see what data it contains (OPTIONAL)
print("First rows of the metadata file:")
print(df.head())

# Show the available columns 
#print("\nColumns in the table:", df.columns.tolist())

# Transpose the table so that each row is a solar panel and the columns are their data
df_clean = df.T.reset_index(drop=True)

# Display the first rows already ordered correctly
print("First rows organized:")
print(df_clean.head())

# Count how many different types of anomalies exist in the entire dataset
print("\nNumber of images for each anomaly type:")
print(df_clean['anomaly_class'].value_counts())


# ------------------ VISUALIZATION AND VISUAL ANALYSIS OF THE SAMPLES (OPTIONAL) ----------------

# Select 4 random samples
samples = df_clean.sample(4)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, (_, row) in enumerate(samples.iterrows()):
    img_path = './solar_data/InfraredSolarModules/' + row['image_filepath']
    label = row['anomaly_class']
    
    img = mpimg.imread(img_path)
    
    # Use the 'inferno' or 'hot' colormap to give a real thermal camera appearance. Use 'gray' for a more classic look
    im = axes[i].imshow(img, cmap='inferno')
    axes[i].set_title(f"Clase: {label}", fontsize=11)
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# ------------------ PREPARATION OF THE DATASET FOR TRAIN/VAL/TEST ----------------


IMG_SIZE = (40, 40)
BATCH_SIZE = 32 

# Create the binary column correctly using 'No-Anomaly' as class '0'
df_clean['binary_class'] = df_clean['anomaly_class'].apply(
    lambda x: '0' if str(x).strip() == 'No-Anomaly' else '1'
)

# First split: Isolate the Test Set (10%) with stratification
df_train_val, df_test = train_test_split(
    df_clean, 
    test_size=0.10, 
    random_state=42, 
    stratify=df_clean['binary_class'] 
)

# Second split: Divide the remaining 90% into Train (80%) and Validation (20%)
df_train, df_val = train_test_split(
    df_train_val, 
    test_size=0.20, 
    random_state=42, 
    stratify=df_train_val['binary_class']
)

print(f"Dataset Distribution:")
print(f"   - Train: {len(df_train)} images")
print(f"   - Val:      {len(df_val)} images")
print(f"   - Test:   {len(df_test)} images")

# Configure Keras Image Generators
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory='./solar_data/InfraredSolarModules/',
    x_col='image_filepath',
    y_col='binary_class',
    target_size=IMG_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=True 
)

val_gen = val_test_datagen.flow_from_dataframe(
    dataframe=df_val,
    directory='./solar_data/InfraredSolarModules/',
    x_col='image_filepath',
    y_col='binary_class',
    target_size=IMG_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_gen = val_test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory='./solar_data/InfraredSolarModules/',
    x_col='image_filepath',
    y_col='binary_class',
    target_size=IMG_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=False 
)


# ------------------ MODEL CREATION AND COMPILATION ----------------


# Upload pretrainedMobileNetV2 (whitout the top layer) and add our own layers for binary classification
# Look!: MobileNetV2 expects images of at least 32x32, and as our images are 40x40, it will work perfectly.
base_model_bin = MobileNetV2(input_shape=(40, 40, 3),
                             include_top=False,
                             weights='imagenet')
base_model_bin.trainable = False

# Freeze the base model so that its weights are not updated during training
base_model_bin.trainable = False

# Create our sequential model by connecting MobileNet with our final layers
model_binary = Sequential([
    base_model_bin,
    GlobalAveragePooling2D(), # Add a global average pooling layer to reduce the spatial dimensions of the feature maps
    Dense(64, activation='relu'), # Add a fully connected layer with 64 neurons and ReLU activation
    Dropout(0.5), # Prevent overfitting by randomly dropping 50% of the neurons during training
    Dense(1, activation='sigmoid') # Add the final output layer with 1 neuron (for binary classification) and sigmoid activation to get a probability for the positive class
])

# Compile the model
model_binary.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

# Show the summary
model_binary.summary()



#-------------------- TRAINING THE MODEL --------------------- 

# Training during N epochs
N_EPOCHS = 10
history_bin = model_binary.fit(
    train_gen,
    validation_data=val_gen,
    epochs=N_EPOCHS
)

print("\n¡Binary training completed!")


#------------ VISUALIZATION OF THE TRAINING/VALIDATIONHISTORY ---------------------


# Subtract the data from training history
acc = history_bin.history['accuracy']
val_acc = history_bin.history['val_accuracy']
loss = history_bin.history['loss']
val_loss = history_bin.history['val_loss']
epochs_range = range(len(acc))

# Create a figure with two subplots side by side
plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training')
plt.plot(epochs_range, val_acc, label='Validation')
plt.legend(loc='lower right')
plt.title('Model Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training')
plt.plot(epochs_range, val_loss, label='Validation')
plt.legend(loc='upper right')
plt.title('Model Loss')

plt.show()


# ------------- MODEL EVALUATION WITH THE TEST SET -----------------

# Evaluate our model 'model_binary' with this totally new test set
test_loss, test_accuracy = model_binary.evaluate(test_gen)

print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# ------------- VISUALIZATION OF THE PREDICTIONS WITH THE TEST SET -----------------

# Obtain a batch of images from the Test generator (configured with shuffle=False)
images_test, labels_test = next(test_gen)

# Perform predictions with our binary model
preds_test = model_binary.predict(images_test)

# Configure the visualization of 8 examples (2 rows x 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(8):
    img = images_test[i]
    real_val = labels_test[i]
    real_label = "Anomaly" if real_val == 1 else "No-Anomaly"
    
    # Obtain the probability (sigmoid output between 0 and 1)
    prob = preds_test[i][0]
    pred_label = "Anomaly" if prob > 0.5 else "No-Anomaly"
    
    # Color green if the model was correct, red if it was wrong
    color = 'green' if real_label == pred_label else 'red'
    
    # Show the thermal image
    axes[i].imshow(img)
    axes[i].set_title(f"Real: {real_label}\nPred: {pred_label}\nConfianza: {prob:.2f}", color=color, fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
plt.show()