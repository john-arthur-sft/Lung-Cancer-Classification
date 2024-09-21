import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Sequential


# root Folders
test_root = "dataset/test"
train_root = "dataset/train"
valid = "dataset/valid"

# test
test_adenocarcinoma = "dataset/test/adenocarcinoma"
test_lc_carcinoma = "dataset/test/large.cell.carcinoma"
test_normal = "dataset/test/normal"
test_sc_carcinoma = "dataset/test/squamous.cell.carcinoma"

#train
train_adenocarcinoma = "dataset/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib"
train_lc_carcinoma = "dataset/train/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa"
train_normal = "dataset/train/normal"
train_sc_carcinoma = "dataset/train/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa"

#validation
valid_adenocarcinoma = "dataset/valid/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib"
valid_lc_carcinoma = "dataset/valid/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa"
valid_normal = "dataset/valid/normal"
valid_sc_carcinoma = "dataset/valid/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa"

print("[ln25: 1]:  Data files initialized successfully.\n")

print("[ln27: 2]: Summary\n")
print(f"Type          \t\t Test     \t\t Train     \t\t Validation\n"
      f"Adenocarcinoma\t\t {len(test_adenocarcinoma)}     \t\t {len(train_adenocarcinoma)}     \t\t {len(valid_adenocarcinoma)}\n"
      f"LC_carcinoma  \t\t {len(test_lc_carcinoma)}       \t\t {len(train_lc_carcinoma)}     \t\t {len(valid_lc_carcinoma)}\n"
      f"SC_carcinoma  \t\t {len(test_sc_carcinoma)}       \t\t {len(train_sc_carcinoma)}     \t\t {len(valid_sc_carcinoma)}\n"
      f"Normal        \t\t {len(test_normal)}     \t\t {len(train_normal)}     \t\t {len(valid_normal)}\n"
      f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

print("[ln35: 3]: Data Preparation Initiated")

batch_size = 8
img_h, img_w = 255,255

test_datagen = ImageDataGenerator(
      rescale=1./255,
      target_size=(255,255),
      horizontal_flip=True,
      brightness_range=(0,7)
)


train_datagen =ImageDataGenerator(
      rescale=1./255,
      horizontal_flip=True,
      brightness_range=(0,7)
)

train_generator = train_datagen.flow_from_directory(
      train_root,
      batch_size=batch_size
)

valid_generator = test_datagen.flow_from_directory(
      test_root,
      batch_size = batch_size
)

print("[ln69: 4]: Data Preparation Completed.\n")
print("[ln70: 5]: Building ResNet Model")

resNet_model = Sequential()
pretrained_model = ResNet50(
      include_top = False, input_shape = (img_h, img_w, 3), pooling='avg', weights='imagenet'
)

for layer in pretrained_model.layers:
      layer.trainable = False


resNet_model.add(pretrained_model)
resNet_model.add(layers.Flatten())
resNet_model.add(layers.Dense(1024, activation="relu"))
resNet_model.add(layers.Dense(4, activation="softmax"))

resNet_model.compile(optimizer= tf.keras.optimizers.Adam(0.001), loss= "categorical_crossentropy", metrics=['accuracy'])
history = resNet_model.fit(train_generator, validation_data = valid_generator, epochs=10)
