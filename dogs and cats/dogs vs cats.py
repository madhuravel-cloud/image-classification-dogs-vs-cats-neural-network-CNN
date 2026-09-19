from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

classifier = Sequential()

classifier.add(Conv2D(filters=32, kernel_size=3, input_shape=(128,128,3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=2, strides=2))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

classifier.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    r"C:\Users\user\Downloads\dogs-vs-cats\train\train",
    target_size=(128,128),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    r"C:\Users\user\Downloads\dogs-vs-cats\train\train",
    target_size=(128,128),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

classifier.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator
)

ui = image.load_img(
    r"C:\Users\user\Downloads\dog_test.jpg",
    target_size=(128,128)
)

ui = image.img_to_array(ui)
ui = ui / 255.0
ui = np.expand_dims(ui, axis=0)

result = classifier.predict(ui)

if result[0][0] > 0.5:
    prediction = "dog"
else:
    prediction = "cat"

print("The given image:", prediction)
print("Prediction value:", result[0][0])

print("Train:", train_generator.samples)
print("Validation:", validation_generator.samples)
print("Classes:", train_generator.class_indices)