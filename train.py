import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from data_loader import load_dataset
from model import build_model
from utils import save_results

# Load dataset
path = load_dataset()

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7,1.3]
)

train_data = datagen.flow_from_directory(
    path,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    path,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

optimizers = {
    "adam": tf.keras.optimizers.Adam(),
    "sgd": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "rmsprop": tf.keras.optimizers.RMSprop()
}

histories = {}

for name, opt in optimizers.items():
    print(f"\nTraining with {name}")

    model = build_model()

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(f"best_{name}.h5", save_best_only=True)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=5,
        callbacks=[checkpoint]
    )

    histories[name] = history.history

    loss, acc = model.evaluate(val_data)
    save_results(name, acc)

# Plot comparison
for name in histories:
    plt.plot(histories[name]['val_accuracy'], label=name)

plt.title("Optimizer Comparison")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()
