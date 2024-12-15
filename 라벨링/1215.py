import tensorflow as tf
from tensorflow.keras import layers, models

# PilotNet 모델 정의
def create_pilotnet_model():
    model = models.Sequential()

    # Convolutional Layers
    model.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)))
    model.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flattening the output
    model.add(layers.Flatten())

    # Fully Connected Layers
    model.add(layers.Dense(1164, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))

    # Output Layer for 5 Classes
    model.add(layers.Dense(5, activation='softmax'))

    return model

# 모델 생성
model = create_pilotnet_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
