#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


## 

def build_model1():
  model = Sequential([
      layers.Flatten(input_shape=(32, 32, 3)),
      layers.Dense(128),
      layers.LeakyReLU(alpha=0.01),

      layers.Dense(128),
      layers.LeakyReLU(alpha=0.01),

      layers.Dense(128),
      layers.LeakyReLU(alpha=0.01),

      layers.Dense(10)  # logits (no activation)
  ])
  return model

def build_model2():
  model = Sequential([
    # Input: (32, 32, 3)

    layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu",
                  input_shape=(32, 32, 3)),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
    layers.BatchNormalization(),

    # Four more Conv2D+BatchNorm pairs, stride defaults to 1
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),

    # Conv 2D: 128 filters, 3x3, same padding (no activation specified in prompt)
    layers.Conv2D(128, (3, 3), padding="same"),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(10)  # logits
])
  return model

def build_model3():
  model = Sequential([
      # First conv stays standard Conv2D
      layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu",
                    input_shape=(32, 32, 3)),
      layers.BatchNormalization(),

      # All remaining conv layers become depthwise-separable convs
      layers.SeparableConv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
      layers.BatchNormalization(),

      # Four more pairs, stride defaults to 1
      layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu"),
      layers.BatchNormalization(),

      layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu"),
      layers.BatchNormalization(),

      layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu"),
      layers.BatchNormalization(),

      layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu"),
      layers.BatchNormalization(),

      # Final conv becomes separable too (prompt says “all except first”)
      layers.SeparableConv2D(128, (3, 3), padding="same"),
      layers.BatchNormalization(),

      layers.Flatten(),
      layers.Dense(10)  # logits
  ])
  return model

def build_model50k():
  model = None # Add code to define model 1.
  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
    # Convert to float and normalize to [0, 1]
    train_images = train_images.astype("float32") / 255.0
    test_images  = test_images.astype("float32") / 255.0
  ########################################
 ## Split training into training and validation
  # CIFAR-10 train has 50,000 images. We'll use 45,000 train / 5,000 val.
    val_size = 5000
    val_images = train_images[-val_size:]
    val_labels = train_labels[-val_size:]
    train_images = train_images[:-val_size]
    train_labels = train_labels[:-val_size]
  ########################################
  ## Build and train model 1
    model1 = build_model1()
  # compile and train model 1.
  
    model1.compile(
      optimizer="adam",
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=["accuracy"]
  )

    model1.summary()

    history1 = model1.fit(
      train_images, train_labels,
      epochs=30,
      validation_data=(val_images, val_labels),
      batch_size=64,
      verbose=2
  )
  ## Build, compile, and train model 2 (DS Convolutions)
    model2 = build_model2()
model2.compile(
      optimizer="adam",
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=["accuracy"]
  )
model2.summary()

history2 = model2.fit(
      train_images, train_labels,
      epochs=30,
      validation_data=(val_images, val_labels),
      batch_size=64,
      verbose=2
  )

  ### Repeat for model 3 and your best sub-50k params model
  # Load image (must be 32x32 RGB)
test_img = np.array(keras.utils.load_img(
    './test_image_bird.jpg',   # <-- rename your saved file to match
    color_mode='rgb',
    target_size=(32, 32)
), dtype=np.float32)

# Preprocess exactly like CIFAR-10
test_img = test_img / 255.0

# Add batch dimension: (32,32,3) -> (1,32,32,3)
test_img = np.expand_dims(test_img, axis=0)

# Predict (your model outputs logits since last layer has no activation)
logits = model2.predict(test_img, verbose=0)          # shape (1,10)
probs  = tf.nn.softmax(logits, axis=1).numpy()[0]    # convert logits -> probabilities

pred_idx = int(np.argmax(probs))
pred_label = class_names[pred_idx]
pred_conf = float(probs[pred_idx])

print("Predicted:", pred_label, "confidence:", pred_conf)

# (Optional) top-3 guesses
top3 = np.argsort(-probs)[:3]
print("Top-3:")
for i in top3:
    print(f"  {class_names[int(i)]}: {float(probs[int(i)]):.4f}")


### Build, compile, and train model 3 (Depthwise-Separable Conv)
    model3 = build_model3()

    model3.compile(
      optimizer="adam",
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=["accuracy"]
  )

    model3.summary()

    history3 = model3.fit(
      train_images, train_labels,
      epochs=30,
      validation_data=(val_images, val_labels),
      batch_size=64,
      verbose=2
  )

  # Optional: evaluate on test set
    test_loss3, test_acc3 = model3.evaluate(test_images, test_labels, verbose=0)
    print(f"Model3 Test Accuracy: {test_acc3:.4f}, Test Loss: {test_loss3:.4f}")