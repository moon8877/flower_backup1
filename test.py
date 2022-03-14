import tensorflow as tf
from tensorflow.keras import Input, Model, layers, models # 建立CNN架構
import numpy as np # 資料前處理
# Build Model
def CNN_Model(input_shape=(28,28,1), number_classes=10):
  # define Input layer
  
  input_tensor = Input(shape=input_shape) # Input: convert normal numpy to Tensor (float32)

  # define layer connection
  representation_model = models.Sequential([
      layers.Conv2D(filters = 32, kernel_size=(3, 3), activation="relu",input_shape=input_shape),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Conv2D(filters = 64, kernel_size=(3, 3), activation="relu"),
      layers.MaxPooling2D(pool_size=(2, 2)),
      ])
  representation_model._name = "representation"
 # x = layers.Conv2D(filters = 32, kernel_size=(3, 3), activation="relu")(input_tensor)
 # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
 # x = layers.Conv2D(filters = 64, kernel_size=(3, 3), activation="relu")(x)
 # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  
  x = representation_model(input_tensor)
  personalize_model = models.Sequential([
      layers.Flatten(),
      layers.Dropout(0.5),
      layers.Dense(number_classes, activation="softmax"),
     ])
  personalize_model._name = "personalize"
  outputs = personalize_model(x)
 # x = layers.Flatten()(x)
 # x = layers.Dropout(0.5)(x)
 # outputs = layers.Dense(number_classes, activation="softmax")(x)

  # define model
  
  model = Model(inputs=input_tensor, outputs=outputs, name="mnist_model")
  
  return model

if __name__ == "__main__":
    model = CNN_Model()
    print(model.get_layer('personalize').get_weights())
     
    for layer in model.get_layer('personalize').layers:
        print(layer.trainable)
    


