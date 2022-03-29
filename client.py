import flwr as fl 
import tensorflow as tf
from tensorflow.keras import backend as K
#import tensorflow as tf 
from tensorflow.keras import Input, Model, layers, models # 建立CNN架構
import numpy as np # 資料前處理
import random
from tensorflow.keras.initializers import GlorotUniform
import argparse # CmmandLine 輸入控制參數
import os # 更改tensorflow的Log訊息的顯示模式

# Make TensorFlow logs less verbose (減少不必要的訊息顯示)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
count = 0
pre_model_person = 0 
pre_model_present = 0
'''
Step 1. Build Local Model (建立本地模型)
'''
# Hyperparameter超參數
num_classes = 10
input_shape = (28, 28, 1)
# Build Model
def CNN_Model(input_shape, number_classes):
  # define Input layer
  input_tensor = Input(shape=input_shape) # Input: convert normal numpy to Tensor (float32)

  # define layer connection
  # representation_model
  representation_model = models.Sequential([
      layers.Conv2D(filters = 32, kernel_size=(3, 3), activation="relu",input_shape=input_shape),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Conv2D(filters = 64, kernel_size=(3, 3), activation="relu"),
      layers.MaxPooling2D(pool_size=(2, 2)),
      ])
  representation_model._name = "representation"
  x = representation_model(input_tensor)
  #personalize_model
  personalize_model = models.Sequential([
      layers.Flatten(),
      layers.Dropout(0.5),
      layers.Dense(number_classes, activation="softmax"),
     ])
  personalize_model._name = "personalize"

  outputs = personalize_model(x)

  # define model
  model = Model(inputs=input_tensor, outputs=outputs, name="mnist_model")
  return model
def alpha_update(alpha,global_representation_parameter,local_representation_parameter,local_personalize_parameter,num_classes,input_shape,x_train,y_train):
    grad_alpha = 0
    alpha_n = 0
    ipt = Input(input_shape)
    #train data and label (add the batch size dim)
    x_train = tf.expand_dims(x_train,axis=0)
    y_train = tf.expand_dims(y_train,axis=0)
    #create model
    representation_model = models.Sequential([
        layers.Conv2D(filters = 32, kernel_size=(3, 3), activation="relu",input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters = 64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
      ])
    representation_model._name = "representation"
    x = representation_model(ipt)
    #personalize_model
    personalize_model = models.Sequential([
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
       ])
    personalize_model._name = "personalize"
    outputs = personalize_model(x)

  # define model
    model = Model(inputs=ipt, outputs=outputs)
  #set global model parameter
    model.get_layer("representation").set_weights(global_representation_parameter)
    model.get_layer("personalize").set_weights(local_personalize_parameter)
    for layer in model.get_layer("representation").layers:
        layer.trainable = True
    for layer in model.get_layer("personalize").layers:
        layer.trainable = False
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    '''
    ipt = Input(input_shape)
    out = layers.Conv2D(filters = 32, kernel_size=(3, 3), activation="relu",input_shape=input_shape)(ipt)
    out = layers.MaxPooling2D(pool_size=(2, 2))(out)
    out = layers.Conv2D(filters = 64, kernel_size=(3, 3), activation="relu")(out)
    out = layers.MaxPooling2D(pool_size=(2, 2))(out)
    out = layers.Flatten()(out)
    out = layers.Dropout(0.5)(out)
    out = layers.Dense(num_classes, activation="softmax")(out)
    model = Model(ipt,out)
    '''
   
    #forward pass
    with tf.GradientTape() as tape:
        pred = model(x_train)
        loss = tf.keras.losses.categorical_crossentropy(y_train,pred)
    #get gradient
    global_grad = tape.gradient(loss, model.trainable_variables)
    
    #set local model parameter
    model.get_layer("representation").set_weights(local_representation_parameter)
    model.get_layer("personalize").set_weights(local_personalize_parameter)
    for layer in model.get_layer("representation").layers:
        layer.trainable = True
    for layer in model.get_layer("personalize").layers:
        layer.trainable = False
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    #forward pass
    with tf.GradientTape() as tape:
        pred = model(x_train)
        loss = tf.keras.losses.categorical_crossentropy(y_train,pred)
    #get gradient
    local_grad = tape.gradient(loss, model.trainable_variables)

    for i in range(len(local_grad)):
        dif = global_representation_parameter[i]-local_representation_parameter[i]
        grad = alpha*global_grad[i] + (1-alpha)*local_grad[i]
        dif = tf.reshape(dif,-1)
        grad = tf.reshape(grad,-1)
        grad_alpha = grad_alpha + tf.tensordot(tf.transpose(dif),grad,1)
    alpha = alpha - 0.5*grad_alpha
    alpha = np.clip(np.array(alpha),0.0,1.0)
    alpha = alpha.item()
    print(alpha)
    return_parameter = global_representation_parameter
    return return_parameter

'''
Step 2. Load local dataset (引入本地端資料集)，對Dataset進行切割
'''

def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    #assert idx in range(10) # limit it can't more than 10...
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # Train 60000-5000, Test 10000

    # Data preprocessing
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    sample_size = 60000 #x_train.shape[0]
    sample_size -= 5000 # Server-side 預留做驗證
    training_size = int(sample_size/3)
    testing_size = int(10000/3)

    return ( 
        x_train[idx * training_size : (idx + 1) * training_size],
        y_train[idx * training_size : (idx + 1) * training_size],
    ), (
        x_test[idx * testing_size : (idx + 1) * testing_size],
        y_test[idx * testing_size : (idx + 1) * testing_size],
    )

'''
Step 3. Define Flower client (定義client的相關設定: 接收Server-side的global model weight、hyperparameters)
'''
class MnistClient(fl.client.NumPyClient):
    # Class初始化: local model、dataset
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    # 此時已無作用，原用來取得 local model 的 ini-weight，
    # 目前初始權重值是來自 Server-side 而非 client 自己
    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        #count -> training round
        #pre_model_person->previous round personalize layer
        global count
        global pre_model_person
        global pre_model_present
        global ada_model_present
        print("client start {} time\n".format(count))
        # Update local model parameters
        self.model.set_weights(parameters)
               
        # tet hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        print("batch_size{}  epoch{}".format(batch_size,epochs))

        #return global model set personalize layer parameter        
        if(count==0):
            pre_model_person = self.model.get_layer("personalize").get_weights()
            pre_model_present = self.model.get_layer("representation").get_weights()
        if(count!=0):
            self.model.get_layer("personalize").set_weights(pre_model_person)
            ada_model_present = alpha_update(0.5,self.model.get_layer("representation").get_weights(),pre_model_present,pre_model_person,10,(28,28,1),self.x_train[0],self.y_train[0])
            self.model.get_layer("representation").set_weights(ada_model_present)
        #fix representation layer - > train personalize layer
        for layer in self.model.get_layer("representation").layers:
            layer.trainable = False
        for layer in self.model.get_layer("personalize").layers:
            layer.trainable = True
        self.model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

        print("fix representation layer - > train personalize layer:acc")
        # Train the model using hyperparameters from config
        # (依 Server-side 的 hyperparameters 進行訓練)
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        #fix personalize layer -> train representation layer
        for layer in self.model.get_layer("representation").layers:
            layer.trainable = True
        for layer in self.model.get_layer("personalize").layers:
            layer.trainable = False
        print("fix personalize layer -> train representation layer:acc")
        self.model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )
        
        count = count + 1
        #save personaluze layer and representation layer (parameter)
        pre_model_person = self.model.get_layer("personalize").get_weights()
        pre_model_present = self.model.get_layer("representation").get_weights()

        # Return updated model parameters and results
        # 將訓練後的權重、資料集筆數、正確率/loss值等，回傳至server-side
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
        }
        print(" local model on test set")
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        print("local model test acc{}".format(accuracy))
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)
        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}

'''
Step 4. Create an instance of our flower client and add one line to actually run this client. (建立Client-to-Server的連線)
'''
def main() -> None:
    # Parse command line argument `partition`
    # 從 CommandLine 輸入 Client 編號，對 Dataset進行切割
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    args = parser.parse_args()
    global count
    # Load and compile Keras model
    model = CNN_Model(input_shape=(28, 28, 1), number_classes=10)
    #model.summary()
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    # Load a subset of CIFAR-10 to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = MnistClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("localhost:8080", client=client) # windows

if __name__ == "__main__":
    main()
    print("client end\n")
