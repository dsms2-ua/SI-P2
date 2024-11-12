import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import numpy as np
import random

def cargarNormalizarDatos():
    # Cargar el dataset CIFAR-10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalizar los datos
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    return x_train, y_train, x_test, y_test

def crearModeloSesion1(neuronas, funcion, optimizador, validationSplit, epocas, batchSize, x_train, y_train):
    #Crear el modelo
    model = tf.keras.Sequential([tf.keras.layers.Dense(neuronas, activation=funcion, input_shape=(3072,)), 
                                tf.keras.layers.Dense(10, activation='softmax')])

    # Compilar el modelo
    model.compile(optimizer=optimizador, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #Entrenar el modelo
    history = model.fit(x_train, y_train, validation_split = validationSplit, epochs = epocas, batch_size = batchSize)

    return model, history

def evaluarModelo(x_test, y_test, model):
    #Evaluamos el modelo con el conjunto de pruebas
    test_loss, test_acc = model.evaluate(x_test, y_test)
    return test_loss, test_acc

#Definimos la función para guardar el modelo
def saveModelIfBetter(test_acc, model):
    #Aquí leo de una carpeta para ver:
    #   1. Leo de un archivo de texto el mejor error hasta el momento
    #   2. Si es mejor que el error actual, guardo el modelo
    #   3. Guardo el error actual en el archivo de texto
    #   4. Si los archivos no existen, los creo

    #1. Leo de un archivo de texto el mejor error hasta el momento
    try:
        with open("dataInfo/bestValue.txt", "r") as archivo:
            best_error = float(archivo.read())
    except FileNotFoundError:
        print("No se ha encontrado el archivo best_error.txt")
        
    #2. Si es mejor que el error actual, guardo el modelo
    if test_acc > best_error:
        model.save("dataInfo/bestModel.h5")
        print("Modelo guardado")
        
    #3. Guardo el error actual en el archivo de texto
    with open("dataInfo/bestValue.txt", "w") as archivo:
        pass #Lo abrimos en modeo escritura y así se borra
    with open("dataInfo/bestValue.txt", "w") as archivo:
        archivo.write(str(test_acc))


def plotModel(history):
    #Gráfica de entrenamiento
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión en validación')
    plt.title("Precisión del modelo")
    plt.xlabel("Época")
    plt.ylabel("Precisión")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #Cargamos el modelo y normalizamos
    x_train, y_train, x_test, y_test = cargarNormalizarDatos()
    
    
    saveModelIfBetter()
