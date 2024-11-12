#Sólo carga los datos de CIFAR-10 con Keras
#Selecciona tres imágenes aletroias del conjunto de prueba
#Nos dice de qué clase son

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import random

def representarImagen(imagen, titulo):
    plt.figure()
    plt.imshow(imagen)
    plt.title(titulo)
    plt.axis('off')
    plt.show()

# Cargar el dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Coge 3 número aleatorios entre 0 y el número de imágenes en el conjunto de prueba
indices = random.sample(range(len(x_test)), 3)

#Para cada imagen la lee, la representa y nos dice a qué clase pertenece
for prueba in indices:
    imagen = x_test[prueba]
    clase = y_test[prueba][0]
    representarImagen(imagen, f"Clase: {clase}")
    print(f"Clase: {clase}")