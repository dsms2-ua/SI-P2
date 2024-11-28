import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import time, itertools

def cargarNormalizarDatos():
    # Cargamos el dataset (conjunto de entrenamiento y de test) CIFAR-10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalizamos los datos al rango [0, 1]
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    return x_train, y_train, x_test, y_test

def cargarDatosCNN():
    # Cargamos el dataset CIFAR-10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalizamos los datos
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def crearModeloSesion1(neuronas, funcion, optimizador, neuronasSalida, activacionSalida):
    #Crear el modelo
    model = tf.keras.Sequential([tf.keras.layers.Dense(neuronas, activation=funcion, input_shape=(3072,)), 
                                tf.keras.layers.Dense(neuronasSalida, activation=activacionSalida)])

    # Compilar el modelo
    model.compile(optimizer=optimizador, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def entrenarModelo(model, x_train, y_train, validationSplit, epocas, batchSize):
    #Entrenar el modelo
    start_time = time.time()
    
    #Definimos el callback
    early_stopping = EarlyStopping(
        monitor='val_loss',    # Métrica a monitorear
        mode='min',            # 'min' porque buscamos minimizar la pérdida
        patience=15,            # Número de épocas de paciencia
        restore_best_weights=True  # Restaura los mejores pesos encontrados
    )
    
    history = model.fit(x_train, y_train, validation_split = validationSplit, epochs = epocas, batch_size = batchSize, callbacks = [early_stopping])

    elapsed_time = time.time() - start_time
    return history, elapsed_time

def evaluarModelo(x_test, y_test, model):
    #Evaluamos el modelo con el conjunto de pruebas
    test_loss, test_acc = model.evaluate(x_test, y_test)
    return test_loss, test_acc

def mostrarMatrizConfusion(model, x_test, y_test, directorio):
    #Genera y muestra la matriz de confusión del modelo.
    
    etiquetas_clases = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo', 'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas_clases)
    
    fig, ax = plt.subplots(figsize=(8,8))
    
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    
    plt.savefig(directorio)
    plt.close(fig)
    
def loadData(prueba):
    directorio = "pruebas/pruebas" + prueba
    
    #Creamos las listas para almacenar los datos de archivos
    datos, tiempo, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss = [], [], [], [], [], [], [], []
    
    #Filtramos los archivos .txt y los ordenamos
    archivos = [file for file in os.listdir(directorio) if file.endswith(".txt")]
    if prueba != "2D":
        archivos_ordenados = sorted(archivos, key=lambda x: int(x.split('_')[1].split('.')[0]))
    else:
        archivos_ordenados = sorted(archivos, key=lambda x: x.split('_')[1].split('.')[0])
    
    #Llenamos las listas con los datos
    for file in archivos_ordenados:
        with open(os.path.join(directorio, file), "r") as archivo:
            lines = archivo.readlines()
            
            #Extraemos el dato a evaluar
            dato = file.split('_')[1].split('.')[0]
            #Probamos a convertir el dato a entero, si diera error nos lo quedamos como string
            try:
                dato = int(dato)
            except:
                dato = dato
            datos.append(dato)
            
            #Extraemos los datos para las listas
            prom_tiempo = float(lines[1].split(": ")[1].strip().split(" ")[0])
            prom_train_acc = float(lines[2].split(": ")[1].strip())
            prom_train_loss = float(lines[3].split(": ")[1].strip())
            prom_val_acc = float(lines[4].split(": ")[1].strip())
            prom_val_loss = float(lines[5].split(": ")[1].strip())
            prom_test_acc = float(lines[6].split(": ")[1].strip())
            prom_test_loss = float(lines[7].split(": ")[1].strip())
            
            #Agregamos a las listas
            tiempo.append(prom_tiempo)
            train_acc.append(prom_train_acc)
            train_loss.append(prom_train_loss)
            val_acc.append(prom_val_acc)
            val_loss.append(prom_val_loss)
            test_acc.append(prom_test_acc)
            test_loss.append(prom_test_loss)
    
    return tiempo, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss, datos


def plotModel(prueba):
    tiempo, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss, datos = loadData(prueba)
    
    #Podemos crear las gráficas
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    if prueba == "2B":
        label = "Épocas"
    elif prueba == "2C":
        label = "Batch size"
    elif prueba == "2D":
        label = "Función de activación"
    elif prueba == "2E":
        label = "Número de neuronas"
    elif prueba == "H1" or prueba == "H2":
        label = "Kernel"
    
    #1. Gráfica de líneas para entrenamiento y validación
    axs[0].plot(datos, train_acc, label="Precisión en entrenamiento", marker='.')
    axs[0].plot(datos, val_acc, label="Precisión en validación", marker='.')
    axs[0].plot(datos, train_loss, label="Pérdida en entrenamiento", marker='.')
    axs[0].plot(datos, val_loss, label="Pérdida en validación", marker='.')
    
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title("Entrenamiento y Validación")
    axs[0].set_xlabel(label)

    #2. Gráfica de barras para test
    ancho = 0.3
    x = np.arange(len(datos))
    test_acc_percent = [acc*100 for acc in test_acc]
    test_loss_percent = [loss*10 for loss in test_loss]
    
    axs[1].bar(x - ancho, test_acc_percent, width=ancho, label="Precisión en test (%)", color="blue")
    axs[1].bar(x, test_loss_percent, width=ancho, label="Pérdida en test (%)", color="red")
    axs[1].bar(x + ancho, tiempo, width=ancho, label="Tiempo de entrenamiento (s)", color="green")

    axs[1].legend()
    axs[1].grid()
    axs[1].set_title("Test y Tiempo")
    axs[1].set_xlabel(label)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(datos)
    
    plt.tight_layout()
    directorio = "pruebas/pruebas" + prueba
    output_path = directorio + "/grafica.png"
    plt.savefig(output_path)
    plt.close(fig)
    
    

#Creamos el modelo básico para crear la estructura de la red neuronal
def tarea_2A():
    x_train, y_train, x_test, y_test = cargarNormalizarDatos()
    model = crearModeloSesion1(32, 'sigmoid', 'adam', 10, 'softmax')
    history, elapsed_time = entrenarModelo(model, x_train, y_train, 0.1, 100, 128)

#En esta tarea analizamos el número de épocas que evitan sobreentrenamiento
def tarea_2B(inicio, final):
    #Cargamos el modelo y normalizamos
    x_train, y_train, x_test, y_test = cargarNormalizarDatos()
    
    # Parámetros del modelo y entrenamiento
    neuronas = 32
    funcion = 'sigmoid'
    optimizador = 'adam'
    neuronasSalida = 10
    activacionSalida = 'softmax'
    validationSplit = 0.1
    batchSize = 128
    repeticiones = 5
    
    epocas = []
    
    for i in range(inicio, final + 2, 2):
        epocas.append(i)
        
    mejorPrecision = 0
        
    #Entrenamos el modelo según el número de repeticiones
    for epoca in epocas:
        # Variables para acumular los promedios
        total_time = 0
        train_acc_list, train_loss_list, val_acc_list, val_loss_list, test_acc_list, test_loss_list, models = [], [], [], [], [], [], []
        for _ in range(repeticiones):
            #Creamos el modelo
            model = crearModeloSesion1(neuronas, funcion, optimizador, neuronasSalida, activacionSalida)
            models.append(model)

            history, elapsed_time = entrenarModelo(model, x_train, y_train, validationSplit, epoca, batchSize)
            test_loss, test_acc = evaluarModelo(x_test, y_test, model)
            
            if test_acc > mejorPrecision:
                mejorPrecision = test_acc
                model.save("modelos/2B.h5")
            
            # Acumulamos los tiempos y los resultados de cada repetición
            total_time += elapsed_time
            train_acc_list.append(history.history['accuracy'][-1])
            train_loss_list.append(history.history['loss'][-1])
            val_acc_list.append(history.history['val_accuracy'][-1])
            val_loss_list.append(history.history['val_loss'][-1])
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            
        #Para cada valor de épocas, genero la matriz de confusión para el mejor
        for i in range(repeticiones):
            if test_acc_list[i] == max(test_acc_list):
                mostrarMatrizConfusion(models[i], x_test, y_test, f"pruebas/pruebas2B/confusion_epoca_{epoca}.png")

        # Cálculo de los promedios
        avg_time = total_time / repeticiones
        avg_train_acc = np.mean(train_acc_list)
        avg_train_loss = np.mean(train_loss_list)
        avg_val_acc = np.mean(val_acc_list)
        avg_val_loss = np.mean(val_loss_list)
        avg_test_acc = np.mean(test_acc_list)
        avg_test_loss = np.mean(test_loss_list)
        
        # Resultados en archivo
        try:
            with open(f"pruebas/pruebas2B/epocas_{epoca}.txt", "w") as archivo:
                archivo.write(f"Epocas: {epoca}\n")
                archivo.write(f"Promedio de tiempo de entrenamiento: {avg_time:.2f} segundos\n")
                archivo.write(f"Promedio de precisión en entrenamiento: {avg_train_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en entrenamiento: {avg_train_loss:.4f}\n")
                archivo.write(f"Promedio de precisión en validación: {avg_val_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en validación: {avg_val_loss:.4f}\n")
                archivo.write(f"Promedio de precisión en pruebas: {avg_test_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en pruebas: {avg_test_loss:.4f}\n")
        except FileNotFoundError:
            print("No se pudo guardar el archivo")

    #Representamos
    plotModel("2B")
    
#En esta tarea analizamos el tamaño del batch que evita sobreentrenamiento
def tarea_2C():
    #Cargamos el modelo y normalizamos
    x_train, y_train, x_test, y_test = cargarNormalizarDatos()
    
    # Parámetros del modelo y entrenamiento
    neuronas = 32
    funcion = 'sigmoid'
    optimizador = 'adam'
    neuronasSalida = 10
    activacionSalida = 'softmax'
    validationSplit = 0.1
    epocas = 100  # Valor de la tarea 2B
    repeticiones = 5
    
    batch_size = [16, 32, 64, 128, 256, 512]
    
    mejorPrecision = 0
     
    
    #Entrenamos el modelo según el batch_size
    for batch in batch_size:
        # Variables para acumular los promedios
        total_time = 0
        train_acc_list, train_loss_list, val_acc_list, val_loss_list, test_acc_list, test_loss_list, models = [], [], [], [], [], [], []
            
        for _ in range(repeticiones):
            #Creamos el modelo
            model = crearModeloSesion1(neuronas, funcion, optimizador, neuronasSalida, activacionSalida)
            history, elapsed_time = entrenarModelo(model, x_train, y_train, validationSplit, epocas, batch)
            models.append(model)
            test_loss, test_acc = evaluarModelo(x_test, y_test, model)
            
            if test_acc > mejorPrecision:
                mejorPrecision = test_acc
                model.save("modelos/2C.h5")
            
            # Acumulamos los tiempos y los resultados de cada repetición
            total_time += elapsed_time
            train_acc_list.append(history.history['accuracy'][-1])
            train_loss_list.append(history.history['loss'][-1])
            val_acc_list.append(history.history['val_accuracy'][-1])
            val_loss_list.append(history.history['val_loss'][-1])
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            
        #Para cada valor de épocas, genero la matriz de confusión para el mejor
        for i in range(repeticiones):
            if test_acc_list[i] == max(test_acc_list):
                mostrarMatrizConfusion(models[i], x_test, y_test, f"pruebas/pruebas2C/confusion_batch_{batch}.png")

        # Cálculo de los promedios
        avg_time = total_time / repeticiones
        avg_train_acc = np.mean(train_acc_list)
        avg_train_loss = np.mean(train_loss_list)
        avg_val_acc = np.mean(val_acc_list)
        avg_val_loss = np.mean(val_loss_list)
        avg_test_acc = np.mean(test_acc_list)
        avg_test_loss = np.mean(test_loss_list)
        
        # Resultados en archivo
        try:
            with open(f"pruebas/pruebas2C/batch_{batch}.txt", "w") as archivo:
                archivo.write(f"Batch size: {batch}\n")
                archivo.write(f"Promedio de tiempo de entrenamiento: {avg_time:.2f} segundos\n")
                archivo.write(f"Promedio de precisión en entrenamiento: {avg_train_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en entrenamiento: {avg_train_loss:.4f}\n")
                archivo.write(f"Promedio de precisión en validación: {avg_val_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en validación: {avg_val_loss:.4f}\n")
                archivo.write(f"Promedio de precisión en pruebas: {avg_test_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en pruebas: {avg_test_loss:.4f}\n")
        except FileNotFoundError:
            print("No se pudo guardar el archivo")
    
    plotModel("2C")
            
def tarea_2D():
    #Cargamos el modelo y normalizamos
    x_train, y_train, x_test, y_test = cargarNormalizarDatos()
    
    #Parámetros del modelo y entrenamiento
    neuronas = 32
    optimizador = 'adam'
    neuronasSalida = 10
    activacionSalida = 'softmax'
    validationSplit = 0.1
    epocas = 100  # Valor de la tarea 2B
    repeticiones = 5 
    batch_size = 256 #Valor de la tarea 2C

    funciones = ['sigmoid', 'relu', 'tanh']
    
    mejorPrecision = 0
    
    for funcion in funciones:
        total_time = 0
        train_acc_list, train_loss_list, val_acc_list, val_loss_list, test_acc_list, test_loss_list, models = [], [], [], [], [], [], []
        
        for _ in range(repeticiones):
            #Creamos el modelo
            model = crearModeloSesion1(neuronas, funcion, optimizador, neuronasSalida, activacionSalida)
            history, elapsed_time = entrenarModelo(model, x_train, y_train, validationSplit, epocas, batch_size)
            models.append(model)
            test_loss, test_acc = evaluarModelo(x_test, y_test, model)
            
            if test_acc > mejorPrecision:
                mejorPrecision = test_acc
                model.save("modelos/2D.h5")
            
            # Acumulamos los tiempos y los resultados de cada repetición
            total_time += elapsed_time
            train_acc_list.append(history.history['accuracy'][-1])
            train_loss_list.append(history.history['loss'][-1])
            val_acc_list.append(history.history['val_accuracy'][-1])
            val_loss_list.append(history.history['val_loss'][-1])
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            
        #Para cada valor de épocas, genero la matriz de confusión para el mejor
        for i in range(repeticiones):
            if test_acc_list[i] == max(test_acc_list):
                mostrarMatrizConfusion(models[i], x_test, y_test, f"pruebas/pruebas2D/confusion_funcion_{funcion}.png")
        
        # Cálculo de los promedios
        avg_time = total_time / repeticiones
        avg_train_acc = np.mean(train_acc_list)
        avg_train_loss = np.mean(train_loss_list)
        avg_val_acc = np.mean(val_acc_list)
        avg_val_loss = np.mean(val_loss_list)
        avg_test_acc = np.mean(test_acc_list)
        avg_test_loss = np.mean(test_loss_list)
        
        # Resultados en archivo
        try:
            with open(f"pruebas/pruebas2D/funcion_{funcion}.txt", "w") as archivo:
                archivo.write(f"Funcion: {funcion}\n")
                archivo.write(f"Promedio de tiempo de entrenamiento: {avg_time:.2f} segundos\n")
                archivo.write(f"Promedio de precisión en entrenamiento: {avg_train_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en entrenamiento: {avg_train_loss:.4f}\n")
                archivo.write(f"Promedio de precisión en validación: {avg_val_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en validación: {avg_val_loss:.4f}\n")
                archivo.write(f"Promedio de precisión en pruebas: {avg_test_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en pruebas: {avg_test_loss:.4f}\n")
        except FileNotFoundError:
            print("No se pudo guardar el archivo")
            
    plotModel("2D")


#Aquí optimizamos el número de neuronas en la capa oculta  
def tarea_2E():      
    #Cargamos y normalizamos el dataset
    x_train, y_train, x_test, y_test = cargarNormalizarDatos()
    
    #Parámetros del modelo y entrenamiento
    optimizador = 'adam'
    neuronasSalida = 10
    activacionSalida = 'softmax'
    validationSplit = 0.1
    epocas = 100  # Valor de la tarea 2B
    repeticiones = 5 
    batch_size = 256 #Valor de la tarea 2C
    funcion = 'sigmoid' #Valor de la tarea 2D
    
    neuronas = [16, 32, 64, 128, 256, 512]
    
    mejorPrecision = 0
    
    for neurona in neuronas:
        total_time = 0
        train_acc_list, train_loss_list, val_acc_list, val_loss_list, test_acc_list, test_loss_list, models = [], [], [], [], [], [], []
        
        for _ in range(repeticiones):
            #Creamos el modelo
            model = crearModeloSesion1(neurona, funcion, optimizador, neuronasSalida, activacionSalida)

            history, elapsed_time = entrenarModelo(model, x_train, y_train, validationSplit, epocas, batch_size)
            models.append(model)
            test_loss, test_acc = evaluarModelo(x_test, y_test, model)
            
            if test_acc > mejorPrecision:
                mejorPrecision = test_acc
                model.save("modelos/2E.h5")
            
            # Acumulamos los tiempos y los resultados de cada repetición
            total_time += elapsed_time
            train_acc_list.append(history.history['accuracy'][-1])
            train_loss_list.append(history.history['loss'][-1])
            val_acc_list.append(history.history['val_accuracy'][-1])
            val_loss_list.append(history.history['val_loss'][-1])
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            
        #Para cada valor de épocas, genero la matriz de confusión para el mejor
        for i in range(repeticiones):
            if test_acc_list[i] == max(test_acc_list):
                mostrarMatrizConfusion(models[i], x_test, y_test, f"pruebas/pruebas2E/confusion_neuronas_{neurona}.png")
        
        # Cálculo de los promedios
        avg_time = total_time / repeticiones
        avg_train_acc = np.mean(train_acc_list)
        avg_train_loss = np.mean(train_loss_list)
        avg_val_acc = np.mean(val_acc_list)
        avg_val_loss = np.mean(val_loss_list)
        avg_test_acc = np.mean(test_acc_list)
        avg_test_loss = np.mean(test_loss_list)
        
        # Resultados en archivo
        try:
            with open(f"pruebas/pruebas2E/neurona_{neurona}.txt", "w") as archivo:
                archivo.write(f"Neurona: {neurona}\n")
                archivo.write(f"Promedio de tiempo de entrenamiento: {avg_time:.2f} segundos\n")
                archivo.write(f"Promedio de precisión en entrenamiento: {avg_train_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en entrenamiento: {avg_train_loss:.4f}\n")
                archivo.write(f"Promedio de precisión en validación: {avg_val_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en validación: {avg_val_loss:.4f}\n")
                archivo.write(f"Promedio de precisión en pruebas: {avg_test_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en pruebas: {avg_test_loss:.4f}\n")
        except FileNotFoundError:
            print("No se pudo guardar el archivo")
            
    plotModel("2E")
    
def generarCombinaciones(batch, funcion, neuronas):
    combinaciones = []
    for capas in range(1, 4):
        for config in itertools.product(itertools.product(neuronas, funcion), repeat=capas):
            combinaciones.append(config)
            
    for combinacion in combinaciones:
        print(combinacion)
        
    print(len(combinaciones))
    return combinaciones

#Aquí añadimos una, dos y tres capas ocultas, optimizando todas las variables anteriores de nuevo        
def tarea_2F():
    #Cargamos y normalizamos el dataset
    x_train, y_train, x_test, y_test = cargarNormalizarDatos()
    
    #Parámetros del modelo y entrenamiento
    optimizador = 'adam'
    neuronasSalida = 10
    activacionSalida = 'softmax'
    validationSplit = 0.1
    repeticiones = 5 
    
    batch_sizes = [16, 32, 64, 128, 256, 512]
    funciones = ['sigmoid', 'relu', 'tanh']
    neuronas = [16, 32, 64, 128, 256, 512]
    max_epocas = 100 #Para el EarlyStopping
    
    combinaciones = generarCombinaciones(batch_sizes, funciones, neuronas)
    
    #Cogeremos 10 combinaciones aleatorias para ver cuales son las mejores
    combinaciones = 10

#Creamos una función genérica para crear modelos CNN con o sin pooling
def createModelCNN(pooling, kernel):
    model = Sequential([Conv2D(16, (kernel, kernel), activation='relu', input_shape=(32, 32, 3))])
    
    if pooling:
        model.add(MaxPooling2D((2, 2)))
        
    model.add(Conv2D(32, (kernel, kernel), activation='relu'))
    
    if pooling:
        model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    
    return model

#En esta tarea vamos a crear redes convolucionales  
def tareaG():
    #Cargamos y normalizamos el dataset para CNN
    x_train, y_train, x_test, y_test = cargarDatosCNN()
    
    #Creamos un callback para evitar ajustar las épocas
    early_stopping = EarlyStopping(
        monitor='val_loss',    # Métrica a monitorear
        mode='min',            # 'min' porque buscamos minimizar la pérdida
        patience=20,            # Número de épocas de paciencia
        restore_best_weights=True)  # Restaura los mejores pesos encontrados
    
    repeticiones = 5
    
    #Entrenamos los modelos varias veces, sacamos las medias y las guardamos en un archivo
    train_acc_list, train_loss_list, val_acc_list, val_loss_list, test_acc_list, test_loss_list, models = [], [], [], [], [], [], []
    for _ in range(repeticiones):
        totalTime = 0
        #Modelo sin pooling
        model = createModelCNN(False, 3)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        #Entrenamos el modelo
        start = time.time()
        history = model.fit(x_train, y_train, validation_split=0.1, epochs=60, batch_size=256, callbacks=[early_stopping])
        end = time.time()
        
        models.append(model)
        
        #Evaluamos el modelo
        test_loss, test_acc = model.evaluate(x_test, y_test)
        
        #Guardamos los resultados
        totalTime += end - start
        train_acc_list.append(history.history['accuracy'][-1])
        train_loss_list.append(history.history['loss'][-1])
        val_acc_list.append(history.history['val_accuracy'][-1])
        val_loss_list.append(history.history['val_loss'][-1])
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        
    #Sacamos las medias y escribimos en un archivo
    avg_time = totalTime / repeticiones
    avg_train_acc = np.mean(train_acc_list)
    avg_train_loss = np.mean(train_loss_list)
    avg_val_acc = np.mean(val_acc_list)
    avg_val_loss = np.mean(val_loss_list)
    avg_test_acc = np.mean(test_acc_list)
    avg_test_loss = np.mean(test_loss_list)
    
    try:
        with open("pruebas/pruebasG/sin_pooling.txt", "w") as archivo:
            archivo.write(f"Promedio de tiempo de entrenamiento: {avg_time:.2f} segundos\n")
            archivo.write(f"Promedio de precisión en entrenamiento: {avg_train_acc:.4f}\n")
            archivo.write(f"Promedio de pérdida en entrenamiento: {avg_train_loss:.4f}\n")
            archivo.write(f"Promedio de precisión en validación: {avg_val_acc:.4f}\n")
            archivo.write(f"Promedio de pérdida en validación: {avg_val_loss:.4f}\n")
            archivo.write(f"Promedio de precisión en pruebas: {avg_test_acc:.4f}\n")
            archivo.write(f"Promedio de pérdida en pruebas: {avg_test_loss:.4f}\n")
    except FileNotFoundError:
        print("No se pudo guardar el archivo")
        
    #Generamos la matriz de confusión del mejor modelo
    for i in range(repeticiones):
        if test_acc_list[i] == max(test_acc_list):
            mostrarMatrizConfusion(models[i], x_test, y_test, "pruebas/pruebasG/confusion_sin_pooling.png")
            

    #Modelo con pooling
    train_acc_list, train_loss_list, val_acc_list, val_loss_list, test_acc_list, test_loss_list, models = [], [], [], [], [], [], []
    
    for _ in range(repeticiones):
        totalTime = 0
        #Modelo sin pooling
        model = createModelCNN(True, 3)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        #Entrenamos el modelo
        start = time.time()
        history = model.fit(x_train, y_train, validation_split=0.1, epochs=60, batch_size=256, callbacks=[early_stopping])
        end = time.time()
        
        models.append(model)
        
        #Evaluamos el modelo
        test_loss, test_acc = model.evaluate(x_test, y_test)
        
        #Guardamos los resultados
        totalTime += end - start
        train_acc_list.append(history.history['accuracy'][-1])
        train_loss_list.append(history.history['loss'][-1])
        val_acc_list.append(history.history['val_accuracy'][-1])
        val_loss_list.append(history.history['val_loss'][-1])
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
    
    #Sacamos las medias y escribimos en un archivo
    avg_time = totalTime / repeticiones
    avg_train_acc = np.mean(train_acc_list)
    avg_train_loss = np.mean(train_loss_list)
    avg_val_acc = np.mean(val_acc_list)
    avg_val_loss = np.mean(val_loss_list)
    avg_test_acc = np.mean(test_acc_list)
    avg_test_loss = np.mean(test_loss_list)
    
    try:
        with open("pruebas/pruebasG/con_pooling.txt", "w") as archivo:
            archivo.write(f"Promedio de tiempo de entrenamiento: {avg_time:.2f} segundos\n")
            archivo.write(f"Promedio de precisión en entrenamiento: {avg_train_acc:.4f}\n")
            archivo.write(f"Promedio de pérdida en entrenamiento: {avg_train_loss:.4f}\n")
            archivo.write(f"Promedio de precisión en validación: {avg_val_acc:.4f}\n")
            archivo.write(f"Promedio de pérdida en validación: {avg_val_loss:.4f}\n")
            archivo.write(f"Promedio de precisión en pruebas: {avg_test_acc:.4f}\n")
            archivo.write(f"Promedio de pérdida en pruebas: {avg_test_loss:.4f}\n")
    except FileNotFoundError:
        print("No se pudo guardar el archivo")
    
    #Generamos la matriz de confusión del mejor modelo
    for i in range(repeticiones):
        if test_acc_list[i] == max(test_acc_list):
            mostrarMatrizConfusion(models[i], x_test, y_test, "pruebas/pruebasG/confusion_con_pooling.png")

#En esta tarea vamos a ajustar el kernel_size de la CNN
def tareaH():
    #Cargamos y normalizamos el dataset para CNN
    x_train, y_train, x_test, y_test = cargarDatosCNN()
    
    #Creamos un callback para evitar ajustar las épocas
    early_stopping = EarlyStopping(
        monitor='val_loss',    # Métrica a monitorear
        mode='min',            # 'min' porque buscamos minimizar la pérdida
        patience=20,            # Número de épocas de paciencia
        restore_best_weights=True)  # Restaura los mejores pesos encontrados
    
    repeticiones = 5
    
    kernels = [3, 4, 5]
    mejorPrecision = 0
    
    for kernel in kernels:
        #Entrenamos los modelos varias veces, sacamos las medias y las guardamos en un archivo  
        train_acc_list, train_loss_list, val_acc_list, val_loss_list, test_acc_list, test_loss_list= [], [], [], [], [], []
        for _ in range(repeticiones):
            totalTime = 0
            #Modelo sin pooling
            model = createModelCNN(False, kernel)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            #Entrenamos el modelo
            start = time.time()
            history = model.fit(x_train, y_train, validation_split=0.1, epochs=60, batch_size=256, callbacks=[early_stopping])
            end = time.time()
            
            #Evaluamos el modelo
            test_loss, test_acc = model.evaluate(x_test, y_test)
            
            #Guardamos los resultados
            totalTime += end - start
            train_acc_list.append(history.history['accuracy'][-1])
            train_loss_list.append(history.history['loss'][-1])
            val_acc_list.append(history.history['val_accuracy'][-1])
            val_loss_list.append(history.history['val_loss'][-1])
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            
            if test_acc > mejorPrecision:
                mejorPrecision = test_acc
                model.save(f"modelos/H.h5")
            
        #Sacamos las medias y escribimos en un archivo
        avg_time = totalTime / repeticiones
        avg_train_acc = np.mean(train_acc_list)
        avg_train_loss = np.mean(train_loss_list)
        avg_val_acc = np.mean(val_acc_list)
        avg_val_loss = np.mean(val_loss_list)
        avg_test_acc = np.mean(test_acc_list)
        avg_test_loss = np.mean(test_loss_list)
        
        #Escribimos en un archivo
        try:
            with open(f"pruebas/pruebasH1/kernel_{kernel}.txt", "w") as archivo:
                archivo.write(f"Kernel: {kernel}x{kernel}\n")
                archivo.write(f"Promedio de tiempo de entrenamiento: {avg_time:.2f} segundos\n")
                archivo.write(f"Promedio de precisión en entrenamiento: {avg_train_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en entrenamiento: {avg_train_loss:.4f}\n")
                archivo.write(f"Promedio de precisión en validación: {avg_val_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en validación: {avg_val_loss:.4f}\n")
                archivo.write(f"Promedio de precisión en pruebas: {avg_test_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en pruebas: {avg_test_loss:.4f}\n")
        except FileNotFoundError:
            print("No se pudo guardar el archivo")
            
        train_acc_list, train_loss_list, val_acc_list, val_loss_list, test_acc_list, test_loss_list= [], [], [], [], [], []
        for _ in range(repeticiones):
            totalTime = 0
            #Modelo sin pooling
            model = createModelCNN(True, kernel)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            #Entrenamos el modelo
            start = time.time()
            history = model.fit(x_train, y_train, validation_split=0.1, epochs=60, batch_size=256, callbacks=[early_stopping])
            end = time.time()
            
            #Evaluamos el modelo
            test_loss, test_acc = model.evaluate(x_test, y_test)
            
            #Guardamos los resultados
            totalTime += end - start
            train_acc_list.append(history.history['accuracy'][-1])
            train_loss_list.append(history.history['loss'][-1])
            val_acc_list.append(history.history['val_accuracy'][-1])
            val_loss_list.append(history.history['val_loss'][-1])
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            
            if test_acc > mejorPrecision:
                mejorPrecision = test_acc
                model.save(f"modelos/H.h5")
            
        #Sacamos las medias y escribimos en un archivo
        avg_time = totalTime / repeticiones
        avg_train_acc = np.mean(train_acc_list)
        avg_train_loss = np.mean(train_loss_list)
        avg_val_acc = np.mean(val_acc_list)
        avg_val_loss = np.mean(val_loss_list)
        avg_test_acc = np.mean(test_acc_list)
        avg_test_loss = np.mean(test_loss_list)
        
        #Escribimos en un archivo
        try:
            with open(f"pruebas/pruebasH2/kernel_{kernel}.txt", "w") as archivo:
                archivo.write(f"Kernel: {kernel}x{kernel}\n")
                archivo.write(f"Promedio de tiempo de entrenamiento: {avg_time:.2f} segundos\n")
                archivo.write(f"Promedio de precisión en entrenamiento: {avg_train_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en entrenamiento: {avg_train_loss:.4f}\n")
                archivo.write(f"Promedio de precisión en validación: {avg_val_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en validación: {avg_val_loss:.4f}\n")
                archivo.write(f"Promedio de precisión en pruebas: {avg_test_acc:.4f}\n")
                archivo.write(f"Promedio de pérdida en pruebas: {avg_test_loss:.4f}\n")
        except FileNotFoundError:
            print("No se pudo guardar el archivo")
            
    #Generamos la matriz de confusión del mejor modelo
    model = tf.keras.models.load_model("modelos/H.h5")
    mostrarMatrizConfusion(model, x_test, y_test, "pruebas/pruebasG/confusion.png")

#En esta tarea, trataremos de encontrar el mejor modelo convolucional
#añadiendo diferentes capas con varios números de neuronas y filtros
#TODO: RECOMENDABLE MIRAR OTROS ESTUDIOS PREVIOS PARA EMPEZAR POR AHÍ
def tareaI():
    #Cargamos los datasets
    x_train, y_train, x_test, y_test = cargarDatosCNN()
    
    #Creamos un callback para evitar ajustar las épocas
    early_stopping = EarlyStopping(
        monitor='val_loss',    # Métrica a monitorear
        mode='min',            # 'min' porque buscamos minimizar la pérdida
        patience=25,            # Número de épocas de paciencia
        restore_best_weights=True)  # Restaura los mejores pesos encontrados
    
    mejorPrecision = 0  
    
    model = tf.keras.models.Sequential([
        #Capa de entrada
        tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=x_train.shape[1:], activation='relu'),
        
        #Capas ocultas
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        #Capa de salida
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(x_train, y_train, validation_split=0.1, epochs=150, batch_size=256, callbacks=[early_stopping])
    test_loss, test_acc = model.evaluate(x_test, y_test)

def main():
    #tarea_2A()
    #tarea_2B(98, 102)
    #tarea_2C()
    #tarea_2D()
    #tarea_2E()
    #tarea_2F()
    #tareaG()
    #tareaH()
    tareaI()
    
    pass

if __name__ == "__main__":
    main()
