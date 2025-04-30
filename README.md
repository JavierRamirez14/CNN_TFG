# Funcionamiento y aplicación de redes neuronales para la clasificación de imágenes

![License](https://img.shields.io/badge/license-MIT-blue)
![Python Version](https://img.shields.io/badge/python-3.11%2B-brightgreen)

Este repositorio contiene el desarrollo completo de una red neuronal convolucional (CNN) **implementada desde cero**, sin usar librerías especializadas en deep learning como TensorFlow o PyTorch. Todo ha sido construido utilizando únicamente librerías de bajo nivel como `numpy`, `scipy` con el objetivo de demostrar el funcionamiento interno de las CNNs.

Soy Javier Ramírez, estudiante del Grado en Gestión de la Información y Contenidos Digitales de la Universidad Carlos III de Madrid. Este proyecto forma parte de mi Trabajo de Fin de Grado (TFG) y tiene como objetivo aplicar deep learning para la **detección de tumores cerebrales** en imágenes de resonancia magnética (MRI), utilizando el dataset ["Brain Tumor MRI Dataset"](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) de Kaggle.

---

## Tabla de Contenidos

- [1. Descripción del Proyecto](#1-descripción-del-proyecto)
- [2. Trabajo Académico (TFG)](#2-trabajo-académico-tfg)
- [3. Estructura del Código](#3-estructura-del-código)
- [4. Resultados](#4-resultados)
- [5. Comparativa con otros modelos](#5-comparativa-con-otros-modelos)
- [6. Ejecución del Proyecto](#6-ejecución-del-proyecto)
  - [6.1 Entorno local (CPU)](#61-entorno-local-cpu)
  - [6.2 Entorno con GPU (Kaggle / Google Colab)](#62-entorno-con-gpu-kaggle--google-colab)
- [7. Licencia](#7-licencia)

---

## 1. Descripción del Proyecto

El objetivo principal de este proyecto es **entender, implementar y optimizar una red neuronal convolucional desde cero**, sin apoyarse en frameworks de alto nivel. Para demostrar su funcionamiento, he aplicado la red a un problema real, como es la **detección automática de tumores cerebrales en imágenes médicas**.

Este trabajo incluye:
- El desarrollo de la CNN, implementando manualmente algoritmos como el descenso del gradiente, backpropagation, convoluciones, pooling, funciones de activación, funciones de pérdida, etc.
- La aplicación del modelo desarrollado para la clasificación de tumores cerebrales.
- Una comparación entre la CNN desarrollada con un modelo idéntico (misma arquitectura e hiperparámetros) en TensorFlow y una red preentrenada (ResNet-50).

---

## 2. Trabajo Académico (TFG)

Este repositorio está vinculado a mi Trabajo de Fin de Grado (TFG), actualmente en desarrollo. El documento final se publicará en formato PDF e incluirá una **explicación detallada del funcionamiento de cada uno de los algoritmos implementados**, desde lo más básico como la propagación hacia adelante y la multiplicación de matrices hasta cosas un poco más complejas como el algoritmo de backpropagation y las capas convolucionales. Todos estos conceptos se explican de una forma clara y sencilla, incluyendo ejemplos, con el objetivo de explorar y comprender a fondo el funcionamiento interno de las redes neuronales. Será una guía útil tanto para estudiantes que no conocen mucho sobre la IA como para cualquier persona interesada en aprender cómo construir una red neuronal convolucional desde cero.

---

## 3. Estructura del Código

El código está dividido en módulos (archivos .py), donde cada uno contienene la implementación de un algoritmo o concepto distinto, con el objetivo de mantener una estructura limpia y reutilizable y favorecer la comprensión:

- `capa.py`: Estructura general de una capa.
- `dense.py`: Capa densa (fully connected layer).
- `activations.py`: Funciones de activación (ReLU, Softmax).
- `loss.py`: Funciones de pérdida y sus derivadas para el backpropagation (Categorical Cross Entropy, Accuracy).
- `convolution.py`: Capa de convolución sin optimizar para favorecer la comprensión.
- `pooling.py`: Capa de MaxPooling sin optimizar para favorecer la comprensión.
- `fast_convolution.py`: Capa de convolución optimizada para un mayor rendimiento.
- `fast_pooling.py`: Capa de MaxPooling optimizada para un mayor rendimiento.
- `reshape.py`: Capa para modificar la forma de matrices (Aplanar).
- `network.py`: Funciones de entrenamiento, validación y prueba.
- `ejecucion_cpu.ipynb`: Prueba de la CNN desarrollada sobre MNIST y el dataset de Kaggle. El entrenamiento de los modelos no está completo ya que es una prueba que se ejecuta en local por lo que no se puede usar GPU.
- `ejecucion_gpu.ipynb`: Entrenamiento y comparación de los 3 modelos usando un entorno virtual con GPU.
- `requirements.txt`: Archivo con las librerias y versiones necesarias para poder ejecutar el código.

---

## 4. Resultados

La CNN desarrollada se entrenó con los datos de train del dataset *Bran Tumor MRI Dataset* de Kaggle y se evaluó su rendimiento con el conjunto de test, presentando los siguientes resultados:

> **Nota:** Puedes consultar todos los resultados, métricas y gráficos de forma más detallada en el [notebook de Kaggle](https://www.kaggle.com/code/jramirez14/funcionamiento-y-aplicaci-n-de-redes-neuronales).

- **Accuracy: 95.27%**
- **AUC ROC promedio: 0.99**

![Matriz de Confusión](https://github.com/JavierRamirez14/CNN_TFG/blob/main/resultados/matriz_confusion.png)

![Reporte de Clasificación](https://github.com/JavierRamirez14/CNN_TFG/blob/main/resultados/reporte_clasificacion.png)

---

## 5. Comparativa con otros modelos

El modelo desarrollado desde cero se comparó con los siguientes modelos:

- Un modelo idéntico (misma arquitectura e hiperparámetros) implementado en Tensorflow.
- Un modelo preentrenado (Resnet-50) haciendo fine-tuning.

Los resultados fueron los siguientes:

![Comparación de Métricas Globales entre Modelos](https://github.com/JavierRamirez14/CNN_TFG/blob/main/resultados/comparacion_metricas_modelos.png)

## 6. Ejecución del Proyecto

### Entorno local (CPU)

Para probar la CNN en tu entorno local, sigue los siguientes pasos::

1. Clona el repositorio:
   ```bash
   git clone https://github.com/JavierRamirez14/CNN_TFG.git
   cd CNN_TFG

2. Instala las dependencias:

    Recomiendo instalarlas en un entorno virtual. Para crearlo:
    ```bash
    python -m venv venv
    ```
    Para activarlo en Windows:
    ```bash
    venv\Scripts\activate
    ```
    Para activarlo en Linux o Mac:
    ```bash
    source venv/bin/activate
    ```
    Ahora, con el entorno activado, se instalan las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
3. Ejecuta el archivo `ejecucion_cpu.ipynb`

### 6.2 Entorno con GPU (Kaggle / Google Colab)

Para ejecutar el modelo aprovechando la aceleración por GPU, existen dos alternativas:

#### Opción 1: Kaggle

Puedes ejecutar el proyecto directamente desde un entorno virtual de Kaggle, que ya incluye el dataset como input del entorno, por lo que **no es necesario descargar ni subir los datos manualmente**. Simplemente accede al siguiente enlace, haz clic en editar y ejecuta el código:

[Notebook en Kaggle: Funcionamiento y aplicación de redes neuronales](https://www.kaggle.com/code/jramirez14/funcionamiento-y-aplicaci-n-de-redes-neuronales)

Kaggle ofrece **30 horas gratuitas de GPU por semana**, lo cual permite entrenar completamente los modelos y visualizar todos los resultados y comentarios generados.

#### Opción 2: Google Colab

También puedes utilizar Google Colab para ejecutar el proyecto mediante el archivo [`ejecucion_gpu.ipynb`](ejecucion_gpu.ipynb). Para ello, debes subir tu archivo `kaggle.json` para autenticarte y descargar el dataset desde la API de Kaggle. Todo el procedimiento está explicado y automatizado en el notebook correspondiente.

> ⚠️ **Nota:** Debido a la **falta de acceso a recursos de GPU en Google Colab** durante el desarrollo, **no fue posible ejecutar el entrenamiento completo ni generar los resultados y comentarios finales** en esta plataforma. Se recomienda utilizar **Kaggle** si se desea consultar las métricas completas y comparativas finales del proyecto.

> ⚠️ **Importante:** Asegúrate de que el entorno virtual (ya sea Kaggle o Colab) está configurado para usar **GPU**. De lo contrario, el entrenamiento será extremadamente lento.   

---

## Licencia

This project is licensed under the [MIT License](LICENSE).
