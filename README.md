# Reconocimiento_Numeros_ML
Ejemplo práctico: reconocimiento básico de números escritos con Scikit-learn y OpenCV

# Reconocimiento de Números Escritos con Scikit-learn y OpenCV 🔢🧠

Este repositorio contiene una implementación avanzada de **Percepción Computacional** que integra **Visión Artificial** con **Machine Learning**. El proyecto se enfoca en el entrenamiento de un modelo predictivo para identificar dígitos manuscritos y agruparlos en números completos a partir de una imagen.

## 💻 Especificaciones del Entorno de Desarrollo
Para asegurar la reproducibilidad del script `reconocedor_numeros_sklearn_opencv.py`, se detallan las especificaciones exactas del sistema utilizado:

* **Sistema Operativo:** Windows 11 Home (Versión 25H2) 🪟
* **Lenguaje:** Python 3.14.3 (64-bit) 🐍
* **IDE:** Visual Studio Code
* **Librerías:** OpenCV (`opencv-python`), Scikit-learn, NumPy, Matplotlib

## 📁 Estructura del Repositorio
* `reconocedor_numeros_sklearn_opencv.py`: Pipeline completo de entrenamiento, segmentación y clasificación.
* `imagen_ejemplo_2.jpg`: Imagen fuente con dígitos manuscritos para las pruebas de reconocimiento.

## 🚀 Metodología de Procesamiento
El flujo de trabajo implementado en el código sigue estos pasos críticos:

1. **Entrenamiento del Modelo:** Uso del dataset *Digits* y un clasificador **SVC (Support Vector Classification)** con escalado estándar de datos para maximizar la precisión.
2. **Segmentación Morfológica:** Aplicación de filtros y detección de componentes para aislar cada trazo independiente (ROI) sobre el fondo claro.
3. **Normalización de ROI:** Redimensionamiento de cada componente detectado a una matriz de 8x8 píxeles para asegurar la compatibilidad con los datos de entrenamiento.
4. **Predicción y Agrupación:** Clasificación individual de dígitos y uso de lógica de proximidad para reconstruir números de múltiples cifras.



## 🛠️ Instalación y Uso
1. **Instalar dependencias:**
   ```powershell
   pip install opencv-python scikit-learn numpy matplotlib
