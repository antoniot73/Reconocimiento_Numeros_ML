"""
Aplicación sencilla para reconocer números escritos en una imagen
usando Scikit-learn para clasificación y OpenCV para segmentación.

Requisitos:
    py -m pip install scikit-learn opencv-python numpy matplotlib

Ejecución:
    py reconocedor_numeros_sklearn_opencv.py

Descripción:
    1. Entrena un clasificador SVM con el dataset Digits de Scikit-learn.
    2. Usa OpenCV para detectar componentes oscuros sobre fondo claro.
    3. Filtra componentes con tamaño compatible con dígitos.
    4. Convierte cada componente a un formato 8x8 similar al dataset de entrenamiento.
    5. Clasifica cada dígito y agrupa dígitos cercanos para formar números.
    6. Guarda imágenes del pipeline en la carpeta "salidas2".
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class DigitoDetectado:
    """
    Estructura para almacenar la información de un dígito detectado.
    """
    x: int
    y: int
    w: int
    h: int
    etiqueta: int
    confianza: float


@dataclass
class NumeroDetectado:
    """
    Estructura para almacenar la información de un número compuesto.
    """
    texto: str
    x: int
    y: int
    w: int
    h: int
    fila: int


def configurar_logger() -> None:
    """
    Configura el sistema de bitácora del programa.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def entrenar_modelo(random_state: int = 42) -> Pipeline:
    """
    Entrena un clasificador SVM usando el dataset Digits de Scikit-learn.

    Args:
        random_state: Semilla para reproducibilidad.

    Returns:
        Pipeline entrenado.
    """
    datos = load_digits()
    X = datos.data
    y = datos.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    modelo = Pipeline([
        ("escalado", StandardScaler()),
        ("clasificador", SVC(kernel="rbf", C=5.0, gamma="scale", random_state=random_state))
    ])

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    exactitud = accuracy_score(y_test, y_pred)

    logging.info("Modelo entrenado con el dataset Digits")
    logging.info("Exactitud de validación interna: %.4f", exactitud)

    return modelo


def cargar_imagen(ruta_imagen: str) -> np.ndarray:
    """
    Carga una imagen desde disco.

    Args:
        ruta_imagen: Ruta de la imagen de entrada.

    Returns:
        Imagen cargada en formato BGR.

    Raises:
        FileNotFoundError: Si la imagen no existe.
        ValueError: Si la imagen no puede abrirse.
    """
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No existe la imagen: {ruta_imagen}")

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        raise ValueError(f"No fue posible abrir la imagen: {ruta_imagen}")

    logging.info("Imagen cargada correctamente: %s", ruta_imagen)
    logging.info(
        "Dimensiones: alto=%d, ancho=%d, canales=%d",
        imagen.shape[0],
        imagen.shape[1],
        imagen.shape[2],
    )

    return imagen


def preprocesar_imagen(
    imagen_bgr: np.ndarray,
    umbral_binario: int = 120
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte la imagen a escala de grises y genera una binaria invertida.

    Args:
        imagen_bgr: Imagen de entrada en BGR.
        umbral_binario: Umbral fijo para separar tinta oscura de fondo claro.

    Returns:
        Tupla con:
        - imagen en escala de grises
        - imagen binaria invertida
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, umbral_binario, 255, cv2.THRESH_BINARY_INV)

    logging.info("Preprocesamiento completado: escala de grises + binarización invertida")
    return gris, binaria


def detectar_componentes_candidatos(
    imagen_binaria: np.ndarray,
    margen_borde: int = 120,
    area_minima: int = 2200,
    area_maxima: int = 9000,
    ancho_minimo: int = 50,
    ancho_maximo: int = 180,
    alto_minimo: int = 100,
    alto_maximo: int = 220
) -> List[Tuple[int, int, int, int, int]]:
    """
    Detecta componentes conectados compatibles con dígitos.

    Args:
        imagen_binaria: Imagen binaria invertida.
        margen_borde: Margen para descartar objetos pegados a los bordes.
        area_minima: Área mínima del componente.
        area_maxima: Área máxima del componente.
        ancho_minimo: Ancho mínimo permitido.
        ancho_maximo: Ancho máximo permitido.
        alto_minimo: Alto mínimo permitido.
        alto_maximo: Alto máximo permitido.

    Returns:
        Lista de tuplas (x, y, w, h, area).
    """
    alto, ancho = imagen_binaria.shape
    numero_etiquetas, _, estadisticas, _ = cv2.connectedComponentsWithStats(
        imagen_binaria,
        8
    )

    candidatos: List[Tuple[int, int, int, int, int]] = []

    for indice in range(1, numero_etiquetas):
        x, y, w, h, area = estadisticas[indice]

        cumple_tamano = (
            ancho_minimo <= w <= ancho_maximo and
            alto_minimo <= h <= alto_maximo and
            area_minima <= area <= area_maxima
        )

        lejos_del_borde = (
            x > margen_borde and
            y > margen_borde - 20 and
            (x + w) < (ancho - margen_borde) and
            (y + h) < (alto - 50)
        )

        if cumple_tamano and lejos_del_borde:
            candidatos.append((int(x), int(y), int(w), int(h), int(area)))

    logging.info("Componentes candidatos detectados: %d", len(candidatos))
    return candidatos


def dibujar_candidatos(
    imagen_bgr: np.ndarray,
    candidatos: List[Tuple[int, int, int, int, int]]
) -> np.ndarray:
    """
    Dibuja las cajas de los componentes candidatos.

    Args:
        imagen_bgr: Imagen original.
        candidatos: Lista de componentes candidatos.

    Returns:
        Imagen con cajas de candidatos.
    """
    imagen_candidatos = imagen_bgr.copy()

    for indice, (x, y, w, h, area) in enumerate(candidatos, start=1):
        cv2.rectangle(imagen_candidatos, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(
            imagen_candidatos,
            f"C{indice} A={area}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    return imagen_candidatos


def convertir_componente_a_8x8(
    imagen_gris: np.ndarray,
    bbox: Tuple[int, int, int, int, int],
    tam_salida: int = 8,
    padding: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte un componente detectado en una representación 8x8 compatible
    con el estilo del dataset Digits.

    Args:
        imagen_gris: Imagen en escala de grises.
        bbox: Tupla (x, y, w, h, area).
        tam_salida: Tamaño de salida.
        padding: Relleno alrededor del componente.

    Returns:
        Tupla con:
        - vector de características en rango 0-16
        - imagen reducida 8x8 para depuración
    """
    x, y, w, h, _ = bbox
    roi = imagen_gris[y:y + h, x:x + w]

    _, roi_binaria = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY_INV)

    lado = max(w, h) + 2 * padding
    lienzo = np.zeros((lado, lado), dtype=np.uint8)

    offset_x = (lado - w) // 2
    offset_y = (lado - h) // 2
    lienzo[offset_y:offset_y + h, offset_x:offset_x + w] = roi_binaria

    reducida = cv2.resize(lienzo, (tam_salida, tam_salida), interpolation=cv2.INTER_AREA)
    caracteristicas = (reducida.astype(np.float32) / 255.0 * 16.0).reshape(1, -1)

    return caracteristicas, reducida


def clasificar_digitos(
    modelo: Pipeline,
    imagen_gris: np.ndarray,
    candidatos: List[Tuple[int, int, int, int, int]]
) -> List[DigitoDetectado]:
    """
    Clasifica los componentes detectados como dígitos.

    Args:
        modelo: Modelo entrenado de Scikit-learn.
        imagen_gris: Imagen en escala de grises.
        candidatos: Lista de componentes candidatos.

    Returns:
        Lista de dígitos clasificados.
    """
    digitos: List[DigitoDetectado] = []

    for bbox in candidatos:
        caracteristicas, _ = convertir_componente_a_8x8(imagen_gris, bbox)
        etiqueta = int(modelo.predict(caracteristicas)[0])

        decision = modelo.decision_function(caracteristicas)
        confianza = float(np.max(decision))

        x, y, w, h, _ = bbox
        digitos.append(
            DigitoDetectado(
                x=x,
                y=y,
                w=w,
                h=h,
                etiqueta=etiqueta,
                confianza=confianza,
            )
        )

    digitos.sort(key=lambda item: (item.y, item.x))
    logging.info("Dígitos clasificados: %d", len(digitos))
    return digitos


def agrupar_por_filas(
    digitos: List[DigitoDetectado],
    tolerancia_vertical: int = 80
) -> List[List[DigitoDetectado]]:
    """
    Agrupa dígitos según su cercanía vertical para formar filas.

    Args:
        digitos: Lista de dígitos detectados.
        tolerancia_vertical: Distancia máxima entre centros verticales para pertenecer a la misma fila.

    Returns:
        Lista de filas; cada fila es una lista de dígitos.
    """
    if not digitos:
        return []

    digitos_ordenados = sorted(
        digitos,
        key=lambda item: (item.y + item.h // 2, item.x)
    )
    filas: List[List[DigitoDetectado]] = []

    for digito in digitos_ordenados:
        centro_y = digito.y + digito.h // 2
        agregado = False

        for fila in filas:
            centros = [elemento.y + elemento.h // 2 for elemento in fila]
            centro_promedio = int(np.mean(centros))

            if abs(centro_y - centro_promedio) <= tolerancia_vertical:
                fila.append(digito)
                agregado = True
                break

        if not agregado:
            filas.append([digito])

    for fila in filas:
        fila.sort(key=lambda item: item.x)

    return filas


def construir_numero_desde_grupo(
    grupo: List[DigitoDetectado],
    fila: int
) -> NumeroDetectado:
    """
    Construye un objeto NumeroDetectado a partir de un grupo de dígitos.

    Args:
        grupo: Lista de dígitos pertenecientes al mismo número.
        fila: Índice de fila.

    Returns:
        Número detectado.
    """
    texto = "".join(str(digito.etiqueta) for digito in grupo)
    x_min = min(digito.x for digito in grupo)
    y_min = min(digito.y for digito in grupo)
    x_max = max(digito.x + digito.w for digito in grupo)
    y_max = max(digito.y + digito.h for digito in grupo)

    return NumeroDetectado(
        texto=texto,
        x=x_min,
        y=y_min,
        w=x_max - x_min,
        h=y_max - y_min,
        fila=fila
    )


def agrupar_digitos_en_numeros(
    digitos: List[DigitoDetectado],
    factor_espaciado: float = 1.5
) -> List[NumeroDetectado]:
    """
    Agrupa dígitos cercanos horizontalmente para formar números de varios dígitos.

    Args:
        digitos: Lista de dígitos detectados.
        factor_espaciado: Multiplicador del ancho promedio para decidir si un salto horizontal
                          inicia un nuevo número.

    Returns:
        Lista de números detectados.
    """
    filas = agrupar_por_filas(digitos)
    numeros: List[NumeroDetectado] = []

    for indice_fila, fila in enumerate(filas):
        if not fila:
            continue

        anchos = [digito.w for digito in fila]
        umbral_salto = int(np.mean(anchos) * factor_espaciado)

        grupo_actual: List[DigitoDetectado] = [fila[0]]

        for actual in fila[1:]:
            previo = grupo_actual[-1]
            separacion = actual.x - (previo.x + previo.w)

            if separacion > umbral_salto:
                numeros.append(construir_numero_desde_grupo(grupo_actual, indice_fila))
                grupo_actual = [actual]
            else:
                grupo_actual.append(actual)

        if grupo_actual:
            numeros.append(construir_numero_desde_grupo(grupo_actual, indice_fila))

    logging.info("Números agrupados: %d", len(numeros))
    return numeros


def anotar_digitos(
    imagen_bgr: np.ndarray,
    digitos: List[DigitoDetectado]
) -> np.ndarray:
    """
    Dibuja los dígitos individuales detectados.

    Args:
        imagen_bgr: Imagen original.
        digitos: Lista de dígitos detectados.

    Returns:
        Imagen anotada con dígitos.
    """
    anotada = imagen_bgr.copy()

    for digito in digitos:
        cv2.rectangle(
            anotada,
            (digito.x, digito.y),
            (digito.x + digito.w, digito.y + digito.h),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            anotada,
            str(digito.etiqueta),
            (digito.x, max(30, digito.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 140, 0),
            2,
            cv2.LINE_AA,
        )

    return anotada


def anotar_resultados(
    imagen_bgr: np.ndarray,
    digitos: List[DigitoDetectado],
    numeros: List[NumeroDetectado]
) -> np.ndarray:
    """
    Dibuja cajas y etiquetas sobre la imagen original.

    Args:
        imagen_bgr: Imagen original.
        digitos: Lista de dígitos detectados.
        numeros: Lista de números agrupados.

    Returns:
        Imagen anotada.
    """
    anotada = anotar_digitos(imagen_bgr, digitos)

    for numero in numeros:
        cv2.rectangle(
            anotada,
            (numero.x, numero.y),
            (numero.x + numero.w, numero.y + numero.h),
            (255, 0, 0),
            3,
        )
        cv2.putText(
            anotada,
            f"Num: {numero.texto}",
            (numero.x, numero.y + numero.h + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return anotada


def guardar_resultados(
    carpeta_salida: str,
    imagen_gris: np.ndarray,
    imagen_binaria: np.ndarray,
    imagen_candidatos: np.ndarray,
    imagen_digitos: np.ndarray,
    imagen_anotada: np.ndarray
) -> None:
    """
    Guarda imágenes de salida del pipeline.

    Args:
        carpeta_salida: Carpeta de salida.
        imagen_gris: Imagen en escala de grises.
        imagen_binaria: Imagen binaria intermedia.
        imagen_candidatos: Imagen con componentes candidatos.
        imagen_digitos: Imagen con dígitos individuales anotados.
        imagen_anotada: Imagen final anotada con números completos.
    """
    os.makedirs(carpeta_salida, exist_ok=True)

    archivos = {
        "01_gris.png": imagen_gris,
        "02_binaria.png": imagen_binaria,
        "03_candidatos.png": imagen_candidatos,
        "04_digitos.png": imagen_digitos,
        "05_anotada.png": imagen_anotada,
    }

    for nombre, imagen in archivos.items():
        ruta_salida = os.path.join(carpeta_salida, nombre)
        cv2.imwrite(ruta_salida, imagen)
        logging.info("Archivo guardado: %s", ruta_salida)


def mostrar_resultados(
    imagen_original: np.ndarray,
    imagen_binaria: np.ndarray,
    imagen_anotada: np.ndarray
) -> None:
    """
    Muestra resultados usando Matplotlib.

    Args:
        imagen_original: Imagen original en BGR.
        imagen_binaria: Imagen binaria intermedia.
        imagen_anotada: Imagen final anotada.
    """
    original_rgb = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)
    anotada_rgb = cv2.cvtColor(imagen_anotada, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Imagen original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(imagen_binaria, cmap="gray")
    plt.title("Binaria invertida")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(anotada_rgb)
    plt.title("Resultado anotado")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def imprimir_reporte(
    ruta_imagen: str,
    digitos: List[DigitoDetectado],
    numeros: List[NumeroDetectado],
    carpeta_salida: str
) -> None:
    """
    Imprime un reporte del proceso en consola.

    Args:
        ruta_imagen: Imagen analizada.
        digitos: Lista de dígitos detectados.
        numeros: Lista de números detectados.
        carpeta_salida: Carpeta donde se guardaron resultados.
    """
    print("\n--- REPORTE DEL PROCESO ---")
    print(f"Imagen analizada    : {ruta_imagen}")
    print(f"Dígitos detectados  : {len(digitos)}")
    print(f"Números detectados  : {len(numeros)}")
    print(f"Carpeta de resultados: {carpeta_salida}")

    if numeros:
        print("\nNúmeros encontrados:")
        for indice, numero in enumerate(numeros, start=1):
            print(
                f"  {indice}. {numero.texto} | "
                f"bbox=({numero.x}, {numero.y}, {numero.w}, {numero.h})"
            )
    else:
        print("\nNo se detectaron números agrupados.")


def main() -> None:
    """
    Función principal del programa.
    """
    configurar_logger()

    ruta_imagen = "imagen_ejemplo_2.jpg"
    carpeta_salida = "salidas2"
    mostrar_ventanas = True

    try:
        modelo = entrenar_modelo()
        imagen = cargar_imagen(ruta_imagen)
        gris, binaria = preprocesar_imagen(imagen)

        candidatos = detectar_componentes_candidatos(binaria)
        imagen_candidatos = dibujar_candidatos(imagen, candidatos)

        digitos = clasificar_digitos(modelo, gris, candidatos)
        numeros = agrupar_digitos_en_numeros(digitos)

        imagen_digitos = anotar_digitos(imagen, digitos)
        imagen_anotada = anotar_resultados(imagen, digitos, numeros)

        guardar_resultados(
            carpeta_salida,
            gris,
            binaria,
            imagen_candidatos,
            imagen_digitos,
            imagen_anotada,
        )

        imprimir_reporte(ruta_imagen, digitos, numeros, carpeta_salida)

        if mostrar_ventanas:
            mostrar_resultados(imagen, binaria, imagen_anotada)

    except FileNotFoundError as exc:
        logging.error("Archivo no encontrado: %s", exc)
        print("No se encontró la imagen de entrada. Verifica que imagen_ejemplo_2.jpg esté en la misma carpeta del script.")
    except ValueError as exc:
        logging.error("Error de valor: %s", exc)
        print("Se produjo un problema al leer o procesar la imagen.")
    except Exception as exc:
        logging.exception("Error inesperado: %s", exc)
        print("Ocurrió un error inesperado durante la ejecución del programa.")


if __name__ == "__main__":
    main()
