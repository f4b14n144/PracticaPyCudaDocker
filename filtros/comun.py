import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "Imagenes" / "entrada"
REFERENCIAS = RAIZ / "referencias"

from .constantes import LAPLACIANO, MASCARA_LAPLACIANO, MASCARA_NITIDEZ, MEDIA, NITIDEZ, NOMBRES, PERSONALIZADA

TOLERANCIA = 1e-4


def mascara_media(k):

    return np.full((k, k), 1.0 / (k * k), dtype=np.float32)


def cargar_gris(ruta):
    with Image.open(ruta) as img:
        if img.mode != "L":
            img = img.convert("L")
        datos = np.asarray(img, dtype=np.uint8)
    return np.ascontiguousarray(datos)


def guardar_gris(ruta, datos_uint8):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(datos_uint8, mode="L").save(ruta)


def ruta_imagen(resolucion):
    return ENTRADA / f"{resolucion}.png"


def verificar_sha256(carpeta=ENTRADA):
    resultados = {}
    for linea in (Path(carpeta) / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        esperado, nombre = linea.split()
        ruta = Path(carpeta) / nombre
        real = hashlib.sha256(ruta.read_bytes()).hexdigest() if ruta.exists() else None
        resultados[nombre] = real == esperado
    return resultados


def convertir_a_visualizacion(salida, filtro):
    
    v = salida.astype(np.float32, copy=True)
    if filtro == LAPLACIANO:
        v += np.float32(128.0)
    v = np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))
    return np.clip(v, 0, 255).astype(np.uint8)


def errores(referencia, valores):
    d = np.abs(referencia.astype(np.float64) - valores.astype(np.float64))
    return float(d.max()), float(d.mean())


def imagen_lcg(ancho, alto):
    estado = 12345
    datos = bytearray(ancho * alto)
    for i in range(ancho * alto):
        estado = (estado * 1103515245 + 12345) & 0xFFFFFFFF
        datos[i] = (estado >> 16) & 0xFF
    return np.frombuffer(bytes(datos), dtype=np.uint8).reshape(alto, ancho).copy()


def nombre_referencia(etiqueta, filtro, k, version, hilos=None):
    sufijo = version if hilos is None else f"{version}{hilos}"
    return REFERENCIAS / f"{etiqueta}_{NOMBRES[filtro]}_k{k}_{sufijo}.f32"


def cargar_referencia(etiqueta, filtro, k, version, alto, ancho, hilos=None):
    ruta = nombre_referencia(etiqueta, filtro, k, version, hilos)
    if not ruta.exists():
        return None
    datos = np.fromfile(ruta, dtype=np.float32)
    if datos.size != ancho * alto:
        raise ValueError(f"{ruta} tiene {datos.size} valores, se esperaban {ancho * alto}")
    return datos.reshape(alto, ancho)
