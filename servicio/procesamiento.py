import io
import math
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from filtros.comun import convertir_a_visualizacion
from filtros.constantes import (LAPLACIANO, MASCARA_LAPLACIANO, MASCARA_NITIDEZ, MEDIA, NITIDEZ,
                                NOMBRES, PERSONALIZADA)

FILTROS_POR_NOMBRE = {nombre: codigo for codigo, nombre in NOMBRES.items()}
FORMATOS_PERMITIDOS = {"PNG", "JPEG", "MPO", "BMP", "TIFF", "WEBP", "GIF", "PPM", "TGA", "ICO"}
AJUSTES = ("recorte", "desplazar128", "normalizar")
LIMITE_COEFICIENTE = 1e4


class ErrorEntrada(Exception):

    def __init__(self, codigo, mensaje, http=400):
        super().__init__(mensaje)
        self.codigo = codigo
        self.mensaje = mensaje
        self.http = http


@dataclass(frozen=True)
class Limites:
    max_bytes: int = 64 * 1024 * 1024
    max_lado: int = 16384
    max_pixeles: int = 100_000_000
    k_min: int = 3
    k_max: int = 31
    hilos_permitidos: tuple = (32, 64, 128, 256, 512, 1024)


@dataclass
class Especificacion:
    filtro: int
    k: int
    mascara: np.ndarray
    kernel: int
    ajuste: str

    @property
    def nombre(self):
        return NOMBRES[self.filtro]

    @property
    def nombre_kernel(self):
        return {MEDIA: "kernelFiltroMedia", NITIDEZ: "kernelFiltroNitidez",
                LAPLACIANO: "kernelFiltroLaplaciano", PERSONALIZADA: "kernelFiltroMascara"}[self.kernel]


@dataclass
class Resultado:
    salida: np.ndarray
    visual: np.ndarray
    ancho: int
    alto: int
    bloques: int
    tiempos: dict = field(default_factory=dict)


def mascara_cruz(k, centro, brazo):
    m = np.zeros((k, k), dtype=np.float32)
    r = k // 2
    m[r, :] = brazo
    m[:, r] = brazo
    m[r, r] = centro
    return m


def mascara_predefinida(filtro, k):
    r = k // 2
    if filtro == MEDIA:
        return np.full((k, k), 1.0 / (k * k), dtype=np.float32)
    if filtro == LAPLACIANO:
        return mascara_cruz(k, -4.0 * r, 1.0)
    if filtro == NITIDEZ:
        return mascara_cruz(k, 1.0 + 4.0 * r, -1.0)
    raise ErrorEntrada("filtro_desconocido", f"filtro sin mascara predefinida: {filtro}")


def _entero(valor, nombre, defecto):
    if valor is None or valor == "":
        return defecto
    try:
        return int(str(valor).strip())
    except ValueError:
        raise ErrorEntrada("parametro_invalido", f"'{nombre}' debe ser un entero (recibido: {valor!r})")


def _mascara_usuario(datos, k):
    if datos is None:
        raise ErrorEntrada("mascara_requerida", "el filtro personalizado necesita el campo 'mascara'")
    if not isinstance(datos, list) or len(datos) != k or any(not isinstance(f, list) or len(f) != k for f in datos):
        raise ErrorEntrada("mascara_invalida", f"'mascara' debe ser una matriz {k}x{k} (lista de listas)")
    try:
        m = np.array(datos, dtype=np.float64)
    except (TypeError, ValueError):
        raise ErrorEntrada("mascara_invalida", "'mascara' solo puede contener numeros")
    if not np.all(np.isfinite(m)) or np.any(np.abs(m) > LIMITE_COEFICIENTE):
        raise ErrorEntrada("mascara_invalida", f"los coeficientes deben ser finitos y |c| <= {LIMITE_COEFICIENTE:g}")
    return m.astype(np.float32)


def construir_especificacion(filtro, k=None, mascara=None, ajuste=None, limites=Limites()):
    nombre = (filtro or "").strip().lower()
    if nombre not in FILTROS_POR_NOMBRE:
        raise ErrorEntrada("filtro_desconocido",
                           f"filtro desconocido '{filtro}'; use uno de: {', '.join(FILTROS_POR_NOMBRE)}")
    codigo = FILTROS_POR_NOMBRE[nombre]
    k = _entero(k, "k", 3)
    if k < limites.k_min or k > limites.k_max or k % 2 == 0:
        raise ErrorEntrada("mascara_no_permitida",
                           f"tamano de mascara {k} no permitido; debe ser impar entre {limites.k_min} y {limites.k_max}")

    if codigo == PERSONALIZADA:
        m = _mascara_usuario(mascara, k)
        kernel = PERSONALIZADA
    else:
        m = mascara_predefinida(codigo, k)
        kernel = codigo if (codigo == MEDIA or k == 3) else PERSONALIZADA

    if ajuste in (None, ""):
        ajuste = {LAPLACIANO: "desplazar128"}.get(codigo, "recorte")
    if ajuste not in AJUSTES:
        raise ErrorEntrada("parametro_invalido", f"ajuste '{ajuste}' no valido; use uno de: {', '.join(AJUSTES)}")
    return Especificacion(filtro=codigo, k=k, mascara=m, kernel=kernel, ajuste=ajuste)


def validar_hilos(hilos, limites=Limites()):
    hilos = _entero(hilos, "hilos", 256)
    if hilos not in limites.hilos_permitidos:
        raise ErrorEntrada("parametro_invalido",
                           f"hilos por bloque {hilos} no permitido; use uno de: {limites.hilos_permitidos}")
    return hilos


def decodificar(datos, limites=Limites()):
    if not datos:
        raise ErrorEntrada("archivo_vacio", "no se recibio ninguna imagen")
    if len(datos) > limites.max_bytes:
        raise ErrorEntrada("archivo_grande",
                           f"el archivo pesa {len(datos) / 2**20:.1f} MiB; el limite es {limites.max_bytes / 2**20:.0f} MiB", 413)
    try:
        with Image.open(io.BytesIO(datos)) as img:
            formato = img.format
            ancho, alto = img.size
            if formato not in FORMATOS_PERMITIDOS:
                raise ErrorEntrada("formato_no_soportado",
                                   f"formato {formato} no soportado; use {', '.join(sorted(FORMATOS_PERMITIDOS))}", 415)
            _validar_dimensiones(ancho, alto, limites)
            img.load()
            arr = a_grises(img)
    except ErrorEntrada:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise ErrorEntrada("archivo_invalido", f"el archivo no es una imagen valida ({type(e).__name__})", 415)
    except (OSError, ValueError, SyntaxError) as e:
        raise ErrorEntrada("archivo_invalido", f"la imagen esta danada o no se puede leer: {e}", 415)
    return arr, formato


def a_grises(img):
    img = ImageOps.exif_transpose(img) or img
    if img.mode == "L":
        g = img
    elif img.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        v = np.asarray(img, dtype=np.float64)
        if img.mode.startswith("I;16"):
            v = v / 257.0
        elif v.size and (v.min() < 0 or v.max() > 255):
            lo, hi = float(v.min()), float(v.max())
            v = (v - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(v)
        return np.ascontiguousarray(np.clip(np.rint(v), 0, 255).astype(np.uint8))
    elif img.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        fondo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        g = Image.alpha_composite(fondo, rgba).convert("L")
    else:
        g = img.convert("L")
    return np.ascontiguousarray(np.asarray(g, dtype=np.uint8))


def decodificar_cabecera(datos, limites=Limites()):
    if not datos:
        raise ErrorEntrada("archivo_vacio", "no se recibio ninguna imagen")
    if len(datos) > limites.max_bytes:
        raise ErrorEntrada("archivo_grande",
                           f"el archivo pesa {len(datos) / 2**20:.1f} MiB; el limite es {limites.max_bytes / 2**20:.0f} MiB", 413)
    try:
        with Image.open(io.BytesIO(datos)) as img:
            formato, (ancho, alto) = img.format, img.size
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError) as e:
        raise ErrorEntrada("archivo_invalido", f"el archivo no es una imagen valida ({type(e).__name__})", 415)
    if formato not in FORMATOS_PERMITIDOS:
        raise ErrorEntrada("formato_no_soportado",
                           f"formato {formato} no soportado; use {', '.join(sorted(FORMATOS_PERMITIDOS))}", 415)
    _validar_dimensiones(ancho, alto, limites)
    return formato, ancho, alto


def _validar_dimensiones(ancho, alto, limites):
    if ancho < 1 or alto < 1:
        raise ErrorEntrada("dimensiones_invalidas", "la imagen no tiene pixeles")
    if ancho > limites.max_lado or alto > limites.max_lado or ancho * alto > limites.max_pixeles:
        raise ErrorEntrada("dimensiones_excedidas",
                           f"la imagen es {ancho}x{alto}; el maximo es {limites.max_lado} px por lado "
                           f"y {limites.max_pixeles / 1e6:.0f} MP en total", 413)


def a_visualizacion(salida, esp):
    if esp.ajuste == "normalizar":
        lo, hi = float(salida.min()), float(salida.max())
        if math.isclose(lo, hi):
            return np.zeros(salida.shape, dtype=np.uint8)
        v = (salida.astype(np.float32) - np.float32(lo)) * np.float32(255.0 / (hi - lo))
        return convertir_a_visualizacion(v, MEDIA)
    if esp.ajuste == "desplazar128":
        return convertir_a_visualizacion(salida, LAPLACIANO)
    return convertir_a_visualizacion(salida, MEDIA)


def codificar_png(visual):
    buf = io.BytesIO()
    Image.fromarray(visual, mode="L").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def aplicar(gpu, imagen, esp, hilos):
    mascara = esp.mascara if esp.kernel == PERSONALIZADA else None
    return gpu.ejecutar(imagen, esp.kernel, k=esp.k, hilos_por_bloque=hilos, mascara=mascara)


def procesar(gpu, datos, esp, hilos, limites=Limites()):
    t0 = time.perf_counter()
    imagen, formato = decodificar(datos, limites)
    t1 = time.perf_counter()
    salida, tg = aplicar(gpu, imagen, esp, hilos)
    t2 = time.perf_counter()
    visual = a_visualizacion(salida, esp)
    t3 = time.perf_counter()
    alto, ancho = imagen.shape
    tiempos = {
        "decodificacion": (t1 - t0) * 1000,
        "reserva_y_liberacion": tg["t_llamada"] - tg["t_conjunto"],
        "cpu_a_gpu": tg["t_cpu_a_gpu"],
        "kernel": tg["t_kernel"],
        "gpu_a_cpu": tg["t_gpu_a_cpu"],
        "gpu_conjunto": tg["t_conjunto"],
        "gpu_llamada": tg["t_llamada"],
        "conversion": (t3 - t2) * 1000,
    }
    return Resultado(salida=salida, visual=visual, ancho=ancho, alto=alto,
                     bloques=tg["bloques"], tiempos=tiempos), formato
