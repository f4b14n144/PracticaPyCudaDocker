import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from concurrent.futures import TimeoutError as FuturoTimeout
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

from . import procesamiento as proc
from .motor import ColaLlena, GPUNoDisponible, MotorGPU

log = logging.getLogger("servicio")
RAIZ_STATIC = Path(__file__).resolve().parent / "static"
ID_VALIDO = re.compile(r"^[0-9a-f]{32}$")


def _env_int(nombre, defecto):
    return int(os.environ.get(nombre, defecto))


class Config:
    MAX_MB = _env_int("MAX_MB", 100)
    MAX_LADO = _env_int("MAX_LADO", 16384)
    MAX_MPIX = _env_int("MAX_MPIX", 100)
    MAX_COLA = _env_int("MAX_COLA", 4)
    TIMEOUT_S = _env_int("TIMEOUT_S", 120)
    TTL_RESULTADOS_S = _env_int("TTL_RESULTADOS_S", 900)
    DIR_RESULTADOS = os.environ.get("DIR_RESULTADOS") or os.path.join(tempfile.gettempdir(), "convolucion-resultados")


def crear_app(motor=None, config=Config):
    limites = proc.Limites(max_bytes=config.MAX_MB * 2**20, max_lado=config.MAX_LADO,
                           max_pixeles=config.MAX_MPIX * 1_000_000)
    Image.MAX_IMAGE_PIXELS = limites.max_pixeles
    dir_res = Path(config.DIR_RESULTADOS)
    dir_res.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = limites.max_bytes + 1 * 2**20
    motor = motor or MotorGPU(max_cola=config.MAX_COLA)
    motor.iniciar(esperar=True)
    app.extensions["motor"] = motor

    def limpiar():
        limite = time.time() - config.TTL_RESULTADOS_S
        for f in dir_res.glob("*"):
            try:
                if f.stat().st_mtime < limite:
                    f.unlink()
            except OSError:
                pass

    def bucle_limpieza():
        while True:
            time.sleep(min(60, config.TTL_RESULTADOS_S))
            limpiar()

    threading.Thread(target=bucle_limpieza, name="limpieza", daemon=True).start()

    def error(codigo, mensaje, http, id_solicitud=None):
        cuerpo = {"error": {"codigo": codigo, "mensaje": mensaje}}
        if id_solicitud:
            cuerpo["id"] = id_solicitud
        resp = jsonify(cuerpo)
        resp.status_code = http
        if id_solicitud:
            resp.headers["X-Request-ID"] = id_solicitud
        log.warning("solicitud %s -> %s %s: %s", id_solicitud, http, codigo, mensaje)
        return resp

    @app.errorhandler(RequestEntityTooLarge)
    def demasiado_grande(_):
        return error("archivo_grande", f"la solicitud supera el limite de {config.MAX_MB} MiB", 413)

    @app.errorhandler(404)
    def no_encontrado(_):
        return error("no_encontrado", "recurso no encontrado", 404)

    @app.errorhandler(405)
    def metodo(_):
        return error("metodo_no_permitido", "metodo HTTP no permitido en esta ruta", 405)

    @app.get("/")
    def indice():
        return send_from_directory(RAIZ_STATIC, "index.html")

    @app.get("/static/<path:nombre>")
    def estaticos(nombre):
        return send_from_directory(RAIZ_STATIC, nombre)

    @app.get("/api/salud")
    def salud():
        estado = motor.estado()
        estado["commit"] = os.environ.get("GIT_COMMIT", "desconocido")
        return jsonify(estado), (200 if estado["gpu_disponible"] else 503)

    @app.get("/api/config")
    def configuracion():
        return jsonify({
            "filtros": list(proc.FILTROS_POR_NOMBRE),
            "k_min": limites.k_min, "k_max": limites.k_max,
            "hilos_permitidos": list(limites.hilos_permitidos),
            "ajustes": list(proc.AJUSTES),
            "max_mb": config.MAX_MB, "max_lado": limites.max_lado, "max_mpix": config.MAX_MPIX,
            "formatos": sorted(proc.FORMATOS_PERMITIDOS),
            "ttl_resultados_s": config.TTL_RESULTADOS_S,
        })

    @app.post("/api/procesar")
    def procesar():
        t_recibido = time.perf_counter()
        id_sol = uuid.uuid4().hex
        try:
            archivo = request.files.get("imagen")
            if archivo is None:
                raise proc.ErrorEntrada("archivo_vacio", "falta el campo 'imagen' (multipart/form-data)")
            mascara = request.form.get("mascara")
            if mascara:
                try:
                    mascara = json.loads(mascara)
                except json.JSONDecodeError:
                    raise proc.ErrorEntrada("mascara_invalida", "'mascara' debe ser JSON, p. ej. [[0,1,0],[1,-4,1],[0,1,0]]")
            esp = proc.construir_especificacion(request.form.get("filtro"), request.form.get("k"),
                                                mascara, request.form.get("ajuste"), limites)
            hilos = proc.validar_hilos(request.form.get("hilos"), limites)
            guardar_f32 = request.form.get("f32", "0") in ("1", "true", "si")
            datos = archivo.read()
        except proc.ErrorEntrada as e:
            return error(e.codigo, e.mensaje, e.http, id_sol)

        try:
            proc.decodificar_cabecera(datos, limites)
        except proc.ErrorEntrada as e:
            return error(e.codigo, e.mensaje, e.http, id_sol)

        try:
            futuro = motor.enviar(lambda gpu: proc.procesar(gpu, datos, esp, hilos, limites))
        except ColaLlena as e:
            return error("cola_llena", str(e), 503, id_sol)
        except GPUNoDisponible as e:
            return error("gpu_no_disponible", f"no hay GPU CUDA utilizable: {e}", 503, id_sol)

        try:
            (resultado, formato), espera_ms = futuro.result(timeout=config.TIMEOUT_S)
        except FuturoTimeout:
            futuro.cancel()
            return error("tiempo_agotado", f"la solicitud supero {config.TIMEOUT_S} s", 504, id_sol)
        except proc.ErrorEntrada as e:
            return error(e.codigo, e.mensaje, e.http, id_sol)
        except GPUNoDisponible as e:
            return error("gpu_no_disponible", f"no hay GPU CUDA utilizable: {e}", 503, id_sol)
        except Exception as e:
            nombre = type(e).__name__
            if "MemoryError" in nombre or "out of memory" in str(e).lower():
                return error("memoria_gpu", f"memoria insuficiente en la GPU: {e}", 507, id_sol)
            log.exception("fallo procesando %s", id_sol)
            return error("error_gpu", f"error durante el procesamiento ({nombre}): {e}", 500, id_sol)

        t0 = time.perf_counter()
        png = proc.codificar_png(resultado.visual)
        t1 = time.perf_counter()
        (dir_res / f"{id_sol}.png").write_bytes(png)
        if guardar_f32:
            resultado.salida.astype(np.float32).tofile(dir_res / f"{id_sol}.f32")
        t2 = time.perf_counter()

        tiempos = dict(resultado.tiempos)
        tiempos["espera_cola"] = espera_ms
        tiempos["codificacion_png"] = (t1 - t0) * 1000
        tiempos["escritura"] = (t2 - t1) * 1000
        tiempos["servidor_total"] = (t2 - t_recibido) * 1000
        cuerpo = {
            "id": id_sol,
            "filtro": esp.nombre, "kernel": esp.nombre_kernel, "k": esp.k, "ajuste": esp.ajuste,
            "mascara": esp.mascara.tolist(),
            "hilos_por_bloque": hilos, "bloques": resultado.bloques,
            "ancho": resultado.ancho, "alto": resultado.alto, "formato_entrada": formato,
            "rango_salida": [float(resultado.salida.min()), float(resultado.salida.max())],
            "tiempos_ms": {k: round(v, 4) for k, v in tiempos.items()},
            "alcance_tiempos": {
                "kernel": "eventos CUDA alrededor del lanzamiento",
                "cpu_a_gpu / gpu_a_cpu": "perf_counter + synchronize; memoria paginable",
                "gpu_conjunto": "cpu_a_gpu + kernel + gpu_a_cpu (sin reserva)",
                "gpu_llamada": "gpu_conjunto + reserva y liberacion de memoria",
                "servidor_total": "desde recibir la solicitud hasta escribir el PNG (incluye espera en cola)",
            },
            "url_resultado": f"/api/resultados/{id_sol}.png",
            "url_f32": f"/api/resultados/{id_sol}.f32" if guardar_f32 else None,
            "expira_en_s": config.TTL_RESULTADOS_S,
        }
        log.info("solicitud %s ok: %s k=%d %dx%d kernel=%.3f ms servidor=%.1f ms", id_sol, esp.nombre, esp.k,
                 resultado.ancho, resultado.alto, tiempos["kernel"], tiempos["servidor_total"])
        resp = jsonify(cuerpo)
        resp.headers["X-Request-ID"] = id_sol
        return resp

    @app.get("/api/resultados/<id_sol>.<ext>")
    def resultado(id_sol, ext):
        if not ID_VALIDO.match(id_sol) or ext not in ("png", "f32"):
            return error("no_encontrado", "identificador de resultado invalido", 404)
        ruta = dir_res / f"{id_sol}.{ext}"
        if not ruta.exists():
            return error("no_encontrado", "el resultado no existe o ya expiro", 404, id_sol)
        descargar = request.args.get("descargar") == "1"
        return send_file(ruta, mimetype="image/png" if ext == "png" else "application/octet-stream",
                         as_attachment=descargar, download_name=f"resultado_{id_sol}.{ext}", max_age=0)

    return app
