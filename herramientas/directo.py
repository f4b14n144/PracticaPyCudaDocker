import argparse
import csv
import json
import time
from pathlib import Path

from servicio import procesamiento as proc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imagen")
    ap.add_argument("--filtro", default="media")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mascara", help="JSON k x k (solo personalizada)")
    ap.add_argument("--ajuste", default=None)
    ap.add_argument("--hilos", type=int, default=256)
    ap.add_argument("--repeticiones", type=int, default=1)
    ap.add_argument("--png", help="guardar la salida visual")
    ap.add_argument("--f32", help="guardar la salida float32 cruda")
    ap.add_argument("--csv", help="agregar filas al CSV (mismas columnas que herramientas/experimentos.py)")
    ap.add_argument("--modo", default="directo-anfitrion", help="p. ej. directo-anfitrion o directo-contenedor")
    ap.add_argument("--entorno", default="")
    a = ap.parse_args()

    from filtros.gpu import FiltrosGPU
    esp = proc.construir_especificacion(a.filtro, a.k, json.loads(a.mascara) if a.mascara else None, a.ajuste)
    hilos = proc.validar_hilos(a.hilos)
    datos = Path(a.imagen).read_bytes()
    t0 = time.perf_counter()
    with FiltrosGPU() as gpu:
        print(json.dumps({"inicio_contexto_ms": gpu.t_contexto_ms, "inicio_compilacion_ms": gpu.t_compilacion_ms,
                          "dispositivo": gpu.nombre, "arranque_total_ms": (time.perf_counter() - t0) * 1000}))
        filas = []
        for rep in range(a.repeticiones + 1):
            t_ini = time.perf_counter()
            res, _ = proc.procesar(gpu, datos, esp, hilos)
            png = proc.codificar_png(res.visual)
            total = (time.perf_counter() - t_ini) * 1000
            filas.append({"id_solicitud": f"directo-{rep}", "experimento": "integracion", "modo": a.modo,
                          "filtro": esp.nombre, "resolucion": res.ancho, "k": esp.k, "hilos": hilos,
                          "bloques": res.bloques, "repeticion": rep, "concurrencia": 1, "estado_http": "",
                          **{f"t_{k}_ms": round(v, 4) for k, v in res.tiempos.items()},
                          "t_servidor_total_ms": round(total, 4), "entorno": a.entorno,
                          "marca_tiempo": time.strftime("%Y-%m-%dT%H:%M:%S")})
            print(json.dumps({"repeticion": rep, "filtro": esp.nombre, "kernel": esp.nombre_kernel, "k": esp.k,
                              "ancho": res.ancho, "alto": res.alto, "bloques": res.bloques,
                              "tiempos_ms": {k: round(v, 4) for k, v in res.tiempos.items()},
                              "procesamiento_total_ms": round(total, 3)}))
    if a.csv:
        from herramientas.experimentos import COLUMNAS
        ruta = Path(a.csv)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not ruta.exists()
        with ruta.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNAS, extrasaction="ignore", restval="")
            if nuevo:
                w.writeheader()
            w.writerows(filas)
    if a.png:
        Path(a.png).write_bytes(png)
    if a.f32:
        res.salida.tofile(a.f32)


if __name__ == "__main__":
    main()
