import argparse
import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

ETAPAS = ("espera_cola", "decodificacion", "reserva_y_liberacion", "cpu_a_gpu", "kernel", "gpu_a_cpu",
          "gpu_conjunto", "gpu_llamada", "conversion", "codificacion_png", "escritura", "servidor_total")
COLUMNAS = (["id_solicitud", "experimento", "modo", "filtro", "resolucion", "k", "hilos", "bloques", "repeticion",
             "ronda", "concurrencia", "estado_http", "codigo_error"]
            + [f"t_{e}_ms" for e in ETAPAS]
            + ["t_cliente_ms", "intervalo_ronda_ms", "exitosas_ronda", "throughput_rps", "entorno", "marca_tiempo"])

_cache = {}


def leer(ruta):
    if ruta not in _cache:
        _cache[ruta] = Path(ruta).read_bytes()
    return _cache[ruta]


def solicitud(sesion, url, ruta, filtro, k, hilos):
    import requests
    datos = leer(ruta)
    fila = {"filtro": filtro, "k": k, "hilos": hilos, "resolucion": Path(ruta).stem,
            "codigo_error": "", "marca_tiempo": datetime.now().isoformat(timespec="milliseconds")}
    t0 = time.perf_counter()
    try:
        r = sesion.post(f"{url}/api/procesar", files={"imagen": (Path(ruta).name, datos, "image/png")},
                        data={"filtro": filtro, "k": str(k), "hilos": str(hilos)}, timeout=300)
        fila["estado_http"] = r.status_code
        try:
            j = r.json()
        except ValueError:
            j = {}
        fila["id_solicitud"] = j.get("id", r.headers.get("X-Request-ID", ""))
        if r.status_code == 200:
            g = sesion.get(url + j["url_resultado"], timeout=300)
            _ = g.content
            if g.status_code != 200:
                fila["codigo_error"] = f"descarga_{g.status_code}"
            fila["bloques"] = j["bloques"]
            fila.update({f"t_{e}_ms": j["tiempos_ms"].get(e, "") for e in ETAPAS})
        else:
            fila["codigo_error"] = j.get("error", {}).get("codigo", f"http_{r.status_code}")
    except requests.RequestException as e:
        fila["estado_http"] = 0
        fila["codigo_error"] = type(e).__name__
    fila["t_cliente_ms"] = round((time.perf_counter() - t0) * 1000, 4)
    return fila


class Escritor:
    def __init__(self, ruta, modo, entorno):
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        nuevo = not Path(ruta).exists()
        self.f = open(ruta, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=COLUMNAS, extrasaction="ignore", restval="")
        if nuevo:
            self.w.writeheader()
        self.modo, self.entorno = modo, entorno

    def escribir(self, fila, **extra):
        fila.update(extra, modo=self.modo, entorno=self.entorno)
        self.w.writerow(fila)
        self.f.flush()
        print(f"{fila['experimento']:<12} {fila['filtro']:<10} k={fila['k']:<2} {fila['resolucion']:<5} "
              f"rep={fila.get('repeticion', '')} ronda={fila.get('ronda', '')} c={fila.get('concurrencia', '')} "
              f"http={fila.get('estado_http')} {fila['codigo_error']} kernel={fila.get('t_kernel_ms', '')} "
              f"servidor={fila.get('t_servidor_total_ms', '')} cliente={fila['t_cliente_ms']}")


def secuencial(esc, sesion, a, experimento, configs):
    for filtro, k, res in configs:
        ruta = str(Path(a.imagenes) / f"{res}.png")
        for rep in range(a.reps + 1):
            esc.escribir(solicitud(sesion, a.url, ruta, filtro, k, a.hilos),
                         experimento=experimento, repeticion=rep, concurrencia=1)


def concurrencia(esc, a):
    import requests
    ruta = str(Path(a.imagenes) / "4000.png")
    leer(ruta)
    esc.escribir(solicitud(requests.Session(), a.url, ruta, "media", 9, a.hilos),
                 experimento="concurrencia", repeticion=0, concurrencia=1)
    for n in (1, 2):
        for ronda in range(1, a.reps + 1):
            barrera = threading.Barrier(n)
            filas = [None] * n

            def cliente(i):
                s = requests.Session()
                barrera.wait()
                filas[i] = solicitud(s, a.url, ruta, "media", 9, a.hilos)

            hilos = [threading.Thread(target=cliente, args=(i,)) for i in range(n)]
            t0 = time.perf_counter()
            for h in hilos:
                h.start()
            for h in hilos:
                h.join()
            intervalo = (time.perf_counter() - t0) * 1000
            exitosas = sum(1 for f in filas if f["estado_http"] == 200 and not f["codigo_error"])
            for i, f in enumerate(filas):
                esc.escribir(f, experimento="concurrencia", repeticion=i + 1, ronda=ronda, concurrencia=n,
                             intervalo_ronda_ms=round(intervalo, 4), exitosas_ronda=exitosas,
                             throughput_rps=round(exitosas / (intervalo / 1000), 4))


def main():
    import requests
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:5000")
    ap.add_argument("--imagenes", default="Imagenes/entrada")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--hilos", type=int, default=256, help="hilos por bloque CUDA")
    ap.add_argument("--salida", default="experimentos/servicio.csv")
    ap.add_argument("--modo", default="servicio-docker")
    ap.add_argument("--entorno", default="")
    ap.add_argument("--experimentos", default="resolucion,filtros,concurrencia")
    a = ap.parse_args()

    sesion = requests.Session()
    salud = sesion.get(f"{a.url}/api/salud", timeout=30).json()
    if not salud.get("gpu_disponible"):
        raise SystemExit(f"el servicio no tiene GPU: {salud.get('error')}")
    if not a.entorno:
        a.entorno = f"{salud.get('dispositivo')} commit={salud.get('commit')}"
    arranque = Path(a.salida).with_name(Path(a.salida).stem + "_arranque.json")
    arranque.parent.mkdir(parents=True, exist_ok=True)
    arranque.write_text(json.dumps({"marca_tiempo": datetime.now().isoformat(timespec="seconds"),
                                    "url": a.url, "modo": a.modo, "entorno": a.entorno,
                                    "primera_solicitud_es_fria": salud.get("atendidas") == 0,
                                    "alcance": "inicio_contexto_ms = cuda.init + make_context; "
                                               "inicio_compilacion_ms = SourceModule (nvcc) + get_function + prepare; "
                                               "inicio_total_ms = ambos + lectura del .cu",
                                    **salud}, indent=2), encoding="utf-8")
    print(f"arranque -> {arranque}: {json.dumps({k: salud.get(k) for k in ('inicio_contexto_ms', 'inicio_compilacion_ms', 'atendidas')})}")
    if salud.get("atendidas"):
        print("aviso: el servicio ya atendio solicitudes; reinicie el contenedor si quiere registrar la primera en frio")

    esc = Escritor(a.salida, a.modo, a.entorno)
    pedidos = [e.strip() for e in a.experimentos.split(",")]
    if "resolucion" in pedidos:
        secuencial(esc, sesion, a, "resolucion", [("media", 9, r) for r in (2000, 4000, 6000)])
    if "filtros" in pedidos:
        secuencial(esc, sesion, a, "filtros",
                   [("media", 3, 4000), ("media", 9, 4000), ("media", 21, 4000), ("nitidez", 3, 4000), ("laplaciano", 3, 4000)])
    if "concurrencia" in pedidos:
        concurrencia(esc, a)
    esc.f.close()
    print(f"CSV: {a.salida}")


if __name__ == "__main__":
    main()
