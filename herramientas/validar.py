import argparse
import csv
import hashlib
import io
import json
import threading
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from filtros.comun import convertir_a_visualizacion, imagen_lcg
from filtros.constantes import LAPLACIANO, MEDIA, NITIDEZ
from servicio import procesamiento as proc

TOLERANCIA = 1e-4
HILOS = (128, 256, 512)
CASOS = (("media", 3), ("media", 9), ("media", 21), ("nitidez", 3), ("laplaciano", 3))
TAMANOS = ((1, 1), (7, 5), (513, 1), (1, 257), (1001, 777))
MANUAL = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], np.uint8)
ESPERADO_MANUAL = {
    "media": np.array([120, 210, 160, 270, 450, 330, 240, 390, 280], np.float32).reshape(3, 3) / np.float32(9),
    "nitidez": np.array([-10, 10, 70, 70, 50, 130, 230, 190, 310], np.float32).reshape(3, 3),
    "laplaciano": np.array([20, 10, -40, -30, 0, -70, -160, -110, -220], np.float32).reshape(3, 3),
}
CONVERSION = ((-10, NITIDEZ, 0), (310, NITIDEZ, 255), (-220, LAPLACIANO, 0), (-30, LAPLACIANO, 98),
              (200, LAPLACIANO, 255), (120 / 9, MEDIA, 13), (160 / 9, MEDIA, 18))

filas = []
fallos = 0


def post(url, campos, archivo, nombre="img.png"):
    limite = uuid.uuid4().hex
    partes = [f'--{limite}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
              for k, v in campos.items()]
    partes.append(f'--{limite}\r\nContent-Disposition: form-data; name="imagen"; filename="{nombre}"\r\n'
                  f'Content-Type: application/octet-stream\r\n\r\n'.encode() + archivo + b"\r\n")
    partes.append(f"--{limite}--\r\n".encode())
    req = urllib.request.Request(url + "/api/procesar", data=b"".join(partes), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={limite}"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def obtener(url):
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def a_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def servicio_f32(url, img_png, forma, filtro, k, hilos):
    st, j = post(url, {"filtro": filtro, "k": k, "hilos": hilos, "f32": 1}, img_png)
    if st != 200:
        return st, j, None
    return st, j, np.frombuffer(obtener(url + j["url_f32"]), np.float32).reshape(forma)


def referencia(carpeta, etiqueta, filtro, k, forma, version="cpu"):
    if carpeta is None:
        return None
    ruta = Path(carpeta) / f"{etiqueta}_{filtro}_k{k}_{version}.f32"
    if not ruta.exists():
        return None
    return np.fromfile(ruta, np.float32).reshape(forma)


def registrar(prueba, caso, comparacion, ref, valores, ok_extra=True, detalle=""):
    global fallos
    sin_escribir = int(np.isnan(valores).sum()) if valores is not None else -1
    if valores is None or ref is None or sin_escribir:
        emax = emed = float("nan")
        ok = False
    else:
        d = np.abs(ref.astype(np.float64) - valores.astype(np.float64))
        emax, emed = float(d.max()), float(d.mean())
        ok = emax <= TOLERANCIA
    ok = ok and ok_extra
    fallos += not ok
    filas.append({"prueba": prueba, "caso": caso, "comparacion": comparacion, "error_abs_max": emax,
                  "error_abs_medio": emed, "pixeles_sin_escribir": sin_escribir, "tolerancia": TOLERANCIA,
                  "cumple": "si" if ok else "no", "detalle": detalle})
    print(f"{'[OK]   ' if ok else '[FALLA]'} {prueba} {caso} {comparacion} max={emax:.3g} medio={emed:.3g} {detalle}",
          flush=True)


def comprobar(prueba, caso, ok, detalle=""):
    global fallos
    fallos += not ok
    filas.append({"prueba": prueba, "caso": caso, "comparacion": "", "error_abs_max": "", "error_abs_medio": "",
                  "pixeles_sin_escribir": "", "tolerancia": "", "cumple": "si" if ok else "no", "detalle": detalle})
    print(f"{'[OK]   ' if ok else '[FALLA]'} {prueba} {caso} {detalle}", flush=True)


def prueba_entradas(carpeta):
    print("== entradas (sha256)")
    sumas = Path(carpeta) / "SHA256SUMS.txt"
    for linea in sumas.read_text().split("\n"):
        if linea.strip():
            esperado, nombre = linea.split()
            real = hashlib.sha256((Path(carpeta) / nombre).read_bytes()).hexdigest()
            comprobar("entradas", nombre, real == esperado, real[:16])


def prueba_manual(url, gpu, refs):
    print("== matriz manual 3x3")
    for filtro, esperado in ESPERADO_MANUAL.items():
        for h in HILOS:
            _, j, serv = servicio_f32(url, a_png(MANUAL), (3, 3), filtro, 3, h)
            registrar("manual", f"{filtro}_h{h}", "servicio_vs_a_mano", esperado, serv, detalle=j.get("id", ""))
            esp = proc.construir_especificacion(filtro, 3)
            directo, _ = gpu.ejecutar(MANUAL, esp.kernel, k=3, hilos_por_bloque=h, centinela=np.nan)
            registrar("manual", f"{filtro}_h{h}", "directo_vs_a_mano", esperado, directo)
        cpu = referencia(refs, "manual", filtro, 3, (3, 3))
        if cpu is not None:
            registrar("manual", filtro, "c_secuencial_vs_a_mano", esperado, cpu)
    print("== conversion 0-255")
    for valor, f, esperado in CONVERSION:
        v = int(convertir_a_visualizacion(np.array([valor], np.float32), f)[0])
        comprobar("conversion", f"{f}_{valor:.2f}", v == esperado, f"{v} esperado {esperado}")


def prueba_limites(url, gpu, refs):
    print("== tamanos no multiplos de los hilos")
    for ancho, alto in TAMANOS:
        img = imagen_lcg(ancho, alto)
        etiqueta = f"lcg{ancho}x{alto}"
        for filtro, k in CASOS:
            cpu = referencia(refs, etiqueta, filtro, k, (alto, ancho))
            esp = proc.construir_especificacion(filtro, k)
            for h in HILOS:
                directo, _ = gpu.ejecutar(img, esp.kernel, k=k, hilos_por_bloque=h, centinela=np.nan)
                _, j, serv = servicio_f32(url, a_png(img), (alto, ancho), filtro, k, h)
                caso = f"{etiqueta}_{filtro}_k{k}_h{h}"
                registrar("limites", caso, "servicio_vs_directo", directo, serv,
                          detalle=f"bloques={j.get('bloques')} resto={(ancho * alto) % h}")
                if cpu is not None:
                    registrar("limites", caso, "servicio_vs_c_secuencial", cpu, serv)


def prueba_imagenes(url, gpu, refs, entrada, figuras):
    print("== imagenes reales")
    configs = [(f, k, r, 256) for r in (2000, 4000) for f, k in CASOS]
    configs += [("media", 9, 4000, 128), ("media", 9, 4000, 512), ("media", 9, 6000, 256)]
    cache = {}
    for filtro, k, res, h in configs:
        if res not in cache:
            cache[res] = (Path(entrada) / f"{res}.png").read_bytes()
        datos = cache[res]
        esp = proc.construir_especificacion(filtro, k)
        directo, _ = proc.procesar(gpu, datos, esp, h)
        st, j, serv = servicio_f32(url, datos, (res, res), filtro, k, h)
        png = np.asarray(Image.open(io.BytesIO(obtener(url + j["url_resultado"])))) if st == 200 else None
        d_png = int(np.abs(png.astype(np.int16) - directo.visual.astype(np.int16)).max()) if png is not None else -1
        caso = f"{res}_{filtro}_k{k}_h{h}"
        registrar("imagen", caso, "servicio_vs_directo", directo.salida, serv, ok_extra=d_png == 0,
                  detalle=f"png_max={d_png} id={j.get('id', '')}")
        cpu = referencia(refs, f"img{res}", filtro, k, (res, res))
        if cpu is not None:
            registrar("imagen", caso, "servicio_vs_c_secuencial", cpu, serv)
        cuda = referencia(refs, f"img{res}", filtro, k, (res, res), f"cuda{h}")
        if cuda is not None:
            registrar("imagen", caso, "servicio_vs_cuda_c", cuda, serv)
        if figuras and res == 2000 and h == 256:
            Image.fromarray(png).save(Path(figuras) / f"{filtro}_{k}_2000.png")
        del serv, directo


def evidencia_visual(entrada, figuras):
    print("== evidencia visual")
    img = np.asarray(Image.open(Path(entrada) / "2000.png").convert("L"))
    Image.fromarray(img).save(Path(figuras) / "original_2000.png")
    y0, x0, lado = 900, 900, 300
    partes = [img]
    for nombre in ("media_3", "media_9", "media_21", "nitidez_3", "laplaciano_3"):
        partes.append(np.asarray(Image.open(Path(figuras) / f"{nombre}_2000.png")))
    sep = np.full((lado, 8), 255, np.uint8)
    mosaico = np.hstack([p for a in partes for p in (a[y0:y0 + lado, x0:x0 + lado], sep)][:-1])
    Image.fromarray(mosaico).save(Path(figuras) / "recortes_2000.png")
    comprobar("visual", "recortes_2000", True, "original, media 3, 9, 21, nitidez, laplaciano; region 300x300 en (900,900)")


def prueba_invalidas(url):
    print("== entradas invalidas")
    png_ok = a_png(imagen_lcg(50, 50))
    max_mb = json.loads(obtener(url + "/api/config"))["max_mb"]
    casos = [("archivo_no_valido", {"filtro": "media"}, b"esto no es una imagen", 415),
             ("archivo_grande", {"filtro": "media"}, b"\0" * (max_mb * 2**20 + 2**21), 413),
             ("filtro_desconocido", {"filtro": "sobel"}, png_ok, 400),
             ("mascara_no_permitida", {"filtro": "media", "k": 4}, png_ok, 400),
             ("hilos_no_permitidos", {"filtro": "media", "hilos": 100}, png_ok, 400),
             ("recuperacion", {"filtro": "media", "k": 3}, png_ok, 200)]
    for nombre, campos, archivo, esperado in casos:
        try:
            st, j = post(url, campos, archivo)
        except (urllib.error.URLError, ConnectionError) as e:
            st, j = (413 if esperado == 413 else 0), {"error": {"codigo": f"conexion cerrada {type(e).__name__}"}}
        codigo = j.get("error", {}).get("codigo", "ok")
        mensaje = j.get("error", {}).get("mensaje", "")
        comprobar("invalidas", nombre, st == esperado, f"{st} {codigo} {mensaje}")


def prueba_aislamiento(url):
    print("== aislamiento")
    entradas = {"A": np.full((64, 96), 40, np.uint8), "B": np.full((80, 48), 200, np.uint8)}
    resp = {}

    def enviar(n):
        resp[n] = post(url, {"filtro": "media", "k": 3, "f32": 1}, a_png(entradas[n]))

    hilos = [threading.Thread(target=enviar, args=(n,)) for n in entradas]
    [h.start() for h in hilos]
    [h.join() for h in hilos]
    ids = set()
    for n, img in entradas.items():
        st, j = resp[n]
        with Image.open(io.BytesIO(obtener(url + j["url_resultado"]))) as r:
            tam = r.size
        centro = np.frombuffer(obtener(url + j["url_f32"]), np.float32).reshape(img.shape)[10, 10]
        ids.add(j["id"])
        comprobar("aislamiento", n, st == 200 and tam == (img.shape[1], img.shape[0]) and abs(centro - img[0, 0]) < 1e-4,
                  f"id={j['id']} tamano={tam} centro={centro:.1f}")
    comprobar("aislamiento", "ids_distintos", len(ids) == 2)


def prueba_cola(url, datos):
    print("== cola llena")
    salud = json.loads(obtener(url + "/api/salud"))
    n = int(salud["max_cola"]) + 4
    resp = [None] * n
    barrera = threading.Barrier(n)

    def rafaga(i):
        barrera.wait()
        try:
            resp[i] = post(url, {"filtro": "media", "k": 21}, datos)
        except (urllib.error.URLError, ConnectionError) as e:
            resp[i] = (0, {"error": {"codigo": type(e).__name__}})

    hilos = [threading.Thread(target=rafaga, args=(i,)) for i in range(n)]
    [h.start() for h in hilos]
    [h.join() for h in hilos]
    n200 = sum(1 for st, _ in resp if st == 200)
    n503 = sum(1 for st, j in resp if st == 503 and j.get("error", {}).get("codigo") == "cola_llena")
    otros = n - n200 - n503
    ids = [j["id"] for st, j in resp if st == 200]
    recupera = post(url, {"filtro": "media", "k": 3}, a_png(imagen_lcg(20, 20)))[0] == 200
    comprobar("cola", f"rafaga_{n}", n200 >= 1 and n503 >= 1 and otros == 0 and len(set(ids)) == n200 and recupera,
              f"max_cola={salud['max_cola']} 200={n200} 503={n503} otros={otros} recupera={recupera}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5000")
    ap.add_argument("--entrada", default="Imagenes/entrada")
    ap.add_argument("--referencias", default=None)
    ap.add_argument("--figuras", default="experimentos/figuras")
    ap.add_argument("--salida", default="experimentos/validacion_servicio.csv")
    a = ap.parse_args()
    Path(a.figuras).mkdir(parents=True, exist_ok=True)
    from filtros.gpu import FiltrosGPU

    prueba_entradas(a.entrada)
    with FiltrosGPU() as gpu:
        prueba_manual(a.url, gpu, a.referencias)
        prueba_limites(a.url, gpu, a.referencias)
        prueba_imagenes(a.url, gpu, a.referencias, a.entrada, a.figuras)
    evidencia_visual(a.entrada, a.figuras)
    prueba_invalidas(a.url)
    prueba_aislamiento(a.url)
    prueba_cola(a.url, (Path(a.entrada) / "6000.png").read_bytes())

    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    with open(a.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"\n{len(filas)} pruebas, {fallos} fallos -> {a.salida}")
    print("TODO OK" if fallos == 0 else "HAY FALLOS")
    raise SystemExit(0 if fallos == 0 else 1)


if __name__ == "__main__":
    main()
