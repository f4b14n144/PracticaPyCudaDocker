import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from importlib import metadata


def version(paquete):
    try:
        return metadata.version(paquete)
    except metadata.PackageNotFoundError:
        return None


def comando(*args):
    if not shutil.which(args[0]):
        return None
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"error: {e}"


def ram_gib():
    try:
        with open("/proc/meminfo") as f:
            return round(int(f.readline().split()[1]) / 2**20, 2)
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida")
    a = ap.parse_args()
    info = {
        "anfitrion": socket.gethostname(),
        "en_contenedor": os.path.exists("/.dockerenv"),
        "so": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "nucleos_logicos": os.cpu_count(),
        "ram_gib": ram_gib(),
        "python": sys.version.split()[0],
        "paquetes": {p: version(p) for p in ("pycuda", "numpy", "pillow", "flask", "werkzeug", "gunicorn")},
        "commit": os.environ.get("GIT_COMMIT") or comando("git", "rev-parse", "HEAD"),
        "nvcc": comando("nvcc", "--version"),
        "nvidia_smi": comando("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
    }
    try:
        import pycuda.driver as cuda
        cuda.init()
        d = cuda.Device(0)
        v = cuda.get_driver_version()
        info["gpu"] = {"nombre": d.name(), "memoria_mib": d.total_memory() // 2**20,
                       "capacidad_computo": "%d.%d" % d.compute_capability(),
                       "cuda_controlador_anfitrion": f"{v // 1000}.{(v % 1000) // 10}",
                       "cuda_compilado_pycuda": ".".join(map(str, cuda.get_version()))}
    except Exception as e:
        info["gpu"] = {"error": f"{type(e).__name__}: {e}"}
    texto = json.dumps(info, indent=2, ensure_ascii=False)
    if a.salida:
        os.makedirs(os.path.dirname(a.salida) or ".", exist_ok=True)
        with open(a.salida, "w", encoding="utf-8") as f:
            f.write(texto)
    print(texto)


if __name__ == "__main__":
    main()
