import logging
import queue
import threading
import time
from concurrent.futures import Future

log = logging.getLogger("motor")


class ColaLlena(Exception):
    pass


class GPUNoDisponible(Exception):
    pass


class MotorGPU:
    def __init__(self, max_cola=4, dispositivo=0, fabrica=None):
        self.max_cola = max_cola
        self.dispositivo = dispositivo
        self._fabrica = fabrica
        self._cola = queue.Queue(maxsize=max_cola)
        self._listo = threading.Event()
        self._hilo = None
        self._ocupado = False
        self.gpu = None
        self.error_inicio = None
        self.info = {}
        self.atendidas = 0

    def iniciar(self, esperar=True):
        if self._hilo is None:
            self._hilo = threading.Thread(target=self._bucle, name="hilo-gpu", daemon=True)
            self._hilo.start()
        if esperar:
            self._listo.wait()
        return self

    def detener(self):
        if self._hilo is not None:
            self._cola.put(None)
            self._hilo.join(timeout=10)

    def _crear_gpu(self):
        if self._fabrica is not None:
            return self._fabrica()
        from filtros.gpu import FiltrosGPU
        return FiltrosGPU(dispositivo=self.dispositivo)

    def _bucle(self):
        t0 = time.perf_counter()
        try:
            self.gpu = self._crear_gpu()
            self.info = {
                "dispositivo": getattr(self.gpu, "nombre", "?"),
                "memoria_mib": getattr(self.gpu, "memoria_mib", None),
                "max_hilos_por_bloque": getattr(self.gpu, "max_hilos", None),
                "inicio_contexto_ms": getattr(self.gpu, "t_contexto_ms", None),
                "inicio_compilacion_ms": getattr(self.gpu, "t_compilacion_ms", None),
                "inicio_total_ms": (time.perf_counter() - t0) * 1000,
            }
            log.info("GPU lista: %s", self.info)
        except Exception as e:
            self.error_inicio = f"{type(e).__name__}: {e}"
            log.error("No se pudo inicializar la GPU: %s", self.error_inicio)
        finally:
            self._listo.set()

        while True:
            trabajo = self._cola.get()
            if trabajo is None:
                break
            funcion, futuro, t_encolado = trabajo
            if not futuro.set_running_or_notify_cancel():
                continue
            espera_ms = (time.perf_counter() - t_encolado) * 1000
            self._ocupado = True
            try:
                if self.gpu is None:
                    raise GPUNoDisponible(self.error_inicio or "GPU no inicializada")
                futuro.set_result((funcion(self.gpu), espera_ms))
            except BaseException as e:
                futuro.set_exception(e)
            finally:
                self._ocupado = False
                self.atendidas += 1

        if self.gpu is not None and hasattr(self.gpu, "cerrar"):
            self.gpu.cerrar()

    @property
    def disponible(self):
        return self.gpu is not None

    def estado(self):
        return {
            "gpu_disponible": self.disponible,
            "error": self.error_inicio,
            "en_cola": self._cola.qsize(),
            "procesando": self._ocupado,
            "max_cola": self.max_cola,
            "atendidas": self.atendidas,
            **self.info,
        }

    def enviar(self, funcion):
        if self._listo.is_set() and self.gpu is None:
            raise GPUNoDisponible(self.error_inicio or "GPU no inicializada")
        futuro = Future()
        try:
            self._cola.put_nowait((funcion, futuro, time.perf_counter()))
        except queue.Full:
            raise ColaLlena(f"hay {self.max_cola} solicitudes en espera; intente de nuevo en unos segundos")
        return futuro
