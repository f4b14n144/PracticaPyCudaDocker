import io
import tempfile
import threading
import time
import unittest

import numpy as np
from PIL import Image

from filtros.constantes import LAPLACIANO, MASCARA_LAPLACIANO, MASCARA_NITIDEZ, MEDIA, NITIDEZ, PERSONALIZADA
from servicio import procesamiento as proc
from servicio.motor import ColaLlena, GPUNoDisponible, MotorGPU
from servicio.web import Config, crear_app


class GPUFalsa:
    nombre, memoria_mib, max_hilos, t_contexto_ms, t_compilacion_ms = "falsa", 1, 1024, 0.0, 0.0

    def __init__(self):
        self.ultima = None

    def ejecutar(self, entrada, filtro, k=3, hilos_por_bloque=256, mascara=None, **kw):
        self.ultima = dict(filtro=filtro, k=k, hilos=hilos_por_bloque, mascara=mascara)
        return entrada.astype(np.float32), {
            "t_cpu_a_gpu": 1.0, "t_kernel": 2.0, "t_gpu_a_cpu": 1.0, "t_conjunto": 4.0, "t_llamada": 5.0,
            "bloques": (entrada.size + hilos_por_bloque - 1) // hilos_por_bloque}

    def cerrar(self):
        pass


def sin_gpu():
    raise RuntimeError("sin GPU")


def png(ancho, alto, modo="RGB", formato="PNG"):
    buf = io.BytesIO()
    Image.new(modo, (ancho, alto), (10, 20, 30) if modo == "RGB" else 100).save(buf, format=formato)
    return buf.getvalue()


class ConfigPrueba(Config):
    MAX_MB, MAX_LADO, MAX_MPIX, MAX_COLA, TIMEOUT_S, TTL_RESULTADOS_S = 1, 64, 1, 2, 10, 600
    DIR_RESULTADOS = tempfile.mkdtemp()


def codigo_de(prueba, fn):
    with prueba.assertRaises(proc.ErrorEntrada) as ctx:
        fn()
    return ctx.exception.codigo


class TestMascaras(unittest.TestCase):
    def test_3x3_son_las_originales(self):
        np.testing.assert_array_equal(proc.mascara_predefinida(NITIDEZ, 3), np.array(MASCARA_NITIDEZ, np.float32))
        np.testing.assert_array_equal(proc.mascara_predefinida(LAPLACIANO, 3), np.array(MASCARA_LAPLACIANO, np.float32))

    def test_media_k9(self):
        m = proc.mascara_predefinida(MEDIA, 9)
        self.assertEqual(m.shape, (9, 9))
        np.testing.assert_allclose(m, 1 / 81, rtol=1e-6)
        self.assertAlmostEqual(float(m.sum()), 1.0, places=5)

    def test_cruz_generalizada_k5(self):
        lap, nit = proc.mascara_predefinida(LAPLACIANO, 5), proc.mascara_predefinida(NITIDEZ, 5)
        self.assertAlmostEqual(float(lap.sum()), 0.0)
        self.assertEqual(lap[2, 2], -8.0)
        self.assertAlmostEqual(float(nit.sum()), 1.0)
        self.assertEqual(nit[2, 2], 9.0)


class TestEspecificacion(unittest.TestCase):
    def test_kernel_elegido(self):
        self.assertEqual(proc.construir_especificacion("media", 21).kernel, MEDIA)
        self.assertEqual(proc.construir_especificacion("nitidez", 3).kernel, NITIDEZ)
        self.assertEqual(proc.construir_especificacion("nitidez", 5).kernel, PERSONALIZADA)
        self.assertEqual(proc.construir_especificacion("laplaciano", "3").ajuste, "desplazar128")
        self.assertEqual(proc.construir_especificacion("media", 3).ajuste, "recorte")
        self.assertEqual(proc.construir_especificacion("Media", " 9 ").k, 9)

    def test_errores(self):
        c = lambda *a: codigo_de(self, lambda: proc.construir_especificacion(*a))
        self.assertEqual(c("sobel", 3), "filtro_desconocido")
        self.assertEqual(c("media", 4), "mascara_no_permitida")
        self.assertEqual(c("media", 33), "mascara_no_permitida")
        self.assertEqual(c("media", "abc"), "parametro_invalido")
        self.assertEqual(c("personalizada", 3, None), "mascara_requerida")
        self.assertEqual(c("personalizada", 3, [[1, 2], [3, 4]]), "mascara_invalida")
        self.assertEqual(c("personalizada", 3, [[1e5] * 3] * 3), "mascara_invalida")
        self.assertEqual(c("personalizada", 3, [["a"] * 3] * 3), "mascara_invalida")
        self.assertEqual(c("media", 3, None, "xx"), "parametro_invalido")

    def test_personalizada_valida(self):
        esp = proc.construir_especificacion("personalizada", 3, [[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.assertEqual(esp.kernel, PERSONALIZADA)
        self.assertEqual(esp.mascara.dtype, np.float32)

    def test_hilos(self):
        self.assertEqual(proc.validar_hilos(256), 256)
        self.assertEqual(proc.validar_hilos(None), 256)
        self.assertEqual(codigo_de(self, lambda: proc.validar_hilos(100)), "parametro_invalido")


class TestDecodificar(unittest.TestCase):
    def test_png(self):
        arr, formato = proc.decodificar(png(10, 7))
        self.assertEqual((formato, arr.shape, arr.dtype), ("PNG", (7, 10), np.uint8))
        self.assertEqual(proc.decodificar_cabecera(png(10, 7)), ("PNG", 10, 7))

    def test_invalidos(self):
        for fn in (proc.decodificar, proc.decodificar_cabecera):
            casos = [((b"hola",), "archivo_invalido", 415),
                     ((png(4, 4, formato="PCX"),), "formato_no_soportado", 415),
                     ((png(4, 4), proc.Limites(max_bytes=10)), "archivo_grande", 413),
                     ((png(10, 10), proc.Limites(max_lado=5)), "dimensiones_excedidas", 413),
                     ((b"",), "archivo_vacio", 400)]
            for args, codigo, http in casos:
                with self.subTest(fn=fn.__name__, codigo=codigo), self.assertRaises(proc.ErrorEntrada) as ctx:
                    fn(*args)
                self.assertEqual((ctx.exception.codigo, ctx.exception.http), (codigo, http))


class TestVisualizacion(unittest.TestCase):
    def vis(self, valores, ajuste):
        return proc.a_visualizacion(np.array(valores, np.float32), proc.construir_especificacion("media", 3, None, ajuste))

    def test_ajustes(self):
        np.testing.assert_array_equal(self.vis([[-3.4, 2.5, 300.0]], "recorte"), [[0, 3, 255]])
        np.testing.assert_array_equal(self.vis([[-128.2, 0.0, 200.0]], "desplazar128"), [[0, 128, 255]])
        np.testing.assert_array_equal(self.vis([[10.0, 20.0, 30.0]], "normalizar"), [[0, 128, 255]])
        np.testing.assert_array_equal(self.vis([[5.0, 5.0]], "normalizar"), [[0, 0]])


class TestProcesar(unittest.TestCase):
    def test_media_usa_kernel_original(self):
        gpu = GPUFalsa()
        res, formato = proc.procesar(gpu, png(8, 6, "L"), proc.construir_especificacion("media", 3), 256)
        self.assertEqual((formato, res.ancho, res.alto, res.bloques), ("PNG", 8, 6, 1))
        self.assertTrue({"decodificacion", "cpu_a_gpu", "kernel", "gpu_a_cpu", "gpu_conjunto", "gpu_llamada",
                         "reserva_y_liberacion", "conversion"} <= set(res.tiempos))
        self.assertIsNone(gpu.ultima["mascara"])

    def test_laplaciano_5_usa_kernel_mascara(self):
        gpu = GPUFalsa()
        proc.procesar(gpu, png(8, 6, "L"), proc.construir_especificacion("laplaciano", 5), 256)
        self.assertEqual(gpu.ultima["filtro"], PERSONALIZADA)
        self.assertEqual(gpu.ultima["mascara"].shape, (5, 5))


class TestMotor(unittest.TestCase):
    def test_enviar(self):
        m = MotorGPU(fabrica=GPUFalsa).iniciar()
        valor, espera = m.enviar(lambda g: 42).result(timeout=5)
        self.assertEqual(valor, 42)
        self.assertGreaterEqual(espera, 0)
        m.detener()

    def test_sin_gpu_no_cae_a_cpu(self):
        m = MotorGPU(fabrica=sin_gpu).iniciar()
        self.assertFalse(m.estado()["gpu_disponible"])
        self.assertIn("sin GPU", m.estado()["error"])
        with self.assertRaises(GPUNoDisponible):
            m.enviar(lambda g: 1)
        m.detener()

    def test_cola_llena(self):
        m = MotorGPU(max_cola=1, fabrica=GPUFalsa).iniciar()
        ev = threading.Event()
        m.enviar(lambda g: ev.wait(5))
        for _ in range(200):
            if m.estado()["procesando"]:
                break
            time.sleep(0.01)
        f2 = m.enviar(lambda g: 2)
        with self.assertRaises(ColaLlena):
            m.enviar(lambda g: 3)
        ev.set()
        self.assertEqual(f2.result(timeout=5)[0], 2)
        m.detener()


class TestWeb(unittest.TestCase):
    def setUp(self):
        self.app = crear_app(motor=MotorGPU(fabrica=GPUFalsa), config=ConfigPrueba)
        self.c = self.app.test_client()

    def tearDown(self):
        self.app.extensions["motor"].detener()

    def post(self, datos, cliente=None, **campos):
        return (cliente or self.c).post("/api/procesar", data={"imagen": (io.BytesIO(datos), "x.png"), **campos},
                                        content_type="multipart/form-data")

    def test_salud_y_config(self):
        r = self.c.get("/api/salud")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["gpu_disponible"])
        self.assertIn("media", self.c.get("/api/config").get_json()["filtros"])

    def test_procesar(self):
        r = self.post(png(8, 8), filtro="media", k="3")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertRegex(j["id"], r"^[0-9a-f]{32}$")
        self.assertEqual(r.headers["X-Request-ID"], j["id"])
        g = self.c.get(j["url_resultado"])
        self.assertEqual((g.status_code, g.mimetype), (200, "image/png"))
        self.assertIn("kernel", j["tiempos_ms"])
        self.assertIsNone(j["url_f32"])

    def test_aislamiento(self):
        j1 = self.post(png(8, 8), filtro="media", k="3").get_json()
        j2 = self.post(png(12, 5), filtro="nitidez", k="3").get_json()
        self.assertNotEqual(j1["id"], j2["id"])
        for j, tam in ((j1, (8, 8)), (j2, (12, 5))):
            with Image.open(io.BytesIO(self.c.get(j["url_resultado"]).data)) as im:
                self.assertEqual(im.size, tam)

    def test_f32(self):
        j = self.post(png(6, 4, "L"), filtro="laplaciano", k="3", f32="1").get_json()
        self.assertIsNotNone(j["url_f32"])
        self.assertEqual(len(self.c.get(j["url_f32"]).data), 6 * 4 * 4)

    def test_errores_y_recuperacion(self):
        r = self.c.post("/api/procesar", data={"filtro": "media"}, content_type="multipart/form-data")
        self.assertEqual((r.status_code, r.get_json()["error"]["codigo"]), (400, "archivo_vacio"))
        casos = [((b"no es imagen",), {"filtro": "media"}, 415, "archivo_invalido"),
                 ((png(8, 8),), {"filtro": "x"}, 400, "filtro_desconocido"),
                 ((png(8, 8),), {"filtro": "media", "k": "4"}, 400, "mascara_no_permitida"),
                 ((png(100, 100),), {"filtro": "media"}, 413, "dimensiones_excedidas"),
                 ((png(8, 8),), {"filtro": "personalizada", "k": "3", "mascara": "no-json"}, 400, "mascara_invalida")]
        for args, campos, http, codigo in casos:
            with self.subTest(codigo=codigo):
                r = self.post(*args, **campos)
                self.assertEqual((r.status_code, r.get_json()["error"]["codigo"]), (http, codigo))
        self.assertEqual(self.post(b"\x00" * (2 * 1024 * 1024), filtro="media").status_code, 413)
        self.assertEqual(self.post(png(8, 8), filtro="media").status_code, 200)

    def test_resultado_inexistente(self):
        self.assertEqual(self.c.get("/api/resultados/noesid.png").status_code, 404)
        self.assertEqual(self.c.get("/api/resultados/" + "a" * 32 + ".png").status_code, 404)

    def test_sin_gpu(self):
        app2 = crear_app(motor=MotorGPU(fabrica=sin_gpu), config=ConfigPrueba)
        c2 = app2.test_client()
        self.assertEqual(c2.get("/api/salud").status_code, 503)
        r = self.post(png(8, 8), cliente=c2, filtro="media", k="3")
        self.assertEqual((r.status_code, r.get_json()["error"]["codigo"]), (503, "gpu_no_disponible"))
        app2.extensions["motor"].detener()


if __name__ == "__main__":
    unittest.main()
