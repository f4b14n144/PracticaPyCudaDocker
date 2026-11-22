# PracticaPyCudaDocker

Practica 03 de Computacion Paralela. Es la practica 02 ([PracticaPyCuda](https://github.com/f4b14n144/PracticaPyCuda), commit `6aa24a3ffe8c218c618c26668d69a2e12b8e49ab`) pero convertida en un servicio web con Flask + PyCUDA y metida en Docker.

Navegador -> Flask (gunicorn) -> cola -> hilo GPU -> PyCUDA -> kernels -> PNG -> navegador

Los kernels son los mismos de la practica 02 (`filtros/kernels`), mismos filtros (media 3/9/21, nitidez 3x3, laplaciano 3x3), misma extension con ceros, un pixel por hilo y bloques solo en X. Se agrego `kernelFiltroMascara` para mascaras personalizadas y nitidez/laplaciano de mas de 3x3.

## Que hay

- `filtros/` kernels y `FiltrosGPU` (lo de la practica 02, se agrego el free de memoria en un finally)
- `servicio/procesamiento.py` validacion y procesamiento sin nada de HTTP, lo usan el servicio y `herramientas/directo.py`
- `servicio/motor.py` un solo hilo crea el contexto CUDA y atiende todo en orden (FIFO), cola de `MAX_COLA` y si se llena 503
- `servicio/web.py` la API
- `servicio/static/` la interfaz
- `herramientas/` validar, experimentos, directo, resumen y entorno
- `experimentos/` csv, figuras y evidencia

## Docker

Necesita en el host: GPU NVIDIA, driver compatible con CUDA 13.x, Docker y NVIDIA Container Toolkit (en Windows ya viene con Docker Desktop + WSL2). El driver va en el host, la imagen trae CUDA 13.4.3 (nvcc, porque PyCUDA compila los kernels al arrancar), Python 3.12 de Ubuntu 24.04 y las versiones de `requirements.txt`.

```
docker run --rm --gpus all nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04 nvidia-smi
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t convolucion-pycuda:0.3 .
docker run -d --name convolucion --gpus all -p 5000:5000 convolucion-pycuda:0.3
curl http://localhost:5000/api/salud
```

En PowerShell `$(git rev-parse HEAD)` funciona igual. Luego abrir http://localhost:5000 (o la IP del host desde otra maquina).

Variables (`-e`): `MAX_MB` 100, `MAX_LADO` 16384, `MAX_MPIX` 100, `MAX_COLA` 4, `TIMEOUT_S` 120, `TTL_RESULTADOS_S` 900, `HILOS_HTTP` 8.

## API

`POST /api/procesar` multipart/form-data

| campo | | |
|---|---|---|
| imagen | si | PNG, JPEG, WEBP, BMP, TIFF, GIF, PPM, TGA, ICO (se revisa el formato real) |
| filtro | si | media, nitidez, laplaciano, personalizada |
| k | no (3) | impar de 3 a 31 |
| mascara | solo personalizada | json kxk, ej `[[0,1,0],[1,-4,1],[0,1,0]]` |
| hilos | no (256) | 32, 64, 128, 256, 512, 1024 |
| ajuste | no | recorte, desplazar128, normalizar |
| f32 | no | 1 para guardar la salida float32 cruda |

```
curl -F imagen=@Imagenes/entrada/4000.png -F filtro=media -F k=9 http://localhost:5000/api/procesar
```

Responde 200 con `id`, `kernel`, `bloques`, `tiempos_ms` (espera_cola, decodificacion, reserva_y_liberacion, cpu_a_gpu, kernel, gpu_a_cpu, conversion, codificacion_png, escritura, servidor_total), `url_resultado` y `url_f32`. El resultado se baja de `GET /api/resultados/<id>.png` (`?descargar=1`) y se borra a los `TTL_RESULTADOS_S`.

Errores: `{"error": {"codigo", "mensaje"}, "id"}`

- 400 archivo_vacio, filtro_desconocido, mascara_no_permitida, mascara_invalida, parametro_invalido
- 413 archivo_grande, dimensiones_excedidas
- 415 archivo_invalido, formato_no_soportado
- 503 cola_llena, gpu_no_disponible (no se pasa a CPU)
- 507 memoria_gpu, 500 error_gpu, 504 tiempo_agotado

Tambien `GET /api/salud` (gpu, cola, tiempos de inicio, commit) y `GET /api/config`.

## Tiempos

- kernel con eventos CUDA, transferencias con perf_counter + synchronize, memoria paginable
- reserva y liberacion aparte, no entra en gpu_conjunto
- contexto y compilacion se miden una vez al arrancar (`/api/salud`), no entran en las mediciones
- servidor_total desde que llega la solicitud hasta escribir el PNG, incluye espera en cola
- cliente desde que se envia hasta bajar el PNG completo

## Validacion y experimentos

Las imagenes son las mismas de las practicas 01 y 02 (`Imagenes/entrada`, con `SHA256SUMS.txt`). Las referencias .f32 de C/C++ y CUDA C salen de `exportar_salidas.exe` de la practica 01 ([PracticaCUDAvsCPU](https://github.com/Jellz01/PracticaCUDAvsCPU), commit `1a721f59d8d0ee83d39a62f8551af53b46bc7c03`), no se suben por el tamano.

```
docker run -d --name convolucion --gpus all -p 5000:5000 -v <ruta>\Imagenes\entrada:/datos/entrada:ro -v <ruta>\referencias:/datos/referencias:ro -v <ruta>\experimentos:/datos/experimentos convolucion-pycuda:0.3

docker exec convolucion python3 -m unittest discover -s pruebas -t .
docker exec convolucion python3 -m herramientas.validar --entrada /datos/entrada --referencias /datos/referencias --figuras /datos/experimentos/figuras --salida /datos/experimentos/validacion_servicio.csv

python -m herramientas.experimentos --url http://localhost:5000 --reps 5 --salida experimentos/servicio.csv
python -m herramientas.directo Imagenes/entrada/4000.png --filtro media --k 9 --repeticiones 5 --modo directo-anfitrion --csv experimentos/directo_anfitrion.csv
docker exec convolucion python3 -m herramientas.directo /datos/entrada/4000.png --filtro media --k 9 --repeticiones 5 --modo directo-contenedor --csv /datos/experimentos/directo_contenedor.csv
python -m herramientas.resumen experimentos/servicio.csv experimentos/directo_anfitrion.csv experimentos/directo_contenedor.csv --salida experimentos/resumen
python -m herramientas.entorno --salida experimentos/evidencia/entorno_anfitrion.json
docker exec convolucion python3 -m herramientas.entorno --salida /datos/experimentos/evidencia/entorno_contenedor.json
```

`validar` hace las pruebas de la practica 02 (matriz 3x3 a mano, 1x1, 7x5, 513x1, 1x257 y 1001x777 con 128/256/512 hilos, conversion 0-255, imagenes 2000/4000/6000 contra C/C++ y CUDA C) pero pasando por el servicio, ademas compara servicio vs PyCUDA directo, entradas invalidas, aislamiento de dos solicitudes y la cola llena.

En los experimentos la repeticion 0 es calentamiento y no se cuenta, 256 hilos, 5 repeticiones (5 rondas en concurrencia).

Fabian Mendoza y Joseph Wellesley

## Resultados

Corrido en una laptop ASUS ROG Strix G16 (Core Ultra 9 275HX, 32 GB, RTX 5060 Laptop 8 GB, driver 621.14, Windows 11 + Docker Desktop 4.99 con WSL2). La imagen se construyo desde un clone limpio del commit `bf2fc7d3b14099e4b39a34df6c4e4d366649844e` (imagen `sha256:998a2b32cdb4...`) y todas las mediciones son con ese commit, el siguiente commit solo agrega `experimentos/`.

- `experimentos/validacion_servicio.csv` 231 pruebas, 0 fallos, error 0 contra C/C++, CUDA C y PyCUDA directo
- `experimentos/servicio.csv`, `directo_anfitrion.csv`, `directo_contenedor.csv` mediciones crudas
- `experimentos/resumen.csv` y `resumen.md` medianas e IQR
- `experimentos/figuras/` original, salidas y recortes
- `experimentos/evidencia/` build, imagen, nvidia-smi, salud, logs del servicio y entorno
