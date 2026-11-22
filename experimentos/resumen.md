| experimento | modo | filtro | resolucion | k | hilos | concurrencia | n | kernel ms (mediana [IQR]) | servidor ms | cliente ms | errores |
|---|---|---|---|---|---|---|---|---|---|---|---|
| concurrencia | servicio-docker | media | 4000 | 9 | 256 | 1 | 5 | 14.560 [0.361] | 836.146 [67.688] | 910.934 [61.763] | 0 |
| concurrencia | servicio-docker | media | 4000 | 9 | 256 | 2 | 10 | 14.506 [0.146] | 1025.230 [505.953] | 1098.214 [546.976] | 0 |
| filtros | servicio-docker | laplaciano | 4000 | 3 | 256 | 1 | 5 | 7.168 [0.348] | 2023.644 [816.434] | 2115.953 [824.560] | 0 |
| filtros | servicio-docker | media | 4000 | 21 | 256 | 1 | 5 | 20.660 [0.408] | 1306.457 [353.108] | 1390.521 [396.578] | 0 |
| filtros | servicio-docker | media | 4000 | 3 | 256 | 1 | 5 | 7.227 [0.120] | 1276.613 [290.497] | 1340.445 [289.038] | 0 |
| filtros | servicio-docker | media | 4000 | 9 | 256 | 1 | 5 | 14.506 [0.117] | 791.051 [19.129] | 856.510 [16.256] | 0 |
| filtros | servicio-docker | nitidez | 4000 | 3 | 256 | 1 | 5 | 7.231 [0.138] | 1214.147 [605.474] | 1270.840 [673.377] | 0 |
| integracion | directo-anfitrion | media | 4000 | 9 | 256 | 1 | 5 | 7.361 [5.508] | 405.771 [21.835] | - | 0 |
| integracion | directo-contenedor | media | 4000 | 9 | 256 | 1 | 5 | 1.915 [0.033] | 942.887 [315.616] | - | 0 |
| resolucion | servicio-docker | media | 2000 | 9 | 256 | 1 | 5 | 0.565 [0.149] | 188.879 [175.927] | 241.938 [151.543] | 0 |
| resolucion | servicio-docker | media | 4000 | 9 | 256 | 1 | 5 | 14.468 [6.896] | 741.995 [232.137] | 770.285 [242.159] | 0 |
| resolucion | servicio-docker | media | 6000 | 9 | 256 | 1 | 5 | 32.828 [0.096] | 1972.935 [67.391] | 2057.790 [52.047] | 0 |

Primera ejecucion de cada configuracion (repeticion 0, excluida del analisis):

| experimento | modo | filtro | resolucion | k | kernel ms | servidor ms | cliente ms |
|---|---|---|---|---|---|---|---|
| resolucion | servicio-docker | media | 2000 | 9 | 4.1673 | 209.4359 | 268.3636 |
| resolucion | servicio-docker | media | 4000 | 9 | 1.9117 | 1025.5361 | 1082.5401 |
| resolucion | servicio-docker | media | 6000 | 9 | 32.9796 | 2121.3055 | 2209.1648 |
| filtros | servicio-docker | media | 4000 | 3 | 4.2019 | 850.5115 | 894.2842 |
| filtros | servicio-docker | media | 4000 | 9 | 14.609 | 939.9618 | 1004.0133 |
| filtros | servicio-docker | media | 4000 | 21 | 20.5549 | 819.712 | 884.9053 |
| filtros | servicio-docker | nitidez | 4000 | 3 | 7.5362 | 2228.6038 | 2298.9057 |
| filtros | servicio-docker | laplaciano | 4000 | 3 | 7.0796 | 1124.5364 | 1194.933 |
| concurrencia | servicio-docker | media | 4000 | 9 | 14.4138 | 1081.2184 | 1167.7944 |
| integracion | directo-anfitrion | media | 4000 | 9 | 4.7106 | 480.1443 |  |
| integracion | directo-contenedor | media | 4000 | 9 | 2.2314 | 1347.228 |  |
