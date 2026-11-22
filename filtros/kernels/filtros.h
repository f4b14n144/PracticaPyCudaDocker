#ifndef FILTROS_H
#define FILTROS_H

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

enum TipoFiltro {
    FILTRO_MEDIA = 1,
    FILTRO_NITIDEZ = 2,
    FILTRO_LAPLACIANO = 3
};

HD inline int pixelConCeros(const unsigned char* entrada, int ancho, int alto, int x, int y) {
    if (x < 0 || x >= ancho || y < 0 || y >= alto) return 0;
    return entrada[y * ancho + x];
}

HD inline float calcularMedia(const unsigned char* entrada, int ancho, int alto, int x, int y, int ksize) {
    int radio = ksize / 2;
    int suma = 0;
    for (int ky = -radio; ky <= radio; ky++)
        for (int kx = -radio; kx <= radio; kx++)
            suma += pixelConCeros(entrada, ancho, alto, x + kx, y + ky);
    return (float)suma / (float)(ksize * ksize);
}

HD inline float calcularNitidez(const unsigned char* entrada, int ancho, int alto, int x, int y) {
    const int mascara[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
    int suma = 0;
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++)
            suma += mascara[ky + 1][kx + 1] * pixelConCeros(entrada, ancho, alto, x + kx, y + ky);
    return (float)suma;
}

HD inline float calcularLaplaciano(const unsigned char* entrada, int ancho, int alto, int x, int y) {
    const int mascara[3][3] = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};
    int suma = 0;
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++)
            suma += mascara[ky + 1][kx + 1] * pixelConCeros(entrada, ancho, alto, x + kx, y + ky);
    return (float)suma;
}

struct TiemposCUDA {
    double h2d_ms;
    double kernel_ms;
    double d2h_ms;
    double total_ms;
    int bloques;
};

void aplicarFiltroMedia(const unsigned char* entrada, float* salida, int ancho, int alto, int ksize);
void aplicarFiltroNitidez(const unsigned char* entrada, float* salida, int ancho, int alto);
void aplicarFiltroLaplaciano(const unsigned char* entrada, float* salida, int ancho, int alto);

unsigned char convertirAVisualizacion(float valor, TipoFiltro filtro);

void filtroMediaCUDA(const unsigned char* h_entrada, float* h_salida, int ancho, int alto, int ksize,
                     int hilosPorBloque, TiemposCUDA* tiempos);
void filtroNitidezCUDA(const unsigned char* h_entrada, float* h_salida, int ancho, int alto,
                       int hilosPorBloque, TiemposCUDA* tiempos);
void filtroLaplacianoCUDA(const unsigned char* h_entrada, float* h_salida, int ancho, int alto,
                          int hilosPorBloque, TiemposCUDA* tiempos);

int maxHilosPorBloqueCUDA();

#endif
