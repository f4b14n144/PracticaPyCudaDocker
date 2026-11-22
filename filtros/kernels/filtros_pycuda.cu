#include "filtros.h"

extern "C" __global__ void kernelFiltroMedia(const unsigned char* entrada, float* salida,
                                             int ancho, int alto, int ksize) {
    int indice = blockIdx.x * blockDim.x + threadIdx.x;
    if (indice >= ancho * alto) return;
    int y = indice / ancho;
    int x = indice % ancho;
    salida[indice] = calcularMedia(entrada, ancho, alto, x, y, ksize);
}

extern "C" __global__ void kernelFiltroNitidez(const unsigned char* entrada, float* salida,
                                               int ancho, int alto) {
    int indice = blockIdx.x * blockDim.x + threadIdx.x;
    if (indice >= ancho * alto) return;
    int y = indice / ancho;
    int x = indice % ancho;
    salida[indice] = calcularNitidez(entrada, ancho, alto, x, y);
}

extern "C" __global__ void kernelFiltroLaplaciano(const unsigned char* entrada, float* salida,
                                                  int ancho, int alto) {
    int indice = blockIdx.x * blockDim.x + threadIdx.x;
    if (indice >= ancho * alto) return;
    int y = indice / ancho;
    int x = indice % ancho;
    salida[indice] = calcularLaplaciano(entrada, ancho, alto, x, y);
}

extern "C" __global__ void kernelFiltroMascara(const unsigned char* entrada, float* salida,
                                               int ancho, int alto,
                                               const float* mascara, int ksize) {
    int indice = blockIdx.x * blockDim.x + threadIdx.x;
    if (indice >= ancho * alto) return;
    int y = indice / ancho;
    int x = indice % ancho;
    int radio = ksize / 2;
    float suma = 0.0f;
    for (int ky = -radio; ky <= radio; ky++)
        for (int kx = -radio; kx <= radio; kx++)
            suma += mascara[(ky + radio) * ksize + (kx + radio)] *
                    (float)pixelConCeros(entrada, ancho, alto, x + kx, y + ky);
    salida[indice] = suma;
}
