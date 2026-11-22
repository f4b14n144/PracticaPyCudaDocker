import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

CLAVE = ("experimento", "modo", "filtro", "resolucion", "k", "hilos", "concurrencia")
METRICAS = ("t_espera_cola_ms", "t_decodificacion_ms", "t_cpu_a_gpu_ms", "t_kernel_ms", "t_gpu_a_cpu_ms",
            "t_gpu_conjunto_ms", "t_conversion_ms", "t_codificacion_png_ms", "t_servidor_total_ms", "t_cliente_ms")


def flotante(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def cuantil(datos, q):
    if len(datos) == 1:
        return datos[0]
    return statistics.quantiles(datos, n=4, method="inclusive")[{0.25: 0, 0.75: 2}[q]]


def estadisticos(valores):
    v = sorted(x for x in valores if x is not None)
    if not v:
        return None
    p25, p75 = cuantil(v, 0.25), cuantil(v, 0.75)
    return {"n": len(v), "mediana": statistics.median(v), "p25": p25, "p75": p75, "iqr": p75 - p25,
            "min": v[0], "max": v[-1], "desv": statistics.stdev(v) if len(v) > 1 else 0.0}


def exitosa(f):
    return str(f.get("estado_http", "")) in ("200", "") and not f.get("codigo_error")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--salida", default="experimentos/resumen")
    a = ap.parse_args()

    filas = []
    for ruta in a.csv:
        with open(ruta, newline="", encoding="utf-8") as f:
            filas += list(csv.DictReader(f))

    grupos, primeras, errores = defaultdict(list), [], defaultdict(list)
    rondas = defaultdict(dict)
    for f in filas:
        clave = tuple(f.get(c, "") for c in CLAVE)
        if str(f.get("repeticion")) == "0":
            primeras.append(f)
            continue
        if not exitosa(f):
            errores[clave].append(f"{f.get('id_solicitud', '')}:{f.get('estado_http', '')}:{f.get('codigo_error', '')}")
            continue
        grupos[clave].append(f)
        if f.get("ronda") and f.get("throughput_rps"):
            rondas[clave][f["ronda"]] = flotante(f["throughput_rps"])

    salida = Path(a.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    columnas = list(CLAVE) + ["metrica", "n", "mediana", "p25", "p75", "iqr", "min", "max", "desv", "unidad",
                              "errores", "detalle_errores"]
    lineas_md = ["| " + " | ".join(CLAVE) + " | n | kernel ms (mediana [IQR]) | servidor ms | cliente ms | errores |",
                 "|" + "---|" * (len(CLAVE) + 5)]
    with open(salida.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=columnas)
        w.writeheader()
        for clave in sorted(set(grupos) | set(errores)):
            base = dict(zip(CLAVE, clave))
            err = errores.get(clave, [])
            celdas = {}
            for m in METRICAS:
                e = estadisticos([flotante(f.get(m)) for f in grupos.get(clave, [])])
                if e:
                    w.writerow({**base, "metrica": m, **{k: round(v, 4) for k, v in e.items()}, "unidad": "ms",
                                "errores": len(err), "detalle_errores": ";".join(err)})
                    celdas[m] = f"{e['mediana']:.3f} [{e['iqr']:.3f}]"
            if clave in rondas:
                e = estadisticos(list(rondas[clave].values()))
                w.writerow({**base, "metrica": "throughput_ronda", **{k: round(v, 4) for k, v in e.items()},
                            "unidad": "solicitudes/s", "errores": len(err), "detalle_errores": ";".join(err)})
            n = len(grupos.get(clave, []))
            lineas_md.append("| " + " | ".join(clave) + f" | {n} | {celdas.get('t_kernel_ms', '-')} | "
                             f"{celdas.get('t_servidor_total_ms', '-')} | {celdas.get('t_cliente_ms', '-')} | {len(err)} |")

    lineas_md += ["", "Primera ejecucion de cada configuracion (repeticion 0, excluida del analisis):", "",
                  "| experimento | modo | filtro | resolucion | k | kernel ms | servidor ms | cliente ms |", "|---|---|---|---|---|---|---|---|"]
    for f in primeras:
        lineas_md.append(f"| {f.get('experimento')} | {f.get('modo')} | {f.get('filtro')} | {f.get('resolucion')} | "
                         f"{f.get('k')} | {f.get('t_kernel_ms')} | {f.get('t_servidor_total_ms')} | {f.get('t_cliente_ms', '')} |")
    salida.with_suffix(".md").write_text("\n".join(lineas_md) + "\n", encoding="utf-8")
    print(f"{salida.with_suffix('.csv')} y {salida.with_suffix('.md')}")
    graficas(grupos, salida.parent)


def graficas(grupos, carpeta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no instalado: se omiten las graficas")
        return
    etapas = ("t_decodificacion_ms", "t_cpu_a_gpu_ms", "t_kernel_ms", "t_gpu_a_cpu_ms",
              "t_conversion_ms", "t_codificacion_png_ms")
    for experimento in ("resolucion", "filtros"):
        claves = [c for c in sorted(grupos) if c[0] == experimento and c[1].startswith("servicio")]
        if not claves:
            continue
        etiquetas = [f"{c[2]} k{c[4]} {c[3]}" for c in claves]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        abajo = [0.0] * len(claves)
        for e in etapas:
            med = [statistics.median([flotante(f.get(e)) or 0 for f in grupos[c]]) for c in claves]
            ax.bar(etiquetas, med, bottom=abajo, label=e[2:-3])
            abajo = [x + y for x, y in zip(abajo, med)]
        cliente = [statistics.median([flotante(f.get("t_cliente_ms")) or 0 for f in grupos[c]]) for c in claves]
        ax.plot(etiquetas, cliente, "ko-", label="cliente total")
        ax.set_ylabel("ms (mediana)")
        ax.set_title(f"Experimento {experimento}: distribucion del tiempo")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(carpeta / f"grafica_{experimento}.png", dpi=150)
        plt.close(fig)
        print(f"grafica_{experimento}.png")


if __name__ == "__main__":
    main()
