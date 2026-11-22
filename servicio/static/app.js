"use strict";

const $ = (s) => document.querySelector(s);
const estado = {
  cfg: null,
  archivo: null,
  urlOriginal: null,
  filtro: "media",
  k: 3,
  matriz: [],
  procesando: false,
};

const DESCRIPCIONES = {
  media: "Promedio k×k (1/k² en cada celda). Kernel original de la práctica 02 para cualquier k.",
  nitidez: "Identidad − laplaciano. 3×3 = [0 −1 0; −1 5 −1; 0 −1 0] (kernel original); k > 3 generaliza la cruz.",
  laplaciano: "Segunda derivada en cruz. 3×3 = [0 1 0; 1 −4 1; 0 1 0] (kernel original); k > 3 generaliza la cruz.",
  personalizada: "Coeficientes libres (admite fracciones como 1/16). Se ejecuta con kernelFiltroMascara.",
};

const ETAPAS = [
  ["espera_cola", "Espera en cola", "#9aa0a6", "Desde que se encola hasta que el hilo GPU la toma"],
  ["decodificacion", "Decodificación", "#c9a227", "Lectura de la imagen y conversión a grises (CPU)"],
  ["reserva_y_liberacion", "Reserva/liberación", "#8e6bbf", "cuda.mem_alloc + free (fuera del tiempo conjunto)"],
  ["cpu_a_gpu", "CPU → GPU", "#3d8bd9", "memcpy_htod, memoria paginable, perf_counter + synchronize"],
  ["kernel", "Kernel", "#76b900", "Eventos CUDA alrededor del lanzamiento"],
  ["gpu_a_cpu", "GPU → CPU", "#2fb3a8", "memcpy_dtoh, memoria paginable, perf_counter + synchronize"],
  ["conversion", "Conversión 0–255", "#d9822b", "float32 → uint8 según el ajuste"],
  ["codificacion_png", "Codificación PNG", "#d6564b", "Pillow, compress_level=1"],
  ["escritura", "Escritura", "#b0466e", "Guardar el PNG (y .f32) con nombre único"],
];

const ceros = (k) => Array.from({ length: k }, () => Array(k).fill(0));
function cruz(k, centro, brazo) {
  const m = ceros(k), r = (k - 1) / 2;
  for (let i = 0; i < k; i++) { m[r][i] = brazo; m[i][r] = brazo; }
  m[r][r] = centro;
  return m;
}
function incrustar(k, base) {
  const m = ceros(k), d = (k - base.length) / 2;
  base.forEach((fila, i) => fila.forEach((v, j) => { m[i + d][j + d] = v; }));
  return m;
}
function predefinida(filtro, k) {
  const r = (k - 1) / 2;
  if (filtro === "media") return Array.from({ length: k }, () => Array(k).fill(1 / (k * k)));
  if (filtro === "laplaciano") return cruz(k, -4 * r, 1);
  if (filtro === "nitidez") return cruz(k, 1 + 4 * r, -1);
  return null;
}
function plantilla(nombre, k) {
  const r = (k - 1) / 2;
  switch (nombre) {
    case "identidad": { const m = ceros(k); m[r][r] = 1; return m; }
    case "gauss": {
      const s = Math.max(k / 6, 0.5);
      const m = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) =>
        Math.exp(-((i - r) ** 2 + (j - r) ** 2) / (2 * s * s))));
      const suma = m.flat().reduce((a, b) => a + b, 0);
      return m.map((f) => f.map((v) => v / suma));
    }
    case "sobelx": return incrustar(k, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]);
    case "sobely": return incrustar(k, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]);
    case "relieve": return incrustar(k, [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]);
    case "bordes": { const m = Array.from({ length: k }, () => Array(k).fill(-1)); m[r][r] = k * k - 1; return m; }
    default: return ceros(k);
  }
}
function redimensionar(m, k) {
  const n = ceros(k), d = (k - m.length) / 2;
  m.forEach((fila, i) => fila.forEach((v, j) => {
    const a = i + d, b = j + d;
    if (a >= 0 && a < k && b >= 0 && b < k) n[a][b] = v;
  }));
  return n;
}

function formato(v) {
  if (!Number.isFinite(v)) return "";
  if (Number.isInteger(v)) return String(v);
  const t = Math.abs(v) >= 1 ? v.toFixed(3) : v.toPrecision(3);
  return String(parseFloat(t));
}
function leerNumero(texto) {
  const s = String(texto).trim().replace(",", ".");
  if (s === "" || s === "-") return 0;
  const frac = s.match(/^(-?\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)$/);
  if (frac) return Number(frac[2]) === 0 ? NaN : Number(frac[1]) / Number(frac[2]);
  return /^-?(\d+\.?\d*|\.\d+)(e-?\d+)?$/i.test(s) ? Number(s) : NaN;
}

function kernelQueSeUsa() {
  if (estado.filtro === "media") return "kernelFiltroMedia";
  if (estado.filtro !== "personalizada" && estado.k === 3)
    return estado.filtro === "nitidez" ? "kernelFiltroNitidez" : "kernelFiltroLaplaciano";
  return "kernelFiltroMascara";
}

function pintarCelda(input, v, maxAbs) {
  const alfa = maxAbs > 0 ? Math.min(Math.abs(v) / maxAbs, 1) * 0.55 : 0;
  input.style.setProperty("--color", v < 0 ? "var(--neg)" : "var(--pos)");
  input.style.setProperty("--alfa", Number.isFinite(v) ? alfa.toFixed(3) : 0);
  input.style.borderColor = Number.isFinite(v) ? "" : "rgb(var(--neg))";
}

function actualizarPieMascara() {
  const plano = estado.matriz.flat();
  const maxAbs = Math.max(...plano.filter(Number.isFinite).map(Math.abs), 0);
  document.querySelectorAll("#mascara input").forEach((inp, idx) => pintarCelda(inp, plano[idx], maxAbs));
  const suma = plano.reduce((a, b) => a + b, 0);
  $("#sumaMascara").textContent = Number.isFinite(suma) ? `Σ = ${formato(suma)}` : "Σ = celda inválida";
  $("#kernelUsado").textContent = kernelQueSeUsa();
}

function dibujarMascara() {
  const k = estado.k, cont = $("#mascara");
  const celda = k <= 5 ? 56 : k <= 9 ? 46 : k <= 15 ? 40 : 34;
  cont.style.gridTemplateColumns = `repeat(${k}, auto)`;
  cont.style.setProperty("--celda", celda + "px");
  cont.style.setProperty("--letra", (k <= 9 ? 12 : 10) + "px");
  const frag = document.createDocumentFragment();
  const r = (k - 1) / 2;
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
    const inp = document.createElement("input");
    inp.type = "text";
    inp.inputMode = "decimal";
    inp.value = formato(estado.matriz[i][j]);
    inp.title = `fila ${i - r}, columna ${j - r}`;
    inp.dataset.i = i; inp.dataset.j = j;
    if (i === r && j === r) inp.classList.add("centro");
    frag.appendChild(inp);
  }
  cont.replaceChildren(frag);
  actualizarPieMascara();
}

$("#mascara").addEventListener("input", (e) => {
  const inp = e.target;
  estado.matriz[+inp.dataset.i][+inp.dataset.j] = leerNumero(inp.value);
  if (estado.filtro !== "personalizada") elegirFiltro("personalizada", { conservar: true });
  actualizarPieMascara();
});
$("#mascara").addEventListener("focusout", (e) => {
  const v = estado.matriz[+e.target.dataset.i]?.[+e.target.dataset.j];
  if (Number.isFinite(v)) e.target.value = formato(v);
});

function elegirFiltro(filtro, { conservar = false } = {}) {
  estado.filtro = filtro;
  document.querySelectorAll("#filtros button").forEach((b) =>
    b.classList.toggle("activo", b.dataset.filtro === filtro));
  $("#descFiltro").textContent = DESCRIPCIONES[filtro];
  $("#predefinidas").hidden = filtro !== "personalizada";
  if (!conservar) {
    estado.matriz = filtro === "personalizada" ? redimensionar(estado.matriz, estado.k) : predefinida(filtro, estado.k);
    dibujarMascara();
  } else {
    actualizarPieMascara();
  }
  dibujarRapidos();
}

function elegirK(k) {
  estado.k = k;
  $("#k").value = k;
  $("#kValor").textContent = `${k} × ${k}`;
  estado.matriz = estado.filtro === "personalizada" ? redimensionar(estado.matriz, k) : predefinida(estado.filtro, k);
  dibujarMascara();
  dibujarRapidos();
}

function dibujarRapidos() {
  const tam = [3, 5, 7, 9, 15, 21, 31].filter((v) => !estado.cfg || v <= estado.cfg.k_max);
  $("#kRapidos").replaceChildren(...tam.map((v) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = `${v}×${v}`;
    b.classList.toggle("activo", v === estado.k);
    b.onclick = () => elegirK(v);
    return b;
  }));
}

document.querySelectorAll("#filtros button").forEach((b) => b.addEventListener("click", () => elegirFiltro(b.dataset.filtro)));
$("#k").addEventListener("input", (e) => elegirK(+e.target.value));
document.querySelectorAll("#predefinidas button").forEach((b) => b.addEventListener("click", () => {
  estado.matriz = plantilla(b.dataset.pre, estado.k);
  if (b.dataset.pre === "sobelx" || b.dataset.pre === "sobely" || b.dataset.pre === "relieve") $("#ajuste").value = "desplazar128";
  dibujarMascara();
}));

function mostrarMensaje(html, ok = false) {
  const m = $("#mensaje");
  m.innerHTML = html;
  m.classList.toggle("ok", ok);
  m.hidden = !html;
}
const escapar = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const mib = (b) => (b / 1048576).toFixed(b < 1048576 ? 2 : 1) + " MiB";

function elegirArchivo(f) {
  if (!f) return;
  mostrarMensaje("");
  if (estado.cfg && f.size > estado.cfg.max_mb * 1048576) {
    mostrarMensaje(`El archivo pesa ${mib(f.size)}; el límite es ${estado.cfg.max_mb} MiB.`);
  }
  estado.archivo = f;
  if (estado.urlOriginal) URL.revokeObjectURL(estado.urlOriginal);
  estado.urlOriginal = URL.createObjectURL(f);
  const img = new Image();
  img.onload = () => {
    $("#dimOriginal").textContent = `${img.naturalWidth}×${img.naturalHeight}`;
    $("#infoArchivo").innerHTML = `<b>${escapar(f.name)}</b><span>${mib(f.size)}</span><span>${img.naturalWidth}×${img.naturalHeight}</span>`;
  };
  img.onerror = () => {
    $("#dimOriginal").textContent = "";
    $("#infoArchivo").innerHTML = `<b>${escapar(f.name)}</b><span>${mib(f.size)}</span><span>vista previa no disponible</span>`;
  };
  img.src = estado.urlOriginal;
  $("#infoArchivo").hidden = false;
  ponerFoto("#fotoOriginal", img);
  actualizarBoton();
}

function ponerFoto(sel, img) {
  const caja = $(sel);
  caja.classList.remove("cargando", "con-zoom");
  img.alt = "";
  img.addEventListener("click", () => {
    img.classList.toggle("zoom");
    caja.classList.toggle("con-zoom", img.classList.contains("zoom"));
  });
  caja.replaceChildren(img);
}

const zona = $("#zona");
$("#archivo").addEventListener("change", (e) => elegirArchivo(e.target.files[0]));
zona.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#archivo").click(); } });
["dragenter", "dragover"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.add("encima"); }));
["dragleave", "drop"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.remove("encima"); }));
zona.addEventListener("drop", (e) => elegirArchivo(e.dataTransfer.files[0]));

function actualizarBoton() {
  const b = $("#procesar");
  b.disabled = estado.procesando || !estado.archivo;
  b.classList.toggle("procesando", estado.procesando);
  b.querySelector(".etiqueta").textContent = estado.procesando ? "Procesando..." : "Procesar";
}

$("#formulario").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (estado.procesando || !estado.archivo) return;
  if (estado.filtro === "personalizada" && estado.matriz.flat().some((v) => !Number.isFinite(v))) {
    mostrarMensaje("La máscara tiene celdas inválidas (marcadas en rojo).");
    return;
  }
  const datos = new FormData();
  datos.append("imagen", estado.archivo);
  datos.append("filtro", estado.filtro);
  datos.append("k", estado.k);
  datos.append("hilos", $("#hilos").value);
  if ($("#ajuste").value) datos.append("ajuste", $("#ajuste").value);
  if ($("#f32").checked) datos.append("f32", "1");
  if (estado.filtro === "personalizada") datos.append("mascara", JSON.stringify(estado.matriz));

  estado.procesando = true;
  actualizarBoton();
  mostrarMensaje("");
  $("#fotoResultado").classList.add("cargando");
  $("#fotoResultado").replaceChildren();

  const t0 = performance.now();
  try {
    const resp = await fetch("/api/procesar", { method: "POST", body: datos });
    let cuerpo;
    try { cuerpo = await resp.json(); } catch { cuerpo = null; }
    if (!resp.ok) {
      const err = cuerpo?.error || { codigo: `http_${resp.status}`, mensaje: resp.statusText || "error del servidor" };
      throw Object.assign(new Error(err.mensaje), { codigo: err.codigo, http: resp.status, id: cuerpo?.id });
    }
    const respImg = await fetch(cuerpo.url_resultado);
    if (!respImg.ok) throw Object.assign(new Error("no se pudo descargar el resultado"), { codigo: "descarga", http: respImg.status });
    const blob = await respImg.blob();
    const clienteMs = performance.now() - t0;
    mostrarResultado(cuerpo, blob, clienteMs);
  } catch (err) {
    $("#fotoResultado").classList.remove("cargando");
    $("#fotoResultado").innerHTML = '<span class="vacio">Sin resultado</span>';
    const extra = err.id ? `<br><code>id ${escapar(err.id)}</code>` : "";
    const cod = err.codigo ? `<code>${escapar(err.http || "")} ${escapar(err.codigo)}</code> · ` : "";
    mostrarMensaje(err.codigo ? `${cod}${escapar(err.message)}${extra}` : `No se pudo contactar al servidor: ${escapar(err.message)}`);
  } finally {
    estado.procesando = false;
    actualizarBoton();
    consultarSalud();
  }
});

let urlResultado = null;
function mostrarResultado(r, blob, clienteMs, { historial = true } = {}) {
  if (urlResultado) URL.revokeObjectURL(urlResultado);
  urlResultado = URL.createObjectURL(blob);
  const img = new Image();
  img.src = urlResultado;
  ponerFoto("#fotoResultado", img);
  $("#dimResultado").textContent = `${r.ancho}×${r.alto} · ${r.filtro} ${r.k}×${r.k}`;
  const d = $("#descargar");
  d.href = r.url_resultado + "?descargar=1";
  d.download = `${r.filtro}_k${r.k}_${r.id}.png`;
  d.hidden = false;
  const f = $("#descargarF32");
  f.hidden = !r.url_f32;
  if (r.url_f32) { f.href = r.url_f32 + "?descargar=1"; f.download = `${r.filtro}_k${r.k}_${r.id}.f32`; }
  mostrarTiempos(r, clienteMs);
  if (historial) agregarHistorial(r, clienteMs);
}

const ms = (v) => (v >= 100 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3));

function mostrarTiempos(r, clienteMs) {
  const t = r.tiempos_ms;
  $("#tiempos").hidden = false;
  $("#idSolicitud").textContent = `id ${r.id} · ${r.kernel} · ${r.bloques} bloques × ${r.hilos_por_bloque} hilos`;
  $("#kpiKernel").textContent = ms(t.kernel) + " ms";
  $("#kpiServidor").textContent = ms(t.servidor_total) + " ms";
  $("#kpiCliente").textContent = ms(clienteMs) + " ms";

  const sumaEtapas = ETAPAS.reduce((a, [k]) => a + (t[k] || 0), 0);
  const red = Math.max(clienteMs - sumaEtapas, 0);
  const filas = [...ETAPAS.map(([k, n, c, a]) => [n, t[k] || 0, c, a]),
    ["Red, HTTP y descarga", red, "#5f6b7a", "Cliente total − etapas del servidor (incluye subida y descarga)"]];

  $("#barraTiempos").replaceChildren(...filas.map(([n, v, c]) => {
    const div = document.createElement("div");
    div.style.width = (100 * v / clienteMs) + "%";
    div.style.background = c;
    div.title = `${n}: ${ms(v)} ms`;
    return div;
  }));

  const totales = [
    ["GPU conjunto", t.gpu_conjunto, "CPU→GPU + kernel + GPU→CPU (sin reserva)"],
    ["Servidor total", t.servidor_total, r.alcance_tiempos.servidor_total],
    ["Cliente total", clienteMs, "performance.now(): envío → PNG descargado"],
  ];
  $("#tablaTiempos").innerHTML =
    filas.map(([n, v, c, a]) => `<tr><td><span class="muestra" style="background:${c}"></span>${n}</td><td>${ms(v)}</td><td>${escapar(a)}</td></tr>`).join("") +
    totales.map(([n, v, a]) => `<tr><td><b>${n}</b></td><td><b>${ms(v)}</b></td><td>${escapar(a)}</td></tr>`).join("");
}

function agregarHistorial(r, clienteMs) {
  $("#historialCaja").hidden = false;
  const tr = document.createElement("tr");
  const hora = new Date().toLocaleTimeString();
  tr.innerHTML = `<td>${hora}</td><td>${escapar(r.filtro)}</td><td>${r.k}</td><td>${r.ancho}×${r.alto}</td><td>${r.hilos_por_bloque}</td>` +
    `<td>${ms(r.tiempos_ms.kernel)}</td><td>${ms(r.tiempos_ms.servidor_total)}</td><td>${ms(clienteMs)}</td><td>${r.id.slice(0, 8)}</td>`;
  tr.title = "Ver este resultado";
  tr.onclick = async () => {
    const resp = await fetch(r.url_resultado);
    if (!resp.ok) { mostrarMensaje("Ese resultado ya expiró en el servidor."); return; }
    mostrarResultado(r, await resp.blob(), clienteMs, { historial: false });
  };
  $("#historial").prepend(tr);
}

async function consultarSalud() {
  const caja = $("#estadoGpu"), txt = caja.querySelector(".texto");
  try {
    const s = await (await fetch("/api/salud")).json();
    caja.classList.toggle("ok", s.gpu_disponible);
    caja.classList.toggle("mal", !s.gpu_disponible);
    txt.textContent = s.gpu_disponible
      ? `${s.dispositivo} · ${s.memoria_mib} MiB · cola ${s.en_cola}/${s.max_cola}${s.procesando ? " · ocupada" : ""}`
      : `GPU no disponible: ${s.error}`;
    caja.title = s.gpu_disponible
      ? `Contexto ${s.inicio_contexto_ms?.toFixed(1)} ms · compilación ${s.inicio_compilacion_ms?.toFixed(1)} ms · ${s.atendidas} solicitudes`
      : s.error;
  } catch {
    caja.classList.remove("ok"); caja.classList.add("mal");
    txt.textContent = "Servidor no responde";
  }
}

async function iniciar() {
  try {
    estado.cfg = await (await fetch("/api/config")).json();
    $("#k").min = estado.cfg.k_min;
    $("#k").max = estado.cfg.k_max;
    $("#hilos").replaceChildren(...estado.cfg.hilos_permitidos.map((h) => new Option(h, h, h === 256, h === 256)));
    $("#zonaLimites").textContent = `${estado.cfg.formatos.join(", ")} · máx. ${estado.cfg.max_mb} MiB · ${estado.cfg.max_lado}px por lado`;
  } catch {
    mostrarMensaje("No se pudo leer la configuración del servidor.");
  }
  elegirK(3);
  elegirFiltro("media");
  consultarSalud();
  setInterval(consultarSalud, 10000);
}
iniciar();
