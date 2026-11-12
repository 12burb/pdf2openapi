// Everything happens in this tab: pdf.js reads the PDF, Pyodide runs the pdf2openapi parser.
import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs";
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/pyodide.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs";

const $ = (id) => document.getElementById(id);
const status = (text, err = false) => {
  $("status").textContent = text;
  $("status").className = err ? "err" : "";
};

let current = null; // { name, spec }

const pyReady = (async () => {
  const pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/" });
  await pyodide.loadPackage(["pyyaml", "micropip"]);
  const manifest = await (await fetch("wheel.json", { cache: "no-cache" })).json();
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(new URL(manifest.wheel, location.href).href, { deps: false });
  pyodide.runPython("import pdf2openapi.browser as browser");
  return pyodide;
})();

pyReady.then(
  () => status("Ready. Drop a PDF."),
  (e) => status(`Could not load the parser: ${e}`, true),
);

// Words as PyMuPDF would report them: [x0, y0, x1, y1, text] in points, top-left origin.
// pdf.js splits text into fragments mid-word ("G" + "as"); glue fragments that touch on the
// same baseline back together before splitting on whitespace.
async function pdfWords(data) {
  const pdf = await pdfjsLib.getDocument({ data }).promise;
  const meta = await pdf.getMetadata().catch(() => null);
  const title = meta?.info?.Title || "";
  const pages = [];
  for (let n = 1; n <= pdf.numPages; n++) {
    const page = await pdf.getPage(n);
    const viewport = page.getViewport({ scale: 1 });
    const content = await page.getTextContent();
    const words = [];
    let run = null;
    const flush = () => {
      if (!run) return;
      const cw = (run.x1 - run.x0) / run.text.length;
      for (const m of run.text.matchAll(/\S+/g)) {
        const x0 = run.x0 + m.index * cw;
        words.push([x0, run.y - run.h * 0.8, x0 + m[0].length * cw, run.y + run.h * 0.2, m[0]]);
      }
      run = null;
    };
    for (const item of content.items) {
      if (!item.str) continue;
      const [, b, , d, e, f] = item.transform;
      const h = Math.hypot(b, d) || item.height || 10;
      const [x, y] = viewport.convertToViewportPoint(e, f);
      const touching =
        run &&
        Math.abs(y - run.y) < 1 &&
        x - run.x1 > -0.5 * h && // kerned pairs ("T" + "o") overlap
        x - run.x1 < 0.22 * h && // a real space glyph is wider than this
        !/\s$/.test(run.text) &&
        !/^\s/.test(item.str);
      if (touching) {
        run.text += item.str;
        run.x1 = x + item.width;
      } else {
        flush();
        if (item.str.trim()) run = { x0: x, x1: x + item.width, y, h, text: item.str };
      }
      if (item.hasEOL) flush();
    }
    flush();
    pages.push(words);
    if (n % 20 === 0) status(`Reading page ${n} of ${pdf.numPages}...`);
  }
  return { title, pages, numPages: pdf.numPages };
}

async function extractPdf(file) {
  const data = new Uint8Array(await file.arrayBuffer());
  const { title, pages, numPages } = await pdfWords(data);
  status(`Extracting endpoints from ${numPages} pages...`);
  const pyodide = await pyReady;
  const fn = pyodide.globals.get("browser").extract_json;
  const out = JSON.parse(fn(file.name, title, JSON.stringify(pages), $("format").value));
  fn.destroy();
  return out;
}

async function onMainFile(file) {
  if (!file) return;
  try {
    status(`Reading ${file.name}...`);
    const out = await extractPdf(file);
    current = { name: file.name, spec: out.spec, text: out.text };
    $("count").textContent = out.count;
    $("name").textContent = file.name;
    $("out").value = out.text;
    $("tip").hidden = !out.tip;
    $("tip").textContent = out.tip
      ? `${out.tip} The command-line version can do that with --ai or a local Ollama model.`
      : "";
    const ws = out.warnings;
    $("warnings").hidden = !ws.length;
    $("warnings").querySelector("summary").textContent = `${ws.length} warning${ws.length === 1 ? "" : "s"}`;
    $("warnings").querySelector("ul").replaceChildren(
      ...ws.map((w) => Object.assign(document.createElement("li"), { textContent: w })),
    );
    $("result").classList.add("show");
    $("diffsec").classList.add("show");
    $("diffout").hidden = true;
    status(out.count ? "Done. Nothing was uploaded." : "No endpoints found in this PDF.");
  } catch (e) {
    console.error(e);
    status(`Could not read ${file.name}: ${e.message || e}`, true);
  }
}

async function onOldFile(file) {
  if (!file || !current) return;
  try {
    status(`Comparing with ${file.name}...`);
    const pyodide = await pyReady;
    let oldSpec;
    if (/\.pdf$/i.test(file.name)) {
      oldSpec = (await extractPdf(file)).spec;
    } else {
      const text = await file.text();
      oldSpec = /^\s*[{[]/.test(text)
        ? JSON.parse(text)
        : pyodide.runPython(`import json, yaml; json.dumps(yaml.safe_load(${JSON.stringify(text)}))`) && JSON.parse(pyodide.runPython(`json.dumps(yaml.safe_load(${JSON.stringify(text)}))`));
    }
    const fn = pyodide.globals.get("browser").diff_json;
    const report = JSON.parse(fn(JSON.stringify(oldSpec), JSON.stringify(current.spec)));
    fn.destroy();
    $("diffout").textContent = report.markdown;
    $("diffout").hidden = false;
    status(`${report.breaking_count} breaking change${report.breaking_count === 1 ? "" : "s"} between ${file.name} and ${current.name}.`);
  } catch (e) {
    console.error(e);
    status(`Could not compare: ${e.message || e}`, true);
  }
}

function wireDrop(dropId, inputId, handler) {
  const drop = $(dropId);
  const input = $(inputId);
  input.addEventListener("change", () => handler(input.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    handler(e.dataTransfer.files[0]);
  });
}
wireDrop("drop", "file", onMainFile);
wireDrop("drop2", "file2", onOldFile);

$("format").addEventListener("change", async () => {
  if (!current) return;
  const pyodide = await pyReady;
  const dump = pyodide.pyimport("pdf2openapi.output").dump;
  current.text = dump(pyodide.toPy(current.spec), $("format").value);
  dump.destroy();
  $("out").value = current.text;
});

$("copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText($("out").value);
  status("Copied to clipboard.");
});

$("download").addEventListener("click", () => {
  if (!current) return;
  const ext = $("format").value === "json" ? "json" : "yaml";
  const blob = new Blob([$("out").value], { type: "text/plain" });
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob),
    download: current.name.replace(/\.pdf$/i, "") + `.openapi.${ext}`,
  });
  a.click();
  URL.revokeObjectURL(a.href);
});
window.pdfWords = pdfWords; // used by tests
window.pdfjsLib = pdfjsLib;
