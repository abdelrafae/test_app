// --- Helpers ---
const $ = (id) => document.getElementById(id);
const fmt = (x) => (Number.isFinite(x) ? x.toFixed(5) : "—");

// Update units shown beside results
function applyUnitLabels() {
  const unit = $("unit").value;
  $("qtUnit").textContent = `STB/${unit}`;
  $("tUnit").textContent = unit;
}

// Show/hide b-factor input depending on decline type
function toggleB() {
  const d = $("decline").value;
  $("bRow").style.display = d === "hyp" ? "grid" : "none";
}

// Arps calculations
function compute() {
  // parse inputs
  const decline = $("decline").value;      // "exp" | "hyp" | "har"
  const unit = $("unit").value;            // display only
  const qi = parseFloat($("qi").value);
  const di = parseFloat($("di").value);
  const t = parseFloat($("t").value);
  const qe = parseFloat($("qecon").value);
  const b = parseFloat($("b").value);

  // basic validation
  if (!(qi > 0) || !(di > 0) || !(qe > 0) || !(t >= 0)) {
    alert("Please enter positive qi, Di, q_econ, and non-negative t.");
    return;
  }
  if (decline === "hyp" && !(b > 0 && b < 1)) {
    alert("For hyperbolic decline, use 0 < b < 1.");
    return;
  }

  let qt, np, te, eur;

  if (decline === "exp") {
    // q = qi * e^(-Di t)
    qt = qi * Math.exp(-di * t);
    // Np = (qi - q) / Di
    np = (qi - qt) / di;
    // t_e = (1/Di) ln(qi / qe)
    te = (1 / di) * Math.log(qi / qe);
    // EUR = (qi - qe) / Di
    eur = (qi - qe) / di;
  } else if (decline === "har") {
    // q = qi / (1 + Di t)
    qt = qi / (1 + di * t);
    // Np = (qi/Di) ln(1 + Di t)
    np = (qi / di) * Math.log(1 + di * t);
    // t_e = (qi/qe - 1) / Di
    te = (qi / qe - 1) / di;
    // EUR = Np(t_e)
    eur = (qi / di) * Math.log(1 + di * te);
  } else {
    // hyperbolic: 0 < b < 1
    const u = 1 + b * di * t;
    // q = qi / (1 + b Di t)^(1/b)
    qt = qi / Math.pow(u, 1 / b);
    // Np = qi / (Di(1-b)) * [1 - (1 + b Di t)^((b-1)/b)]
    np = (qi / (di * (1 - b))) * (1 - Math.pow(u, (b - 1) / b));
    // t_e = ((qi/qe)^b - 1) / (b Di)
    te = (Math.pow(qi / qe, b) - 1) / (b * di);
    // EUR = Np(t_e)
    const ue = 1 + b * di * te;
    eur = (qi / (di * (1 - b))) * (1 - Math.pow(ue, (b - 1) / b));
  }

  $("qt").textContent = fmt(qt);
  $("np").textContent = fmt(np);
  $("te").textContent = fmt(te);
  $("eur").textContent = fmt(eur);
}

// Clear results
function clearResults() {
  ["qt", "np", "te", "eur"].forEach((id) => ($(id).textContent = "-"));
}

// Events
$("decline").addEventListener("change", () => { toggleB(); });
$("unit").addEventListener("change", applyUnitLabels);
$("compute").addEventListener("click", compute);
$("clear").addEventListener("click", clearResults);

// Init
toggleB();
applyUnitLabels();
$("year").textContent = new Date().getFullYear();

// --- PWA: register service worker ---
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch(console.error);
  });
}
