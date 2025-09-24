import math, io, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyscript import document, display, js

EPS = 1e-12
P_EPS = 1e-12  # clamp for probabilities

def clamp(x, lo, hi): return max(min(x, hi), lo)

# Acklam inverse-normal approximation
def norminv(p: float) -> float:
    p = clamp(p, P_EPS, 1 - P_EPS)
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    plow=0.02425; phigh=1-plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif phigh < p:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p-0.5; r=q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def lhs(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    # Latin Hypercube Sampling, U(0,1)
    cut = np.linspace(0, 1, n+1)
    H = np.zeros((n, k))
    for j in range(k):
        r = rng.random(n)
        rd = r*(cut[1:]-cut[:-1]) + cut[:-1]
        idx = np.arange(n)
        rng.shuffle(idx)
        H[:, j] = rd[idx]
    return H

def cholesky(A: np.ndarray) -> np.ndarray:
    # simple Cholesky (expect symmetric PSD)
    return np.linalg.cholesky(A + 1e-12*np.eye(A.shape[0]))

def infer_normal(xmin, xmax, z):
    mean = 0.5*(xmin+xmax)
    std = (xmax-xmin)/(2*max(z, EPS))
    return mean, max(std, 1e-16)

def infer_lognormal(xmin, xmax, z):
    xmin = max(xmin, 1e-16); xmax = max(xmax, 1e-16)
    lnmin, lnmax = math.log(xmin), math.log(xmax)
    mu = 0.5*(lnmin+lnmax)
    sigma = (lnmax-lnmin)/(2*max(z, EPS))
    return mu, max(sigma, 1e-16)

def betainv_approx(p, a, b):
    # Peizer–Pratt style normal approx
    mu = a/(a+b)
    sigma = math.sqrt(a*b/(((a+b)**2)*(a+b+1)))
    x = mu + sigma*norminv(p)
    return clamp(x, 1e-9, 1-1e-9)

def map_dist(dist, xmin, xmax, u, z):
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    dist = dist.lower()
    if dist == "fixed":
        return np.full_like(u, 0.5*(xmin+xmax), dtype=float)
    if dist == "uniform":
        return xmin + (xmax-xmin)*u
    if dist == "triangular":
        a, b, c = xmin, xmax, 0.5*(xmin+xmax)
        denom = max(b-a, EPS)
        Fc = (c-a)/denom
        out = np.empty_like(u)
        left = u < Fc
        out[left] = a + np.sqrt(u[left]*(b-a)*max(c-a, 0))
        out[~left] = b - np.sqrt((1-u[~left])*(b-a)*max(b-c, 0))
        return out
    if dist == "pert":
        a, b, m, lam = xmin, xmax, 0.5*(xmin+xmax), 4.0
        denom = max(b-a, EPS)
        alpha = 1 + lam*(m-a)/denom
        beta  = 1 + lam*(b-m)/denom
        x = np.array([betainv_approx(float(ui), alpha, beta) for ui in u])
        return a + x*(b-a)
    if dist == "normal":
        mean, std = infer_normal(xmin, xmax, z)
        x = np.array([mean + std*norminv(float(ui)) for ui in u])
        return x
    if dist == "lognormal":
        mu, sigma = infer_lognormal(max(xmin,1e-16), max(xmax,1e-16), z)
        x = np.array([math.exp(mu + sigma*norminv(float(ui))) for ui in u])
        return x
    raise ValueError(f"Unsupported distribution: {dist}")

def percentile(a, q):
    return float(np.percentile(a, q))

def read_vars():
    # pull variable cards from DOM
    names  = [el.value.strip() for el in document.querySelectorAll(".v_name")]
    dists  = [el.value for el in document.querySelectorAll(".v_dist")]
    mins   = [float(el.value) for el in document.querySelectorAll(".v_min")]
    maxs   = [float(el.value) for el in document.querySelectorAll(".v_max")]
    if len(set(names)) != len(names):
        raise ValueError("Variable names must be unique.")
    return names, dists, mins, maxs

def read_corr(k: int):
    use_corr = bool(document.getElementById("useCorr").checked)
    if not use_corr or k <= 1:
        return None
    inputs = document.querySelectorAll(".corr")
    C = np.zeros((k, k), dtype=float)
    for el in inputs:
        i = int(el.getAttribute("data-i"))
        j = int(el.getAttribute("data-j"))
        v = float(el.value)
        if i == j: v = 1.0
        v = max(min(v, 0.95), -0.95)
        C[i, j] = v
    # enforce symmetry
    C = 0.5*(C + C.T)
    np.fill_diagonal(C, 1.0)
    # ensure PD
    _ = cholesky(C)
    return C

def set_metric(id_, val):
    document.getElementById(id_).innerText = f"{val:.5f}"

def render_table(names, y, samples, n_show=10):
    headers = names + ["OUTPUT"]
    html = "<table><thead><tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr></thead><tbody>"
    m = min(n_show, len(y))
    for i in range(m):
        row = [f"{samples[nm][i]:.5f}" for nm in names] + [f"{y[i]:.5f}"]
        html += "<tr>" + "".join([f"<td>{v}</td>" for v in row]) + "</tr>"
    html += "</tbody></table>"
    document.getElementById("preview").innerHTML = html

def plot_hist(y):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(y, bins=50)
    ax.set_title("Output Distribution")
    ax.set_xlabel("OUTPUT"); ax.set_ylabel("Frequency")
    display(fig, target="hist", append=False)
    plt.close(fig)

def plot_cdf(y):
    a = np.sort(y)
    x = a
    n = len(a)
    p = np.linspace(0, 1, n)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(x, p)
    ax.set_title("Empirical CDF")
    ax.set_xlabel("OUTPUT"); ax.set_ylabel("Cumulative Probability")
    display(fig, target="cdf", append=False)
    plt.close(fig)

def plot_tornado(names, samples, y):
    rs = []
    y_mean = float(np.mean(y))
    y_var = float(np.sum((y - y_mean)**2))
    for nm in names:
        xi = samples[nm]
        xm = float(np.mean(xi))
        num = float(np.sum((xi - xm)*(y - y_mean)))
        den = math.sqrt(float(np.sum((xi - xm)**2)) * (y_var if y_var>0 else 1.0))
        r = num/den if den>0 else 0.0
        rs.append((nm, r))
    rs.sort(key=lambda t: abs(t[1]))
    labs = [t[0] for t in rs]; vals = [t[1] for t in rs]
    fig, ax = plt.subplots(figsize=(6,3))
    ax.barh(labs, vals, color=["#22c55e" if v>=0 else "#ef4444" for v in vals])
    ax.set_title("Pearson Correlation with OUTPUT")
    ax.set_xlabel("Correlation")
    display(fig, target="tornado", append=False)
    plt.close(fig)

def csv_escape(s): 
    return '"' + str(s).replace('"','""') + '"'

def download_blob(filename: str, text: str):
    blob = js.Blob.new([text], { "type": "text/csv;charset=utf-8;" })
    url = js.URL.createObjectURL(blob)
    a = js.document.createElement("a")
    a.href = url
    a.download = filename
    js.document.body.appendChild(a)
    a.click()
    js.document.body.removeChild(a)
    js.URL.revokeObjectURL(url)

def run_sim(event=None):
    try:
        n = int(document.getElementById("iters").value)
        seed = int(document.getElementById("seed").value) if document.getElementById("seed").value else 42
        use_lhs = bool(document.getElementById("useLhs").checked)
        coverage = int(document.getElementById("coverage").value)
        expr = document.getElementById("expr").value.strip()
        names, dists, mins, maxs = read_vars()
        k = len(names)
        z = norminv(0.5 + coverage/200.0)

        rng = np.random.default_rng(seed)
        U = lhs(n, k, rng) if use_lhs else rng.random((n, k))
        # Correlations via Gaussian copula
        C = read_corr(k)
        if C is not None:
            Z = np.vectorize(norminv)(U)
            L = cholesky(C)
            Zc = Z @ L.T
            from math import erf, sqrt
            U = 0.5*(1.0 + erf(Zc/ math.sqrt(2)))  # normcdf

        # Map marginals
        samples = {}
        for j in range(k):
            samples[names[j]] = map_dist(dists[j], mins[j], maxs[j], U[:, j], z)

        # Evaluate expression safely (math/numpy only)
        cleaned = expr.replace("^", "**")
        safe_globals = {"__builtins__": {}}
        safe_locals = {
            **samples,
            "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
            "sqrt": np.sqrt, "abs": np.abs, "minimum": np.minimum, "maximum": np.maximum,
            "where": np.where, "clip": np.clip, "pi": math.pi
        }
        y = eval(cleaned, safe_globals, safe_locals).astype(float).reshape(-1)

        mean = float(np.mean(y))
        std  = float(np.std(y, ddof=1))
        p10, p50, p90 = (float(np.percentile(y, q)) for q in (10, 50, 90))

        set_metric("mMean", mean)
        set_metric("mStd", std)
        set_metric("mP10", p10)
        set_metric("mP50", p50)
        set_metric("mP90", p90)

        render_table(names, y, samples)
        plot_hist(y); plot_cdf(y); plot_tornado(names, samples, y)

        # stash data for downloads
        global _last_data
        _last_data = (names, y, samples, mean, std, p10, p50, p90)

    except Exception as e:
        js.alert(f"Error: {e}")

def clear_results(event=None):
    for id_ in ["mMean","mStd","mP10","mP50","mP90"]:
        document.getElementById(id_).innerText="-"
    document.getElementById("preview").innerHTML=""
    for t in ["hist","cdf","tornado"]:
        document.getElementById(t).innerHTML=""

_last_data = None

def download_data(event=None):
    global _last_data
    if not _last_data:
        js.alert("Run the simulation first.")
        return
    names, y, samples, *_ = _last_data
    headers = names + ["OUTPUT"]
    lines = [",".join(headers)]
    n = len(y)
    for i in range(n):
        row = [f"{samples[nm][i]}" for nm in names] + [f"{y[i]}"]
        lines.append(",".join(row))
    download_blob("mc_bounds_results.csv", "\n".join(lines))

def download_summary(event=None):
    global _last_data
    if not _last_data:
        js.alert("Run the simulation first.")
        return
    _, _, _, mean, std, p10, p50, p90 = _last_data
    lines = ["metric,value",
             f"mean,{mean}",
             f"std,{std}",
             f"p10,{p10}",
             f"p50,{p50}",
             f"p90,{p90}"]
    download_blob("mc_bounds_summary.csv", "\n".join(lines))
