// ===== Utility =====
const $ = (id) => document.getElementById(id);
const fmt = (x) => (Number.isFinite(x) ? x.toFixed(5) : "—");
const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

// Safer constants
const EPS = 1e-12;
const P_EPS = 1e-12; // protect prob inputs to inverse CDFs

function seedPRNG(seed){ // Mulberry32
  let t = seed >>> 0;
  return function(){
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  }
}

// Normal CDF / inv (Acklam) with clamping
function erf(x){
  const sign = Math.sign(x);
  x = Math.abs(x);
  const a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429;
  const p=0.3275911;
  const t=1/(1+p*x);
  const y=1-((((a5*t+a4)*t+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
  return sign*y;
}
function normcdf(x){ return 0.5 * (1 + erf(x / Math.SQRT2)); }
function norminv(p){
  p = clamp(p, P_EPS, 1 - P_EPS);
  const a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00];
  const b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01];
  const c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00];
  const d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00];
  const plow=0.02425, phigh=1-plow;
  let q, r;
  if(p<plow){
    q=Math.sqrt(-2*Math.log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  } else if (phigh<p){
    q=Math.sqrt(-2*Math.log(1-p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  } else {
    q=p-0.5; r=q*q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
  }
}

// Percentile
function percentile(arr, q){
  const a = Float64Array.from(arr).sort();
  const idx = (q/100)*(a.length-1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo===hi) return a[lo];
  return a[lo] + (a[hi]-a[lo])*(idx-lo);
}

// Seeded Fisher–Yates shuffle
function shuffleInPlace(arr, rng){
  for(let i=arr.length-1;i>0;i--){
    const j = Math.floor(rng()*(i+1));
    const t = arr[i]; arr[i]=arr[j]; arr[j]=t;
  }
}

// LHS (seeded; no Math.random())
function lhs(n, k, rng){
  const cuts = Array.from({length:n+1}, (_,i)=>i/n);
  const H = Array.from({length:n}, ()=>Array(k).fill(0));
  for(let j=0;j<k;j++){
    const rands = Array.from({length:n}, ()=>rng());
    const rd = rands.map((u,i)=> u*(cuts[i+1]-cuts[i]) + cuts[i]);
    const idx = Array.from({length:n}, (_,i)=>i);
    shuffleInPlace(idx, rng);
    for(let i=0;i<n;i++) H[i][j] = rd[idx[i]];
  }
  return H;
}

// Cholesky + helpers
function cholesky(A){
  const n=A.length;
  const L=Array.from({length:n}, ()=>Array(n).fill(0));
  for(let i=0;i<n;i++){
    for(let j=0;j<=i;j++){
      let sum=0;
      for(let k=0;k<j;k++) sum += L[i][k]*L[j][k];
      if(i===j){
        const v=A[i][i]-sum;
        if(!(v>EPS)) throw new Error("Correlation matrix not positive definite");
        L[i][j]=Math.sqrt(v);
      }else{
        L[i][j]=(A[i][j]-sum)/L[j][j];
      }
    }
  }
  return L;
}
function matVec(L, z){
  const n=L.length; const out=new Array(n).fill(0);
  for(let i=0;i<n;i++){
    let s=0; for(let k=0;k<=i;k++) s+=L[i][k]*z[k];
    out[i]=s;
  }
  return out;
}

// Distributions from bounds (min,max) only
function inferNormal(xmin,xmax,z){
  const mean=(xmin+xmax)/2;
  const std = (xmax-xmin)/(2*Math.max(z, EPS));
  return [mean, Math.max(std, 1e-16)];
}
function inferLognormal(xmin,xmax,z){
  xmin=Math.max(xmin,1e-16); xmax=Math.max(xmax,1e-16);
  const lnmin=Math.log(xmin), lnmax=Math.log(xmax);
  const mu=(lnmin+lnmax)/2; 
  const sigma=(lnmax-lnmin)/(2*Math.max(z, EPS));
  return [mu, Math.max(sigma,1e-16)];
}
function mapDist(dist, xmin, xmax, u, z){
  if(xmin>xmax){ const t=xmin; xmin=xmax; xmax=t; }
  const n=u.length; const out=new Float64Array(n);
  const d=dist.toLowerCase();
  if(d==="fixed"){ const val=0.5*(xmin+xmax); out.fill(val); return out; }
  if(d==="uniform"){ for(let i=0;i<n;i++) out[i]=xmin+(xmax-xmin)*u[i]; return out; }
  if(d==="triangular"){
    const a=xmin, b=xmax, c=(xmin+xmax)/2;
    const denom = Math.max(b-a, EPS);
    const Fc = (c-a)/denom;
    for(let i=0;i<n;i++){
      const ui=clamp(u[i], P_EPS, 1-P_EPS);
      if(ui<Fc){ out[i]= a + Math.sqrt(ui * (b-a) * Math.max(c-a,0)); }
      else { out[i]= b - Math.sqrt((1-ui) * (b-a) * Math.max(b-c,0)); }
    }
    return out;
  }
  if(d==="pert"){
    // symmetric Beta-PERT with lambda=4 (a=min, m=(a+b)/2, b=max)
    const a=xmin, b=xmax, m=(a+b)/2, lam=4;
    const denom = Math.max(b-a, EPS);
    const alpha = 1 + lam*(m-a)/denom;
    const beta  = 1 + lam*(b-m)/denom;
    function betainv(p, A, B){
      // normal approx
      const mu=A/(A+B), sigma=Math.sqrt(A*B/((A+B)*(A+B)*(A+B+1)));
      const x = mu + sigma*norminv(p);
      return clamp(x, 1e-9, 1-1e-9);
    }
    for(let i=0;i<n;i++){
      const x = betainv(clamp(u[i], P_EPS, 1-P_EPS), alpha, beta);
      out[i] = a + x*(b-a);
    }
    return out;
  }
  if(d==="normal"){
    const [mean,std]=inferNormal(xmin,xmax,z);
    for(let i=0;i<n;i++){ out[i]= mean + std*norminv(clamp(u[i], P_EPS, 1-P_EPS)); }
    return out;
  }
  if(d==="lognormal"){
    const [mu,sigma]=inferLognormal(Math.max(xmin,1e-16),Math.max(xmax,1e-16),z);
    for(let i=0;i<n;i++){ out[i]= Math.exp(mu + sigma*norminv(clamp(u[i], P_EPS, 1-P_EPS))); }
    return out;
  }
  throw new Error("Unsupported distribution: "+dist);
}

// UI: Variables
const DIST_LIST = ["fixed","uniform","triangular","normal","lognormal","PERT"];

function buildVars(){
  const k = parseInt($("kVars").value);
  const wrap = $("vars"); wrap.innerHTML="";
  for(let i=0;i<k;i++){
    const v = document.createElement("div");
    v.className="vcard";
    v.innerHTML = `
      <div class="vgrid">
        <div><label>Name</label><input id="v_name_${i}" value="X${i+1}"></div>
        <div><label>Distribution</label>
          <select id="v_dist_${i}">${DIST_LIST.map((d,j)=>`<option value="${d}" ${i===1&&j===3?"selected":(j===1?"selected":"")}>${d}</option>`).join("")}</select>
        </div>
        <div><label>min</label><input id="v_min_${i}" type="number" step="0.00001" value="${(i===0)?0:"0.00000"}"></div>
        <div><label>max</label><input id="v_max_${i}" type="number" step="0.00001" value="${(i===0)?1:"1.00000"}"></div>
      </div>
    `;
    wrap.appendChild(v);
  }
  buildCorr(); // refresh corr table
}

function getVarSpecs(){
  const k = parseInt($("kVars").value);
  const specs = [];
  for(let i=0;i<k;i++){
    const name = $("v_name_"+i).value.trim();
    const dist = $("v_dist_"+i).value;
    const xmin = parseFloat($("v_min_"+i).value);
    const xmax = parseFloat($("v_max_"+i).value);
    if(!name) throw new Error("Variable names must be non-empty.");
    specs.push({name, dist, xmin, xmax});
  }
  const set = new Set(specs.map(s=>s.name));
  if(set.size !== specs.length) throw new Error("Variable names must be unique.");
  return specs;
}

// Correlation matrix UI
function buildCorr(){
  const k = parseInt($("kVars").value);
  const section = $("corrSection");
  const useCorr = $("useCorr").checked;
  section.style.display = (useCorr && k>1) ? "" : "none";
  if(!(useCorr && k>1)) return;

  const tbl = document.createElement("table");
  tbl.style.borderCollapse="collapse";
  tbl.style.width="100%";
  const header = document.createElement("tr");
  header.innerHTML = "<th></th>" + Array.from({length:k}, (_,j)=>`<th>X${j+1}</th>`).join("");
  tbl.appendChild(header);
  for(let i=0;i<k;i++){
    const tr = document.createElement("tr");
    tr.innerHTML = `<th style="text-align:left">X${i+1}</th>` + Array.from({length:k}, (_,j)=>{
      const val = (i===j)?1:0;
      return `<td><input data-corr="1" data-i="${i}" data-j="${j}" type="number" step="0.01" min="-0.95" max="0.95" value="${val}" style="width:80px"></td>`
    }).join("");
    tbl.appendChild(tr);
  }
  $("corrTable").innerHTML="";
  $("corrTable").appendChild(tbl);
}

$("applyK").addEventListener("click", buildVars);
$("useCorr").addEventListener("change", buildCorr);

// Expression evaluator (restricted)
function safeEval(expr, scope){
  const cleaned = expr.replace(/\^/g, "**"); // allow ^
  if(/[;={}[\]`]|function|=>|while|for|if|switch|new|class|import|export/i.test(cleaned)){
    throw new Error("Invalid expression");
  }
  const keys = Object.keys(scope);
  const extras = ["sin","cos","tan","exp","log","sqrt","abs","min","max","PI","E","where","clip"];
  const allowed = keys.concat(extras);
  const f = new Function(...allowed, `return (${cleaned});`);
  const args = allowed.map(k=>{
    if(k==="PI") return Math.PI;
    if(k==="E") return Math.E;
    if(k==="where") return (cond,a,b)=>cond? a:b;
    if(k==="clip") return (x,lo,hi)=> clamp(x,lo,hi);
    // expose Math.* directly
    if(k==="sin") return Math.sin;
    if(k==="cos") return Math.cos;
    if(k==="tan") return Math.tan;
    if(k==="exp") return Math.exp;
    if(k==="log") return Math.log;
    if(k==="sqrt") return Math.sqrt;
    if(k==="abs") return Math.abs;
    if(k==="min") return Math.min;
    if(k==="max") return Math.max;
    return scope[k];
  });
  return f(...args);
}

// CSV helpers
function toCSV(headers, rows){
  const esc = (s)=>(''+s).replace(/"/g,'""');
  let out = headers.join(",")+"\n";
  for(const r of rows){
    out += r.map(v=> `"${esc(v)}"`).join(",")+"\n";
  }
  return out;
}
function download(name, text){
  const blob = new Blob([text], {type:"text/csv;charset=utf-8;"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href=url; a.download=name; a.click();
  URL.revokeObjectURL(url);
}

// Main run
$("run").addEventListener("click", () => {
  try {
    const n = parseInt($("iters").value);
    const seed = parseInt($("seed").value)||42;
    const useLhs = $("useLhs").checked;
    const useCorr = $("useCorr").checked;
    const coverage = parseInt($("coverage").value);
    const expr = $("expr").value.trim();
    const specs = getVarSpecs();

    if(n<=0) throw new Error("Iterations must be positive");
    if(!expr) throw new Error("Expression is required");

    // coverage → z for central coverage
    const z = norminv(0.5 + coverage/200);

    // Base U(0,1)
    const rng = seedPRNG(seed);
    let U = useLhs ? lhs(n, specs.length, rng) : Array.from({length:n}, ()=> Array.from({length:specs.length}, ()=>clamp(rng(), P_EPS, 1-P_EPS)));

    // Gaussian copula correlation (optional)
    if(useCorr && specs.length>1){
      // read matrix from UI
      const inputs = document.querySelectorAll("input[data-corr='1']");
      const k = specs.length;
      const C = Array.from({length:k}, ()=>Array(k).fill(0));
      inputs.forEach(inp=>{
        const i=parseInt(inp.dataset.i), j=parseInt(inp.dataset.j);
        let v = parseFloat(inp.value);
        if(i===j) v=1.0;
        v = clamp(v, -0.95, 0.95);
        C[i][j]=v;
      });
      for(let i=0;i<k;i++) for(let j=0;j<k;j++){ C[i][j] = (C[i][j]+C[j][i])/2; if(i===j) C[i][j]=1; }
      const L = cholesky(C);

      const Ucorr = new Array(n);
      for(let i=0;i<n;i++){
        const Zi = U[i].map(u=> norminv(u));
        const Zc = matVec(L, Zi);
        Ucorr[i] = Zc.map(z=> normcdf(z));
      }
      U = Ucorr;
    }

    // Map marginals from bounds
    const samples = {};
    for(let j=0;j<specs.length;j++){
      const s = specs[j];
      const ucol = Float64Array.from(U.map(row=>row[j]));
      samples[s.name] = mapDist(s.dist, s.xmin, s.xmax, ucol, z);
    }

    // Evaluate expression row-wise
    const names = specs.map(s=>s.name);
    const y = new Float64Array(n);
    for(let i=0;i<n;i++){
      const scope = {};
      for(const nm of names) scope[nm] = samples[nm][i];
      y[i] = safeEval(expr, scope);
      if(!Number.isFinite(y[i])) throw new Error("Expression produced non-finite value at row "+i);
    }

    // Summary stats
    const mean = y.reduce((a,b)=>a+b,0)/n;
    const sd = Math.sqrt(y.reduce((s,yi)=>s+(yi-mean)*(yi-mean),0)/Math.max(n-1,1));
    const p10 = percentile(y,10), p50=percentile(y,50), p90=percentile(y,90);

    $("mMean").textContent = fmt(mean);
    $("mStd").textContent  = fmt(sd);
    $("mP10").textContent  = fmt(p10);
    $("mP50").textContent  = fmt(p50);
    $("mP90").textContent  = fmt(p90);

    // Preview (first 10)
    const rows = [];
    const headers = names.concat(["OUTPUT"]);
    for(let i=0;i<Math.min(10,n);i++){
      const r = names.map(nm=> fmt(samples[nm][i]));
      r.push(fmt(y[i]));
      rows.push(r);
    }
    let html = "<table><thead><tr>"+ headers.map(h=>`<th>${h}</th>`).join("") +"</tr></thead><tbody>";
    for(const r of rows){ html += "<tr>"+ r.map(v=>`<td>${v}</td>`).join("") + "</tr>"; }
    html += "</tbody></table>";
    $("preview").innerHTML = html;

    // Charts
    drawHist($("hist"), y, 50);
    drawCDF($("cdf"), y);

    // Tornado (Pearson correlation)
    const corPairs = [];
    for(const nm of names){
      const xi = samples[nm];
      const xm = xi.reduce((a,b)=>a+b,0)/n;
      const num = xi.reduce((s,xx,idx)=> s + (xx - xm)*(y[idx]-mean), 0);
      const den = Math.sqrt( xi.reduce((s,xx)=> s + (xx - xm)*(xx - xm),0) ) * Math.sqrt( y.reduce((s,yy)=> s + (yy - mean)*(yy - mean),0) );
      const r = den>0? num/den : 0;
      corPairs.push([nm, r]);
    }
    corPairs.sort((a,b)=> Math.abs(a[1]) - Math.abs(b[1]));
    drawTornado($("tornado"), corPairs);

    // Downloads
    $("dlData").onclick = ()=>{
      const allRows = [];
      for(let i=0;i<n;i++){
        const r = names.map(nm=> samples[nm][i]);
        r.push(y[i]);
        allRows.push(r);
      }
      download("mc_bounds_results.csv", toCSV(headers, allRows));
    };
    $("dlSummary").onclick = ()=>{
      const sHeaders=["metric","value"];
      const sRows=[["mean",mean],["std",sd],["p10",p10],["p50",p50],["p90",p90]];
      download("mc_bounds_summary.csv", toCSV(sHeaders, sRows));
    };

  } catch (e){
    alert(e.message||e);
    console.error(e);
  }
});

$("clear").addEventListener("click", ()=>{
  ["mMean","mStd","mP10","mP50","mP90"].forEach(id=> $(id).textContent="-");
  $("preview").innerHTML="";
  const ctxs=["hist","cdf","tornado"].map(id=>$(id).getContext("2d"));
  ctxs.forEach(ctx=>{ ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height); });
});

$("applyK").addEventListener("click", buildVars);
$("applyK").click();

// Drawing helpers (same as before, with extra guards)
function drawHist(canvas, data, bins){
  const ctx = canvas.getContext("2d");
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);
  const min = Math.min(...data), max = Math.max(...data);
  const span = Math.max(max-min, EPS);
  const binW = span/bins;
  const counts = new Array(bins).fill(0);
  for(const v of data){
    let b = Math.floor((v-min)/binW);
    if(b>=bins) b=bins-1;
    if(b<0) b=0;
    counts[b]++;
  }
  const cmax = Math.max(...counts,1);
  ctx.strokeStyle="#e5e7eb"; ctx.beginPath();
  ctx.moveTo(40,H-30); ctx.lineTo(W-10,H-30); ctx.lineTo(W-10,10); ctx.stroke();
  const barW = (W-60)/bins;
  ctx.fillStyle="#2563eb88";
  for(let i=0;i<bins;i++){
    const h = (H-50) * (counts[i]/cmax);
    ctx.fillRect(40 + i*barW, H-30 - h, barW-1, h);
  }
}
function drawCDF(canvas, data){
  const ctx = canvas.getContext("2d");
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);
  const a = Float64Array.from(data).sort();
  const n=a.length;
  const min=a[0], max=a[n-1];
  const span = Math.max(max-min, EPS);
  ctx.strokeStyle="#e5e7eb"; ctx.beginPath();
  ctx.moveTo(40,H-30); ctx.lineTo(W-10,H-30); ctx.lineTo(W-10,10); ctx.stroke();
  ctx.strokeStyle="#2563eb"; ctx.beginPath();
  for(let i=0;i<n;i++){
    const x = 40 + (W-50) * ( (a[i]-min) / span );
    const y = H-30 - (H-40) * ( i/(n-1) );
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();
}
function drawTornado(canvas, pairs){
  const ctx = canvas.getContext("2d");
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);
  const marginL=120, marginR=10, marginB=30, marginT=10;
  const n = pairs.length;
  const bw = (H - marginT - marginB)/Math.max(n,1);
  const rmax = Math.max(...pairs.map(p=>Math.abs(p[1])), 1e-6);
  ctx.fillStyle="#000"; ctx.font="12px sans-serif";
  for(let i=0;i<n;i++){
    const y = marginT + i*bw + bw*0.6;
    ctx.fillText(pairs[i][0], 8, y);
  }
  ctx.strokeStyle="#e5e7eb"; ctx.beginPath();
  ctx.moveTo(marginL, H-marginB); ctx.lineTo(W-marginR, H-marginB); ctx.stroke();
  for(let i=0;i<n;i++){
    const r = pairs[i][1];
    const x0 = marginL + (W - marginL - marginR)/2;
    const len = ((W - marginL - marginR)/2) * (Math.abs(r)/rmax);
    ctx.fillStyle= r>=0 ? "#22c55eaa" : "#ef4444aa";
    if(r>=0){ ctx.fillRect(x0, marginT + i*bw + 4, len, bw-8); }
    else    { ctx.fillRect(x0 - len, marginT + i*bw + 4, len, bw-8); }
  }
}

// PWA
if("serviceWorker" in navigator){
  window.addEventListener("load", ()=>{
    navigator.serviceWorker.register("sw.js").catch(console.error);
  });
}
document.getElementById("year").textContent = new Date().getFullYear();
