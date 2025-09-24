// Simple placeholder app logic
document.getElementById("year").textContent = new Date().getFullYear();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch(console.error);
  });
}
