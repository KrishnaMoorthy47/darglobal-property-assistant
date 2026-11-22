(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  let token = "";
  try { token = sessionStorage.getItem("pia.admin") || ""; } catch { /* storage unavailable */ }
  const charts = {};
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  async function api(path) {
    const r = await fetch(path, { headers: { "X-Admin-Token": token } });
    if (r.status === 401) throw new Error("unauthorized");
    return r.json();
  }

  async function login() {
    token = $("#token").value.trim() || token;
    try {
      await api("/api/admin/metrics?hours=1");
      try { sessionStorage.setItem("pia.admin", token); } catch { /* ignore */ }
      $("#login").hidden = true; $("#dash").hidden = false;
      load();
    } catch {
      $("#loginErr").textContent = "Invalid token.";
    }
  }
  $("#loginBtn").onclick = login;
  $("#token").onkeydown = (e) => e.key === "Enter" && login();
  $("#refresh").onclick = () => load();
  $("#hours").onchange = () => load();
  if (token) login();

  function chart(id, type, labels, datasets, opts = {}) {
    charts[id]?.destroy();
    Chart.defaults.color = css("--muted");
    Chart.defaults.borderColor = css("--line");
    Chart.defaults.font.family = "Inter, system-ui, sans-serif";
    charts[id] = new Chart(document.getElementById(id), {
      type, data: { labels, datasets },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: datasets.length > 1 || type === "doughnut", position: "bottom" } }, ...opts },
    });
  }
  const palette = ["#c9a96e", "#0e7a5f", "#3b5b92", "#b54708", "#7a5af8", "#98a2b3", "#d92d20"];

  async function load() {
    const hours = $("#hours").value;
    const [d, traces] = await Promise.all([api(`/api/admin/metrics?hours=${hours}`), api("/api/admin/traces?limit=60")]);
    const m = d.metrics;
    const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
    const kpis = [
      [m.requests, "Requests"], [m.sessions, "Sessions"],
      [m.latency_ms.ttft.p50 ? `${(m.latency_ms.ttft.p50 / 1000).toFixed(1)}s` : "–", "Time to first token (p50)"],
      [m.latency_ms.total.p95 ? `${(m.latency_ms.total.p95 / 1000).toFixed(1)}s` : "–", "Total latency (p95)"],
      [pct(m.cache_hit_rate), "Cache hit rate"], [pct(m.fallback_rate), "Provider fallback rate"],
      [m.feedback.satisfaction == null ? "–" : pct(m.feedback.satisfaction), `Satisfaction (👍${m.feedback.up} 👎${m.feedback.down})`],
      [m.errors, "Errors"],
    ];
    $("#kpis").innerHTML = kpis.map(([v, l]) => `<div class="panel kpi"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`).join("");

    const hrs = Object.keys(m.by_hour);
    chart("cHour", "bar", hrs.map((h) => h.slice(5)), [{ label: "requests", data: Object.values(m.by_hour), backgroundColor: palette[0], borderRadius: 4 }]);
    const stages = ["router", "retrieval", "ttft", "llm", "total"];
    chart("cLat", "bar", stages, [
      { label: "p50", data: stages.map((s) => m.latency_ms[s].p50), backgroundColor: palette[2], borderRadius: 4 },
      { label: "p95", data: stages.map((s) => m.latency_ms[s].p95), backgroundColor: palette[0], borderRadius: 4 },
    ]);
    chart("cIntent", "doughnut", Object.keys(m.intents), [{ data: Object.values(m.intents), backgroundColor: palette }]);
    chart("cModel", "doughnut", Object.keys(m.models).map((x) => x.split("/").pop()), [{ data: Object.values(m.models), backgroundColor: palette }]);

    $("#keys").innerHTML = `<tr><th>Key</th><th>Requests today / cap</th><th>Status</th></tr>` +
      (d.keys || []).map((k) => {
        const w = k.daily_cap ? Math.min(100, (k.used_today / k.daily_cap) * 100) : 0;
        return `<tr><td>${esc(k.key)}</td>
          <td><div style="display:flex;gap:8px;align-items:center"><div class="bar"><i style="width:${w}%"></i></div>${k.used_today} / ${k.daily_cap ?? "no cap"}</div></td>
          <td class="${k.blocked_for_s ? "bad" : "ok"}">${k.blocked_for_s ? `daily limit hit (retry in ${Math.round(k.blocked_for_s / 60)} min)` : "ok"}</td></tr>`;
      }).join("");

    $("#prov").innerHTML = `<tr><th>Chain</th><th>Endpoint</th><th>Used today / safe limit</th><th>Daily cap</th><th>RPM</th><th>Circuit</th></tr>` +
      d.providers.map((p) => {
        const w = Math.min(100, (p.used_today / Math.max(1, p.safe_limit)) * 100);
        return `<tr><td>${esc(p.chain)}</td><td>${esc(p.endpoint)}</td>
          <td><div style="display:flex;gap:8px;align-items:center"><div class="bar"><i style="width:${w}%"></i></div>${p.used_today} / ${p.safe_limit}</div></td>
          <td>${p.daily_cap}</td><td>${p.rpm_limit}</td>
          <td class="${p.circuit_open ? "bad" : "ok"}">${p.circuit_open ? `open (${p.open_for_s}s)` : "closed"}</td></tr>`;
      }).join("");

    if (d.eval) {
      const e = d.eval, s = e.summary || {};
      $("#eval").innerHTML = `<table class="t">${Object.entries(s).map(([k, v]) =>
        `<tr><td>${esc(k.replaceAll("_", " "))}</td><td><b>${typeof v === "number" && v <= 1 && !k.includes("ms") && !k.includes("cases") ? pct(v) : esc(v)}</b></td></tr>`).join("")}</table>
        <div class="muted" style="margin-top:6px;font-size:12px">Run ${esc(e.run_at)} · ${e.cases} cases · mode: ${esc(e.mode)}</div>`;
    }
    const c = (d.manifest || {}).counts || {};
    $("#kbase").innerHTML = `<table class="t">
      <tr><td>Built</td><td>${esc(d.manifest.built_at)}</td></tr>
      <tr><td>Embeddings</td><td>${esc(d.embeddings_in_use)}</td></tr>
      <tr><td>DarGlobal projects / chunks</td><td>${c.darglobal_projects} / ${c.darglobal_chunks}</td></tr>
      <tr><td>Wasalt listings</td><td>${c.wasalt_listings} (${esc(JSON.stringify(c.wasalt_by_purpose || {}))})</td></tr>
      <tr><td>Top cities</td><td>${esc(Object.entries(c.wasalt_by_city || {}).slice(0, 6).map(([k, v]) => `${k} ${v}`).join(", "))}</td></tr>
      <tr><td>Response cache</td><td>${d.cache.entries} entries · ${d.cache.hits} hits</td></tr>
      <tr><td>Log storage</td><td>${esc(m.storage)}${d.durable_logs && d.durable_logs.enabled ? ` · written ${d.durable_logs.written} · failed ${d.durable_logs.failed} · dropped ${d.durable_logs.dropped}` : " (local only, resets on restart)"}</td></tr></table>`;

    $("#traces").innerHTML = `<tr><th>Time (UTC)</th><th>Question</th><th>Intent</th><th>Model</th><th>Total</th><th>Fallbacks</th><th>👍/👎</th></tr>` +
      traces.map((t, i) => `<tr class="trace" data-i="${i}">
        <td>${new Date(t.ts * 1000).toISOString().slice(5, 16).replace("T", " ")}</td>
        <td>${esc(t.question.slice(0, 90))}</td><td>${esc(t.intent)}${t.cache_hit ? " · cache" : ""}</td>
        <td>${esc((t.model || "").split("/").pop())}</td><td>${Math.round(t.total_ms)} ms</td>
        <td class="${t.attempts.length > 1 ? "warn" : ""}">${Math.max(0, t.attempts.length - 1)}</td>
        <td>${t.feedback === 1 ? "👍" : t.feedback === -1 ? "👎" : ""}</td></tr>`).join("");
    document.querySelectorAll(".trace").forEach((row) => (row.onclick = () => {
      const t = traces[Number(row.dataset.i)];
      const pre = $("#traceDetail");
      pre.textContent = JSON.stringify(t, null, 2); pre.hidden = false;
      pre.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }));
  }
})();
