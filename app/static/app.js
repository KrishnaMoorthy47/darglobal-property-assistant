(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const chat = $("#chat"), input = $("#input"), form = $("#composer"), send = $("#send");
  const store = {
    get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  };

  let source = store.get("pia.source", "any");
  let history = [];          // [{role, content}]
  let busy = false;
  const sessionId = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random())).slice(0, 24);

  // theme
  const savedTheme = store.get("pia.theme", null);
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  $("#themeBtn").onclick = () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    store.set("pia.theme", document.documentElement.dataset.theme);
  };

  // sidebar
  $("#menuBtn").onclick = () => $("#sidebar").classList.toggle("open");
  document.querySelectorAll(".segmented button").forEach((b) => {
    const on = b.dataset.source === source;
    b.classList.toggle("active", on); b.setAttribute("aria-checked", on);
    b.onclick = () => {
      source = b.dataset.source; store.set("pia.source", source);
      document.querySelectorAll(".segmented button").forEach((x) => {
        x.classList.toggle("active", x === b); x.setAttribute("aria-checked", x === b);
      });
    };
  });
  $("#newChat").onclick = () => {
    history = [];
    chat.querySelectorAll(".msg").forEach((m) => m.remove());
    $("#hero").style.display = "";
    $("#sidebar").classList.remove("open");
    input.focus();
  };

  // composer
  const autosize = () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; };
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
  });
  form.addEventListener("submit", (e) => { e.preventDefault(); ask(input.value.trim()); });
  document.querySelectorAll("#suggestions button").forEach((b) => (b.onclick = () => ask(b.textContent)));

  // rendering
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : "#");

  function renderMarkdown(text, sources) {
    text = (text || "").replace(/【(\d{1,2})[^】]*】/g, "[$1]");  // some models cite as 【1】 or 【1†source】
    const html = DOMPurify.sanitize(marked.parse(text, { breaks: true }), { ADD_ATTR: ["target"] });
    const byN = Object.fromEntries((sources || []).map((s) => [String(s.n), s]));
    // [3] -> link to source 3
    return html.replace(/\[(\d{1,2})\](?![^<]*<\/a>)/g, (m, n) => {
      const s = byN[n];
      return s ? `<a class="cite" href="${esc(safeUrl(s.url))}" target="_blank" rel="noopener" title="${esc(s.title)}">${n}</a>` : m;
    });
  }

  function cardHtml(c) {
    const tags = [c.type, c.beds && `${c.beds} bed`, c.area_sqm && `${Math.round(c.area_sqm)} m²`, c.status, c.brand]
      .filter(Boolean).slice(0, 4).map((t) => `<span class="tag">${esc(t)}</span>`).join("");
    const img = c.image ? `<img src="${esc(safeUrl(c.image))}" alt="" loading="lazy" referrerpolicy="no-referrer" onerror="this.remove()">` : "";
    const label = c.source === "wasalt" ? "Wasalt" : "DarGlobal";
    return `<a class="card" href="${esc(safeUrl(c.url))}" target="_blank" rel="noopener">
      <div class="card-img">${img}<span class="badge ${c.source}">${label}${c.purpose === "rent" ? " · Rent" : ""}</span></div>
      <div class="card-body">
        <div class="card-name">${esc(c.name)}</div>
        <div class="card-loc">${esc(c.location || "")}</div>
        <div class="card-price">${esc(c.price || (c.source === "darglobal" ? "Price on request" : ""))}</div>
        <div class="card-tags">${tags}</div>
      </div></a>`;
  }

  function addUser(text) {
    $("#hero").style.display = "none";
    const el = document.createElement("div");
    el.className = "msg user";
    el.innerHTML = `<div class="bubble" dir="auto">${esc(text)}</div>`;
    chat.appendChild(el);
  }

  function addBot() {
    const el = document.createElement("div");
    el.className = "msg bot";
    el.innerHTML = `<div class="answer" dir="auto"><div class="thinking"><span class="typing"><span></span><span></span><span></span></span><span class="phase">Understanding your question…</span></div><div class="content"></div></div>
      <div class="cards" hidden></div><details class="sources" hidden><summary></summary><ol></ol></details><div class="meta" hidden></div>`;
    chat.appendChild(el);
    return el;
  }

  const scrollDown = () => { chat.scrollTop = chat.scrollHeight; };

  // chat
  async function ask(text) {
    if (!text || busy) return;
    busy = true; send.disabled = true;
    input.value = ""; autosize();
    addUser(text);
    const el = addBot();
    const content = el.querySelector(".content"), thinking = el.querySelector(".thinking");
    let answer = "", meta = { sources: [], cards: [] }, done = null;
    scrollDown();

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Session-Id": sessionId },
        body: JSON.stringify({ message: text, history, session_id: sessionId, source }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || `Request failed (${resp.status})`);
      }
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "", pending = false;
      const paint = () => { pending = false; content.innerHTML = renderMarkdown(answer, meta.sources); scrollDown(); };

      while (true) {
        const { value, done: end } = await reader.read();
        if (end) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, idx).trim(); buf = buf.slice(idx + 2);
          if (!line.startsWith("data:")) continue;
          const ev = JSON.parse(line.slice(5));
          if (ev.type === "meta") {
            meta = ev;
            thinking.querySelector(".phase").textContent = "Writing the answer…";
            if (ev.cards && ev.cards.length) {
              const cards = el.querySelector(".cards");
              cards.innerHTML = ev.cards.map(cardHtml).join(""); cards.hidden = false;
            }
          } else if (ev.type === "token") {
            thinking.hidden = true;
            answer += ev.text;
            if (!pending) { pending = true; requestAnimationFrame(paint); }
          } else if (ev.type === "done") {
            done = ev;
          } else if (ev.type === "error") {
            throw new Error(ev.message);
          }
        }
      }
      paint();
      thinking.hidden = true;
      renderSources(el, meta.sources);
      renderMeta(el, done, meta);
      history.push({ role: "user", content: text }, { role: "assistant", content: answer.slice(0, 1500) });
      history = history.slice(-12);
    } catch (err) {
      thinking.hidden = true;
      content.innerHTML = `<p class="error">${esc(err.message || "Network error. Please try again.")}</p>`;
    } finally {
      busy = false; send.disabled = false; input.focus(); scrollDown();
    }
  }

  function renderSources(el, sources) {
    if (!sources || !sources.length) return;
    const d = el.querySelector(".sources");
    d.querySelector("summary").textContent = `${sources.length} sources`;
    d.querySelector("ol").innerHTML = sources.map((s) =>
      `<li value="${s.n}"><a href="${esc(safeUrl(s.url))}" target="_blank" rel="noopener">${esc(s.title)}</a><span class="src-kind">${s.source === "wasalt" ? "Wasalt" : "DarGlobal"}</span></li>`).join("");
    d.hidden = false;
  }

  function renderMeta(el, done, meta) {
    if (!done) return;
    const m = el.querySelector(".meta");
    const total = meta.total_matches ? `<span class="pill">${meta.total_matches} matches</span>` : "";
    m.innerHTML = `${total}<span class="spacer"></span>
      <button data-v="1" aria-label="Helpful">👍</button><button data-v="-1" aria-label="Not helpful">👎</button><button data-copy aria-label="Copy">Copy</button>`;
    m.hidden = false;
    m.querySelectorAll("[data-v]").forEach((b) => (b.onclick = () => {
      m.querySelectorAll("[data-v]").forEach((x) => x.classList.toggle("on", x === b));
      fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json", "X-Session-Id": sessionId },
        body: JSON.stringify({ trace_id: done.trace_id, value: Number(b.dataset.v) }) }).catch(() => {});
    }));
    m.querySelector("[data-copy]").onclick = (e) => {
      navigator.clipboard?.writeText(el.querySelector(".content").innerText).then(() => { e.target.textContent = "Copied"; });
    };
  }
})();
