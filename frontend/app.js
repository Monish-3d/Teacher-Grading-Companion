/* Marks-Grader frontend — talks to the FastAPI backend in server.py */
"use strict";

// Same-origin when served by FastAPI; fall back to localhost if opened as a file.
const API = location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

const state = { subject: null, subjects: [] };

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/* ------------------------------------------------------------------ utils */
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 4200);
}

function loading(btn, on) {
  btn.disabled = on;
  btn.classList.toggle("is-loading", on);
  $(".spinner", btn).hidden = !on;
}

async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON */ }
  if (!res.ok) {
    const detail = (data && (data.detail || data.message)) || `Request failed (${res.status})`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;" }[c]));
}

function band(score) {            // score out of 10 -> qualitative chip
  if (score >= 7.5) return { cls: "hi",  txt: "Strong" };
  if (score >= 5)   return { cls: "mid", txt: "Fair" };
  return { cls: "lo", txt: "Needs work" };
}

/* Build an SVG score ring. score is out of 10. */
function ring(score, { size = 118, stroke = 11, grad = "gradWC" } = {}) {
  const r = size / 2 - stroke;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(1, score / 10));
  const off = c * (1 - pct);
  const cx = size / 2;
  const big = size >= 90;
  return `
  <svg class="ring" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
    <circle class="ring-track" cx="${cx}" cy="${cx}" r="${r}" stroke-width="${stroke}"/>
    <circle class="ring-fill" cx="${cx}" cy="${cx}" r="${r}" stroke-width="${stroke}"
            stroke="url(#${grad})" stroke-dasharray="${c.toFixed(1)}"
            stroke-dashoffset="${c.toFixed(1)}" data-off="${off.toFixed(1)}"
            transform="rotate(-90 ${cx} ${cx})"/>
    <text class="ring-label" x="50%" y="${big ? "48%" : "52%"}" text-anchor="middle"
          dominant-baseline="middle" font-size="${big ? size*0.27 : size*0.3}">${score.toFixed(1)}</text>
    ${big ? `<text class="ring-cap" x="50%" y="66%" text-anchor="middle" font-size="${size*0.11}">/ 10</text>` : ""}
  </svg>`;
}

/* animate every ring inside a container from full offset -> target */
function animateRings(root) {
  requestAnimationFrame(() => {
    $$(".ring-fill", root).forEach(el => { el.style.strokeDashoffset = el.dataset.off; });
  });
}

/* Attach drag/drop + click behaviour to a dropzone. */
function bindDropzone(dz, input, title, onset) {
  const setFile = (file) => {
    if (!file) return;
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      toast("Please choose a PDF file."); return;
    }
    dz._file = file;
    title.textContent = "📎 " + file.name;
    dz.classList.add("has-file");
    onset && onset(file);
  };
  input.addEventListener("change", () => setFile(input.files[0]));
  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", e => setFile(e.dataTransfer.files[0]));
}

/* ------------------------------------------------------------------ subjects */
function renderSubjects() {
  const box = $("#subjectPicker");
  const pills = state.subjects.map(s => {
    const building = s.status && s.status !== "ready";
    const active = s.id === state.subject;
    return `<button class="subject-pill ${active ? "is-active" : ""} ${building ? "building" : ""}"
             data-id="${s.id}" ${building ? "disabled" : ""} title="${building ? "Building knowledge base…" : ""}">
             ${esc(s.label)}${building ? "<span class='pill-dot'></span>" : ""}</button>`;
  }).join("");
  box.innerHTML = pills + `<button class="subject-pill add-pill" id="addSubjectBtn">＋ Add textbook</button>`;

  $$(".subject-pill", box).forEach(p => {
    if (p.id === "addSubjectBtn") { p.addEventListener("click", openAddModal); return; }
    if (p.disabled) return;
    p.addEventListener("click", () => { state.subject = p.dataset.id; renderSubjects(); });
  });
}

async function fetchSubjects({ pickDefault = false } = {}) {
  const subs = await api("/api/subjects");
  state.subjects = subs;
  if (pickDefault && !state.subject && subs[0]) state.subject = subs[0].id;
  if (!subs.find(s => s.id === state.subject) && subs[0]) state.subject = subs[0].id; // stale selection
  renderSubjects();
  return subs;
}

/* Poll until a subject finishes building (or errors). */
function pollSubjectReady(id, onTick) {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const tick = async () => {
      let subs;
      try { subs = await fetchSubjects(); } catch (e) { return reject(e); }
      const s = subs.find(x => x.id === id);
      if (!s) return reject(new Error("Subject disappeared."));
      if (s.status === "ready") return resolve(s);
      if (String(s.status).startsWith("error")) return reject(new Error(s.status.replace(/^error:\s*/, "")));
      if (Date.now() - started > 15 * 60 * 1000) return reject(new Error("Timed out while building."));
      onTick && onTick(s);
      setTimeout(tick, 2500);
    };
    tick();
  });
}

/* ------------------------------------------------------------------ add-textbook modal */
function openAddModal() {
  const m = $("#addModal");
  $("#add-name").value = "";
  const dz = $("#add-dropzone");
  dz._file = null;
  dz.classList.remove("has-file");
  $("#add-file-title").textContent = "Drop a textbook PDF here or click to browse";
  $("#add-status").hidden = true;
  $("#add-status").classList.remove("err");
  $("#add-run").disabled = true;
  m.hidden = false;
}
function closeAddModal() { $("#addModal").hidden = true; }

function updateAddRunState() {
  $("#add-run").disabled = !($("#add-name").value.trim() && $("#add-dropzone")._file);
}

async function runAddSubject() {
  const name = $("#add-name").value.trim();
  const file = $("#add-dropzone")._file;
  if (!name) { toast("Give the subject a name."); return; }
  if (!file) { toast("Choose a textbook PDF."); return; }

  const btn = $("#add-run");
  const status = $("#add-status");
  loading(btn, true);
  status.classList.remove("err");
  status.hidden = false;
  status.textContent = "Uploading…";

  try {
    const fd = new FormData();
    fd.append("label", name);
    fd.append("file", file);
    const res = await api("/api/subjects", { method: "POST", body: fd });

    status.innerHTML = `<span class="spin-inline"></span>Building the knowledge base — reading, chunking and embedding the textbook. This can take a minute or two for a full book…`;

    await pollSubjectReady(res.id);
    state.subject = res.id;
    await fetchSubjects();
    toast(`“${name}” is ready — you can grade and generate questions from it.`);
    closeAddModal();
  } catch (e) {
    status.classList.add("err");
    status.textContent = "Couldn't build that textbook: " + e.message;
  } finally {
    loading(btn, false);
  }
}

/* ------------------------------------------------------------------ tabs */
function setupTabs() {
  const tabs = $$(".tab");
  const glow = $("#tabGlow");
  const move = (tab) => { glow.style.left = tab.offsetLeft + "px"; glow.style.width = tab.offsetWidth + "px"; };
  tabs.forEach(tab => tab.addEventListener("click", () => {
    tabs.forEach(t => t.classList.remove("is-active"));
    tab.classList.add("is-active");
    const name = tab.dataset.tab;
    $$(".panel").forEach(p => p.classList.toggle("is-active", p.dataset.panel === name));
    move(tab);
  }));
  const active = $(".tab.is-active");
  if (active) move(active);
  window.addEventListener("resize", () => { const a = $(".tab.is-active"); if (a) move(a); });
}

/* ------------------------------------------------------------------ 1) grade single */
async function runGrade() {
  const question = $("#g-question").value.trim();
  const answer   = $("#g-answer").value.trim();
  if (!question || !answer) { toast("Please enter both a question and an answer."); return; }
  const btn = $("#g-run");
  loading(btn, true);
  try {
    const r = await api("/api/grade", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject: state.subject, question, answer }),
    });
    renderGrade(r);
  } catch (e) {
    toast(e.message);
  } finally {
    loading(btn, false);
  }
}

function renderGrade(r) {
  const b = band(r.final_score);
  const box = $("#g-result");
  box.innerHTML = `
    <div class="result-in">
      <div class="score-hero">
        ${ring(r.final_score, { size: 118, grad: "gradWC" })}
        <div class="score-hero-text">
          <h3>Final score</h3>
          <p>Weighted blend · 0.6 LLM / 0.2 similarity / 0.2 keyword</p>
          <span class="chip ${b.cls}">${b.txt}</span>
        </div>
      </div>
      <div class="signals">
        <div class="signal">${ring(r.llm_score,        { size: 78, stroke: 8, grad: "gradCool" })}<div class="mini-cap">LLM rubric</div></div>
        <div class="signal">${ring(r.similarity_score, { size: 78, stroke: 8, grad: "gradWarm" })}<div class="mini-cap">Similarity</div></div>
        <div class="signal">${ring(r.keyword_score,    { size: 78, stroke: 8, grad: "gradMint" })}<div class="mini-cap">Keywords</div></div>
      </div>
      <div class="feedback">
        <div class="fb-cap">Feedback</div>
        <p>${esc(r.feedback || "No feedback returned.")}</p>
      </div>
    </div>`;
  animateRings(box);
}

/* ------------------------------------------------------------------ 2) grade sheet */
async function runSheet() {
  const dz = $("#dropzone");
  const file = dz._file;
  if (!file) { toast("Upload a PDF answer sheet first."); return; }
  const btn = $("#s-run");
  loading(btn, true);
  try {
    const fd = new FormData();
    fd.append("subject", state.subject);
    fd.append("file", file);
    const data = await api("/api/grade-sheet", { method: "POST", body: fd });
    renderSheet(data);
  } catch (e) {
    toast(e.message);
  } finally {
    loading(btn, false);
  }
}

function renderSheet(data) {
  const box = $("#s-result");
  const rows = data.results.map((r, i) => {
    if (r.error) {
      return `<div class="q-item"><div class="q-num">Question ${i + 1}</div>
        <div class="q-text">${esc(r.question || "(unreadable)")}</div>
        <div class="q-fb">Could not grade: ${esc(r.error)}</div></div>`;
    }
    return `<div class="q-item">
      <div class="q-head">
        <div>
          <div class="q-num">Question ${i + 1}</div>
          <div class="q-text">${esc(r.question)}</div>
        </div>
        <div class="q-badge">${r.final_score.toFixed(1)}</div>
      </div>
      <div class="q-ans">${esc(r.answer)}</div>
      <div class="q-fb">${esc(r.feedback || "")}</div>
    </div>`;
  }).join("");

  box.innerHTML = `
    <div class="result-in">
      <div class="sheet-summary">
        ${ring(data.average, { size: 96, stroke: 9, grad: "gradWC" })}
        <div>
          <h3>Average score</h3>
          <p>${data.count} question${data.count === 1 ? "" : "s"} graded from the sheet</p>
        </div>
      </div>
      <div class="q-list">${rows || "<p class='placeholder'>No gradable questions found.</p>"}</div>
    </div>`;
  animateRings(box);
}

/* ------------------------------------------------------------------ 3) generate mcqs */
async function runMcq() {
  const topic = $("#m-topic").value.trim();
  if (!topic) { toast("Enter a topic to generate questions."); return; }
  const difficulty = $("#m-difficulty").value;
  const num = Math.max(1, Math.min(10, parseInt($("#m-count").value || "5", 10)));
  const btn = $("#m-run");
  loading(btn, true);
  try {
    const data = await api("/api/generate-mcqs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject: state.subject, topic, difficulty, num_questions: num }),
    });
    renderMcq(data);
  } catch (e) {
    toast(e.message);
  } finally {
    loading(btn, false);
  }
}

function renderMcq(data) {
  const box = $("#m-result");
  if (!data.questions || !data.questions.length) {
    box.innerHTML = "<div class='placeholder'><p>No questions came back — try a broader topic.</p></div>";
    return;
  }
  const letters = ["A", "B", "C", "D", "E", "F"];
  const cards = data.questions.map((q, qi) => {
    const opts = (q.options || []).map((o, oi) => {
      const correct = String(o).trim() === String(q.answer).trim();
      return `<div class="opt" data-correct="${correct}">
        <span class="bullet">${letters[oi] || "•"}</span><span>${esc(o)}</span></div>`;
    }).join("");
    return `<div class="mcq" data-answer="${esc(q.answer)}">
      <div class="mcq-q"><span class="mcq-i">Q${qi + 1}.</span>${esc(q.question)}</div>
      <div class="opts">${opts}</div>
      <div class="reveal">✓ Correct answer: ${esc(q.answer)}</div>
    </div>`;
  }).join("");

  box.innerHTML = `
    <div class="result-in">
      <div class="mcq-head">
        <h3>${data.questions.length} question${data.questions.length === 1 ? "" : "s"} · ${esc(data.difficulty)}</h3>
        <p>Topic: ${esc(data.topic)} — click an option to check it.</p>
      </div>
      <div class="mcq-list">${cards}</div>
    </div>`;

  $$(".mcq", box).forEach(card => {
    $$(".opt", card).forEach(opt => opt.addEventListener("click", () => {
      if (card.classList.contains("answered")) return;
      card.classList.add("answered");
      const chosenCorrect = opt.dataset.correct === "true";
      $$(".opt", card).forEach(o => { if (o.dataset.correct === "true") o.classList.add("correct"); });
      if (!chosenCorrect) opt.classList.add("wrong");
      $(".reveal", card).classList.add("show");
    }));
  });
}

/* ------------------------------------------------------------------ health + boot */
async function pingHealth() {
  const el = $("#health");
  try {
    await api("/api/health");
    el.textContent = "backend online";
    el.className = "ok";
  } catch (_) {
    el.textContent = "backend offline";
    el.className = "down";
  }
}

function boot() {
  setupTabs();

  // sheet dropzone
  bindDropzone($("#dropzone"), $("#s-file"), $("#dropzone-title"), () => { $("#s-run").disabled = false; });
  // modal dropzone
  bindDropzone($("#add-dropzone"), $("#add-file"), $("#add-file-title"), updateAddRunState);

  $("#g-run").addEventListener("click", runGrade);
  $("#s-run").addEventListener("click", runSheet);
  $("#m-run").addEventListener("click", runMcq);

  // modal wiring
  $("#add-name").addEventListener("input", updateAddRunState);
  $("#add-run").addEventListener("click", runAddSubject);
  $("#addClose").addEventListener("click", closeAddModal);
  $("#addModal").addEventListener("click", e => { if (e.target.id === "addModal") closeAddModal(); });
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeAddModal(); });

  fetchSubjects({ pickDefault: true });
  pingHealth();
}

document.addEventListener("DOMContentLoaded", boot);
