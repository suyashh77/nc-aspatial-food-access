
(() => {
  const root = document.querySelector('.nt');
  const fmtInt = d3.format(','), pct = d3.format('.1%'), pct0 = d3.format('.0%'), f3 = d3.format('.3f'), f2 = d3.format('.2f');
  const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });
  const FMT = {
    reviews_clustered: (v) => fmtInt(v), reviews_with_text: (v) => fmtInt(v), stores: (v) => fmtInt(v), topics: (v) => fmtInt(v),
    teresa_topics: (v) => fmtInt(v), outlier_share: (v) => pct(v), coherence_cv: (v) => f3(v), topic_diversity: (v) => f3(v),
    teresa_coherence: (v) => f3(v), teresa_diversity: (v) => f3(v),
    topic: (v) => String(v), reviews: (v) => fmtInt(v), share: (v) => pct(v), avg_stars: (v) => f2(v), membership: (v) => f2(v),
    raleigh_share: (v) => pct0(v),
  };
  const fmt = (field, v) => (v === null || v === undefined || v === '') ? '—' : (FMT[field] ? FMT[field](Number(v)) : String(v));
  const resolve = (c) => { const s = document.createElement('span'); s.style.color = c; root.appendChild(s); const v = getComputedStyle(s).color; s.remove(); return v; };
  const withAlpha = (c, a) => { const x = d3.color(c); if (!x) return c; x.opacity = a; return x.formatRgb(); };

  let reviewKind = 'typical', sortKey = 'reviews', sortDir = -1, query = '';
  const selected = () => { const t = dash.params().topic; return t === undefined || t === null ? 0 : Number(t); };
  const pick = (t) => dash.setParams({ topic: Number(t) });
  if (dash.params().topic === undefined) dash.setParams({ topic: 0 });

  const fail = (el) => { el.innerHTML = ''; const d = document.createElement('div'); d.className = 'nt-fail'; d.textContent = 'Failed to load'; el.appendChild(d); };
  const done = (el) => el.classList.remove('dash-skeleton');

  // ---------- summary ----------
  function drawSummary() {
    const S = dash.data('summary');
    root.querySelectorAll('[data-source="summary"]').forEach((el) => {
      if (S.status === 'ok') { el.textContent = fmt(el.dataset.field, S.data[0][el.dataset.field]); done(el); }
      else if (S.status === 'error') { el.textContent = '—'; done(el); }
      else { el.textContent = ''; el.classList.add('dash-skeleton'); }
    });
  }

  // ---------- selected topic ----------
  function topicRow(T) { const t = selected(); return T.data.find((r) => Number(r.topic) === t) || T.data.find((r) => Number(r.topic) === 0); }
  function drawDetail(T) {
    const box = root.querySelector('#topic');
    const els = box.querySelectorAll('[data-source="topics"][data-field]');
    if (T.status !== 'ok') { els.forEach((el) => { el.textContent = '—'; el.removeAttribute('data-where'); }); return; }
    const r = topicRow(T);
    els.forEach((el) => { el.setAttribute('data-where', 'topic=' + encodeURIComponent(r.topic)); el.textContent = fmt(el.dataset.field, r[el.dataset.field]); });
    const ramp = d3.interpolateRgb(resolve('var(--cds-chart-muted)'), resolve(dash.colors[0]));
    const stars = [1, 2, 3, 4, 5].map((k) => ({ k, v: Number(r['star' + k]) || 0, c: ramp((k - 1) / 4) }));
    const bar = box.querySelector('.nt-stars'), key = box.querySelector('.nt-starkey');
    bar.replaceChildren(...stars.map((s) => { const i = document.createElement('i'); i.style.width = (s.v * 100) + '%'; i.style.background = s.c; return i; }));
    bar.setAttribute('aria-label', 'Star rating mix: ' + stars.map((s) => s.k + ' star ' + pct0(s.v)).join(', '));
    key.replaceChildren(...stars.map((s) => { const sp = document.createElement('span'); const i = document.createElement('i'); i.style.background = s.c; sp.append(i, s.k + '★ ' + pct0(s.v)); return sp; }));
  }

  // ---------- map ----------
  const mapBox = root.querySelector('.nt-map'), canvas = mapBox.querySelector('canvas'), tip = mapBox.querySelector('.nt-tip');
  let mapState = null;
  function drawMap(M, T) {
    if (M.status === 'error' || T.status === 'error') { fail(mapBox); return; }
    if (M.status !== 'ok' || T.status !== 'ok') return;
    if (!mapBox.contains(canvas)) mapBox.replaceChildren(canvas, tip);
    const W = Math.max(280, mapBox.clientWidth), H = Math.round(Math.min(560, Math.max(320, W * 0.72))), dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); canvas.style.height = H + 'px';
    mapBox.style.minHeight = H + 'px';
    const pts = M.data, sel = selected(), pad = 10;
    const x = d3.scaleLinear(d3.extent(pts, (d) => Number(d.x)), [pad, W - pad]);
    const y = d3.scaleLinear(d3.extent(pts, (d) => Number(d.y)), [H - pad, pad]);
    const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const cSel = resolve(dash.colors[0]), cOth = withAlpha(resolve('var(--cds-chart-muted)'), 0.55), cOut = withAlpha(resolve('var(--cds-chart-muted)'), 0.22);
    const dot = (d, r) => { ctx.moveTo(x(d.x) + r, y(d.y)); ctx.arc(x(d.x), y(d.y), r, 0, Math.PI * 2); };
    ctx.beginPath(); ctx.fillStyle = cOut; for (const d of pts) if (Number(d.topic) === -1 && sel !== -1) dot(d, 1.1); ctx.fill();
    ctx.beginPath(); ctx.fillStyle = cOth; for (const d of pts) if (Number(d.topic) !== -1 && Number(d.topic) !== sel) dot(d, 1.2); ctx.fill();
    ctx.beginPath(); ctx.fillStyle = cSel; for (const d of pts) if (Number(d.topic) === sel) dot(d, 1.9); ctx.fill();
    const r = topicRow(T);
    if (r && Number(r.topic) !== -1) {
      const lx = x(Number(r.cx)), ly = y(Number(r.cy));
      ctx.font = '600 12px ' + getComputedStyle(root).fontFamily; ctx.textBaseline = 'middle';
      const text = r.label, tw = ctx.measureText(text).width;
      const tx = Math.min(Math.max(lx + 8, 4), W - tw - 8);
      ctx.fillStyle = resolve('var(--color-panel)'); ctx.globalAlpha = 0.85; ctx.fillRect(tx - 4, ly - 9, tw + 8, 18); ctx.globalAlpha = 1;
      ctx.fillStyle = resolve('var(--color-fg)'); ctx.fillText(text, tx, ly);
    }
    root.querySelector('.lg-sel').style.background = cSel; root.querySelector('.lg-oth').style.background = cOth; root.querySelector('.lg-out').style.background = cOut;
    const labels = new Map(T.data.map((t) => [Number(t.topic), t.label]));
    mapState = { qt: d3.quadtree(pts, (d) => x(d.x), (d) => y(d.y)), labels, W, H };
  }
  const nearest = (e) => { if (!mapState) return null; const b = canvas.getBoundingClientRect(); return { d: mapState.qt.find(e.clientX - b.left, e.clientY - b.top, 10), px: e.clientX - b.left, py: e.clientY - b.top }; };
  canvas.addEventListener('pointermove', (e) => {
    const n = nearest(e); if (!n || !n.d) { tip.hidden = true; return; }
    const t = Number(n.d.topic);
    tip.replaceChildren(); const k = document.createElement('div'); k.className = 'k'; k.textContent = t === -1 ? 'No topic' : 'Topic ' + t;
    const v = document.createElement('div'); v.textContent = mapState.labels.get(t) || ''; tip.append(k, v); tip.hidden = false;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = Math.min(n.px + 12, mapState.W - tw - 4) + 'px'; tip.style.top = Math.min(n.py + 12, mapState.H - th - 4) + 'px';
  });
  canvas.addEventListener('pointerleave', () => (tip.hidden = true));
  canvas.addEventListener('click', (e) => { const n = nearest(e); if (n && n.d) pick(n.d.topic); });

  // ---------- reviews ----------
  const revBox = root.querySelector('.nt-reviews');
  const stars = (n) => (n >= 1 && n <= 5) ? '★'.repeat(n) + '☆'.repeat(5 - n) : 'no rating';
  function drawReviews(R) {
    if (R.status === 'error') { fail(revBox); return; }
    if (R.status !== 'ok') { revBox.replaceChildren(); revBox.classList.add('dash-skeleton'); return; }
    done(revBox);
    const t = selected();
    let rows = R.data.filter((r) => Number(r.topic) === t && (t === -1 || r.kind === reviewKind));
    if (reviewKind === 'typical') rows.sort((a, b) => Number(b.membership) - Number(a.membership));
    revBox.replaceChildren(...rows.map((r) => {
      const el = document.createElement('article'); el.className = 'nt-rev';
      const meta = document.createElement('div'); meta.className = 'meta';
      const st = document.createElement('span'); st.className = 'st'; st.textContent = stars(Number(r.stars));
      const where = document.createElement('span'); where.textContent = [r.store, r.city].filter(Boolean).join(' · ');
      const mem = document.createElement('span'); mem.textContent = 'membership ' + f2(Number(r.membership));
      meta.append(st, where, mem);
      const p = document.createElement('p'); p.textContent = r.text;
      el.append(meta, p); return el;
    }));
    if (!rows.length) { const p = document.createElement('p'); p.className = 'nt-note'; p.textContent = 'No sample reviews for this topic.'; revBox.appendChild(p); }
  }
  const setKind = (k) => { reviewKind = k; root.querySelector('#rv-typical').setAttribute('aria-pressed', String(k === 'typical')); root.querySelector('#rv-random').setAttribute('aria-pressed', String(k === 'random')); drawReviews(dash.data('reviews_sample')); };
  root.querySelector('#rv-typical').addEventListener('click', () => setKind('typical'));
  root.querySelector('#rv-random').addEventListener('click', () => setKind('random'));

  // ---------- topics table ----------
  const tbody = root.querySelector('table[data-source="topics"] tbody');
  const NUMERIC = new Set(['topic', 'reviews', 'avg_stars', 'stores']);
  function drawTable(T) {
    if (T.status !== 'ok') { tbody.replaceChildren(); return; }
    const max = d3.max(T.data.filter((r) => Number(r.topic) !== -1), (r) => Number(r.reviews)) || 1;
    const q = query.trim().toLowerCase(), sel = selected(), accent = resolve(dash.colors[0]), muted = resolve('var(--cds-chart-muted)');
    root.style.setProperty('--nt-accent', accent);
    let rows = T.data.filter((r) => !q || [r.label, r.keywords, r.keybert, r.mmr, r.raleigh_label].join(' ').toLowerCase().includes(q));
    rows = rows.slice().sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      const c = NUMERIC.has(sortKey) ? Number(av) - Number(bv) : String(av).localeCompare(String(bv));
      return c * sortDir;
    });
    tbody.replaceChildren(...rows.map((r) => {
      const tr = document.createElement('tr'); if (Number(r.topic) === sel) tr.className = 'on';
      const td = (text, cls) => { const c = document.createElement('td'); if (cls) c.className = cls; c.textContent = text; return c; };
      const n = document.createElement('td'); n.className = 'num';
      const wrap = document.createElement('div'); wrap.className = 'nt-bar';
      const bar = document.createElement('i'); bar.style.width = (Number(r.topic) === -1 ? 0 : Number(r.reviews) / max * 6) + 'rem'; bar.style.background = Number(r.topic) === sel ? accent : muted;
      const lab = document.createElement('span'); lab.textContent = fmtInt(Number(r.reviews));
      wrap.append(bar, lab); n.appendChild(wrap);
      tr.append(td(String(r.topic), 'num'), td(r.label, 'nt-label'), n, td(f2(Number(r.avg_stars)), 'num'), td(fmtInt(Number(r.stores)), 'num'), td(r.raleigh_label, 'nt-label'));
      tr.addEventListener('click', () => pick(r.topic));
      return tr;
    }));
    root.querySelectorAll('table[data-source="topics"] th').forEach((th) => th.setAttribute('aria-sort', th.dataset.field === sortKey ? (sortDir > 0 ? 'ascending' : 'descending') : 'none'));
  }
  root.querySelectorAll('table[data-source="topics"] th button').forEach((b) => b.addEventListener('click', () => {
    const k = b.dataset.sort; if (k === sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = (k === 'label' || k === 'raleigh_label' || k === 'topic') ? 1 : -1; }
    drawTable(dash.data('topics'));
  }));
  root.querySelector('#topic-search').addEventListener('input', (e) => { query = e.target.value; drawTable(dash.data('topics')); });

  // ---------- grid ----------
  const gbody = root.querySelector('table.nt-grid tbody');
  function drawGrid(G) {
    if (G.status !== 'ok') { gbody.replaceChildren(); return; }
    gbody.replaceChildren(...G.data.map((r) => {
      const tr = document.createElement('tr');
      const td = (text, cls) => { const c = document.createElement('td'); if (cls) c.className = cls; c.textContent = text; return c; };
      if (r.chosen === true || r.chosen === 'true') tr.className = 'on';
      tr.append(td(String(r.min_cluster_size), 'num'), td(fmtInt(Number(r.n_topics)), 'num'), td(pct(Number(r.outlier_share)), 'num'), td(f3(Number(r.coherence_cv)), 'num'), td(f3(Number(r.topic_diversity)), 'num'));
      return tr;
    }));
  }

  dash.onData(() => {
    const T = dash.data('topics');
    drawSummary();
    drawDetail(T);
    drawMap(dash.data('map_points'), T);
    drawReviews(dash.data('reviews_sample'));
    drawTable(T);
    drawGrid(dash.data('grid'));
  });

  const named = document.getElementById(location.hash.slice(1));
  if (named) requestAnimationFrame(() => { named.scrollIntoView(); dash.setLink(named.id); });
})();
