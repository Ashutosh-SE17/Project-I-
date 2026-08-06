/* ============================================================
   Electrametrics — frontend

   Leaflet replaces the Folium iframe: a static iframe can't send
   click events to the parent page, so district selection needs
   the map rendered in-page.
   ============================================================ */

const PARTY = {
  'rsp':             { label: 'Rastriya Swatantra',  colour: '#1B7FA8', symbol: 'bell'   },
  'uml':             { label: 'CPN (UML)',           colour: '#C8322D', symbol: 'sun'    },
  'nc':              { label: 'Nepali Congress',     colour: '#2C7A4B', symbol: 'tree'   },
  'maoist':          { label: 'Maoist Centre',       colour: '#8B2635', symbol: 'hammer' },
  'shram sanskriti': { label: 'Shram Sanskriti',     colour: '#B5762A', symbol: 'spade'  },
};
const NO_PARTY = { label: 'Unaffiliated', colour: '#9A96A6', symbol: 'ballot' };

/* Ballot symbols, drawn rather than emoji — emoji render
   inconsistently across platforms and can't inherit colour. */
const SYMBOL = {
  bell:   '<path d="M12 3v2M6.5 20h11a1 1 0 0 0 .95-1.32C17.6 16 17 13.6 17 11a5 5 0 0 0-10 0c0 2.6-.6 5-1.45 7.68A1 1 0 0 0 6.5 20Z"/><path d="M10.5 20a1.5 1.5 0 0 0 3 0"/>',
  sun:    '<circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.2 5.2l1.6 1.6M17.2 17.2l1.6 1.6M18.8 5.2l-1.6 1.6M6.8 17.2l-1.6 1.6"/>',
  tree:   '<path d="M12 3 6.5 11h3L5 18h14l-4.5-7h3L12 3Z"/><path d="M12 18v3.5"/>',
  hammer: '<path d="M14.5 3 21 9.5l-3 3-6.5-6.5 3-3Z"/><path d="M12.5 8 4 16.5V21h4.5L17 12.5"/>',
  spade:  '<path d="M12 3c0 4-6.5 5.5-6.5 10A4.5 4.5 0 0 0 12 16a4.5 4.5 0 0 0 6.5-3C18.5 8.5 12 7 12 3Z"/><path d="M12 16v5"/>',
  ballot: '<rect x="3.5" y="5.5" width="17" height="13" rx="2"/><path d="M8 11.5l2.5 2.5L16 9"/>',
};

/* Races. district must match the DISTRICT field in the GeoJSON —
   note the file spells Chitwan as "Chitawan". */
const RACES = [
  { seat: 'Jhapa-5',         district: 'Jhapa',      total: 103000,
    a: { key: 'balen',              name: 'Balendra Shah',    party: 'rsp',             votes: 68348 },
    b: { key: 'kp-oli',             name: 'KP Sharma Oli',    party: 'uml',             votes: 18734 } },
  { seat: 'Chitwan-2',       district: 'Chitawan',   total: 78000,
    a: { key: 'rabi-lamichhane',    name: 'Rabi Lamichhane',  party: 'rsp',             votes: 54402 },
    b: { key: 'mina-kharel',        name: 'Mina Kharel',      party: 'nc',              votes: 14564 } },
  { seat: 'Sunsari-1',       district: 'Sunsari',    total: 80000,
    a: { key: 'harka-sampang',      name: 'Harka Sampang',    party: 'shram sanskriti', votes: 35741 },
    b: { key: 'goma-tamang',        name: 'Goma Tamang',      party: 'rsp',             votes: 27249 } },
  { seat: 'Sarlahi-4',       district: 'Sarlahi',    total: 78000,
    a: { key: 'amresh-kumar-singh', name: 'Amresh Kumar Singh', party: 'rsp',           votes: 35688 },
    b: { key: 'gagan-thapa',        name: 'Gagan Thapa',      party: 'nc',              votes: 22838 } },
  { seat: 'Eastern Rukum-1', district: 'Rukum East', total: 18000,
    a: { key: 'prachanda',          name: 'Prachanda',        party: 'maoist',          votes: 10240 },
    b: { key: 'leelamani-gautam',   name: 'Leelamani Gautam', party: 'uml',             votes: 3462 } },
];

const byDistrict = Object.fromEntries(RACES.map(r => [r.district.toLowerCase(), r]));
const $ = id => document.getElementById(id);

let partyData = null;
let activeRace = null;

/* ── helpers ─────────────────────────────────────────────── */

const party = k => PARTY[k] || NO_PARTY;

function chip(partyKey, large) {
  const p = party(partyKey);
  return `<span class="chip${large ? ' chip--lg' : ''}" style="color:${p.colour}"
            title="${p.label}" aria-label="${p.label}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              ${SYMBOL[p.symbol]}
            </svg></span>`;
}

function initials(name) {
  return name.split(/\s+/).slice(0, 2).map(w => w[0]).join('').toUpperCase();
}

function portrait(c) {
  // Drop candidate photos in static/img/candidates/<key>.jpg .
  // Missing files fall back to initials rather than a broken image.
  return `<div class="portrait">
            <img src="/static/img/candidates/${c.key}.jpg" alt="" width="46" height="46"
                 style="width:100%;height:100%;object-fit:cover"
                 onerror="this.replaceWith(document.createTextNode('${initials(c.name)}'))">
          </div>`;
}

const pct = (v, t) => (100 * v / t).toFixed(1);

/* ── map ─────────────────────────────────────────────────── */

const map = L.map('map', {
  center: [28.35, 84.1],
  zoom: 7,
  zoomControl: true,
  attributionControl: false,
  scrollWheelZoom: false,
});

let layer = null;

fetch('/static/data/nepal_districts.geojson')
  .then(r => r.json())
  .then(geo => {
    layer = L.geoJSON(geo, {
      style: styleFor,
      onEachFeature: bindFeature,
    }).addTo(map);
    map.fitBounds(layer.getBounds(), { padding: [12, 12] });
  })
  .catch(() => {
    $('map').innerHTML =
      '<p style="padding:24px;color:#7C7788;font-size:13px">' +
      'Map data not found. Run <code>python prepare_map_data.py</code> ' +
      'to generate static/data/nepal_districts.geojson.</p>';
  });

function styleFor(f) {
  const race = byDistrict[(f.properties.district || '').toLowerCase()];
  if (!race) {
    return { fillColor: '#EFEEE8', color: '#E4E2DA', weight: 0.7, fillOpacity: 1 };
  }
  const winner = race.a.votes >= race.b.votes ? race.a : race.b;
  return {
    fillColor: party(winner.party).colour,
    color: '#FFFFFF',
    weight: 1.2,
    fillOpacity: 0.82,
  };
}

function bindFeature(f, lyr) {
  const name = f.properties.district || '';
  const race = byDistrict[name.toLowerCase()];

  lyr.bindTooltip(race ? `${name} · ${race.seat}` : name,
                  { className: 'seat', sticky: true });

  if (!race) return;

  lyr.on({
    mouseover: e => e.target.setStyle({ weight: 2.4, fillOpacity: 0.95 }),
    mouseout:  e => layer.resetStyle(e.target),
    click:     () => showRace(race),
  });
}

/* ── legend ──────────────────────────────────────────────── */

$('legend').innerHTML = Object.entries(PARTY).map(([k, p]) =>
  `<li><i style="background:${p.colour}"></i>${p.label}</li>`).join('') +
  `<li><i style="background:#EFEEE8;border:1px solid #E4E2DA"></i>No validated race</li>`;

/* ── race detail ─────────────────────────────────────────── */

function showRace(race) {
  activeRace = race;
  $('railEmpty').hidden = true;
  $('railBody').hidden = false;
  $('result').hidden = true;
  $('raceSeat').textContent = `${race.district} · ${race.seat}`;

  const winner = race.a.votes >= race.b.votes ? race.a : race.b;

  $('contest').innerHTML = [race.a, race.b]
    .sort((x, y) => y.votes - x.votes)
    .map((c, i) => runnerRow(c, c === winner, race.total, i))
    .join('');

  // wire the expanders
  $('contest').querySelectorAll('.runner').forEach(btn => {
    btn.addEventListener('click', () => {
      const open = btn.getAttribute('aria-expanded') === 'true';
      btn.setAttribute('aria-expanded', String(!open));
      document.getElementById(btn.dataset.panel).dataset.open = String(!open);
    });
  });

  renderVerdict(race, winner);
  renderBars(race);

  $('candidateName').innerHTML = [race.a, race.b]
    .map(c => `<option value="${c.key.replace(/-/g, ' ')}">${c.name}</option>`).join('');
}

function runnerRow(c, won, total, idx) {
  const p = party(c.party);
  const panelId = `dossier-${c.key}`;
  const prof = (typeof PROFILES !== 'undefined' && PROFILES[c.key]) || null;

  return `
    <button class="runner${won ? ' runner--won' : ''}" type="button"
            aria-expanded="false" aria-controls="${panelId}"
            data-panel="${panelId}">
      ${portrait(c)}
      ${chip(c.party)}
      <div class="runner__id">
        <div class="runner__name">${c.name}</div>
        <div class="runner__party">${p.label}</div>
      </div>
      <div class="runner__votes">${c.votes.toLocaleString()}
        <span class="runner__pct">${pct(c.votes, total)}%</span>
      </div>
      <svg class="runner__caret" viewBox="0 0 24 24" width="14" height="14"
           fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
        <path d="m6 9 6 6 6-6"/>
      </svg>
    </button>

    <div class="dossier" id="${panelId}" data-open="false">
      <div class="dossier__inner">
        ${prof ? `
          <dl>
            <div><dt>Position</dt><dd class="is-role">${prof.role}</dd></div>
            <div><dt>Education</dt><dd>${prof.education}</dd></div>
            <div><dt>Background</dt><dd>${prof.background}</dd></div>
          </dl>` : `
          <dl><div><dt>Profile</dt><dd>No background recorded.</dd></div></dl>`}
      </div>
    </div>`;
}

function renderVerdict(race, winner) {
  if (!partyData) { $('verdict').hidden = true; return; }
  const row = (partyData.races || []).find(d => d.constituency === race.seat);
  if (!row) { $('verdict').hidden = true; return; }

  $('verdict').hidden = false;
  const ok = row.national_correct;
  $('verdict').innerHTML = `
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor"
         stroke-width="2" style="flex:none;color:${ok ? '#2C7A4B' : '#B3382F'}">
      ${ok ? '<path d="M20 6 9 17l-5-5"/>' : '<path d="M18 6 6 18M6 6l12 12"/>'}
    </svg>
    <span>Party sentiment favoured <b>${party(row.national_predicted).label}</b>.
      ${party(winner.party).label} won the seat —
      ${ok ? 'prediction correct' : 'prediction incorrect'}.</span>`;
}

function renderBars(race) {
  if (!partyData) return;
  const keys = [...new Set([race.a.party, race.b.party])];
  const stats = partyData.national_party_scores || {};

  $('partyBars').innerHTML = keys.map(k => {
    const s = stats[k] || {};
    const mean = s.mean ?? s.mean_polarity ?? 0;
    const n = s.n ?? 0;
    const p = party(k);
    const w = Math.min(50, Math.abs(mean) * 50);
    const side = mean >= 0 ? `left:50%;width:${w}%` : `right:50%;width:${w}%`;
    return `
      <div class="bar">
        <div class="bar__top">
          <span>${p.label}</span>
          <span class="bar__n">${mean >= 0 ? '+' : ''}${mean.toFixed(3)} · n=${n}</span>
        </div>
        <div class="bar__track">
          <div class="bar__mid"></div>
          <div class="bar__fill" style="${side};background:${p.colour}"></div>
        </div>
      </div>`;
  }).join('');
}

/* ── candidate analysis ──────────────────────────────────── */

$('runBtn').addEventListener('click', analyse);
$('railClose').addEventListener('click', () => {
  $('railBody').hidden = true;
  $('railEmpty').hidden = false;
  if (layer) layer.setStyle(styleFor);
});

async function analyse() {
  const candidate = $('candidateName').value;
  if (!candidate) return;

  $('runBtn').disabled = true;
  $('loading').hidden = false;
  $('result').hidden = true;

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate, source: 'YouTube' }),
    });
    const data = await res.json();

    if (data.error) {
      $('loading').textContent = data.error;
      return;
    }
    renderResult(data);
  } catch (err) {
    $('loading').textContent = 'Could not reach the server. Is it still running?';
  } finally {
    $('runBtn').disabled = false;
    if (!$('result').hidden) $('loading').hidden = true;
  }
}

function renderResult(d) {
  $('loading').hidden = true;
  $('result').hidden = false;

  const p = d.win_probability ?? 0;
  $('meterFill').style.width = p + '%';
  $('meterVal').textContent = p.toFixed(1) + '%';

  const ci = d.confidence_interval;
  $('ciText').textContent = ci ? `95% CI ${ci[0]}–${ci[1]} · n=${d.n_comments ?? 0}` : '';

  const pos = d.counts?.Positive ?? 0;
  const neg = d.counts?.Negative ?? 0;
  const tot = pos + neg || 1;
  $('split').innerHTML = `
    <div class="split__seg" style="flex:${pos};background:#2C7A4B">${Math.round(100 * pos / tot)}%</div>
    <div class="split__seg" style="flex:${neg};background:#B3382F">${Math.round(100 * neg / tot)}%</div>`;

  const sb = d.script_breakdown || {};
  const label = { deva: 'Devanagari', latin: 'Romanised', mixed: 'Mixed' };
  $('scriptTable').innerHTML = Object.entries(sb).map(([k, v]) =>
    `<tr><th>${label[k] || k}</th><td>${(v.mean ?? 0).toFixed(3)} · n=${v.count ?? 0}</td></tr>`
  ).join('') || '<tr><th>No breakdown</th><td>—</td></tr>';

  const m = d.model_metrics || {};
  $('modelTable').innerHTML = `
    <tr><th>Macro-F1</th><td>${(m.macro_f1 ?? 0).toFixed(3)}</td></tr>
    <tr><th>Accuracy</th><td>${(m.accuracy ?? 0).toFixed(3)}</td></tr>
    <tr><th>Majority baseline</th><td>${(m.baseline ?? 0).toFixed(3)}</td></tr>
    <tr><th>Labelled comments</th><td>${m.n ?? '—'}</td></tr>`;

  renderEvidence(d);
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

/* ── boot ────────────────────────────────────────────────── */

fetch('/api/parties')
  .then(r => r.ok ? r.json() : null)
  .then(d => {
    partyData = d;
    if (!d) return;
    const correct = (d.races || []).filter(x => x.national_correct).length;
    const total = (d.races || []).length;
    if (total) {
      $('corpusMeta').innerHTML =
        `party sentiment ${correct}/${total} seats<br>9,123 comments · 10 candidates`;
    }
  })
  .catch(() => {});
