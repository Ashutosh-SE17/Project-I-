/* ============================================================
   extras.js — stats band, party standings, comment evidence

   Load AFTER app.js. Reads the PARTY / SYMBOL / RACES globals
   and the partyData already fetched there.
   ============================================================ */

/* ── stats band ──────────────────────────────────────────── */

function renderBand(d) {
  const races = d?.races || [];
  const correct = races.filter(r => r.national_correct).length;
  const scores = d?.national_party_scores || {};
  const totalN = Object.values(scores).reduce((a, s) => a + (s.n || 0), 0);

  document.getElementById('band').innerHTML = `
    <div class="band__cell">
      <div class="band__val">${correct}<small>/${races.length || 5}</small></div>
      <div class="band__key">Seats predicted</div>
    </div>
    <div class="band__cell">
      <div class="band__val">0.814</div>
      <div class="band__key">Model macro-F1</div>
    </div>
    <div class="band__cell">
      <div class="band__val">0.943</div>
      <div class="band__key">Annotator agreement</div>
    </div>
    <div class="band__cell">
      <div class="band__val">${totalN.toLocaleString()}</div>
      <div class="band__key">Comments scored</div>
    </div>
    <div class="band__cell">
      <div class="band__val">10</div>
      <div class="band__key">Candidates tracked</div>
    </div>`;
}

/* ── party standings ─────────────────────────────────────── */

function renderStandings(d) {
  const scores = d?.national_party_scores || {};
  const races = d?.races || [];

  const seatsWon = {};
  races.forEach(r => {
    const w = r.actual_party || r.national_predicted;
    if (w) seatsWon[w] = (seatsWon[w] || 0) + 1;
  });

  const rows = Object.entries(scores)
    .map(([k, s]) => ({ k, mean: s.mean_polarity ?? s.mean ?? 0, n: s.n ?? 0 }))
    .sort((a, b) => b.mean - a.mean);

  document.getElementById('standings').innerHTML = rows.map(r => {
    const p = PARTY[r.k] || { label: r.k, colour: '#9A96A6', symbol: 'ballot' };
    const w = Math.min(50, Math.abs(r.mean) * 50);
    const side = r.mean >= 0 ? `left:50%;width:${w}%` : `right:50%;width:${w}%`;
    const seats = seatsWon[r.k] || 0;
    return `
      <div class="standing">
        <span class="chip" style="color:${p.colour}" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"
               stroke-linecap="round" stroke-linejoin="round">${SYMBOL[p.symbol]}</svg>
        </span>
        <div class="standing__id">
          <div class="standing__name">${p.label}</div>
          <div class="standing__seats">${seats} of ${races.length || 5} seats</div>
          <div class="standing__track">
            <div class="standing__mid"></div>
            <div class="standing__fill" style="${side};background:${p.colour}"></div>
          </div>
        </div>
        <div class="standing__val">${r.mean >= 0 ? '+' : ''}${r.mean.toFixed(3)}
          <span class="standing__n">n=${r.n.toLocaleString()}</span>
        </div>
      </div>`;
  }).join('');
}

/* ── comment evidence ────────────────────────────────────── */

const CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M20 6 9 17l-5-5"/></svg>';
const CROSS = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M18 6 6 18M6 6l12 12"/></svg>';

let evidence = { positive: [], negative: [], tab: 'positive' };

function renderEvidence(d) {
  evidence.positive = d.sample_positive || [];
  evidence.negative = d.sample_negative || [];

  // fall back to the legacy combined key if the backend wasn't updated
  if (!evidence.positive.length && !evidence.negative.length) {
    const all = d.sample_data || [];
    evidence.positive = all.filter(r => r.sentiment_label === 'Positive');
    evidence.negative = all.filter(r => r.sentiment_label === 'Negative');
  }

  const v = d.verified_count;
  const strip = document.getElementById('verifiedStrip');
  if (v && v.labelled) {
    strip.hidden = false;
    strip.innerHTML = `
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor"
           stroke-width="2" style="flex:none;color:var(--violet)">
        <path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/>
      </svg>
      <span><b>${v.agreeing} of ${v.labelled}</b> comments shown here were also
        hand-labelled. Badges mark where the model matched the human label.</span>`;
  } else {
    strip.hidden = true;
  }

  document.getElementById('tabPos').textContent = `Positive · ${evidence.positive.length}`;
  document.getElementById('tabNeg').textContent = `Negative · ${evidence.negative.length}`;
  paintTab();
}

function paintTab() {
  const rows = evidence[evidence.tab];
  const list = document.getElementById('quoteList');

  list.innerHTML = rows.length
    ? rows.map(r => `
        <div class="quote" data-s="${r.sentiment_label}">
          ${escapeHtml(String(r.text).slice(0, 220))}${badge(r)}
        </div>`).join('')
    : '<p style="font-size:12.5px;color:var(--muted);margin:0">No comments in this class.</p>';

  list.dataset.open = 'false';
  const btn = document.getElementById('moreBtn');
  btn.setAttribute('aria-expanded', 'false');
  btn.hidden = rows.length <= 3;
  btn.querySelector('span').textContent = `Show all ${rows.length}`;

  document.getElementById('tabPos').setAttribute('aria-selected', String(evidence.tab === 'positive'));
  document.getElementById('tabNeg').setAttribute('aria-selected', String(evidence.tab === 'negative'));
}

function badge(r) {
  if (r.human_label == null) return '';
  const ok = r.agrees;
  return `<span class="quote__tag quote__tag--${ok ? 'ok' : 'dif'}"
            title="Human label: ${r.human_label}">${ok ? CHECK : CROSS}${ok ? 'verified' : 'differs'}</span>`;
}

/* ── wiring ──────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('tabPos').addEventListener('click', () => { evidence.tab = 'positive'; paintTab(); });
  document.getElementById('tabNeg').addEventListener('click', () => { evidence.tab = 'negative'; paintTab(); });

  document.getElementById('moreBtn').addEventListener('click', e => {
    const btn = e.currentTarget;
    const list = document.getElementById('quoteList');
    const open = btn.getAttribute('aria-expanded') === 'true';
    btn.setAttribute('aria-expanded', String(!open));
    list.dataset.open = String(!open);
    btn.querySelector('span').textContent =
      open ? `Show all ${evidence[evidence.tab].length}` : 'Show fewer';
  });

  fetch('/api/parties')
    .then(r => r.ok ? r.json() : null)
    .then(d => { if (d) { renderBand(d); renderStandings(d); } })
    .catch(() => {});
});
