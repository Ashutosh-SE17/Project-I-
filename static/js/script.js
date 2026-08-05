let pieChartInstance = null;

function getMapFileForCandidate(candidate, winProbability = null) {
    const normalized = String(candidate).toLowerCase();

    if (normalized.includes('balen')) return 'nepal_map_balen.html';
    if (normalized.includes('gagan')) return 'nepal_map_gagan.html';
    if (normalized.includes('kp') || normalized.includes('prachanda')) return 'nepal_map_kp.html';

    return 'nepal_map_default.html';
}

function updateMapForCandidate(candidate, winProbability = null) {
    const iframe = document.querySelector('.map-frame');
    if (!iframe) return;
    iframe.src = `/static/${getMapFileForCandidate(candidate, winProbability)}`;
}

async function runAnalysis() {
    const candidate = document.getElementById('candidateName').value;
    const source = document.getElementById('dataSource').value;
    const dashboard = document.getElementById('dashboard');
    const loading = document.getElementById('loading');

    if (!candidate) {
        alert("Please select a candidate.");
        return;
    }

    dashboard.classList.add('hidden');
    loading.classList.remove('hidden');

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ candidate, source })
        });

        const data = await response.json();
        if (data.error) {
            alert(data.error);
            return;
        }

        updateMapForCandidate(candidate, data.win_probability);
        renderDashboard(data);

    } catch (error) {
        console.error('Error:', error);
        alert("Failed to fetch analysis.");
    } finally {
        loading.classList.add('hidden');
        dashboard.classList.remove('hidden');
    }
}

function renderDashboard(data) {
    // Safety check to prevent the "Failed to fetch" alert
    if (!data.counts || !data.sample_data) {
        console.error("Missing data keys:", data);
        alert("Data received but format is incorrect. Check Console.");
        return;
    }

    const meterFill = document.getElementById('meterFill');
    meterFill.style.width = `${data.win_probability}%`;
    meterFill.innerText = `${data.win_probability}%`;

    // CONFIDENCE INTERVAL
    const ciText = document.getElementById('ciText');
    if (ciText) {
        if (Array.isArray(data.confidence_interval) && data.confidence_interval.length === 2) {
            const [lo, hi] = data.confidence_interval;
            ciText.innerText = `${data.win_probability}% (95% CI: ${lo}–${hi})`;
        } else {
            ciText.innerText = '';
        }
    }

    // PIE CHART - binary model: Positive / Negative only
    const ctxPie = document.getElementById('pieChart').getContext('2d');
    if (pieChartInstance) pieChartInstance.destroy();
    const pos = data.counts.Positive || data.counts.positive || 0;
    const neg = data.counts.Negative || data.counts.negative || 0;
    pieChartInstance = new Chart(ctxPie, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Negative'],
            datasets: [{
                data: [pos, neg],
                backgroundColor: ['#2ecc71', '#e74c3c']
            }]
        }
    });

    // MODEL PERFORMANCE
    renderModelMetrics(data.model_metrics);

    // SCRIPT BREAKDOWN
    renderScriptBreakdown(data.script_breakdown);

    // TABLE - Only render if sample_data exists
    const tableBody = document.querySelector('#dataTable tbody');
    tableBody.innerHTML = '';
    data.sample_data.forEach(row => {
        const sentimentColor = row.sentiment_label === 'Positive' ? '#27ae60' :
                              row.sentiment_label === 'Negative' ? '#c0392b' : '#7f8c8d';
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.text ? row.text.substring(0, 100) : 'No text'}...</td>
            <td style="color:${sentimentColor}; font-weight:bold;">${row.sentiment_label || 'Neutral'}</td>
        `;
        tableBody.appendChild(tr);
    });
}

function renderModelMetrics(metrics) {
    const container = document.getElementById('modelMetrics');
    if (!container) return;
    container.innerHTML = '';

    if (!metrics || metrics.macro_f1 === undefined) {
        container.innerHTML = '<p>No model metrics available.</p>';
        return;
    }

    const rows = [
        ['Macro-F1', metrics.macro_f1.toFixed(3)],
        ['Accuracy', `${(metrics.accuracy * 100).toFixed(1)}%`],
        ['Majority baseline', `${(metrics.baseline * 100).toFixed(1)}%`],
    ];
    rows.forEach(([label, value]) => {
        const row = document.createElement('div');
        row.className = 'stat-row';
        row.innerHTML = `<span>${label}</span><span>${value}</span>`;
        container.appendChild(row);
    });
}

function renderScriptBreakdown(breakdown) {
    const tableBody = document.querySelector('#scriptTable tbody');
    if (!tableBody) return;
    tableBody.innerHTML = '';

    if (!breakdown) return;

    const scriptLabels = { deva: 'Devanagari', latin: 'Latin', mixed: 'Mixed' };
    ['deva', 'latin', 'mixed'].forEach(script => {
        if (!breakdown[script]) return;
        const { count, mean } = breakdown[script];
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${scriptLabels[script]}</td>
            <td>${count}</td>
            <td>${Number(mean).toFixed(3)}</td>
        `;
        tableBody.appendChild(tr);
    });
}

async function loadPartyData() {
    const tableBody = document.querySelector('#partyTable tbody');
    if (!tableBody) return;

    try {
        const response = await fetch('/api/parties');
        const data = await response.json();
        if (data.error || !data.national_party_scores) {
            tableBody.innerHTML = `<tr><td colspan="3">${data.error || 'No party data available.'}</td></tr>`;
            return;
        }

        tableBody.innerHTML = '';
        Object.entries(data.national_party_scores).forEach(([party, s]) => {
            if (s.n === 0) return;
            const cls = s.mean_polarity >= 0 ? 'positive' : 'negative';
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${party}</td>
                <td>${s.n}</td>
                <td class="${cls}">${s.mean_polarity.toFixed(3)}</td>
            `;
            tableBody.appendChild(tr);
        });
    } catch (error) {
        console.error('Error loading party data:', error);
        tableBody.innerHTML = '<tr><td colspan="3">Failed to load party data.</td></tr>';
    }
}

document.addEventListener('DOMContentLoaded', loadPartyData);