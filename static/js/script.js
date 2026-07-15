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

    // PIE CHART - Handle case sensitivity
    const ctxPie = document.getElementById('pieChart').getContext('2d');
    if (pieChartInstance) pieChartInstance.destroy();
    const pos = data.counts.Positive || data.counts.positive || 0;
    const neg = data.counts.Negative || data.counts.negative || 0;
    const neu = data.counts.Neutral || data.counts.neutral || 0;
    pieChartInstance = new Chart(ctxPie, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Negative', 'Neutral'],
            datasets: [{
                data: [pos, neg, neu],
                backgroundColor: ['#2ecc71', '#e74c3c', '#95a5a6']
            }]
        }
    });

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