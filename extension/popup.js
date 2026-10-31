// Page text is untrusted: build list items with textContent, never innerHTML.
function addItem(list, parts) {
  const li = document.createElement('li');
  parts.forEach(([tag, text, style], i) => {
    if (i > 0) li.appendChild(document.createElement('br'));
    const el = document.createElement(tag);
    el.textContent = text;
    if (style) el.style.cssText = style;
    li.appendChild(el);
  });
  list.appendChild(li);
  return li;
}

// The decision trace: which clauses were consulted, how they voted, and how close
// the opposite verdict was. Collapsed by default so the list stays readable.
function addWhy(li, trace) {
  if (!trace) return;
  const details = document.createElement('details');
  details.style.cssText = 'margin-top:4px; font-size:11px; color:#555;';
  const summary = document.createElement('summary');
  summary.textContent = 'Why?';
  summary.style.cssText = 'cursor:pointer; color:#667eea; font-weight:600;';
  details.appendChild(summary);

  const line = (text, style) => {
    const div = document.createElement('div');
    div.textContent = text;
    div.style.cssText = style || 'margin-top:3px;';
    details.appendChild(div);
  };
  line(`Decided by ${trace.verdict_by} using ${trace.rule} retrieval`, 'margin-top:4px; color:#888;');
  (trace.evidence || []).forEach(e => line(
    `${e.label === 'non_ethical' ? '✗' : '✓'} ${e.score} — "${e.clause}"${e.category ? ` [${e.category}]` : ''}`,
    'margin-top:3px; padding-left:6px; border-left:2px solid #eee;'));
  if (trace.vote) {
    line(`Vote: unfair ${trace.vote.non_ethical || 0} vs fair ${trace.vote.ethical || 0}`);
  }
  if (trace.closest_opposite) {
    line(`Closest opposite match scored ${trace.closest_opposite.score} (margin ${trace.margin})`);
  }
  if (trace.negation_flip) {
    line('A match was flipped: it says the opposite of this clause.', 'margin-top:3px; color:#9a6508;');
  }
  li.appendChild(details);
}

function renderPoints(points, list, countEl, color) {
  list.innerHTML = '';
  countEl.innerText = points.length;
  if (points.length === 0) {
    addItem(list, [['span', 'None found']]);
    return;
  }
  points.forEach(point => {
    const parts = [
      ['strong', point.text],
      ['small', `Confidence: ${(point.confidence * 100).toFixed(0)}%`, `color:${color}`]
    ];
    if (point.category) {
      parts.push(['small', `Type: ${point.category}`, `color:${color}; font-size:10px; font-weight:600;`]);
    }
    if (point.matched_clause) {
      parts.push(['small', `Similar to: "${point.matched_clause}"`, 'color:#888; font-size:10px; font-style:italic;']);
    }
    addWhy(addItem(list, parts), point.trace);
  });
}

const BACKEND = 'http://127.0.0.1:5001';
const CHOICES = ['rag', 'classifier', 'generator'];

// The dropdown contents, including the one-line reason under each, come from the
// backend so the wording lives in one place.
function buildChoices() {
  fetch(`${BACKEND}/options`)
    .then(res => res.json())
    .then(options => {
      chrome.storage.local.get(CHOICES, saved => {
        CHOICES.forEach(kind => {
          const select = document.getElementById(`${kind}Select`);
          const why = document.getElementById(`${kind}Why`);
          select.innerHTML = '';
          options[kind].forEach(option => {
            const el = document.createElement('option');
            el.value = option.id;
            el.textContent = option.available ? option.name : `${option.name} — not trained yet`;
            el.disabled = !option.available;
            el.dataset.why = option.why;
            select.appendChild(el);
          });
          const stored = options[kind].find(o => o.id === saved[kind] && o.available);
          select.value = (stored || options[kind].find(o => o.default)).id;
          const showWhy = () => { why.textContent = select.selectedOptions[0].dataset.why; };
          showWhy();
          select.addEventListener('change', () => {
            showWhy();
            chrome.storage.local.set({[kind]: select.value});
          });
        });
      });
    })
    .catch(() => {
      document.getElementById('status').innerText =
        "❌ Cannot reach the backend. Start it with: python3 run.py";
    });
}

buildChoices();

document.getElementById('analyzeBtn').addEventListener('click', () => {
  const status = document.getElementById('status');
  const results = document.getElementById('results');
  const riskBadge = document.getElementById('riskBadge');

  const startTime = Date.now();
  status.innerText = "Analyzing page...";
  const ticker = setInterval(() => {
    const seconds = Math.round((Date.now() - startTime) / 1000);
    status.innerText = `Analyzing page... ${seconds}s`
      + (seconds > 8 ? " (the Qwen summary model is slow; 'Plain-language templates' is instant)" : "");
  }, 1000);
  results.style.display = "none";

  // Get the active tab in the current window
  chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
    if (!tabs || tabs.length === 0) {
      clearInterval(ticker);
      status.innerText = "❌ No active tab found";
      return;
    }

    const activeTab = tabs[0];
    chrome.tabs.sendMessage(activeTab.id, {action: "getText"}, (response) => {
      if (chrome.runtime.lastError || !response || !response.text) {
        clearInterval(ticker);
        status.innerText = "❌ Cannot read this page. Reload the page, then try again.";
        return;
      }

      chrome.tabs.sendMessage(activeTab.id, {action: "clearHighlights"});

      // some pipeline options take over a minute on their first request
      let reached = false;   // distinguishes "never got a response" from "crashed rendering one"
      const timeout = new AbortController();
      const timer = setTimeout(() => timeout.abort(), 180000);

      fetch(`${BACKEND}/process`, {
        signal: timeout.signal,
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          text: response.text,
          links: response.links || [],
          rag: document.getElementById('ragSelect').value,
          classifier: document.getElementById('classifierSelect').value,
          generator: document.getElementById('generatorSelect').value
        })
      })
      .then(res => {
        reached = true;
        if (!res.ok) throw new Error(`The backend returned an error (HTTP ${res.status}). Check its terminal.`);
        return res.json();
      })
      .then(data => {
        clearTimeout(timer); clearInterval(ticker);
        const actualTime = ((Date.now() - startTime) / 1000).toFixed(2);

        // Apply highlighting to the page
        if (data.highlighted_indices && data.highlighted_indices.length > 0) {
          chrome.tabs.sendMessage(activeTab.id, {
            action: "highlightText",
            highlights: data.highlighted_indices
          });
        }

        const unclear = (data.unclear_points || []).length;
        status.innerText = `✅ Done in ${actualTime}s (Backend: ${data.processing_time_seconds}s) | `
          + `${data.clauses_analyzed} clauses${unclear ? `, ${unclear} unclear` : ''} | ${data.total_text_analyzed_chars} chars`;
        results.style.display = "block";

        riskBadge.style.display = "inline-block";
        riskBadge.innerText = data.risk_level + " Risk";
        riskBadge.className = "risk-badge risk-" + data.risk_level.toLowerCase();

        const summaryList = document.getElementById('summaryList');
        summaryList.innerHTML = '';
        (data.summary || []).forEach(sentence => addItem(summaryList, [['span', sentence]]));

        renderPoints(data.ethical_points || [], document.getElementById('ethicalList'), document.getElementById('ethicalCount'), '#28a745');
        renderPoints(data.non_ethical_points || [], document.getElementById('nonEthicalList'), document.getElementById('nonEthicalCount'), '#dc3545');

        // Display links
        const links = (data.links || []).filter(url => /^https?:\/\//.test(url));
        const linksList = document.getElementById('linksList');
        document.getElementById('linksSection').style.display = links.length ? 'block' : 'none';
        document.getElementById('linksCount').innerText = links.length;
        linksList.innerHTML = '';
        links.forEach(url => {
          const li = document.createElement('li');
          const a = document.createElement('a');
          a.href = url;
          a.target = '_blank';
          a.textContent = url;
          li.appendChild(a);
          linksList.appendChild(li);
        });
      })
      .catch(err => {
        clearTimeout(timer); clearInterval(ticker);
        results.style.display = "none";   // never leave an older result under an error
        if (err.name === 'AbortError') {
          status.innerText = "❌ Timed out after 3 minutes. Try the 'Plain-language templates' summary, "
            + "or a verdict other than Legal-BERT, which is slow on its first run.";
        } else if (!reached) {
          status.innerText = "❌ Cannot reach the backend. Start it with: python3 run.py";
        } else {
          status.innerText = `❌ Analysis arrived but could not be displayed: ${err.message}`;
        }
        console.error(err);
      });
    });
  });
});

// Add clear highlights button functionality
if (document.getElementById('clearHighlightsBtn')) {
  document.getElementById('clearHighlightsBtn').addEventListener('click', () => {
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
      if (tabs && tabs.length > 0) {
        chrome.tabs.sendMessage(tabs[0].id, {action: "clearHighlights"});
      }
    });
  });
}
