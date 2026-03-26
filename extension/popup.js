document.getElementById('analyzeBtn').addEventListener('click', () => {
  const status = document.getElementById('status');
  const results = document.getElementById('results');
  const ethicalList = document.getElementById('ethicalList');
  const nonEthicalList = document.getElementById('nonEthicalList');
  const riskBadge = document.getElementById('riskBadge');
  const ethicalCount = document.getElementById('ethicalCount');
  const nonEthicalCount = document.getElementById('nonEthicalCount');
  
  const startTime = Date.now();
  status.innerText = "Analyzing full page (up to 2,00,000+ words)... (Expected: 5-10 seconds)";
  results.style.display = "none";
  ethicalList.innerHTML = "";
  nonEthicalList.innerHTML = "";
  
  // Get the active tab in the current window
  chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
    if (!tabs || tabs.length === 0) {
      status.innerText = "❌ Error: No active tab found";
      return;
    }
    
    const activeTab = tabs[0];
    chrome.tabs.sendMessage(activeTab.id, {action: "getText"}, (response) => {
      if (chrome.runtime.lastError || !response || !response.text) {
        status.innerText = "❌ Error: Refresh page and try again";
        return;
      }
      
      // Show highlighting in progress
      chrome.tabs.sendMessage(activeTab.id, {action: "clearHighlights"});
      
      fetch('http://127.0.0.1:5001/process', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          text: response.text,
          links: response.links || []
        })
      })
      .then(res => res.json())
      .then(data => {
        const endTime = Date.now();
        const actualTime = ((endTime - startTime) / 1000).toFixed(2);
        
        // Apply highlighting to the page
        if (data.highlighted_indices && data.highlighted_indices.length > 0) {
          chrome.tabs.sendMessage(activeTab.id, {
            action: "highlightText",
            highlights: data.highlighted_indices
          });
        }
        
        status.innerText = `✅ Done in ${actualTime}s (Backend: ${data.processing_time_seconds}s) | Analyzed ${data.total_text_analyzed_chars} chars | ${data.clauses_analyzed} clauses`;
        results.style.display = "block";
        
        riskBadge.style.display = "inline-block";
        riskBadge.innerText = data.risk_level + " Risk";
        riskBadge.className = "risk-badge risk-" + data.risk_level.toLowerCase();
        
        // Ethical points
        if (data.ethical_points && data.ethical_points.length > 0) {
          data.ethical_points.forEach(point => {
            const li = document.createElement('li');
            let reasoningHtml = '';
            if (point.reasoning) {
              reasoningHtml = `<br><small style="color:#555; font-size: 11px;">${point.reasoning}</small>`;
            }
            li.innerHTML = `<strong>${point.text}</strong><br><small style="color:#28a745">Confidence: ${(point.confidence * 100).toFixed(0)}%</small>${reasoningHtml}`;
            ethicalList.appendChild(li);
          });
          ethicalCount.innerText = data.ethical_points.length;
        } else {
          ethicalList.innerHTML = "<li>None found</li>";
          ethicalCount.innerText = "0";
        }
        
        // Non-ethical points
        if (data.non_ethical_points && data.non_ethical_points.length > 0) {
          data.non_ethical_points.forEach(point => {
            const li = document.createElement('li');
            let reasoningHtml = '';
            if (point.reasoning) {
              reasoningHtml = `<br><small style="color:#555; font-size: 11px;">${point.reasoning}</small>`;
            }
            li.innerHTML = `<strong>${point.text}</strong><br><small style="color:#dc3545">Confidence: ${(point.confidence * 100).toFixed(0)}%</small>${reasoningHtml}`;
            nonEthicalList.appendChild(li);
          });
          nonEthicalCount.innerText = data.non_ethical_points.length;
        } else {
          nonEthicalList.innerHTML = "<li>None found</li>";
          nonEthicalCount.innerText = "0";
        }
        
        // Display links
        const linksList = document.getElementById('linksList');
        const linksSection = document.getElementById('linksSection');
        const linksCount = document.getElementById('linksCount');
        
        if (data.links && data.links.length > 0) {
          linksSection.style.display = 'block';
          linksList.innerHTML = '';
          data.links.forEach(link => {
            const li = document.createElement('li');
            li.innerHTML = `<a href="${link}" target="_blank" style="color:#667eea; text-decoration:underline; word-break:break-all;">${link}</a>`;
            linksList.appendChild(li);
          });
          linksCount.innerText = data.links.length;
        } else {
          linksSection.style.display = 'none';
        }
      })
      .catch(err => {
        status.innerText = "❌ Error: Backend not running on port 5001";
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