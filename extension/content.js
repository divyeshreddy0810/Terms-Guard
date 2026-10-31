chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getText") {
    // Prefer the document's main content: document.body.innerText starts with the
    // nav bar and menus, which is what the backend would otherwise analyse first.
    const main = document.querySelector('main, article, [role="main"], #content, .main-content');
    const text = (main || document.body).innerText;
    
    // Extract all links
    const links = [];
    const linkElements = document.querySelectorAll('a[href]');
    linkElements.forEach(link => {
      const href = link.href;
      const linkText = link.textContent.trim();
      if (href && href.startsWith('http')) {
        links.push({
          url: href,
          text: linkText || href
        });
      }
    });
    
    sendResponse({
      text: text.substring(0, 10000000),
      links: links
    });
  } else if (request.action === "highlightText") {
    highlightAnalyzedText(request.highlights);
  } else if (request.action === "clearHighlights") {
    clearAllHighlights();
  }
  return true;
});

// Function to highlight analyzed text on the page using simpler approach
function highlightAnalyzedText(highlights) {
  const colors = {
    ethical: { bg: '#d4edda', border: '#28a745' },
    non_ethical: { bg: '#f8d7da', border: '#dc3545' }
  };
  
  // Create a replacer function to highlight text directly
  highlights.forEach(highlight => {
    const { text, type } = highlight;
    if (!text) return;
    
    const color = colors[type];
    const safeText = text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    // No 'g' flag: a global regex's test() keeps lastIndex between nodes and skips matches
    const matcher = new RegExp(safeText, 'i');
    const splitter = new RegExp(`(${safeText})`, 'gi');

    // Highlight in body by traversing all text nodes
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
      // Skip non-rendered text (inline JSON in <script> often repeats the page text)
      { acceptNode: n => /^(SCRIPT|STYLE|NOSCRIPT|TEXTAREA)$/.test(n.parentNode.nodeName) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT },
      false
    );

    let node;
    const nodesToReplace = [];

    while (node = walker.nextNode()) {
      if (matcher.test(node.textContent)) {
        nodesToReplace.push(node);
      }
    }

    nodesToReplace.forEach(node => {
      const span = document.createElement('span');

      // split() with a capture group puts the matches at odd indices
      node.textContent.split(splitter).forEach((part, i) => {
        if (part && i % 2 === 1) {
          const highlightSpan = document.createElement('span');
          highlightSpan.className = 'analyzed-text-highlight';
          highlightSpan.setAttribute('data-type', type);
          highlightSpan.style.backgroundColor = color.bg;
          highlightSpan.style.borderLeft = `3px solid ${color.border}`;
          highlightSpan.style.borderRadius = '2px';
          highlightSpan.style.padding = '2px 4px';
          highlightSpan.style.cursor = 'pointer';
          highlightSpan.title = `${type.charAt(0).toUpperCase() + type.slice(1)} clause`;
          highlightSpan.textContent = part;
          span.appendChild(highlightSpan);
        } else if (part) {
          span.appendChild(document.createTextNode(part));
        }
      });
      
      if (node.parentNode) {
        node.parentNode.replaceChild(span, node);
      }
    });
  });
  
  // Add stylesheet
  if (!document.getElementById('t-c-analyzer-style')) {
    const style = document.createElement('style');
    style.id = 't-c-analyzer-style';
    style.textContent = `
      .analyzed-text-highlight {
        transition: background-color 0.2s ease;
      }
      .analyzed-text-highlight:hover {
        opacity: 0.85 !important;
        box-shadow: 0 0 3px currentColor;
      }
    `;
    document.head.appendChild(style);
  }
}

// Function to clear all highlights
function clearAllHighlights() {
  const highlights = document.querySelectorAll('.analyzed-text-highlight');
  highlights.forEach(highlight => {
    const parent = highlight.parentNode;
    while (highlight.firstChild) {
      parent.insertBefore(highlight.firstChild, highlight);
    }
    parent.removeChild(highlight);
  });
}