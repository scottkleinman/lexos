/**
 * Citation View
 *
 * Displays a single bibliography entry with formatted citation
 */

import { Cite, CITATION_STYLES } from './lib/citation.js';

/**
 * Helper function to ensure paths are absolute
 */
function ensureAbsolutePath(path) {
  if (!path) return path;
  // If path already starts with /, it's absolute
  if (path.startsWith('/')) return path;
  // If path starts with http:// or https://, it's already absolute
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  // Otherwise, make it absolute by prepending /
  return '/' + path;
}

/**
 * Load configuration
 */
async function loadConfig() {
  try {
    const cacheBuster = Date.now();
    const response = await fetch(`/config.json?v=${cacheBuster}`, {
      cache: 'no-cache'
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.warn('[Citation] Failed to load config.json, using defaults:', error);
    return {
      bibliography: { path: 'sample_data/bibliography.json' }
    };
  }
}

/**
 * Render the citation view for a specific document
 */
export async function renderCitationView(docId) {
  console.log('[Citation] Rendering citation view for document:', docId);

  const container = document.getElementById('main-view');

  if (!container) {
    console.error('[Citation] Main container not found');
    return;
  }

  // Show loading state
  container.innerHTML = `
    <div class="container mt-4">
      <div class="text-center">
        <div class="spinner-border text-primary" role="status">
          <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-2">Loading citation...</p>
      </div>
    </div>
  `;

  try {
    // Try to load from bibliography.json first (has proper CSL JSON format)
    let cslData = null;
    let doc = null;

    try {
      // Load bibliography data
      const config = await loadConfig();
      const bibliographyPath = config?.bibliography?.path || 'sample_data/bibliography.json';
      const absolutePath = ensureAbsolutePath(bibliographyPath);
      console.log('[Citation] Loading bibliography from:', absolutePath);
      const response = await fetch(absolutePath);

      if (response.ok) {
        const bibliographyData = await response.json();

        // Add _docIndex to each entry if not present
        bibliographyData.forEach((entry, index) => {
          if (entry._docIndex === undefined) {
            entry._docIndex = index;
          }
        });

        // Find the entry for this document
        const bibEntry = bibliographyData.find(entry => entry._docIndex === docId);

        if (bibEntry) {
          console.log('[Citation] Using bibliography.json data');
          cslData = bibEntry;
          doc = bibEntry;
        }
      }
    } catch (bibError) {
      console.log('[Citation] Bibliography not available, falling back to metadata:', bibError.message);
    }

    // Fallback to metadata if bibliography not available
    if (!cslData) {
      doc = window.dfrState.metadata[docId];
      if (!doc) {
        throw new Error(`Document ${docId} not found`);
      }
      console.log('[Citation] Using metadata for CSL conversion');
      cslData = convertToCslJson(doc, docId);
    }

    // Generate citations using our custom Cite class
    const cite = new Cite(cslData);

    // Generate citation formats for all available styles
    const citationFormats = {};
    for (const style of CITATION_STYLES) {
      try {
        citationFormats[style.value] = {
          html: await cite.format('bibliography', { format: 'html', template: style.value }),
          text: await cite.format('bibliography', { format: 'text', template: style.value })
        };
      } catch (error) {
        console.warn(`[Citation] Could not generate format for ${style.label}:`, error.message);
        // Use a fallback for this style
        citationFormats[style.value] = {
          html: `<em>Style "${style.label}" not available</em>`,
          text: `Style "${style.label}" not available`
        };
      }
    }

    const bibtex = await cite.format('bibtex');
    const ris = await cite.format('ris');

    // Store citation formats for event listeners
    window.currentCitationFormats = citationFormats;

    // Render the view
    container.innerHTML = generateCitationHTML(doc, docId, {
      citationFormats: citationFormats,
      bibtex: bibtex,
      ris: ris,
      csl: cslData
    });

    // Setup event listeners
    setupCitationEventListeners();

  } catch (error) {
    console.error('[Citation] Error rendering citation:', error);
    container.innerHTML = `
      <div class="container mt-4">
        <div class="alert alert-danger">
          <h4 class="alert-heading">Error Loading Citation</h4>
          <p>${error.message}</p>
          <hr>
          <p class="mb-0">
            <a href="#/bibliography" class="alert-link">Return to Bibliography</a>
          </p>
        </div>
      </div>
    `;
  }
}

/**
 * Convert document metadata to CSL JSON format
 */
function convertToCslJson(doc, docId) {
  const csl = {
    id: `doc-${docId}`,
    type: 'article-journal' // Default type
  };

  // Map common fields
  if (doc.title) csl.title = doc.title;
  if (doc.journaltitle) csl['container-title'] = doc.journaltitle;
  if (doc.volume) csl.volume = doc.volume;
  if (doc.issue) csl.issue = doc.issue;
  if (doc.pagerange) csl.page = doc.pagerange;
  if (doc.doi) {
    const doi = Array.isArray(doc.doi) ? doc.doi[0] : doc.doi;
    csl.DOI = doi;
  }
  if (doc.url) csl.URL = doc.url;

  // Parse authors
  if (doc.author) {
    csl.author = parseAuthors(doc.author);
  }

  // Parse date
  if (doc.pubdate) {
    const date = parseDate(doc.pubdate);
    if (date) csl.issued = date;
  } else if (doc.year) {
    csl.issued = { 'date-parts': [[parseInt(doc.year)]] };
  }

  return csl;
}

/**
 * Parse author string into CSL JSON format
 */
function parseAuthors(authorString) {
  if (!authorString) return [];

  const authors = authorString.split(/;\s*/);
  return authors.map(author => {
    const parts = author.split(',').map(s => s.trim());
    if (parts.length >= 2) {
      return {
        family: parts[0],
        given: parts[1]
      };
    } else {
      return { literal: author };
    }
  });
}

/**
 * Parse date string into CSL JSON format
 */
function parseDate(dateString) {
  if (!dateString) return null;

  // Try to parse YYYY-MM-DD format
  const match = dateString.match(/(\d{4})-?(\d{2})?-?(\d{2})?/);
  if (match) {
    const parts = [parseInt(match[1])];
    if (match[2]) parts.push(parseInt(match[2]));
    if (match[3]) parts.push(parseInt(match[3]));
    return { 'date-parts': [parts] };
  }

  // Try to parse just year
  const yearMatch = dateString.match(/(\d{4})/);
  if (yearMatch) {
    return { 'date-parts': [[parseInt(yearMatch[1])]] };
  }

  return null;
}

/**
 * Generate HTML for citation view
 */
function generateCitationHTML(doc, docId, citations) {
  // Generate dropdown options for all citation styles
  const styleOptions = CITATION_STYLES.map((style, index) =>
    `<option value="${style.value}" ${index === 0 ? 'selected' : ''}>${style.label}</option>`
  ).join('\n                ');

  // Get the default style (first in list)
  const defaultStyle = CITATION_STYLES[0].value;

  return `
    <div class="container mt-4">
      <!-- Header -->
      <div class="mb-4">
        <h2>Citation</h2>
      </div>

      <!-- Citation Display with Style Selector -->
      <div class="card mb-4">
        <div class="card-header">
          <div class="d-flex justify-content-between align-items-center">
            <h5 class="mb-0">Formatted Citation</h5>
            <div class="d-flex align-items-center">
              <label for="citation-style-select" class="me-2 mb-0"><small>Style:</small></label>
              <select id="citation-style-select" class="form-select form-select-sm" style="width: auto;">
                ${styleOptions}
              </select>
            </div>
          </div>
        </div>
        <div class="card-body">
          <div id="citation-display" class="citation-display mb-3" style="font-size: 1.1rem; line-height: 1.6;">
            ${citations.citationFormats[defaultStyle]?.html || 'Citation not available'}
          </div>
          <button class="btn btn-sm btn-primary" id="copy-citation-btn">
            <i class="bi bi-clipboard"></i> Copy to Clipboard
          </button>
        </div>
      </div>

      <!-- Export Options -->
      <div class="card mb-4">
        <div class="card-header">
          <h5 class="mb-0">Export Citation</h5>
        </div>
        <div class="card-body">
          <!-- Format Tabs -->
          <ul class="nav nav-tabs mb-3" id="citationTabs" role="tablist">
            <li class="nav-item" role="presentation">
              <button class="nav-link active" id="bibtex-tab" data-bs-toggle="tab"
                      data-bs-target="#bibtex-pane" type="button" role="tab">
                BibTeX
              </button>
            </li>
            <li class="nav-item" role="presentation">
              <button class="nav-link" id="ris-tab" data-bs-toggle="tab"
                      data-bs-target="#ris-pane" type="button" role="tab">
                RIS
              </button>
            </li>
            <li class="nav-item" role="presentation">
              <button class="nav-link" id="csl-tab" data-bs-toggle="tab"
                      data-bs-target="#csl-pane" type="button" role="tab">
                CSL JSON
              </button>
            </li>
          </ul>

          <!-- Tab Content -->
          <div class="tab-content" id="citationTabContent">
            <!-- BibTeX -->
            <div class="tab-pane fade show active" id="bibtex-pane" role="tabpanel">
              <pre class="border p-3 bg-light" style="white-space: pre-wrap;">${escapeHtml(citations.bibtex)}</pre>
              <button class="btn btn-sm btn-primary copy-btn" data-target="bibtex-pane">
                <i class="bi bi-clipboard"></i> Copy to Clipboard
              </button>
            </div>

            <!-- RIS -->
            <div class="tab-pane fade" id="ris-pane" role="tabpanel">
              <pre class="border p-3 bg-light" style="white-space: pre-wrap;">${escapeHtml(citations.ris)}</pre>
              <button class="btn btn-sm btn-primary copy-btn" data-target="ris-pane">
                <i class="bi bi-clipboard"></i> Copy to Clipboard
              </button>
            </div>

            <!-- CSL JSON -->
            <div class="tab-pane fade" id="csl-pane" role="tabpanel">
              <pre class="border p-3 bg-light" style="white-space: pre-wrap;">${escapeHtml(JSON.stringify(citations.csl, null, 2))}</pre>
              <button class="btn btn-sm btn-primary copy-btn" data-target="csl-pane">
                <i class="bi bi-clipboard"></i> Copy to Clipboard
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Document Metadata -->
      <div class="card">
        <div class="card-header">
          <h5 class="mb-0">Document Metadata</h5>
        </div>
        <div class="card-body">
          <dl class="row">
            ${Object.entries(doc).map(([key, value]) => `
              <dt class="col-sm-3">${escapeHtml(key)}</dt>
              <dd class="col-sm-9">${escapeHtml(String(value))}</dd>
            `).join('')}
          </dl>
        </div>
      </div>
    </div>
  `;
}

/**
 * Setup event listeners for citation view
 */
function setupCitationEventListeners() {
  // Get citation formats from the stored data
  const citationFormats = window.currentCitationFormats;

  // Style selector for formatted citation
  const styleSelect = document.getElementById('citation-style-select');
  const citationDisplay = document.getElementById('citation-display');

  if (styleSelect && citationDisplay) {
    styleSelect.addEventListener('change', function() {
      const selectedStyle = this.value;
      citationDisplay.innerHTML = citationFormats[selectedStyle].html;
    });
  }

  // Copy formatted citation button
  const copyCitationBtn = document.getElementById('copy-citation-btn');
  if (copyCitationBtn) {
    copyCitationBtn.addEventListener('click', function() {
      const selectedStyle = styleSelect.value;
      const text = citationFormats[selectedStyle].text;

      navigator.clipboard.writeText(text).then(() => {
        // Show success feedback
        const originalText = this.innerHTML;
        this.innerHTML = '<i class="bi bi-check"></i> Copied!';
        this.classList.remove('btn-primary');
        this.classList.add('btn-success');

        setTimeout(() => {
          this.innerHTML = originalText;
          this.classList.remove('btn-success');
          this.classList.add('btn-primary');
        }, 2000);
      }).catch(err => {
        console.error('Failed to copy:', err);
        alert('Failed to copy to clipboard');
      });
    });
  }

  // Copy to clipboard functionality for export formats
  const copyButtons = document.querySelectorAll('.copy-btn');

  copyButtons.forEach(button => {
    button.addEventListener('click', function() {
      const targetId = this.getAttribute('data-target');
      const pane = document.getElementById(targetId);
      const pre = pane.querySelector('pre');
      const text = pre.textContent;

      navigator.clipboard.writeText(text).then(() => {
        // Show success feedback
        const originalText = this.innerHTML;
        this.innerHTML = '<i class="bi bi-check"></i> Copied!';
        this.classList.remove('btn-primary');
        this.classList.add('btn-success');

        setTimeout(() => {
          this.innerHTML = originalText;
          this.classList.remove('btn-success');
          this.classList.add('btn-primary');
        }, 2000);
      }).catch(err => {
        console.error('Failed to copy:', err);
        alert('Failed to copy to clipboard');
      });
    });
  });
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}
