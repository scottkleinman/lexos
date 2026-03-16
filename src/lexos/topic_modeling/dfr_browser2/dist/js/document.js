// Document View

import { extractTopicWords } from './state-utils.js';
import { getTopicLabel, getMetadataValue } from './topic-config.js';

// Helper function to ensure paths work on any sub-path deployment
function ensureAbsolutePath(path) {
  if (!path) return path;
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  if (path.startsWith('/')) return (window.dfrBasePath || '') + path;
  return path;
}

// Config management
let appConfig = null;
let dataSourceCache = null;

async function loadConfig() {
  try {
    const cacheBuster = Date.now();
    const response = await fetch(`/config.json?v=${cacheBuster}`, {
      cache: 'no-cache'
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    appConfig = await response.json();
    return appConfig;
  } catch (error) {
    console.warn('Failed to load config.json, using defaults:', error);
    appConfig = {
      embargo: false,
      language: { default: 'en' }
    };
    return appConfig;
  }
}

// Load and cache data source file
async function loadDataSource() {
  if (dataSourceCache) {
    return dataSourceCache;
  }

  try {
    console.log('Checking metadata.csv for data_source column...');
    const metadataResponse = await fetch(ensureAbsolutePath('sample_data/metadata.csv'));
    if (!metadataResponse.ok) {
      throw new Error(`Failed to load metadata.csv: HTTP ${metadataResponse.status}`);
    }

    const metadataText = await metadataResponse.text();
    const metadataLines = metadataText.trim().split('\n');

    if (metadataLines.length === 0) {
      throw new Error('Metadata file is empty');
    }

    const headers = metadataLines[0].split(',').map(h => h.trim());
    const dataSourceColIndex = headers.findIndex(h => h.toLowerCase() === 'data_source');

    if (dataSourceColIndex !== -1) {
      console.log('Found data_source column in metadata, loading individual files...');

      dataSourceCache = [];

      for (let i = 1; i < metadataLines.length; i++) {
        const columns = metadataLines[i].split(',');
        const dataSourcePath = columns[dataSourceColIndex]?.trim();

        if (dataSourcePath) {
          try {
            const docResponse = await fetch(ensureAbsolutePath(dataSourcePath));
            if (docResponse.ok) {
              const docText = await docResponse.text();
              dataSourceCache[i - 1] = {
                id: i - 1,
                docId: columns[0] || `doc${i}`,
                text: docText.trim()
              };
            } else {
              dataSourceCache[i - 1] = {
                id: i - 1,
                docId: columns[0] || `doc${i}`,
                text: ''
              };
            }
          } catch (docError) {
            dataSourceCache[i - 1] = {
              id: i - 1,
              docId: columns[0] || `doc${i}`,
              text: ''
            };
          }
        } else {
          dataSourceCache[i - 1] = {
            id: i - 1,
            docId: columns[0] || `doc${i}`,
            text: ''
          };
        }
      }

      console.log(`Loaded ${dataSourceCache.filter(doc => doc.text).length} documents with text`);

    } else if (appConfig?.data_source) {
      console.log('No data_source column in metadata, using config data_source file...');

      const response = await fetch(ensureAbsolutePath(appConfig.data_source));
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const text = await response.text();
      const lines = text.trim().split('\n');

      dataSourceCache = lines.map((line, index) => {
        const columns = line.split('\t');
        return {
          id: index,
          docId: columns[0] || `doc${index + 1}`,
          text: columns.length === 2 ? columns[1] : (columns.length >= 3 ? columns[2] : '')
        };
      });

      console.log(`Loaded ${dataSourceCache.length} documents from single data source file`);
    } else {
      throw new Error('No data source configured');
    }

    return dataSourceCache;
  } catch (error) {
    console.error('Failed to load data source:', error);
    return null;
  }
}

// Display document text in modal
function showDocumentText(documentId, documentText, docRef) {
  const existingModal = document.getElementById('documentTextModal');
  if (existingModal) {
    existingModal.remove();
  }

  const modalHtml = `
    <div class="modal fade" id="documentTextModal" tabindex="-1" aria-labelledby="documentTextModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-lg">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="documentTextModalLabel">Document Text: ${documentId}</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <div class="document-info mb-3">
              <small class="text-muted">
                ${docRef ? `Reference: ${docRef}` : ''}
              </small>
            </div>
            <div class="document-text" style="max-height: 400px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px;">
              <p style="white-space: pre-wrap; line-height: 1.6; margin: 0;">${documentText}</p>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
          </div>
        </div>
      </div>
    </div>
  `;

  document.body.insertAdjacentHTML('beforeend', modalHtml);

  const modal = new bootstrap.Modal(document.getElementById('documentTextModal'));
  modal.show();

  document.getElementById('documentTextModal').addEventListener('hidden.bs.modal', function() {
    this.remove();
  });
}

// Handle View Text button click
async function handleViewTextClick(referenceData) {
  console.log('handleViewTextClick called with:', referenceData);
  try {
    await loadDataSource();

    if (!dataSourceCache) {
      alert('Unable to load document text data source.');
      return;
    }

    let docId;
    if (typeof referenceData.id === 'string' && referenceData.id.includes('_')) {
      const parts = referenceData.id.split('_');
      docId = parseInt(parts[parts.length - 1]);
    } else {
      docId = parseInt(referenceData.id);
    }

    const document = dataSourceCache[docId];

    if (document && document.text) {
      showDocumentText(
        document.docId,
        document.text,
        referenceData.formattedCitation || referenceData.title
      );
    } else {
      alert(`Document text not found for ID: ${referenceData.id}`);
    }
  } catch (error) {
    console.error('Error loading document text:', error);
    alert('Error loading document text. Please try again.');
  }
}

window.handleViewTextClick = handleViewTextClick;

export async function loadDocumentView(topicKeys, docTopic, metadata, referenceData = null, docLengths = null) {
  const main = document.getElementById('main-view');

  await loadConfig();

  let documentTopics = [];
  let totalTokens = 0;

  if (referenceData && metadata && docTopic && topicKeys) {
    console.log('Reference data:', referenceData);

    let docIndex = -1;

    // Strategy 1: Direct docIndex match
    if (referenceData.docIndex !== undefined) {
      docIndex = referenceData.docIndex;
      console.log(`Strategy 1 - Direct docIndex: ${docIndex}`);
    }

    // Strategy 2: Direct docNum match
    if (docIndex === -1 && referenceData.id) {
      docIndex = metadata.findIndex(doc => parseInt(doc.docNum) === parseInt(referenceData.id));
      console.log(`Strategy 2 - Direct docNum match for ID ${referenceData.id}: index ${docIndex}`);
    }

    // Strategy 3: Match by title content
    if (docIndex === -1 && referenceData.title) {
      docIndex = metadata.findIndex(doc =>
        doc.title && doc.title.toLowerCase().includes(referenceData.title.toLowerCase())
      );
      console.log(`Strategy 3 - Title content match: index ${docIndex}`);
    }

    if (docIndex !== -1 && docTopic[docIndex]) {
      const docTopicProportions = docTopic[docIndex];

      totalTokens = (docLengths && docLengths[docIndex] !== undefined)
        ? docLengths[docIndex]
        : 0;
      console.log(`Token count for document ${docIndex}: ${totalTokens} tokens`);

      const settings = await window.getSettings();
      const extractWords = settings.wordsInTopic || 50;
      const displayWords = 15;

      console.log(`Extracting enhanced words for document topic display...`);
      let allEnhancedWords = null;
      try {
        allEnhancedWords = await extractTopicWords(topicKeys.length, extractWords);
      } catch (error) {
        console.log(`State file extraction failed:`, error.message);
      }

      documentTopics = topicKeys.map((topic, topicIndex) => {
        const proportion = docTopicProportions[topicIndex] || 0;
        const tokens = Math.round(totalTokens * proportion);
        const percentage = (proportion * 100).toFixed(1);

        let topWords;

        const enhancedTopicWords = allEnhancedWords && allEnhancedWords[topicIndex];

        if (enhancedTopicWords && Array.isArray(enhancedTopicWords) && enhancedTopicWords.length > 0) {
          const displayWordsArray = enhancedTopicWords.slice(0, displayWords);
          topWords = displayWordsArray.join(', ') + (enhancedTopicWords.length > displayWords ? '...' : '');
        } else {
          if (topic.words && Array.isArray(topic.words) && topic.words.length > 0) {
            const displayWordsArray = topic.words.slice(0, displayWords);
            topWords = displayWordsArray.join(', ') + (topic.words.length > displayWords ? '...' : '');
          } else {
            topWords = `Topic ${topicIndex + 1}`;
          }
        }

        return {
          topicNumber: topicIndex + 1,
          topicIndex: topicIndex,
          topWords: topWords,
          proportion: proportion,
          percentage: percentage,
          tokens: tokens
        };
      });

      documentTopics = documentTopics
        .filter(topic => topic.tokens > 0)
        .sort((a, b) => b.tokens - a.tokens);

      console.log(`Found ${documentTopics.length} topics with non-zero proportions`);
    } else {
      console.log('Document not found in metadata or no topic data available');
    }
  }

  let html = `<div class="card"><div class="card-body">`;

  html += `<h3>Document View</h3>`;

  if (referenceData) {
    // Try to load formatted citation from bibliography if available
    let bibliographyCitation = null;
    if (referenceData.docIndex !== undefined) {
      try {
        const bibliographyPath = appConfig?.bibliography?.path || 'sample_data/bibliography.json';
        const response = await fetch(ensureAbsolutePath(bibliographyPath));
        if (response.ok) {
          const bibliographyData = await response.json();

          // Add _docIndex to each entry if not present (same as bibliography.js does)
          bibliographyData.forEach((doc, index) => {
            if (doc._docIndex === undefined) {
              doc._docIndex = index;
            }
          });

          const bibEntry = bibliographyData.find(doc => doc._docIndex === referenceData.docIndex);

          if (bibEntry) {
            // Use the formatted citation from bibliography
            bibliographyCitation = bibEntry['formatted-citation'] || bibEntry._formattedCitation;
          }
        }
      } catch (error) {
        // Bibliography not available, will fall back to metadata citation
        console.log('[Document] Bibliography not available, using metadata citation');
      }
    }

    html += `<div class="alert alert-info mb-4">`;
    html += `<h5 class="alert-heading"><i class="bi bi-bookmark"></i> Reference Information</h5>`;

    let citationHtml = '';
    // Priority 1: Use bibliography citation if available
    if (bibliographyCitation) {
      citationHtml = `<div class="citation-text mb-3">${bibliographyCitation}</div>`;
    }
    // Priority 2: Use formatted citation from metadata
    else if (referenceData.formattedCitation) {
      citationHtml = `<div class="citation-text mb-3"><strong>${referenceData.formattedCitation}</strong></div>`;
    }
    // Priority 3: Construct citation from metadata fields
    else if (referenceData.title || referenceData.author || referenceData.year) {
      citationHtml = `<div class="citation-text mb-3">`;
      if (referenceData.title) citationHtml += `<strong>${referenceData.title}</strong>`;
      if (referenceData.author) citationHtml += `, ${referenceData.author}`;
      if (referenceData.year) citationHtml += ` (${referenceData.year})`;
      citationHtml += `</div>`;
    }
    html += citationHtml;

    html += `<div class="d-flex justify-content-between align-items-center">`;
    html += `<div>`;
    html += `<span class="badge bg-secondary me-2">Tokens: ${totalTokens}</span>`;

    // Cite button - always visible
    html += `<button type="button" class="btn btn-primary btn-sm" onclick="window.page('/citation/${referenceData.docIndex}')">`;
    html += `<i class="bi bi-quote"></i> Cite`;
    html += `</button>`;


    if (!appConfig?.embargo) {
      html += `<button type="button" class="btn btn-primary btn-sm" onclick="handleViewTextClick(${JSON.stringify(referenceData).replace(/"/g, '&quot;')})">`;
      html += `View Text`;
      html += `</button>`;
    }

    html += `</div>`;
    html += `</div>`;
    html += `</div>`;
  }

  if (referenceData) {
    html += `<div class="card">`;
    html += `<div class="card-header"><h5 class="mb-0">Topic Analysis</h5></div>`;
    html += `<div class="card-body">`;

    if (documentTopics.length > 0) {
      html += `<div class="table-responsive">`;
      html += `<table class="table table-striped table-hover">`;
      html += `<thead class="table-dark">`;
      html += `<tr>`;
      html += `<th scope="col" style="width: 80px; white-space: nowrap;">Topic</th>`;
      html += `<th scope="col" style="width: 260px;">Top Words</th>`;
      html += `<th scope="col" style="width: 130px;">Proportion</th>`;
      html += `<th scope="col" style="width: 80px;">%</th>`;
      html += `<th scope="col" style="width: 60px;">Tokens</th>`;
      html += `</tr>`;
      html += `</thead>`;
      html += `<tbody>`;

      documentTopics.forEach(topic => {
        html += `<tr>`;
        html += `<td style="white-space: nowrap;"><a href="#" onclick="navigateToTopicFromDocument(${topic.topicIndex}); return false;" class="text-decoration-none">Topic ${topic.topicNumber}</a></td>`;
        html += `<td style="word-wrap: break-word; max-width: 260px;"><a href="#" onclick="navigateToTopicFromDocument(${topic.topicIndex}); return false;" class="text-decoration-none text-muted">${topic.topWords}</a></td>`;

        const barWidth = Math.round(parseFloat(topic.percentage) * 0.9);
        html += `<td><div style="display:inline-block;width:90px;height:18px;vertical-align:middle;background:#ffffff;border:1px solid #dee2e6;border-radius:9px;overflow:hidden;box-shadow:inset 0 1px 2px rgba(0,0,0,0.1);"><div style="height:18px;width:${barWidth}px;background:linear-gradient(90deg, #0d6efd 0%, #084298 100%);border-radius:8px 0 0 8px;"></div></div></td>`;

        html += `<td><span class="badge bg-primary">${topic.percentage}%</span></td>`;
        html += `<td><span class="fw-bold">${topic.tokens}</span></td>`;
        html += `</tr>`;
      });

      html += `</tbody>`;
      html += `</table>`;
      html += `</div>`;

      html += `<div class="mt-3">`;
      html += `<small class="text-muted">`;
      html += `<strong>Column Guide:</strong> <strong>Proportion</strong> bars show the share of this document assigned to each topic. <strong>%</strong> shows what percent of this document's tokens are assigned to each topic. <strong>Tokens</strong> is the actual token count for each topic in this document.<br><span class="text-warning"><i class="bi bi-exclamation-triangle"></i> In rare cases, the model may assign 100% of a document to a single topic, even if not all words in the document are among the topic's top words. This reflects the topic model's output, not a literal word match.</span>`;
      html += `</small>`;
      html += `</div>`;
    } else {
      html += `<p class="text-muted">No topic data available for this document.</p>`;
    }

    html += `</div>`;
    html += `</div>`;
  } else {
    html += `<div class="alert alert-warning">`;
    html += `<h5 class="alert-heading"><i class="bi bi-info-circle"></i> No Document Selected</h5>`;
    html += `<p>To view a document's topic analysis, please:</p>`;
    html += `<ul>`;
    html += `<li>Click on a document in the <strong>Topic View</strong> (from the "Top Documents" table)</li>`;
    html += `<li>Or use a direct link like <code>/document/0</code> (where 0 is the document index)</li>`;
    html += `</ul>`;
    html += `<p class="mb-0">You can also navigate back to the <a href="/">Overview</a> to explore topics.</p>`;
    html += `</div>`;
  }

  html += `</div></div>`;

  main.innerHTML = html;
}

// Global function to navigate to topic from document view
window.navigateToTopicFromDocument = function(topicIndex) {
  window.page(`/topic/${topicIndex + 1}`); // Use 1-based numbering in URL
};
