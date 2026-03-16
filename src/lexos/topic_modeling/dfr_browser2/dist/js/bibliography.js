// Bibliography View
/*
Available in browser console:
await window.bibliographyCache.info()    // Check cache status
await window.bibliographyCache.clear()   // Clear cache
await window.bibliographyCache.reload()  // Clear and reload from network
*/

// Helper function to ensure paths work on any sub-path deployment
function ensureAbsolutePath(path) {
  if (!path) return path;
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  if (path.startsWith('/')) return (window.dfrBasePath || '') + path;
  return path;
}

// Global configuration object
let appConfig = null;
let bibliographyData = null;

// Load application configuration
async function loadConfig() {
  try {
    const response = await fetch('config.json');
    appConfig = await response.json();
    return appConfig;
  } catch (error) {
    console.error('Failed to load configuration:', error);
    // Fallback to default English configuration
    appConfig = {
      language: {
        default: 'en',
        configs: {
          'en': { locale: 'en-US' }
        }
      }
    };
    return appConfig;
  }
}

// IndexedDB helper functions for caching
async function getCachedBibliography() {
  return new Promise((resolve) => {
    const request = indexedDB.open('BibliographyDB', 1);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('bibliography')) {
        db.createObjectStore('bibliography');
      }
    };

    request.onsuccess = () => {
      const db = request.result;

      // Check if object store exists before trying to access it
      if (!db.objectStoreNames.contains('bibliography')) {
        console.log('Bibliography object store not found, cache is empty');
        db.close();
        resolve(null);
        return;
      }

      const transaction = db.transaction(['bibliography'], 'readonly');
      const store = transaction.objectStore('bibliography');
      const getRequest = store.get('data');

      getRequest.onsuccess = () => {
        const result = getRequest.result;
        db.close();
        // Cache expires after 24 hours
        if (result && Date.now() - result.timestamp < 24 * 60 * 60 * 1000) {
          console.log('Bibliography loaded from IndexedDB cache');
          resolve(result.data);
        } else {
          if (result) {
            console.log('Bibliography cache expired, will fetch from network');
          }
          resolve(null);
        }
      };

      getRequest.onerror = () => {
        console.log('Error reading from IndexedDB cache');
        db.close();
        resolve(null);
      };
    };

    request.onerror = () => {
      console.log('Error opening IndexedDB');
      resolve(null);
    };
  });
}

async function cacheBibliography(data) {
  return new Promise((resolve) => {
    const request = indexedDB.open('BibliographyDB', 1);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('bibliography')) {
        db.createObjectStore('bibliography');
      }
    };

    request.onsuccess = () => {
      const db = request.result;

      // Check if object store exists before trying to access it
      if (!db.objectStoreNames.contains('bibliography')) {
        console.log('Bibliography object store not found, cannot cache');
        db.close();
        resolve();
        return;
      }

      const transaction = db.transaction(['bibliography'], 'readwrite');
      const store = transaction.objectStore('bibliography');

      store.put({
        data: data,
        timestamp: Date.now()
      }, 'data');

      transaction.oncomplete = () => {
        console.log('Bibliography cached to IndexedDB');
        db.close();
        resolve();
      };

      transaction.onerror = () => {
        console.log('Error caching bibliography to IndexedDB');
        db.close();
        resolve();
      };
    };

    request.onerror = () => {
      console.log('Error opening IndexedDB for caching');
      resolve();
    };
  });
}

async function clearBibliographyCache() {
  return new Promise((resolve) => {
    const request = indexedDB.open('BibliographyDB', 1);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('bibliography')) {
        db.createObjectStore('bibliography');
      }
    };

    request.onsuccess = () => {
      const db = request.result;

      // Check if object store exists before trying to access it
      if (!db.objectStoreNames.contains('bibliography')) {
        console.log('Bibliography object store not found, cache is already empty');
        db.close();
        resolve();
        return;
      }

      const transaction = db.transaction(['bibliography'], 'readwrite');
      const store = transaction.objectStore('bibliography');

      const deleteRequest = store.delete('data');

      deleteRequest.onsuccess = () => {
        console.log('Bibliography cache entry deleted successfully');
      };

      deleteRequest.onerror = () => {
        console.log('Error deleting bibliography cache entry');
      };

      transaction.oncomplete = () => {
        console.log('Bibliography cache cleared');
        db.close();
        resolve();
      };

      transaction.onerror = () => {
        console.log('Error clearing bibliography cache');
        db.close();
        resolve();
      };
    };

    request.onerror = () => {
      console.log('Error opening IndexedDB for cache clearing');
      resolve();
    };
  });
}

// Get cache info for debugging/UI display
async function getBibliographyCacheInfo() {
  return new Promise((resolve) => {
    const request = indexedDB.open('BibliographyDB', 1);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('bibliography')) {
        db.createObjectStore('bibliography');
      }
    };

    request.onsuccess = () => {
      const db = request.result;

      // Check if object store exists before trying to access it
      if (!db.objectStoreNames.contains('bibliography')) {
        console.log('Bibliography object store not found, cache is empty');
        db.close();
        resolve({ cached: false });
        return;
      }

      const transaction = db.transaction(['bibliography'], 'readonly');
      const store = transaction.objectStore('bibliography');
      const getRequest = store.get('data');

      getRequest.onsuccess = () => {
        const result = getRequest.result;
        db.close();
        if (result) {
          const cacheAge = Date.now() - result.timestamp;
          const isExpired = cacheAge > 24 * 60 * 60 * 1000;
          resolve({
            cached: true,
            expired: isExpired,
            timestamp: result.timestamp,
            ageMinutes: Math.round(cacheAge / (1000 * 60)),
            entries: result.data ? result.data.length : 0,
            sizeEstimate: JSON.stringify(result.data).length
          });
        } else {
          resolve({ cached: false });
        }
      };

      getRequest.onerror = () => {
        db.close();
        resolve({ cached: false });
      };
    };

    request.onerror = () => resolve({ cached: false });
  });
}

// Parse metadata CSV format
async function parseMetadataCSV(text) {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return [];

  const headers = lines[0].split(',').map(h => h.trim());
  const documents = [];

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',');
    const doc = {};
    headers.forEach((header, index) => {
      doc[header] = values[index] ? values[index].trim() : '';
    });
    documents.push(doc);
  }

  return documents;
}

// Convert metadata row to CSL (Citation Style Language) format
function metadataToCSL(metadata, index) {
  return {
    id: metadata.id || metadata.docNum || metadata.docName || `doc_${index}`,
    type: 'article-journal',
    title: metadata.title || 'Untitled',
    author: metadata.author ? [{ literal: metadata.author }] : [],
    issued: metadata.year ? { 'date-parts': [[parseInt(metadata.year)]] } : undefined,
    'container-title': metadata.journaltitle || metadata.journal || undefined,
    volume: metadata.volume || undefined,
    issue: metadata.issue || undefined,
    page: metadata.pages || undefined,
    publisher: metadata.publisher || undefined,
    DOI: metadata.doi || undefined,
    URL: metadata.url || undefined,
    // Store original metadata with the array index (docIndex)
    _metadata: metadata,
    _docIndex: index  // Use the array index directly
  };
}

// Generate Chicago-style citation from CSL data
function generateChicagoCitation(cslDoc) {
  let citation = '';

  // Author
  if (cslDoc.author && cslDoc.author.length > 0) {
    const authors = cslDoc.author.map(a => a.literal || `${a.family || ''}, ${a.given || ''}`.trim()).join(', ');
    citation += authors + '. ';
  }

  // Title
  if (cslDoc.title) {
    citation += `"${cslDoc.title}." `;
  }

  // Container (journal/book)
  if (cslDoc['container-title']) {
    citation += `<i>${cslDoc['container-title']}</i>`;

    // Volume and issue
    if (cslDoc.volume) {
      citation += ` ${cslDoc.volume}`;
      if (cslDoc.issue) {
        citation += `, no. ${cslDoc.issue}`;
      }
    }

    // Year
    if (cslDoc.issued && cslDoc.issued['date-parts'] && cslDoc.issued['date-parts'][0]) {
      citation += ` (${cslDoc.issued['date-parts'][0][0]})`;
    }

    // Pages
    if (cslDoc.page) {
      citation += `: ${cslDoc.page}`;
    }

    citation += '.';
  }

  return citation || 'Citation unavailable';
}

// Load bibliography from metadata.csv fallback
async function loadBibliographyFromMetadata() {
  console.log('Loading bibliography from metadata.csv fallback...');

  try {
    const response = await fetch(ensureAbsolutePath('sample_data/metadata.csv'));
    if (!response.ok) {
      throw new Error(`Failed to load metadata.csv: HTTP ${response.status}`);
    }

    const text = await response.text();
    const metadata = await parseMetadataCSV(text);

    // Convert to CSL format with index
    const cslDocs = metadata.map((doc, index) => metadataToCSL(doc, index));

    // Add formatted citations
    cslDocs.forEach(doc => {
      doc._formattedCitation = generateChicagoCitation(doc);
    });

    // Mark as metadata fallback
    cslDocs._source = 'metadata-fallback';
    cslDocs._sourceTimestamp = Date.now();

    console.log(`Loaded ${cslDocs.length} documents from metadata.csv`);
    return cslDocs;
  } catch (error) {
    console.error('Error loading bibliography from metadata:', error);
    return [];
  }
}

// Load bibliography data (from JSON file or metadata fallback)
async function loadBibliographyData() {
  // Try to load from cache first
  const cached = await getCachedBibliography();
  if (cached) {
    bibliographyData = cached;
    return bibliographyData;
  }

  // Try to load from bibliography.json
  if (!appConfig) {
    await loadConfig();
  }

  const bibliographyPath = appConfig?.bibliography?.path || 'sample_data/bibliography.json';

  try {
    console.log(`Loading bibliography from ${bibliographyPath}...`);
    const response = await fetch(ensureAbsolutePath(bibliographyPath));

    if (response.ok) {
      bibliographyData = await response.json();

      // Add _docIndex to each document based on array position
      bibliographyData.forEach((doc, index) => {
        if (doc._docIndex === undefined) {
          doc._docIndex = index;
        }
      });

      console.log(`Loaded ${bibliographyData.length} documents from bibliography.json`);

      // Cache the data
      await cacheBibliography(bibliographyData);

      return bibliographyData;
    } else {
      console.warn(`Bibliography file not found (${response.status}), falling back to metadata.csv`);
      throw new Error('Bibliography file not available');
    }
  } catch (error) {
    console.warn('Could not load bibliography.json, using metadata.csv fallback');

    // Fallback to metadata.csv
    bibliographyData = await loadBibliographyFromMetadata();

    // Try to cache the fallback data
    if (bibliographyData && bibliographyData.length > 0) {
      await cacheBibliography(bibliographyData);
    }

    return bibliographyData;
  }
}

// Get current language from config
function getCurrentLanguage() {
  return appConfig?.language?.default || 'en';
}

// Get all available CSL fields from bibliography data
// Get all available CSL fields from bibliography data with statistics
function getAvailableCSLFields(bibliographyData) {
  const fieldSet = new Set();
  const fieldCounts = {};

  bibliographyData.forEach(doc => {
    Object.keys(doc).forEach(key => {
      if (key !== 'formatted-citation' && !key.startsWith('_')) {
        fieldSet.add(key);
        fieldCounts[key] = (fieldCounts[key] || 0) + 1;
      }
    });
  });

  // Convert to array and sort by frequency (most common first), then alphabetically
  const fields = Array.from(fieldSet).sort((a, b) => {
    const countDiff = (fieldCounts[b] || 0) - (fieldCounts[a] || 0);
    if (countDiff !== 0) return countDiff;
    return a.localeCompare(b);
  });

  return fields.map(field => ({
    key: field,
    name: field.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    count: fieldCounts[field] || 0,
    percentage: Math.round((fieldCounts[field] || 0) / bibliographyData.length * 100)
  }));
}

// Generate custom sorting modal HTML
function generateCustomSortingModal(availableFields) {
  return `
    <div class="modal fade" id="customSortModal" tabindex="-1" aria-labelledby="customSortModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-lg">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="customSortModalLabel">
              <i class="bi bi-sort-alpha-down"></i> Custom Sorting Options
            </h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>

          <div class="modal-body">
            <p class="text-muted mb-4">
              Select up to 3 fields to define your custom sorting priority.
              Fields will be sorted in the order you select them.
            </p>

            <div class="row">
              <div class="col-md-6">
                <h6 class="mb-3">
                  <a href="#json-csl-info" id="json-csl-link" class="text-decoration-none">
                    Available JSON-CSL Fields
                    <i class="bi bi-info-circle ms-1" title="Click to learn about JSON-CSL"></i>
                  </a>
                </h6>
                <div class="border rounded p-3" style="max-height: 400px; overflow-y: auto;">
                  <div id="available-fields" class="list-group list-group-flush">
                    ${availableFields.map(field => `
                      <div class="list-group-item list-group-item-action available-field-item"
                           data-field="${field.key}"
                           style="cursor: pointer; border: none; padding: 8px 12px;">
                        <div class="d-flex justify-content-between align-items-center">
                          <div>
                            <strong>${field.name}</strong>
                            <br>
                            <small class="text-muted">${field.key}</small>
                          </div>
                          <div class="text-end">
                            <span class="badge bg-secondary">${field.percentage}%</span>
                            <br>
                            <small class="text-muted">${field.count} entries</small>
                          </div>
                        </div>
                      </div>
                    `).join('')}
                  </div>
                </div>
              </div>

              <div class="col-md-6">
                <h6 class="mb-3">Selected Sorting Order</h6>
                <div class="border rounded p-3" style="min-height: 400px;">
                  <div id="selected-fields" class="d-grid gap-2">
                    <div class="text-muted text-center py-5">
                      <i class="bi bi-arrow-left-square fs-3"></i>
                      <p class="mt-2">Click fields from the left to add them here</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div class="mt-3">
              <small class="text-muted">
                <i class="bi bi-info-circle"></i>
                Tip: The percentage shows how many bibliography entries contain each field.
                Higher percentages indicate more useful sorting fields.
              </small>
            </div>
          </div>

          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
            <button type="button" class="btn btn-outline-warning" id="reset-custom-sort">
              <i class="bi bi-arrow-clockwise"></i> Reset
            </button>
            <button type="button" class="btn btn-primary" id="apply-custom-sort" disabled>
              <i class="bi bi-check-lg"></i> Apply Custom Sort
            </button>
          </div>

          <!-- JSON-CSL Information Section (initially hidden) -->
          <div id="json-csl-info" class="mt-4 p-4 bg-light rounded border" style="display: none;">
            <h5 class="mb-3">
              <i class="bi bi-file-earmark-code"></i>
              JSON-CSL Fields Reference
            </h5>

            <p class="mb-3">
              Citation Style Language (CSL) JSON is a structured format for representing bibliographic information.
              This system uses JSON-CSL fields to organize and sort your bibliography entries.
            </p>

            <div class="row">
              <div class="col-md-6">
                <h6>Common Core Fields</h6>
                <div class="table-responsive">
                  <table class="table table-sm">
                    <thead>
                      <tr>
                        <th>Field</th>
                        <th>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr><td><code>title</code></td><td>Title of the work</td></tr>
                      <tr><td><code>author</code></td><td>Author(s) information</td></tr>
                      <tr><td><code>issued</code></td><td>Publication date</td></tr>
                      <tr><td><code>container-title</code></td><td>Journal, book, or collection title</td></tr>
                      <tr><td><code>publisher</code></td><td>Publishing organization</td></tr>
                      <tr><td><code>volume</code></td><td>Volume number</td></tr>
                      <tr><td><code>issue</code></td><td>Issue number</td></tr>
                      <tr><td><code>page</code></td><td>Page numbers</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div class="col-md-6">
                <h6>Additional Fields</h6>
                <div class="table-responsive">
                  <table class="table table-sm">
                    <thead>
                      <tr>
                        <th>Field</th>
                        <th>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr><td><code>DOI</code></td><td>Digital Object Identifier</td></tr>
                      <tr><td><code>URL</code></td><td>Web address</td></tr>
                      <tr><td><code>abstract</code></td><td>Summary or abstract</td></tr>
                      <tr><td><code>keyword</code></td><td>Subject keywords</td></tr>
                      <tr><td><code>language</code></td><td>Publication language</td></tr>
                      <tr><td><code>type</code></td><td>Item type (article, book, etc.)</td></tr>
                      <tr><td><code>editor</code></td><td>Editor information</td></tr>
                      <tr><td><code>ISBN</code></td><td>International Standard Book Number</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div class="mt-3">
              <h6>Sorting Tips</h6>
              <ul class="list-unstyled small">
                <li><i class="bi bi-check-circle text-success"></i> <strong>Primary sort:</strong> Use fields with high percentage coverage (like 'issued' for chronological)</li>
                <li><i class="bi bi-check-circle text-success"></i> <strong>Secondary sort:</strong> Add 'author' or 'title' to break ties</li>
                <li><i class="bi bi-check-circle text-success"></i> <strong>Specialized sort:</strong> Use 'container-title' to group by journal or publication</li>
                <li><i class="bi bi-info-circle text-info"></i> <strong>Field coverage:</strong> The percentage shows how many entries contain each field</li>
              </ul>
            </div>

            <div class="mt-3">
              <h6>Learn More</h6>
              <p class="small mb-3">
                For complete JSON-CSL documentation, visit the
                <a href="https://citeproc-js.readthedocs.io/en/latest/csl-json/markup.html" target="_blank" rel="noopener">
                  Citation Style Language documentation <i class="bi bi-box-arrow-up-right"></i>
                </a>
              </p>
            </div>

            <div class="text-center">
              <button type="button" class="btn btn-outline-primary btn-sm" id="return-to-top">
                <i class="bi bi-arrow-up-circle"></i> Return to Top
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

// Setup custom sorting modal event handlers
function setupCustomSortingModal() {
  const selectedFields = [];
  const maxFields = 3;

  // Handle clicking on available fields
  document.addEventListener('click', function(e) {
    if (e.target.closest('.available-field-item')) {
      const fieldItem = e.target.closest('.available-field-item');
      const fieldKey = fieldItem.getAttribute('data-field');

      // Check if already selected or at max capacity
      if (selectedFields.includes(fieldKey)) {
        return; // Already selected
      }

      if (selectedFields.length >= maxFields) {
        return; // Max capacity reached
      }

      // Add to selected fields
      selectedFields.push(fieldKey);

      // Update UI
      updateSelectedFieldsDisplay();
      updateAvailableFieldsState();
      updateApplyButtonState();
    }
  });

  // Handle clicking on selected fields (to remove them)
  document.addEventListener('click', function(e) {
    if (e.target.closest('.selected-field-item')) {
      const fieldItem = e.target.closest('.selected-field-item');
      const fieldKey = fieldItem.getAttribute('data-field');

      // Remove from selected fields
      const index = selectedFields.indexOf(fieldKey);
      if (index > -1) {
        selectedFields.splice(index, 1);
      }

      // Update UI
      updateSelectedFieldsDisplay();
      updateAvailableFieldsState();
      updateApplyButtonState();
    }
  });

  // Handle reset button
  document.addEventListener('click', function(e) {
    if (e.target.closest('#reset-custom-sort')) {
      selectedFields.length = 0; // Clear array
      updateSelectedFieldsDisplay();
      updateAvailableFieldsState();
      updateApplyButtonState();
    }
  });

  // Handle apply button
  document.addEventListener('click', function(e) {
    if (e.target.closest('#apply-custom-sort')) {
      console.log('Apply button clicked, selectedFields:', selectedFields);

      // Close the modal
      const modal = bootstrap.Modal.getInstance(document.getElementById('customSortModal'));
      if (modal) {
        modal.hide();
      }

      // Update the sort-by dropdown with custom fields
      updateSortByDropdown(selectedFields);

      // Set the dropdown to custom and trigger refresh
      const sortBySelect = document.getElementById('sort-by');
      if (sortBySelect) {
        sortBySelect.value = 'custom';
        // Store custom fields for sorting function
        sortBySelect.customFields = [...selectedFields];
        console.log('Custom fields stored:', sortBySelect.customFields);

        // Trigger refresh to apply the custom sorting
        if (window.refreshBibliography) {
          console.log('Calling refreshBibliography...');
          window.refreshBibliography();
        } else {
          console.error('refreshBibliography not found on window');
        }
      }

      // Show success message
      showCustomSortMessage(selectedFields);
    }
  });

  // Handle JSON-CSL info link
  document.addEventListener('click', function(e) {
    if (e.target.closest('#json-csl-link')) {
      e.preventDefault();
      const infoSection = document.getElementById('json-csl-info');
      if (infoSection) {
        infoSection.style.display = 'block';
        infoSection.scrollIntoView({ behavior: 'smooth' });
      }
    }
  });

  // Handle return to top button
  document.addEventListener('click', function(e) {
    if (e.target.closest('#return-to-top')) {
      const modalBody = document.querySelector('#customSortModal .modal-body');
      if (modalBody) {
        modalBody.scrollTop = 0;
      }
      const infoSection = document.getElementById('json-csl-info');
      if (infoSection) {
        infoSection.style.display = 'none';
      }
    }
  });

  // Update selected fields display
  function updateSelectedFieldsDisplay() {
    const container = document.getElementById('selected-fields');
    if (!container) return;

    if (selectedFields.length === 0) {
      container.innerHTML = `
        <div class="text-muted text-center py-5">
          <i class="bi bi-arrow-left-square fs-3"></i>
          <p class="mt-2">Click fields from the left to add them here</p>
        </div>
      `;
    } else {
      container.innerHTML = selectedFields.map((fieldKey, index) => {
        const fieldData = getFieldDisplayName(fieldKey);
        return `
          <div class="selected-field-item border rounded p-3 bg-light" data-field="${fieldKey}" style="cursor: pointer;">
            <div class="d-flex justify-content-between align-items-center">
              <div>
                <span class="badge bg-primary me-2">${index + 1}</span>
                <strong>${fieldData.name}</strong>
                <br>
                <small class="text-muted">${fieldKey}</small>
              </div>
              <div>
                <i class="bi bi-x-circle text-danger fs-5"></i>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }
  }

  // Update available fields state (disable selected ones)
  function updateAvailableFieldsState() {
    const availableItems = document.querySelectorAll('.available-field-item');
    availableItems.forEach(item => {
      const fieldKey = item.getAttribute('data-field');
      const isSelected = selectedFields.includes(fieldKey);
      const isAtMaxCapacity = selectedFields.length >= maxFields;

      if (isSelected) {
        item.classList.add('disabled', 'bg-secondary');
        item.style.opacity = '0.5';
        item.style.cursor = 'not-allowed';
      } else if (isAtMaxCapacity) {
        item.classList.add('disabled');
        item.style.opacity = '0.6';
        item.style.cursor = 'not-allowed';
      } else {
        item.classList.remove('disabled', 'bg-secondary');
        item.style.opacity = '1';
        item.style.cursor = 'pointer';
      }
    });
  }

  // Update apply button state
  function updateApplyButtonState() {
    const applyBtn = document.getElementById('apply-custom-sort');
    if (applyBtn) {
      applyBtn.disabled = selectedFields.length === 0;
    }
  }

  // Get field display name
  function getFieldDisplayName(fieldKey) {
    return {
      key: fieldKey,
      name: fieldKey.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    };
  }

  // Update sort-by dropdown with custom fields
  function updateSortByDropdown(fields) {
    const sortBySelect = document.getElementById('sort-by');
    if (!sortBySelect || fields.length === 0) return;

    // Find the custom option
    const customOption = sortBySelect.querySelector('option[value="custom"]');
    if (customOption) {
      const fieldNames = fields.map(key => getFieldDisplayName(key).name);
      const displayText = `${fieldNames.join(' → ')}`;
      customOption.textContent = displayText;
    }
  }

  // Show custom sort message (success)
  function showCustomSortMessage(fields) {
    const fieldNames = fields.map(key => getFieldDisplayName(key).name).join(' → ');
    const alertHtml = `
      <div class="alert alert-success alert-dismissible fade show" role="alert">
        <i class="bi bi-check-circle"></i>
        <strong>Custom sorting applied:</strong> ${fieldNames}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
      </div>
    `;

    const container = document.getElementById('bibliography-container');
    if (container) {
      // Insert at the top of the container
      container.insertAdjacentHTML('afterbegin', alertHtml);

      // Auto-dismiss after 5 seconds
      setTimeout(() => {
        const alert = container.querySelector('.alert');
        if (alert) {
          alert.remove();
        }
      }, 5000);
    }
  }
}

// Extract value from CSL document for a given field
function extractFromCSL(doc, field) {
  if (field === 'author') {
    if (doc.author && doc.author.length > 0) {
      return doc.author.map(a => a.literal || `${a.family || ''}, ${a.given || ''}`.trim()).join(', ');
    }
    return '';
  }

  if (field === 'year' || field === 'issued') {
    if (doc.issued && doc.issued['date-parts'] && doc.issued['date-parts'][0]) {
      return doc.issued['date-parts'][0][0];
    }
    return 0;
  }

  return doc[field] || '';
}

// Extract CSL field value with proper handling
function extractCSLFieldValue(doc, fieldKey) {
  // Special handling for common fields
  if (fieldKey === 'author') {
    return extractFromCSL(doc, 'author');
  }

  if (fieldKey === 'year' || fieldKey === 'issued') {
    return extractFromCSL(doc, 'year');
  }

  // Handle nested fields
  if (fieldKey.includes('.')) {
    const parts = fieldKey.split('.');
    let value = doc;
    for (const part of parts) {
      value = value?.[part];
      if (value === undefined) return '';
    }
    return value;
  }

  return doc[fieldKey] || '';
}

// Check if a field contains numeric data
function isNumericField(fieldKey) {
  const numericFields = ['year', 'volume', 'issue', 'page', 'number'];
  return numericFields.includes(fieldKey);
}

// Check if a field contains date data
function isDateField(fieldKey) {
  const dateFields = ['issued', 'accessed', 'event-date', 'original-date'];
  return dateFields.includes(fieldKey);
}

// Sort documents based on sorting criteria
function sortDocuments(docs, sortBy, sortOrder, locale) {
  const sorted = [...docs];

  const compareFunction = (a, b) => {
    let valA, valB;

    if (sortBy === 'year-author') {
      const yearA = extractFromCSL(a, 'year') || 'Unknown';
      const yearB = extractFromCSL(b, 'year') || 'Unknown';
      if (yearA !== yearB) {
        // Sort Unknown to the end
        if (yearA === 'Unknown') return sortOrder === 'asc' ? 1 : -1;
        if (yearB === 'Unknown') return sortOrder === 'asc' ? -1 : 1;
        return sortOrder === 'asc' ? yearA - yearB : yearB - yearA;
      }
      valA = extractFromCSL(a, 'author').toLowerCase();
      valB = extractFromCSL(b, 'author').toLowerCase();
    } else if (sortBy === 'author-year') {
      valA = extractFromCSL(a, 'author').toLowerCase();
      valB = extractFromCSL(b, 'author').toLowerCase();
      if (valA !== valB) {
        return sortOrder === 'asc' ? valA.localeCompare(valB, locale) : valB.localeCompare(valA, locale);
      }
      const yearA = extractFromCSL(a, 'year') || 'Unknown';
      const yearB = extractFromCSL(b, 'year') || 'Unknown';
      // Sort Unknown to the end
      if (yearA === 'Unknown') return sortOrder === 'asc' ? 1 : -1;
      if (yearB === 'Unknown') return sortOrder === 'asc' ? -1 : 1;
      return sortOrder === 'asc' ? yearA - yearB : yearB - yearA;
    } else if (sortBy === 'title') {
      valA = extractFromCSL(a, 'title').toLowerCase();
      valB = extractFromCSL(b, 'title').toLowerCase();
    } else if (sortBy === 'journal') {
      valA = extractFromCSL(a, 'container-title').toLowerCase();
      valB = extractFromCSL(b, 'container-title').toLowerCase();
    } else if (sortBy === 'custom') {
      const sortBySelect = document.getElementById('sort-by');
      const customFields = sortBySelect?.customFields || ['year'];
      const customOrders = sortBySelect?.customOrders || ['asc'];

      for (let i = 0; i < customFields.length; i++) {
        const field = customFields[i];
        const order = customOrders[i];

        let fieldValA = extractCSLFieldValue(a, field);
        let fieldValB = extractCSLFieldValue(b, field);

        // Handle numeric/date fields
        if (isNumericField(field) || isDateField(field)) {
          fieldValA = parseFloat(fieldValA) || 0;
          fieldValB = parseFloat(fieldValB) || 0;
          if (fieldValA !== fieldValB) {
            return order === 'asc' ? fieldValA - fieldValB : fieldValB - fieldValA;
          }
        } else {
          // String comparison
          fieldValA = String(fieldValA).toLowerCase();
          fieldValB = String(fieldValB).toLowerCase();
          if (fieldValA !== fieldValB) {
            return order === 'asc' ? fieldValA.localeCompare(fieldValB, locale) : fieldValB.localeCompare(fieldValA, locale);
          }
        }
      }
      return 0;
    }

    if (typeof valA === 'number' && typeof valB === 'number') {
      return sortOrder === 'asc' ? valA - valB : valB - valA;
    }

    return sortOrder === 'asc' ? valA.localeCompare(valB, locale) : valB.localeCompare(valA, locale);
  };

  sorted.sort(compareFunction);
  return sorted;
}

// Generate navigation menu HTML
// Generate navigation menu based on sort type
function generateNavigationHTML(sortBy, bibliographyData, langConfig) {
  let isAlphabetSort = sortBy === 'author-year' || sortBy === 'title';
  let sortField = sortBy === 'author-year' ? 'author' : 'title';

  // Handle custom sorting
  if (sortBy === 'custom') {
    const sortBySelect = document.getElementById('sort-by');
    const customFields = sortBySelect?.customFields || [];
    if (customFields.length > 0) {
      const firstField = customFields[0];
      // Determine if first field should use alphabet or year navigation
      isAlphabetSort = !(firstField === 'year' || isNumericField(firstField) || isDateField(firstField));
      sortField = firstField;
    }
  }

  if (isAlphabetSort) {
    // Get alphabet from language config
    const alphabet = langConfig.alphabet || ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];

    // Find which letters actually have content
    const usedLetters = new Set();

    bibliographyData.forEach(doc => {
      const value = extractCSLFieldValue(doc, sortField);
      if (value && value.length > 0) {
        const firstLetter = String(value).charAt(0).toUpperCase();
        usedLetters.add(firstLetter);
      }
    });

    let html = `<div class="sticky-top" style="top: 20px;">`;
    html += `<div class="card">`;
    html += `<div class="card-header bg-success text-white">`;
    html += `<h6 class="mb-0"><i class="bi bi-alphabet-uppercase"></i> Jump to Letter</h6>`;
    html += `</div>`;
    html += `<div class="card-body p-2" style="max-height: 450px; overflow-y: auto;">`;
    html += `<div class="nav nav-pills" style="flex-wrap: wrap;">`;

    alphabet.forEach(letter => {
      const hasContent = usedLetters.has(letter);
      const linkClass = hasContent ? 'nav-link py-1 px-2 text-success small me-1 mb-1' : 'nav-link py-1 px-2 text-muted small me-1 mb-1';
      const clickable = hasContent ? `data-letter="${letter}" style="cursor: pointer;"` : 'style="cursor: default;"';
      html += `<a class="${linkClass}" ${clickable}>${letter}</a>`;
    });

    html += `</div>`;
    html += `</div>`;
    html += `</div>`;
    html += `</div>`;

    return html;
  } else {
    // Year navigation for date-based sorting
    const uniqueYears = [...new Set(bibliographyData.map(doc => extractFromCSL(doc, 'year') || 'Unknown'))].sort((a, b) => {
      if (a === 'Unknown') return 1;
      if (b === 'Unknown') return -1;
      return parseInt(a) - parseInt(b);
    });

    let html = `<div class="sticky-top" style="top: 20px;">`;
    html += `<div class="card">`;
    html += `<div class="card-header bg-primary text-white">`;
    html += `<h6 class="mb-0"><i class="bi bi-calendar-event"></i> Jump to Year</h6>`;
    html += `</div>`;
    html += `<div class="card-body p-2" style="max-height: 450px; overflow-y: auto;">`;
    html += `<div class="nav nav-pills flex-column">`;

    uniqueYears.forEach(year => {
      const linkClass = 'nav-link py-1 px-2 text-primary small';
      const clickable = `data-year="${year}" style="cursor: pointer;"`;
      html += `<a class="${linkClass}" ${clickable}>${year}</a>`;
    });

    html += `</div>`;
    html += `</div>`;
    html += `</div>`;
    html += `</div>`;

    return html;
  }
}

// Generate bibliography HTML with grouping
function generateBibliographyHTML(sortedDocs, groupingType = 'year', langConfig = null) {
  let html = '';
  let currentGroup = null;

  sortedDocs.forEach((doc, index) => {
    let group = null;

    if (groupingType === 'year') {
      group = extractFromCSL(doc, 'year') || 'Unknown';
    } else if (groupingType === 'alphabet') {
      const firstChar = (extractFromCSL(doc, 'author') || extractFromCSL(doc, 'title') || 'Unknown')[0].toUpperCase();
      group = /[A-Z]/.test(firstChar) ? firstChar : '#';
    }

    // Add group header if this is a new group
    if (group !== null && group !== currentGroup) {
      if (currentGroup !== null) {
        html += '</div>'; // Close previous group
      }

      const groupId = groupingType === 'year' ? `year-${group}` : `letter-${group}`;
      html += `<div class="bibliography-group mb-4" id="${groupId}">`;
      html += `<h4 class="border-bottom pb-2 mb-3">${group}</h4>`;

      currentGroup = group;
    } else if (groupingType === 'none' && currentGroup === null) {
      html += '<div class="bibliography-group mb-4">';
      currentGroup = 'ungrouped';
    }

    // Add entry
    html += generateEntryHTML(doc);
  });

  if (currentGroup !== null) {
    html += '</div>'; // Close final group
  }

  return html;
}

// Generate individual entry HTML
// Generate individual entry HTML
function generateEntryHTML(doc) {
  let html = `<div class="bibliography-entry mb-3 border rounded p-3">`;

  // Use the pre-formatted citation if available
  if (doc['formatted-citation']) {
    html += `<div class="citation-text mb-3">${doc['formatted-citation']}</div>`;
  } else if (doc._formattedCitation) {
    html += `<div class="citation-text mb-3">${doc._formattedCitation}</div>`;
  } else {
    // Fallback to manual formatting
    let citation = [];

    // Author (bold)
    const author = extractFromCSL(doc, 'author');
    if (author) {
      citation.push(`<span class="fw-bold">${author}</span>`);
    }

    // Title (italic)
    const title = extractFromCSL(doc, 'title');
    if (title) {
      citation.push(`<span class="fst-italic">${title}</span>`);
    }

    // Journal/Container
    const journal = extractFromCSL(doc, 'container-title');
    if (journal) {
      citation.push(`<span>${journal}</span>`);
    }

    // Year
    const year = extractFromCSL(doc, 'year');
    if (year) {
      citation.push(`<span>(${year})</span>`);
    }

    // Join all parts with periods and spaces
    if (citation.length > 0) {
      html += `<div class="citation-text mb-3">${citation.join('. ') + '.'}</div>`;
    }
  }

  // Add action buttons
  html += `<div class="d-flex gap-2 align-items-center">`;

  // View Document button - store docIndex directly as a simple data attribute
  const docIndex = doc._docIndex !== undefined ? doc._docIndex : -1;
  html += `<button type="button" class="btn btn-outline-primary btn-sm view-document-btn" `;
  html += `data-doc-index="${docIndex}" `;
  html += `data-doc-id="${doc.id || 'unknown'}">`;
  html += `<i class="bi bi-file-text"></i> View Document Topics`;
  html += `</button>`;

  // Cite button
  html += `<button type="button" class="btn btn-outline-secondary btn-sm cite-document-btn" `;
  html += `data-doc-index="${docIndex}" `;
  html += `data-doc-id="${doc.id || 'unknown'}">`;
  html += `<i class="bi bi-quote"></i> Cite`;
  html += `</button>`;

  // Show document ID if available
  if (doc.id) {
    html += `<small class="text-muted ms-2">ID: ${doc.id}</small>`;
  }

  html += `</div>`;
  html += `</div>`;
  return html;
}

// Main export function
export async function loadBibliography(topicKeys, docTopic, metadata) {
  const main = document.getElementById('main-view');

  // Load configuration if not already loaded
  if (!appConfig) {
    await loadConfig();
  }

  // Load bibliography data if not already loaded
  if (!bibliographyData) {
    // Show loading indicator
    main.innerHTML = `
      <div class="card">
        <div class="card-body text-center">
          <h3>Bibliography</h3>
          <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
          </div>
          <p class="mt-2">Loading bibliography data...</p>
        </div>
      </div>`;

    await loadBibliographyData();
  }

  const currentLang = getCurrentLanguage();
  const langConfig = appConfig.language.configs[currentLang];
  const locale = langConfig.locale;

  let html = `<div class="card"><div class="card-body">`;
  html += `<h3>Bibliography</h3>`;

  // Add cache status info
  try {
    const cacheInfo = await getBibliographyCacheInfo();
    if (cacheInfo.cached) {
      const statusClass = cacheInfo.expired ? 'text-warning' : 'text-success';
      const statusText = cacheInfo.expired ? 'expired cache' : 'cached';
      html += `<div class="alert alert-info alert-dismissible fade show mb-3" role="alert">
        <small>
          <i class="bi bi-database"></i>
          Loaded ${cacheInfo.entries.toLocaleString()} entries from ${statusText}
          (cached ${cacheInfo.ageMinutes} minutes ago, ~${Math.round(cacheInfo.sizeEstimate / 1024 / 1024 * 10) / 10}MB)
        </small>
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
      </div>`;
    }
  } catch (error) {
    console.warn('Could not get cache info:', error);
  }

  // Add status info if bibliography was generated from metadata fallback
  if (bibliographyData && bibliographyData._source === 'metadata-fallback') {
    const fallbackAge = Math.round((Date.now() - bibliographyData._sourceTimestamp) / (1000 * 60));
    html += `<div class="alert alert-warning alert-dismissible fade show mb-3" role="alert">
      <small>
        <i class="bi bi-exclamation-triangle"></i>
        Bibliography generated from metadata.csv fallback (${fallbackAge} minutes ago)
        - bibliography.json was not available
      </small>
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    </div>`;
  }

  // Add navbar with sorting form
  html += `<nav class="navbar navbar-expand-lg navbar-light bg-light mb-3">`;
  html += `<div class="container-fluid">`;
  html += `<form class="d-flex align-items-center flex-wrap gap-2">`;
  html += `<label for="sort-by" class="form-label me-2 mb-0">Sort:</label>`;
  html += `<select id="sort-by" class="form-select me-2" style="width: auto;">`;
  html += `<option value="year-author" selected>Year → Author</option>`;
  html += `<option value="author-year">Author → Year</option>`;
  html += `<option value="title">Title</option>`;
  html += `<option value="journal">Journal</option>`;
  html += `<option value="custom">Custom Sorting...</option>`;
  html += `</select>`;
  html += `<select id="sort-order" class="form-select me-2" style="width: auto;">`;
  html += `<option value="asc" selected>in ascending order</option>`;
  html += `<option value="desc">in descending order</option>`;
  html += `</select>`;
  html += `<button type="button" class="btn btn-outline-primary btn-sm" id="custom-sort-btn" data-bs-toggle="modal" data-bs-target="#customSortModal">`;
  html += `<i class="bi bi-sliders"></i> Advanced Sort`;
  html += `</button>`;
  html += `</form>`;
  html += `</div>`;
  html += `</nav>`;

  html += `<div class="row">`;
  html += `<div class="col-md-2">`;
  html += `<div id="navigation-menu">`;

  // Create navigation menu
  if (bibliographyData && bibliographyData.length > 0) {
    html += generateNavigationHTML('year-author', bibliographyData, langConfig);
  }

  html += `</div>`;
  html += `</div>`;
  html += `<div class="col-md-10">`;
  html += `<div id="bibliography-content">`;

  // Generate initial bibliography content
  if (bibliographyData && bibliographyData.length > 0) {
    const sortBy = 'year-author';
    const sortOrder = 'asc';
    const sortedDocs = sortDocuments(bibliographyData, sortBy, sortOrder, locale);
    html += generateBibliographyHTML(sortedDocs, 'year', langConfig);
  } else {
    html += `<p class="text-muted">No documents available for bibliography.</p>`;
  }

  html += `</div>`;
  html += `</div>`;
  html += `</div>`;
  html += `</div></div>`;

  // Add custom sorting modal
  if (bibliographyData && bibliographyData.length > 0) {
    const availableFields = getAvailableCSLFields(bibliographyData);
    html += generateCustomSortingModal(availableFields);
  }

  main.innerHTML = html;

  // Add event listeners
  const sortBySelect = document.getElementById('sort-by');
  const sortOrderSelect = document.getElementById('sort-order');

  // Function to refresh bibliography content
  window.refreshBibliography = function() {
    const sortBy = sortBySelect.value;
    const sortOrder = sortOrderSelect.value;

    const sortedDocs = sortDocuments(bibliographyData, sortBy, sortOrder, locale);

    // Determine grouping type
    let groupingType = 'none';
    if (sortBy === 'year-author') {
      groupingType = 'year';
    } else if (sortBy === 'author-year' || sortBy === 'title') {
      groupingType = 'alphabet';
    } else if (sortBy === 'custom') {
      const customFields = sortBySelect?.customFields || [];
      if (customFields.length > 0) {
        const firstField = customFields[0];
        if (firstField === 'year' || isNumericField(firstField) || isDateField(firstField)) {
          groupingType = 'year';
        } else {
          groupingType = 'alphabet';
        }
      }
    }

    // Update bibliography content
    const bibliographyContent = document.getElementById('bibliography-content');
    bibliographyContent.innerHTML = generateBibliographyHTML(sortedDocs, groupingType, langConfig);

    // Update navigation menu
    const navigationMenu = document.getElementById('navigation-menu');
    navigationMenu.innerHTML = generateNavigationHTML(sortBy, bibliographyData, langConfig);

    // Attach navigation event listeners
    attachNavigationListeners(sortBy);
  };

  // Function to attach navigation event listeners
  function attachNavigationListeners(sortBy) {
    const isAlphabetSort = sortBy === 'author-year' || sortBy === 'title';

    if (isAlphabetSort) {
      // Alphabet navigation
      const letterLinks = main.querySelectorAll('[data-letter]');
      letterLinks.forEach(link => {
        link.addEventListener('click', (e) => {
          e.preventDefault();
          const letter = link.getAttribute('data-letter');
          const target = document.getElementById(`letter-${letter}`);
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        });
      });
    } else {
      // Year navigation
      const yearLinks = main.querySelectorAll('[data-year]');
      yearLinks.forEach(link => {
        link.addEventListener('click', (e) => {
          e.preventDefault();
          const year = link.getAttribute('data-year');
          const target = document.getElementById(`year-${year}`);
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        });
      });
    }
  }

  // Add sort change listeners
  if (sortBySelect) {
    sortBySelect.addEventListener('change', () => {
      if (sortBySelect.value === 'custom') {
        // Show custom sorting modal
        const modal = new bootstrap.Modal(document.getElementById('customSortModal'));
        modal.show();
      } else {
        window.refreshBibliography();
      }
    });
  }

  if (sortOrderSelect) {
    sortOrderSelect.addEventListener('change', () => {
      window.refreshBibliography();
    });
  }

  // Setup custom sorting modal
  setupCustomSortingModal();

  // Attach initial navigation listeners
  attachNavigationListeners('year-author');

  // Add event listeners for "View Document" buttons
  document.addEventListener('click', function(e) {
    if (e.target.closest('.view-document-btn')) {
      e.preventDefault();
      const button = e.target.closest('.view-document-btn');
      const docIndex = parseInt(button.getAttribute('data-doc-index'));
      const docId = button.getAttribute('data-doc-id');

      if (docIndex !== -1 && !isNaN(docIndex) && docIndex >= 0) {
        // Navigate to document view
        window.page(`/document/${docIndex}`);
      } else {
        console.warn('[Bibliography] Invalid document index:', docIndex, 'DocId:', docId);
        alert('Document not found in the current dataset.');
      }
    }

    // Handle "Cite" button clicks
    if (e.target.closest('.cite-document-btn')) {
      e.preventDefault();
      const button = e.target.closest('.cite-document-btn');
      const docIndex = parseInt(button.getAttribute('data-doc-index'));
      const docId = button.getAttribute('data-doc-id');

      if (docIndex !== -1 && !isNaN(docIndex) && docIndex >= 0) {
        // Navigate to citation view
        window.page(`/citation/${docIndex}`);
      } else {
        console.warn('[Bibliography] Invalid document index:', docIndex, 'DocId:', docId);
        alert('Document not found in the current dataset.');
      }
    }
  });

  // Expose cache management functions to console
  window.bibliographyCache = {
    info: getBibliographyCacheInfo,
    clear: clearBibliographyCache,
    reload: async function() {
      await clearBibliographyCache();
      bibliographyData = null;
      await loadBibliography(topicKeys, docTopic, metadata);
    }
  };
}