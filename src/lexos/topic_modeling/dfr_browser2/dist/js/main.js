// Main entry point for DFR Browser 2 with page.js routing
import { loadOverview } from './overview.js';
import { loadTopicView } from './topic.js';
import { loadDocumentView } from './document.js';
import { loadWordView } from './word.js';
import { loadBibliography } from './bibliography.js';
import { loadWordList } from './wordlist.js';
import { loadAboutView } from './about.js';
import { loadDiagnosticsView } from './diagnostics.js';
import { renderCitationView } from './citation.js';
import CacheManager from './cache-manager.js';
import CachedDataLoader from './cached-data-loader.js';
import ErrorHandler from './error-handler.js';
import DataValidator from './data-validator.js';
import { getMetadataValue, validateTopicConfig } from './topic-config.js';

// Make error handler globally available
window.ErrorHandler = ErrorHandler;
window.DataValidator = DataValidator;

// Global state to store loaded data
window.dfrState = {
  topicKeys: null,
  docTopic: null,
  metadata: null,
  topicCoords: null,
  docTopicCounts: null,
  docLengths: {},
  config: null,
  dataLoaded: false // Track if data has been loaded
};

// Utility: Parse topic-keys.txt (flexible)
function parseTopicKeys(text) {
  const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
  // Skip header if present (detect if first line doesn't start with a number)
  const startIdx = /^\d/.test(lines[0]) ? 0 : 1;
  return lines.slice(startIdx).map(line => {
    const parts = line.split(/\t|\s{2,}/); // tab or multiple spaces
    const topicNum = parts[0];
    const weight = parts[1];
    const words = parts.slice(2).join(' ').split(/\s+/);
    return {
      topic: Number(topicNum),
      weight: Number(weight),
      words: words.filter(Boolean)
    };
  });
}

// Utility: Parse doc-topics.txt (flexible)
function parseDocTopics(text) {
  const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
  // Skip header if present (detect if first line doesn't start with a number)
  const startIdx = /^\d/.test(lines[0]) ? 0 : 1;
  return lines.slice(startIdx).map(line => {
    const parts = line.split(/\t|\s+/); // tab or whitespace
    // MALLET format: first two columns are docNum and docName, rest are topic proportions
    return parts.slice(2).map(Number);
  });
}

// Utility: Parse metadata.csv (properly handles quoted fields with commas)
function parseMetadata(text) {
  const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
  if (lines.length < 2) return [];

  // Parse CSV line with proper quote handling
  function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      const nextChar = line[i + 1];

      if (char === '"') {
        if (inQuotes && nextChar === '"') {
          // Escaped quote
          current += '"';
          i++; // Skip next quote
        } else {
          // Toggle quote state
          inQuotes = !inQuotes;
        }
      } else if (char === ',' && !inQuotes) {
        // End of field
        result.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }

    // Add last field
    result.push(current.trim());
    return result;
  }

  const headers = parseCSVLine(lines[0]);
  return lines.slice(1).map(line => {
    const values = parseCSVLine(line);
    const obj = {};
    headers.forEach((h, i) => obj[h] = values[i]);
    return obj;
  });
}

// Rewrite data-route nav links to use correct base path
function updateNavLinks() {
  const base = window.dfrBasePath || '';
  document.querySelectorAll('[data-route]').forEach(el => {
    el.href = base + el.dataset.route;
  });
}

// Initialize the application
async function init() {
  console.log('[DFR] Initializing application...');
  console.log('[DFR] page.js available:', typeof window.page);
    // Accessibility: Add event listener for Settings link to open modal
    const settingsLink = document.getElementById('settings-link');
    if (settingsLink) {
      settingsLink.addEventListener('click', function(e) {
        e.preventDefault();
        // If using page.js for routing, trigger settings route
        if (window.page) {
          window.page('/settings');
        } else {
          // Fallback: Show modal directly if available
          const settingsModal = document.getElementById('settings-modal');
          if (settingsModal) {
            settingsModal.style.display = 'block';
            settingsModal.setAttribute('aria-modal', 'true');
            settingsModal.setAttribute('role', 'dialog');
            settingsModal.focus();
          }
        }
      });
    }

  // Load configuration with error handling
  try {
    const response = await fetch('config.json');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const configText = await response.text();
    try {
      window.dfrState.config = JSON.parse(configText);
      console.log('[DFR] Config loaded:', window.dfrState.config);
    } catch (parseError) {
      ErrorHandler.handleConfigError(
        'Invalid JSON in config.json',
        parseError.message
      );
      throw parseError;
    }

    // Apply branding
    applyBranding(window.dfrState.config);

    // Validate topic configuration
    const topicValidation = validateTopicConfig();
    if (topicValidation.warnings.length > 0) {
      console.warn('[DFR] Topic configuration warnings:', topicValidation.warnings);
    }
  } catch (error) {
    if (error.message.includes('404')) {
      ErrorHandler.handleConfigError(
        'Configuration file not found',
        'Could not load config.json - using default configuration'
      );
    } else if (!error.message.includes('JSON')) {
      ErrorHandler.handleConfigError(
        'Error loading configuration',
        error.message
      );
    }

    // Use minimal default config
    window.dfrState.config = {
      brand: { name: 'DFR Browser 2', url: '#' },
      topic_keys_file: 'sample_data/topic-keys.txt',
      doc_topic_file: 'sample_data/doc-topic.txt',
      metadata_file: 'sample_data/metadata.csv',
      topic_coords_file: 'sample_data/topic_coords.csv',
      topic_state_file: 'sample_data/topic-state.gz'
    };
    console.warn('[DFR] Using default config');
  }

  // Setup page.js routes
  console.log('[DFR] Setting up routes...');
  setupRoutes();
  console.log('[DFR] Routes set up');

  // Configure router WITHOUT hashbang mode (use HTML5 history)
  console.log('[DFR] Configuring router with HTML5 history mode...');
  window.page.base(window.dfrBasePath || '');

  // Rewrite nav hrefs to use the correct base path
  updateNavLinks();



  // Setup file upload handler
  setupFileUploadHandler();

  // Check if we should show upload form or auto-load data
  await showUploadFormOrAutoLoad();

  // Start router with click interception AFTER data is loaded
  // This ensures the initial route dispatch happens with loaded data
  window.page({ click: true, popstate: true, dispatch: true });

  console.log('[DFR] Router started');
}

// Apply branding from config
function applyBranding(config) {
  if (config && config.brand) {
    const brand = config.brand;
    const brandLink = document.getElementById('brand-link');

    if (brand.name) {
      brandLink.textContent = brand.name;
      document.title = brand.name;
    }

    if (brand.url && brand.url !== '#') {
      brandLink.href = brand.url;
      brandLink.target = '_blank';
    }

    if (brand.logo) {
      brandLink.innerHTML = `<img src="${brand.logo}" alt="${brand.name || 'Logo'}" height="30" class="me-2">${brand.name || ''}`;
    }
  }

  // Update favicon if specified
  const faviconPath = config?.brand?.favicon || config?.favicon;
  if (faviconPath) {
    let favicon = document.getElementById('favicon');
    if (!favicon) {
      favicon = document.createElement('link');
      favicon.id = 'favicon';
      favicon.rel = 'icon';
      favicon.type = 'image/png';
      document.head.appendChild(favicon);
    }
    favicon.href = faviconPath;
  }
}

// Make settings globally accessible
window.getSettings = async function() {
  // Load default settings from config
  let defaultSettings = {
    showHiddenTopics: false,
    wordsInLists: 15,
    wordsInTopic: 50,
    wordsInCircles: 6,
    topicDocs: 20
  };

  try {
    // Try to load defaults from config.json
    if (window.dfrState.config && window.dfrState.config.settings && window.dfrState.config.settings.defaults) {
      defaultSettings = { ...defaultSettings, ...window.dfrState.config.settings.defaults };
    }
  } catch (error) {
    console.warn('Could not load config settings, using hardcoded defaults');
  }

  // Get user settings from localStorage or use defaults
  const savedSettings = localStorage.getItem('dfrBrowserSettings');
  if (savedSettings) {
    try {
      const userSettings = JSON.parse(savedSettings);
      return { ...defaultSettings, ...userSettings };
    } catch (error) {
      console.error('Error parsing saved settings:', error);
    }
  }

  return defaultSettings;
};

// Cache management functions
window.clearCache = async function() {
  try {
    await CacheManager.clearAll();
    console.log('[DFR] All caches cleared');
    return true;
  } catch (err) {
    console.error('[DFR] Error clearing cache:', err);
    return false;
  }
};

window.getCacheStats = async function() {
  try {
    const stats = await CacheManager.getStats();
    console.log('[DFR] Cache statistics:', stats);
    return stats;
  } catch (err) {
    console.error('[DFR] Error getting cache stats:', err);
    return null;
  }
};

window.pruneCache = async function(maxAgeDays = 7) {
  try {
    const maxAge = maxAgeDays * 24 * 60 * 60 * 1000;
    const deleted = await CacheManager.pruneOldEntries(maxAge);
    console.log(`[DFR] Pruned ${deleted} old cache entries`);
    return deleted;
  } catch (err) {
    console.error('[DFR] Error pruning cache:', err);
    return 0;
  }
};

// Function to ensure data is loaded before accessing routes
async function ensureDataLoaded() {
  if (window.dfrState.dataLoaded) {
    return true; // Data already loaded
  }

  console.log('[DFR] Data not loaded, attempting auto-load...');

  // Try to auto-load data
  if (!window.dfrState.config) {
    // Load config first if not already loaded
    try {
      const response = await fetch('config.json');
      window.dfrState.config = await response.json();
      console.log('[DFR] Config loaded for auto-load');
    } catch (error) {
      console.error('[DFR] Could not load config:', error);
      alert('Error: Could not load configuration file. Please check that config.json exists.');
      return false;
    }
  }

  // Attempt to auto-load data
  try {
    await autoLoadData();
    return true;
  } catch (error) {
    console.error('[DFR] Auto-load failed:', error);
    alert('Sample data not loaded: ' + error.message + '\n\nPlease go to the home page to load data.');
    window.page('/');
    return false;
  }
}

// Function to populate the Topic dropdown menu
async function populateTopicDropdown(topicKeys) {
  const dropdownMenu = document.getElementById('topicDropdownMenu');
  if (!dropdownMenu) return;

  // Clear existing items
  dropdownMenu.innerHTML = '';

  if (!topicKeys || topicKeys.length === 0) {
    dropdownMenu.innerHTML = '<li><a class="dropdown-item" href="#">No topics loaded</a></li>';
    return;
  }

  // Get settings to determine how many words to show and whether to show hidden topics
  const settings = await window.getSettings();
  const wordsToShow = settings.wordsInLists || 15;
  const showHidden = settings.showHiddenTopics;

  // Get hidden topic indices from config
  const config = window.dfrState.config || {};
  const hiddenTopics = (config.topics && config.topics.hidden) ? config.topics.hidden : [];

  // Add menu items for each topic, filtering hidden topics unless showHidden is true
  topicKeys.forEach((tk, index) => {
    const topicNum = index + 1; // 1-based numbering
    const isHidden = hiddenTopics.includes(topicNum);
    if (isHidden && !showHidden) return;
    const topWords = tk.words.slice(0, wordsToShow).join(' ');
    const menuText = `Topic ${topicNum}: ${topWords}`;

    const li = document.createElement('li');
    const a = document.createElement('a');
    a.className = 'dropdown-item';
    a.href = (window.dfrBasePath || '') + `/topic/${topicNum}`; // Use 1-based numbering in URL
    a.textContent = menuText;

    li.appendChild(a);
    dropdownMenu.appendChild(li);
  });
// Helper to get visible topic indices for all views
function getVisibleTopicIndices() {
  const config = window.dfrState.config || {};
  const hiddenTopics = (config.topics && config.topics.hidden) ? config.topics.hidden : [];
  const settings = JSON.parse(localStorage.getItem('dfrBrowserSettings') || '{}');
  const showHidden = settings.showHiddenTopics;
  const topicKeys = window.dfrState.topicKeys || [];
  return topicKeys.map((tk, idx) => idx)
    .filter(idx => {
      const topicNum = idx + 1;
      return showHidden || !hiddenTopics.includes(topicNum);
    });
}
}

// Helper function to ensure paths work on any sub-path deployment
function ensureAbsolutePath(path) {
  if (!path) return path;
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  // Legacy absolute paths get the base prepended; relative paths resolve via <base> tag
  if (path.startsWith('/')) return (window.dfrBasePath || '') + path;
  return path;
}

// Auto-load files from config paths
async function autoLoadData() {
  try {
    const config = window.dfrState.config;
    const topicKeysPath = ensureAbsolutePath(config.topic_keys_file || 'data/topic-keys.txt');
    const docTopicPath = ensureAbsolutePath(config.doc_topic_file || 'data/doc-topic.txt');
    const metadataPath = ensureAbsolutePath(config.metadata_file || 'data/metadata.csv');
    const coordsPath = ensureAbsolutePath(config.topic_coords_file || 'data/topic_coords.csv');

    console.log('[DFR] Loading data files with caching...');

    // Load files using cached data loader with error handling
    let tkRes, dtRes, mdRes, coordsRes;
    try {
      // Load with caching - pass the parser functions
      const results = await Promise.all([
        CachedDataLoader.loadTopicKeys(topicKeysPath, parseTopicKeys).catch(err => {
          ErrorHandler.handleFileError(topicKeysPath, err, 'topic keys file');
          throw err;
        }),
        CachedDataLoader.loadDocTopics(docTopicPath, parseDocTopics).catch(err => {
          ErrorHandler.handleFileError(docTopicPath, err, 'doc-topics file');
          throw err;
        }),
        CachedDataLoader.loadMetadata(metadataPath, parseMetadata).catch(err => {
          ErrorHandler.handleFileError(metadataPath, err, 'metadata file');
          throw err;
        }),
        fetch(coordsPath).then(r => r.ok ? r.text() : null).catch(() => null)
      ]);

      // Assign results
      window.dfrState.topicKeys = results[0];
      window.dfrState.docTopic = results[1];
      window.dfrState.metadata = results[2];
      coordsRes = results[3];

      // Validate individual files
      console.log('[DFR] Validating loaded data...');
      const validation = await DataValidator.validateAll(
        window.dfrState.topicKeys,
        window.dfrState.docTopic,
        window.dfrState.metadata
      );

      // Check if validation passed
      if (!validation.topicKeys.valid || !validation.docTopics.valid ||
          !validation.metadata.valid || !validation.integrity.valid) {
        console.error('[DFR] Data validation failed');
        // Errors already shown by validator
        return;
      }

      // Show data quality report
      const qualityReport = DataValidator.getDataQualityReport(
        window.dfrState.topicKeys,
        window.dfrState.docTopic,
        window.dfrState.metadata
      );
      console.log('[DFR] Data Quality Report:', qualityReport);

    } catch (err) {
      console.error('[DFR] Error loading data files:', err);
      return;
    }

    // Derive docLengths and docTopicCounts from state file if available
    try {
      const stateUtils = await import('./state-utils.js');
      window.dfrState.docLengths = await stateUtils.extractDocLengths();
      window.dfrState.docTopicCounts = await stateUtils.extractDocTopicCounts(window.dfrState.topicKeys.length);
      console.log('[DFR] Successfully extracted data from state file');
    } catch (e) {
      console.warn('[DFR] Warning: Could not extract doc lengths or topic counts from state file. Some features may be missing.');
    }

    // Parse topic coordinates (optional)
    window.dfrState.topicCoords = null;
    if (coordsRes) {
      try {
        // Parse CSV: topic,x,y (header optional)
        const lines = coordsRes.trim().split(/\r?\n/).filter(l => l.trim());
        const startIdx = /^topic/i.test(lines[0]) ? 1 : 0;
        window.dfrState.topicCoords = lines.slice(startIdx).map(line => {
          const [topic, x, y] = line.split(/,|\t/).map(s => s.trim());
          return { topic: Number(topic), x: Number(x), y: Number(y) };
        });
        console.log('[DFR] Topic coordinates loaded successfully');
      } catch (e) {
        console.warn('[DFR] Warning: Could not parse topic_coords.csv. Default topic layout will be used.');
      }
    } else {
      console.warn('[DFR] Warning: topic_coords.csv not found. Default topic layout will be used.');
    }

    await populateTopicDropdown(window.dfrState.topicKeys);

    console.log('[DFR] Data loaded successfully');
    console.log('[DFR] Topics:', window.dfrState.topicKeys?.length);
    console.log('[DFR] Documents:', window.dfrState.docTopic?.length);
    console.log('[DFR] Metadata:', window.dfrState.metadata?.length);

    // Mark data as loaded
    window.dfrState.dataLoaded = true;

    // Show footer after data is loaded
    const footer = document.getElementById('app-footer');
    if (footer) {
      footer.style.display = 'block';
    }
  } catch (e) {
    console.error('[DFR] Sample data not loaded:', e);
    throw e; // Re-throw so ensureDataLoaded can catch it
  }
}

// Show upload form or auto-load data from config paths
async function showUploadFormOrAutoLoad() {
  try {
    const config = window.dfrState.config;
    if (config && config.show_upload_form !== false) {
      // Show the upload form as a modal overlay
      const form = document.getElementById('data-upload-form');
      if (form) {
        form.style.display = 'block';
        form.classList.add('upload-modal');
      }
    } else {
      // Hide the form if not needed and auto-load data from config paths
      const form = document.getElementById('data-upload-form');
      if (form) form.style.display = 'none';
      await autoLoadData();
    }
  } catch (e) {
    console.warn('[DFR] Could not check show_upload_form:', e);
  }
}

// Handle file uploads
function setupFileUploadHandler() {
  const form = document.getElementById('data-upload-form');
  if (!form) return;

  form.addEventListener('submit', async function(e) {
    e.preventDefault();
    const topicKeysFile = document.getElementById('topic-keys').files[0];
    const docTopicFile = document.getElementById('doc-topic').files[0];
    const metadataFile = document.getElementById('metadata').files[0];
    const coordsFile = document.getElementById('topic-coords').files[0];

    // Helper for error display
    function showError(msg) {
      console.error('[DFR] ' + msg);
      alert(msg);
    }
    function showWarning(msg) {
      console.warn('[DFR] ' + msg);
    }

    // All data loaded here is from the files selected in the upload form
    const readerPromises = [];
    readerPromises.push(topicKeysFile ? topicKeysFile.text() : Promise.reject('Topic keys file is required.'));
    readerPromises.push(docTopicFile ? docTopicFile.text() : Promise.reject('Doc-topic file is required.'));
    readerPromises.push(metadataFile ? metadataFile.text() : Promise.reject('Metadata file is required.'));
    readerPromises.push(coordsFile ? coordsFile.text() : null);

    try {
      const [tkText, dtText, mdText, coordsText] = await Promise.all(readerPromises);

      try {
        window.dfrState.topicKeys = parseTopicKeys(tkText);
        if (!window.dfrState.topicKeys.length) throw new Error('No topics found in topic keys file.');
      } catch (e) {
        showError('Failed to parse topic keys: ' + e.message);
        return;
      }
      try {
        window.dfrState.docTopic = parseDocTopics(dtText);
        if (!window.dfrState.docTopic.length) throw new Error('No doc-topics found in doc-topic file.');
      } catch (e) {
        showError('Failed to parse doc-topic file: ' + e.message);
        return;
      }
      try {
        window.dfrState.metadata = parseMetadata(mdText);
        if (!window.dfrState.metadata.length) throw new Error('No metadata found in metadata file.');
      } catch (e) {
        showError('Failed to parse metadata: ' + e.message);
        return;
      }

      // Derive docLengths and docTopicCounts from state file if available
      try {
        const stateUtils = await import('./state-utils.js');
        window.dfrState.docLengths = await stateUtils.extractDocLengths();
        window.dfrState.docTopicCounts = await stateUtils.extractDocTopicCounts(window.dfrState.topicKeys.length);
      } catch (e) {
        showWarning('Warning: Could not extract doc lengths or topic counts from state file. Some features may be missing.');
      }

      window.dfrState.topicCoords = null;
      if (coordsText) {
        try {
          // Parse CSV: topic,x,y (header optional)
          const lines = coordsText.trim().split(/\r?\n/).filter(l => l.trim());
          const startIdx = /^topic/i.test(lines[0]) ? 1 : 0;
          window.dfrState.topicCoords = lines.slice(startIdx).map(line => {
            const [topic, x, y] = line.split(/,|\t/).map(s => s.trim());
            return { topic: Number(topic), x: Number(x), y: Number(y) };
          });
        } catch (e) {
          showWarning('Warning: Could not parse topic_coords.csv. Default topic layout will be used.');
        }
      } else {
        showWarning('Warning: topic_coords.csv not uploaded. Default topic layout will be used.');
      }

      await populateTopicDropdown(window.dfrState.topicKeys);

      // Mark data as loaded
      window.dfrState.dataLoaded = true;

      // Show footer after data is loaded
      const footer = document.getElementById('app-footer');
      if (footer) {
        footer.style.display = 'block';
      }

      // Load overview with the data
      window.page('/');

      // Hide the upload form modal after successful upload
      form.style.display = 'none';
      form.classList.remove('upload-modal');

      alert('Data uploaded from form files!');
    } catch (err) {
      showError('Error: ' + err);
    }
  });
}

// Initialize floating Back to Top button (used on all pages)
function initializeBackToTopButton() {
  // Remove existing button and listener if present
  const existingBtn = document.getElementById('floating-back-to-top');
  if (existingBtn) {
    existingBtn.remove();
  }
  if (window.backToTopScrollListener) {
    window.removeEventListener('scroll', window.backToTopScrollListener);
    window.backToTopScrollListener = null;
  }

  // Create the button
  const floatingTopButton = document.createElement('button');
  floatingTopButton.id = 'floating-back-to-top';
  floatingTopButton.className = 'btn btn-primary btn-lg';
  floatingTopButton.innerHTML = '<i class="bi bi-arrow-up-circle-fill" style="font-size: 1.8rem;"></i>';
  floatingTopButton.style.cssText = `
    position: fixed;
    bottom: 30px;
    right: 30px;
    z-index: 1000;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    opacity: 0;
    visibility: hidden;
    transform: translateY(10px);
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
  `;
  floatingTopButton.setAttribute('aria-label', 'Back to Top');
  floatingTopButton.title = 'Return to Top';

  document.body.appendChild(floatingTopButton);

  // Show/hide based on scroll position
  function toggleFloatingButton() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

    if (scrollTop > 250) {
      floatingTopButton.style.opacity = '1';
      floatingTopButton.style.visibility = 'visible';
      floatingTopButton.style.transform = 'translateY(0)';
    } else {
      floatingTopButton.style.opacity = '0';
      floatingTopButton.style.visibility = 'hidden';
      floatingTopButton.style.transform = 'translateY(10px)';
    }
  }

  window.backToTopScrollListener = toggleFloatingButton;
  window.addEventListener('scroll', window.backToTopScrollListener);

  // Click handler to scroll to top
  floatingTopButton.addEventListener('click', function(e) {
    e.preventDefault();
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });
}

// Initialize the application

// Setup all routes
function setupRoutes() {
  // Add global middleware to show spinner and scroll to top on every route change
  window.page('*', (ctx, next) => {
    // Show loading spinner immediately for all routes
    const main = document.getElementById('main-view');
    if (main) {
      main.innerHTML = `
        <div class="card">
          <div class="card-body text-center">
            <h3>Loading...</h3>
            <div class="spinner-border text-primary" role="status">
              <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Please wait...</p>
          </div>
        </div>`;
    }
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' });

    // Initialize back to top button for all routes
    initializeBackToTopButton();

    next();
  });

  // Overview route (home)
  window.page('/', async () => {
    console.log('[DFR] Route: Overview');
    if (!await ensureDataLoaded()) return;
    setActiveNav('/');
    loadOverview(
      window.dfrState.topicKeys,
      window.dfrState.docTopic,
      window.dfrState.metadata,
      window.dfrState.topicCoords,
      window.dfrState.docTopicCounts
    );
  });

  // Topic view route with parameter (1-based numbering in URL)
  window.page('/topic/:id', async (ctx) => {
    if (!await ensureDataLoaded()) return;
    const topicNumber = parseInt(ctx.params.id, 10); // 1-based from URL
    const topicId = topicNumber - 1; // Convert to 0-based index
    console.log('[DFR] Route: Topic', topicNumber, '(index:', topicId + ')');
    setActiveNav('/topic');
    loadTopicView(
      window.dfrState.topicKeys,
      window.dfrState.docTopic,
      window.dfrState.metadata,
      topicId,
      window.dfrState.docTopicCounts
    );
  });

  // Document view route with parameter
  window.page('/document/:docIndex?', async (ctx) => {
    if (!await ensureDataLoaded()) return;
    const docIndexParam = ctx.params.docIndex;
    console.log('[DFR] Route: Document', docIndexParam);
    setActiveNav('/document');

    // Build reference data from docIndex
    let referenceData = null;
    if (docIndexParam !== undefined && window.dfrState.metadata) {
      const docIndex = parseInt(docIndexParam);
      if (!isNaN(docIndex) && docIndex >= 0 && docIndex < window.dfrState.metadata.length) {
        const doc = window.dfrState.metadata[docIndex];
        const title = getMetadataValue(doc, 'title') || 'Untitled';
        const author = getMetadataValue(doc, 'author') || 'Unknown author';
        const year = getMetadataValue(doc, 'year') || getMetadataValue(doc, 'pubdate') || 'Unknown date';

        referenceData = {
          docIndex: docIndex,
          id: getMetadataValue(doc, 'id') || `doc_${docIndex}`,
          title: title,
          author: author,
          year: year,
          formattedCitation: getMetadataValue(doc, 'formatted-citation') ||
            `${title}. ${author}. ${year}.`
        };
      }
    }

    loadDocumentView(
      window.dfrState.topicKeys,
      window.dfrState.docTopic,
      window.dfrState.metadata,
      referenceData,
      window.dfrState.docLengths
    );
  });

  // Word view route with parameter
  window.page('/word/:word?', async (ctx) => {
    if (!await ensureDataLoaded()) return;
    const word = ctx.params.word;
    console.log('[DFR] Route: Word', word);
    setActiveNav('/word');
    await loadWordView(
      window.dfrState.topicKeys,
      window.dfrState.docTopic,
      window.dfrState.metadata,
      word
    );
  });

  // Bibliography route
  window.page('/bibliography', async () => {
    if (!await ensureDataLoaded()) return;
    console.log('[DFR] Route: Bibliography');
    setActiveNav('/bibliography');
    loadBibliography(
      window.dfrState.metadata
    );
  });

  // Citation route
  window.page('/citation/:id', async (ctx) => {
    if (!await ensureDataLoaded()) return;
    console.log('[DFR] Route: Citation view:', ctx.params.id);
    setActiveNav('/bibliography');
    const docIndex = parseInt(ctx.params.id);
    await renderCitationView(docIndex);
  });

  // Word list/index route
  window.page('/wordlist', async () => {
    if (!await ensureDataLoaded()) return;
    console.log('[DFR] Route: Word List');
    setActiveNav('/wordlist');
    loadWordList(
      window.dfrState.topicKeys,
      window.dfrState.docTopic,
      window.dfrState.metadata
    );
  });

  // About route
  window.page('/about', () => {
    console.log('[DFR] Route: About');
    setActiveNav('/about');
    loadAboutView();
  });

  // Diagnostics route
  window.page('/diagnostics', () => {
    console.log('[DFR] Route: Diagnostics');
    setActiveNav('/diagnostics');
    loadDiagnosticsView();
  });

  // Settings route (modal, not a page)
  window.page('/settings', (ctx) => {
    console.log('[DFR] Route: Settings');
    showSettingsModal();
    // Navigate back to home (settings is a modal, not a separate page)
    window.page('/');
  });

  // 404 fallback
  window.page('*', () => {
    console.log('[DFR] Route: 404');
    document.getElementById('main-view').innerHTML = `
      <div class="alert alert-warning">
        <h4>Page not found</h4>
        <p>The page you're looking for doesn't exist.</p>
        <a href="/" class="btn btn-primary">Go to Overview</a>
      </div>
    `;
  });
}

// Set active navigation link
function setActiveNav(path) {
  // Remove active class from all nav links
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.remove('active');
  });

  // Add active class to matching link
  const navLink = document.querySelector(`.nav-link[data-route="${path}"]`);
  if (navLink) {
    navLink.classList.add('active');
  }
}

// Settings modal placeholder
function showSettingsModal() {
  // Remove any existing modal
  const existingModal = document.getElementById('settingsModal');
  if (existingModal) {
    existingModal.remove();
  }

  // Create modal HTML with tabs for General Settings and Cache Management
  const modalHTML = `
    <div class="modal fade" id="settingsModal" tabindex="-1" role="dialog" aria-labelledby="settings_title" aria-modal="true" aria-hidden="true">
      <div class="modal-dialog modal-lg" role="document">
        <div class="modal-content">
          <div class="modal-header">
            <h4 class="modal-title" id="settings_title">Settings</h4>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close Settings Modal"></button>
          </div>
          <div class="modal-body">
            <!-- Tabs Navigation -->
            <ul class="nav nav-tabs" id="settingsTabs" role="tablist">
              <li class="nav-item" role="presentation">
                <button class="nav-link active" id="general-tab" data-bs-toggle="tab" data-bs-target="#general-panel" type="button" role="tab" aria-controls="general-panel" aria-selected="true">
                  <i class="bi bi-gear me-2"></i>General Settings
                </button>
              </li>
              <li class="nav-item" role="presentation">
                <button class="nav-link" id="cache-tab" data-bs-toggle="tab" data-bs-target="#cache-panel" type="button" role="tab" aria-controls="cache-panel" aria-selected="false">
                  <i class="bi bi-hdd-stack me-2"></i>Cache Management
                </button>
              </li>
            </ul>

            <!-- Tabs Content -->
            <div class="tab-content mt-3" id="settingsTabsContent">
              <!-- General Settings Tab -->
              <div class="tab-pane show active" id="general-panel" role="tabpanel" aria-labelledby="general-tab" style="display: block;">
                <div class="pt-3">
                  <p id="settings_desc" class="help mb-3">Use these controls to adjust how much information is displayed on some of the browser pages.</p>
                  <form role="form" aria-label="Settings Form">
                    <fieldset>
                      <legend class="visually-hidden">Topic Display Settings</legend>
                      <div id="reveal_hidden" class="mb-3" role="group" aria-labelledby="showHiddenTopicsLabel">
                        <div class="form-check">
                          <input class="form-check-input" type="checkbox" id="showHiddenTopics" aria-labelledby="showHiddenTopicsLabel" aria-checked="false">
                          <label class="form-check-label" for="showHiddenTopics" id="showHiddenTopicsLabel">Show hidden topics</label>
                        </div>
                      </div>
                      <div id="n_words_list" class="mb-3" role="group" aria-labelledby="wordsInListsLabel">
                        <div class="row align-items-center">
                          <div class="col-4">
                            <input type="number" class="form-control" min="1" max="50" value="15" id="wordsInLists" aria-labelledby="wordsInListsLabel" aria-valuemin="1" aria-valuemax="50">
                          </div>
                          <div class="col-8">
                            <label for="wordsInLists" class="form-label mb-0" id="wordsInListsLabel">Topic top words in lists</label>
                          </div>
                        </div>
                      </div>
                      <div id="n_words_topic" class="mb-3" role="group" aria-labelledby="wordsInTopicLabel">
                        <div class="row align-items-center">
                          <div class="col-4">
                            <input type="number" class="form-control" min="1" max="100" value="50" id="wordsInTopic" aria-labelledby="wordsInTopicLabel" aria-valuemin="1" aria-valuemax="100">
                          </div>
                          <div class="col-8">
                            <label for="wordsInTopic" class="form-label mb-0" id="wordsInTopicLabel">Topic top words on the Topic and Word pages</label>
                          </div>
                        </div>
                      </div>
                      <div id="n_words_in_circles" class="mb-3" role="group" aria-labelledby="wordsInCirclesLabel">
                        <div class="row align-items-center">
                          <div class="col-4">
                            <input type="number" class="form-control" min="1" max="20" value="6" id="wordsInCircles" aria-labelledby="wordsInCirclesLabel" aria-valuemin="1" aria-valuemax="20">
                          </div>
                          <div class="col-8">
                            <label for="wordsInCircles" class="form-label mb-0" id="wordsInCirclesLabel">Words displayed in Overview circles</label>
                          </div>
                        </div>
                      </div>
                      <div id="n_topic_docs" class="mb-3" role="group" aria-labelledby="topicDocsLabel">
                        <div class="row align-items-center">
                          <div class="col-4">
                            <input type="number" class="form-control" min="1" max="5605" value="20" id="topicDocs" aria-labelledby="topicDocsLabel" aria-valuemin="1" aria-valuemax="5605">
                          </div>
                          <div class="col-8">
                            <label for="topicDocs" class="form-label mb-0" id="topicDocsLabel">Top articles on the topic page</label>
                          </div>
                        </div>
                      </div>
                    </fieldset>

                    <!-- Action buttons for General Settings -->
                    <div class="mt-4 d-flex gap-2 justify-content-end">
                      <button type="button" class="btn btn-secondary" data-bs-dismiss="modal" aria-label="Close Settings Modal">Close</button>
                      <button type="button" class="btn btn-primary" onclick="saveSettings()" aria-label="Save Settings">Save Settings</button>
                    </div>
                  </form>
                </div>
              </div>

              <!-- Cache Management Tab -->
              <div class="tab-pane" id="cache-panel" role="tabpanel" aria-labelledby="cache-tab" style="display: none;">
                <div class="pt-3">
                  <p class="text-muted small mb-3" id="cacheManagementDesc">Caching improves performance by storing parsed data locally. Clear caches if you've updated your data files.</p>

                  <!-- Cache Statistics Card -->
                  <div class="card mb-3">
                    <div class="card-header bg-light">
                      <h6 class="mb-0"><i class="bi bi-bar-chart me-2"></i>Cache Statistics</h6>
                    </div>
                    <div class="card-body">
                      <div id="cache-stats">
                        <p class="text-muted small mb-0">Loading cache statistics...</p>
                      </div>
                    </div>
                  </div>

                  <!-- Quick Actions -->
                  <div class="mb-3">
                    <h6 class="mb-2"><i class="bi bi-lightning me-2"></i>Quick Actions</h6>
                    <div class="row g-2">
                      <div class="col-4">
                        <button type="button" class="btn btn-outline-primary btn-sm w-100" onclick="refreshCacheStats()" aria-label="Refresh Cache Statistics">
                          <i class="bi bi-arrow-clockwise me-1"></i>Refresh
                        </button>
                      </div>
                      <div class="col-4">
                        <button type="button" class="btn btn-outline-warning btn-sm w-100" onclick="pruneCacheEntries()" aria-label="Remove Old Cache Entries">
                          <i class="bi bi-calendar-x me-1"></i>Prune
                        </button>
                      </div>
                      <div class="col-4">
                        <button type="button" class="btn btn-warning btn-sm w-100" onclick="clearAllCaches()" aria-label="Clear All Caches">
                          <i class="bi bi-trash me-1"></i>Clear All
                        </button>
                      </div>
                    </div>
                  </div>

                  <!-- Individual Cache Clear -->
                  <div class="mb-3">
                    <h6 class="mb-2"><i class="bi bi-boxes me-2"></i>Clear Individual Caches</h6>
                    <div class="row g-2">
                      <div class="col-4">
                        <button type="button" class="btn btn-sm btn-outline-secondary w-100" onclick="clearBibliographyCache()" aria-label="Clear Bibliography Cache">
                          <i class="bi bi-book me-1"></i>Bibliography
                        </button>
                      </div>
                      <div class="col-4">
                        <button type="button" class="btn btn-sm btn-outline-secondary w-100" onclick="clearMetadataCache()" aria-label="Clear Metadata Cache">
                          <i class="bi bi-file-text me-1"></i>Metadata
                        </button>
                      </div>
                      <div class="col-4">
                        <button type="button" class="btn btn-sm btn-outline-secondary w-100" onclick="clearTopicsCache()" aria-label="Clear Topics Cache">
                          <i class="bi bi-grid-3x3 me-1"></i>Topics
                        </button>
                      </div>
                      <div class="col-4">
                        <button type="button" class="btn btn-sm btn-outline-secondary w-100" onclick="clearVocabularyCache()" aria-label="Clear Vocabulary Cache">
                          <i class="bi bi-alphabet me-1"></i>Vocabulary
                        </button>
                      </div>
                      <div class="col-4">
                        <button type="button" class="btn btn-sm btn-outline-secondary w-100" onclick="clearStateCache()" aria-label="Clear State Cache">
                          <i class="bi bi-database me-1"></i>State
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Add modal to body
  document.body.insertAdjacentHTML('beforeend', modalHTML);

  // Initialize tab switching manually
  const generalTab = document.getElementById('general-tab');
  const cacheTab = document.getElementById('cache-tab');
  const generalPanel = document.getElementById('general-panel');
  const cachePanel = document.getElementById('cache-panel');

  generalTab.addEventListener('click', function(e) {
    e.preventDefault();
    // Show general, hide cache
    generalPanel.style.display = 'block';
    cachePanel.style.display = 'none';
    generalTab.classList.add('active');
    cacheTab.classList.remove('active');
    generalPanel.classList.add('show', 'active');
    cachePanel.classList.remove('show', 'active');
  });

  cacheTab.addEventListener('click', function(e) {
    e.preventDefault();
    // Show cache, hide general
    generalPanel.style.display = 'none';
    cachePanel.style.display = 'block';
    cacheTab.classList.add('active');
    generalTab.classList.remove('active');
    cachePanel.classList.add('show', 'active');
    generalPanel.classList.remove('show', 'active');
  });

  // Load current settings
  loadCurrentSettings();

  // Show modal using Bootstrap 5
  const modal = new bootstrap.Modal(document.getElementById('settingsModal'));
  modal.show();

  // Clean up when modal is hidden and navigate away from /settings
  document.getElementById('settingsModal').addEventListener('hidden.bs.modal', function () {
    this.remove();
    // If still on /settings, navigate to home to prevent modal reopening
    if (window.location.pathname === '/settings') {
      window.page('/');
    }
  });
}

// Load current settings from localStorage or defaults
async function loadCurrentSettings() {
  const settings = await window.getSettings();

  document.getElementById('showHiddenTopics').checked = settings.showHiddenTopics;
  document.getElementById('wordsInLists').value = settings.wordsInLists;
  document.getElementById('wordsInTopic').value = settings.wordsInTopic;
  document.getElementById('wordsInCircles').value = settings.wordsInCircles;
  document.getElementById('topicDocs').value = settings.topicDocs;

  // Load cache stats
  await refreshCacheStats();
}

// Save settings to localStorage
window.saveSettings = async function() {
  const settings = {
    showHiddenTopics: document.getElementById('showHiddenTopics').checked,
    wordsInLists: parseInt(document.getElementById('wordsInLists').value),
    wordsInTopic: parseInt(document.getElementById('wordsInTopic').value),
    wordsInCircles: parseInt(document.getElementById('wordsInCircles').value),
    topicDocs: parseInt(document.getElementById('topicDocs').value)
  };

  localStorage.setItem('dfrBrowserSettings', JSON.stringify(settings));

  // Close modal
  const modal = bootstrap.Modal.getInstance(document.getElementById('settingsModal'));
  modal.hide();

  // Show brief notification
  const alertDiv = document.createElement('div');
  alertDiv.className = 'alert alert-success alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
  alertDiv.style.zIndex = '9999';
  alertDiv.innerHTML = `
    <i class="bi bi-check-circle"></i> Settings saved! Reloading page...
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
  `;
  document.body.appendChild(alertDiv);

  // Refresh topic dropdown with new words in lists setting
  if (window.dfrState.topicKeys) {
    await populateTopicDropdown(window.dfrState.topicKeys);
  }

  // Reload the current page after a brief delay to show the notification
  setTimeout(() => {
    alertDiv.remove();
    // Get current path and reload it
    const currentPath = window.location.pathname;
    window.page.show(currentPath);
  }, 800);
};

// Handle navigation link clicks to use page.js
document.addEventListener('DOMContentLoaded', () => {
  // Show loading spinner immediately - earliest possible moment
  const main = document.getElementById('main-view');
  if (main) {
    main.innerHTML = `
      <div class="card">
        <div class="card-body text-center">
          <h3>Loading...</h3>
          <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">Loading...</span>
          </div>
          <p class="mt-2">Initializing application...</p>
        </div>
      </div>`;
  }

  console.log('[DFR] DOM Content Loaded');
  console.log('[DFR] Checking if page.js is loaded:', typeof window.page);

  // Make sure page.js is loaded before initializing
  if (typeof window.page === 'undefined') {
    console.error('[DFR] ERROR: page.js is not loaded!');
    console.error('[DFR] Check if the script path is correct: ../js/index.js');
    return;
  }

  // With hash-based routing, we don't need to intercept clicks
  // The browser handles hash changes natively and page.js picks them up

  // Initialize the app
  init();
});

// Cache Management Functions

window.refreshCacheStats = async function() {
  const statsContainer = document.getElementById('cache-stats');
  if (!statsContainer) return;

  statsContainer.innerHTML = '<p class="text-muted small">Loading cache statistics...</p>';

  try {
    // Import CacheManager
    const CacheManager = (await import('./cache-manager.js')).default;
    const detailedStats = await CacheManager.getCacheStats();

    if (!detailedStats) {
      statsContainer.innerHTML = '<p class="text-danger small">Error loading cache statistics</p>';
      return;
    }

    const { totalFiles, totalSize, stores } = detailedStats;

    // Also check BibliographyDB
    let bibliographyDBCount = 0;
    let bibliographyDBSize = 0;
    try {
      const bibInfo = await new Promise((resolve) => {
        const request = indexedDB.open('BibliographyDB', 1);
        request.onsuccess = () => {
          const db = request.result;
          if (!db.objectStoreNames.contains('bibliography')) {
            resolve({ count: 0, size: 0 });
            return;
          }

          const transaction = db.transaction(['bibliography'], 'readonly');
          const store = transaction.objectStore('bibliography');
          const getRequest = store.getAll();

          getRequest.onsuccess = () => {
            const items = getRequest.result || [];
            const size = items.reduce((sum, item) => sum + JSON.stringify(item).length, 0);
            resolve({ count: items.length, size });
          };

          getRequest.onerror = () => resolve({ count: 0, size: 0 });
        };
        request.onerror = () => resolve({ count: 0, size: 0 });
      });

      bibliographyDBCount = bibInfo.count;
      bibliographyDBSize = bibInfo.size;
    } catch (err) {
      console.warn('[DFR] Could not read BibliographyDB stats:', err);
    }

    let html = '<div class="mb-2">';
    html += '<div class="d-flex justify-content-between mb-2">';
    html += `<span class="text-muted small">Total Files:</span>`;
    html += `<strong>${totalFiles + bibliographyDBCount}</strong>`;
    html += '</div>';
    html += '<div class="d-flex justify-content-between mb-2">';
    html += `<span class="text-muted small">Total Size:</span>`;
    html += `<strong>${formatBytes(totalSize + bibliographyDBSize)}</strong>`;
    html += '</div>';
    html += '</div>';

    html += '<hr class="my-2">';
    html += '<div class="small">';

    Object.entries(stores).forEach(([name, data]) => {
      const label = name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      html += `<div class="d-flex justify-content-between mb-1">`;
      html += `<span class="text-muted">${label}:</span>`;
      html += `<span>${data.count} (${formatBytes(data.size)})</span>`;
      html += '</div>';
    });

    // Add BibliographyDB stats
    if (bibliographyDBCount > 0 || bibliographyDBSize > 0) {
      html += `<div class="d-flex justify-content-between mb-1">`;
      html += `<span class="text-muted">Bibliography DB:</span>`;
      html += `<span>${bibliographyDBCount} (${formatBytes(bibliographyDBSize)})</span>`;
      html += '</div>';
    }

    html += '</div>';

    statsContainer.innerHTML = html;
  } catch (err) {
    console.error('[DFR] Error loading cache stats:', err);
    statsContainer.innerHTML = '<p class="text-danger small">Error loading cache statistics</p>';
  }
};window.clearAllCaches = async function() {
  if (!confirm('Are you sure you want to clear all caches? This will remove all stored data and cannot be undone.')) {
    return;
  }

  const statsContainer = document.getElementById('cache-stats');
  if (statsContainer) {
    statsContainer.innerHTML = '<p class="text-muted small"><i class="bi bi-hourglass-split"></i> Clearing caches...</p>';
  }

  try {
    // Clear cache-manager stores
    const success = await window.clearCache();

    // Also clear BibliographyDB
    if (window.bibliographyCache && window.bibliographyCache.clear) {
      await window.bibliographyCache.clear();
    } else {
      // Fallback: directly delete BibliographyDB
      await new Promise((resolve) => {
        const request = indexedDB.deleteDatabase('BibliographyDB');
        request.onsuccess = () => {
          console.log('[DFR] BibliographyDB deleted');
          resolve();
        };
        request.onerror = () => {
          console.warn('[DFR] Error deleting BibliographyDB');
          resolve();
        };
        request.onblocked = () => {
          console.warn('[DFR] BibliographyDB deletion blocked');
          resolve();
        };
      });
    }

    if (success) {
      alert('All caches cleared successfully!');
      await window.refreshCacheStats();
    } else {
      alert('Error clearing caches. Please check the console for details.');
    }
  } catch (err) {
    console.error('[DFR] Error clearing caches:', err);
    alert('Error clearing caches: ' + err.message);
  }
};

window.pruneCacheEntries = async function() {
  const statsContainer = document.getElementById('cache-stats');
  if (statsContainer) {
    statsContainer.innerHTML = '<p class="text-muted small"><i class="bi bi-hourglass-split"></i> Pruning old entries...</p>';
  }

  try {
    const deleted = await window.pruneCache(7); // 7 days

    alert(`Removed ${deleted} old cache entries (older than 7 days)`);
    await window.refreshCacheStats();
  } catch (err) {
    console.error('[DFR] Error pruning caches:', err);
    alert('Error pruning caches: ' + err.message);
  }
};

// Individual cache clear functions
window.clearBibliographyCache = async function() {
  if (!confirm('Clear bibliography cache? This will reload bibliography data on next view.')) {
    return;
  }

  console.log('[DFR] Starting bibliography cache clear...');

  try {
    const CacheManager = (await import('./cache-manager.js')).default;

    // Clear from cache-manager store
    console.log('[DFR] Clearing BIBLIOGRAPHY store from cache-manager...');
    await CacheManager.clearStore(CacheManager.stores.BIBLIOGRAPHY);
    console.log('[DFR] BIBLIOGRAPHY store cleared');

    // Also clear the BibliographyDB database
    console.log('[DFR] Attempting to clear BibliographyDB...');

    // IMPORTANT: Close all open connections to BibliographyDB first
    console.log('[DFR] Closing any open BibliographyDB connections...');

    // If bibliography module has an open connection, close it
    if (window.bibliographyCache && window.bibliographyCache.db) {
      console.log('[DFR] Closing bibliographyCache.db connection');
      window.bibliographyCache.db.close();
      window.bibliographyCache.db = null;
    }

    // Wait a moment for connections to close
    await new Promise(resolve => setTimeout(resolve, 100));

    // Now delete the database
    const deleted = await new Promise((resolve) => {
      console.log('[DFR] Requesting BibliographyDB deletion...');
      const request = indexedDB.deleteDatabase('BibliographyDB');

      request.onsuccess = () => {
        console.log('[DFR] BibliographyDB deletion successful');
        resolve(true);
      };

      request.onerror = (event) => {
        console.error('[DFR] Error deleting BibliographyDB:', event);
        resolve(false);
      };

      request.onblocked = () => {
        console.warn('[DFR] BibliographyDB deletion still blocked after closing connections');
        alert('Warning: Cannot delete BibliographyDB. It may be in use by another tab. Please:\n1. Close all other tabs with this app open\n2. Refresh this page\n3. Try clearing the cache again');
        resolve(false);
      };

      // Timeout after 5 seconds
      setTimeout(() => {
        console.warn('[DFR] BibliographyDB deletion timeout');
        resolve(false);
      }, 5000);
    });

    if (deleted) {
      showSuccessAlert('Bibliography cache cleared successfully!');
    } else {
      alert('Bibliography cache partially cleared. BibliographyDB could not be deleted. Try refreshing the page and clearing again.');
    }

    await window.refreshCacheStats();
  } catch (err) {
    console.error('[DFR] Error clearing bibliography cache:', err);
    console.error('[DFR] Error stack:', err.stack);
    alert('Error clearing bibliography cache: ' + err.message);
  }
};

window.clearMetadataCache = async function() {
  if (!confirm('Clear metadata cache? This will reload metadata on next view.')) {
    return;
  }

  try {
    const CacheManager = (await import('./cache-manager.js')).default;
    await CacheManager.clearStore(CacheManager.stores.METADATA);
    showSuccessAlert('Metadata cache cleared successfully!');
    await window.refreshCacheStats();
  } catch (err) {
    console.error('[DFR] Error clearing metadata cache:', err);
    alert('Error clearing metadata cache: ' + err.message);
  }
};

window.clearTopicsCache = async function() {
  if (!confirm('Clear topics cache? This will reload topic data on next view.')) {
    return;
  }

  try {
    const CacheManager = (await import('./cache-manager.js')).default;
    await CacheManager.clearStore(CacheManager.stores.TOPIC_KEYS);
    await CacheManager.clearStore(CacheManager.stores.DOC_TOPICS);
    showSuccessAlert('Topics cache cleared successfully!');
    await window.refreshCacheStats();
  } catch (err) {
    console.error('[DFR] Error clearing topics cache:', err);
    alert('Error clearing topics cache: ' + err.message);
  }
};

window.clearVocabularyCache = async function() {
  if (!confirm('Clear vocabulary cache? This will reload vocabulary data on next view.')) {
    return;
  }

  try {
    const CacheManager = (await import('./cache-manager.js')).default;
    await CacheManager.clearStore(CacheManager.stores.VOCABULARY);
    showSuccessAlert('Vocabulary cache cleared successfully!');
    await window.refreshCacheStats();
  } catch (err) {
    console.error('[DFR] Error clearing vocabulary cache:', err);
    alert('Error clearing vocabulary cache: ' + err.message);
  }
};

window.clearStateCache = async function() {
  if (!confirm('Clear state cache? This will reload state file data on next view.')) {
    return;
  }

  try {
    const CacheManager = (await import('./cache-manager.js')).default;
    await CacheManager.clearStore(CacheManager.stores.STATE);
    showSuccessAlert('State cache cleared successfully!');
    await window.refreshCacheStats();
  } catch (err) {
    console.error('[DFR] Error clearing state cache:', err);
    alert('Error clearing state cache: ' + err.message);
  }
};

// Helper function to format bytes
function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Show success alert
function showSuccessAlert(message) {
  const alertDiv = document.createElement('div');
  alertDiv.className = 'alert alert-success alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
  alertDiv.style.zIndex = '9999';
  alertDiv.innerHTML = `
    ${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
  `;
  document.body.appendChild(alertDiv);

  // Auto-dismiss after 3 seconds
  setTimeout(() => {
    alertDiv.remove();
  }, 3000);
}
