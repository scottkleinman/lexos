// Word List View
import { loadWordView } from './word.js';
import { extractFullVocabulary } from './state-utils.js';
import CachedDataLoader from './cached-data-loader.js';

// Helper function to ensure paths work on any sub-path deployment
function ensureAbsolutePath(path) {
  if (!path) return path;
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  if (path.startsWith('/')) return (window.dfrBasePath || '') + path;
  return path;
}

// Global configuration object
let appConfig = null;

// Load application configuration
async function loadConfig() {
  try {
    const response = await fetch(ensureAbsolutePath('config.json'));
    appConfig = await response.json();
    return appConfig;
  } catch (error) {
    console.error('Failed to load configuration:', error);
    // Fallback to default English configuration
    appConfig = {
      language: {
        default: 'en',
        configs: {
          'en': {
            name: 'English',
            alphabet: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
            locale: 'en-US'
          }
        }
      }
    };
    return appConfig;
  }
}

// Get current language setting (default from config)
function getCurrentLanguage() {
  return appConfig?.language?.default || 'en';
}

// Normalize character for grouping (remove diacritics for most languages)
function normalizeCharacter(char, langConfig) {
  if (langConfig.locale === 'zh-CN') {
    // For Chinese, we might group by radical or frequency
    return char;
  }

  // Remove diacritics for most Latin-based languages
  return char.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toUpperCase();
}

// Get the grouping character for a word
function getGroupingCharacter(word, langConfig) {
  if (!word) return '#';

  const firstChar = word.charAt(0);

  if (langConfig.locale === 'zh-CN') {
    // For Chinese, you might implement radical-based grouping
    // This is a simplified approach - in practice you'd use a radical lookup
    return firstChar;
  }

  const normalized = normalizeCharacter(firstChar, langConfig);

  // Check if the normalized character is in the language's alphabet
  if (langConfig.alphabet && langConfig.alphabet.includes(normalized)) {
    return normalized;
  }

  // For characters not in the alphabet, group under '#'
  return '#';
}

// Global state for vocabulary mode
let useCompleteVocabulary = false;
let completeVocabulary = null;
let basicVocabulary = null;
// Store parameters for toggle functionality
let currentTopicKeys = null;
let currentDocTopic = null;
let currentMetadata = null;

export async function loadWordList(topicKeys, docTopic, metadata) {
  const main = document.getElementById('main-view');

  // Store parameters for toggle functionality
  currentTopicKeys = topicKeys;
  currentDocTopic = docTopic;
  currentMetadata = metadata;

  // Load configuration if not already loaded
  if (!appConfig) {
    await loadConfig();
  }

  const currentLang = getCurrentLanguage();
  const langConfig = appConfig.language.configs[currentLang];

  // Cache basic vocabulary from topic keys
  if (!basicVocabulary) {
    const uniqueWords = new Set();
    if (topicKeys && topicKeys.length > 0) {
      topicKeys.forEach(topic => {
        if (topic.words && Array.isArray(topic.words)) {
          topic.words.forEach(word => {
            uniqueWords.add(word.toLowerCase());
          });
        }
      });
    }
    basicVocabulary = Array.from(uniqueWords).sort((a, b) => {
      return a.localeCompare(b, langConfig.locale);
    });
  }

  // Load complete vocabulary in background if not already loaded or loading
  if (!completeVocabulary && !window.completeVocabLoading) {
    window.completeVocabLoading = true;

    // Start loading complete vocabulary asynchronously with caching
    CachedDataLoader.loadFullVocabulary(extractFullVocabulary).then(fullVocab => {
      if (fullVocab && fullVocab.length > 0) {
        completeVocabulary = fullVocab.map(word => word.toLowerCase()).sort((a, b) => {
          return a.localeCompare(b, langConfig.locale);
        });

        // Update the button text once loaded (only if still on word list view)
        const completeBtn = document.getElementById('vocab-complete');
        if (completeBtn && !useCompleteVocabulary) {
          completeBtn.textContent = `Complete (${completeVocabulary.length} words)`;
        }
      }
      window.completeVocabLoading = false;
    }).catch(error => {
      console.error('Failed to load complete vocabulary:', error);
      window.completeVocabLoading = false;
    });
  }

  // Get complete vocabulary from state file if requested and already cached
  if (useCompleteVocabulary && !completeVocabulary) {
    try {
      const fullVocab = await CachedDataLoader.loadFullVocabulary(extractFullVocabulary);
      if (fullVocab && fullVocab.length > 0) {
        completeVocabulary = fullVocab.map(word => word.toLowerCase()).sort((a, b) => {
          return a.localeCompare(b, langConfig.locale);
        });
      }
    } catch (error) {
      console.error('Failed to load complete vocabulary:', error);
      completeVocabulary = null;
    }
  }

  // Choose which vocabulary to use
  const sortedWords = useCompleteVocabulary && completeVocabulary ? completeVocabulary : basicVocabulary;

  // Group words by character
  const wordGroups = {};
  sortedWords.forEach(word => {
    const groupChar = getGroupingCharacter(word, langConfig);
    if (!wordGroups[groupChar]) {
      wordGroups[groupChar] = [];
    }
    wordGroups[groupChar].push(word);
  });

  // 3. Create the HTML with language-aware alphabet navigation
  let html = `<div class="card"><div class="card-body">`;
  html += `<div class="d-flex justify-content-between align-items-center mb-3">`;
  html += `<div>`;
  html += `<h3 class="mb-1">All words prominent in any topic</h3>`;

  // Set the message based on current vocabulary mode
  const vocabMessage = useCompleteVocabulary
    ? 'Click the Basic button to view only words prominent in topics.'
    : 'Only words prominent in topics are listed. Click the complete button to see the complete vocabulary.';

  html += `<p id="vocab-info" class="text-muted mb-0" style="font-style: italic;">${vocabMessage}</p>`;
  html += `</div>`;
  html += `<div class="btn-group" role="group">`;
  html += `<button type="button" class="btn ${!useCompleteVocabulary ? 'btn-primary' : 'btn-outline-primary'}" id="vocab-basic">`;
  html += `Basic (${basicVocabulary.length} words)`;
  html += `</button>`;
  html += `<button type="button" class="btn ${useCompleteVocabulary ? 'btn-primary' : 'btn-outline-primary'}" id="vocab-complete">`;
  html += `Complete${completeVocabulary ? ` (${completeVocabulary.length} words)` : ' (loading...)'}`;
  html += `</button>`;
  html += `</div>`;
  html += `</div>`;

  // Add sticky alphabet/character navbar
  const rtlClass = langConfig.rtl ? 'dir="rtl"' : '';
  html += `<nav class="navbar navbar-expand-lg navbar-light bg-light mb-3 position-sticky" style="top: 0; z-index: 1000;" ${rtlClass}>`;
  html += `<div class="container-fluid justify-content-center">`;
  html += `<div class="navbar-nav flex-row flex-wrap justify-content-center">`;

  // Generate alphabet/character navigation
  const sortedGroupKeys = Object.keys(wordGroups).sort((a, b) => {
    // Sort '#' to the end
    if (a === '#') return 1;
    if (b === '#') return -1;

    if (langConfig.alphabet && langConfig.alphabet.length > 0) {
      const aIndex = langConfig.alphabet.indexOf(a);
      const bIndex = langConfig.alphabet.indexOf(b);
      if (aIndex !== -1 && bIndex !== -1) {
        return aIndex - bIndex;
      }
    }

    return a.localeCompare(b, langConfig.locale);
  });

  // Create navigation for available characters/letters
  const allChars = langConfig.alphabet && langConfig.alphabet.length > 0 ?
    [...langConfig.alphabet, '#'] :
    sortedGroupKeys;

  allChars.forEach(char => {
    const hasWords = sortedGroupKeys.includes(char);
    const linkClass = hasWords ? 'nav-link px-2 text-primary' : 'nav-link px-2 text-muted';
    const clickable = hasWords ? `data-letter="${char}" style="cursor: pointer;"` : 'style="cursor: default;"';
    html += `<a class="${linkClass}" ${clickable}>${char}</a>`;
  });

  html += `</div></div></nav>`;

  // Display word groups
  sortedGroupKeys.forEach(char => {
    html += `<div class="word-group mb-3" id="letter-${char}">`;
    html += `<h5 class="fw-bold text-primary">${char}</h5>`;
    html += `<div class="word-links" ${rtlClass}>`;

    // Create comma-separated links for each word
    const wordLinks = wordGroups[char].map(word => {
      return `<a href="#" class="word-link text-decoration-none" data-word="${word}">${word}</a>`;
    }).join(', ');

    html += wordLinks;
    html += `</div></div>`;
  });

  html += `</div></div>`;
  main.innerHTML = html;

  // Add click event listeners to word links
  const wordLinks = main.querySelectorAll('.word-link');
  wordLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      const word = this.getAttribute('data-word');
      // Use router to update URL and trigger view
      window.page(`/word/${encodeURIComponent(word)}`);
    });
  });

  // Add click event listeners to alphabet navigation
  // Use setTimeout to ensure DOM is fully rendered
  setTimeout(() => {
    const alphabetLinks = document.querySelectorAll('.navbar [data-letter]');
    alphabetLinks.forEach(link => {
      link.addEventListener('click', function(e) {
        e.preventDefault();
        const letter = this.getAttribute('data-letter');
        const targetSection = document.getElementById(`letter-${letter}`);
        if (targetSection) {
          // Get the navbar height to offset the scroll
          const navbar = document.querySelector('.position-sticky');
          const navbarHeight = navbar ? navbar.offsetHeight : 60; // fallback to 60px

          // Calculate the target position minus navbar height
          const targetPosition = targetSection.offsetTop - navbarHeight - 10; // extra 10px padding

          // Smooth scroll to the adjusted position
          window.scrollTo({
            top: targetPosition,
            behavior: 'smooth'
          });
        }
      });
    });
  }, 0);

  // Add click event listeners to vocabulary toggle buttons
  setTimeout(() => {
    const basicBtn = document.getElementById('vocab-basic');
    const completeBtn = document.getElementById('vocab-complete');

    if (basicBtn) {
      basicBtn.addEventListener('click', async function(e) {
        e.preventDefault();
        if (useCompleteVocabulary === false) return; // Already in basic mode

        useCompleteVocabulary = false;
        await loadWordList(currentTopicKeys, currentDocTopic, currentMetadata);
      });
    }

    if (completeBtn) {
      completeBtn.addEventListener('click', async function(e) {
        e.preventDefault();
        if (useCompleteVocabulary === true) return; // Already in complete mode

        useCompleteVocabulary = true;
        await loadWordList(currentTopicKeys, currentDocTopic, currentMetadata);
      });
    }
  }, 0);
}
