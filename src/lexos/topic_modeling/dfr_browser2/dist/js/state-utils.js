// State utilities for handling MALLET topic-state files

// Helper function to ensure paths work on any sub-path deployment
function ensureAbsolutePath(path) {
  if (!path) return path;
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  if (path.startsWith('/')) return (window.dfrBasePath || '') + path;
  return path;
}

// Global cache for parsed state data
let parsedStateData = null;
let stateFileConfig = null;

// Extract doc-topic counts matrix: docTopicCounts[doc][topic] = count
export async function extractDocTopicCounts(topicCount) {
  const stateData = await loadStateFile();
  if (!stateData) return [];

  // First, determine the number of documents
  let maxDoc = -1;
  for (const line of stateData.lines) {
    if (line.startsWith('#') || line.trim() === '') continue;
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 6) {
      const docId = parseInt(parts[0]);
      if (!isNaN(docId) && docId > maxDoc) maxDoc = docId;
    }
  }
  const docCount = maxDoc + 1;

  // Initialize matrix
  const docTopicCounts = Array(docCount).fill(null).map(() => Array(topicCount).fill(0));

  for (const line of stateData.lines) {
    if (line.startsWith('#') || line.trim() === '') continue;
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 6) {
      const docId = parseInt(parts[0]);
      const topicId = parseInt(parts[5]);
      if (!isNaN(docId) && !isNaN(topicId) && docId >= 0 && topicId >= 0 && topicId < topicCount) {
        docTopicCounts[docId][topicId]++;
      }
    }
  }

  console.log(`✅ Extracted doc-topic counts: ${docCount} documents, ${topicCount} topics`);
  return docTopicCounts;
}

// Extract doc lengths: docLengths[doc] = token count
export async function extractDocLengths() {
  const stateData = await loadStateFile();
  if (!stateData) return {};

  const docLengths = {};
  for (const line of stateData.lines) {
    if (line.startsWith('#') || line.trim() === '') continue;
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 6) {
      const docId = parseInt(parts[0]);
      if (!isNaN(docId)) {
        docLengths[docId] = (docLengths[docId] || 0) + 1;
      }
    }
  }

  console.log(`✅ Extracted doc lengths for ${Object.keys(docLengths).length} documents`);
  return docLengths;
}

// Function to get state file configuration
async function getStateFileConfig() {
  if (stateFileConfig) return stateFileConfig;

  try {
    const response = await fetch('config.json');
    const config = await response.json();
    stateFileConfig = {
      path: ensureAbsolutePath(config.topic_state_file || 'sample_data/topic-state.gz'),
      available: true
    };
  } catch (error) {
    console.warn('Could not load config, using default state file path');
    stateFileConfig = {
      path: ensureAbsolutePath('sample_data/topic-state.gz'),
      available: true
    };
  }

  return stateFileConfig;
}

// Function to load and parse the state file (cached)
export async function loadStateFile() {
  // Return cached data if available
  if (parsedStateData) {
    return parsedStateData;
  }

  try {
    const config = await getStateFileConfig();
    const response = await fetch(config.path);

    if (!response.ok) {
      console.log('⚠️ State file not found at:', config.path);
      return null;
    }

    console.log('📊 State file found, parsing...');

    // Handle compressed file
    const stream = response.body.pipeThrough(new DecompressionStream('gzip'));
    const text = await new Response(stream).text();

    const lines = text.split('\n').filter(line => line.trim());
    console.log(`📊 Processing ${lines.length} lines from state file`);

    // Store raw lines for different parsing needs
    parsedStateData = {
      lines: lines,
      parsed: false
    };

    console.log('✅ State file loaded successfully');
    return parsedStateData;

  } catch (error) {
    console.error('❌ Error loading state file:', error);
    return null;
  }
}

// Function to extract word frequencies for topics from state data
export async function extractTopicWords(topicCount, wordsCount = 50) {
  console.log(`🔍 Extracting ${wordsCount} words for ${topicCount} topics`);

  const stateData = await loadStateFile();
  if (!stateData) {
    console.log('⚠️ No state data available, falling back to keys file');
    return null;
  }

  // Initialize topic word counts
  const topicWordCounts = Array(topicCount).fill(null).map(() => ({}));

  // Parse each line: doc source pos typeindex type topic
  let validLines = 0;

  for (const line of stateData.lines) {
    // Skip header and comment lines
    if (line.startsWith('#') || line.trim() === '') continue;

    const parts = line.trim().split(/\s+/);

    if (parts.length >= 6) {
      const topicId = parseInt(parts[5]); // topic is the last column
      const word = parts[4]; // word is the 5th column (type)

      if (topicId >= 0 && topicId < topicCount && word && word !== 'NA') {
        topicWordCounts[topicId][word] = (topicWordCounts[topicId][word] || 0) + 1;
        validLines++;
      }
    }
  }

  console.log(`📈 Processed ${validLines} valid word-topic assignments`);

  // Convert to sorted word lists
  const topicWordLists = topicWordCounts.map((wordCounts) => {
    return Object.entries(wordCounts)
      .sort(([,a], [,b]) => b - a)
      .map(([word]) => word)
      .slice(0, wordsCount);
  });

  console.log(`✅ Successfully extracted words from state file`);
  return topicWordLists;
}

export async function extractFullVocabulary() {
  console.log('📚 Extracting full vocabulary from state file...');

  const stateData = await loadStateFile();
  if (!stateData) {
    console.log('⚠️ No state data available for vocabulary extraction');
    return [];
  }

  const vocabulary = new Set();

  // Parse each line to collect all unique words
  for (const line of stateData.lines) {
    if (line.startsWith('#') || line.trim() === '') continue;

    const parts = line.trim().split(/\s+/);
    if (parts.length >= 6) {
      const word = parts[4]; // word is the 5th column (type)
      if (word && word !== 'NA') {
        vocabulary.add(word);
      }
    }
  }

  const vocabArray = Array.from(vocabulary).sort();
  console.log(`📖 Extracted ${vocabArray.length} unique words from state file`);
  return vocabArray;
}

export function clearStateCache() {
  parsedStateData = null;
  stateFileConfig = null;
}
