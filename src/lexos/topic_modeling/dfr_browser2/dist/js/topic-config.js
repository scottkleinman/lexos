/**
 * Topic Configuration Utilities
 * Handles custom topic labels, hidden topics, and metadata field mappings
 */

/**
 * Get custom label for a topic, or default label if none configured
 * @param {number} topicIndex - 0-based topic index
 * @returns {string} The topic label
 */
export function getTopicLabel(topicIndex) {
  const config = window.dfrState?.config?.topics;

  // Check if this topic has a custom label
  if (config?.labels && config.labels[topicIndex] !== undefined) {
    return config.labels[topicIndex];
  }

  // Return default label (1-based numbering)
  return `Topic ${topicIndex + 1}`;
}

/**
 * Check if a topic should be hidden
 * @param {number} topicIndex - 0-based topic index
 * @returns {boolean} True if topic should be hidden
 */
export function isTopicHidden(topicIndex) {
  const config = window.dfrState?.config?.topics;
  // Use correct settings key
  const settings = JSON.parse(localStorage.getItem('dfrBrowserSettings') || '{}');
  if (settings.showHiddenTopics) {
    return false;
  }
  // Hidden topics are 1-based in config, topicIndex is 0-based
  if (config?.hidden && Array.isArray(config.hidden)) {
    return config.hidden.includes(topicIndex + 1);
  }
  return false;
}

export function getVisibleTopicIndices(topicKeys) {
  const config = window.dfrState?.config?.topics;
  const settings = JSON.parse(localStorage.getItem('dfrBrowserSettings') || '{}');
  const showHidden = settings.showHiddenTopics;
  if (!topicKeys) return [];
  return topicKeys.map((tk, idx) => idx)
    .filter(idx => {
      if (showHidden) return true;
      return !(config?.hidden && Array.isArray(config.hidden) && config.hidden.includes(idx + 1));
    });
}

/**
 * Filter topics to exclude hidden ones
 * @param {Array} topics - Array of topic objects with topic property
 * @returns {Array} Filtered topics array
 */
export function filterHiddenTopics(topics) {
  return topics.filter(t => !isTopicHidden(t.topic));
}

/**
 * Get metadata field mapping
 * Maps custom field names to standard internal names
 * @param {string} internalField - Internal field name (e.g., 'title', 'author')
 * @returns {string} The actual field name to use in metadata
 */
export function getMetadataField(internalField) {
  const config = window.dfrState?.config?.metadata?.fieldMappings;

  if (config && config[internalField]) {
    return config[internalField];
  }

  // Return default field name
  return internalField;
}

/**
 * Get value from metadata using field mapping
 * @param {Object} doc - Document metadata object
 * @param {string} internalField - Internal field name
 * @returns {*} The field value
 */
export function getMetadataValue(doc, internalField) {
  const actualField = getMetadataField(internalField);
  return doc[actualField];
}

/**
 * Get all configured topics with their labels and visibility
 * @param {number} totalTopics - Total number of topics
 * @returns {Array} Array of topic info objects
 */
export function getAllTopicInfo(totalTopics) {
  const topics = [];
  for (let i = 0; i < totalTopics; i++) {
    topics.push({
      index: i,
      label: getTopicLabel(i),
      hidden: isTopicHidden(i),
      number: i + 1 // 1-based numbering
    });
  }
  return topics;
}

/**
 * Validate topic configuration
 * @returns {Object} Validation result with warnings
 */
export function validateTopicConfig() {
  const warnings = [];
  const config = window.dfrState?.config?.topics;

  if (!config) {
    return { valid: true, warnings: [] };
  }

  // Check for duplicate labels
  if (config.labels) {
    const labelValues = Object.values(config.labels);
    const duplicates = labelValues.filter((label, index) => labelValues.indexOf(label) !== index);
    if (duplicates.length > 0) {
      warnings.push(`Duplicate topic labels found: ${[...new Set(duplicates)].join(', ')}`);
    }
  }

  // Check for hidden topics with labels (might be intentional but worth noting)
  if (config.labels && config.hidden) {
    const labeledAndHidden = Object.keys(config.labels)
      .map(Number)
      .filter(idx => config.hidden.includes(idx));

    if (labeledAndHidden.length > 0) {
      warnings.push(`Topics with custom labels are also hidden: ${labeledAndHidden.map(i => i + 1).join(', ')}`);
    }
  }

  return {
    valid: true,
    warnings
  };
}
