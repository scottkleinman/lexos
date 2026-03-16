/**
 * Data Validator for DFR Browser 2
 * Validates data files and checks for integrity issues
 */

import ErrorHandler from './error-handler.js';

const DataValidator = {

  /**
   * Validate topic keys data
   */
  validateTopicKeys(topicKeys, filename = 'topic-keys.txt') {
    const errors = [];
    const warnings = [];

    // Check if data exists
    if (!topicKeys || topicKeys.length === 0) {
      errors.push('No topic data found');
      ErrorHandler.handleValidationError(
        'Topic keys file is empty or invalid',
        `File: ${filename}`,
        [
          'Ensure the file contains topic data',
          'Check that the file format matches MALLET output',
          'Verify the file is not corrupted'
        ]
      );
      return { valid: false, errors, warnings };
    }

    // Validate structure
    topicKeys.forEach((topic, index) => {
      if (!topic.hasOwnProperty('topic') && !topic.hasOwnProperty('id')) {
        errors.push(`Topic ${index} missing id/topic field`);
      }
      if (!topic.words || !Array.isArray(topic.words)) {
        errors.push(`Topic ${index} missing or invalid words array`);
      }
      if (topic.words && topic.words.length === 0) {
        warnings.push(`Topic ${index} has no words`);
      }
    });

    // Check for duplicate topic IDs
    const topicIds = topicKeys.map(t => t.topic || t.id);
    const duplicates = topicIds.filter((id, index) => topicIds.indexOf(id) !== index);
    if (duplicates.length > 0) {
      errors.push(`Duplicate topic IDs found: ${duplicates.join(', ')}`);
    }

    if (errors.length > 0) {
      ErrorHandler.handleValidationError(
        'Topic keys data validation failed',
        errors.join('\n'),
        [
          'Check the file format matches MALLET topic-keys.txt output',
          'Verify there are no duplicate topic numbers',
          'Ensure each topic has associated words'
        ]
      );
      return { valid: false, errors, warnings };
    }

    console.log(`✅ Topic keys validated: ${topicKeys.length} topics`);
    if (warnings.length > 0) {
      console.warn(`⚠️ Warnings:`, warnings);
    }

    return { valid: true, errors: [], warnings };
  },

  /**
   * Validate doc-topics data
   */
  validateDocTopics(docTopics, filename = 'doc-topics.txt') {
    const errors = [];
    const warnings = [];

    if (!docTopics || docTopics.length === 0) {
      errors.push('No document-topic data found');
      ErrorHandler.handleValidationError(
        'Doc-topics file is empty or invalid',
        `File: ${filename}`,
        [
          'Ensure the file contains document-topic distributions',
          'Check that the file format matches MALLET output',
          'Verify the file is not corrupted'
        ]
      );
      return { valid: false, errors, warnings };
    }

    // Validate structure
    docTopics.forEach((doc, index) => {
      if (!Array.isArray(doc)) {
        errors.push(`Document ${index} is not an array of topic proportions`);
      }
      if (doc.length === 0) {
        warnings.push(`Document ${index} has no topic assignments`);
      }

      // Check if proportions sum to approximately 1
      const sum = doc.reduce((a, b) => a + b, 0);
      if (Math.abs(sum - 1.0) > 0.01) {
        warnings.push(`Document ${index} proportions sum to ${sum.toFixed(3)}, not 1.0`);
      }
    });

    if (errors.length > 0) {
      ErrorHandler.handleValidationError(
        'Doc-topics data validation failed',
        errors.join('\n'),
        [
          'Check the file format matches MALLET doc-topics.txt output',
          'Verify each document has topic proportions',
          'Ensure proportions are valid numbers'
        ]
      );
      return { valid: false, errors, warnings };
    }

    console.log(`✅ Doc-topics validated: ${docTopics.length} documents`);
    if (warnings.length > 0) {
      console.warn(`⚠️ Warnings:`, warnings);
    }

    return { valid: true, errors: [], warnings };
  },

  /**
   * Validate metadata
   */
  validateMetadata(metadata, filename = 'metadata.csv') {
    const errors = [];
    const warnings = [];

    if (!metadata || metadata.length === 0) {
      errors.push('No metadata found');
      ErrorHandler.handleValidationError(
        'Metadata file is empty or invalid',
        `File: ${filename}`,
        [
          'Ensure the file contains document metadata',
          'Check that the CSV format is correct',
          'Verify the file has headers and data rows'
        ]
      );
      return { valid: false, errors, warnings };
    }

    // Check for required fields
    const firstDoc = metadata[0];
    const requiredFields = ['docNum', 'docName', 'title', 'year'];

    requiredFields.forEach(field => {
      if (!firstDoc.hasOwnProperty(field)) {
        errors.push(`Missing required field: ${field}`);
      }
    });

    if (errors.length > 0) {
      ErrorHandler.handleValidationError(
        'Metadata validation failed - missing required fields',
        `File: ${filename}\nMissing fields: ${errors.join(', ')}`,
        [
          'Ensure metadata CSV has required fields: docNum, docName, title, year',
          'Check the field names match exactly (case-sensitive)',
          'Verify the CSV header row is present'
        ]
      );
      return { valid: false, errors, warnings };
    }

    // Check for empty critical fields
    let emptyTitles = 0;
    let emptyDates = 0;
    metadata.forEach((doc, index) => {
      if (!doc.title || doc.title.trim() === '') emptyTitles++;
      if (!doc.date && !doc.year) emptyDates++;
    });

    if (emptyTitles > 0) {
      warnings.push(`${emptyTitles} documents have empty titles`);
    }
    if (emptyDates > 0) {
      warnings.push(`${emptyDates} documents have no date/year`);
    }

    if (errors.length > 0) {
      ErrorHandler.handleValidationError(
        'Metadata validation failed',
        errors.join('\n'),
        [
          'Ensure metadata CSV is properly formatted',
          'Check that the CSV has valid column headers',
          'Verify the file is not corrupted'
        ]
      );
      return { valid: false, errors, warnings };
    }

    console.log(`✅ Metadata validated: ${metadata.length} documents`);
    if (warnings.length > 0) {
      console.warn(`⚠️ Warnings:`, warnings);
    }

    return { valid: true, errors: [], warnings };
  },

  /**
   * Validate data integrity across files
   */
  validateDataIntegrity(topicKeys, docTopics, metadata) {
    const errors = [];
    const warnings = [];

    // Check topic count consistency
    const topicCount = topicKeys.length;
    const docTopicTopicCount = docTopics.length > 0 ? docTopics[0].length : 0;

    if (topicCount !== docTopicTopicCount) {
      errors.push(
        `Topic count mismatch: topic-keys has ${topicCount} topics, ` +
        `but doc-topics has ${docTopicTopicCount} topics per document`
      );
    }

    // Check document count consistency
    const docCount = docTopics.length;
    const metadataCount = metadata.length;

    if (docCount !== metadataCount) {
      warnings.push(
        `Document count mismatch: doc-topics has ${docCount} documents, ` +
        `but metadata has ${metadataCount} documents`
      );
    }

    // Check for data completeness
    const completeness = {
      topicKeys: topicKeys.length > 0,
      docTopics: docTopics.length > 0,
      metadata: metadata.length > 0
    };

    const missingData = Object.entries(completeness)
      .filter(([_, exists]) => !exists)
      .map(([name, _]) => name);

    if (missingData.length > 0) {
      errors.push(`Missing data: ${missingData.join(', ')}`);
    }

    if (errors.length > 0) {
      ErrorHandler.handleIntegrityError(
        'Data integrity check failed',
        errors.join('\n'),
        [
          'Ensure all data files are from the same MALLET run',
          'Verify that topic counts match across files',
          'Check that document counts are consistent',
          'Re-export data from MALLET if necessary'
        ]
      );
      return { valid: false, errors, warnings };
    }

    console.log(`✅ Data integrity validated`);
    console.log(`   - Topics: ${topicCount}`);
    console.log(`   - Documents: ${docCount}`);
    console.log(`   - Metadata records: ${metadataCount}`);

    if (warnings.length > 0) {
      console.warn(`⚠️ Integrity warnings:`, warnings);
    }

    return { valid: true, errors: [], warnings };
  },

  /**
   * Comprehensive validation of all data
   */
  async validateAll(topicKeys, docTopics, metadata) {
    console.log('🔍 Starting comprehensive data validation...');

    const results = {
      topicKeys: this.validateTopicKeys(topicKeys),
      docTopics: this.validateDocTopics(docTopics),
      metadata: this.validateMetadata(metadata),
      integrity: { valid: true, errors: [], warnings: [] }
    };

    // Only check integrity if individual validations passed
    if (results.topicKeys.valid && results.docTopics.valid && results.metadata.valid) {
      results.integrity = this.validateDataIntegrity(topicKeys, docTopics, metadata);
    }

    const allValid = Object.values(results).every(r => r.valid);

    if (allValid) {
      console.log('✅ All data validation passed');
    } else {
      console.error('❌ Data validation failed');
    }

    return results;
  },

  /**
   * Get data quality report
   */
  getDataQualityReport(topicKeys, docTopics, metadata) {
    const report = {
      summary: {
        topics: topicKeys.length,
        documents: docTopics.length,
        metadataRecords: metadata.length
      },
      completeness: {
        topicWords: topicKeys.filter(t => t.words && t.words.length > 0).length / topicKeys.length,
        documentAssignments: docTopics.filter(d => d.length > 0).length / docTopics.length,
        metadataFields: this.calculateMetadataCompleteness(metadata)
      },
      issues: {
        emptyTopics: topicKeys.filter(t => !t.words || t.words.length === 0).length,
        emptyDocuments: docTopics.filter(d => d.length === 0).length,
        missingTitles: metadata.filter(d => !d.title || d.title.trim() === '').length,
        missingDates: metadata.filter(d => !d.date && !d.year).length
      }
    };

    return report;
  },

  /**
   * Calculate metadata completeness
   */
  calculateMetadataCompleteness(metadata) {
    if (metadata.length === 0) return {};

    const fields = Object.keys(metadata[0]);
    const completeness = {};

    fields.forEach(field => {
      const nonEmpty = metadata.filter(d => d[field] && d[field].toString().trim() !== '').length;
      completeness[field] = (nonEmpty / metadata.length * 100).toFixed(1) + '%';
    });

    return completeness;
  }
};

export default DataValidator;
