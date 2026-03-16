/**
 * Citation.js wrapper for DFR Browser
 *
 * This module provides a wrapper around citation-js with support for multiple citation styles
 * using CSL (Citation Style Language) templates.
 */

// Import citation-js from CDN (loaded via script tag in HTML)
// This will be available as window.Cite after loading from CDN

/**
 * Available citation styles with their CSL template names
 */
export const CITATION_STYLES = {
  'apa': { name: 'APA 7th Edition', template: 'apa' },
  'apa-6': { name: 'APA 6th Edition', template: 'apa-6th-edition' },
  'mla': { name: 'MLA 9th Edition', template: 'modern-language-association' },
  'mla-8': { name: 'MLA 8th Edition', template: 'modern-language-association-8th-edition' },
  'chicago': { name: 'Chicago Manual of Style (Author-Date)', template: 'chicago-author-date' },
  'chicago-note': { name: 'Chicago Manual of Style (Notes)', template: 'chicago-note-bibliography' },
  'harvard': { name: 'Harvard', template: 'harvard-cite-them-right' },
  'vancouver': { name: 'Vancouver', template: 'vancouver' },
  'ieee': { name: 'IEEE', template: 'ieee' },
  'nature': { name: 'Nature', template: 'nature' },
  'science': { name: 'Science', template: 'science' },
  'acs': { name: 'ACS (American Chemical Society)', template: 'american-chemical-society' },
  'ama': { name: 'AMA (American Medical Association)', template: 'american-medical-association' },
  'apa-annotated': { name: 'APA Annotated Bibliography', template: 'apa-annotated-bibliography' },
  'turabian': { name: 'Turabian (Author-Date)', template: 'turabian-author-date' },
  'bluebook': { name: 'Bluebook Law Review', template: 'bluebook-law-review' }
};

/**
 * Citation wrapper class
 */
export class Cite {
  constructor(data) {
    // Check if citation-js is loaded
    if (typeof window.Cite === 'undefined') {
      throw new Error('Citation.js library not loaded. Please include citation-js via CDN.');
    }

    // Create citation-js instance
    this.cite = new window.Cite(data);
    this.data = Array.isArray(data) ? data : [data];
  }

  /**
   * Format citation in the specified style
   *
   * @param {string} format - Output format ('bibliography', 'bibtex', 'ris', 'csl-json')
   * @param {Object} options - Formatting options
   * @param {string} options.template - Citation style template (e.g., 'apa', 'mla', 'chicago')
   * @param {string} options.format - Output format ('html' or 'text')
   * @param {string} options.lang - Language code (default: 'en-US')
   * @returns {string} Formatted citation
   */
  async format(format, options = {}) {
    const template = options.template || 'apa';
    const outputFormat = options.format || 'html';
    const lang = options.lang || 'en-US';

    try {
      switch (format) {
        case 'bibliography':
          return await this.formatBibliography(template, outputFormat, lang);

        case 'bibtex':
          return this.cite.format('bibtex');

        case 'ris':
          return this.cite.format('ris');

        case 'csl-json':
          return JSON.stringify(this.data, null, 2);

        default:
          throw new Error(`Unsupported format: ${format}`);
      }
    } catch (error) {
      console.error('[Citation] Format error:', error);
      // Fallback to basic formatting
      return this.formatFallback(outputFormat);
    }
  }

  /**
   * Format as bibliography using CSL styles
   */
  async formatBibliography(template, outputFormat, lang) {
    try {
      // Check if the template exists in our mapping
      const styleInfo = CITATION_STYLES[template];
      const cslTemplate = styleInfo ? styleInfo.template : template;

      // Use citation-js to format
      const formatted = this.cite.format('bibliography', {
        format: outputFormat,
        template: cslTemplate,
        lang: lang
      });

      return formatted;
    } catch (error) {
      console.error(`[Citation] Error formatting with template "${template}":`, error);
      // Try with default APA template
      if (template !== 'apa') {
        console.log('[Citation] Falling back to APA template');
        return this.cite.format('bibliography', {
          format: outputFormat,
          template: 'apa',
          lang: lang
        });
      }
      throw error;
    }
  }

  /**
   * Fallback formatting when citation-js fails
   */
  formatFallback(outputFormat) {
    const entry = this.data[0];
    if (!entry) return '';

    const parts = [];

    // Authors
    if (entry.author && entry.author.length > 0) {
      const authors = entry.author.map(a => {
        if (a.literal) return a.literal;
        return `${a.family}, ${a.given || ''}`;
      });
      parts.push(authors.join(', & '));
    }

    // Year
    if (entry.issued && entry.issued['date-parts']) {
      parts.push(`(${entry.issued['date-parts'][0][0]})`);
    }

    // Title
    if (entry.title) {
      parts.push(entry.title);
    }

    // Journal
    if (entry['container-title']) {
      const journal = outputFormat === 'html'
        ? `<em>${this.stripHtml(entry['container-title'])}</em>`
        : this.stripHtml(entry['container-title']);
      parts.push(journal);
    }

    return parts.join('. ') + '.';
  }

  /**
   * Strip HTML tags
   */
  stripHtml(text) {
    if (!text) return text;
    return text.replace(/<[^>]*>/g, '');
  }
}

/**
 * Initialize citation-js with plugins
 * This should be called after citation-js is loaded from CDN
 */
export async function initializeCitationJS() {
  if (typeof window.Cite === 'undefined') {
    console.warn('[Citation] Citation.js not loaded from CDN');
    return false;
  }

  try {
    // Citation-js plugins are already included in the CDN bundle
    // No additional initialization needed for basic functionality
    console.log('[Citation] Citation.js initialized successfully');
    return true;
  } catch (error) {
    console.error('[Citation] Error initializing citation-js:', error);
    return false;
  }
}

/**
 * Load CSL style template
 * CSL styles are loaded from the official CSL style repository
 */
export async function loadCSLStyle(styleName) {
  try {
    const url = `https://raw.githubusercontent.com/citation-style-language/styles/master/${styleName}.csl`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to load CSL style: ${styleName}`);
    }

    const cslXml = await response.text();

    // Register the style with citation-js
    if (window.Cite && window.Cite.CSL) {
      window.Cite.CSL.register.addTemplate(styleName, cslXml);
      console.log(`[Citation] Loaded CSL style: ${styleName}`);
      return true;
    }

    return false;
  } catch (error) {
    console.error(`[Citation] Error loading CSL style "${styleName}":`, error);
    return false;
  }
}
