/**
 * Citation formatting library for DFR Browser
 *
 * Provides citation formatting in multiple styles without external dependencies
 */

/**
 * Available citation styles
 */
export const CITATION_STYLES = [
  { value: 'chicago', label: 'Chicago (Author-Date)' },
  { value: 'chicago-note-bibliography', label: 'Chicago Manual of Style (Notes)' },
  { value: 'mla', label: 'MLA 9th Edition' },
  { value: 'apa', label: 'APA 7th Edition' },
  { value: 'harvard', label: 'Harvard' },
  { value: 'vancouver', label: 'Vancouver' },
  { value: 'ieee', label: 'IEEE' }
];

export class Cite {
  constructor(data) {
    this.data = Array.isArray(data) ? data : [data];
  }

  /**
   * Helper to strip HTML tags for text output
   */
  stripHtml(text) {
    if (!text) return text;
    return text.replace(/<[^>]*>/g, '');
  }

  /**
   * Helper to handle container-title that may already contain HTML
   */
  formatContainer(containerTitle, isHtml) {
    if (!containerTitle) return '';

    // If already contains HTML tags, use as-is for HTML output or strip for text
    if (containerTitle.includes('<em>')) {
      return isHtml ? containerTitle : this.stripHtml(containerTitle);
    }

    // Otherwise, add italics for HTML output
    return isHtml ? `<em>${containerTitle}</em>` : containerTitle;
  }

  /**
   * Format citation in specified style
   */
  async format(format, options = {}) {
    const template = options.template || 'apa';
    const outputFormat = options.format || 'html';

    switch (format) {
      case 'bibliography':
        return this.formatBibliography(template, outputFormat);
      case 'bibtex':
        return this.formatBibtex();
      case 'ris':
        return this.formatRis();
      default:
        throw new Error(`Unsupported format: ${format}`);
    }
  }

  /**
   * Format as bibliography
   */
  formatBibliography(template, format) {
    const citations = this.data.map(entry => {
      switch (template) {
        case 'mla':
        case 'modern-language-association':
          return this.formatMLA(entry, format);
        case 'chicago':
        case 'chicago-author-date':
          return this.formatChicago(entry, format);
        case 'harvard':
        case 'harvard-cite-them-right':
          return this.formatHarvard(entry, format);
        case 'vancouver':
          return this.formatVancouver(entry, format);
        case 'ieee':
          return this.formatIEEE(entry, format);
        case 'apa':
        case 'apa-6th-edition':
        default:
          return this.formatAPA(entry, format);
      }
    });

    if (format === 'html') {
      return citations.map(c => `<div class="csl-entry">${c}</div>`).join('\n');
    } else {
      return citations.join('\n\n');
    }
  }

  /**
   * Format single entry in APA style
   */
  formatAPA(entry, format) {
    const parts = [];
    const isHtml = format === 'html';

    // Authors
    if (entry.author && entry.author.length > 0) {
      const authorStr = this.formatAuthorsAPA(entry.author);
      parts.push(authorStr);
    }

    // Year
    if (entry.issued && entry.issued['date-parts']) {
      const year = entry.issued['date-parts'][0][0];
      parts.push(`(${year}).`);
    }

    // Title
    if (entry.title) {
      parts.push(entry.title + '.');
    }

    // Container (journal)
    if (entry['container-title']) {
      const container = this.formatContainer(entry['container-title'], isHtml);
      let containerPart = container;

      // Volume
      if (entry.volume) {
        containerPart += `, ${isHtml ? '<em>' : ''}${entry.volume}${isHtml ? '</em>' : ''}`;
      }

      // Issue
      if (entry.issue) {
        containerPart += `(${entry.issue})`;
      }

      // Pages
      if (entry.page) {
        containerPart += `, ${entry.page}`;
      }

      parts.push(containerPart + '.');
    }

    // DOI
    if (entry.DOI) {
      const doi = Array.isArray(entry.DOI) ? entry.DOI[0] : entry.DOI;
      if (isHtml) {
        parts.push(`<a href="https://doi.org/${doi}" target="_blank">https://doi.org/${doi}</a>`);
      } else {
        parts.push(`https://doi.org/${doi}`);
      }
    } else if (entry.URL) {
      if (isHtml) {
        parts.push(`<a href="${entry.URL}" target="_blank">${entry.URL}</a>`);
      } else {
        parts.push(entry.URL);
      }
    }

    return parts.join(' ');
  }

  /**
   * Format authors in APA style
   */
  formatAuthorsAPA(authors) {
    if (authors.length === 0) return '';

    const formatAuthor = (author) => {
      if (author.literal) return author.literal;
      const given = author.given ? ` ${author.given.charAt(0)}.` : '';
      return `${author.family},${given}`;
    };

    if (authors.length === 1) {
      return formatAuthor(authors[0]);
    } else if (authors.length === 2) {
      return `${formatAuthor(authors[0])}, & ${formatAuthor(authors[1])}`;
    } else {
      const formatted = authors.slice(0, -1).map(formatAuthor);
      return `${formatted.join(', ')}, & ${formatAuthor(authors[authors.length - 1])}`;
    }
  }

  /**
   * Format single entry in MLA style
   */
  formatMLA(entry, format) {
    const parts = [];
    const isHtml = format === 'html';

    // Authors (Last, First)
    if (entry.author && entry.author.length > 0) {
      const authorStr = this.formatAuthorsMLA(entry.author);
      parts.push(authorStr + '.');
    }

    // Title (in quotes)
    if (entry.title) {
      parts.push(`"${entry.title}."`);
    }

    // Container (journal) in italics
    if (entry['container-title']) {
      const container = this.formatContainer(entry['container-title'], isHtml);
      parts.push(container + ',');
    }

    // Volume and issue
    if (entry.volume) {
      let volIssue = `vol. ${entry.volume}`;
      if (entry.issue) {
        volIssue += `, no. ${entry.issue}`;
      }
      parts.push(volIssue + ',');
    }

    // Year
    if (entry.issued && entry.issued['date-parts']) {
      const year = entry.issued['date-parts'][0][0];
      parts.push(`${year},`);
    }

    // Pages (pp. notation)
    if (entry.page) {
      parts.push(`pp. ${entry.page}.`);
    }

    // DOI or URL
    if (entry.DOI) {
      const doi = Array.isArray(entry.DOI) ? entry.DOI[0] : entry.DOI;
      if (isHtml) {
        parts.push(`<a href="https://doi.org/${doi}" target="_blank">https://doi.org/${doi}</a>.`);
      } else {
        parts.push(`https://doi.org/${doi}.`);
      }
    } else if (entry.URL) {
      if (isHtml) {
        parts.push(`<a href="${entry.URL}" target="_blank">${entry.URL}</a>.`);
      } else {
        parts.push(`${entry.URL}.`);
      }
    }

    return parts.join(' ');
  }

  /**
   * Format authors in MLA style
   */
  formatAuthorsMLA(authors) {
    if (authors.length === 0) return '';

    const formatAuthor = (author, isFirst) => {
      if (author.literal) return author.literal;
      if (isFirst) {
        // First author: Last, First
        return `${author.family}, ${author.given || ''}`;
      } else {
        // Other authors: First Last
        return `${author.given || ''} ${author.family}`;
      }
    };

    if (authors.length === 1) {
      return formatAuthor(authors[0], true);
    } else if (authors.length === 2) {
      return `${formatAuthor(authors[0], true)}, and ${formatAuthor(authors[1], false)}`;
    } else {
      const first = formatAuthor(authors[0], true);
      const rest = authors.slice(1, -1).map(a => formatAuthor(a, false));
      const last = formatAuthor(authors[authors.length - 1], false);
      return `${first}, ${rest.join(', ')}, and ${last}`;
    }
  }

  /**
   * Format single entry in Chicago style
   */
  formatChicago(entry, format) {
    const parts = [];
    const isHtml = format === 'html';

    // Authors (Last, First)
    if (entry.author && entry.author.length > 0) {
      const authorStr = this.formatAuthorsChicago(entry.author);
      parts.push(authorStr + '.');
    }

    // Title (in quotes)
    if (entry.title) {
      parts.push(`"${entry.title}."`);
    }

    // Container (journal) in italics
    if (entry['container-title']) {
      const container = this.formatContainer(entry['container-title'], isHtml);
      let containerPart = container;

      // Volume
      if (entry.volume) {
        containerPart += ` ${entry.volume}`;
      }

      // Issue
      if (entry.issue) {
        containerPart += `, no. ${entry.issue}`;
      }

      parts.push(containerPart);
    }

    // Year and pages
    if (entry.issued && entry.issued['date-parts']) {
      const year = entry.issued['date-parts'][0][0];
      let yearPart = `(${year})`;
      if (entry.page) {
        yearPart += `: ${entry.page}`;
      }
      parts.push(yearPart + '.');
    }

    // DOI or URL
    if (entry.DOI) {
      const doi = Array.isArray(entry.DOI) ? entry.DOI[0] : entry.DOI;
      if (isHtml) {
        parts.push(`<a href="https://doi.org/${doi}" target="_blank">https://doi.org/${doi}</a>.`);
      } else {
        parts.push(`https://doi.org/${doi}.`);
      }
    } else if (entry.URL) {
      if (isHtml) {
        parts.push(`<a href="${entry.URL}" target="_blank">${entry.URL}</a>.`);
      } else {
        parts.push(`${entry.URL}.`);
      }
    }

    return parts.join(' ');
  }

  /**
   * Format authors in Chicago style
   */
  formatAuthorsChicago(authors) {
    if (authors.length === 0) return '';

    const formatAuthor = (author, isFirst) => {
      if (author.literal) return author.literal;
      if (isFirst) {
        // First author: Last, First
        return `${author.family}, ${author.given || ''}`;
      } else {
        // Other authors: First Last
        return `${author.given || ''} ${author.family}`;
      }
    };

    if (authors.length === 1) {
      return formatAuthor(authors[0], true);
    } else if (authors.length === 2) {
      return `${formatAuthor(authors[0], true)}, and ${formatAuthor(authors[1], false)}`;
    } else if (authors.length === 3) {
      return `${formatAuthor(authors[0], true)}, ${formatAuthor(authors[1], false)}, and ${formatAuthor(authors[2], false)}`;
    } else {
      // More than 3 authors: use "et al."
      return `${formatAuthor(authors[0], true)} et al.`;
    }
  }

  /**
   * Format single entry in Harvard style
   */
  formatHarvard(entry, format) {
    // Harvard is very similar to APA
    return this.formatAPA(entry, format);
  }

  /**
   * Format single entry in Vancouver style (numbered references)
   */
  formatVancouver(entry, format) {
    const parts = [];
    const isHtml = format === 'html';

    // Authors (last name + initials, up to 6 authors)
    if (entry.author && entry.author.length > 0) {
      const authorStr = this.formatAuthorsVancouver(entry.author);
      parts.push(authorStr + '.');
    }

    // Title
    if (entry.title) {
      parts.push(entry.title + '.');
    }

    // Journal (abbreviated if possible)
    if (entry['container-title']) {
      const journal = this.formatContainer(entry['container-title'], isHtml);
      let journalPart = journal;

      // Year
      if (entry.issued && entry.issued['date-parts']) {
        const year = entry.issued['date-parts'][0][0];
        journalPart += ` ${year}`;
      }

      // Volume and issue
      if (entry.volume) {
        journalPart += `;${entry.volume}`;
        if (entry.issue) {
          journalPart += `(${entry.issue})`;
        }
      }

      // Pages
      if (entry.page) {
        journalPart += `:${entry.page}`;
      }

      parts.push(journalPart + '.');
    }

    // DOI
    if (entry.DOI) {
      const doi = Array.isArray(entry.DOI) ? entry.DOI[0] : entry.DOI;
      parts.push(`doi: ${doi}`);
    }

    return parts.join(' ');
  }

  /**
   * Format authors in Vancouver style
   */
  formatAuthorsVancouver(authors) {
    if (authors.length === 0) return '';

    const formatAuthor = (author) => {
      if (author.literal) return author.literal;
      const initials = author.given ? author.given.split(' ').map(n => n.charAt(0)).join('') : '';
      return `${author.family} ${initials}`;
    };

    if (authors.length <= 6) {
      return authors.map(formatAuthor).join(', ');
    } else {
      // More than 6: list first 6 then "et al."
      const first6 = authors.slice(0, 6).map(formatAuthor).join(', ');
      return `${first6}, et al`;
    }
  }

  /**
   * Format single entry in IEEE style
   */
  formatIEEE(entry, format) {
    const parts = [];
    const isHtml = format === 'html';

    // Authors (initials + last name)
    if (entry.author && entry.author.length > 0) {
      const authorStr = this.formatAuthorsIEEE(entry.author);
      parts.push(authorStr + ',');
    }

    // Title in quotes
    if (entry.title) {
      parts.push(`"${entry.title},"`);
    }

    // Journal in italics
    if (entry['container-title']) {
      const journal = this.formatContainer(entry['container-title'], isHtml);
      let journalPart = journal;

      // Volume
      if (entry.volume) {
        journalPart += `, vol. ${entry.volume}`;
      }

      // Issue
      if (entry.issue) {
        journalPart += `, no. ${entry.issue}`;
      }

      // Pages
      if (entry.page) {
        journalPart += `, pp. ${entry.page}`;
      }

      // Year
      if (entry.issued && entry.issued['date-parts']) {
        const year = entry.issued['date-parts'][0][0];
        journalPart += `, ${year}`;
      }

      parts.push(journalPart + '.');
    }

    // DOI
    if (entry.DOI) {
      const doi = Array.isArray(entry.DOI) ? entry.DOI[0] : entry.DOI;
      parts.push(`doi: ${doi}.`);
    }

    return parts.join(' ');
  }

  /**
   * Format authors in IEEE style
   */
  formatAuthorsIEEE(authors) {
    if (authors.length === 0) return '';

    const formatAuthor = (author) => {
      if (author.literal) return author.literal;
      const initials = author.given ? author.given.split(' ').map(n => `${n.charAt(0)}.`).join(' ') : '';
      return `${initials} ${author.family}`;
    };

    return authors.map(formatAuthor).join(', ');
  }

  /**
   * Format as BibTeX
   */
  formatBibtex() {
    return this.data.map(entry => {
      const type = entry.type || 'article';
      const id = entry.id || 'entry1';

      let bibtex = `@${type}{${id},\n`;

      if (entry.author) {
        const authors = entry.author.map(a =>
          a.literal || `${a.family}, ${a.given || ''}`
        ).join(' and ');
        bibtex += `  author = {${authors}},\n`;
      }

      if (entry.title) bibtex += `  title = {${entry.title}},\n`;
      if (entry['container-title']) bibtex += `  journal = {${entry['container-title']}},\n`;
      if (entry.volume) bibtex += `  volume = {${entry.volume}},\n`;
      if (entry.issue) bibtex += `  number = {${entry.issue}},\n`;
      if (entry.page) bibtex += `  pages = {${entry.page}},\n`;
      if (entry.issued) bibtex += `  year = {${entry.issued['date-parts'][0][0]}},\n`;
      if (entry.DOI) {
        const doi = Array.isArray(entry.DOI) ? entry.DOI[0] : entry.DOI;
        bibtex += `  doi = {${doi}},\n`;
      }
      if (entry.URL) bibtex += `  url = {${entry.URL}},\n`;

      bibtex += '}';
      return bibtex;
    }).join('\n\n');
  }

  /**
   * Format as RIS
   */
  formatRis() {
    return this.data.map(entry => {
      let ris = 'TY  - JOUR\n';

      if (entry.author) {
        entry.author.forEach(a => {
          const name = a.literal || `${a.family}, ${a.given || ''}`;
          ris += `AU  - ${name}\n`;
        });
      }

      if (entry.title) ris += `TI  - ${entry.title}\n`;
      if (entry['container-title']) ris += `JO  - ${entry['container-title']}\n`;
      if (entry.volume) ris += `VL  - ${entry.volume}\n`;
      if (entry.issue) ris += `IS  - ${entry.issue}\n`;
      if (entry.page) ris += `SP  - ${entry.page}\n`;
      if (entry.issued) ris += `PY  - ${entry.issued['date-parts'][0][0]}\n`;
      if (entry.DOI) {
        const doi = Array.isArray(entry.DOI) ? entry.DOI[0] : entry.DOI;
        ris += `DO  - ${doi}\n`;
      }
      if (entry.URL) ris += `UR  - ${entry.URL}\n`;

      ris += 'ER  - \n';
      return ris;
    }).join('\n');
  }
}
