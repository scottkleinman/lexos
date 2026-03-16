# Citation System - Implementation Guide

## Overview

The DFR Browser includes a **custom citation formatting library** that provides professional citation formatting in multiple academic styles without external dependencies. This lightweight implementation supports 6 major citation styles with the ability to easily add more.

## Architecture

### Custom Implementation (No External Dependencies)

We use a **local citation library** (`/dist/js/lib/citation.js`) instead of relying on external CDNs. This approach provides:

- **Fast loading**: No network delays
- **Offline support**: Works without internet connection
- **No external dependencies**: Everything is self-contained
- **Full control**: Easy to customize and extend
- **Reliability**: No CDN outages or version changes

## Current Citation Styles

The system supports **6 citation styles**:

1. **APA 7th Edition** - Psychology, Education, Social Sciences
2. **MLA 9th Edition** - Literature, Humanities
3. **Chicago (Author-Date)** - History, Arts, Humanities
4. **Harvard** - General academic (UK/Australia)
5. **Vancouver** - Medicine, Life Sciences
6. **IEEE** - Engineering, Computer Science

## File Structure

```txt
dist/
├── js/
│   ├── citation.js              # Citation view controller
│   └── lib/
│       ├── citation.js          # Core citation formatting library (ACTIVE)
│       └── citation-full.js     # Extended reference implementation (for future use)
├── css/
│   └── style.css                # Citation view styling
└── docs/
    └── CITATION_JS_INTEGRATION.md  # This file
```

## How It Works

### 1. Citation Data Loading

The citation view (`/dist/js/citation.js`):

1. Loads bibliography data from `bibliography.json` (CSL JSON format)
2. Falls back to metadata if bibliography not available
3. Converts metadata to CSL JSON format if needed

### 2. Citation Generation

```javascript
import { Cite, CITATION_STYLES } from './lib/citation.js';

// Create citation instance
const cite = new Cite(cslData);

// Generate all citation formats
for (const style of CITATION_STYLES) {
  citationFormats[style.value] = {
    html: await cite.format('bibliography', {
      format: 'html',
      template: style.value
    }),
    text: await cite.format('bibliography', {
      format: 'text',
      template: style.value
    })
  };
}
```

### 3. Display and Interaction

- **Dropdown menu**: Lists all available citation styles
- **Instant switching**: Pre-generated formats allow instant style changes
- **Copy to clipboard**: Copies plain text version of current style
- **Export formats**: BibTeX, RIS, and CSL JSON available

## Adding New Citation Styles

### Step 1: Add to CITATION_STYLES Array

Edit `/dist/js/lib/citation.js`:

```javascript
export const CITATION_STYLES = [
  { value: 'chicago', label: 'Chicago (Author-Date)' },
  { value: 'chicago-note', label: 'Chicago Manual of Style (Notes)' },
  { value: 'mla', label: 'MLA 9th Edition' },
  { value: 'apa', label: 'APA 7th Edition' },
  { value: 'harvard', label: 'Harvard' },
  { value: 'vancouver', label: 'Vancouver' },
  { value: 'ieee', label: 'IEEE' },
  // Add your new style here:
  { value: 'ams', label: 'AMS (American Mathematical Society)' }
];
```

A good source of citation styles is the [Zotero Style Repository](https://www.zotero.org/styles).

### Step 2: Add Formatting Method

Add a new formatting method in `/dist/js/lib/citation.js`:

```javascript
/**
 * Format single entry in AMS style
 */
formatAMS(entry, format) {
  const parts = [];
  const isHtml = format === 'html';

  // Authors (First Last format)
  if (entry.author && entry.author.length > 0) {
    const authorStr = this.formatAuthorsAMS(entry.author);
    parts.push(authorStr + ',');
  }

  // Title in italics
  if (entry.title) {
    const title = isHtml ? `<em>${entry.title}</em>` : entry.title;
    parts.push(title + ',');
  }

  // Journal
  if (entry['container-title']) {
    const journal = this.formatContainer(entry['container-title'], isHtml);
    parts.push(journal);
  }

  // Volume and issue
  if (entry.volume) {
    let volPart = entry.volume;
    if (entry.issue) {
      volPart += `(${entry.issue})`;
    }
    parts.push(volPart);
  }

  // Year and pages
  if (entry.issued && entry.issued['date-parts']) {
    const year = entry.issued['date-parts'][0][0];
    let yearPart = `(${year})`;
    if (entry.page) {
      yearPart += `, ${entry.page}`;
    }
    parts.push(yearPart + '.');
  }

  return parts.join(' ');
}

/**
 * Format authors in AMS style
 */
formatAuthorsAMS(authors) {
  if (authors.length === 0) return '';

  const formatAuthor = (author) => {
    if (author.literal) return author.literal;
    return `${author.given || ''} ${author.family}`;
  };

  return authors.map(formatAuthor).join(', ');
}
```

### Step 3: Add to formatBibliography Switch

Update the `formatBibliography()` method to include your new style:

```javascript
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
      case 'ams':  // Add your new style here
        return this.formatAMS(entry, format);
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
```

### Step 4: Test Your New Style

1. Reload the browser
2. Navigate to any citation view
3. Select your new style from the dropdown
4. Verify the formatting matches the style guide
5. Test the "Copy to Clipboard" function

## Citation Style Guidelines

When implementing a new citation style, consider these elements:

### Author Formatting

- **APA/Chicago**: Last, F. M.
- **MLA**: Last, First Middle
- **Vancouver**: Last FM
- **IEEE**: F. M. Last

### Title Formatting

- **APA**: Sentence case, no quotes
- **MLA**: Title Case in "Quotes"
- **Chicago**: Title Case in "Quotes"
- **Vancouver**: Sentence case, no quotes

### Journal Formatting

- **APA/MLA/Chicago**: *Italicized*
- **Vancouver**: Abbreviated, not italicized
- **IEEE**: *Italicized*, abbreviated

### Date/Volume/Issue

- **APA**: (2024). *Journal*, *volume*(issue), pages.
- **MLA**: *Journal* vol. X, no. Y, 2024, pp. XX-XX.
- **Chicago**: *Journal* X, no. Y (2024): XX-XX.
- **Vancouver**: Journal. 2024;X(Y):XX-XX.

## Helper Methods Available

The `Cite` class provides several helper methods:

### `stripHtml(text)`

Removes HTML tags from text (used for plain text output):

```javascript
this.stripHtml('<em>Journal Name</em>'); // Returns: "Journal Name"
```

### `formatContainer(containerTitle, isHtml)`

Handles journal/container titles that may already contain HTML:

```javascript
// If title already has <em> tags, uses as-is for HTML or strips for text
this.formatContainer('<em>Nature</em>', true);  // '<em>Nature</em>'
this.formatContainer('<em>Nature</em>', false); // 'Nature'

// If plain text, adds italics for HTML
this.formatContainer('Nature', true);  // '<em>Nature</em>'
this.formatContainer('Nature', false); // 'Nature'
```

### Author Formatting Methods

- `formatAuthorsAPA(authors)` - APA style (Last, F. M.)
- `formatAuthorsMLA(authors)` - MLA style (Last, First)
- `formatAuthorsChicago(authors)` - Chicago style (with et al. for 4+)
- `formatAuthorsVancouver(authors)` - Vancouver style (Last FM, up to 6)
- `formatAuthorsIEEE(authors)` - IEEE style (F. M. Last)

## Export Formats

The library supports multiple export formats beyond formatted bibliographies:

### BibTeX

```javascript
const bibtex = await cite.format('bibtex');
// Returns: @article{doc-0, author = {...}, ...}
```

### RIS

```javascript
const ris = await cite.format('ris');
// Returns: TY  - JOUR\nAU  - ...\n...
```

### CSL JSON

```javascript
const cslJson = JSON.stringify(cslData, null, 2);
// Returns: Structured CSL JSON
```

## CSS Styling

Citation view styling in `/dist/css/style.css`:

```css
/* Citation style selector dropdown */
#citation-style-select {
  min-width: 100px;
  color: var(--color-neutral-900) !important;
  background-color: white !important;
}

/* Citation display area */
.citation-display {
  padding: 1rem;
  background-color: var(--color-neutral-50);
  border: 1px solid var(--color-neutral-300);
  border-radius: 0.375rem;
}
```

## Reference: CSL JSON Format

The library expects CSL JSON format for input:

```json
{
  "id": "001244776600001",
  "type": "article-journal",
  "author": [
    {
      "family": "Hermann",
      "given": "Erik"
    }
  ],
  "issued": {
    "date-parts": [[2024]]
  },
  "title": "Article Title",
  "container-title": "<em>Journal Name</em>",
  "volume": "180",
  "issue": "2",
  "page": "10-20",
  "DOI": ["10.1016/j.example.2024.12345"]
}
```

## Advanced: Loading External CSL Styles

For advanced users, the `/dist/js/lib/citation-full.js` file includes a reference implementation for loading CSL styles from external sources:

```javascript
import { loadCSLStyle } from './lib/citation-full.js';

// Load from CSL repository
await loadCSLStyle('american-sociological-association');
```

This feature is available but not currently implemented in the main citation view. It could be added for users who want to use official CSL templates from the [CSL Style Repository](https://github.com/citation-style-language/styles).

## Testing

To verify the citation system:

1. Navigate to any document's Citation view
2. Dropdown should show 6 citation styles
3. Switching styles should instantly update the citation
4. "Copy to Clipboard" should copy the plain text version
5. Export formats (BibTeX, RIS, CSL JSON) should work
6. All text should be clearly visible

## Troubleshooting

**If citations don't display:**

- Check browser console for JavaScript errors
- Verify `bibliography.json` exists and has valid CSL JSON
- Check that CSL JSON has required fields (author, title, etc.)

**If a style looks incorrect:**

- Compare with official style guide
- Check the formatting method for that style
- Verify CSL JSON has all necessary fields (volume, issue, pages, etc.)

**If switching styles doesn't work:**

- Check browser console for errors
- Verify all styles are in `CITATION_STYLES` array
- Ensure event listeners are properly attached

## Resources

- **CSL Specification**: <https://citationstyles.org/>
- **CSL JSON Schema**: <https://github.com/citation-style-language/schema>
- **CSL Style Repository**: <https://github.com/citation-style-language/styles>
- **CSL Editor**: <https://editor.citationstyles.org/>
