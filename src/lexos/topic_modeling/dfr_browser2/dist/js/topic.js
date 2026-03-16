// Topic view - Shows detailed information about a specific topic
import { extractTopicWords } from './state-utils.js';
import { getTopicLabel } from './topic-config.js';

// Store current topic data for filtering
let currentTopicData = null;
let currentSelectedYear = null;

export async function loadTopicView(topicKeys, docTopic, metadata, topicId, docTopicCounts) {
  const mainView = document.getElementById('main-view');

  // Check if data is loaded
  if (!topicKeys || !docTopic || !metadata) {
    mainView.innerHTML = `
      <div class="alert alert-info">
        <h4>No Data Loaded</h4>
        <p>Please load data to view topic details.</p>
      </div>
    `;
    return;
  }

  if (topicId === undefined || topicId < 0 || topicId >= topicKeys.length) {
    mainView.innerHTML = `
      <div class="alert alert-warning">
        <h4>Invalid Topic</h4>
        <p>Topic ${topicId} does not exist.</p>
      </div>
    `;
    return;
  }

// Global configuration object
let appConfig = null;

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

  // Get user settings
  const settings = await window.getSettings();
  const wordsCount = settings.wordsInTopic || 50;

  // Load bibliography data for formatted citations
  let bibliographyData = null;
  try {
    const config = appConfig || { bibliography: { path: 'data/bibliography.json' } };
    const bibliographyPath = config?.bibliography?.path || 'data/bibliography.json';
    const fullPath = bibliographyPath.startsWith('/') ? bibliographyPath : '/' + bibliographyPath;
    const response = await fetch(fullPath);
    if (response.ok) {
      bibliographyData = await response.json();
      // Add _docIndex to each entry if not present (same as bibliography.js does)
      bibliographyData.forEach((doc, index) => {
        if (doc._docIndex === undefined) {
          doc._docIndex = index;
        }
      });
    }
  } catch (error) {
    console.log('[Topic] Bibliography not available, will use metadata citations');
  }

  // Try to get enhanced word list from state file
  const enhancedWordLists = await extractTopicWords(topicKeys.length, wordsCount);
  const topic = topicKeys[topicId];

  // Use enhanced words if available, otherwise fall back to original
  const topicWords = (enhancedWordLists && enhancedWordLists[topicId] && enhancedWordLists[topicId].length > 0)
    ? enhancedWordLists[topicId]
    : (topic.words || []);

  // Store current topic data globally for filtering
  currentTopicData = {
    docTopic,
    metadata,
    topicNumber: topicId,
    docTopicCounts,
    settings,
    bibliographyData
  };

  // Reset any existing year filter
  currentSelectedYear = null;

  // Calculate topic proportions over time
  const timeSeriesData = calculateTopicTimeSeries(docTopic, metadata, topicId);

  let html = `
    <div class="container-fluid">
      <div class="row">
        <div class="col-12">
          <h2>${getTopicLabel(topicId)} Analysis</h2>
          <!-- Topic Statistics Section -->
          <div class="mb-4">
            ${generateTopicStatsHTML(docTopic, topicId, timeSeriesData)}
          </div>
        </div>
      </div>

      <div class="row">
        <!-- Left Column: Word Distribution -->
        <div class="col-md-3 col-lg-4">
          <div class="card">
            <div class="card-header">
              <h5 class="mb-0">Topic Word Distribution</h5>
            </div>
            <div class="card-body p-0">
              ${generateWordDistributionHTML(topicWords, topicId, settings)}
            </div>
          </div>
        </div>

        <!-- Right Column: Time Series and Top Documents -->
        <div class="col-md-9 col-lg-8">
          <div class="card mb-4">
            <div class="card-header">
              <h5 class="mb-0">Topic Proportion Over Time</h5>
            </div>
            <div class="card-body">
              <div id="topic-timeseries-chart">
                ${generateTimeSeriesHTML(timeSeriesData, topicId)}
              </div>
              <div id="year-filter-controls" class="mt-2" style="display: none;">
                <div class="alert alert-info py-2">
                  <div class="d-flex justify-content-between align-items-center">
                    <span><strong>Year Filter Active:</strong> <span id="selected-year-display"></span></span>
                    <button class="btn btn-outline-secondary btn-sm" onclick="clearYearFilter()">
                      ✕ Clear Filter
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="card">
            <div class="card-header">
              <h5 class="mb-0">Top Documents</h5>
            </div>
            <div class="card-body" id="top-documents-container">
              ${generateTopDocumentsHTML(docTopic, metadata, topicId, docTopicCounts, settings, bibliographyData)}
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  mainView.innerHTML = html;
}

function generateWordDistributionHTML(topicWords, topicNumber, settings = {}) {
  if (!topicWords || topicWords.length === 0) {
    return `<p class="text-muted">No word data available for Topic ${topicNumber + 1}</p>`;
  }

  const wordsToShow = settings.wordsInTopic || 50;
  const displayWords = topicWords.slice(0, wordsToShow);
  const maxWeight = displayWords.length;

  let html = `
    <div class="topic-words-table">
      <p class="mb-3 px-3 pt-3">All ${displayWords.length} top-ranked words for this topic:</p>
      <div class="table-responsive" style="max-height: 600px; overflow-y: auto;">
        <table class="table table-sm table-striped table-hover mb-0">
          <thead class="table-dark">
            <tr>
              <th>Word</th>
              <th class="text-end">Weight</th>
            </tr>
          </thead>
          <tbody>
  `;

  displayWords.forEach((word, index) => {
    const weight = maxWeight - index;
    const barWidth = Math.round((weight / maxWeight) * 100);

    html += `
      <tr>
        <td>
          <a href="word/${encodeURIComponent(word)}" class="fw-bold text-primary word-link" title="Click to view word details">
            ${word}
          </a>
        </td>
        <td class="text-end">
          <div class="d-flex align-items-center justify-content-end">
            <small class="text-muted me-2">${weight}</small>
            <div style="
              width: 80px;
              height: 16px;
              background: #e9ecef;
              border-radius: 8px;
              overflow: hidden;
              border: 1px solid #dee2e6;
            ">
              <div style="
                height: 16px;
                width: ${barWidth}%;
                background: linear-gradient(90deg, #0d6efd 0%, #6ea8ff 100%);
                border-radius: 7px 0 0 7px;
              "></div>
            </div>
          </div>
        </td>
      </tr>
    `;
  });

  html += `
          </tbody>
        </table>
      </div>
      <div class="mt-3 px-3 pb-3">
        <small class="text-muted">
          <strong>Instructions:</strong> Words are ranked by importance within the topic.
          Click on any word to see its details.
        </small>
      </div>
    </div>
  `;

  return html;
}

function calculateTopicTimeSeries(docTopic, metadata, topicNumber) {
  const yearData = {};

  docTopic.forEach((docTopics, docIndex) => {
    if (docIndex < metadata.length) {
      const doc = metadata[docIndex];
      const year = doc.year;

      if (!year) return;

      if (!yearData[year]) {
        yearData[year] = {
          year: year,
          totalDocs: 0,
          topicSum: 0,
          avgProportion: 0
        };
      }

      yearData[year].totalDocs += 1;
      yearData[year].topicSum += (docTopics[topicNumber] || 0);
    }
  });

  const timeSeriesData = Object.values(yearData).map(yearInfo => {
    yearInfo.avgProportion = yearInfo.totalDocs > 0 ? yearInfo.topicSum / yearInfo.totalDocs : 0;
    return yearInfo;
  }).sort((a, b) => parseInt(a.year) - parseInt(b.year));

  return timeSeriesData;
}

function generateTimeSeriesHTML(timeSeriesData, topicNumber) {
  if (!timeSeriesData || timeSeriesData.length === 0) {
    return `<p class="text-muted">No temporal data available for Topic ${topicNumber + 1}</p>`;
  }

  const maxProportion = Math.max(...timeSeriesData.map(d => d.avgProportion));
  const chartHeight = 180;

  let html = `
    <div class="time-series-chart">
      <p class="mb-2">Average topic proportion by year (${timeSeriesData.length} years)</p>
      <div style="position: relative; min-height: ${chartHeight + 40}px; width: 95%; margin: 0 auto;">
  `;

  const availableWidth = 95;
  const barSpacing = 2;
  const leftMargin = 30;
  const rightMargin = 10;
  const chartAreaWidth = `calc(${availableWidth}% - ${leftMargin + rightMargin}px)`;
  const barWidthCalc = `calc((${chartAreaWidth} - ${(timeSeriesData.length - 1) * barSpacing}px) / ${timeSeriesData.length})`;

  timeSeriesData.forEach((dataPoint, index) => {
    const barHeight = maxProportion > 0 ? (dataPoint.avgProportion / maxProportion) * (chartHeight - 50) : 0;
    const leftPosCalc = `calc(${leftMargin}px + ${index} * (${barWidthCalc} + ${barSpacing}px))`;
    const bottomPos = 40;

    const clickHandler = `filterDocumentsByYear(${dataPoint.year}, ${topicNumber})`;
    const tooltip = `Year: ${dataPoint.year}, Avg: ${(dataPoint.avgProportion * 100).toFixed(2)}%`;

    html += `
      <div style="
        position: absolute;
        left: ${leftPosCalc};
        bottom: ${bottomPos}px;
        width: ${barWidthCalc};
        height: ${barHeight}px;
        background: linear-gradient(to top, #0d6efd, #6ea8ff);
        border-radius: 2px 2px 0 0;
        cursor: pointer;
        transition: opacity 0.2s;
      "
      onmouseover="this.style.opacity='0.8'"
      onmouseout="this.style.opacity='1'"
      onclick="${clickHandler}"
      title="${tooltip}">
      </div>

      <div style="
        position: absolute;
        left: calc(${leftPosCalc} + ${barWidthCalc} / 2);
        bottom: 5px;
        font-size: ${timeSeriesData.length > 20 ? '9px' : '10px'};
        color: #666;
        transform: translateX(-50%) ${timeSeriesData.length > 15 ? 'rotate(-45deg)' : 'rotate(0deg)'};
        transform-origin: center;
        white-space: nowrap;
      ">
        ${dataPoint.year}
      </div>
    `;
  });

  // Y-axis labels
  for (let i = 0; i <= 4; i++) {
    const value = (maxProportion * i) / 4;
    const yPos = chartHeight - 30 - (i * (chartHeight - 50) / 4);

    html += `
      <div style="
        position: absolute;
        left: 0px;
        top: ${yPos - 6}px;
        font-size: 9px;
        color: #666;
        width: 25px;
        text-align: right;
      ">
        ${(value * 100).toFixed(0)}%
      </div>
    `;
  }

  html += `
      </div>
      <div class="mt-2">
        <small class="text-muted">
          <strong>Instructions:</strong> Hover over bars to see details.
          <strong>Click bars to filter documents by year.</strong>
        </small>
      </div>
    </div>
  `;

  return html;
}

function generateTopicStatsHTML(docTopic, topicNumber, timeSeriesData) {
  let totalProportion = 0;
  let docsWithTopic = 0;
  let maxProportion = 0;

  docTopic.forEach(docTopics => {
    const proportion = docTopics[topicNumber] || 0;
    totalProportion += proportion;
    if (proportion > 0) docsWithTopic++;
    if (proportion > maxProportion) maxProportion = proportion;
  });

  const avgProportion = docTopic.length > 0 ? totalProportion / docTopic.length : 0;
  const coverage = docTopic.length > 0 ? (docsWithTopic / docTopic.length) * 100 : 0;

  const yearRange = timeSeriesData.length > 0 ?
    `${timeSeriesData[0].year} - ${timeSeriesData[timeSeriesData.length - 1].year}` :
    'No data';

  const peakYear = timeSeriesData.reduce((peak, current) =>
    current.avgProportion > peak.avgProportion ? current : peak,
    { year: 'None', avgProportion: 0 }
  );

  return `
    <div class="row">
      <div class="col-md-3">
        <div class="stat-card text-center p-2 border rounded">
          <h6 class="text-muted mb-1 small">Average Proportion</h6>
          <h5 class="text-primary mb-0">${(avgProportion * 100).toFixed(2)}%</h5>
          <small class="text-muted">All Years</small>
        </div>
      </div>
      <div class="col-md-3">
        <div class="stat-card text-center p-2 border rounded">
          <h6 class="text-muted mb-1 small">Document Coverage</h6>
          <h5 class="text-success mb-0">${coverage.toFixed(1)}%</h5>
          <small class="text-muted">${docsWithTopic} of ${docTopic.length} docs</small>
        </div>
      </div>
      <div class="col-md-3">
        <div class="stat-card text-center p-2 border rounded">
          <h6 class="text-muted mb-1 small">Peak Year</h6>
          <h5 class="text-warning mb-0">${peakYear.year}</h5>
          <small class="text-muted">${(peakYear.avgProportion * 100).toFixed(2)}% avg</small>
        </div>
      </div>
      <div class="col-md-3">
        <div class="stat-card text-center p-2 border rounded">
          <h6 class="text-muted mb-1 small">Time Range</h6>
          <h5 class="text-info mb-0">${timeSeriesData.length}</h5>
          <small class="text-muted">${yearRange}</small>
        </div>
      </div>
    </div>
  `;
}

function generateTopDocumentsHTML(docTopic, metadata, topicNumber, docTopicCounts = null, settings = {}, bibliographyData = null) {
  if (!docTopic || !metadata || docTopic.length === 0) {
    return `<p class="text-muted">No document data available for Topic ${topicNumber + 1}</p>`;
  }

  const documentProportions = [];

  docTopic.forEach((doc, docIndex) => {
    if (doc && doc.length > topicNumber) {
      const proportion = parseFloat(doc[topicNumber]) || 0;
      if (proportion > 0) {
        const metadataDoc = metadata[docIndex];
        if (metadataDoc) {
          // Get topic tokens from docTopicCounts if available
          if (docTopicCounts && docTopicCounts[docIndex] && docTopicCounts[docIndex][topicNumber] !== undefined) {
            const topicTokens = parseInt(docTopicCounts[docIndex][topicNumber]) || 0;

            // Get formatted citation from bibliography if available
            let citation = null;
            if (bibliographyData && bibliographyData[docIndex]) {
              // Direct array access since bibliography should be in same order as metadata
              citation = bibliographyData[docIndex]['formatted-citation'] || bibliographyData[docIndex]._formattedCitation;
            }

            // Fallback to metadata citation
            if (!citation) {
              citation = metadataDoc['formatted-citation'] ||
                        `${metadataDoc.title || 'Untitled'}. ${metadataDoc.author || 'Unknown author'}. ${metadataDoc.year || metadataDoc.pubdate || 'Unknown date'}.`;
            }

            documentProportions.push({
              docIndex,
              proportion,
              topicTokens,
              citation,
              id: metadataDoc.id || `doc_${docIndex}`
            });
          }
        }
      }
    }
  });

  documentProportions.sort((a, b) => b.proportion - a.proportion);
  const docsToShow = settings.topicDocs || 20;
  const topDocs = documentProportions.slice(0, docsToShow);

  if (topDocs.length === 0) {
    return `<p class="text-muted">No documents found with significant presence of Topic ${topicNumber + 1}. Make sure the state file is loaded for token counts.</p>`;
  }

  const totalTopTokens = topDocs.reduce((sum, doc) => sum + doc.topicTokens, 0);
  const minProportion = topDocs[topDocs.length - 1].proportion;
  const maxProportion = topDocs[0].proportion;
  const proportionRange = maxProportion - minProportion;

  let html = `
    <div class="top-documents">
      <p class="mb-2">Top ${topDocs.length} documents with highest topic proportion</p>
      <div class="mb-3">
        <small class="text-muted">
          <strong>Column Guide:</strong> <strong>Proportion</strong> bars show the share of each document devoted to this topic.
          <strong>%</strong> shows the same proportion as a percentage value.
          <strong>Tokens</strong> is the topic token count for each document. Click rows for document details.
        </small>
      </div>

      <div class="table-responsive">
        <table class="table table-sm table-hover">
          <thead class="table-dark">
            <tr>
              <th style="width: 55%;">Document</th>
              <th style="width: 20%;" title="Visual representation of topic strength in document">Proportion</th>
              <th style="width: 10%; text-align: right;" title="Percentage of document devoted to this topic">%</th>
              <th style="width: 15%; text-align: right;" title="Number of topic-related tokens">Tokens</th>
            </tr>
          </thead>
          <tbody>
  `;

  topDocs.forEach((doc) => {
    // Use absolute proportion value (0-100%) for both bars and % column
    const barWidth = Math.round(doc.proportion * 100);
    const proportionPercent = (doc.proportion * 100).toFixed(1);

    html += `
      <tr style="cursor: pointer;"
          onclick="window.page('/document/${doc.docIndex}')"
          title="Click to view document details">
        <td>
          <div style="max-width: 500px;">
            <small>${doc.citation}</small>
          </div>
        </td>
        <td>
          <div style="
            width: 100%;
            height: 16px;
            background: #e9ecef;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #dee2e6;
          ">
            <div style="
              height: 16px;
              width: ${barWidth}%;
              background: linear-gradient(90deg, #28a745 0%, #6fcf7f 100%);
              border-radius: 7px 0 0 7px;
            "></div>
          </div>
        </td>
        <td style="text-align: right;">
          <span class="text-muted">${proportionPercent}%</span>
        </td>
        <td style="text-align: right;">
          <span class="fw-bold text-success">${doc.topicTokens.toLocaleString()}</span>
        </td>
      </tr>
    `;
  });

  html += `
          </tbody>
        </table>
      </div>
    </div>
  `;

  return html;
}

// Global function to handle document navigation
window.navigateToDocument = function(docId, citation, docIndex) {
  // Navigate using docIndex as URL parameter for permalink support
  window.page(`/document/${docIndex}`);
};

// Global function to handle word navigation
window.navigateToWord = function(word) {
  window.page(`/word/${encodeURIComponent(word)}`);
};

// Global function to filter documents by year
window.filterDocumentsByYear = function(year, topicNumber) {
  if (!currentTopicData) return;

  currentSelectedYear = year;

  const { docTopic, metadata, docTopicCounts, settings, bibliographyData } = currentTopicData;

  // Filter documents by year
  const filteredDocs = metadata
    .map((doc, idx) => ({ doc, idx }))
    .filter(({ doc }) => doc.year === year);

  // Update the display
  const container = document.getElementById('top-documents-container');
  if (container) {
    container.innerHTML = generateFilteredDocumentsHTML(
      docTopic,
      filteredDocs,
      topicNumber,
      docTopicCounts,
      settings,
      year,
      bibliographyData
    );
  }

  // Show year filter controls
  const filterControls = document.getElementById('year-filter-controls');
  const yearDisplay = document.getElementById('selected-year-display');
  if (filterControls && yearDisplay) {
    yearDisplay.textContent = year;
    filterControls.style.display = 'block';
  }
};

// Global function to clear year filter
window.clearYearFilter = function() {
  if (!currentTopicData) return;

  currentSelectedYear = null;

  const { docTopic, metadata, topicNumber, docTopicCounts, settings, bibliographyData } = currentTopicData;

  // Restore full document list
  const container = document.getElementById('top-documents-container');
  if (container) {
    container.innerHTML = generateTopDocumentsHTML(
      docTopic,
      metadata,
      topicNumber,
      docTopicCounts,
      settings,
      bibliographyData
    );
  }

  // Hide year filter controls
  const filterControls = document.getElementById('year-filter-controls');
  if (filterControls) {
    filterControls.style.display = 'none';
  }
};

function generateFilteredDocumentsHTML(docTopic, filteredDocs, topicNumber, docTopicCounts, settings, year, bibliographyData = null) {
  const documentProportions = [];

  filteredDocs.forEach(({ doc: metadataDoc, idx: docIndex }) => {
    const docTopics = docTopic[docIndex];
    if (docTopics && docTopics.length > topicNumber) {
      const proportion = parseFloat(docTopics[topicNumber]) || 0;
      if (proportion > 0) {
        // Get topic tokens from docTopicCounts if available
        if (docTopicCounts && docTopicCounts[docIndex] && docTopicCounts[docIndex][topicNumber] !== undefined) {
          const topicTokens = parseInt(docTopicCounts[docIndex][topicNumber]) || 0;

          // Get formatted citation from bibliography if available
          let citation = null;
          if (bibliographyData && bibliographyData[docIndex]) {
            // Direct array access since bibliography should be in same order as metadata
            citation = bibliographyData[docIndex]['formatted-citation'] || bibliographyData[docIndex]._formattedCitation;
          }

          // Fallback to metadata citation
          if (!citation) {
            citation = metadataDoc['formatted-citation'] ||
                      `${metadataDoc.title || 'Untitled'}. ${metadataDoc.author || 'Unknown author'}. ${metadataDoc.year || 'Unknown date'}.`;
          }

          documentProportions.push({
            docIndex,
            proportion,
            topicTokens,
            citation,
            id: metadataDoc.id || `doc_${docIndex}`
          });
        }
      }
    }
  });

  documentProportions.sort((a, b) => b.proportion - a.proportion);
  const docsToShow = settings.topicDocs || 20;
  const topDocs = documentProportions.slice(0, docsToShow);

  if (topDocs.length === 0) {
    return `<p class="text-muted">No documents found for year ${year} with Topic ${topicNumber + 1}</p>`;
  }

  const totalTopTokens = topDocs.reduce((sum, doc) => sum + doc.topicTokens, 0);
  const minProportion = topDocs[topDocs.length - 1].proportion;
  const maxProportion = topDocs[0].proportion;
  const proportionRange = maxProportion - minProportion;

  let html = `
    <div class="top-documents">
      <p class="mb-2">Top ${topDocs.length} documents from ${year} with highest topic proportion</p>
      <div class="table-responsive">
        <table class="table table-sm table-hover">
          <thead class="table-dark">
            <tr>
              <th style="width: 55%;">Document</th>
              <th style="width: 20%;">Proportion</th>
              <th style="width: 10%; text-align: right;">%</th>
              <th style="width: 15%; text-align: right;">Tokens</th>
            </tr>
          </thead>
          <tbody>
  `;

  topDocs.forEach((doc) => {
    // Use absolute proportion value (0-100%) for both bars and % column
    const barWidth = Math.round(doc.proportion * 100);
    const proportionPercent = (doc.proportion * 100).toFixed(1);

    html += `
      <tr style="cursor: pointer;"
          onclick="window.page('/document/${doc.docIndex}')"
          title="Click to view document details">
        <td>
          <div style="max-width: 500px;">
            <small>${doc.citation}</small>
          </div>
        </td>
        <td>
          <div style="
            width: 100%;
            height: 16px;
            background: #e9ecef;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #dee2e6;
          ">
            <div style="
              height: 16px;
              width: ${barWidth}%;
              background: linear-gradient(90deg, #28a745 0%, #6fcf7f 100%);
              border-radius: 7px 0 0 7px;
            "></div>
          </div>
        </td>
        <td style="text-align: right;">
          <span class="text-muted">${proportionPercent}%</span>
        </td>
        <td style="text-align: right;">
          <span class="fw-bold text-success">${doc.topicTokens.toLocaleString()}</span>
        </td>
      </tr>
    `;
  });

  html += `
          </tbody>
        </table>
      </div>
    </div>
  `;

  return html;
}
