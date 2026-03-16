// Overview view - Grid, Scaled, List, and Stacked visualizations
import { getTopicLabel, isTopicHidden, filterHiddenTopics, getVisibleTopicIndices } from './topic-config.js';

// Store data globally for this module
let overviewData = null;
let currentMode = 'grid';
let resizeTimeout = null;

export async function loadOverview(topicKeys, docTopic, metadata, topicCoords, docTopicCounts) {
  const mainView = document.getElementById('main-view');

  // Check if data is loaded
  if (!topicKeys || !docTopic || !metadata) {
    mainView.innerHTML = `
      <div class="alert alert-info">
        <h4>No Data Loaded</h4>
        <p>Please load data to view the overview.</p>
      </div>
    `;
    return;
  }

  console.log('[Overview] Rendering with data:', {
    topics: topicKeys.length,
    documents: docTopic.length,
    metadata: metadata.length
  });

  // Get user settings (keep spinner visible while this happens)
  const settings = await window.getSettings();
  const wordsInCircles = settings.wordsInCircles || 6;

  // Store data globally for navigation
  overviewData = {
    topicKeys,
    docTopic,
    metadata,
    topicCoords,
    docTopicCounts,
    settings
  };

  // NOW render the card with the mode navbar and a container for the visualization
  // This replaces the spinner from the route handler
  mainView.innerHTML = `
    <div class="card">
      <div class="card-body">
        <nav class="navbar navbar-expand-lg navbar-light bg-light mb-3">
          <div class="container-fluid">
            <ul class="navbar-nav me-auto mb-2 mb-lg-0">
              <li class="nav-item"><a class="nav-link mode-link active" href="#" id="mode-grid">Grid</a></li>
              <li class="nav-item"><a class="nav-link mode-link" href="#" id="mode-scaled">Scaled</a></li>
              <li class="nav-item"><a class="nav-link mode-link" href="#" id="mode-list">List</a></li>
              <li class="nav-item"><a class="nav-link mode-link" href="#" id="mode-stacked">Stacked</a></li>
              <li class="nav-item"><a class="nav-link mode-link" href="${window.dfrBasePath || ''}/diagnostics">Diagnostics</a></li>
            </ul>
            <span class="navbar-text ms-auto" id="overview-navbar-message">Click a circle for more about a topic</span>
            <span id="conditional_view_help" class="navbar-text model_view_conditional" style="display:none; margin-left:1em;">y-axis:</span>
            <div id="conditional_choice" class="btn-group btn-group-sm model_view_conditional" role="group" aria-label="y-axis toggle" style="display:none; margin-left:0.5em;">
              <button type="button" id="nav_model_conditional_frac" class="btn btn-outline-primary active">%</button>
              <button type="button" id="nav_model_conditional_raw" class="btn btn-outline-primary">word counts</button>
            </div>
            <span id="reset-zoom-container" style="display:none; margin-left:1em;">
              <button class="btn btn-outline-secondary btn-sm" id="reset-zoom-btn">Reset Zoom</button>
            </span>
          </div>
        </nav>
        <div id="overview-vis"></div>
      </div>
    </div>
  `;

  // Set up event listeners
  document.getElementById('mode-grid').addEventListener('click', (e) => {
    e.preventDefault();
    currentMode = 'grid';
    setActiveMode('mode-grid');
    setNavbarMessage('Click a circle for more about a topic');
    hideResetZoom();
    hideConditionalControls();
    renderGridView(topicKeys, docTopic, wordsInCircles);
  });

  document.getElementById('mode-scaled').addEventListener('click', (e) => {
    e.preventDefault();
    currentMode = 'scaled';
    setActiveMode('mode-scaled');
    setNavbarMessage('Scroll to zoom; shift-drag to pan; click for more about a topic');
    hideConditionalControls();
    renderScaledView(topicKeys, topicCoords, docTopic, wordsInCircles);
  });

  document.getElementById('mode-list').addEventListener('click', (e) => {
    e.preventDefault();
    currentMode = 'list';
    setActiveMode('mode-list');
    setNavbarMessage('Click a column label to sort; click a row for more about a topic');
    hideResetZoom();
    hideConditionalControls();
    renderListView(topicKeys, docTopic, metadata, settings);
  });

  document.getElementById('mode-stacked').addEventListener('click', (e) => {
    e.preventDefault();
    currentMode = 'stacked';
    setActiveMode('mode-stacked');
    setNavbarMessage('Scroll to zoom; shift-drag to pan; click for more about a topic');

    // Show conditional controls if we have docTopicCounts
    if (docTopicCounts) {
      document.querySelectorAll('.model_view_conditional').forEach(el => {
        el.style.display = '';
      });
    }

    renderStackedView(topicKeys, docTopic, metadata, docTopicCounts);
  });

  // Reset zoom button
  document.getElementById('reset-zoom-btn').addEventListener('click', () => {
    if (typeof window.resetScaledZoom === 'function') {
      window.resetScaledZoom();
    }
  });

  // Handle window resize
  window.removeEventListener('resize', handleOverviewResize);
  window.addEventListener('resize', handleOverviewResize);

  // Render default grid view
  currentMode = 'grid';
  setActiveMode('mode-grid');
  renderGridView(topicKeys, docTopic, wordsInCircles);
}

// Handle resize with debouncing
function handleOverviewResize() {
  if (resizeTimeout) {
    clearTimeout(resizeTimeout);
  }

  resizeTimeout = setTimeout(() => {
    if (!overviewData) return;

    const { topicKeys, docTopic, metadata, topicCoords, docTopicCounts, settings } = overviewData;
    const wordsInCircles = settings.wordsInCircles || 6;

    // Re-render current mode
    switch (currentMode) {
      case 'grid':
        renderGridView(topicKeys, docTopic, wordsInCircles);
        break;
      case 'scaled':
        renderScaledView(topicKeys, topicCoords, docTopic, wordsInCircles);
        break;
      case 'stacked':
        renderStackedView(topicKeys, docTopic, metadata, docTopicCounts);
        break;
      case 'list':
        renderListView(topicKeys, docTopic, metadata, settings);
        break;
    }
  }, 250); // 250ms debounce
}

function setActiveMode(activeId) {
  document.querySelectorAll('.mode-link').forEach(link => link.classList.remove('active'));
  const activeLink = document.getElementById(activeId);
  if (activeLink) activeLink.classList.add('active');
}

function setNavbarMessage(msg) {
  const msgElement = document.getElementById('overview-navbar-message');
  if (msgElement) msgElement.textContent = msg;
}

function hideResetZoom() {
  const resetContainer = document.getElementById('reset-zoom-container');
  if (resetContainer) resetContainer.style.display = 'none';
}

function hideConditionalControls() {
  document.querySelectorAll('.model_view_conditional').forEach(el => {
    el.style.display = 'none';
  });
}

// Grid View - Hexagonal grid with topic circles and words
function renderGridView(topicKeys, docTopic, wordsInCircles) {
  const container = document.getElementById('overview-vis');
  container.innerHTML = '';
  const width = container.offsetWidth || 800;

  // Calculate grid dimensions
  const n = topicKeys.length;
  const cols = Math.ceil(Math.sqrt(n));
  const rows = Math.ceil(n / cols);

  // Calculate proper sizing
  const cellWidth = width / (cols + 0.5); // Account for stagger offset
  const baseCircleSize = cellWidth * 0.5;
  const circleRadius = baseCircleSize / 1.33;

  // Calculate height based on rows and circle size
  const rowSpacing = circleRadius * 2.0;
  const topPadding = circleRadius * 1.0 + 15;
  const bottomPadding = circleRadius * 0.2;
  const calculatedHeight = (rows * rowSpacing) + topPadding + bottomPadding;

  const svg = d3.select(container)
    .append('svg')
    .attr('width', width)
    .attr('height', calculatedHeight);

  // Calculate topic prominence for stroke width
  let topicProminence = [];
  if (docTopic) {
    const topicCounts = Array(topicKeys.length).fill(0);
    const topicSums = Array(topicKeys.length).fill(0);
    docTopic.forEach((docT) => {
      if (!docT) return;
      docT.forEach((proportion, topicIdx) => {
        topicSums[topicIdx] += proportion;
        topicCounts[topicIdx] += 1;
      });
    });
    topicProminence = topicSums.map((sum, i) =>
      topicCounts[i] > 0 ? sum / topicCounts[i] : 0
    );

    // Normalize to 0-1 range with power scaling
    const minProminence = Math.min(...topicProminence);
    const maxProminence = Math.max(...topicProminence);
    const range = maxProminence - minProminence;

    topicProminence = topicProminence.map(p => {
      const normalized = range > 0 ? (p - minProminence) / range : 0;
      return Math.pow(normalized, 0.5); // Square root for more spread
    });
  } else {
    topicProminence = Array(topicKeys.length).fill(0.5);
  }

  const cellSize = baseCircleSize;

  topicKeys.forEach((tk, i) => {
    // Skip hidden topics
    if (isTopicHidden(i)) return;

    const col = i % cols;
    const row = Math.floor(i / cols);

    // Stagger alternate rows by half a cell (hexagonal grid pattern)
    const horizontalOffset = (row % 2 === 0) ? 0 : 0.5;
    const cx = (col + 0.5 + horizontalOffset) * cellWidth;
    const cy = (row * rowSpacing) + topPadding;

    // Calculate stroke width based on prominence
    const strokeWidth = 0.3 + (topicProminence[i] * 5.7);

    // Group for circle and words
    const g = svg.append('g')
      .attr('class', 'topic-circle-group')
      .attr('transform', `translate(${cx},${cy})`)
      .style('cursor', 'pointer');

    g.append('circle')
      .attr('r', cellSize / 1.33)
      .attr('fill', 'rgba(240, 240, 240, 0.6)')
      .attr('stroke', 'rgba(200, 200, 200, 0.6)')
      .attr('stroke-width', strokeWidth)
      .attr('data-topic', tk.topic);

    // Top words
    const words = tk.words.slice(0, wordsInCircles);
    const baseFontSize = Math.max(10, 24 - wordsInCircles);

    // Center-weighted arrangement
    const fontSizes = [];
    const yOffsets = [];

    if (words.length === 1) {
      fontSizes.push(baseFontSize);
      yOffsets.push(0);
    } else {
      // First word in center
      fontSizes.push(baseFontSize);
      yOffsets.push(0);

      // Remaining words alternately above and below
      const verticalSpacing = Math.min(18, 70 / Math.max(1, words.length - 1));

      for (let j = 1; j < words.length; j++) {
        const fontSize = Math.max(8, baseFontSize - (j * 1.5));
        fontSizes.push(fontSize);

        const isAbove = (j % 2 === 1);
        const level = Math.ceil(j / 2);
        const yOffset = isAbove ? -level * verticalSpacing : level * verticalSpacing;
        yOffsets.push(yOffset);
      }
    }

    words.forEach((word, idx) => {
      g.append('text')
        .attr('x', 0)
        .attr('y', yOffsets[idx])
        .attr('text-anchor', 'middle')
        .attr('font-size', fontSizes[idx])
        .attr('fill', 'rgba(0, 0, 0, 0.6)')
        .text(word);
    });

    // Hover overlay
    g.append('circle')
      .attr('r', cellSize / 1.33)
      .attr('fill', 'rgba(51, 153, 255, 0.9)')
      .attr('opacity', 0)
      .attr('class', 'topic-hover-rect');

    g.append('text')
      .attr('x', 0)
      .attr('y', 0)
      .attr('text-anchor', 'middle')
      .attr('font-size', 24)
      .attr('fill', '#000')
      .attr('class', 'topic-hover-label')
      .attr('opacity', 0)
      .text(getTopicLabel(tk.topic));

    g.on('mouseover', function() {
      g.select('.topic-hover-rect').attr('opacity', 0.85);
      g.select('.topic-hover-label').attr('opacity', 1);
    });

    g.on('mouseout', function() {
      g.select('.topic-hover-rect').attr('opacity', 0);
      g.select('.topic-hover-label').attr('opacity', 0);
    });

    g.on('click', function() {
      window.page(`/topic/${tk.topic + 1}`); // Use 1-based numbering in URL
    });
  });
}

// List View - Table of topics
function renderListView(topicKeys, docTopic, metadata, settings) {
  const container = document.getElementById('overview-vis');
  const wordsToShow = settings.wordsInLists || 15;

  // Precompute topic percents (average proportion across all documents)
  const topicCounts = Array(topicKeys.length).fill(0);
  const topicSums = Array(topicKeys.length).fill(0);
  docTopic.forEach((docT) => {
    if (!docT) return;
    docT.forEach((proportion, topicIdx) => {
      topicSums[topicIdx] += proportion;
      topicCounts[topicIdx] += 1;
    });
  });
  const topicPercents = topicSums.map((sum, i) =>
    topicCounts[i] > 0 ? sum / topicCounts[i] : 0
  );

  // Extract years from metadata
  const years = Array.from(new Set(metadata.map(m => m.year).filter(y => y))).sort();

  // Compute topic time-series (proportion per year)
  function computeTopicYearSeries() {
    const topicCount = topicKeys.length;
    const yearCount = years.length;
    const topicYearSums = Array.from({ length: topicCount }, () => Array(yearCount).fill(0));
    const topicYearCounts = Array(yearCount).fill(0);

    metadata.forEach((doc, docIdx) => {
      const yearIdx = years.indexOf(doc.year);
      if (yearIdx === -1) return;
      const docT = docTopic[docIdx];
      if (Array.isArray(docT)) {
        docT.forEach((proportion, topicIdx) => {
          if (topicIdx >= 0 && topicIdx < topicCount) {
            topicYearSums[topicIdx][yearIdx] += proportion;
          }
        });
        topicYearCounts[yearIdx] += 1;
      }
    });

    // Normalize by doc count per year
    return topicYearSums.map((arr) =>
      arr.map((sum, yearIdx) =>
        topicYearCounts[yearIdx] > 0 ? sum / topicYearCounts[yearIdx] : 0
      )
    );
  }

  const topicYearSeries = years.length > 0 ? computeTopicYearSeries() : [];

  // Sorting state
  let sortKey = 'topic';
  let sortAsc = true;

  function renderTable() {
    const headings = [
      { key: 'topic', label: 'Topic' },
      { key: 'overtime', label: 'Over Time' },
      { key: 'words', label: 'Top Words' },
      { key: 'percent', label: 'Proportion' }
    ];

    let sorted = topicKeys
      .filter((tk, i) => !isTopicHidden(i)) // Filter hidden topics
      .map((tk, originalIdx) => {
        // Need to get the original index for proper data lookup
        const i = tk.topic;
        const yearSeries = topicYearSeries[i] || [];
        const peakIdx = yearSeries.length > 0
          ? yearSeries.reduce((maxIdx, val, idx, arr) => (val > arr[maxIdx] ? idx : maxIdx), 0)
          : 0;

        return {
          topic: tk.topic,
          words: tk.words.slice(0, wordsToShow).join(', '),
          percent: topicPercents[i],
          idx: i,
          yearSeries,
          peakIdx
        };
      });

    // Sort data
    sorted.sort((a, b) => {
      if (sortKey === 'topic')
        return sortAsc ? a.topic - b.topic : b.topic - a.topic;
      if (sortKey === 'words')
        return sortAsc ? a.words.localeCompare(b.words) : b.words.localeCompare(a.words);
      if (sortKey === 'percent')
        return sortAsc ? a.percent - b.percent : b.percent - a.percent;
      if (sortKey === 'overtime')
        return sortAsc ? a.peakIdx - b.peakIdx : b.peakIdx - a.peakIdx;
      return 0;
    });

    let html = `<div class="table-responsive"><table class="table table-striped"><thead><tr>`;

    headings.forEach(h => {
      // Determine sort icon based on current sort state
      let sortIcon;
      if (sortKey === h.key) {
        sortIcon = sortAsc ? 'bi-sort-up' : 'bi-sort-down';
      } else {
        sortIcon = 'bi-sort-up';
      }

      let widthStyle = '';
      if (h.key === 'topic') {
        widthStyle = 'width: 100px; min-width: 100px;';
      } else if (h.key === 'overtime') {
        widthStyle = 'width: 140px; min-width: 140px;';
      } else if (h.key === 'percent') {
        widthStyle = 'width: 200px; min-width: 200px;';
      }

      html += `<th style="cursor:pointer; ${widthStyle}" id="sort-${h.key}">${h.label} <i class="bi ${sortIcon}"></i></th>`;
    });

    html += `</tr></thead><tbody>`;

    sorted.forEach(row => {
      html += `<tr style="cursor:pointer;" data-topic="${row.topic}">`;
      html += `<td>${getTopicLabel(row.topic)}</td>`;

      // Over Time D3 histogram placeholder
      html += `<td><div class="overtime-chart" data-topic="${row.idx}"></div></td>`;

      html += `<td>${row.words}</td>`;

      // Proportion bar
      const barWidth = Math.round(row.percent * 100 * 2);
      html += `<td><div style="display:inline-block;width:120px;height:16px;vertical-align:middle;background:#e9ecef;border-radius:8px;overflow:hidden;"><div style="height:16px;width:${barWidth}px;background:#0d6efd;"></div></div> <span style="margin-left:8px;">${(row.percent * 100).toFixed(1)}%</span></td>`;

      html += `</tr>`;
    });

    html += `</tbody></table></div>`;
    container.innerHTML = html;

    // Render D3 histograms for Over Time column
    sorted.forEach(row => {
      const sel = d3.select(`.overtime-chart[data-topic='${row.idx}']`);
      const svgW = 60, svgH = 18, barW = svgW / years.length;
      const svg = sel.append('svg').attr('width', svgW).attr('height', svgH);

      let series = Array.isArray(row.yearSeries) ? row.yearSeries : [];
      if (series.length !== years.length) series = Array(years.length).fill(0);

      let maxVal = Math.max(...series);
      if (!isFinite(maxVal) || maxVal <= 0) maxVal = 1;

      svg.selectAll('rect')
        .data(series)
        .enter()
        .append('rect')
        .attr('x', (d, j) => j * barW)
        .attr('y', d => svgH - Math.round((d / maxVal) * (svgH - 2)))
        .attr('width', barW)
        .attr('height', d => Math.round((d / maxVal) * (svgH - 2)))
        .attr('fill', '#111');
    });

    // Add sort handlers
    headings.forEach(h => {
      const sortHeader = document.getElementById(`sort-${h.key}`);
      if (sortHeader) {
        sortHeader.onclick = function() {
          if (sortKey === h.key) sortAsc = !sortAsc;
          else {
            sortKey = h.key;
            sortAsc = true;
          }
          renderTable();
        };
      }
    });

    // Add row click handlers
    const tableRows = container.querySelectorAll('tr[data-topic]');
    tableRows.forEach(row => {
      row.addEventListener('click', () => {
        const topicIdx = parseInt(row.getAttribute('data-topic'));
        window.page(`/topic/${topicIdx + 1}`); // Use 1-based numbering in URL
      });
    });
  }

  renderTable();
}

// Scaled View - Topics positioned using coordinates or computed layout
function renderScaledView(topicKeys, topicCoords, docTopic, wordsInCircles) {
  const container = document.getElementById('overview-vis');
  container.innerHTML = '';

  const width = container.offsetWidth || 800;
  const height = 500;

  // Calculate circle size to match Grid view
  const n = topicKeys.length;
  const cols = Math.ceil(Math.sqrt(n));
  const cellWidth = width / (cols + 0.5);
  const baseCircleSize = cellWidth * 0.5;
  const circleRadius = baseCircleSize / 1.33;

  // Calculate topic prominence for stroke width
  let topicProminence = [];
  if (docTopic) {
    const topicCounts = Array(topicKeys.length).fill(0);
    const topicSums = Array(topicKeys.length).fill(0);
    docTopic.forEach((docT) => {
      if (!docT) return;
      docT.forEach((proportion, topicIdx) => {
        topicSums[topicIdx] += proportion;
        topicCounts[topicIdx] += 1;
      });
    });
    topicProminence = topicSums.map((sum, i) =>
      topicCounts[i] > 0 ? sum / topicCounts[i] : 0
    );

    const minProminence = Math.min(...topicProminence);
    const maxProminence = Math.max(...topicProminence);
    const range = maxProminence - minProminence;

    topicProminence = topicProminence.map(p => {
      const normalized = range > 0 ? (p - minProminence) / range : 0;
      return Math.pow(normalized, 0.5);
    });
  } else {
    topicProminence = Array(topicKeys.length).fill(0.5);
  }

  let coords = null;
  if (topicCoords && topicCoords.length > 0) {
    // Use provided coordinates where available
    console.log('[Scaled] Using topic coordinates:', topicCoords.length, 'for', topicKeys.length, 'topics');

    const topicToCoord = new Map(
      topicCoords.map(tc => [tc.topic, [tc.x, tc.y]])
    );

    // For topics without coordinates, compute a grid position
    coords = topicKeys.map((tk, i) => {
      const coord = topicToCoord.get(tk.topic);
      if (coord) {
        return coord;
      } else {
        // Fallback: use grid position for missing coordinates
        const col = i % cols;
        const row = Math.floor(i / cols);
        // Scale grid positions to roughly match the coordinate range (-1 to 1)
        const x = (col / cols) * 2 - 1;
        const y = (row / Math.ceil(topicKeys.length / cols)) * 2 - 1;
        return [x, y];
      }
    });
  } else {
    console.log('[Scaled] No topic coordinates available, using grid layout');
    // Simple grid-based fallback
    coords = topicKeys.map((_, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      // Scale to -1 to 1 range like real coordinates
      const x = (col / cols) * 2 - 1;
      const y = (row / Math.ceil(topicKeys.length / cols)) * 2 - 1;
      return [x, y];
    });
  }

  // Create SVG with zoom
  const svg = d3.select(container)
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  const g = svg.append('g');

  const zoom = d3.zoom()
    .scaleExtent([0.5, 10])
    .on('zoom', function(event) {
      g.attr('transform', event.transform);
    });

  svg.call(zoom);
  window._scaledZoomSvg = svg;
  window._scaledZoomBehavior = zoom;

  // Scale coordinates to fit
  const xs = coords.map(c => c[0]);
  const ys = coords.map(c => c[1]);
  const xScale = d3.scaleLinear()
    .domain([Math.min(...xs), Math.max(...xs)])
    .range([80, width - 80]);
  const yScale = d3.scaleLinear()
    .domain([Math.min(...ys), Math.max(...ys)])
    .range([80, height - 80]);

  topicKeys.forEach((tk, i) => {
    // Skip hidden topics
    if (isTopicHidden(i)) return;

    const cx = xScale(coords[i][0]);
    const cy = yScale(coords[i][1]);
    const strokeWidth = 0.3 + (topicProminence[i] * 5.7);

    const group = g.append('g')
      .attr('transform', `translate(${cx},${cy})`)
      .style('cursor', 'pointer');

    group.append('circle')
      .attr('r', circleRadius)
      .attr('fill', 'rgba(240, 240, 240, 0.6)')
      .attr('stroke', 'rgba(200, 200, 200, 0.6)')
      .attr('stroke-width', strokeWidth)
      .attr('data-topic', tk.topic);

    // Top words with center-weighted arrangement
    const words = tk.words.slice(0, wordsInCircles);
    const baseFontSize = Math.max(10, 24 - wordsInCircles);
    const fontSizes = [];
    const yOffsets = [];

    if (words.length === 1) {
      fontSizes.push(baseFontSize);
      yOffsets.push(0);
    } else {
      fontSizes.push(baseFontSize);
      yOffsets.push(0);

      const verticalSpacing = Math.min(18, 70 / Math.max(1, words.length - 1));

      for (let j = 1; j < words.length; j++) {
        const fontSize = Math.max(8, baseFontSize - (j * 1.5));
        fontSizes.push(fontSize);

        const isAbove = (j % 2 === 1);
        const level = Math.ceil(j / 2);
        const yOffset = isAbove ? -level * verticalSpacing : level * verticalSpacing;
        yOffsets.push(yOffset);
      }
    }

    words.forEach((word, idx) => {
      group.append('text')
        .attr('x', 0)
        .attr('y', yOffsets[idx])
        .attr('text-anchor', 'middle')
        .attr('font-size', fontSizes[idx])
        .attr('fill', 'rgba(0, 0, 0, 0.6)')
        .text(word);
    });

    // Hover overlay
    group.append('circle')
      .attr('r', circleRadius)
      .attr('fill', 'rgba(51, 153, 255, 0.9)')
      .attr('opacity', 0)
      .attr('class', 'topic-hover-rect');

    group.append('text')
      .attr('x', 0)
      .attr('y', 0)
      .attr('text-anchor', 'middle')
      .attr('font-size', 24)
      .attr('fill', '#000')
      .attr('class', 'topic-hover-label')
      .attr('opacity', 0)
      .text(getTopicLabel(tk.topic));

    group.on('mouseover', function() {
      group.select('.topic-hover-rect').attr('opacity', 0.85);
      group.select('.topic-hover-label').attr('opacity', 1);
    });

    group.on('mouseout', function() {
      group.select('.topic-hover-rect').attr('opacity', 0);
      group.select('.topic-hover-label').attr('opacity', 0);
    });

    group.on('click', function() {
      window.page(`/topic/${tk.topic + 1}`); // Use 1-based numbering in URL
    });
  });

  // Reset zoom function
  window.resetScaledZoom = function() {
    if (window._scaledZoomSvg && window._scaledZoomBehavior) {
      window._scaledZoomSvg
        .transition()
        .duration(500)
        .call(window._scaledZoomBehavior.transform, d3.zoomIdentity);
    }
  };

  // Show reset zoom button
  const resetContainer = document.getElementById('reset-zoom-container');
  if (resetContainer) {
    resetContainer.style.display = 'inline';
  }
}

// Stacked View - Streamgraph of topics over time
function renderStackedView(topicKeys, docTopic, metadata, docTopicCounts) {
  const container = document.getElementById('overview-vis');
  container.innerHTML = '';

  const width = container.offsetWidth || 900;
  const height = 500;

  // Extract years from metadata
  const years = Array.from(new Set(metadata.map(m => m.year))).filter(y => y).sort();
  const topicCount = topicKeys.length;

  if (years.length === 0) {
    container.innerHTML = `
      <div class="alert alert-warning">
        <h4>No Date Information</h4>
        <p>The metadata does not contain year information needed for the stacked view.</p>
      </div>
    `;
    return;
  }

  // Helper: get topic proportions/counts by year
  function getSeries(mode) {
    // Use getVisibleTopicIndices to filter topics
    const visibleIndices = getVisibleTopicIndices(topicKeys);
    const series = visibleIndices.map(i => ({
      topic: topicKeys[i].topic,
      values: years.map(y => ({ year: y, value: 0 }))
    }));

    years.forEach((year, yIdx) => {
      const docs = metadata
        .map((doc, idx) => ({ doc, idx }))
        .filter(({ doc }) => doc.year === year);

      if (mode === 'frac') {
        docs.forEach(({ idx: docIdx }) => {
          const docT = docTopic[docIdx];
          if (!docT) return;
          visibleIndices.forEach((topicIdx, sIdx) => {
            series[sIdx].values[yIdx].value += docT[topicIdx] || 0;
          });
        });
      } else if (mode === 'raw' && docTopicCounts) {
        docs.forEach(({ idx: docIdx }) => {
          const docCounts = docTopicCounts[docIdx];
          if (!docCounts) return;
          visibleIndices.forEach((topicIdx, sIdx) => {
            series[sIdx].values[yIdx].value += docCounts[topicIdx] || 0;
          });
        });
      }
    });

    // For 'frac', normalize by total for each year
    if (mode === 'frac') {
      years.forEach((year, yIdx) => {
        const total = series.reduce((sum, s) => sum + s.values[yIdx].value, 0);
        if (total > 0) {
          series.forEach(s => {
            s.values[yIdx].value /= total;
          });
        }
      });
    }

    return series;
  }

  // Initial mode: 'frac'
  let mode = 'frac';
  let series = getSeries(mode);

  // Order topics by prominence
  const order = series
    .map((s, i) => ({ i, sum: s.values.reduce((a, b) => a + b.value, 0) }))
    .sort((a, b) => b.sum - a.sum)
    .map(o => o.i);

  const orderedSeries = order.map(i => series[i]);

  // Stack data for streamgraph
  // Use only visible topics for stacking
  const visibleTopicCount = orderedSeries.length;
  const dataByYear = years.map((year, yIdx) =>
    orderedSeries.map(s => s.values[yIdx].value)
  );

  const stack = d3.stack()
    .keys(d3.range(visibleTopicCount))
    .order(d3.stackOrderNone)
    .offset(d3.stackOffsetWiggle);

  const stackedData = stack(dataByYear);

  // Scales
  const x = d3.scalePoint()
    .domain(years)
    .range([60, width - 60]);

  let y;
  if (mode === 'frac') {
    y = d3.scaleLinear()
      .domain([
        d3.min(stackedData, layer => d3.min(layer, d => d[0])),
        d3.max(stackedData, layer => d3.max(layer, d => d[1]))
      ])
      .range([height - 40, 40]);
  } else {
    const maxTotal = d3.max(dataByYear, yearArr => Math.abs(d3.sum(yearArr)));
    y = d3.scaleLinear()
      .domain([-maxTotal, maxTotal])
      .range([height - 40, 40]);
  }

  const color = d3.scaleOrdinal(d3.schemeCategory10);

  const svg = d3.select(container)
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  const g = svg.append('g');

  // Zoom
  const zoom = d3.zoom()
    .scaleExtent([0.5, 10])
    .on('zoom', function(event) {
      g.attr('transform', event.transform);
    });

  svg.call(zoom);
  window._stackedZoomSvg = svg;
  window._stackedZoomBehavior = zoom;

  window.resetScaledZoom = function() {
    if (window._stackedZoomSvg && window._stackedZoomBehavior) {
      window._stackedZoomSvg
        .transition()
        .duration(500)
        .call(window._stackedZoomBehavior.transform, d3.zoomIdentity);
    }
  };

  // Area generator
  const area = d3.area()
    .x((d, i) => x(years[i]))
    .y0(d => y(d[0]))
    .y1(d => y(d[1]))
    .curve(d3.curveCatmullRom);

  // Draw streamgraph
  g.selectAll('.layer')
    .data(stackedData)
    .enter()
    .append('path')
    .attr('class', 'layer')
    .attr('d', area)
    .attr('fill', (d, i) => color(i))
    .attr('stroke', '#333')
    .attr('opacity', 0.85)
    .on('mouseover', function(event, d) {
      d3.select(this).attr('opacity', 1);

      let tooltip = d3.select('#stacked-tooltip');
      if (tooltip.empty()) {
        tooltip = d3.select('body')
          .append('div')
          .attr('id', 'stacked-tooltip')
          .style('position', 'absolute')
          .style('pointer-events', 'none')
          .style('background', 'rgba(255,255,255,0.97)')
          .style('border', '1px solid #888')
          .style('border-radius', '4px')
          .style('padding', '6px 12px')
          .style('font-size', '15px')
          .style('color', '#222')
          .style('box-shadow', '0 2px 8px rgba(0,0,0,0.15)')
          .style('z-index', 1000);
      }

      const topicIdx = order[d.index];
      const topic = topicKeys[topicIdx];
      const topicNum = topic.topic + 1;
      const words = topic.words.slice(0, 5).join(', ');
      tooltip
        .style('display', 'block')
        .html(`<b>Topic ${topicNum}:</b> ${words}`);
    })
    .on('mousemove', function(event) {
      const tooltip = d3.select('#stacked-tooltip');
      if (!tooltip.empty()) {
        tooltip
          .style('left', event.pageX + 16 + 'px')
          .style('top', event.pageY - 10 + 'px');
      }
    })
    .on('mouseout', function() {
      d3.select(this).attr('opacity', 0.85);
      d3.select('#stacked-tooltip').remove();
    })
    .on('click', function(event, d) {
      const topicIdx = order[d.index];
      const topic = topicKeys[topicIdx];
      window.page(`/topic/${topic.topic + 1}`); // Use 1-based numbering in URL
    });

  // X axis
  g.append('g')
    .attr('transform', `translate(0,${height - 40})`)
    .call(d3.axisBottom(x));

  // Y axis (hidden ticks)
  const yAxisG = g.append('g')
    .attr('transform', `translate(60,0)`)
    .call(d3.axisLeft(y).tickSize(0));

  yAxisG.selectAll('.tick text').remove();
  yAxisG.selectAll('.domain').remove();

  // Conditional view controls (% vs raw counts) - only if docTopicCounts available
  if (docTopicCounts) {
    const fracBtn = document.getElementById('nav_model_conditional_frac');
    const rawBtn = document.getElementById('nav_model_conditional_raw');

    if (fracBtn && rawBtn) {
      function setActiveConditional(activeId) {
        fracBtn.classList.remove('active');
        rawBtn.classList.remove('active');
        document.getElementById(activeId).classList.add('active');
      }

      function updateStacked() {
        g.selectAll('.layer').remove();
        d3.select('#stacked-tooltip').remove();

        let newSeries = getSeries(mode);
        const orderedSeries = order.map(i => newSeries[i]);
        const dataByYear = years.map((year, yIdx) =>
          orderedSeries.map(s => s.values[yIdx].value)
        );
        const newStackedData = stack(dataByYear);

        if (mode === 'frac') {
          y.domain([
            d3.min(newStackedData, layer => d3.min(layer, d => d[0])),
            d3.max(newStackedData, layer => d3.max(layer, d => d[1]))
          ]);
        } else {
          const maxTotal = d3.max(dataByYear, yearArr => Math.abs(d3.sum(yearArr)));
          y.domain([-maxTotal, maxTotal]);
        }

        g.selectAll('.layer')
          .data(newStackedData)
          .enter()
          .append('path')
          .attr('class', 'layer')
          .attr('d', area)
          .attr('fill', (d, i) => color(i))
          .attr('stroke', '#333')
          .attr('opacity', 0.85)
          .on('mouseover', function(event, d) {
            d3.select(this).attr('opacity', 1);
            let tooltip = d3.select('#stacked-tooltip');
            if (tooltip.empty()) {
              tooltip = d3.select('body')
                .append('div')
                .attr('id', 'stacked-tooltip')
                .style('position', 'absolute')
                .style('pointer-events', 'none')
                .style('background', 'rgba(255,255,255,0.97)')
                .style('border', '1px solid #888')
                .style('border-radius', '4px')
                .style('padding', '6px 12px')
                .style('font-size', '15px')
                .style('color', '#222')
                .style('box-shadow', '0 2px 8px rgba(0,0,0,0.15)')
                .style('z-index', 1000);
            }
            const topicIdx = order[d.index];
            const topic = topicKeys[topicIdx];
            const topicNum = topic.topic + 1;
            const words = topic.words.slice(0, 5).join(', ');
            tooltip.style('display', 'block').html(`<b>Topic ${topicNum}:</b> ${words}`);
          })
          .on('mousemove', function(event) {
            const tooltip = d3.select('#stacked-tooltip');
            if (!tooltip.empty()) {
              tooltip.style('left', event.pageX + 16 + 'px').style('top', event.pageY - 10 + 'px');
            }
          })
          .on('mouseout', function() {
            d3.select(this).attr('opacity', 0.85);
            d3.select('#stacked-tooltip').remove();
          })
          .on('click', function(event, d) {
            const topicIdx = order[d.index];
            const topic = topicKeys[topicIdx];
            window.page(`/topic/${topic.topic}`);
          });

        yAxisG.selectAll('.tick text').remove();
        yAxisG.selectAll('.domain').remove();
      }

      fracBtn.addEventListener('click', function(e) {
        e.preventDefault();
        mode = 'frac';
        updateStacked();
        setActiveConditional('nav_model_conditional_frac');
      });

      rawBtn.addEventListener('click', function(e) {
        e.preventDefault();
        mode = 'raw';
        updateStacked();
        setActiveConditional('nav_model_conditional_raw');
      });
    }
  }

  // Show reset zoom button
  const resetContainer = document.getElementById('reset-zoom-container');
  if (resetContainer) {
    resetContainer.style.display = 'inline';
  }
}
