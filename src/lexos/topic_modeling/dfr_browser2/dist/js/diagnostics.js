// Diagnostics view - Topic model quality visualization
import { getTopicLabel } from './topic-config.js';

// Store diagnostics data globally for this module
let diagnosticsData = null;

export async function loadDiagnosticsView() {
  const mainView = document.getElementById('main-view');

  // Check if data is loaded
  if (!window.dfrState.dataLoaded) {
    mainView.innerHTML = `
      <div class="alert alert-info">
        <h4>No Data Loaded</h4>
        <p>Please load data to view diagnostics.</p>
      </div>
    `;
    return;
  }

  // Check if config is loaded
  if (!window.dfrState.config) {
    mainView.innerHTML = `
      <div class="alert alert-warning">
        <h4>Configuration Not Loaded</h4>
        <p>Please wait for the application to finish loading.</p>
      </div>
    `;
    return;
  }

  console.log('[Diagnostics] Loading diagnostics view');

  // Show loading state
  mainView.innerHTML = `
    <div class="d-flex justify-content-center align-items-center" style="min-height: 200px;">
      <div class="spinner-border text-primary" role="status">
        <span class="visually-hidden">Loading diagnostics...</span>
      </div>
    </div>
  `;

  // Get diagnostics file path from config
  const config = window.dfrState.config;
  const diagnosticsPath = config?.diagnostics_file || 'diagnostics.xml';

  try {
    // Use relative path for web server
    const fetchPath = diagnosticsPath.startsWith('/') ? diagnosticsPath.substring(1) : diagnosticsPath;

    console.log('[Diagnostics] Loading diagnostics from:', fetchPath);

    // Try to load diagnostics file
    const diagnosticsResponse = await fetch(fetchPath);
    console.log('[Diagnostics] Fetch response status:', diagnosticsResponse.status, diagnosticsResponse.ok);

    if (!diagnosticsResponse.ok) {
      throw new Error(`Diagnostics file not found at ${fetchPath}`);
    }

    const diagnosticsText = await diagnosticsResponse.text();
    console.log('[Diagnostics] Response text length:', diagnosticsText.length);
    console.log('[Diagnostics] First 100 chars:', diagnosticsText.substring(0, 100));

    // Check if we got HTML instead of XML/JSON (happens when file is missing and server returns index.html)
    if (diagnosticsText.trim().startsWith('<!DOCTYPE html>') || diagnosticsText.trim().startsWith('<html')) {
      throw new Error(`Diagnostics file not found - server returned HTML instead of data file`);
    }

    // Check if it's JSON or XML based on file extension or content
    if (diagnosticsPath.endsWith('.json') || diagnosticsText.trim().startsWith('{')) {
      diagnosticsData = parseDiagnosticsJson(diagnosticsText);
    } else {
      diagnosticsData = parseDiagnosticsXml(diagnosticsText);
    }

    console.log('[Diagnostics] Parsed data:', diagnosticsData);
    console.log('[Diagnostics] Topics count:', diagnosticsData?.topics?.length);

    // Check if we got valid data
    if (!diagnosticsData || !diagnosticsData.topics || diagnosticsData.topics.length === 0) {
      throw new Error(`No valid diagnostics data found in file`);
    }

    renderDiagnosticsView();

  } catch (error) {
    console.error('[Diagnostics] Could not load diagnostics file:', error);
    console.error('[Diagnostics] Expected path:', diagnosticsPath);
    // Clear diagnostics data to ensure we don't use stale data
    diagnosticsData = null;
    // Show message that no diagnostics are available
    console.log('[Diagnostics] Setting mainView.innerHTML to alert message');
    mainView.innerHTML = `
      <div class="alert alert-warning">
        <h4>No Diagnostics Data Available</h4>
        <p>The diagnostics file could not be found. Please ensure a diagnostics file is available in your data directory.</p>
        <p class="mb-0"><small>Expected file: ${diagnosticsPath}</small></p>
      </div>
    `;
    console.log('[Diagnostics] mainView.innerHTML set, length:', mainView.innerHTML.length);
  }
}

function parseDiagnosticsXml(xmlText) {
  const parser = new DOMParser();
  const xmlDoc = parser.parseFromString(xmlText, 'text/xml');

  const topics = Array.from(xmlDoc.querySelectorAll('topic')).map(topicElement => {
    const topicId = parseInt(topicElement.getAttribute('id'));
    const coherence = parseFloat(topicElement.getAttribute('coherence') || '0');
    const exclusivity = parseFloat(topicElement.getAttribute('exclusivity') || '0');
    const documentEntropy = parseFloat(topicElement.getAttribute('document_entropy') || '0');
    // Use exclusivity as a proxy for distinctiveness for the chart
    const distinctiveness = exclusivity;
    // Use tokens as a proxy for size (number of tokens in the topic)
    const size = parseInt(topicElement.getAttribute('tokens') || '0');

    const topWords = Array.from(topicElement.querySelectorAll('word')).map(wordElement => ({
      word: wordElement.textContent.trim(),
      weight: parseFloat(wordElement.getAttribute('prob') || '0')
    }));

    return {
      id: topicId,
      coherence,
      exclusivity,
      documentEntropy,
      distinctiveness,
      size,
      topWords
    };
  });

  const modelMetrics = {
    overallCoherence: topics.length > 0 ? topics.reduce((sum, t) => sum + t.coherence, 0) / topics.length : 0,
    numTopics: topics.length,
    avgCoherence: topics.length > 0 ? topics.reduce((sum, t) => sum + t.coherence, 0) / topics.length : 0,
    avgDistinctiveness: topics.length > 0 ? topics.reduce((sum, t) => sum + t.distinctiveness, 0) / topics.length : 0
  };

  return { topics, modelMetrics };
}

function parseDiagnosticsJson(jsonText) {
  const data = JSON.parse(jsonText);

  // Handle different possible JSON structures
  let topics = [];
  let modelMetrics = {};

  if (data.diagnostics && data.diagnostics.topics) {
    // MALLET-style JSON structure
    topics = data.diagnostics.topics.map(topic => ({
      id: topic.id,
      coherence: topic.coherence || 0,
      exclusivity: topic.exclusivity || 0,
      documentEntropy: topic.document_entropy || topic.documentEntropy || 0,
      distinctiveness: topic.exclusivity || topic.distinctiveness || 0,
      size: topic.size || topic.tokens || 0,
      topWords: topic.words || []
    }));

    const validTopics = topics.filter(t => !isNaN(t.coherence) && !isNaN(t.distinctiveness));
    modelMetrics = {
      overallCoherence: data.diagnostics.model?.coherence || 0,
      numTopics: topics.length,
      avgCoherence: validTopics.length > 0 ? validTopics.reduce((sum, t) => sum + t.coherence, 0) / validTopics.length : 0,
      avgDistinctiveness: validTopics.length > 0 ? validTopics.reduce((sum, t) => sum + t.distinctiveness, 0) / validTopics.length : 0
    };
  } else if (Array.isArray(data)) {
    // Array of topics
    topics = data.map((topic, index) => ({
      id: topic.id || index,
      coherence: topic.coherence || 0,
      exclusivity: topic.exclusivity || 0,
      documentEntropy: topic.document_entropy || topic.documentEntropy || 0,
      distinctiveness: topic.exclusivity || topic.distinctiveness || 0,
      size: topic.size || topic.tokens || 0,
      topWords: topic.words || topic.topWords || []
    }));

    const validTopics = topics.filter(t => !isNaN(t.coherence) && !isNaN(t.distinctiveness));
    modelMetrics = {
      overallCoherence: validTopics.length > 0 ? validTopics.reduce((sum, t) => sum + t.coherence, 0) / validTopics.length : 0,
      numTopics: topics.length,
      avgCoherence: validTopics.length > 0 ? validTopics.reduce((sum, t) => sum + t.coherence, 0) / validTopics.length : 0,
      avgDistinctiveness: validTopics.length > 0 ? validTopics.reduce((sum, t) => sum + t.distinctiveness, 0) / validTopics.length : 0
    };
  }

  return { topics, modelMetrics };
}

function renderDiagnosticsView() {
  const mainView = document.getElementById('main-view');

  mainView.innerHTML = `
    <div class="card">
      <div class="card-header">
        <h2 class="card-title mb-0">Topic Model Diagnostics</h2>
      </div>
      <div class="card-body">
        <!-- Model Overview -->
        <div class="row mb-4">
          <div class="col-md-12">
            <h4 class="d-flex align-items-center">
              Model Overview
              <button type="button" class="btn btn-link text-secondary ms-2 p-0" data-bs-toggle="modal" data-bs-target="#modelOverviewModal" style="text-decoration: none; font-size: 1.2rem;" title="Help">
                <i class="bi bi-question-circle-fill"></i>
              </button>
            </h4>
            <div class="row">
              <div class="col-md-3">
                <div class="card bg-primary text-white h-100">
                  <div class="card-body text-center d-flex flex-column justify-content-center">
                    <h5 class="card-title">${diagnosticsData.modelMetrics.numTopics || 'N/A'}</h5>
                    <p class="card-text mb-0">Total Topics</p>
                    <small class="text-white-50">&nbsp;</small>
                  </div>
                </div>
              </div>
              <div class="col-md-3">
                <div class="card bg-success text-white h-100">
                  <div class="card-body text-center d-flex flex-column justify-content-center">
                    <h5 class="card-title">${diagnosticsData.modelMetrics.avgCoherence ? diagnosticsData.modelMetrics.avgCoherence.toFixed(3) : 'N/A'}</h5>
                    <p class="card-text mb-0">Avg Coherence</p>
                    <small class="text-white-50">Less negative = more coherent</small>
                  </div>
                </div>
              </div>
              <div class="col-md-3">
                <div class="card bg-info text-white h-100">
                  <div class="card-body text-center d-flex flex-column justify-content-center">
                    <h5 class="card-title">${diagnosticsData.modelMetrics.avgDistinctiveness ? diagnosticsData.modelMetrics.avgDistinctiveness.toFixed(3) : 'N/A'}</h5>
                    <p class="card-text mb-0">Avg Exclusivity</p>
                    <small class="text-white-50">Higher = more unique words</small>
                  </div>
                </div>
              </div>
              <div class="col-md-3">
                <div class="card bg-warning text-dark h-100">
                  <div class="card-body text-center d-flex flex-column justify-content-center">
                    <h5 class="card-title">${diagnosticsData.modelMetrics.overallCoherence ? diagnosticsData.modelMetrics.overallCoherence.toFixed(3) : 'N/A'}</h5>
                    <p class="card-text mb-0">Overall Coherence</p>
                    <small class="text-muted">Model quality indicator</small>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Topic Quality Scatter Plot -->
        <div class="row mb-4">
          <div class="col-md-12">
            <h4 class="d-flex align-items-center">
              Topic Quality Analysis
              <button type="button" class="btn btn-link text-secondary ms-2 p-0" data-bs-toggle="modal" data-bs-target="#chartInterpretationModal" style="text-decoration: none; font-size: 1.2rem;" title="Help">
                <i class="bi bi-question-circle-fill"></i>
              </button>
            </h4>
            <div id="topic-quality-chart" style="height: 440px;"></div>
          </div>
        </div>

        <!-- Topic Details Table -->
        <div class="row">
          <div class="col-md-12">
            <h4 class="d-flex align-items-center">
              Topic Details
              <button type="button" class="btn btn-link text-secondary ms-2 p-0" data-bs-toggle="modal" data-bs-target="#tableColumnsModal" style="text-decoration: none; font-size: 1.2rem;" title="Help">
                <i class="bi bi-question-circle-fill"></i>
              </button>
            </h4>
            <div class="table-responsive">
              <table class="table table-striped table-hover" id="diagnostics-table">
                <thead>
                  <tr>
                    <th>Topic</th>
                    <th>Coherence</th>
                    <th>Exclusivity</th>
                    <th>Doc Entropy</th>
                    <th>Token Count</th>
                    <th>Top Words</th>
                  </tr>
                </thead>
                <tbody>
                  ${diagnosticsData.topics.map(topic => `
                    <tr onclick="window.page('/topic/${topic.id + 1}');">
                      <td>${getTopicLabel(topic.id)}</td>
                      <td>${topic.coherence.toFixed(3)}</td>
                      <td>${topic.exclusivity.toFixed(3)}</td>
                      <td>${topic.documentEntropy.toFixed(3)}</td>
                      <td>${topic.size.toLocaleString()}</td>
                      <td>
                        <small>
                          ${topic.topWords.slice(0, 5).map(w => w.word).join(', ')}
                          ${topic.topWords.length > 5 ? '...' : ''}
                        </small>
                      </td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Model Overview Modal -->
    <div class="modal fade" id="modelOverviewModal" tabindex="-1" aria-labelledby="modelOverviewModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-lg">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="modelOverviewModalLabel"><i class="fas fa-lightbulb"></i> Understanding Topic Model Quality</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <p class="mb-3">This diagnostics view helps you evaluate the quality of your topic model using metrics from MALLET. Here's what to look for:</p>
            <div class="row">
              <div class="col-md-6">
                <h6>Key Metrics:</h6>
                <ul class="mb-0">
                  <li><strong>Coherence</strong>: How well top words fit together (less negative = better)</li>
                  <li><strong>Exclusivity</strong>: How unique words are to this topic (higher = more distinctive)</li>
                  <li><strong>Token Count</strong>: Topic size - avoid extremes (too small = unreliable, too large = generic)</li>
                </ul>
              </div>
              <div class="col-md-6">
                <h6>Quality Indicators:</h6>
                <ul class="mb-0">
                  <li><span class="text-success">✓</span> Upper-right topics: High exclusivity, good coherence</li>
                  <li><span class="text-warning">⚠</span> Lower-left topics: Generic or incoherent topics</li>
                  <li><span class="text-danger">✗</span> Extreme sizes: Too small/large topics may be problematic</li>
                </ul>
              </div>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Chart Interpretation Modal -->
    <div class="modal fade" id="chartInterpretationModal" tabindex="-1" aria-labelledby="chartInterpretationModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-lg">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="chartInterpretationModalLabel"><i class="fas fa-info-circle"></i> How to Interpret This Chart</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <ul class="mb-0">
              <li><strong>Coherence (X-axis)</strong>: Measures how often top words co-occur together. Less negative values indicate more coherent topics.</li>
              <li><strong>Exclusivity (Y-axis)</strong>: Measures how unique top words are to this topic. Higher values indicate more distinctive topics.</li>
              <li><strong>Circle Size</strong>: Represents topic size (number of tokens). Larger circles indicate topics with more tokens assigned to them.</li>
              <li><strong>Topic Numbers</strong>: Each circle displays its topic number. Click on a circle to view that topic's details.</li>
              <li><strong>Ideal Topics</strong>: Upper-right quadrant (high exclusivity, less negative coherence) are typically the most interpretable.</li>
            </ul>
            <div class="alert alert-info mt-3">
              <small><strong>Navigation Tip:</strong> Use mouse wheel to zoom in/out, and drag to pan around the chart to access overlapping circles.</small>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Table Columns Modal -->
    <div class="modal fade" id="tableColumnsModal" tabindex="-1" aria-labelledby="tableColumnsModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-lg">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="tableColumnsModalLabel"><i class="fas fa-table"></i> Understanding Table Columns</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <p class="mb-3">This table shows detailed diagnostics for each topic from MALLET. Here's what each column means:</p>

            <h6><strong>Coherence</strong></h6>
            <p>Measures whether the top words in a topic tend to co-occur together in documents. Values are log probabilities (negative numbers). Values closer to zero indicate that top words co-occur more often, suggesting the topic is more coherent and interpretable.</p>

            <h6><strong>Exclusivity</strong></h6>
            <p>Measures how unique the top words are to this topic compared to other topics. Values range from 0 to 1. Higher values indicate the topic's top words are more exclusive to this topic and don't appear prominently in other topics, making the topic more distinctive.</p>

            <h6><strong>Doc Entropy</strong></h6>
            <p>Document entropy measures how the topic is distributed across documents. Low entropy means the topic is concentrated in a few documents (high predictability). High entropy means the topic is spread evenly across many documents. Very low entropy can indicate unusual documents or documents in other languages.</p>

            <h6><strong>Token Count</strong></h6>
            <p>The total number of word tokens assigned to this topic across the entire corpus. Very small counts may indicate unreliable topics (not enough observations). Very large counts may indicate "background" topics with common words that aren't topic-specific.</p>

            <h6><strong>Top Words</strong></h6>
            <p>The five most probable words in this topic, providing a quick preview of the topic's content.</p>

            <div class="alert alert-info mt-3">
              <small><strong>Learn More:</strong> For detailed explanations and mathematical definitions, visit <a href="https://mallet.cs.umass.edu/diagnostics.php" target="_blank">MALLET's diagnostics documentation</a>.</small>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
          </div>
        </div>
      </div>
    </div>
  `;

  // Render the scatter plot
  renderTopicQualityChart();

  // Add window resize handler
  let resizeTimeout;
  window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(function() {
      renderTopicQualityChart();
    }, 250);
  });
}

function renderTopicQualityChart() {
  const chartContainer = document.getElementById('topic-quality-chart');

  if (!chartContainer) {
    console.warn('[Diagnostics] Chart container not found');
    return;
  }

  // Clear existing chart
  d3.select(chartContainer).selectAll('*').remove();

  if (!diagnosticsData || !diagnosticsData.topics || diagnosticsData.topics.length === 0) {
    console.warn('[Diagnostics] No diagnostics data available');
    d3.select(chartContainer)
      .append('div')
      .style('text-align', 'center')
      .style('padding', '50px')
      .text('No diagnostics data available');
    return;
  }

  // Prepare data for D3
  const data = diagnosticsData.topics.map(topic => ({
    x: topic.coherence,
    y: topic.distinctiveness,
    size: topic.size,
    topicId: topic.id,
    label: getTopicLabel(topic.id)
  }));

  // Filter out invalid data points
  const validData = data.filter(d => !isNaN(d.x) && !isNaN(d.y) && isFinite(d.x) && isFinite(d.y));

  if (validData.length === 0) {
    console.warn('[Diagnostics] No valid data points for chart');
    d3.select(chartContainer)
      .append('div')
      .style('text-align', 'center')
      .style('padding', '50px')
      .text('No valid diagnostics data to display');
    return;
  }

  const margin = { top: 20, right: 20, bottom: 60, left: 70 };
  const width = Math.max(chartContainer.clientWidth - margin.left - margin.right, 200);
  const height = 400 - margin.top - margin.bottom;

  const svg = d3.select(chartContainer)
    .append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom);

  // Create a container group with margins
  const container = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  // Create a clip path to prevent content from overflowing
  container.append('defs').append('clipPath')
    .attr('id', 'chart-clip')
    .append('rect')
    .attr('width', width)
    .attr('height', height);

  // Scales
  const xMin = d3.min(validData, d => d.x);
  const xMax = d3.max(validData, d => d.x);
  const yMax = d3.max(validData, d => d.y);

  const xScale = d3.scaleLinear()
    .domain([xMin * 1.1, Math.max(xMax * 1.1, 0)])
    .range([0, width]);

  const yScale = d3.scaleLinear()
    .domain([0, yMax * 1.1])
    .range([height, 0]);

  const sizeScale = d3.scaleSqrt()
    .domain([0, d3.max(validData, d => d.size)])
    .range([5, 20]);

  // Create axes (non-zoomable)
  const xAxis = container.append('g')
    .attr('class', 'x-axis')
    .attr('transform', `translate(0,${height})`)
    .call(d3.axisBottom(xScale).ticks(8));

  const yAxis = container.append('g')
    .attr('class', 'y-axis')
    .call(d3.axisLeft(yScale).ticks(8));

  // Axis labels (non-zoomable)
  container.append('text')
    .attr('class', 'axis-label')
    .attr('transform', `translate(${width/2}, ${height + margin.bottom - 8})`)
    .style('text-anchor', 'middle')
    .style('font-size', '13px')
    .style('font-weight', '500')
    .style('fill', '#495057')
    .text('Coherence (less negative = more coherent)');

  container.append('text')
    .attr('class', 'axis-label')
    .attr('transform', 'rotate(-90)')
    .attr('y', 0 - margin.left + 18)
    .attr('x', 0 - (height / 2))
    .attr('dy', '1em')
    .style('text-anchor', 'middle')
    .style('font-size', '13px')
    .style('font-weight', '500')
    .style('fill', '#495057')
    .text('Exclusivity (higher = more unique words)');

  // Create zoomable group for points
  const zoomGroup = container.append('g')
    .attr('class', 'zoom-group')
    .attr('clip-path', 'url(#chart-clip)');

  // Add zoom behavior to the container
  const zoom = d3.zoom()
    .scaleExtent([0.5, 10])
    .translateExtent([[0, 0], [width, height]])
    .on('zoom', function(event) {
      // Update the zoom group transform
      zoomGroup.attr('transform', event.transform);
    });

  // Apply zoom to the entire container but only transform the zoom group
  svg.call(zoom);

  // Apply zoom to the entire container but only transform the zoom group
  svg.call(zoom);

  // Points with topic numbers
  zoomGroup.selectAll('circle')
    .data(validData)
    .enter()
    .append('circle')
    .attr('class', 'topic-circle')
    .attr('cx', d => xScale(d.x))
    .attr('cy', d => yScale(d.y))
    .attr('r', d => sizeScale(d.size))
    .attr('fill', 'steelblue')
    .attr('opacity', 0.75)
    .attr('stroke', 'white')
    .attr('stroke-width', 2)
    .on('mouseover', function(event, d) {
      d3.select(this)
        .attr('opacity', 0.95)
        .attr('stroke-width', 3)
        .attr('stroke', '#ffc107');
    })
    .on('mouseout', function(event, d) {
      d3.select(this)
        .attr('opacity', 0.75)
        .attr('stroke-width', 2)
        .attr('stroke', 'white');
    })
    .on('click', function(event, d) {
      // Navigate to topic view
      window.page(`/topic/${d.topicId + 1}`);
    })
    .style('cursor', 'pointer');

  // Add topic numbers as text inside circles
  zoomGroup.selectAll('text')
    .data(validData)
    .enter()
    .append('text')
    .attr('class', 'topic-label')
    .attr('x', d => xScale(d.x))
    .attr('y', d => yScale(d.y))
    .attr('text-anchor', 'middle')
    .attr('dy', '0.35em')
    .attr('font-size', d => Math.max(8, Math.min(14, sizeScale(d.size) * 0.55)))
    .attr('fill', 'white')
    .attr('font-weight', 'bold')
    .attr('pointer-events', 'none')
    .style('text-shadow', '0 1px 2px rgba(0,0,0,0.6)')
    .text(d => d.topicId);

  // Add instruction text
  svg.append('text')
    .attr('x', width + margin.left - 10)
    .attr('y', margin.top + 15)
    .attr('text-anchor', 'end')
    .style('font-size', '11px')
    .style('fill', '#6c757d')
    .style('font-style', 'italic')
    .text('Scroll to zoom, drag to pan');
}