// Word View
import { loadWordList } from './wordlist.js';
import { extractTopicWords } from './state-utils.js';
import { getTopicLabel } from './topic-config.js';

// Function to extract enhanced word lists from state file
async function enhanceTopicKeysWithState(topicKeys, wordsCount) {
  try {
    const topicWordLists = await extractTopicWords(topicKeys, wordsCount);
    if (!topicWordLists || topicWordLists.length === 0) {
      throw new Error('No enhanced words extracted');
    }

    const result = topicKeys.map((topic, i) => ({
      ...topic,
      words: topicWordLists[i] && topicWordLists[i].length > 0
        ? topicWordLists[i]
        : topic.words // fallback to original words
    }));

    return result;
  } catch (error) {
    return topicKeys;
  }
}

export async function loadWordView(topicKeys, docTopic, metadata, searchWord = null) {
  const mainView = document.getElementById('main-view');

  // Get user settings for number of words to display
  const settings = await window.getSettings();
  const wordsCount = settings.wordsInTopic || 50;

  // Try to get enhanced word lists from state file, fall back to original keys
  let enhancedTopicKeys;
  try {
    const topicWordLists = await extractTopicWords(topicKeys.length, wordsCount);
    enhancedTopicKeys = topicKeys.map((topic, i) => ({
      ...topic,
      words: topicWordLists[i] && topicWordLists[i].length > 0
        ? topicWordLists[i]
        : topic.words // fallback to original words
    }));
  } catch (error) {
    enhancedTopicKeys = topicKeys;
  }

  // Use enhanced topic keys for the rest of the function
  const topicKeysToUse = enhancedTopicKeys;

  // Store the prefilled word parameter
  const prefilledWord = searchWord;

  const main = document.getElementById('main-view');
  let html = `<div class=\"card\"><div class=\"card-body\">`
  html += `<nav class=\"navbar navbar-expand-lg navbar-light bg-light mb-3\">\n<div class=\"container-fluid\">\n<ul class=\"navbar-nav me-auto mb-2 mb-lg-0\">\n<li class=\"nav-item\">\n<span class=\"navbar-text wordlist-message\" id=\"overview-navbar-message\">Choose a specific word to view from the <a href=\"#\" id=\"word-list-link\">Word Index</a> or a topic page, or enter a word to the right.</span>\n</li>\n</ul>\n<form class=\"d-flex ms-auto\" id=\"word-search-form\">\n<input class=\"form-control form-control-sm me-2\" type=\"search\" placeholder=\"Enter a word\" aria-label=\"Enter a word\" id=\"word-input\">\n<button class=\"btn btn-outline-secondary btn-sm text-nowrap\" type=\"submit\" id=\"list-topics-btn\">List Topics</button>\n</form>\n</div>\n</nav>`
  html += `<div id=\"word-content\"></div>`
  html += `</div></div>`
  main.innerHTML = html;

  // Add form submission event listener
  const form = document.getElementById('word-search-form');
  const wordInput = document.getElementById('word-input');
  const wordContent = document.getElementById('word-content');
  const wordListLink = document.getElementById('word-list-link');

  // Add click event listener for word list link
  wordListLink.addEventListener('click', function(e) {
    e.preventDefault();
    loadWordList(topicKeysToUse, docTopic, metadata);
  });

  form.addEventListener('submit', function(e) {
    e.preventDefault();
    const wordValue = wordInput.value.trim().toLowerCase();
    if (wordValue) {
      // Navigate to the word permalink
      window.page(`/word/${encodeURIComponent(wordValue)}`);
    }
  });

  // Function to display word search results
  function displayWordResults(wordValue) {
    // Check if the word exists in any topic's words and find relevant topics
    let relevantTopics = [];
    if (topicKeysToUse && topicKeysToUse.length > 0) {
      relevantTopics = topicKeysToUse.map((topic, originalIndex) => ({
        ...topic,
        topicNumber: originalIndex + 1, // 1-based numbering for display
        topicIndex: originalIndex // 0-based index for navigation
      })).filter(topic =>
        topic.words.some(word => word.toLowerCase() === wordValue)
      );
    }

    const wordFound = relevantTopics.length > 0;

    // Generate the content
    let contentHTML = `<h2 id="word_header">Prominent topics for <span class="word">${wordValue}</span></h2>
      <div id="word_view_explainer" class="${wordFound ? '' : 'hidden'}">
        <p class="help">Click row labels to go to the corresponding topic page; click a word to show the topic list for that word.</p>
      </div>
      <div class="alert alert-info ${wordFound ? 'hidden' : ''}">
        <p>There are no topics in which this word is prominent.</p>
      </div>`;

    if (wordFound) {
      contentHTML += `<div id="word-topics-visualization"></div>`;
    }

    wordContent.innerHTML = contentHTML;

    // If word found, create the visualization
    if (wordFound) {
      createWordTopicsVisualization(relevantTopics, wordValue);
    }
  }

  // If a prefilled word was provided, display results immediately
  if (prefilledWord) {
    wordInput.value = prefilledWord;
    displayWordResults(prefilledWord);
  }

  // Function to create the word topics visualization (adapted from dfr-browser)
  function createWordTopicsVisualization(topics, searchWord) {
    const container = document.getElementById('word-topics-visualization');
    if (!container) return;

    // Clear previous content
    container.innerHTML = '';

    // Configuration constants
    const rowHeight = 80;
    const margin = { top: 30, right: 20, bottom: 10, left: 120 };
    const maxWords = 38;
    const containerWidth = container.offsetWidth || 800;
    const width = containerWidth - margin.left - margin.right;
    const height = (rowHeight + 30) * (topics.length + 1); // Added 30px margin per row

    // Create SVG
    const svg = d3.select(container)
      .append('svg')
      .attr('width', containerWidth)
      .attr('height', height + margin.top + margin.bottom);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const scaleX = d3.scaleLinear()
      .domain([0, maxWords])
      .range([0, width]);

    const scaleY = d3.scaleLinear()
      .domain([0, Math.max(1, topics.length - 1)])
      .range([rowHeight, (rowHeight + 30) * Math.max(topics.length, 1)]); // Added 30px margin between rows

    const scaleBar = d3.scaleLinear()
      .domain([0, 1])
      .range([0, rowHeight / 2]);

    // Create topic groups
    const topicGroups = g.selectAll('g.topic')
      .data(topics, d => d.topicNumber)
      .enter()
      .append('g')
      .classed('topic', true)
      .attr('transform', (d, i) => `translate(0,${scaleY(i)})`);

    // Add topic labels with wrapping support
    const labelWidth = margin.left - 20; // Available width for labels
    const lineHeight = 18; // Line height for wrapped text

    topicGroups.each(function(d) {
      const group = d3.select(this);
      const labelText = getTopicLabel(d.topicNumber - 1);

      // Create a foreignObject for HTML-based wrapping
      const fo = group.append('foreignObject')
        .attr('x', -margin.left + 10)
        .attr('y', -rowHeight / 2 - 20)
        .attr('width', labelWidth)
        .attr('height', rowHeight)
        .style('cursor', 'pointer')
        .on('click', function(event, d) {
          // Navigate to the topic view using page.js routing
          window.page(`/topic/${d.topicNumber}`);
        });

      fo.append('xhtml:div')
        .style('font-weight', 'bold')
        .style('font-size', '16px')
        .style('text-align', 'right')
        .style('line-height', '1.2')
        .style('word-wrap', 'break-word')
        .style('overflow-wrap', 'break-word')
        .style('hyphens', 'auto')
        .style('display', 'flex')
        .style('align-items', 'center')
        .style('justify-content', 'flex-end')
        .style('height', '100%')
        .style('padding-right', '5px')
        .classed('topic-label', true)
        .text(labelText);
    });

    // Create word groups within each topic
    const wordGroups = topicGroups.selectAll('g.word')
      .data(d => {
        const topWords = d.words.slice(0, maxWords);
        const maxWeight = topWords[0] ? 1 : 0; // Assume first word has highest weight
        return topWords.map((word, i) => ({
          word: word,
          weight: Math.max(0.1, 1 - (i * 0.06)), // Decreasing weight by position
          normalizedWeight: Math.max(0.1, 1 - (i * 0.06)),
          index: i,
          isSearchWord: word.toLowerCase() === searchWord.toLowerCase()
        }));
      })
      .enter()
      .append('g')
      .classed('word', true)
      .classed('selected-word', d => d.isSearchWord)
      .attr('transform', (d, i) => `translate(${scaleX(i)},-${rowHeight / 2})`)
      .style('cursor', 'pointer');

    // Add bars
    wordGroups.append('rect')
      .classed('proportion', true)
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', width / (2 * maxWords))
      .attr('height', d => scaleBar(d.normalizedWeight))
      .attr('fill', d => d.isSearchWord ? '#0d6efd' : '#b3d9ff')
      .on('mouseover', function(event, d) {
        if (!d.isSearchWord) {
          d3.select(this).attr('fill', '#66b3ff');
        }
        d3.select(this.parentNode).classed('hover', true);
      })
      .on('mouseout', function(event, d) {
        if (!d.isSearchWord) {
          d3.select(this).attr('fill', '#b3d9ff');
        }
        d3.select(this.parentNode).classed('hover', false);
      })
      .on('click', function(event, d) {
        // Navigate to word permalink
        window.page(`/word/${encodeURIComponent(d.word)}`);
      });

    // Add word labels
    wordGroups.append('text')
      .attr('x', width / (4 * maxWords))
      .attr('y', -5)
      .attr('transform', `rotate(-45)`)
      .attr('text-anchor', 'start')
      .style('font-size', '12px') // Increased from 10px
      .style('font-weight', d => d.isSearchWord ? 'bold' : 'normal')
      .style('fill', d => d.isSearchWord ? '#0d6efd' : '#333')
      .text(d => d.word)
      .on('mouseover', function() {
        d3.select(this.parentNode).classed('hover', true);
      })
      .on('mouseout', function() {
        d3.select(this.parentNode).classed('hover', false);
      })
      .on('click', function(event, d) {
        // Navigate to word permalink
        window.page(`/word/${encodeURIComponent(d.word)}`);
      });

    // Add CSS for hover effects
    const style = document.createElement('style');
    style.textContent = `
      .word.hover rect { opacity: 0.8; }
      .word.hover text { font-weight: bold; }
      .topic-label:hover { fill: #0d6efd; }
    `;
    document.head.appendChild(style);
  }
}
