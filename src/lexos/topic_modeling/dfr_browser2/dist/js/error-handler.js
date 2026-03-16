/**
 * Error Handler for DFR Browser 2
 * Provides user-friendly error messages and recovery suggestions
 */

const ErrorHandler = {

  /**
   * Error types and their user-friendly messages
   */
  errorTypes: {
    FILE_NOT_FOUND: {
      title: 'File Not Found',
      icon: 'bi-file-earmark-x',
      color: 'danger'
    },
    PARSE_ERROR: {
      title: 'Data Parsing Error',
      icon: 'bi-exclamation-triangle',
      color: 'warning'
    },
    VALIDATION_ERROR: {
      title: 'Data Validation Error',
      icon: 'bi-shield-exclamation',
      color: 'warning'
    },
    NETWORK_ERROR: {
      title: 'Network Error',
      icon: 'bi-wifi-off',
      color: 'danger'
    },
    CACHE_ERROR: {
      title: 'Cache Error',
      icon: 'bi-hdd-x',
      color: 'warning'
    },
    CONFIG_ERROR: {
      title: 'Configuration Error',
      icon: 'bi-gear-wide-connected',
      color: 'warning'
    },
    INTEGRITY_ERROR: {
      title: 'Data Integrity Error',
      icon: 'bi-database-x',
      color: 'danger'
    }
  },

  /**
   * Show error modal with detailed information
   */
  showError(type, message, details = null, suggestions = []) {
    const errorInfo = this.errorTypes[type] || this.errorTypes.PARSE_ERROR;

    // Create error modal HTML
    const modalHtml = `
      <div class="modal fade" id="errorModal" tabindex="-1" aria-labelledby="errorModalLabel" aria-hidden="true">
        <div class="modal-dialog modal-lg">
          <div class="modal-content">
            <div class="modal-header bg-${errorInfo.color} text-white">
              <h5 class="modal-title" id="errorModalLabel">
                <i class="${errorInfo.icon} me-2"></i>${errorInfo.title}
              </h5>
              <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
              <div class="alert alert-${errorInfo.color}" role="alert">
                <strong>${message}</strong>
              </div>

              ${details ? `
                <div class="card mb-3">
                  <div class="card-header">
                    <strong>Details</strong>
                  </div>
                  <div class="card-body">
                    <pre class="mb-0" style="white-space: pre-wrap; font-size: 0.875rem;">${this.escapeHtml(details)}</pre>
                  </div>
                </div>
              ` : ''}

              ${suggestions.length > 0 ? `
                <div class="card">
                  <div class="card-header bg-info text-white">
                    <i class="bi bi-lightbulb me-2"></i><strong>Suggestions</strong>
                  </div>
                  <div class="card-body">
                    <ul class="mb-0">
                      ${suggestions.map(s => `<li>${s}</li>`).join('')}
                    </ul>
                  </div>
                </div>
              ` : ''}
            </div>
            <div class="modal-footer">
              <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
              ${suggestions.length > 0 ? `
                <button type="button" class="btn btn-primary" onclick="location.reload()">
                  <i class="bi bi-arrow-clockwise me-2"></i>Retry
                </button>
              ` : ''}
            </div>
          </div>
        </div>
      </div>
    `;

    // Remove existing error modal if any
    const existing = document.getElementById('errorModal');
    if (existing) existing.remove();

    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('errorModal'));
    modal.show();

    // Log to console
    console.error(`[${type}] ${message}`, details);
  },

  /**
   * Escape HTML to prevent XSS
   */
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },

  /**
   * Handle file loading errors
   */
  handleFileError(filename, error, fileType = 'data file') {
    const suggestions = [
      `Check that <code>${filename}</code> exists in the correct location`,
      `Verify that the file path in <code>config.json</code> is correct`,
      `Make sure the server is running and accessible`,
      `Check browser console for additional error details`
    ];

    if (error.message && error.message.includes('404')) {
      this.showError(
        'FILE_NOT_FOUND',
        `Could not load ${fileType}: ${filename}`,
        `HTTP 404: File not found`,
        suggestions
      );
    } else if (error.message && error.message.includes('network')) {
      this.showError(
        'NETWORK_ERROR',
        `Network error while loading ${fileType}`,
        error.message,
        [
          'Check your internet connection',
          'Verify the server is running',
          'Check for CORS issues in browser console',
          ...suggestions
        ]
      );
    } else {
      this.showError(
        'PARSE_ERROR',
        `Error loading ${fileType}: ${filename}`,
        error.message || error.toString(),
        suggestions
      );
    }
  },

  /**
   * Handle data parsing errors
   */
  handleParseError(filename, error, lineNumber = null) {
    const suggestions = [
      `Check that ${filename} is properly formatted`,
      `Verify the file encoding is UTF-8`,
      `Look for special characters or formatting issues`,
      `Try re-exporting the file from MALLET`,
      `Check the file isn't corrupted or truncated`
    ];

    if (lineNumber) {
      this.showError(
        'PARSE_ERROR',
        `Error parsing ${filename} at line ${lineNumber}`,
        error.message || error.toString(),
        suggestions
      );
    } else {
      this.showError(
        'PARSE_ERROR',
        `Error parsing ${filename}`,
        error.message || error.toString(),
        suggestions
      );
    }
  },

  /**
   * Handle data validation errors
   */
  handleValidationError(message, details, suggestions = []) {
    this.showError('VALIDATION_ERROR', message, details, suggestions);
  },

  /**
   * Handle data integrity errors
   */
  handleIntegrityError(message, details, suggestions = []) {
    this.showError('INTEGRITY_ERROR', message, details, suggestions);
  },

  /**
   * Handle configuration errors
   */
  handleConfigError(message, details = null) {
    const suggestions = [
      'Check that <code>config.json</code> is valid JSON',
      'Verify all required configuration fields are present',
      'Compare with the default <code>config.json</code> template',
      'Check for typos in file paths and settings'
    ];

    this.showError('CONFIG_ERROR', message, details, suggestions);
  },

  /**
   * Handle cache errors
   */
  handleCacheError(operation, error) {
    const suggestions = [
      'Try clearing the browser cache',
      'Check browser IndexedDB quota limits',
      'Try using a different browser',
      'Check browser console for storage errors'
    ];

    this.showError(
      'CACHE_ERROR',
      `Cache ${operation} failed`,
      error.message || error.toString(),
      suggestions
    );
  }
};

export default ErrorHandler;
