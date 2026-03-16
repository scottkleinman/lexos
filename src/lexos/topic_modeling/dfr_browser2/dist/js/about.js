// About view - Shows information about the application and data
export function loadAboutView() {
  const mainView = document.getElementById('main-view');

  const config = window.dfrState.config || {};
  const application = config.application || {};
  const about = config.about || {};
  const brand = config.brand || {};

  // Get title and content from config, with fallbacks
  const title = about.title || `About ${brand.name || 'DFR Browser 2'}`;
  const content = about.content || `
This is a browser-based tool for exploring topic models created with MALLET.

Topic modeling is a method for finding abstract "topics" that occur in a collection of documents.
  `;

  // Initialize markdown-it renderer
  const md = window.markdownit({
    html: true,        // Enable HTML tags in source
    linkify: true,     // Auto-convert URLs to links
    typographer: true, // Enable smart quotes and other typographic replacements
    breaks: false      // Don't convert \n to <br>
  });

  // Render markdown content to HTML
  const formattedContent = md.render(content);

    mainView.innerHTML = `
    <div class="card">
      <div class="card-body">
        <h2>${title}</h2>

        <div class="mt-4">
          ${formattedContent}
        </div>

        ${about.version ? `
        <div class="mt-4">
          <p><small class="text-muted">Version: ${application.version}${about.lastUpdated ? ` | Last Updated: ${application.lastUpdated}` : ''}</small></p>
        </div>
        ` : ''}

        <div class="mt-4">
          <a href="${window.dfrBasePath || '/'}" class="btn btn-secondary">Back to Overview</a>
        </div>
      </div>
    </div>
      <div class="modal fade" id="aboutModal" tabindex="-1" role="dialog" aria-labelledby="about_title" aria-modal="true" aria-describedby="about_desc" aria-hidden="true">
        <div class="modal-dialog" role="document">
          <div class="modal-content">
            <div class="modal-header">
              <h4 class="modal-title" id="about_title">About</h4>
              <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close About Modal"></button>
            </div>
            <div class="modal-body" id="about_desc">
              <div id="about-content" role="document" aria-label="About Content"></div>
            </div>
            <div class="modal-footer" role="group" aria-label="About Modal Actions">
              <button type="button" class="btn btn-secondary" data-bs-dismiss="modal" aria-label="Close About Modal">Close</button>
            </div>
          </div>
        </div>
      </div>
  `;
}
