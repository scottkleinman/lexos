/* This script is injected by the Material for MkDocs theme to ensure that the version selector in the header always displays the correct version number, even if the theme's JavaScript rewrites it later. */
;(function () {
  var TARGET_VERSION = 'v0.1.0-beta.31'
  var SITE_BASE_URL = 'https://scottkleinman.github.io/lexos/'
  var HEADER_BUTTON_SELECTOR = '.md-version__current'
  var HEADER_ELLIPSIS_SELECTOR = '.md-header__ellipsis'
  var HEADER_TOPIC_SELECTOR = '.md-header__topic'
  var VERSION_CONTAINER_SELECTOR = '.md-version'
  var retryDelays = [0, 50, 150, 500, 1000, 2000]

  function createVersionContainer () {
    var container = document.createElement('div')
    container.className = 'md-version'

    var button = document.createElement('button')
    button.className = 'md-version__current'
    button.setAttribute('aria-label', 'Select version')
    button.textContent = TARGET_VERSION

    var list = document.createElement('ul')
    list.className = 'md-version__list'

    var versions = [
      { label: 'dev', value: 'dev' },
      { label: TARGET_VERSION, value: TARGET_VERSION }
    ]

    versions.forEach(function (version) {
      var item = document.createElement('li')
      item.className = 'md-version__item'

      var link = document.createElement('a')
      link.className = 'md-version__link'
      link.href = SITE_BASE_URL + version.value + '/'
      link.textContent = version.label

      item.appendChild(link)
      list.appendChild(item)
    })

    container.appendChild(button)
    container.appendChild(list)
    return container
  }

  function ensureVersionContainerPlacement () {
    var ellipsis = document.querySelector(HEADER_ELLIPSIS_SELECTOR)
    if (!ellipsis) {
      return
    }

    var firstTopic = ellipsis.querySelector(HEADER_TOPIC_SELECTOR)
    if (!firstTopic || !firstTopic.parentNode) {
      return
    }

    var titleSpan = firstTopic.querySelector('.md-ellipsis')

    var versionContainer = ellipsis.querySelector(VERSION_CONTAINER_SELECTOR)
    if (!versionContainer) {
      versionContainer = createVersionContainer()
    }

    // Place the version selector inside the first topic, after the title span.
    if (titleSpan) {
      if (versionContainer.parentNode !== firstTopic || versionContainer.previousElementSibling !== titleSpan) {
        firstTopic.insertBefore(versionContainer, titleSpan.nextSibling)
      }
    } else if (versionContainer.parentNode !== firstTopic) {
      firstTopic.appendChild(versionContainer)
    }
  }

  function applyVersionState () {
    var currentButton = document.querySelector(HEADER_BUTTON_SELECTOR)
    if (currentButton && currentButton.textContent !== TARGET_VERSION) {
      currentButton.textContent = TARGET_VERSION
    }
  }

  function init () {
    retryDelays.forEach(function (delay) {
      setTimeout(function () {
        ensureVersionContainerPlacement()
        applyVersionState()
      }, delay)
    })

    // Keep the closed-menu label pinned even if theme scripts rewrite it later.
    setInterval(function () {
      ensureVersionContainerPlacement()
      applyVersionState()
    }, 1000)
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init)
  } else {
    init()
  }
})()
