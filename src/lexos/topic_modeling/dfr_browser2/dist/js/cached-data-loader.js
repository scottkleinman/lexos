/**
 * Cached Data Loader for DFR Browser 2
 * Wraps data loading with IndexedDB caching
 */

import CacheManager from './cache-manager.js';

const CachedDataLoader = {

  /**
   * Return a stable model ID for the current deployment.
   * Uses the base path (i.e. the sub-directory the app is served from) so
   * that different model deployments always have distinct cache keys even
   * when running on the same origin.
   */
  getModelId() {
    // window.dfrBasePath is set by the inline script in index.html
    const base = (window.dfrBasePath || window.location.pathname).replace(/\/$/, '');
    return base || 'default';
  },

  /**
   * Build a model-scoped cache key so that data from different models is
   * never mixed together in IndexedDB.
   */
  buildCacheKey(url) {
    return `${this.getModelId()}::${url}`;
  },

  /**
   * Get file metadata (size, last modified) for cache versioning.
   * cache: 'no-store' ensures the browser never returns stale headers from
   * its own HTTP cache — the server always answers the HEAD request.
   */
  async getFileInfo(url) {
    try {
      const response = await fetch(url, { method: 'HEAD', cache: 'no-store' });
      if (!response.ok) return null;

      const size = response.headers.get('Content-Length') || '0';
      const lastModified = response.headers.get('Last-Modified') || Date.now().toString();

      return {
        size: size,
        lastModified: lastModified,
        version: `${size}-${lastModified}`
      };
    } catch (err) {
      console.warn('[CachedLoader] Could not get file info:', err);
      return null;
    }
  },

  /**
   * Load metadata with caching
   */
  async loadMetadata(url, parser) {
    const cacheKey = this.buildCacheKey(url);
    const fileInfo = await this.getFileInfo(url);
    const version = fileInfo ? fileInfo.version : null;

    // Try to get from cache
    try {
      const cached = await CacheManager.get(CacheManager.stores.METADATA, cacheKey, version);
      if (cached && cached.data) {
        console.log('[CachedLoader] Using cached metadata');
        return cached.data;
      }
    } catch (err) {
      console.warn('[CachedLoader] Cache retrieval failed:', err);
    }

    // Load from file (bypass browser HTTP cache so we always get fresh data)
    console.log('[CachedLoader] Loading metadata from file');
    const response = await fetch(url, { cache: 'no-store' });
    const text = await response.text();
    const data = parser(text);

    // Store in cache
    if (fileInfo) {
      try {
        await CacheManager.set(
          CacheManager.stores.METADATA,
          cacheKey,
          data,
          { version: version, size: fileInfo.size }
        );
      } catch (err) {
        console.warn('[CachedLoader] Failed to cache metadata:', err);
      }
    }

    return data;
  },

  /**
   * Load topic keys with caching
   */
  async loadTopicKeys(url, parser) {
    const cacheKey = this.buildCacheKey(url);
    const fileInfo = await this.getFileInfo(url);
    const version = fileInfo ? fileInfo.version : null;

    // Try cache
    try {
      const cached = await CacheManager.get(CacheManager.stores.TOPIC_KEYS, cacheKey, version);
      if (cached && cached.data) {
        console.log('[CachedLoader] Using cached topic keys');
        return cached.data;
      }
    } catch (err) {
      console.warn('[CachedLoader] Cache retrieval failed:', err);
    }

    // Load from file
    console.log('[CachedLoader] Loading topic keys from file');
    const response = await fetch(url, { cache: 'no-store' });
    const text = await response.text();
    const data = parser(text);

    // Cache
    if (fileInfo) {
      try {
        await CacheManager.set(
          CacheManager.stores.TOPIC_KEYS,
          cacheKey,
          data,
          { version: version, size: fileInfo.size }
        );
      } catch (err) {
        console.warn('[CachedLoader] Failed to cache topic keys:', err);
      }
    }

    return data;
  },

  /**
   * Load doc-topics with caching
   */
  async loadDocTopics(url, parser) {
    const cacheKey = this.buildCacheKey(url);
    const fileInfo = await this.getFileInfo(url);
    const version = fileInfo ? fileInfo.version : null;

    // Try cache
    try {
      const cached = await CacheManager.get(CacheManager.stores.DOC_TOPICS, cacheKey, version);
      if (cached && cached.data) {
        console.log('[CachedLoader] Using cached doc-topics');
        return cached.data;
      }
    } catch (err) {
      console.warn('[CachedLoader] Cache retrieval failed:', err);
    }

    // Load from file
    console.log('[CachedLoader] Loading doc-topics from file');
    const response = await fetch(url, { cache: 'no-store' });
    const text = await response.text();
    const data = parser(text);

    // Cache
    if (fileInfo) {
      try {
        await CacheManager.set(
          CacheManager.stores.DOC_TOPICS,
          cacheKey,
          data,
          { version: version, size: fileInfo.size }
        );
      } catch (err) {
        console.warn('[CachedLoader] Failed to cache doc-topics:', err);
      }
    }

    return data;
  },

  /**
   * Generic cached file loader
   */
  async loadFile(url, storeName, parser = null) {
    const cacheKey = this.buildCacheKey(url);
    const fileInfo = await this.getFileInfo(url);
    const version = fileInfo ? fileInfo.version : null;

    // Try cache
    try {
      const cached = await CacheManager.get(storeName, cacheKey, version);
      if (cached && cached.data) {
        console.log(`[CachedLoader] Using cached data from ${storeName}`);
        return cached.data;
      }
    } catch (err) {
      console.warn('[CachedLoader] Cache retrieval failed:', err);
    }

    // Load from file
    console.log(`[CachedLoader] Loading from file: ${url}`);
    const response = await fetch(url, { cache: 'no-store' });
    const text = await response.text();
    const data = parser ? parser(text) : text;

    // Cache
    if (fileInfo) {
      try {
        await CacheManager.set(
          storeName,
          cacheKey,
          data,
          { version: version, size: fileInfo.size }
        );
      } catch (err) {
        console.warn(`[CachedLoader] Failed to cache in ${storeName}:`, err);
      }
    }

    return data;
  },

  /**
   * Load full vocabulary with caching
   * Uses state file configuration to determine cache key
   */
  async loadFullVocabulary(extractFn) {
    // Get state file info from config
    let stateFileUrl = 'sample_data/topic-state.gz';
    try {
      const response = await fetch('config.json');
      const config = await response.json();
      stateFileUrl = config.topic_state_file || stateFileUrl;
    } catch (err) {
      console.warn('[CachedLoader] Could not load config for state file path');
    }

    const cacheKey = this.buildCacheKey('full-vocabulary');
    const fileInfo = await this.getFileInfo(stateFileUrl);
    const version = fileInfo ? fileInfo.version : null;

    // Try to get from cache
    try {
      const cached = await CacheManager.get(CacheManager.stores.VOCABULARY, cacheKey, version);
      if (cached && cached.data) {
        console.log('[CachedLoader] Using cached full vocabulary');
        return cached.data;
      }
    } catch (err) {
      console.warn('[CachedLoader] Cache retrieval failed for vocabulary:', err);
    }

    // Extract vocabulary from state file
    console.log('[CachedLoader] Extracting full vocabulary from state file');
    const vocabulary = await extractFn();

    // Cache the vocabulary
    if (fileInfo && vocabulary && vocabulary.length > 0) {
      try {
        await CacheManager.set(
          CacheManager.stores.VOCABULARY,
          cacheKey,
          vocabulary,
          { version: version, wordCount: vocabulary.length }
        );
        console.log(`[CachedLoader] Cached ${vocabulary.length} words`);
      } catch (err) {
        console.warn('[CachedLoader] Failed to cache vocabulary:', err);
      }
    }

    return vocabulary;
  }
};

export default CachedDataLoader;
