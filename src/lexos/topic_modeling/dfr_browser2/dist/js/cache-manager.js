/**
 * Cache Manager for DFR Browser 2
 * Handles IndexedDB caching of parsed data files
 */

const CacheManager = {
  dbName: 'dfr-browser-cache',
  version: 2, // Incremented to add VOCABULARY store
  db: null,

  // Cache store names
  stores: {
    STATE: 'state-data',
    METADATA: 'metadata',
    TOPIC_KEYS: 'topic-keys',
    DOC_TOPICS: 'doc-topics',
    BIBLIOGRAPHY: 'bibliography',
    VOCABULARY: 'vocabulary'
  },

  /**
   * Initialize IndexedDB
   */
  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        console.log('[Cache] IndexedDB initialized');
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        // Create object stores if they don't exist
        Object.values(this.stores).forEach(storeName => {
          if (!db.objectStoreNames.contains(storeName)) {
            const store = db.createObjectStore(storeName, { keyPath: 'id' });
            store.createIndex('timestamp', 'timestamp', { unique: false });
            store.createIndex('version', 'version', { unique: false });
          }
        });

        console.log('[Cache] Object stores created');
      };
    });
  },

  /**
   * Store data in cache
   */
  async set(storeName, key, data, metadata = {}) {
    if (!this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      try {
        const transaction = this.db.transaction([storeName], 'readwrite');
        const store = transaction.objectStore(storeName);

        const cacheEntry = {
          id: key,
          data: data,
          timestamp: Date.now(),
          ...metadata
        };

        const request = store.put(cacheEntry);
        request.onsuccess = () => {
          console.log(`[Cache] Stored ${key} in ${storeName}`);
          resolve(true);
        };
        request.onerror = () => reject(request.error);
      } catch (err) {
        reject(err);
      }
    });
  },

  /**
   * Retrieve data from cache
   */
  async get(storeName, key, version = null) {
    if (!this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      try {
        const transaction = this.db.transaction([storeName], 'readonly');
        const store = transaction.objectStore(storeName);
        const request = store.get(key);

        request.onsuccess = () => {
          const result = request.result;

          // Check if cache exists and version matches (if provided)
          if (result && (!version || result.version === version)) {
            console.log(`[Cache] Retrieved ${key} from ${storeName}`);
            resolve(result);
          } else {
            resolve(null);
          }
        };
        request.onerror = () => reject(request.error);
      } catch (err) {
        reject(err);
      }
    });
  },

  /**
   * Check if cached data exists and is valid
   */
  async has(storeName, key, version = null) {
    try {
      const cached = await this.get(storeName, key, version);
      return cached !== null;
    } catch (err) {
      console.warn('[Cache] Error checking cache:', err);
      return false;
    }
  },

  /**
   * Delete specific cache entry
   */
  async delete(storeName, key) {
    if (!this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      try {
        const transaction = this.db.transaction([storeName], 'readwrite');
        const store = transaction.objectStore(storeName);
        const request = store.delete(key);

        request.onsuccess = () => {
          console.log(`[Cache] Deleted ${key} from ${storeName}`);
          resolve(true);
        };
        request.onerror = () => reject(request.error);
      } catch (err) {
        reject(err);
      }
    });
  },

  /**
   * Clear entire cache store
   */
  async clear(storeName) {
    if (!this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      try {
        const transaction = this.db.transaction([storeName], 'readwrite');
        const store = transaction.objectStore(storeName);
        const request = store.clear();

        request.onsuccess = () => {
          console.log(`[Cache] Cleared ${storeName}`);
          resolve(true);
        };
        request.onerror = () => reject(request.error);
      } catch (err) {
        reject(err);
      }
    });
  },

  /**
   * Clear all caches
   */
  async clearAll() {
    const promises = Object.values(this.stores).map(store =>
      this.clear(store).catch(err => console.warn(`Failed to clear ${store}:`, err))
    );
    await Promise.all(promises);
    console.log('[Cache] All caches cleared');
  },

  /**
   * Get cache statistics
   */
  async getStats() {
    if (!this.db) {
      await this.init();
    }

    const stats = {};

    for (const [name, storeName] of Object.entries(this.stores)) {
      try {
        const transaction = this.db.transaction([storeName], 'readonly');
        const store = transaction.objectStore(storeName);
        const count = await new Promise((resolve) => {
          const request = store.count();
          request.onsuccess = () => resolve(request.result);
          request.onerror = () => resolve(0);
        });
        stats[name] = count;
      } catch (err) {
        stats[name] = 0;
      }
    }

    return stats;
  },

  /**
   * Get detailed cache statistics including size
   */
  async getCacheStats() {
    if (!this.db) {
      await this.init();
    }

    let totalFiles = 0;
    let totalSize = 0;
    const storeStats = {};

    for (const [name, storeName] of Object.entries(this.stores)) {
      try {
        const transaction = this.db.transaction([storeName], 'readonly');
        const store = transaction.objectStore(storeName);

        const items = await new Promise((resolve) => {
          const request = store.getAll();
          request.onsuccess = () => resolve(request.result);
          request.onerror = () => resolve([]);
        });

        let storeSize = 0;
        items.forEach(item => {
          const itemSize = JSON.stringify(item).length;
          storeSize += itemSize;
        });

        totalFiles += items.length;
        totalSize += storeSize;

        storeStats[name] = {
          count: items.length,
          size: storeSize
        };
      } catch (err) {
        storeStats[name] = { count: 0, size: 0 };
      }
    }

    return { totalFiles, totalSize, stores: storeStats };
  },

  /**
   * Clear a specific object store
   * @param {string} storeName - Name of the store to clear
   */
  async clearStore(storeName) {
    console.log(`[CacheManager] clearStore called with: ${storeName}`);

    if (!this.db) {
      console.log('[CacheManager] Database not initialized, initializing...');
      await this.init();
    }

    // Validate store name
    console.log('[CacheManager] Available stores:', Object.values(this.stores));
    if (!Object.values(this.stores).includes(storeName)) {
      const error = `Store ${storeName} does not exist`;
      console.error(`[CacheManager] ${error}`);
      throw new Error(error);
    }

    console.log(`[CacheManager] Clearing store: ${storeName}`);
    const result = await this.clear(storeName);
    console.log(`[CacheManager] Store ${storeName} cleared successfully`);
    return result;
  },

  /**
   * Clear all data from all object stores
   */
  async clearAllStores() {
    return this.clearAll();
  },

  /**
   * Delete the entire IndexedDB database
   * @param {string} dbName - Name of the database to delete
   */
  async deleteDatabase(dbName = null) {
    const targetDb = dbName || this.dbName;

    // Close existing connection
    if (this.db) {
      this.db.close();
      this.db = null;
    }

    return new Promise((resolve, reject) => {
      const request = indexedDB.deleteDatabase(targetDb);

      request.onsuccess = () => {
        console.log(`[CacheManager] Database ${targetDb} deleted successfully`);
        resolve();
      };

      request.onerror = () => {
        console.error(`[CacheManager] Error deleting database ${targetDb}:`, request.error);
        reject(request.error);
      };

      request.onblocked = () => {
        console.warn(`[CacheManager] Delete blocked for ${targetDb}. Close all tabs using this database.`);
      };
    });
  },

  /**
   * Remove old cache entries (older than maxAge milliseconds)
   */
  async pruneOldEntries(maxAge = 7 * 24 * 60 * 60 * 1000) {
    if (!this.db) {
      await this.init();
    }

    const cutoffTime = Date.now() - maxAge;
    let totalDeleted = 0;

    for (const storeName of Object.values(this.stores)) {
      try {
        const transaction = this.db.transaction([storeName], 'readwrite');
        const store = transaction.objectStore(storeName);
        const index = store.index('timestamp');
        const range = IDBKeyRange.upperBound(cutoffTime);

        await new Promise((resolve, reject) => {
          const request = index.openCursor(range);
          request.onsuccess = (event) => {
            const cursor = event.target.result;
            if (cursor) {
              cursor.delete();
              totalDeleted++;
              cursor.continue();
            } else {
              resolve();
            }
          };
          request.onerror = () => reject(request.error);
        });
      } catch (err) {
        console.warn(`Failed to prune ${storeName}:`, err);
      }
    }

    console.log(`[Cache] Pruned ${totalDeleted} old entries`);
    return totalDeleted;
  }
};

// Initialize cache on module load
CacheManager.init().catch(err => {
  console.warn('[Cache] Failed to initialize:', err);
});

export default CacheManager;
