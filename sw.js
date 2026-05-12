// 1. 定義快取名稱，每次更新介面或 app.js 時，請手動將 v1 改成 v2, v3...
const CACHE_NAME = 'ai-search-v1';

// 2. 列出需要離線存取的靜態資源
// 注意：如果你的專案在 GitHub Pages 的子目錄，路徑要包含專案名稱，例如 '/repo-name/index.html'
const ASSETS_TO_CACHE = [
  './',
  './index.html',
  './style.css',
  './app.js',
  './manifest.json',
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js'
];

// --- Service Worker 生命週期管理 ---

// 安裝階段：快取資源
self.addEventListener('install', (event) => {
  console.log('[Service Worker] Installing New Version...');
  
  // 強制讓新版 SW 跳過等待階段，立即進入 Activate
  self.skipWaiting();

  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Caching App Shell & Assets');
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
});

// 激活階段：清理舊快取
self.addEventListener('activate', (event) => {
  console.log('[Service Worker] Activating & Cleaning Old Caches...');
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cache) => {
          if (cache !== CACHE_NAME) {
            console.log('[Service Worker] Deleting Old Cache:', cache);
            return caches.delete(cache);
          }
        })
      );
    }).then(() => {
      // 確保新版 SW 立即控制所有開放的標籤頁
      return self.clients.claim();
    })
  );
});

// 攔截請求
self.addEventListener('fetch', (event) => {
  // 對於 AI 模型檔案 (.onnx) 或 函式庫，採用 Cache-First 策略
  if (event.request.url.includes('.onnx') || event.request.url.includes('cdn')) {
    event.respondWith(
      caches.match(event.request).then((response) => {
        return response || fetch(event.request).then((networkResponse) => {
          return caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, networkResponse.clone());
            return networkResponse;
          });
        });
      })
    );
    return;
  }

  // 對於 HTML/CSS/JS，採用 Network-First 策略，確保介面是最新的
  event.respondWith(
    fetch(event.request).catch(() => {
      return caches.match(event.request);
    })
  );
});
