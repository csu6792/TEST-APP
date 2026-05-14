const CACHE_NAME = "yolo-ai-cache-v1";

const urlsToCache = [

  "./",
  "./index.html",
  "./style.css",
  "./app.js",
  "./manifest.json",
  "./model/yolo.onnx"

];

self.addEventListener("install", (event) => {

  event.waitUntil(

    caches.open(CACHE_NAME)
      .then((cache) => {

        return cache.addAll(urlsToCache);

      })

  );

});

self.addEventListener("fetch", (event) => {

  event.respondWith(

    caches.match(event.request)
      .then((response) => {

        return response || fetch(event.request);

      })

  );

});
