import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const imageUpload = document.getElementById('image-upload');
const imageProxy = document.getElementById('image-proxy');

let detector;
let isLive = true;

// --- PWA 強制更新邏輯 ---
if ('serviceWorker' in navigator) {
    let refreshing = false;
    // 監測到新版 SW 接管後，自動重新整理網頁
    navigator.serviceWorker.addEventListener('controllerchange', () => {
        if (refreshing) return;
        refreshing = true;
        window.location.reload();
    });

    window.addEventListener('load', () => {
        navigator.serviceWorker.register('./sw.js').then(reg => {
            // 每 5 分鐘主動檢查一次伺服器是否有新版本
            setInterval(() => { reg.update(); }, 1000 * 60 * 5);
        });
    });
}

// --- YOLO 初始化與偵測邏輯 ---
async function init() {
    try {
        status.innerText = "⏳ 載入模型中...";
        // 建議開啟 WebGPU 加速 (若硬體支援)
        // 修改 app.js 初始化
       detector = await pipeline('object-detection', 'Xenova/yolov8n', {
       device: 'webgpu' // 強制開啟網頁顯示卡加速
       });
        status.innerText = "✅ 偵測中";
        
        await setupCamera();
        detectFrame();
    } catch (e) {
        status.innerText = "❌ 載入失敗: " + e.message;
    }
}

async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
            audio: false
        });
        video.srcObject = stream;
        return new Promise(resolve => {
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                resolve();
            };
        });
    } catch (e) {
        status.innerText = "❌ 無法開啟相機";
    }
}

async function detectFrame() {
    if (!isLive) return;
    if (detector && video.readyState >= 2) {
        const results = await detector(video);
        renderDetections(results);
    }
    requestAnimationFrame(detectFrame);
}

function renderDetections(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!isLive) ctx.drawImage(imageProxy, 0, 0, canvas.width, canvas.height);

    results.forEach(p => {
        const { xmin, ymin, xmax, ymax } = p.box;
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 4;
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
        ctx.fillStyle = '#00FF00';
        ctx.font = '18px Arial';
        ctx.fillText(`${p.label} ${Math.round(p.score*100)}%`, xmin, ymin > 20 ? ymin - 10 : 20);
    });
}

// --- 照片模式 ---
imageUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    isLive = false;
    status.innerText = "⏳ 分析照片中...";
    const url = URL.createObjectURL(file);
    imageProxy.src = url;
    imageProxy.style.display = 'block';
    video.style.display = 'none';

    imageProxy.onload = async () => {
        canvas.width = imageProxy.naturalWidth;
        canvas.height = imageProxy.naturalHeight;
        const results = await detector(imageProxy.src);
        renderDetections(results);
        status.innerText = "✅ 分析完成 (點擊畫面恢復鏡頭)";
        URL.revokeObjectURL(url);
    };
});

canvas.onclick = () => {
    if (!isLive) {
        isLive = true;
        imageProxy.style.display = 'none';
        video.style.display = 'block';
        status.innerText = "✅ 偵測中";
        setupCamera().then(detectFrame);
    }
};

init();
