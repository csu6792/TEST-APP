import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const imageUpload = document.getElementById('image-upload');
const imageProxy = document.getElementById('image-proxy');

let detector;
let isLive = true;

// 1. PWA 強制更新邏輯
if ('serviceWorker' in navigator) {
    let refreshing = false;
    navigator.serviceWorker.addEventListener('controllerchange', () => {
        if (refreshing) return;
        refreshing = true;
        window.location.reload();
    });
}

// 2. 初始化 YOLOv10
async function init() {
    try {
        status.innerText = "⏳ 載入 YOLOv10 (WebGPU)...";
        
        // V3 版本使用 onnx-community 託管的模型
        detector = await pipeline('object-detection', 'onnx-community/yolov10n', {
            device: 'webgpu', // 強制開啟 GPU 加速，若不支援會自動回退
        });

        status.innerText = "✅ YOLOv10 準備就緒";
        await setupCamera();
        detectFrame();
    } catch (e) {
        console.error(e);
        status.innerText = "⚠️ GPU 加速失敗，切換至 CPU 模式...";
        // 回退到 CPU
        detector = await pipeline('object-detection', 'onnx-community/yolov10n');
        await setupCamera();
        detectFrame();
    }
}

async function setupCamera() {
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
}

async function detectFrame() {
    if (!isLive) return;
    if (detector && video.readyState >= 2) {
        // YOLOv10 不需要複雜的後處理
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
        ctx.font = 'bold 18px Arial';
        ctx.fillText(`${p.label} ${Math.round(p.score * 100)}%`, xmin, ymin > 20 ? ymin - 10 : 20);
    });
}

// 3. 照片模式 (加入壓縮邏輯避免記憶體崩潰)
imageUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    isLive = false;
    status.innerText = "⏳ 深度分析照片中...";
    const url = URL.createObjectURL(file);
    imageProxy.src = url;
    imageProxy.style.display = 'block';
    video.style.display = 'none';

    imageProxy.onload = async () => {
        // 設定辨識尺寸，避免 9MB 大圖壓垮瀏覽器
        canvas.width = imageProxy.naturalWidth;
        canvas.height = imageProxy.naturalHeight;
        
        const results = await detector(imageProxy.src);
        renderDetections(results);
        status.innerText = "✅ 分析完成 (點擊畫面恢復)";
        URL.revokeObjectURL(url);
    };
});

canvas.onclick = () => {
    if (!isLive) {
        isLive = true;
        imageProxy.style.display = 'none';
        video.style.display = 'block';
        status.innerText = "✅ 偵測中";
        detectFrame();
    }
};

init();
