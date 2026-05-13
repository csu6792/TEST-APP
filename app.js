import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const imageUpload = document.getElementById('image-upload');
const imageProxy = document.getElementById('image-proxy');

let detector;
let isLive = true;
let localStream = null;

// PWA 更新邏輯
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.addEventListener('controllerchange', () => window.location.reload());
}

/** 1. 初始化模型 **/
async function init() {
    try {
        status.innerText = "⏳ 載入 YOLOv10 (WebGPU)...";
        detector = await pipeline('object-detection', 'onnx-community/yolov10n', {
            device: 'webgpu',
        });
        status.innerText = "✅ 模型就緒，開啟相機...";
        await startCamera(); // 預設開啟相機
        detectFrame();
    } catch (e) {
        console.error(e);
        status.innerText = "⚠️ GPU 不支援，切換至 CPU...";
        detector = await pipeline('object-detection', 'onnx-community/yolov10n');
        await startCamera();
        detectFrame();
    }
}

/** 2. 啟動相機 (封裝成獨立函式) **/
async function startCamera() {
    try {
        // 如果原本有串流，先停止它
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
        }

        localStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
            audio: false
        });
        
        video.srcObject = localStream;
        video.style.display = 'block';
        imageProxy.style.display = 'none';
        isLive = true;

        return new Promise(resolve => {
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                resolve();
            };
        });
    } catch (e) {
        status.innerText = "❌ 相機開啟失敗: " + e.message;
    }
}

/** 3. 偵測迴圈 **/
async function detectFrame() {
    if (!isLive) return; 
    
    if (detector && video.readyState >= 2) {
        const results = await detector(video);
        renderDetections(results);
    }
    requestAnimationFrame(detectFrame);
}

/** 4. 繪製結果 **/
function renderDetections(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 如果是照片模式，要把照片畫在最底層
    if (!isLive) {
        ctx.drawImage(imageProxy, 0, 0, canvas.width, canvas.height);
    }

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

/** 5. 照片模式處理 **/
imageUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    isLive = false; // 停止 detectFrame 的迴圈邏輯
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
        status.innerText = "✅ 分析完成 (點擊畫面恢復相機)";
        URL.revokeObjectURL(url);
    };
});

/** 6. 點擊 Canvas 或影像恢復相機模式 **/
canvas.onclick = async () => {
    if (!isLive) {
        status.innerText = "🔄 切換回相機...";
        await startCamera();
        detectFrame();
    }
};

init();
