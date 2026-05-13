import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19';

// 取得 DOM 元素
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const imageUpload = document.getElementById('image-upload');
const imageProxy = document.getElementById('image-proxy');

let detector;
let isLive = true;
let localStream = null;

/** 
 * 1. PWA 強制更新邏輯 
 * 當 sw.js 版本更新時，偵測到新 Service Worker 接管立即重整頁面
 **/
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.addEventListener('controllerchange', () => {
        window.location.reload();
    });
}

/** 
 * 2. 啟動相機 
 * 優先執行，避免黑畫面
 **/
async function startCamera() {
    try {
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
        }

        const constraints = {
            video: { 
                facingMode: 'environment', // 優先使用後鏡頭
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };

        localStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = localStream;
        
        // 確保影片播放
        await video.play();
        
        // 同步畫布尺寸
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        video.style.display = 'block';
        imageProxy.style.display = 'none';
        isLive = true;
        
        return true;
    } catch (e) {
        console.error("相機啟動失敗:", e);
        status.innerText = "❌ 無法開啟相機，請檢查權限與 HTTPS";
        return false;
    }
}

/** 
 * 3. 初始化 AI 模型 (YOLOv10n)
 **/
async function initAI() {
    try {
        status.innerText = "⏳ 載入 AI 模型中 (約 25MB)...";
        
        // V3 支援 WebGPU 加速
        detector = await pipeline('object-detection', 'onnx-community/yolov10n', {
            device: 'webgpu', 
        });

        status.innerText = "✅ 辨識系統就緒";
        detectFrame();
    } catch (e) {
        console.warn("WebGPU 初始化失敗，嘗試回退至 CPU:", e);
        try {
            detector = await pipeline('object-detection', 'onnx-community/yolov10n');
            status.innerText = "✅ 辨識系統就緒 (CPU模式)";
            detectFrame();
        } catch (err2) {
            status.innerText = "❌ 模型載入失敗";
        }
    }
}

/** 
 * 4. 即時偵測迴圈 
 **/
async function detectFrame() {
    if (!isLive) return; // 照片模式下停止迴圈

    if (detector && video.readyState >= 2) {
        try {
            const results = await detector(video);
            renderDetections(results);
        } catch (err) {
            console.error("偵測出錯:", err);
        }
    }
    // 持續執行下一訊框
    requestAnimationFrame(detectFrame);
}

/** 
 * 5. 繪製偵測框 
 **/
function renderDetections(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 照片模式需手動補上底圖
    if (!isLive) {
        ctx.drawImage(imageProxy, 0, 0, canvas.width, canvas.height);
    }

    results.forEach(p => {
        const { xmin, ymin, xmax, ymax } = p.box;
        
        // 畫框
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 4;
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
        
        // 畫標籤背景
        const label = `${p.label} ${Math.round(p.score * 100)}%`;
        ctx.fillStyle = '#00FF00';
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(xmin, ymin > 25 ? ymin - 25 : ymin, textWidth + 10, 25);
        
        // 畫文字
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 16px Arial';
        ctx.fillText(label, xmin + 5, ymin > 25 ? ymin - 7 : ymin + 18);
    });
}

/** 
 * 6. 照片上傳處理 (支援 9MB+ 大檔案預處理) 
 **/
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
        // 設定畫布解析度為圖片原始大小
        canvas.width = imageProxy.naturalWidth;
        canvas.height = imageProxy.naturalHeight;
        
        // 執行辨識 (V3 會自動處理 Resize)
        const results = await detector(imageProxy.src);
        renderDetections(results);
        
        status.innerText = "✅ 分析完成 (點擊畫面恢復相機)";
        URL.revokeObjectURL(url);
    };
});

/** 
 * 7. 點擊恢復相機 
 **/
canvas.onclick = async () => {
    if (!isLive) {
        status.innerText = "🔄 恢復即時偵測...";
        const ok = await startCamera();
        if (ok) detectFrame();
    }
};

// 啟動流程：先開鏡頭，再載模型
startCamera().then(() => {
    initAI();
});
