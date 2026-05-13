import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19';

// 設定環境變數：禁用 WebGPU 實驗性功能，強制使用穩定的 WASM
env.allowLocalModels = false;

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
 * 1. PWA 更新監測 
 **/
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.addEventListener('controllerchange', () => {
        window.location.reload();
    });
}

/** 
 * 2. 啟動相機 
 **/
async function startCamera() {
    try {
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
        }

        const constraints = {
            video: { 
                facingMode: 'environment', // 強制後鏡頭
                width: { ideal: 640 },    // 手機版建議不用太高，減少運算負擔
                height: { ideal: 480 }
            },
            audio: false
        };

        localStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = localStream;
        await video.play();
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        video.style.display = 'block';
        imageProxy.style.display = 'none';
        isLive = true;
        return true;
    } catch (e) {
        status.innerText = "❌ 相機失敗: " + e.message;
        return false;
    }
}

/** 
 * 3. 初始化 AI (強制 CPU 模式)
 **/
async function initAI() {
    try {
        status.innerText = "⏳ 載入 AI 模型中 (CPU 模式)...";
        
        // 使用 WASM 設備，dtype 改為 fp32 確保最高相容性
        detector = await pipeline('object-detection', 'onnx-community/yolov10n', {
            device: 'wasm', 
            dtype: 'fp32'
        });

        status.innerText = "✅ 系統就緒 (CPU)";
        detectFrame();
    } catch (e) {
        console.error("AI 初始化失敗:", e);
        status.innerText = "❌ 模型載入失敗，請重整頁面";
    }
}

/** 
 * 4. 偵測迴圈 
 **/
async function detectFrame() {
    if (!isLive) return;

    if (detector && video.readyState >= 2) {
        try {
            // 在 CPU 模式下，我們可以手動限制偵測頻率來避免手機過熱
            const results = await detector(video);
            renderDetections(results);
        } catch (err) {
            console.error(err);
        }
    }
    // 使用 requestAnimationFrame 保持平滑，但 CPU 速度取決於處理器效能
    requestAnimationFrame(detectFrame);
}

/** 
 * 5. 繪圖邏輯 
 **/
function renderDetections(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!isLive) ctx.drawImage(imageProxy, 0, 0, canvas.width, canvas.height);

    results.forEach(p => {
        const { xmin, ymin, xmax, ymax } = p.box;
        
        // 框線樣式
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 3;
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
        
        // 文字標籤
        const label = `${p.label} ${Math.round(p.score * 100)}%`;
        ctx.fillStyle = '#00FF00';
        ctx.font = 'bold 14px Arial';
        ctx.fillText(label, xmin, ymin > 15 ? ymin - 5 : ymin + 15);
    });
}

/** 
 * 6. 照片分析 
 **/
imageUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    isLive = false;
    status.innerText = "⏳ 分析中...";
    
    const url = URL.createObjectURL(file);
    imageProxy.src = url;
    imageProxy.style.display = 'block';
    video.style.display = 'none';

    imageProxy.onload = async () => {
        canvas.width = imageProxy.naturalWidth;
        canvas.height = imageProxy.naturalHeight;
        
        const results = await detector(imageProxy.src);
        renderDetections(results);
        status.innerText = "✅ 完成 (點擊畫面回相機)";
        URL.revokeObjectURL(url);
    };
});

/** 
 * 7. 點擊畫面恢復 
 **/
canvas.onclick = async () => {
    if (!isLive) {
        status.innerText = "🔄 重啟鏡頭...";
        const ok = await startCamera();
        if (ok) detectFrame();
    }
};

// 執行順序：先顯示相機，再啟動 AI
startCamera().then(() => {
    initAI();
});
