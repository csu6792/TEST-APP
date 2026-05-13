import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19';

/** 
 * 第一部分：環境配置 (資工系 Debug 核心)
 * 強制手動指定遠端來源與安全性策略，解決 Unauthorized 錯誤
 **/
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.remoteHost = 'https://huggingface.co';
env.remotePathTemplate = '{model}/resolve/{revision}/';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const imageUpload = document.getElementById('image-upload');
const imageProxy = document.getElementById('image-proxy');

let detector;
let isLive = true;
let localStream = null;

// 1. 啟動相機
async function startCamera() {
    try {
        if (localStream) localStream.getTracks().forEach(t => t.stop());
        
        const constraints = {
            video: { facingMode: 'environment', width: 640 },
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
        status.innerText = "❌ 相機權限失敗";
        return false;
    }
}

// 2. 初始化 AI (加入 Fetch 權限突破)
async function initAI() {
    try {
        status.innerText = "⏳ 正在繞過權限限制載入模型...";
        
        // 使用 yolov8n 作為基礎測試，V3 版本會自動下載 .onnx 檔
        detector = await pipeline('object-detection', 'onnx-community/yolov8n', {
            device: 'wasm', // 手機端 CPU 模式最穩定
            // 【關鍵點】自定義 Fetch 邏輯，避免跨域權限檢查失敗
            fetch_callback: (url, options) => {
                return fetch(url, {
                    ...options,
                    referrerPolicy: "no-referrer", // 隱藏來源網址，解決 403/Unauthorized
                });
            }
        });

        status.innerText = "✅ 辨識系統就緒 (CPU)";
        detectFrame();
    } catch (e) {
        console.error("模型載入失敗:", e);
        status.style.color = "#ff4d4d";
        status.innerText = "❌ 載入失敗: " + e.message.substring(0, 30);
    }
}

// 3. 偵測迴圈
async function detectFrame() {
    if (!isLive) return;
    if (detector && video.readyState >= 2) {
        const results = await detector(video);
        renderDetections(results);
    }
    requestAnimationFrame(detectFrame);
}

// 4. 繪圖邏輯
function renderDetections(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!isLive) ctx.drawImage(imageProxy, 0, 0, canvas.width, canvas.height);

    results.forEach(p => {
        const { xmin, ymin, xmax, ymax } = p.box;
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 3;
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
        
        const label = `${p.label} ${Math.round(p.score * 100)}%`;
        ctx.fillStyle = '#00FF00';
        ctx.font = 'bold 16px Arial';
        ctx.fillText(label, xmin, ymin > 15 ? ymin - 5 : ymin + 15);
    });
}

// 5. 照片上傳與分析
imageUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    isLive = false;
    status.innerText = "⏳ 圖片分析中...";
    const url = URL.createObjectURL(file);
    imageProxy.src = url;
    imageProxy.style.display = 'block';
    video.style.display = 'none';

    imageProxy.onload = async () => {
        canvas.width = imageProxy.naturalWidth;
        canvas.height = imageProxy.naturalHeight;
        const results = await detector(imageProxy.src);
        renderDetections(results);
        status.innerText = "✅ 分析完成 (點擊畫面恢復)";
        URL.revokeObjectURL(url);
    };
});

// 點擊恢復相機
canvas.onclick = async () => {
    if (!isLive) {
        status.innerText = "🔄 重新啟動鏡頭...";
        await startCamera();
        detectFrame();
    }
};

// 執行順序
startCamera().then(initAI);
