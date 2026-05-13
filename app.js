import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19';

// --- 強制根目錄讀取配置 ---
env.allowRemoteModels = false;    // 絕對不連外網
env.localModelPath = './';        // 設定模型路徑為「當前目錄」

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const imageUpload = document.getElementById('image-upload');
const imageProxy = document.getElementById('image-proxy');

let detector;
let isLive = true;

// 1. 啟動相機
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: 640 },
            audio: false
        });
        video.srcObject = stream;
        await video.play();
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        return true;
    } catch (e) {
        status.innerText = "❌ 相機權限失敗";
        return false;
    }
}

// 2. 初始化 AI
async function initAI() {
    try {
        status.innerText = "⏳ 載入根目錄模型檔案...";
        
        // 注意：因為檔案就在根目錄，這裡直接傳入一個代表「目前資料夾」的識別碼
        // Transformers.js 會自動去抓 ./config.json, ./model.onnx 等
        detector = await pipeline('object-detection', './', {
            device: 'wasm', 
        });

        status.innerText = "✅ 本地模型載入成功";
        detectFrame();
    } catch (e) {
        console.error("載入失敗詳情:", e);
        status.style.color = "red";
        status.innerText = "❌ 找不到 model.onnx 或 config.json";
    }
}

// 3. 偵測與繪圖
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
        ctx.lineWidth = 3;
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
        ctx.fillStyle = '#00FF00';
        ctx.fillText(p.label, xmin, ymin > 10 ? ymin - 5 : 10);
    });
}

// 4. 上傳與點擊恢復
imageUpload.addEventListener('change', async (e) => {
    isLive = false;
    const url = URL.createObjectURL(e.target.files[0]);
    imageProxy.src = url;
    imageProxy.style.display = 'block';
    video.style.display = 'none';
    imageProxy.onload = async () => {
        canvas.width = imageProxy.naturalWidth;
        canvas.height = imageProxy.naturalHeight;
        const results = await detector(imageProxy.src);
        renderDetections(results);
    };
});

canvas.onclick = async () => {
    if (!isLive) {
        await startCamera();
        isLive = true;
        detectFrame();
    }
};

startCamera().then(initAI);
