import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19';

// --- 資工系 Debug 配置：強制手動指定模型下載來源 ---
env.allowLocalModels = false;
// 有些手機網路會擋 huggingface.co，我們強制它嘗試從 cdn 下載
env.remoteHost = 'https://huggingface.co/'; 
env.remotePathTemplate = '{model}/resolve/{revision}/';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const imageUpload = document.getElementById('image-upload');
const imageProxy = document.getElementById('image-proxy');

let detector;
let isLive = true;

// 啟動相機
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
        video.style.display = 'block';
        return true;
    } catch (e) {
        status.innerText = "❌ 相機權限遭拒: " + e.message;
        return false;
    }
}

// 初始化 AI
async function initAI() {
    try {
        status.innerText = "⏳ 正在從雲端抓取 AI 權重...";
        
        // 使用更小、更穩定的 yolov8n 進行環境測試
        detector = await pipeline('object-detection', 'onnx-community/yolov8n', {
            device: 'wasm', 
        });

        status.innerText = "✅ 系統就緒";
        detectFrame();
    } catch (e) {
        console.error(e);
        // 如果失敗，嘗試輸出更詳細的錯誤到螢幕上
        status.style.color = "red";
        status.innerText = "❌ 載入失敗原因: " + e.message.slice(0, 30);
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
    results.forEach(p => {
        const { xmin, ymin, xmax, ymax } = p.box;
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 3;
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
        ctx.fillStyle = '#00FF00';
        ctx.fillText(`${p.label}`, xmin, ymin > 10 ? ymin - 5 : 10);
    });
}

// 監聽上傳
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

startCamera().then(initAI);
