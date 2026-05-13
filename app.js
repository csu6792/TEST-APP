import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19';

// 設定本地路徑模式
env.allowRemoteModels = false;
env.localModelPath = './';

const status = document.getElementById('status');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let detector;

async function initAI() {
    try {
        // 使用 pipeline 的回呼函數來追蹤進度
        detector = await pipeline('object-detection', './', {
            device: 'wasm',
            // 檔名必須與 GitHub 上的完全一致
            config: 'config.json',
            model: 'model.onnx',
            processor: 'preprocessor_config.json',
            
            // 加入進度監測
            progress_callback: (data) => {
                if (data.status === 'progress') {
                    status.innerText = `⏳ 載入中: ${Math.round(data.progress)}%`;
                } else if (data.status === 'ready') {
                    status.innerText = "✨ 模型已就緒";
                }
            }
        });

        status.innerText = "✅ 辨識啟動中...";
        startCamera();
    } catch (e) {
        console.error(e);
        status.style.color = "red";
        status.innerText = "❌ 初始化失敗: " + e.message;
    }
}

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
        detectFrame();
    } catch (err) {
        status.innerText = "❌ 相機啟動失敗";
    }
}

async function detectFrame() {
    if (detector && video.readyState >= 2) {
        // 執行辨識
        const results = await detector(video);
        
        // 繪圖
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        results.forEach(p => {
            const { xmin, ymin, xmax, ymax } = p.box;
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 3;
            ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
            
            ctx.fillStyle = '#00FF00';
            ctx.font = '16px Arial';
            ctx.fillText(`${p.label} ${Math.round(p.score * 100)}%`, xmin, ymin > 10 ? ymin - 5 : 15);
        });
    }
    requestAnimationFrame(detectFrame);
}

// 執行
initAI();
