import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19';

// 基礎環境強制設定
env.allowLocalModels = false;
env.localModelPath = './';
// 強制更新 WASM 運算核心，確保支援 YOLOv10 的新架構
env.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/';

const status = document.getElementById('status');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let detector;

async function init() {
    try {
        status.innerText = "⏳ 模型載入中...";
        
        // 讀取根目錄檔案：model.onnx, config.json, preprocessor_config.json
        detector = await pipeline('object-detection', './', {
            device: 'wasm',
            model: 'model.onnx',
            config: 'config.json',
            processor: 'preprocessor_config.json'
        });

        status.innerText = "✅ 載入成功";
        start();
    } catch (err) {
        console.error(err);
        status.style.color = "red";
        // 顯示具體報錯，方便知道是檔案壞了還是路徑錯了
        status.innerText = "❌ 錯誤: " + err.message;
    }
}

async function start() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: 640 },
            audio: false
        });
        video.srcObject = stream;
        await video.play();
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        render();
    } catch (err) {
        status.innerText = "❌ 相機失敗";
    }
}

async function render() {
    if (detector && video.readyState >= 2) {
        const results = await detector(video);
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        results.forEach(obj => {
            const { xmin, ymin, xmax, ymax } = obj.box;
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 3;
            ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
            
            ctx.fillStyle = '#00FF00';
            ctx.font = '16px sans-serif';
            ctx.fillText(`${obj.label}`, xmin, ymin > 15 ? ymin - 5 : 15);
        });
    }
    requestAnimationFrame(render);
}

init();
