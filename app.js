import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.19';

// 取得 UI 元件
const status = document.getElementById('status');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let detector;

/**
 * 1. 初始化環境與模型
 */
async function init() {
    try {
        status.innerText = "⏳ 正在初始化環境...";

        // 設定模型搜尋路徑（指向根目錄）
        env.allowLocalModels = true;
        env.allowRemoteModels = false;
        env.localModelPath = './';

        // 修正：安全地設定 WASM 路徑，避免 undefined 錯誤
        if (env.onnx && env.onnx.wasm) {
            env.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/';
        } else if (env.backends && env.backends.onnx) {
            env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/';
        }

        status.innerText = "⏳ 載入 YOLOv10 模型 (9MB)...";

        // 載入模型：確保根目錄有 model.onnx, config.json, preprocessor_config.json
        detector = await pipeline('object-detection', './', {
            device: 'wasm',
            model: 'model.onnx',
            config: 'config.json',
            processor: 'preprocessor_config.json'
        });

        status.innerText = "✅ 模型就緒，啟動相機...";
        await startCamera();
    } catch (err) {
        console.error("初始化失敗:", err);
        status.style.color = "red";
        status.innerText = "❌ 失敗: " + err.message;
    }
}

/**
 * 2. 啟動相機
 */
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                facingMode: 'environment', // 優先使用後鏡頭
                width: { ideal: 640 },
                height: { ideal: 640 }
            },
            audio: false
        });
        
        video.srcObject = stream;
        await video.play();

        // 調整 Canvas 大小與影片一致
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        render();
    } catch (err) {
        status.innerText = "❌ 無法啟動相機: " + err.message;
    }
}

/**
 * 3. 偵測與繪圖迴圈
 */
async function render() {
    if (detector && video.readyState >= 2) {
        try {
            // 執行 YOLOv10 推論
            const results = await detector(video);

            // 清除畫布
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // 繪製結果
            results.forEach(obj => {
                const { xmin, ymin, xmax, ymax } = obj.box;
                
                // 畫框
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 3;
                ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

                // 畫標籤背景
                ctx.fillStyle = '#00FF00';
                const label = `${obj.label} ${Math.round(obj.score * 100)}%`;
                const textWidth = ctx.measureText(label).width;
                ctx.fillRect(xmin, ymin > 20 ? ymin - 20 : ymin, textWidth + 10, 20);

                // 寫文字
                ctx.fillStyle = '#000000';
                ctx.font = '14px sans-serif';
                ctx.fillText(label, xmin + 5, ymin > 20 ? ymin - 5 : ymin + 15);
            });
        } catch (inferenceErr) {
            console.error("推論出錯:", inferenceErr);
        }
    }
    // 持續執行
    requestAnimationFrame(render);
}

// 啟動程式
init();
