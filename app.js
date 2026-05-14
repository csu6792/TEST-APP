const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const fpsText = document.getElementById("fps");
const statusText = document.getElementById("statusText");

const MODEL_SIZE = 640;
let session;
let isDetecting = false;

// 優化：將臨時 Canvas 移到全域，避免重複創建導致記憶體洩漏
const tempCanvas = document.createElement("canvas");
tempCanvas.width = MODEL_SIZE;
tempCanvas.height = MODEL_SIZE;
const tempCtx = tempCanvas.getContext("2d");

const classNames = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'Kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
  'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
  'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

async function loadModel() {
  try {
    startBtn.disabled = true;
    startBtn.innerText = "模型載入中...";

    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

    // 載入模型
    session = await ort.InferenceSession.create(
      "./model/yolo26n.onnx", 
      { executionProviders: ["wasm"] }
    );

    console.log("Model loaded successfully");
    startBtn.disabled = false; // 載入成功後才啟用按鈕
    statusText.innerText = "AI 模型已就緒";
    startBtn.innerText = "開始偵測";
  } catch (e) {
    console.error("模型載入失敗:", e);
    startBtn.innerText = "載入失敗 (檢查路徑)";
  }
}

// 執行載入
loadModel();

startBtn.onclick = async () => {
  if (isDetecting) return;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false
    });

    video.srcObject = stream;

    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      isDetecting = true;
      detectFrame();
    };
  } catch (err) {
    console.error("無法開啟相機:", err);
    alert("請確保已授權相機權限");
  }
};

async function detectFrame() {
  if (!session || !isDetecting) return;

  const start = performance.now();

  // 1. 影像預處理：繪製到 640x640 的畫布
  tempCtx.drawImage(video, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const imageData = tempCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);

  // 2. 轉換為 Tensor
  const input = preprocess(imageData.data);
  const tensor = new ort.Tensor("float32", input, [1, 3, MODEL_SIZE, MODEL_SIZE]);

  try {
    // 3. 執行推論
    const outputs = await session.run({ images: tensor });
    
    // 注意：這裡假設你的模型輸出 key 為 'output0'
    // 如果報錯，請 console.log(outputs) 檢查正確的 key 名稱
    const outputData = outputs[Object.keys(outputs)[0]].cpuData;

    // 4. 繪圖
    drawBoxes(outputData);

  } catch (err) {
    console.error("推論過程出錯:", err);
  }

  const end = performance.now();
  const fps = Math.round(1000 / (end - start));
  fpsText.innerText = `FPS: ${fps}`;

  requestAnimationFrame(detectFrame);
}

function preprocess(data) {
  const float32Data = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
  // HWC 轉 CHW 格式
  for (let i = 0; i < MODEL_SIZE * MODEL_SIZE; i++) {
    float32Data[i] = data[i * 4] / 255; // R
    float32Data[i + MODEL_SIZE * MODEL_SIZE] = data[i * 4 + 1] / 255; // G
    float32Data[i + MODEL_SIZE * MODEL_SIZE * 2] = data[i * 4 + 2] / 255; // B
  }
  return float32Data;
}

function drawBoxes(data) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const videoW = video.videoWidth;
  const videoH = video.videoHeight;
  const canvasW = canvas.clientWidth;
  const canvasH = canvas.clientHeight;

  const scale = Math.max(canvasW / videoW, canvasH / videoH);
  const xOffset = (canvasW - videoW * scale) / 2;
  const yOffset = (canvasH - videoH * scale) / 2;

  const modelScaleX = videoW / 640;
  const modelScaleY = videoH / 640;

  for (let i = 0; i < data.length; i += 6) {
    const score = data[i + 4];
    if (score < 0.45) continue; // 稍微調低閾值讓反應更靈敏

    // 1. 座標轉換邏輯維持不變 (這是準確度的核心)
    const x = (data[i] * modelScaleX) * scale + xOffset;
    const y = (data[i + 1] * modelScaleY) * scale + yOffset;
    const w = (data[i + 2] - data[i]) * modelScaleX * scale;
    const h = (data[i + 3] - data[i + 1]) * modelScaleY * scale;

    const classId = Math.round(data[i + 5]);
    const label = classNames[classId] || "Object";
    
    // --- VisionOS 視覺設計開始 ---

    // 2. 設置發光與陰影
    ctx.shadowBlur = 15;
    ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
    
    // 3. 繪製主邊框 (使用白色或極淺綠)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";

    // 繪製圓角矩形框 (VisionOS 核心元素)
    drawRoundedRect(ctx, x, y, w, h, 12);
    ctx.stroke();

    // 4. 繪製「懸浮標籤」
    const font = "600 14px -apple-system, system-ui, sans-serif";
    ctx.font = font;
    const txt = `${label.toUpperCase()} ${Math.round(score * 100)}%`;
    const txtWidth = ctx.measureText(txt).width;
    const padding = 10;
    const rectW = txtWidth + padding * 2;
    const rectH = 28;
    const rectX = x;
    const rectY = y - rectH - 8; // 向上偏移，增加呼吸感

    // 標籤背景：深色毛玻璃質感
    ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
    ctx.shadowBlur = 10;
    drawRoundedRect(ctx, rectX, rectY, rectW, rectH, 14);
    ctx.fill();

    // 標籤亮邊 (微光效果)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
    ctx.lineWidth = 1;
    ctx.stroke();

    // 5. 繪製文字
    ctx.shadowBlur = 0; // 文字不需要陰影
    ctx.fillStyle = "white";
    ctx.textBaseline = "middle";
    ctx.fillText(txt, rectX + padding, rectY + rectH / 2);

    // 6. 加分項：四角強化 (Corners)
    drawCorners(ctx, x, y, w, h, 20);
  }
}

/**
 * 輔助函數：畫圓角矩形
 */
function drawRoundedRect(ctx, x, y, w, h, r) {
  if (w < 2 * r) r = w / 2;
  if (h < 2 * r) r = h / 2;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/**
 * 輔助函數：繪製 VisionOS 感的四角強化線段
 */
function drawCorners(ctx, x, y, w, h, len) {
  ctx.strokeStyle = "rgba(0, 255, 136, 1)"; // 轉角使用主題綠色
  ctx.lineWidth = 4;
  
  // 左上
  ctx.beginPath();
  ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y);
  ctx.stroke();
  
  // 右上
  ctx.beginPath();
  ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
  ctx.stroke();
  
  // 左下
  ctx.beginPath();
  ctx.moveTo(x, y + h - len); ctx.lineTo(x, y + h); ctx.lineTo(x + len, y + h);
  ctx.stroke();
  
  // 右下
  ctx.beginPath();
  ctx.moveTo(x + w - len, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - len);
  ctx.stroke();
}
