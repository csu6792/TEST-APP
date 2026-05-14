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

  // 獲取原始視訊尺寸
  const vW = video.videoWidth;
  const vH = video.videoHeight;

  // 計算 crop 尺寸 (以短邊為主，模擬 cover 效果)
  const size = Math.min(vW, vH);
  const sx = (vW - size) / 2;
  const sy = (vH - size) / 2;

  // 將視訊中間的正方形區域繪製到 640x640 的暫存畫布上
  tempCtx.drawImage(video, sx, sy, size, size, 0, 0, MODEL_SIZE, MODEL_SIZE);

  const imageData = tempCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const input = preprocess(imageData.data);
  const tensor = new ort.Tensor("float32", input, [1, 3, MODEL_SIZE, MODEL_SIZE]);

  const outputs = await session.run({ images: tensor });
  const outputData = outputs[Object.keys(outputs)[0]].cpuData;

  // 傳送 sx, sy, size 給繪圖函數進行反向計算
  drawBoxes(outputData, sx, sy, size);

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
  // --- 關鍵修正：確保畫布解析度與顯示尺寸一致 ---
  if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const videoW = video.videoWidth;
  const videoH = video.videoHeight;
  const canvasW = canvas.width;
  const canvasH = canvas.height;

  // 1. 計算 object-fit: cover 的縮放比例與位移
  // 這段邏輯必須與 CSS 的 object-fit 完全對應
  const scale = Math.max(canvasW / videoW, canvasH / videoH);
  const xOffset = (canvasW - videoW * scale) / 2;
  const yOffset = (canvasH - videoH * scale) / 2;

  // YOLO 模型通常是將影像 Resize 到 640x640（不維持比例的拉伸）
  const modelScaleX = videoW / 640;
  const modelScaleY = videoH / 640;

  for (let i = 0; i < data.length; i += 6) {
    const score = data[i + 4];
    if (score < 0.45) continue;

    // 2. 座標轉換：從模型 640x640 -> 視訊原尺寸 -> 螢幕顯示尺寸
    const x1 = data[i] * modelScaleX * scale + xOffset;
    const y1 = data[i + 1] * modelScaleY * scale + yOffset;
    const x2 = data[i + 2] * modelScaleX * scale + xOffset;
    const y2 = data[i + 3] * modelScaleY * scale + yOffset;

    const w = x2 - x1;
    const h = y2 - y1;

    const classId = Math.round(data[i + 5]);
    const label = classNames[classId] || "Object";

    // --- VisionOS 視覺設計 ---
    
    // 繪製外框陰影
    ctx.save(); // 保存狀態以避免陰影污染其他繪圖
    ctx.shadowBlur = 20;
    ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
    
    // 主圓角框 (白色半透明)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 2;
    drawRoundedRect(ctx, x1, y1, w, h, 12);
    ctx.stroke();
    ctx.restore();

    // 懸浮標籤 (Glassmorphism)
    const font = "600 13px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.font = font;
    const txt = `${label.toUpperCase()} ${Math.round(score * 100)}%`;
    const txtWidth = ctx.measureText(txt).width;
    
    const labelW = txtWidth + 20;
    const labelH = 26;
    const labelX = x1;
    const labelY = y1 - labelH - 10; // 懸浮高度

    // 標籤背景
    ctx.fillStyle = "rgba(20, 20, 20, 0.6)";
    ctx.backdropFilter = "blur(10px)"; // 注意：目前僅部分瀏覽器支援 canvas backdropFilter
    drawRoundedRect(ctx, labelX, labelY, labelW, labelH, 13);
    ctx.fill();
    
    // 標籤邊框
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    ctx.lineWidth = 1;
    ctx.stroke();

    // 標籤文字
    ctx.fillStyle = "#ffffff";
    ctx.textBaseline = "middle";
    ctx.fillText(txt, labelX + 10, labelY + labelH / 2 + 1);

    // 四角裝飾 (這讓它看起來更像 AR 偵測)
    drawVisionCorners(ctx, x1, y1, w, h, 15);
  }
}

// 輔助函式：VisionOS 風格轉角
function drawVisionCorners(ctx, x, y, w, h, len) {
  ctx.strokeStyle = "#00ff88"; // 使用你的主題綠色作為焦點
  ctx.lineWidth = 3;
  ctx.lineCap = "round";

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
