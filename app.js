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

// 初始化音訊上下文
const AudioContext = window.AudioContext || window.webkitAudioContext;
const audioCtx = new AudioContext();
const panner = audioCtx.createStereoPanner();
panner.connect(audioCtx.destination);

// 限制語音播報頻率的變數
let lastSpokenLabel = "";
let lastSpokenTime = 0;

function speakObject(label) {
  const now = Date.now();
  // 避免短時間內重複播放 (冷卻時間 3 秒)
  if (label === lastSpokenLabel && now - lastSpokenTime < 3000) return;

  const msg = new SpeechSynthesisUtterance(label);
  msg.lang = "en-US"; // 或是 "zh-TW"
  msg.rate = 1.0;
  msg.pitch = 1.2; // 稍微高一點，聽起來更具未來感
  window.speechSynthesis.speak(msg);

  lastSpokenLabel = label;
  lastSpokenTime = now;
}

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

  // --- 關鍵：解鎖音訊 ---
    if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
    }

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

  const vW = video.videoWidth;
  const vH = video.videoHeight;

  // 計算中間正方形裁切資訊
  const size = Math.min(vW, vH);
  const sx = (vW - size) / 2;
  const sy = (vH - size) / 2;

  tempCtx.drawImage(video, sx, sy, size, size, 0, 0, MODEL_SIZE, MODEL_SIZE);

  const imageData = tempCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const input = preprocess(imageData.data);
  const tensor = new ort.Tensor("float32", input, [1, 3, MODEL_SIZE, MODEL_SIZE]);

  const outputs = await session.run({ images: tensor });
  const outputData = outputs[Object.keys(outputs)[0]].cpuData;

  // --- 關鍵：傳入 sx, sy, size ---
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

function drawBoxes(data, sx, sy, size) {
  // 1. 自動同步 Canvas 畫布解析度與 CSS 顯示大小 (解決模糊與偏移)
  if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const videoW = video.videoWidth;
  const videoH = video.videoHeight;
  const canvasW = canvas.width;
  const canvasH = canvas.height;

  // 2. 計算 object-fit: cover 的顯示縮放與位移
  const displayScale = Math.max(canvasW / videoW, canvasH / videoH);
  const xOffset = (canvasW - videoW * displayScale) / 2;
  const yOffset = (canvasH - videoH * displayScale) / 2;

  // 3. 模型座標還原比例 (因為當初 detectFrame 切的是正方形 size)
  const modelToVideoScale = size / 640;

  for (let i = 0; i < data.length; i += 6) {
    const score = data[i + 4];
    if (score < 0.45) continue; // 信心門檻

     // 在 drawBoxes 迴圈內
     if (score > 0.85) {
      // 觸發語音播報
      speakObject(label);
    
      // 觸發空間音效 (取框框中心點 x)
      //const centerX = x + w / 2;
      //const distanceScale = h / canvasH; // 框越高代表物體越近
      //playSpatialPing(centerX, canvasW, distanceScale);
     }

    // 4. 座標轉換鏈條：模型 -> 視訊裁切區 -> 原始視訊 -> 螢幕 Canvas
    const x = ((data[i] * modelToVideoScale) + sx) * displayScale + xOffset;
    const y = ((data[i + 1] * modelToVideoScale) + sy) * displayScale + yOffset;
    const w = (data[i + 2] - data[i]) * modelToVideoScale * displayScale;
    const h = (data[i + 3] - data[i + 1]) * modelToVideoScale * displayScale;

    const classId = Math.round(data[i + 5]);
    const label = classNames[classId] || "Object";

    // --- 開始繪製 VisionOS 風格介面 ---

    // A. 主體圓角框 (白色半透明線條)
    ctx.save();
    ctx.shadowBlur = 15;
    ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
    ctx.lineWidth = 2;
    drawRoundedRect(ctx, x, y, w, h, 12);
    ctx.stroke();
    ctx.restore();

    // B. 懸浮資訊標籤 (玻璃質感面板)
    const font = "600 13px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    ctx.font = font;
    const txt = `${label.toUpperCase()} ${Math.round(score * 100)}%`;
    const txtWidth = ctx.measureText(txt).width;
    
    const labelW = txtWidth + 20;
    const labelH = 28;
    const labelX = x;
    const labelY = y - labelH - 10; // 向上偏移，增加懸浮感

    // 標籤背景 (深色毛玻璃)
    ctx.fillStyle = "rgba(10, 10, 10, 0.55)";
    drawRoundedRect(ctx, labelX, labelY, labelW, labelH, 14);
    ctx.fill();
    
    // 標籤細邊框
    ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
    ctx.lineWidth = 1;
    ctx.stroke();

    // 標籤文字 (純白)
    ctx.fillStyle = "#ffffff";
    ctx.textBaseline = "middle";
    ctx.fillText(txt, labelX + 10, labelY + labelH / 2 + 1);

    // C. 繪製強化四角 (調用下方的 drawVisionCorners)
    drawVisionCorners(ctx, x, y, w, h, 18);
  }
}

// 輔助函式：VisionOS 風格轉角
/**
 * 繪製 VisionOS 感的四角強化線段
 * @param {CanvasRenderingContext2D} ctx - Canvas 上下文
 * @param {number} x - 左上角 X
 * @param {number} y - 左上角 Y
 * @param {number} w - 寬度
 * @param {number} h - 高度
 * @param {number} len - 線段長度
 */
function drawVisionCorners(ctx, x, y, w, h, len) {
  ctx.save();
  ctx.strokeStyle = "#00ff88"; // Vision 主題綠色
  ctx.lineWidth = 3.5;
  ctx.lineCap = "round"; // 讓線段末端圓潤
  ctx.shadowBlur = 8;
  ctx.shadowColor = "rgba(0, 255, 136, 0.5)";

  // 左上角
  ctx.beginPath();
  ctx.moveTo(x, y + len);
  ctx.lineTo(x, y);
  ctx.lineTo(x + len, y);
  ctx.stroke();

  // 右上角
  ctx.beginPath();
  ctx.moveTo(x + w - len, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + len);
  ctx.stroke();

  // 左下角
  ctx.beginPath();
  ctx.moveTo(x, y + h - len);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x + len, y + h);
  ctx.stroke();

  // 右下角
  ctx.beginPath();
  ctx.moveTo(x + w - len, y + h);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + w, y + h - len);
  ctx.stroke();

  ctx.restore();
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
