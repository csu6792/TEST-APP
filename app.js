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

  // 計算 object-fit: cover 的縮放比例
  const scale = Math.max(canvasW / videoW, canvasH / videoH);
  const xOffset = (canvasW - videoW * scale) / 2;
  const yOffset = (canvasH - videoH * scale) / 2;

  // YOLO 模型輸入為 640x640
  const modelScaleX = videoW / 640;
  const modelScaleY = videoH / 640;

  for (let i = 0; i < data.length; i += 6) {
    const score = data[i + 4];
    if (score < 0.5) continue;

    // 1. 還原回視訊原始像素座標
    const origX1 = data[i] * modelScaleX;
    const origY1 = data[i + 1] * modelScaleY;
    const origX2 = data[i + 2] * modelScaleX;
    const origY2 = data[i + 3] * modelScaleY;

    // 2. 轉換為螢幕顯示座標（加上 Offset 與 Scale）
    const x = origX1 * scale + xOffset;
    const y = origY1 * scale + yOffset;
    const w = (origX2 - origX1) * scale;
    const h = (origY2 - origY1) * scale;

    // 3. 繪製科技感偵測框
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    // 繪製標籤背景
    ctx.fillStyle = "#00ff88";
    const label = classNames[Math.round(data[i+5])] || "OBJ";
    const txt = `${label} ${Math.round(score * 100)}%`;
    ctx.font = "bold 14px Arial";
    const txtWidth = ctx.measureText(txt).width;
    
    ctx.fillRect(x, y - 25, txtWidth + 10, 25);
    ctx.fillStyle = "black";
    ctx.fillText(txt, x + 5, y - 7);
  }
}
