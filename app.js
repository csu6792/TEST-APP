const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const fpsText = document.getElementById("fps");

const MODEL_SIZE = 640;
let session;
let isDetecting = false;

// 優化：將臨時 Canvas 移到全域，避免重複創建導致記憶體洩漏
const tempCanvas = document.createElement("canvas");
tempCanvas.width = MODEL_SIZE;
tempCanvas.height = MODEL_SIZE;
const tempCtx = tempCanvas.getContext("2d");

const classNames = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"
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

  const scaleX = canvas.width / MODEL_SIZE;
  const scaleY = canvas.height / MODEL_SIZE;

  // 這裡維持你原有的解析邏輯
  // 但提醒：若偵測不到框，通常是因為輸出格式不是 [x1, y1, x2, y2, score, label]
  for (let i = 0; i < data.length; i += 6) {
    const score = data[i + 4];
    if (score < 0.5) continue;

    const x1 = data[i] * scaleX;
    const y1 = data[i + 1] * scaleY;
    const x2 = data[i + 2] * scaleX;
    const y2 = data[i + 3] * scaleY;
    const classId = Math.round(data[i + 5]);

    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.fillStyle = "#00ff88";
    ctx.font = "16px Arial";
    const label = `${classNames[classId] || 'Obj'} ${score.toFixed(2)}`;
    ctx.fillText(label, x1, y1 > 20 ? y1 - 5 : y1 + 20);
  }
}
