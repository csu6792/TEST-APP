const video =
  document.getElementById("video");

const canvas =
  document.getElementById("canvas");

const ctx =
  canvas.getContext("2d");

const startBtn =
  document.getElementById("startBtn");

const fpsText =
  document.getElementById("fps");

const MODEL_SIZE = 640;

let session;

let lastTime = performance.now();

const classNames = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck"
];

async function loadModel() {

  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

  session =
    await ort.InferenceSession.create(
      "./model/yolo26n.onnx",
      {
        executionProviders: ["wasm"]
      }
    );

  console.log("model loaded");
}

loadModel();

startBtn.onclick = async () => {

  const stream =
    await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment"
      },
      audio: false
    });

  video.srcObject = stream;

  video.onloadedmetadata = () => {

    canvas.width =
      video.videoWidth;

    canvas.height =
      video.videoHeight;

    detectFrame();

  };

};

async function detectFrame() {

  const start =
    performance.now();

  const tempCanvas =
    document.createElement("canvas");

  tempCanvas.width =
    MODEL_SIZE;

  tempCanvas.height =
    MODEL_SIZE;

  const tempCtx =
    tempCanvas.getContext("2d");

  tempCtx.drawImage(
    video,
    0,
    0,
    MODEL_SIZE,
    MODEL_SIZE
  );

  const imageData =
    tempCtx.getImageData(
      0,
      0,
      MODEL_SIZE,
      MODEL_SIZE
    );

  const input =
    preprocess(imageData.data);

  const tensor =
    new ort.Tensor(
      "float32",
      input,
      [1, 3, MODEL_SIZE, MODEL_SIZE]
    );

  const outputs =
    await session.run({
      images: tensor
    });

  drawBoxes(outputs.output0.cpuData);

  const end =
    performance.now();

  const fps =
    Math.round(1000 / (end - start));

  fpsText.innerText =
    `FPS: ${fps}`;

  requestAnimationFrame(detectFrame);
}

function preprocess(data) {

  const float32Data =
    new Float32Array(
      3 * MODEL_SIZE * MODEL_SIZE
    );

  for (
    let i = 0;
    i < MODEL_SIZE * MODEL_SIZE;
    i++
  ) {

    float32Data[i] =
      data[i * 4] / 255;

    float32Data[
      i + MODEL_SIZE * MODEL_SIZE
    ] =
      data[i * 4 + 1] / 255;

    float32Data[
      i + MODEL_SIZE * MODEL_SIZE * 2
    ] =
      data[i * 4 + 2] / 255;
  }

  return float32Data;
}

function drawBoxes(data) {

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const scaleX = canvas.width / MODEL_SIZE;
  const scaleY = canvas.height / MODEL_SIZE;

  for (let i = 0; i < data.length; i += 6) {

    const x1 = data[i];
    const y1 = data[i + 1];
    const x2 = data[i + 2];
    const y2 = data[i + 3];

    const score = data[i + 4];
    const classId = Math.round(data[i + 5]);

    if (score < 0.5) continue;

    const w = (x2 - x1) * scaleX;
    const h = (y2 - y1) * scaleY;

    const x = x1 * scaleX;
    const y = y1 * scaleY;

    const label = `${classNames[classId] || classId}`;
    const percent = (score * 100).toFixed(0);

    // 🧊 floating card size
    const boxW = 150;
    const boxH = 34;

    const cx = x + w / 2 - boxW / 2;
    const cy = y - 45;

    drawGlassCard(ctx, cx, cy, boxW, boxH, label, percent);

    // （可選）保留淡淡 bounding guide
    ctx.strokeStyle = "rgba(0,255,200,0.25)";
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);
  }
}

function drawGlassCard(ctx, x, y, w, h, label, percent) {

  const r = 12;

  ctx.save();

  // 🌟 glow
  ctx.shadowColor = "rgba(0,255,200,0.25)";
  ctx.shadowBlur = 20;

  // 🧊 glass gradient
  const g = ctx.createLinearGradient(x, y, x, y + h);
  g.addColorStop(0, "rgba(255,255,255,0.14)");
  g.addColorStop(1, "rgba(255,255,255,0.05)");

  ctx.fillStyle = g;

  roundRect(ctx, x, y, w, h, r);
  ctx.fill();

  // border
  ctx.strokeStyle = "rgba(0,255,200,0.35)";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.restore();

  // 🧠 text (Vision Pro style)
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.font = "13px -apple-system, BlinkMacSystemFont";

  ctx.fillText(
    `${label} ${percent}%`,
    x + 10,
    y + 22
  );
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

if ("serviceWorker" in navigator) {

  window.addEventListener("load", () => {

    navigator.serviceWorker.register(
      "./sw.js"
    );

  });

}
