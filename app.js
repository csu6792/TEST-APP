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

  ctx.clearRect(
    0,
    0,
    canvas.width,
    canvas.height
  );

  const scaleX =
    canvas.width / MODEL_SIZE;

  const scaleY =
    canvas.height / MODEL_SIZE;

  for (
    let i = 0;
    i < data.length;
    i += 6
  ) {

    const x1 = data[i];
    const y1 = data[i + 1];
    const x2 = data[i + 2];
    const y2 = data[i + 3];

    const score =
      data[i + 4];

    const classId =
      Math.round(data[i + 5]);

    if (score < 0.5)
      continue;

    const w = x2 - x1;
    const h = y2 - y1;

    ctx.strokeStyle =
      "#00ff88";

    ctx.lineWidth = 3;

    ctx.strokeRect(
      x1 * scaleX,
      y1 * scaleY,
      w * scaleX,
      h * scaleY
    );

    ctx.fillStyle =
      "#00ff88";

    ctx.font =
      "18px Arial";

    const label =
      `${classNames[classId] || classId}
      ${score.toFixed(2)}`;

    ctx.fillRect(
      x1 * scaleX,
      y1 * scaleY - 30,
      140,
      30
    );

    ctx.fillStyle =
      "black";

    ctx.fillText(
      label,
      x1 * scaleX + 5,
      y1 * scaleY - 8
    );
  }
}

if ("serviceWorker" in navigator) {

  window.addEventListener("load", () => {

    navigator.serviceWorker.register(
      "./sw.js"
    );

  });

}
