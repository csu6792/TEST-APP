const upload = document.getElementById("upload");
const image = document.getElementById("image");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let session;

async function loadModel() {

  session = await ort.InferenceSession.create(
    "./model/yolo.onnx"
  );

  console.log("model loaded");
}

loadModel();

upload.addEventListener("change", async (e) => {

  const file = e.target.files[0];

  image.src = URL.createObjectURL(file);

  image.onload = async () => {

    detect(image);

  };

});

async function detect(img) {

  const size = 640;

  canvas.width = img.width;
  canvas.height = img.height;

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = size;
  tempCanvas.height = size;

  const tempCtx = tempCanvas.getContext("2d");

  tempCtx.drawImage(img, 0, 0, size, size);

  const imageData = tempCtx.getImageData(0, 0, size, size);

  const input = preprocessing(imageData.data, size);

  const tensor = new ort.Tensor(
    "float32",
    input,
    [1, 3, size, size]
  );

  const outputs = await session.run({
    images: tensor
  });

  console.log(outputs);

  drawDemoBox();

}

function preprocessing(data, size) {

  const float32Data = new Float32Array(
    3 * size * size
  );

  for (let i = 0; i < size * size; i++) {

    float32Data[i] =
      data[i * 4] / 255;

    float32Data[i + size * size] =
      data[i * 4 + 1] / 255;

    float32Data[i + size * size * 2] =
      data[i * 4 + 2] / 255;
  }

  return float32Data;
}

function drawDemoBox() {

  ctx.strokeStyle = "red";
  ctx.lineWidth = 3;

  ctx.strokeRect(
    100,
    100,
    200,
    200
  );

}
