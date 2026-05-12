import { pipeline, cos_sim } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

let embedder;
let pdfChunks = [];

const status = document.getElementById('status');
const searchArea = document.getElementById('search-area');
const resultsContainer = document.getElementById('results');

// 初始化模型
async function init() {
    try {
        embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        status.innerText = "✅ 準備就緒，請上傳 PDF";
        status.style.background = "#d4edda";
    } catch (e) {
        status.innerText = "❌ 模型載入失敗";
    }
}

// 提取 PDF 文字
async function extractText(file) {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    let text = "";
    for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        text += content.items.map(item => item.str).join(" ") + "\n";
    }
    return text;
}

// 文字切片
function chunkText(text, size = 400) {
    const chunks = [];
    for (let i = 0; i < text.length; i += size) {
        chunks.push(text.substring(i, i + size));
    }
    return chunks;
}

// 處理上傳
document.getElementById('pdf-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    status.innerText = "⏳ 正在讀取 PDF...";
    const fullText = await extractText(file);
    const chunks = chunkText(fullText);

    status.innerText = `⏳ 正在分析向量 (0/${chunks.length})...`;
    pdfChunks = [];
    
    for (let i = 0; i < chunks.length; i++) {
        const output = await embedder(chunks[i], { pooling: 'mean', normalize: true });
        pdfChunks.push({ text: chunks[i], embedding: output.data });
        status.innerText = `⏳ 正在分析向量 (${i + 1}/${chunks.length})...`;
    }

    status.innerText = "✅ 索引完成，可以開始搜尋";
    searchArea.style.display = 'block';
});

// 處理搜尋
document.getElementById('search-btn').onclick = async () => {
    const query = document.getElementById('query').value;
    if (!query) return;

    status.innerText = "🔍 搜尋中...";
    const queryOutput = await embedder(query, { pooling: 'mean', normalize: true });
    
    const scored = pdfChunks.map(chunk => ({
        text: chunk.text,
        score: cos_sim(queryOutput.data, chunk.embedding)
    })).sort((a, b) => b.score - a.score);

    displayResults(scored.slice(0, 3));
    status.innerText = "✅ 搜尋完畢";
};

function displayResults(items) {
    resultsContainer.innerHTML = items.map(item => `
        <div class="result-card">
            <p>${item.text}</p>
            <small>相關係數：${(item.score * 100).toFixed(1)}%</small>
        </div>
    `).join('');
}

init();
