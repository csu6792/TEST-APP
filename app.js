import { pipeline, cos_sim } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

// 全局變數
let embedder;
let pdfChunks = [];

// DOM 元素
const status = document.getElementById('status');
const searchArea = document.getElementById('search-area');
const resultsContainer = document.getElementById('results');
const pdfInput = document.getElementById('pdf-upload');
const searchBtn = document.getElementById('search-btn');
const queryInput = document.getElementById('query');

/**
 * 1. 初始化 AI 模型 (支援多國語言/中文)
 */
async function init() {
    try {
        status.innerText = "⏳ 正在載入中文語意模型 (約 80MB)...";
        status.style.background = "#fff3cd";
        
        // 使用多國語言模型，對中文理解力大幅提升
        embedder = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2');
        
        status.innerText = "✅ 中文模型準備就緒，請上傳 PDF";
        status.style.background = "#d4edda";
    } catch (e) {
        console.error(e);
        status.innerText = "❌ 模型載入失敗，請檢查網路連線";
        status.style.background = "#f8d7da";
    }
}

/**
 * 2. 提取 PDF 文字 (利用 pdf.js)
 */
async function extractText(file) {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    let text = "";
    for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        const pageText = content.items.map(item => item.str).join("");
        text += `第 ${i} 頁：${pageText}\n`;
    }
    return text;
}

/**
 * 3. 語意切片函式 (帶有重疊區塊 Overlap)
 */
function chunkTextWithOverlap(text, size = 300, overlap = 50) {
    const chunks = [];
    let i = 0;
    while (i < text.length) {
        // 取得片段
        let chunk = text.substring(i, i + size);
        chunks.push(chunk);
        // 移動索引：移動距離為 (區塊大小 - 重疊大小)
        i += (size - overlap);
        if (size <= overlap) break; // 防止無限循環
    }
    return chunks;
}

/**
 * 4. 處理文件上傳與向量化
 */
pdfInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file || !embedder) return;

    status.innerText = "⏳ 正在讀取文件內容...";
    const fullText = await extractText(file);
    
    // 中文資訊密度高，300字一個 chunk 效果較好
    const chunks = chunkTextWithOverlap(fullText, 300, 50);
    pdfChunks = [];

    status.innerText = `⏳ 正在分析中文語意 (0/${chunks.length})...`;
    
    // 逐一向量化
    for (let i = 0; i < chunks.length; i++) {
        const output = await embedder(chunks[i], { pooling: 'mean', normalize: true });
        pdfChunks.push({
            text: chunks[i],
            embedding: output.data
        });
        
        // 更新進度條
        status.innerText = `⏳ 向量運算中：${Math.round(((i + 1) / chunks.length) * 100)}%`;
    }

    status.innerText = "✅ 索引建立完成，隨時可以提問！";
    searchArea.style.display = 'block';
});

/**
 * 5. 執行向量搜尋
 */
async function handleSearch() {
    const query = queryInput.value.trim();
    if (!query || !embedder || pdfChunks.length === 0) return;

    status.innerText = "🔍 AI 搜尋中...";
    
    // 將使用者的問題也轉為向量
    const queryOutput = await embedder(query, { pooling: 'mean', normalize: true });
    const queryVec = queryOutput.data;

    // 計算所有片段與問題的 Cosine Similarity (餘弦相似度)
    const scoredResults = pdfChunks.map(chunk => ({
        text: chunk.text,
        score: cos_sim(queryVec, chunk.embedding)
    }));

    // 排序：從高分到低分，並取前 5 名
    const topResults = scoredResults
        .sort((a, b) => b.score - a.score)
        .slice(0, 5);

    renderResults(topResults);
    status.innerText = "✅ 搜尋完畢";
}

/**
 * 6. 渲染結果到畫面
 */
function renderResults(items) {
    resultsContainer.innerHTML = items.map((item, index) => `
        <div class="result-card" style="animation-delay: ${index * 0.1}s">
            <div class="score-tag">相關度: ${(item.score * 100).toFixed(1)}%</div>
            <p>${highlightKeywords(item.text, queryInput.value)}</p>
        </div>
    `).join('');
}

/**
 * 輔助：關鍵字高亮 (選配)
 */
function highlightKeywords(text, query) {
    if (!query) return text;
    const words = query.split(/\s+/);
    let highlighted = text;
    words.forEach(word => {
        if (word.length > 1) {
            const reg = new RegExp(word, 'gi');
            highlighted = highlighted.replace(reg, match => `<mark>${match}</mark>`);
        }
    });
    return highlighted;
}

// 綁定按鈕事件
searchBtn.addEventListener('click', handleSearch);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSearch();
});

// 啟動初始化
init();
