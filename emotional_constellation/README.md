# 情绪星空 (Emotional Constellation)

一个融合 **数据驱动 + 生成艺术 + 网页交互** 的期末项目。  
项目在 **Streamlit Cloud** 上部署，可将文本情绪映射为“星空”可视化。

## 在线演示（待部署后替换链接）
- Streamlit: `https://<your-name>-emotional-constellation.streamlit.app`
- GitHub: `https://github.com/<your-name>/emotional_constellation`

## 功能
- 读取 CSV 或粘贴文本，使用 **NLTK VADER** 进行情感分析（多语言文本在英文词典下以关键词为主）。  
- 将情绪极性与强度映射为星星的颜色、亮度与大小，生成一张“情绪星空”。  
- 可按**时间范围**与**情绪类型**进行过滤，动态刷新画布。  
- 可导出当前星空为 PNG。

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 数据格式
- CSV 至少包含一列 `text`（文本内容），可选列 `timestamp`（YYYY-MM-DD 或可解析日期）、`source`（来源标签）。
- 示例：`sample_data.csv`。

## 说明
- 由于 VADER 词典以英文为主，对中文情绪的识别较依赖英文或常见情绪词。可扩展为繁体/简体中文词典或情绪分类器。
- 在 Streamlit Cloud 首次运行时会自动下载 `vader_lexicon`。

---

## English

A generative, data-driven, interactive Streamlit app that turns text sentiments into a glowing **emotional night sky**.

### Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Data
- CSV with `text` column, optional `timestamp`, `source`.
- See `sample_data.csv`.

### Notes
- Uses **NLTK VADER** sentiment analysis. Works best for English; multilingual support can be improved by adding custom lexicons or models.
