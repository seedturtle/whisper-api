# Faster Whisper Speech-to-Text API
# 免 API Key 的語音轉文字服務

## 部署說明

### 1. 安裝依賴
```bash
pip install faster-whisper flask flask-cors
```

### 2. 運行服務
```bash
python app.py
```

服務會在 `http://0.0.0.0:8080` 啟動

### 3. API 使用方式

**轉換音檔：**
```bash
curl -X POST -F "file=@audio.mp3" http://localhost:8080/transcribe
```

**回應格式：**
```json
{
  "text": "轉換後的文字內容",
  "language": "zh",
  "segments": [...]
}
```

### 4. Zeabur 部署
- 選擇 Python 模板
- 設定 PORT = 8080
- Build Command: `pip install faster-whisper flask flask-cors`
- Start Command: `python app.py`

## 模型大小選擇
- `tiny` - 最快，準確度較低
- `base` - 平衡選擇
- `small` - 較慢，準確度較高
- `medium` - 更慢，更準確
- `large-v3` - 最準確，需要更多資源

預設使用 `base` 模型。
