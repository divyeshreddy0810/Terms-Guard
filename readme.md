# Terms Guard AI 🛡️

A browser extension and backend system that analyzes Terms & Conditions in real-time to identify unethical clauses using Retrieval-Augmented Generation (RAG) and Large Language Models.

## Features

- **Real-time T&C Analysis**: Automatically analyzes terms and conditions on websites
- **Ethical Assessment**: Identifies potentially unethical clauses using RAG and LLM
- **Browser Extension**: Seamless integration with Chrome/Chromium browsers
- **AI-Powered**: Uses Google's FLAN-T5 for intelligent summarization and analysis
- **Semantic Search**: Leverages sentence transformers for accurate clause matching
- **User-Friendly**: Simple popup interface for easy review of findings

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: Chrome Extension (JavaScript, HTML, CSS)
- **NLP Models**:
  - `google/flan-t5-base` for text generation and summarization
  - `sentence-transformers/all-MiniLM-L6-v2` for semantic embeddings
  - `FAISS` for vector similarity search
- **Libraries**: LangChain, Transformers, PyTorch, Flask-CORS

## Project Structure

```
├── backend/
│   ├── app.py              # Flask backend API
│   └── requirements.txt     # Python dependencies
├── extension/
│   ├── manifest.json       # Chrome extension manifest
│   ├── popup.html          # Popup UI
│   ├── popup.js            # Popup logic
│   ├── content.js          # Content script for webpage interaction
│   ├── window.html         # Results window
│   └── icons/              # Extension icons
├── .gitignore              # Git ignore rules
└── readme.md               # This file
```

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Chrome/Chromium browser
- At least 4GB RAM (for model loading)
- 5GB disk space (for pre-trained models)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/divyeshreddy0810/Terms-Guard.git
cd Terms-Guard
```

### 2. Set Up Python Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note**: First-time installation may take 5-10 minutes as it downloads pre-trained models (~2GB).

### 4. Install Browser Extension

1. Open Chrome/Chromium and go to: `chrome://extensions/`
2. Enable **"Developer mode"** (top right)
3. Click **"Load unpacked"**
4. Navigate to the `extension/` folder in this project
5. The extension should appear in your browser toolbar

## How to Run

### Start the Backend Server

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python app.py
```

You should see:
```
LOADING MODELS (This takes 1-2 minutes on first run)...
✅ Models loaded in XX.XX seconds
Running on http://127.0.0.1:5000
```

### Use the Browser Extension

1. Keep the backend server running
2. Open any website with Terms & Conditions
3. Click the **Terms Guard AI** icon in your browser toolbar
4. The extension will analyze the page and display results

## API Endpoints

The backend provides the following endpoints:

- **POST** `/analyze` - Analyze terms and conditions text
  ```bash
  curl -X POST http://localhost:5000/analyze \
    -H "Content-Type: application/json" \
    -d '{"text": "Your T&C text here"}'
  ```

## Troubleshooting

### Models Not Loading
- **Issue**: "CUDA out of memory" or "Model not found"
- **Solution**: Ensure you have at least 4GB free RAM. The first run will download ~2GB of models.

### Extension Not Connecting to Backend
- **Issue**: Extension shows "Connection refused"
- **Solution**: Make sure the Flask server is running on `http://127.0.0.1:5000`

### Slow Analysis
- **Issue**: Analysis takes longer than expected
- **Solution**: This is normal on first run (models are loading). Subsequent runs are faster.

### macOS Specific Issues
- The backend includes fixes for macOS multiprocessing. If you encounter fork safety warnings, they're expected and can be safely ignored.

## Development Notes

- The project uses `FAISS` for efficient vector similarity search
- Pre-computed ethical reference clauses are embedded for comparison
- T5 model generates concise summaries of potentially problematic clauses
- The extension communicates with the backend via HTTP requests

## Future Enhancements

- Real-time highlighting of unethical clauses on webpages
- Multi-language support
- Migration to lighter models for better performance
- Export analysis reports to PDF
- User preferences and filtering options

## License

This project is part of the Foundations of AI Lab coursework at NCI.

## Support

For issues or questions, please open an issue on GitHub or contact the project maintainers.

---

**Happy analyzing! 🔍 Help users understand what they're agreeing to.** ✨
