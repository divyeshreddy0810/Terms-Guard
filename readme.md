# Terms Guard AI 🛡️

A browser extension and backend system that analyzes Terms & Conditions in real-time to identify unethical clauses using Retrieval-Augmented Generation (RAG) and Large Language Models.

## Team Members Details

1.  Divyesh Reddy Ellasiri (25134825)
2.  Sai Vivek Yerninti (25138286)
3.  Mukesh Saren Ramu (25121448)
4.  Alabi Taiwo Oluwatamilore (24335592)

## Features

-   **Real-time T&C Analysis**: Automatically analyzes terms and conditions on websites
-   **Ethical Assessment**: Identifies potentially unethical clauses using RAG and LLM
-   **Browser Extension**: Seamless integration with Chrome/Chromium browsers
-   **AI-Powered**: Uses Google's FLAN-T5 for intelligent summarization and analysis
-   **Semantic Search**: Leverages sentence transformers for accurate clause matching
-   **User-Friendly**: Simple popup interface for easy review of findings

## Technology Stack

-   **Backend**: Flask (Python)
-   **Frontend**: Chrome Extension (JavaScript, HTML, CSS)
-   **NLP Models**:
    -   `google/flan-t5-base` for text generation and summarization
    -   `sentence-transformers/all-MiniLM-L6-v2` for semantic embeddings
    -   `FAISS` for vector similarity search
-   **Libraries**: LangChain, Transformers, PyTorch, Flask-CORS

## Project Structure

```
├── backend/│   ├── app.py              # Flask backend API│   └── requirements.txt     # Python dependencies├── extension/│   ├── manifest.json       # Chrome extension manifest│   ├── popup.html          # Popup UI│   ├── popup.js            # Popup logic│   ├── content.js          # Content script for webpage interaction│   ├── window.html         # Results window│   └── icons/              # Extension icons├── .gitignore              # Git ignore rules└── readme.md               # This file
```

## Prerequisites

-   Python 3.8+
-   pip (Python package manager)
-   Chrome/Chromium browser
-   At least 4GB RAM (for model loading)
-   5GB disk space (for pre-trained models)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/divyeshreddy0810/Terms-Guard.gitcd Terms-Guard
```

### 2. Set Up Python Virtual Environment

```bash
# macOS/Linuxpython3 -m venv venvsource venv/bin/activate# Windowspython -m venv venvvenvScriptsactivate
```

### 3. Install Backend Dependencies

```bash
cd backendpip install -r requirements.txt
```

**Note**: First-time installation may take 5-10 minutes as it downloads pre-trained models (~2GB).

### 4. Install Browser Extension

1.  Open Chrome/Chromium and go to: `chrome://extensions/`
2.  Enable **"Developer mode"** (top right)
3.  Click **"Load unpacked"**
4.  Navigate to the `extension/` folder in this project
5.  The extension should appear in your browser toolbar

## How to Run

### Start the Backend Server

```bash
cd backendsource venv/bin/activate  # On Windows: venvScriptsactivatepython app.py
```

You should see:

```
LOADING MODELS (This takes 1-2 minutes on first run)...✅ Models loaded in XX.XX secondsRunning on http://127.0.0.1:5000
```

### Use the Browser Extension

1.  Keep the backend server running
2.  Open any website with Terms & Conditions
3.  Click the **Terms Guard AI** icon in your browser toolbar
4.  The extension will analyze the page and display results

## API Endpoints

The backend provides the following endpoints:

-   **POST** `/analyze` - Analyze terms and conditions text
    
    ```bash
    curl -X POST http://localhost:5000/analyze   -H "Content-Type: application/json"   d '{"text": "Your T&C text here"}'
    ```
    

## Development Notes

-   The project uses `FAISS` for efficient vector similarity search
-   Pre-computed ethical reference clauses are embedded for comparison
-   T5 model generates concise summaries of potentially problematic clauses
-   The extension communicates with the backend via HTTP requests

## Future Enhancements

-   Real-time highlighting of unethical clauses on webpages
-   Multi-language support
-   Migration to lighter models for better performance
-   Export analysis reports to PDF
-   User preferences and filtering options
-   Text-to-Speech dictation of the terms and conditions after classification

---