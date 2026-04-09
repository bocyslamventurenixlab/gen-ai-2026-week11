# Agentic Read - Multi-Agent Document Intelligence System

A powerful FastAPI + React application that uses multi-agent orchestration to analyze PDF documents with semantic search, intelligent reasoning, and hallucination prevention.

## 🎯 Features

- **Multi-Agent Architecture**
  - 🔒 **SecurityAgent**: Input validation and jailbreak detection
  - 📚 **LibrarianAgent**: Semantic vector search with cosine similarity
  - 🧠 **AnalystAgent**: LLM-powered reasoning with tool calling
  - ✏️ **EditorAgent**: Response verification and hallucination prevention

- **Document Processing**
  - PDF upload with automatic text extraction
  - Vector embedding generation for semantic search
  - Support for multiple document uploads
  - Persistent storage with Supabase

- **User Interface**
  - Modern dark-themed React dashboard
  - Real-time document management
  - Agent execution trace visualization
  - Live upload status feedback

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn
- Git

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd gen-ai-2026-week11

# Create .env file with your credentials
cat > .env << EOF
SUPABASE_URL="your_supabase_url"
SUPABASE_KEY="your_supabase_key"
OPENROUTER_API_KEY="your_openrouter_api_key"
OPENROUTER_URL="https://openrouter.ai/api/v1"
EOF
```

### 2. Backend Setup

```bash
# Install Python dependencies
pip install fastapi uvicorn supabase python-multipart pypdf openai python-dotenv numpy

# Start the backend server
python main.py
```

The backend will run on `http://localhost:8000`

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will run on `http://localhost:5173/`

### 4. Access the Application

Open your browser and navigate to:
```
http://localhost:5173/
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_public_key

# OpenRouter API (LLM Provider)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
OPENROUTER_URL=https://openrouter.ai/api/v1
```

### Supabase Setup

1. Create a Supabase project at https://supabase.com
2. Create the following tables:

**documents table:**
```sql
CREATE TABLE documents (
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  user_id UUID,
  upload_date TIMESTAMP DEFAULT NOW(),
  created_at TIMESTAMP DEFAULT NOW()
);
```

**embeddings table:**
```sql
CREATE TABLE embeddings (
  id BIGSERIAL PRIMARY KEY,
  doc_id BIGINT REFERENCES documents(id),
  content TEXT NOT NULL,
  embedding VECTOR(1536),
  created_at TIMESTAMP DEFAULT NOW()
);
```

**security_logs table (optional):**
```sql
CREATE TABLE security_logs (
  id BIGSERIAL PRIMARY KEY,
  query TEXT,
  type TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 📁 Project Structure

```
gen-ai-2026-week11/
├── main.py                          # FastAPI backend
├── .env                            # Environment variables
├── README.md                       # This file
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # Main React component
│   │   ├── main.jsx               # Entry point
│   │   ├── index.css              # Tailwind styles
│   │   └── App.css                # Component styles
│   ├── public/                    # Static assets
│   ├── package.json               # Frontend dependencies
│   ├── vite.config.js             # Vite configuration
│   ├── tailwind.config.js         # Tailwind configuration
│   ├── postcss.config.js          # PostCSS configuration
│   └── eslint.config.js           # ESLint configuration
```

---

## 🔌 API Endpoints

### GET `/documents`
Retrieve all uploaded documents.

**Response:**
```json
[
  {
    "id": 1,
    "title": "Sarah-Marketing Executive-2025.pdf",
    "upload_date": "2026-02-02T10:35:29.340461+00:00"
  }
]
```

### POST `/upload`
Upload a PDF file and create embeddings.

**Request:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "message": "Upload & Embedding successful",
  "document_id": 1
}
```

### POST `/process`
Process a query against a document using multi-agent orchestration.

**Request:**
```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": 1,
    "query": "Who is Sarah?"
  }'
```

**Response:**
```json
{
  "summary": "Sarah Thompson is a Marketing Director...",
  "key_points": [
    "Sarah Thompson is a Strategic Marketing Executive",
    "Works at BrandSphere Global in New York",
    "Over 10 years of experience"
  ],
  "confidence_score": 0.95,
  "is_safe": true,
  "trace": [
    "Security Cleared",
    "Semantic Retrieval Complete",
    "Reasoning Verified",
    "Schema Validated"
  ]
}
```

---

## 📖 Usage Guide

### 1. Upload Documents

1. Click the **"Upload PDF"** button in the sidebar
2. Select one or more PDF files
3. Wait for the success message: "✓ All X file(s) uploaded successfully!"
4. Documents will appear in the **History** section

### 2. Query Documents

1. Select a document from the **History** list
2. Type your question in the search box
3. Press **Enter** or click the **Send** button
4. View the results:
   - **Agent Summary**: The main answer
   - **Key Points**: Important extracted information
   - **Agent Trace**: Execution path through the agents

### 3. Monitor Agent Execution

The right panel shows the **Agent Trace** visualization:
- 🔒 **Security**: Checks for jailbreak attempts
- 📚 **Retrieval**: Semantic search for relevant content
- 🧠 **Reasoning**: LLM generates response
- ✏️ **Validation**: Formats and verifies output

---

## 🛠️ Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
# Kill the process using port 8000
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

**Supabase connection error:**
- Verify `.env` file has correct SUPABASE_URL and SUPABASE_KEY
- Check network connectivity to Supabase

**Embedding errors:**
- Ensure OPENROUTER_API_KEY is valid
- Check OpenRouter account has available credits

### Frontend Issues

**Port 5173 already in use:**
```bash
cd frontend
npm run dev -- --port 5174
```

**Tailwind CSS not loading:**
```bash
cd frontend
npm install -D @tailwindcss/postcss
npm run dev
```

**Cannot connect to backend:**
- Verify backend is running on http://localhost:8000
- Check CORS is enabled (it is by default)
- Check browser console for detailed error messages

---

## 🔍 Debug Logging

The backend logs detailed information about each agent's execution:

```bash
# Watch logs in real-time
tail -f /tmp/backend.log

# Filter by agent
tail -f /tmp/backend.log | grep "\[Librarian\]"
tail -f /tmp/backend.log | grep "\[Analyst\]"
tail -f /tmp/backend.log | grep "\[Editor\]"
```

### Log Format

```
[Librarian] Embedding query: Who is Sarah?
[Librarian] Found 3 embeddings
[Librarian] Chunk 0: similarity = 0.4363
[Librarian] Returning 3 chunks
[Analyst] Processing query with context length: 2046
[Editor] Parsed result: ['summary', 'key_points', 'confidence_score']
```

---

## 🧠 How It Works

### Multi-Agent Architecture

```
User Query
    ↓
┌─────────────────────────┐
│  SecurityAgent          │ ← Validates input
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│  LibrarianAgent         │ ← Semantic search
│  (Vector Similarity)    │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│  AnalystAgent           │ ← LLM reasoning
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│  EditorAgent            │ ← Format & verify
└──────────┬──────────────┘
           ↓
      Final Response
```

### Vector Search Process

1. **Embed Query**: Convert user question to 1536-dim vector
2. **Fetch Embeddings**: Get all document chunk embeddings
3. **Calculate Similarity**: Cosine similarity score for each chunk
4. **Rank Results**: Sort by similarity (0.0 to 1.0)
5. **Return Top-5**: Most relevant chunks to LLM

### Similarity Scoring

- **0.8-1.0**: Highly relevant (exact match)
- **0.5-0.8**: Very relevant
- **0.3-0.5**: Somewhat relevant
- **0.0-0.3**: Low relevance

---

## 📦 Dependencies

### Backend
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `supabase` - Vector database client
- `openai` - LLM API client
- `pypdf` - PDF text extraction
- `numpy` - Vector calculations
- `python-multipart` - File uploads
- `python-dotenv` - Environment variables

### Frontend
- `react` - UI framework
- `vite` - Build tool
- `tailwindcss` - Styling
- `lucide-react` - Icons

---

## 🚀 Performance Tips

1. **Batch Uploads**: Upload multiple PDFs at once to save time
2. **Query Specificity**: More specific queries return better results
3. **Context Size**: Larger documents take longer to search
4. **Vector Indices**: For production, add Supabase vector indices for faster search

---

## 📝 License

MIT License - feel free to use this project for your own purposes.

---

## 🤝 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review backend logs: `tail -f /tmp/backend.log`
3. Check browser console for frontend errors
4. Verify all environment variables are set correctly

---

**Happy querying! 🚀**
