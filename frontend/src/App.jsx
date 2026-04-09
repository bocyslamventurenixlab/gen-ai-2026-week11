import React, { useState, useEffect } from 'react';
import { Upload, ShieldCheck, Search, Cpu, CheckSquare, Loader2, AlertTriangle, Send, FileText } from 'lucide-react';

const App = () => {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [trace, setTrace] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [error, setError] = useState('');

  const API_BASE = '/api';

  useEffect(() => { 
    fetchDocs();
    // Refresh documents every 5 seconds
    const interval = setInterval(fetchDocs, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchDocs = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (!res.ok) {
        console.error('Failed to fetch documents:', res.status);
        setDocuments([]);
        return;
      }
      const data = await res.json();
      // Ensure data is always an array
      setDocuments(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Error fetching documents:', err);
      setDocuments([]);
    }
  };

  const handleUpload = async (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    // Validate all files are PDFs
    const invalidFiles = Array.from(files).filter(f => !f.name.endsWith('.pdf'));
    if (invalidFiles.length > 0) {
      setUploadStatus(`Only PDF files are allowed. Invalid: ${invalidFiles.map(f => f.name).join(', ')}`);
      return;
    }

    setUploading(true);
    setUploadStatus(`Uploading ${files.length} file(s)...`);

    let successCount = 0;
    let failedFiles = [];

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        try {
          const formData = new FormData();
          formData.append('file', file);

          const res = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
          });

          if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
          }

          const data = await res.json();
          successCount++;
          setUploadStatus(`Uploading ${files.length} file(s)... (${successCount}/${files.length} done)`);
        } catch (err) {
          failedFiles.push(`${file.name} (${err.message})`);
        }
      }

      // Refresh documents list
      await fetchDocs();

      if (failedFiles.length === 0) {
        setUploadStatus(`✓ All ${successCount} file(s) uploaded successfully!`);
      } else {
        setUploadStatus(`✓ ${successCount}/${files.length} uploaded. Failed: ${failedFiles.join(', ')}`);
      }

      // Clear status after 3 seconds
      setTimeout(() => setUploadStatus(''), 3000);
    } catch (err) {
      setUploadStatus(`✗ Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
      // Reset file input
      event.target.value = '';
    }
  };

  const handleProcess = async () => {
    if (!selectedDoc) {
      setError('Please select a document first');
      return;
    }

    if (!query.trim()) {
      setError('Please enter a query');
      return;
    }

    setLoading(true);
    setResult(null);
    setTrace([]);
    setError('');

    try {
      console.log('Processing query:', { document_id: selectedDoc, query });
      
      const res = await fetch(`${API_BASE}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_id: selectedDoc, query })
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(errorData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      console.log('Query response:', data);
      
      setResult(data);
      if (data.trace) setTrace(data.trace);
    } catch (err) {
      console.error('Error processing query:', err);
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0c] text-slate-300 font-sans p-6">
      <div className="max-w-7xl mx-auto grid grid-cols-12 gap-6">
        
        {/* Sidebar */}
        <div className="col-span-12 lg:col-span-3 space-y-6">
          <div className="flex items-center gap-3 p-4 bg-slate-900/50 rounded-2xl border border-slate-800">
            <div className="bg-emerald-500/20 p-2 rounded-lg">
              <ShieldCheck className="text-emerald-500" size={24} />
            </div>
            <h1 className="font-bold text-white text-xl">Agentic Read</h1>
          </div>

          <div className="bg-slate-900/30 p-4 rounded-2xl border border-slate-800/50 space-y-4">
            <h2 className="text-sm font-bold uppercase tracking-widest text-slate-500">History</h2>
            
            {/* Upload Section */}
            <div className="relative">
              <input
                type="file"
                accept=".pdf"
                onChange={handleUpload}
                disabled={uploading}
                multiple
                className="hidden"
                id="pdf-upload"
              />
              <label
                htmlFor="pdf-upload"
                className="flex items-center justify-center gap-2 w-full p-3 bg-emerald-600/20 hover:bg-emerald-600/30 border border-emerald-500/50 rounded-xl cursor-pointer transition-all disabled:opacity-50"
              >
                <Upload size={16} className="text-emerald-500" />
                <span className="text-sm font-semibold text-emerald-400">
                  {uploading ? 'Uploading...' : 'Upload PDF'}
                </span>
              </label>
              {uploadStatus && (
                <p className={`text-xs mt-2 p-2 rounded ${uploadStatus.startsWith('✓') ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
                  {uploadStatus}
                </p>
              )}
            </div>

            <div className="space-y-2">
              {documents.map(doc => (
                <button 
                  key={doc.id}
                  onClick={() => setSelectedDoc(doc.id)}
                  className={`w-full text-left p-3 rounded-xl border text-sm transition-all ${selectedDoc === doc.id ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400' : 'border-transparent hover:bg-slate-800/50'}`}
                >
                  <FileText size={14} className="inline mr-2" />
                  {doc.title}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Main Panel */}
        <div className="col-span-12 lg:col-span-6 space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl">
            <div className="flex flex-col gap-4">
              <div className="flex gap-4">
                <input 
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && !loading && selectedDoc && handleProcess()}
                  placeholder="Ask the Agent team..."
                  disabled={loading}
                  className="flex-1 bg-black/40 border border-slate-800 rounded-2xl px-6 py-4 focus:ring-2 focus:ring-emerald-500/50 outline-none disabled:opacity-50"
                />
                <button 
                  onClick={handleProcess}
                  disabled={loading || !selectedDoc || !query.trim()}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white px-6 rounded-2xl transition-all disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center min-w-[50px]"
                >
                  {loading ? <Loader2 className="animate-spin" size={20} /> : <Send size={20} />}
                </button>
              </div>
              {!selectedDoc && (
                <p className="text-xs text-yellow-400 flex items-center gap-2">
                  ⚠️ Select a document from the list first
                </p>
              )}
              {error && (
                <p className="text-xs text-red-400 flex items-center gap-2">
                  ✗ {error}
                </p>
              )}
            </div>
          </div>

          {result && (
            <div className={`p-8 rounded-3xl border ${result.is_safe ? 'bg-slate-900 border-slate-800' : 'bg-red-950/20 border-red-500/50'}`}>
              {!result.is_safe && <div className="flex items-center gap-2 text-red-500 mb-4 font-bold"><AlertTriangle /> Security Violation Blocked</div>}
              <h2 className="text-2xl font-bold text-white mb-6">Agent Summary</h2>
              <p className="text-lg leading-relaxed text-slate-400 mb-8">{result.summary}</p>
              
              <div className="grid grid-cols-1 gap-4">
                {result.key_points.map((pt, i) => (
                  <div key={i} className="flex gap-4 p-4 bg-black/30 rounded-2xl border border-slate-800">
                    <span className="text-emerald-500 font-mono">0{i+1}</span>
                    <p className="text-sm">{pt}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Trace Panel */}
        <div className="col-span-12 lg:col-span-3 space-y-6">
          <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-6 min-h-[400px]">
            <h2 className="text-sm font-bold uppercase tracking-widest text-slate-500 mb-6">Agent Trace</h2>
            <div className="space-y-8 relative">
              <div className="absolute left-[11px] top-2 bottom-2 w-0.5 bg-slate-800"></div>
              
              {[
                { label: 'Security', icon: ShieldCheck, step: 'Gatekeeper' },
                { label: 'Retrieval', icon: Search, step: 'Librarian' },
                { label: 'Reasoning', icon: Cpu, step: 'Analyst' },
                { label: 'Validation', icon: CheckSquare, step: 'Editor' }
              ].map((item, i) => {
                const isCompleted = trace.length > i;
                return (
                  <div key={i} className="flex gap-4 relative z-10">
                    <div className={`p-1 rounded-full border-4 border-[#0a0a0c] ${isCompleted ? 'bg-emerald-500' : 'bg-slate-800'}`}>
                      <item.icon size={12} className={isCompleted ? 'text-black' : 'text-slate-500'} />
                    </div>
                    <div>
                      <p className={`text-xs font-bold ${isCompleted ? 'text-emerald-500' : 'text-slate-600'}`}>{item.label}</p>
                      <p className="text-[10px] text-slate-500">{item.step}</p>
                    </div>
                  </div>
                );
              })}
            </div>

            {loading && (
              <div className="mt-12 flex items-center gap-3 text-emerald-500 animate-pulse">
                <Loader2 className="animate-spin" size={16} />
                <span className="text-xs font-bold uppercase tracking-tighter">Processing...</span>
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

export default App;