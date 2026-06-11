import { useState } from "react";
import { Brain, PanelRightClose, PanelRightOpen } from "lucide-react";
import QueryInput from "./components/QueryInput";
import ResponsePanel from "./components/ResponsePanel";
import SourceList from "./components/SourceList";
import CompareView from "./components/CompareView";
import ConfigPanel from "./components/ConfigPanel";
import DocumentUpload from "./components/DocumentUpload";
import { useChat } from "./hooks/useChat";
import { useConfig } from "./hooks/useConfig";

export default function App() {
  const { question, answer, sources, elapsed, loading, streaming, ask, askStream, clear } =
    useChat();
  const { config, topK, updateTopK } = useConfig();
  const [configOpen, setConfigOpen] = useState(false);
  const [showCompare, setShowCompare] = useState(false);

  return (
    <div className="app">
      <header className="header">
        <h1>
          <Brain size={20} color="var(--accent)" /> RAG Playground
        </h1>
        <div className="header-actions">
          <ConfigPanel
            config={config}
            topK={topK}
            onTopKChange={updateTopK}
            visible={configOpen}
            onToggle={() => setConfigOpen((o) => !o)}
          />
          <button className="btn" style={{ fontSize: 12 }} onClick={clear} title="Clear chat">
            ✕ Clear
          </button>
          <button
            className="btn"
            style={{ fontSize: 12 }}
            onClick={() => setShowCompare((v) => !v)}
            title="Toggle compare"
          >
            {showCompare ? <PanelRightClose size={14} /> : <PanelRightOpen size={14} />}
          </button>
        </div>
      </header>

      <div className="main-content">
        <div className="chat-area">
          <div className="messages">
            <ResponsePanel
              answer={answer}
              elapsed={elapsed}
              streaming={streaming}
              loading={loading}
            />
            <SourceList sources={sources} />
            {showCompare && <CompareView question={question} onAsk={ask} />}
            <DocumentUpload />
          </div>
          <QueryInput
            onAsk={(q) => ask(q, topK)}
            onStream={(q) => askStream(q, topK)}
            loading={loading}
          />
        </div>
      </div>
    </div>
  );
}
