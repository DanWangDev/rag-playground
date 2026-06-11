import { useState, type FC } from "react";
import { GitCompare } from "lucide-react";

interface Props {
  question: string;
  onAsk: (q: string) => void;
}

const CompareView: FC<Props> = ({ question, onAsk }) => {
  const [ragAnswer, setRagAnswer] = useState("");
  const [llmAnswer, setLlmAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const runCompare = async () => {
    if (!question) return;
    setLoading(true);
    try {
      const res = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      setRagAnswer(data.rag_answer);
      setLlmAnswer(data.llm_only_answer);
    } catch (e) {
      setRagAnswer(`Error: ${e}`);
    }
    setLoading(false);
  };

  return (
    <div className="card">
      <div className="card-header">
        <GitCompare size={14} />
        <span>RAG vs LLM-only</span>
        <button
          className="btn"
          style={{ marginLeft: "auto", fontSize: 11, padding: "4px 10px" }}
          onClick={runCompare}
          disabled={loading || !question}
        >
          Compare
        </button>
      </div>
      {(ragAnswer || llmAnswer) && (
        <div className="compare-grid">
          <div>
            <div style={{ fontSize: 12, color: "var(--green)", marginBottom: 6 }}>
              ✅ With RAG
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
              {ragAnswer}
            </div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: "var(--yellow)", marginBottom: 6 }}>
              ⚡ LLM Only
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
              {llmAnswer}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CompareView;
