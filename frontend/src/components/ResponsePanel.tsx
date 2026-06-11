import type { FC } from "react";
import { Clock } from "lucide-react";

interface Props {
  answer: string;
  elapsed: number;
  streaming: boolean;
  loading: boolean;
}

const ResponsePanel: FC<Props> = ({ answer, elapsed, streaming, loading }) => {
  if (!answer && !loading) return null;

  return (
    <div className="card">
      <div className="card-header">
        <span>🤖 Answer</span>
        {elapsed > 0 && (
          <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 4 }}>
            <Clock size={12} /> {elapsed}ms
          </span>
        )}
      </div>
      {loading && !answer && (
        <div style={{ color: "var(--text-muted)", fontSize: 14 }}>Thinking...</div>
      )}
      <div
        style={{ fontSize: 14, lineHeight: 1.7, whiteSpace: "pre-wrap" }}
        className={streaming ? "streaming-cursor" : ""}
      >
        {answer}
      </div>
    </div>
  );
};

export default ResponsePanel;
