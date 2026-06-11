import { useState, type FC } from "react";
import { Send, Zap } from "lucide-react";

interface Props {
  onAsk: (q: string) => void;
  onStream: (q: string) => void;
  loading: boolean;
}

const QueryInput: FC<Props> = ({ onAsk, onStream, loading }) => {
  const [value, setValue] = useState("");

  const handleSubmit = (stream: boolean) => {
    if (!value.trim() || loading) return;
    if (stream) onStream(value.trim());
    else onAsk(value.trim());
  };

  return (
    <div className="card">
      <div className="input-row">
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(false);
            }
          }}
          placeholder="Ask a question about your documents..."
          disabled={loading}
          autoFocus
        />
        <button
          className="btn btn-primary"
          onClick={() => handleSubmit(false)}
          disabled={loading || !value.trim()}
        >
          <Send size={16} /> Ask
        </button>
        <button
          className="btn"
          onClick={() => handleSubmit(true)}
          disabled={loading || !value.trim()}
          title="Stream response"
        >
          <Zap size={16} />
        </button>
      </div>
    </div>
  );
};

export default QueryInput;
