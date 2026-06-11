import type { FC } from "react";
import { Settings } from "lucide-react";

interface Config {
  chat_model: string;
  embed_model: string;
  top_k: number;
  temperature: number;
  chunk_size: number;
  chunk_overlap: number;
}

interface Props {
  config: Config | null;
  topK: number;
  onTopKChange: (k: number) => void;
  visible: boolean;
  onToggle: () => void;
}

const ConfigPanel: FC<Props> = ({ config, topK, onTopKChange, visible, onToggle }) => {
  return (
    <>
      <button
        className="btn"
        onClick={onToggle}
        style={{ fontSize: 12 }}
        title="Toggle config panel"
      >
        <Settings size={14} /> {visible ? "Hide" : "Config"}
      </button>
      {visible && config && (
        <div className="config-panel">
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Configuration</h3>

          <div>
            <label>Chat Model</label>
            <div style={{ fontSize: 13, color: "var(--accent)" }}>{config.chat_model}</div>
          </div>
          <div>
            <label>Embed Model</label>
            <div style={{ fontSize: 13, color: "var(--accent)" }}>{config.embed_model}</div>
          </div>

          <div>
            <label>Top-K Results: {topK}</label>
            <input
              type="range"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => onTopKChange(Number(e.target.value))}
            />
          </div>

          <div>
            <label>Temperature: {config.temperature}</label>
            <input type="range" min={0} max={2} step={0.1} value={config.temperature} readOnly />
          </div>

          <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 8 }}>
            Chunk size: {config.chunk_size} · Overlap: {config.chunk_overlap}
          </div>
        </div>
      )}
    </>
  );
};

export default ConfigPanel;
