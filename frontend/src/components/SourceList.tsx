import type { FC } from "react";
import { BookOpen } from "lucide-react";

interface Source {
  content: string;
  score: number;
  rank: number;
  source: string;
}

interface Props {
  sources: Source[];
}

const SourceList: FC<Props> = ({ sources }) => {
  if (!sources.length) return null;

  return (
    <div className="card">
      <div className="card-header">
        <BookOpen size={14} />
        <span>Sources ({sources.length})</span>
      </div>
      {sources.map((s, i) => (
        <div key={i} className="source-item">
          <div className="source-score">
            #{s.rank} — relevance: {(s.score * 100).toFixed(0)}% — {s.source}
          </div>
          <div className="source-text">{s.content}</div>
        </div>
      ))}
    </div>
  );
};

export default SourceList;
