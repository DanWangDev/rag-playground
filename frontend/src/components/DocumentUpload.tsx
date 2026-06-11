import { useState, useRef, type FC } from "react";
import { Upload } from "lucide-react";

const DocumentUpload: FC = () => {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setResult("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/documents", { method: "POST", body: formData });
      const data = await res.json();
      setResult(
        data.status === "indexed"
          ? `Indexed "${data.filename}" — ${data.chunks} chunks (total: ${data.total_store_size})`
          : `Error: ${data.error || "Unknown"}`,
      );
    } catch (e) {
      setResult(`Upload failed: ${e}`);
    }
    setUploading(false);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div style={{ marginTop: 16 }}>
      <div
        className="upload-zone"
        onClick={() => inputRef.current?.click()}
      >
        <Upload size={20} style={{ marginBottom: 8 }} />
        <div>{uploading ? "Uploading..." : "Drop .md or .txt files here"}</div>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".md,.txt"
        onChange={handleUpload}
        style={{ display: "none" }}
      />
      {result && (
        <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 8 }}>
          {result}
        </div>
      )}
    </div>
  );
};

export default DocumentUpload;
