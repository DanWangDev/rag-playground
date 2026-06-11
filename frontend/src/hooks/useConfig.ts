import { useState, useEffect } from "react";

interface Config {
  chat_model: string;
  embed_model: string;
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  temperature: number;
  ollama_host: string;
}

export function useConfig() {
  const [config, setConfig] = useState<Config | null>(null);
  const [topK, setTopK] = useState(3);

  useEffect(() => {
    fetch("/api/config")
      .then((r) => r.json())
      .then((c) => {
        setConfig(c);
        setTopK(c.top_k);
      })
      .catch(() => {});
  }, []);

  const updateTopK = (k: number) => {
    setTopK(k);
    fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ top_k: k }),
    }).catch(() => {});
  };

  return { config, topK, updateTopK };
}
