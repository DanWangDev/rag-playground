import { useState, useCallback } from "react";

interface Source {
  content: string;
  score: number;
  rank: number;
  source: string;
}

interface ChatState {
  question: string;
  answer: string;
  sources: Source[];
  elapsed: number;
  loading: boolean;
  streaming: boolean;
}

export function useChat() {
  const [state, setState] = useState<ChatState>({
    question: "",
    answer: "",
    sources: [],
    elapsed: 0,
    loading: false,
    streaming: false,
  });
  const [streamingToken, setStreamingToken] = useState("");

  const ask = useCallback(async (question: string, topK: number = 3) => {
    setState((s) => ({ ...s, question, loading: true, answer: "", sources: [] }));

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: topK }),
      });
      const data = await res.json();
      setState({
        question,
        answer: data.answer,
        sources: data.sources,
        elapsed: data.elapsed_ms,
        loading: false,
        streaming: false,
      });
    } catch (e) {
      setState((s) => ({
        ...s,
        answer: `Error: ${e instanceof Error ? e.message : "Unknown error"}`,
        loading: false,
        streaming: false,
      }));
    }
  }, []);

  const askStream = useCallback(async (question: string, topK: number = 3) => {
    setState((s) => ({ ...s, question, loading: true, answer: "", sources: [], streaming: true }));
    setStreamingToken("");

    try {
      const res = await fetch(`/api/chat/stream?question=${encodeURIComponent(question)}&top_k=${topK}`);
      const reader = res.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.done) {
                setState((s) => ({ ...s, streaming: false }));
              } else if (data.token) {
                setStreamingToken((prev) => prev + data.token);
                setState((s) => ({ ...s, answer: s.answer + data.token }));
              } else if (data.error) {
                setState((s) => ({ ...s, answer: `Error: ${data.error}`, streaming: false }));
              }
            } catch { /* ignore parse errors */ }
          }
        }
      }
      setState((s) => ({ ...s, loading: false, streaming: false }));
    } catch (e) {
      setState((s) => ({
        ...s,
        answer: `Error: ${e instanceof Error ? e.message : "Unknown"}`,
        loading: false,
        streaming: false,
      }));
    }
  }, []);

  const clear = useCallback(() => {
    setState({
      question: "", answer: "", sources: [],
      elapsed: 0, loading: false, streaming: false,
    });
    setStreamingToken("");
  }, []);

  return { ...state, streamingToken, ask, askStream, clear };
}
