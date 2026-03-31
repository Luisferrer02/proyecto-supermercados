"use client";

import { useEffect, useRef, useState } from "react";

interface LogLine {
  type: "log" | "error" | "done";
  msg: string;
}

interface Props {
  url: string;
  onDone?: (success: boolean) => void;
  autoStart?: boolean;
}

export function LiveLog({ url, onDone, autoStart = true }: Props) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [active, setActive] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const esRef = useRef<EventSource | null>(null);

  const start = () => {
    if (esRef.current) esRef.current.close();
    setLines([]);
    setActive(true);

    const es = new EventSource(url);
    esRef.current = es;

    const addLine = (type: LogLine["type"], data: string) => {
      try {
        const { message } = JSON.parse(data);
        setLines((prev) => [...prev, { type, msg: message }]);
      } catch {
        setLines((prev) => [...prev, { type, msg: data }]);
      }
    };

    es.addEventListener("log", (e) => addLine("log", (e as MessageEvent).data));
    es.addEventListener("error", (e) => addLine("error", (e as MessageEvent).data));
    es.addEventListener("done", (e) => {
      const { message } = JSON.parse((e as MessageEvent).data);
      setLines((prev) => [...prev, { type: "done", msg: message }]);
      setActive(false);
      es.close();
      const success = message.includes("code 0") || message.includes("successfully");
      onDone?.(success);
    });

    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) {
        setActive(false);
        es.close();
        onDone?.(false);
      }
    };
  };

  useEffect(() => {
    if (autoStart) start();
    return () => esRef.current?.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        {active && (
          <span className="inline-flex items-center gap-1.5 text-xs text-amber-600">
            <span className="h-2 w-2 rounded-full bg-amber-500 animate-pulse" />
            Running…
          </span>
        )}
        {!active && lines.length > 0 && (
          <span className="text-xs text-muted-foreground">
            {lines[lines.length - 1]?.msg.includes("code 0") ? "✓ Done" : "⚠ Finished"}
          </span>
        )}
      </div>
      <div className="bg-secondary rounded-lg p-4 h-72 overflow-y-auto font-mono text-xs leading-5 border border-border">
        {lines.length === 0 && (
          <span className="text-secondary-foreground/60">Waiting for output…</span>
        )}
        {lines.map((l, i) => (
          <div
            key={i}
            className={
              l.type === "error"
                ? "text-red-400"
                : l.type === "done"
                ? "text-green-400 font-semibold"
                : "text-secondary-foreground"
            }
          >
            {l.msg}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
