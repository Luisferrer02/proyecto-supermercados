"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Upload } from "lucide-react";
import { api } from "@/lib/api";

interface UploadedFile {
  name: string;
  size: number;
  modified: string;
}

function fmt(bytes: number) {
  return bytes > 1024 * 1024
    ? `${(bytes / 1024 / 1024).toFixed(1)} MB`
    : `${(bytes / 1024).toFixed(0)} KB`;
}

export default function UploadPage() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const loadFiles = async () => {
    try {
      const d = await api.listFiles();
      setFiles(d.files ?? []);
    } catch {}
  };

  useEffect(() => { loadFiles(); }, []);

  const uploadFiles = async (selected: FileList | File[]) => {
    setUploading(true);
    const form = new FormData();
    Array.from(selected).forEach((f) => form.append("files", f));
    try {
      await api.uploadFiles(form);
      await loadFiles();
    } finally {
      setUploading(false);
    }
  };

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const deleteFile = async (name: string) => {
    await api.deleteFile(name);
    await loadFiles();
  };

  return (
    <div className="space-y-8 max-w-3xl relative">
      <div
        className="fixed inset-0 bg-cover bg-center opacity-[0.09] blur-sm pointer-events-none -z-10"
        style={{ backgroundImage: "url(/uploadcsv.png)" }}
      />
      <div className="animate-fade-in-up">
        <h1 className="text-3xl font-heading font-extrabold uppercase tracking-tight text-secondary">
          Upload CSV Files
        </h1>
        <p className="text-muted-foreground text-sm mt-2">
          Upload monthly sales CSVs named <code className="bg-muted px-1.5 py-0.5 rounded text-xs">sales_YYYY_MM_monthname.csv</code>
        </p>
      </div>

      {/* Drop zone */}
      <div
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
          dragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".csv"
          className="hidden"
          onChange={(e) => e.target.files && uploadFiles(e.target.files)}
        />
        <Upload size={40} className="text-primary mx-auto mb-3" />
        <p className="font-medium">Drop CSV files here or click to browse</p>
        <p className="text-xs text-muted-foreground mt-1">Accepts sales_*.csv files</p>
        {uploading && <p className="text-sm text-amber-600 mt-2">Uploading…</p>}
      </div>

      {/* File list */}
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle className="text-sm flex items-center justify-between">
            Uploaded files
            <Badge variant="secondary">{files.length}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {files.length === 0 ? (
            <p className="text-sm text-muted-foreground">No files uploaded yet.</p>
          ) : (
            <div className="space-y-1">
              {files.map((f) => (
                <div key={f.name} className="flex items-center justify-between text-sm py-1.5 border-b border-border last:border-0">
                  <div>
                    <span className="font-mono">{f.name}</span>
                    <span className="text-muted-foreground text-xs ml-2">{fmt(f.size)}</span>
                  </div>
                  <button
                    onClick={() => deleteFile(f.name)}
                    className="text-xs text-destructive hover:underline"
                  >
                    Delete
                  </button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
