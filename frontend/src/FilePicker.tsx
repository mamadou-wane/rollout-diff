import { useEffect, useRef, useState, type FormEvent } from "react";
import { post, useResource } from "./api";
import { ErrorNotice, Modal } from "./common";
import type { Evidence, Run } from "./types";
interface Entry {
  name: string;
  path: string;
  kind: "directory" | "trl" | "verl" | "evidence";
}
export function FilePicker({
  mode,
  runId,
  onClose,
  onRun,
  onEvidence,
}: {
  mode: "run" | "evidence";
  runId?: string;
  onClose: () => void;
  onRun: (run: Run) => void;
  onEvidence: (value: Evidence) => void;
}) {
  const [path, setPath] = useState("."),
    [selected, setSelected] = useState<Entry | null>(null);
  const [name, setName] = useState(""),
    [format, setFormat] = useState("trl"),
    [reward, setReward] = useState("");
  const [busy, setBusy] = useState(false),
    [error, setError] = useState<string | null>(null);
  const listing = useResource<{ path: string; entries: Entry[] }>(
    `/api/files?path=${encodeURIComponent(path)}`,
  );
  const pending = useRef<AbortController | null>(null);
  useEffect(() => () => pending.current?.abort(), [mode, runId]);
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!selected) return;
    setBusy(true);
    setError(null);
    const controller = new AbortController();
    pending.current?.abort();
    pending.current = controller;
    try {
      if (mode === "run") {
        const result = await post<Run>(
          "/api/runs",
          {
            path: selected.path,
            format,
            run: name,
            reward_column: format === "trl" ? reward || null : null,
          },
          controller.signal,
        );
        if (controller.signal.aborted) return;
        onRun(result);
      } else {
        const result = await post<Evidence>(
          "/api/evidence",
          {
            path: selected.path,
            run_id: runId ?? null,
          },
          controller.signal,
        );
        if (controller.signal.aborted) return;
        onEvidence(result);
      }
      onClose();
    } catch (error) {
      if (controller.signal.aborted) return;
      setError(
        error instanceof Error ? error.message : "Unable to open this file.",
      );
      setBusy(false);
    }
  }

  function choose(entry: Entry) {
    setSelected(entry);
    if (entry.kind === "trl" || entry.kind === "verl") setFormat(entry.kind);
  }
  return (
    <Modal
      title={mode === "run" ? "Load local run" : "Open saved evidence"}
      onClose={onClose}
    >
      <p className="muted">
        Choose an input within the launch root. Files are read locally.
      </p>
      <div className="pathbar">
        <button
          disabled={path === "." || busy}
          onClick={() => {
            setPath(path.split("/").slice(0, -1).join("/") || ".");
            setSelected(null);
          }}
        >
          Parent directory
        </button>
        <code>{path}</code>
      </div>
      <ErrorNotice message={listing.error} retry={listing.refresh} />
      {listing.loading && <p role="status">Reading directory…</p>}
      <ul className="file-list">
        {listing.data?.entries.map((entry) => (
          <li key={entry.path}>
            {entry.kind === "directory" ? (
              <>
                <button
                  className="file-name"
                  disabled={busy}
                  onClick={() => {
                    setPath(entry.path);
                    setSelected(null);
                  }}
                  aria-label={`Open directory ${entry.name}`}
                >
                  {entry.name}/
                </button>
                {mode === "run" && (
                  <button
                    disabled={busy}
                    onClick={() => choose(entry)}
                    aria-label={`Select directory ${entry.name}`}
                  >
                    Select directory
                  </button>
                )}
              </>
            ) : (
              <button
                disabled={
                  busy ||
                  (mode === "evidence" && entry.kind !== "evidence") ||
                  (mode === "run" && entry.kind === "evidence")
                }
                className={`file-name ${selected?.path === entry.path ? "selected" : ""}`}
                onClick={() => choose(entry)}
                aria-label={`Select ${entry.name}`}
              >
                {entry.name}
                <span className="muted">{entry.kind}</span>
              </button>
            )}
          </li>
        ))}
      </ul>
      {listing.data?.entries.length === 0 && (
        <p className="empty">No supported inputs in this directory.</p>
      )}
      {mode === "run" && (
        <button
          disabled={busy || listing.loading || Boolean(listing.error)}
          onClick={() => setSelected({ name: path, path, kind: "directory" })}
        >
          Use this directory
        </button>
      )}
      <form onSubmit={submit}>
        <p className="selection">
          Selected:{" "}
          <code>{selected?.path ?? "Choose a file or directory"}</code>
        </p>
        {mode === "run" && (
          <div className="form-grid">
            <label>
              Run name
              <input
                aria-label="Run name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                required
                disabled={busy}
              />
            </label>
            <label>
              Input format
              <select
                value={format}
                onChange={(event) => setFormat(event.target.value)}
                disabled={busy}
              >
                <option value="trl">TRL parquet</option>
                <option value="verl">verl JSONL / directory</option>
              </select>
            </label>
            <label className="wide">
              Reward column (optional)
              <input
                value={reward}
                onChange={(event) => setReward(event.target.value)}
                placeholder="Use adapter default"
                disabled={busy || format !== "trl"}
              />
            </label>
          </div>
        )}
        <ErrorNotice message={error} />
        <div className="form-actions">
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            className="primary"
            disabled={busy || !selected || (mode === "run" && !name.trim())}
          >
            {busy
              ? "Reading…"
              : mode === "run"
                ? "Load selected run"
                : "Read selected evidence"}
          </button>
        </div>
      </form>
    </Modal>
  );
}
