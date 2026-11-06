import { useEffect, useRef, useState, type FormEvent } from "react";
import { post, useResource } from "./api";
import { ErrorNotice, Modal } from "./common";
import type { Evidence, Run } from "./types";
interface Entry {
  name: string;
  path: string;
  kind: "directory" | "trl" | "verl" | "evidence";
}
const KINDS: Record<Entry["kind"], string> = {
  directory: "directory",
  trl: "TRL parquet",
  verl: "verl JSONL",
  evidence: "saved evidence",
};
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
  const title = mode === "run" ? "Load a rollout run" : "Open saved evidence";
  return (
    <Modal title={title} onClose={onClose}>
      <div className="picker">
        <div className="modal-heading">
          <h1 className="headline">{title}</h1>
          <button
            type="button"
            className="link"
            onClick={onClose}
            aria-label={`Close ${title.toLowerCase()}`}
          >
            Close
          </button>
        </div>
        <p className="dek">
          {mode === "run"
            ? "Choose TRL parquet or verl JSONL. Analysis stays on this machine, and files stay inside the launch root."
            : "Choose a saved evidence file. Reading stays on this machine, and files stay inside the launch root."}
        </p>
        <p className="picker-path">
          <button
            type="button"
            className="link"
            disabled={path === "." || busy}
            onClick={() => {
              setPath(path.split("/").slice(0, -1).join("/") || ".");
              setSelected(null);
            }}
          >
            Parent directory
          </button>
          <code>{path}</code>
        </p>
        <ErrorNotice message={listing.error} retry={listing.refresh} />
        {listing.loading && (
          <p className="loading" role="status">
            Reading directory…
          </p>
        )}
        <ul className="picker-list">
          {listing.data?.entries.map((entry) => {
            const chosen = selected?.path === entry.path;
            return (
              <li key={entry.path} className={chosen ? "selected" : undefined}>
                {entry.kind === "directory" ? (
                  <div className="picker-row">
                    <code>{entry.name}/</code>
                    <span className="picker-side">
                      {KINDS.directory}
                      <button
                        type="button"
                        className="link"
                        disabled={busy}
                        onClick={() => {
                          setPath(entry.path);
                          setSelected(null);
                        }}
                        aria-label={`Open directory ${entry.name}`}
                      >
                        Open
                      </button>
                      {mode === "run" && (
                        <button
                          type="button"
                          className="link"
                          disabled={busy}
                          onClick={() => choose(entry)}
                          aria-pressed={chosen}
                          aria-label={`Select directory ${entry.name}`}
                        >
                          Select
                        </button>
                      )}
                    </span>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="picker-entry"
                    disabled={
                      busy ||
                      (mode === "evidence" && entry.kind !== "evidence") ||
                      (mode === "run" && entry.kind === "evidence")
                    }
                    onClick={() => choose(entry)}
                    aria-pressed={chosen}
                    aria-label={`Select ${entry.name}`}
                  >
                    <code>{entry.name}</code>
                    <span className="picker-side">{KINDS[entry.kind]}</span>
                  </button>
                )}
              </li>
            );
          })}
        </ul>
        {listing.data?.entries.length === 0 && (
          <p className="quiet-note">No supported inputs in this directory.</p>
        )}
        {mode === "run" && (
          <p className="link-row">
            <button
              type="button"
              className="link"
              disabled={busy || listing.loading || Boolean(listing.error)}
              onClick={() =>
                setSelected({ name: path, path, kind: "directory" })
              }
            >
              Use this directory
            </button>
          </p>
        )}
        <form className="picker-form" onSubmit={submit}>
          <p className="section-note">
            Selected:{" "}
            <code>{selected?.path ?? "Choose a file or directory"}</code>
          </p>
          {mode === "run" && (
            <div className="picker-fields">
              <label className="picker-field">
                Run name
                <input
                  className="field-line"
                  aria-label="Run name"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  required
                  disabled={busy}
                />
              </label>
              <label className="picker-field">
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
              <label className="picker-field wide">
                Reward column (optional)
                <input
                  className="field-line"
                  value={reward}
                  onChange={(event) => setReward(event.target.value)}
                  placeholder="Use adapter default"
                  disabled={busy || format !== "trl"}
                />
              </label>
            </div>
          )}
          <ErrorNotice message={error} />
          <div className="picker-actions">
            <button type="button" className="link" onClick={onClose}>
              Cancel
            </button>
            <button
              type="submit"
              className="pill"
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
      </div>
    </Modal>
  );
}
