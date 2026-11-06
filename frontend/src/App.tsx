import { useEffect, useRef, useState } from "react";
import { downloadReport, initializeToken, request } from "./api";
import { ErrorNotice, figure, known, runCounts, text } from "./common";
import { CompareView } from "./CompareView";
import { EventsView } from "./EventsView";
import { EvidenceView } from "./EvidenceView";
import { FilePicker } from "./FilePicker";
import { ResponseDrawer } from "./ResponseDrawer";
import type {
  Citation,
  CompareSeed,
  Evidence,
  ReaderContext,
  Run,
} from "./types";

type Tab = "Events" | "Compare" | "Saved evidence";
const TABS: Tab[] = ["Events", "Compare", "Saved evidence"];
const STAGES: Record<string, string> = {
  queued: "Run queued",
  loading: "Reading source files",
  analyzing: "Analyzing rollout evidence",
};
function RunMenu({
  run,
  runs,
  rootLabel,
  onSelect,
  onUnload,
}: {
  run: Run;
  runs: Run[];
  rootLabel?: string;
  onSelect: (id: string) => void;
  onUnload: () => Promise<void>;
}) {
  const [open, setOpen] = useState(false),
    [error, setError] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null),
    trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (ref.current?.contains(document.activeElement))
        trigger.current?.focus();
      setOpen(false);
    };
    const onPointer = (event: PointerEvent) => {
      if (!ref.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPointer);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onPointer);
    };
  }, [open]);
  async function exportFile(format: string) {
    setError(null);
    try {
      await downloadReport(
        `/api/runs/${run.id}/export?format=${format}`,
        `rollout-report.${format}`,
      );
    } catch (error) {
      setError(error instanceof Error ? error.message : "Export failed.");
    }
  }
  async function unload() {
    setError(null);
    try {
      await onUnload();
      trigger.current?.focus();
      setOpen(false);
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Unable to unload run.",
      );
    }
  }
  return (
    <div
      className="run-menu"
      ref={ref}
      onBlur={(event) => {
        if (event.relatedTarget && !ref.current?.contains(event.relatedTarget))
          setOpen(false);
      }}
    >
      <button
        ref={trigger}
        type="button"
        className="link"
        aria-expanded={open}
        aria-label={`Run ${run.run}, ${run.status}`}
        onClick={() => setOpen((value) => !value)}
      >
        <code>{run.run}</code>
      </button>
      {open && (
        <section className="popover" aria-label="Run summary">
          {runs.length > 1 && (
            <label className="popover-field">
              Loaded run
              <select
                value={run.id}
                onChange={(event) => onSelect(event.target.value)}
              >
                {runs.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.run}, {item.status}
                  </option>
                ))}
              </select>
            </label>
          )}
          {run.counts ? (
            <>
              <p>{runCounts(run.counts)}</p>
              <p className="section-note">
                {text(run.counts.unscorable_series)} of{" "}
                {text(run.counts.series)} series were unscorable.
              </p>
            </>
          ) : (
            <p className="section-note">Status: {run.status}.</p>
          )}
          {run.source && (
            <p className="section-note">
              {known(run.elapsed_seconds) &&
                `Analyzed in ${figure(run.elapsed_seconds)} s. `}
              Files stay inside <code>{rootLabel ?? "the launch root"}</code>.
            </p>
          )}
          {run.source && (
            <details className="more">
              <summary>Source fingerprint</summary>
              <p className="literal-line">
                <code>{run.source.path}</code>
              </p>
              <p className="literal-line">
                <code>{run.source.canonical_sha256}</code>
              </p>
            </details>
          )}
          <ErrorNotice message={error} />
          <p className="link-row">
            {run.status === "ready" && (
              <>
                <button
                  type="button"
                  className="link"
                  onClick={() => exportFile("json")}
                >
                  Export JSON
                </button>
                <button
                  type="button"
                  className="link"
                  onClick={() => exportFile("html")}
                >
                  Export HTML
                </button>
              </>
            )}
            {["ready", "failed"].includes(run.status) && (
              <button type="button" className="link" onClick={unload}>
                Unload run
              </button>
            )}
          </p>
        </section>
      )}
    </div>
  );
}
export default function App() {
  const [credentials, setCredentials] = useState(() => ({
    token: initializeToken(),
    generation: 0,
  }));
  useEffect(() => {
    const consumeFragment = () => {
      if (!new URLSearchParams(window.location.hash.slice(1)).get("token"))
        return;
      const token = initializeToken();
      setCredentials((previous) => ({
        token,
        generation: previous.generation + 1,
      }));
    };
    window.addEventListener("hashchange", consumeFragment);
    return () => window.removeEventListener("hashchange", consumeFragment);
  }, []);
  return (
    <Investigation key={credentials.generation} token={credentials.token} />
  );
}
function Investigation({ token }: { token: string }) {
  const [session, setSession] = useState<{ root_label: string } | null>(null),
    [runs, setRuns] = useState<Run[]>([]),
    [runId, setRunId] = useState("");
  const [tab, setTab] = useState<Tab>("Events"),
    [picker, setPicker] = useState<"run" | "evidence" | null>(null),
    [citation, setCitation] = useState<{
      value: Citation;
      context: ReaderContext;
    } | null>(null),
    [seed, setSeed] = useState<CompareSeed | null>(null),
    [saved, setSaved] = useState<Evidence | null>(null);
  const [error, setError] = useState<string | null>(null),
    [initialLoading, setInitialLoading] = useState(true),
    [revision, setRevision] = useState(0);
  useEffect(() => {
    if (!token) {
      setInitialLoading(false);
      return;
    }
    let active = true;
    const controller = new AbortController();
    setInitialLoading(true);
    setError(null);
    Promise.all([
      request<{ root_label: string }>("/api/session", {
        signal: controller.signal,
      }),
      request<{ runs: Run[] }>("/api/runs", { signal: controller.signal }),
    ])
      .then(([info, data]) => {
        if (active) {
          setSession(info);
          setRuns(data.runs);
          setRunId((value) => value || data.runs[0]?.id || "");
        }
      })
      .catch((error) => {
        if (active)
          setError(
            error instanceof Error ? error.message : "Session unavailable.",
          );
      })
      .finally(() => {
        if (active) setInitialLoading(false);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [token, revision]);
  useEffect(() => {
    if (
      !runs.some((run) =>
        ["queued", "loading", "analyzing"].includes(run.status),
      )
    )
      return;
    let active = true;
    const controller = new AbortController();
    const timer = setTimeout(
      () =>
        request<{ runs: Run[] }>("/api/runs", { signal: controller.signal })
          .then((data) => {
            if (active) setRuns(data.runs);
          })
          .catch((error) => {
            if (active)
              setError(
                error instanceof Error
                  ? error.message
                  : "Run status unavailable.",
              );
          }),
      700,
    );
    return () => {
      active = false;
      clearTimeout(timer);
      controller.abort();
    };
  }, [runs]);
  const selected = runs.find((run) => run.id === runId);
  const loadButton = useRef<HTMLButtonElement>(null),
    hadRun = useRef(false);
  // Unloading the last run removes the focused run menu.
  useEffect(() => {
    if (hadRun.current && !selected) loadButton.current?.focus();
    hadRun.current = Boolean(selected);
  }, [selected]);
  const current = useRef({ tab, picker });
  current.current = { tab, picker };
  // Panels stay mounted, so a citation can resolve after the user moved on.
  function openCitation(origin: Tab) {
    return (value: Citation, context: ReaderContext) => {
      if (current.current.tab === origin && !current.current.picker)
        setCitation({ value, context });
    };
  }
  function resetRunState() {
    setCitation(null);
    setSaved(null);
    setSeed(null);
  }
  function loaded(run: Run) {
    setRuns((current) => [
      ...current.filter((item) => item.id !== run.id),
      run,
    ]);
    setRunId(run.id);
    setTab("Events");
    resetRunState();
  }
  async function unload() {
    if (!selected) return;
    await request(`/api/runs/${selected.id}`, { method: "DELETE" });
    setRuns((current) => current.filter((item) => item.id !== selected.id));
    setRunId(runs.find((item) => item.id !== selected.id)?.id ?? "");
    resetRunState();
  }
  return (
    <>
      <header className="masthead">
        <span className="brand">Rollout Diff</span>
        <div className="tabs" role="tablist" aria-label="Investigation views">
          {TABS.map((name) => (
            <button
              type="button"
              key={name}
              id={`tab-${name.replaceAll(" ", "-")}`}
              role="tab"
              aria-selected={tab === name}
              aria-controls="investigation-panel"
              tabIndex={tab === name ? 0 : -1}
              onKeyDown={(event) => {
                if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
                  const next =
                    TABS[
                      (TABS.indexOf(tab) +
                        (event.key === "ArrowRight" ? 1 : 2)) %
                        3
                    ];
                  setTab(next);
                  document
                    .getElementById(`tab-${next.replaceAll(" ", "-")}`)
                    ?.focus();
                }
              }}
              onClick={() => {
                setTab(name);
                setCitation(null);
              }}
            >
              {name}
            </button>
          ))}
        </div>
        <div className="run-controls">
          {selected && (
            <RunMenu
              run={selected}
              runs={runs}
              rootLabel={session?.root_label}
              onSelect={(id) => {
                setRunId(id);
                resetRunState();
              }}
              onUnload={unload}
            />
          )}
          <button
            ref={loadButton}
            type="button"
            className="link"
            onClick={() => setPicker("run")}
            disabled={!session}
          >
            Load run
          </button>
        </div>
      </header>
      <main className="page">
        {!token && (
          <ErrorNotice message="No local session token. Open the URL printed by rollout investigate." />
        )}
        <ErrorNotice
          message={error}
          retry={() => setRevision((value) => value + 1)}
        />
        {initialLoading && (
          <p className="loading" role="status">
            Connecting to local investigation…
          </p>
        )}
        <div
          id="investigation-panel"
          role="tabpanel"
          tabIndex={-1}
          aria-labelledby={`tab-${tab.replaceAll(" ", "-")}`}
        >
          <div hidden={tab !== "Saved evidence"}>
            <EvidenceView value={saved} onOpen={() => setPicker("evidence")} />
          </div>
          {selected?.status === "ready" ? (
            <>
              <div hidden={tab !== "Events"}>
                <EventsView
                  key={selected.id}
                  run={selected}
                  onCitation={openCitation("Events")}
                  onCompare={(windows) => {
                    setSeed(windows);
                    setTab("Compare");
                  }}
                />
              </div>
              <div hidden={tab !== "Compare"}>
                <CompareView
                  key={selected.id}
                  run={selected}
                  runs={runs}
                  seed={seed}
                  onCitation={openCitation("Compare")}
                />
              </div>
            </>
          ) : tab === "Saved evidence" ? null : selected?.status === "failed" ? (
            <div className="blank">
              <h1 className="headline">Run could not be loaded</h1>
              <ErrorNotice message={selected.error ?? "Analysis failed."} />
              <p className="link-row">
                <button
                  type="button"
                  className="link"
                  onClick={() => setPicker("run")}
                >
                  Choose another input
                </button>
              </p>
            </div>
          ) : selected ? (
            <div className="blank" role="status">
              <h1 className="headline">
                {STAGES[selected.status] ?? selected.status}
              </h1>
              <p className="dek">
                The local worker is processing <code>{selected.run}</code>. This
                view updates automatically.
              </p>
              <div className="progress" aria-hidden="true">
                <i />
              </div>
            </div>
          ) : !initialLoading && session ? (
            <div className="blank">
              <h1 className="headline">Load a rollout run to begin</h1>
              <p className="dek">
                Browse ranked events, follow feature changes and read exact
                source responses. Analysis stays on this machine, and files stay
                inside <code>{session.root_label}</code>.
              </p>
              <p className="link-row">
                <button
                  type="button"
                  className="pill"
                  onClick={() => setPicker("run")}
                >
                  Choose local input
                </button>
              </p>
              <p className="section-note">
                TRL parquet, or verl JSONL files and directories.
              </p>
            </div>
          ) : null}
        </div>
      </main>
      {picker && (
        <FilePicker
          mode={picker}
          runId={selected?.status === "ready" ? selected.id : undefined}
          onClose={() => setPicker(null)}
          onRun={loaded}
          onEvidence={setSaved}
        />
      )}
      {citation && (
        <ResponseDrawer
          value={citation.value}
          context={citation.context}
          onClose={() => setCitation(null)}
        />
      )}
    </>
  );
}
