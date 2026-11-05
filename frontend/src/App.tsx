import { useEffect, useState } from "react";
import { initializeToken, request } from "./api";
import { Badge, ErrorNotice, Fields, text } from "./common";
import { CompareView } from "./CompareView";
import { EventsView } from "./EventsView";
import { EvidenceView } from "./EvidenceView";
import { FilePicker } from "./FilePicker";
import { ResponseDrawer } from "./ResponseDrawer";
import type { Citation, Evidence, Run } from "./types";

type Tab = "Events" | "Compare" | "Saved evidence";
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
    [citation, setCitation] = useState<Citation | null>(null),
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
  function loaded(run: Run) {
    setRuns((current) => [
      ...current.filter((item) => item.id !== run.id),
      run,
    ]);
    setRunId(run.id);
    setTab("Events");
    setCitation(null);
    setSaved(null);
  }
  async function unload() {
    if (!selected) return;
    try {
      await request(`/api/runs/${selected.id}`, { method: "DELETE" });
      setRuns((current) => current.filter((item) => item.id !== selected.id));
      setRunId(runs.find((item) => item.id !== selected.id)?.id ?? "");
      setCitation(null);
      setSaved(null);
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Unable to unload run.",
      );
    }
  }
  return (
    <>
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            rd
          </span>
          <div>
            <h1>Rollout Diff</h1>
            <p>Local investigation</p>
          </div>
        </div>
        <div className="header-actions">
          <label>
            Loaded run
            <select
              value={runId}
              onChange={(event) => {
                setRunId(event.target.value);
                setCitation(null);
                setSaved(null);
              }}
              aria-label="Loaded run"
            >
              <option value="">Select a loaded run</option>
              {runs.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.run} · {run.status}
                </option>
              ))}
            </select>
          </label>
          <button
            className="primary"
            onClick={() => setPicker("run")}
            disabled={!session}
          >
            Load run
          </button>
        </div>
      </header>
      <main>
        <div className="session-line">
          <span>
            Root <code>{session?.root_label ?? "Connecting…"}</code>
          </span>
          <span>Local analysis only</span>
        </div>
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
        {selected && (
          <section className="run-strip" aria-label="Run summary">
            <div>
              <strong>{selected.run}</strong>
              <Badge warn={selected.status === "failed"}>
                {selected.status}
              </Badge>
              {selected.elapsed_seconds !== null &&
                selected.elapsed_seconds !== undefined && (
                  <span className="muted">
                    {text(selected.elapsed_seconds)} s
                  </span>
                )}
            </div>
            {selected.counts && <Fields data={selected.counts} />}
            <div className="run-actions">
              {selected.source && (
                <details>
                  <summary>Source fingerprint</summary>
                  <code>{selected.source.path}</code>
                  <code className="fingerprint">
                    {selected.source.canonical_sha256}
                  </code>
                </details>
              )}
              {["ready", "failed"].includes(selected.status) && (
                <button onClick={unload}>Unload run</button>
              )}
            </div>
          </section>
        )}
        <div className="tabs" role="tablist" aria-label="Investigation views">
          {(["Events", "Compare", "Saved evidence"] as Tab[]).map((name) => (
            <button
              key={name}
              id={`tab-${name.replaceAll(" ", "-")}`}
              role="tab"
              aria-selected={tab === name}
              aria-controls="investigation-panel"
              tabIndex={tab === name ? 0 : -1}
              onKeyDown={(event) => {
                const names: Tab[] = ["Events", "Compare", "Saved evidence"];
                if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
                  const next =
                    names[
                      (names.indexOf(tab) +
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
        <div
          id="investigation-panel"
          role="tabpanel"
          aria-labelledby={`tab-${tab.replaceAll(" ", "-")}`}
        >
          {tab === "Saved evidence" ? (
            <EvidenceView value={saved} onOpen={() => setPicker("evidence")} />
          ) : selected?.status === "ready" ? (
            tab === "Events" ? (
              <EventsView
                key={selected.id}
                run={selected}
                onCitation={setCitation}
              />
            ) : (
              <CompareView
                key={selected.id}
                run={selected}
                runs={runs}
                onCitation={setCitation}
              />
            )
          ) : selected?.status === "failed" ? (
            <div className="empty">
              <h2>Run could not be loaded</h2>
              <ErrorNotice message={selected.error ?? "Analysis failed."} />
              <button onClick={() => setPicker("run")}>
                Choose another input
              </button>
            </div>
          ) : selected ? (
            <div className="empty" role="status">
              <div className="spinner" />
              <h2>
                {selected.status === "queued"
                  ? "Run queued"
                  : selected.status === "loading"
                    ? "Reading source files"
                    : "Analyzing rollout evidence"}
              </h2>
              <p>
                The local worker is processing {selected.run}. This view updates
                automatically.
              </p>
            </div>
          ) : !initialLoading && session ? (
            <div className="empty welcome">
              <h2>Load a rollout run to begin</h2>
              <p>
                Browse ranked events, follow feature changes and inspect exact
                source responses.
              </p>
              <button className="primary" onClick={() => setPicker("run")}>
                Choose local input
              </button>
              <p className="muted">
                TRL parquet or verl JSONL · files stay within the selected root
              </p>
            </div>
          ) : null}
        </div>
      </main>
      <footer>
        Rollout Diff · Evidence for investigation. Descriptive indicators do not
        establish mechanism.
      </footer>
      {picker && (
        <FilePicker
          mode={picker}
          runId={selected?.status === "ready" ? selected.id : undefined}
          onClose={() => setPicker(null)}
          onRun={loaded}
          onEvidence={setSaved}
        />
      )}{" "}
      {citation && (
        <ResponseDrawer value={citation} onClose={() => setCitation(null)} />
      )}
    </>
  );
}
