import { useEffect, useRef, useState, type FormEvent } from "react";
import { downloadReport, post, request } from "./api";
import {
  Badge,
  DataTable,
  ErrorNotice,
  Fields,
  label,
  rows,
  text,
} from "./common";
import type { Citation, Comparison, Estimate, Run } from "./types";
function EstimateView({ title, value }: { title: string; value: Estimate }) {
  return (
    <section className="estimate">
      <h4>{title}</h4>
      <strong>{text(value.mean ?? value.estimate)}</strong>
      <span>{label(value.status)}</span>
      <span>SE {text(value.se)}</span>
      {value.ci_low !== null && value.ci_low !== undefined && (
        <span>
          95% normal interval [{text(value.ci_low)}, {text(value.ci_high)}]
        </span>
      )}
      <Fields
        data={value}
        fields={["n_prompts", "n_sampling_units", "n_rollouts"]}
      />
    </section>
  );
}
export function CompareView({
  run,
  runs,
  onCitation,
}: {
  run: Run;
  runs: Run[];
  onCitation: (value: Citation) => void;
}) {
  const [other, setOther] = useState(""),
    [aStart, setAStart] = useState(""),
    [aEnd, setAEnd] = useState(""),
    [bStart, setBStart] = useState(""),
    [bEnd, setBEnd] = useState("");
  const [comparison, setComparison] = useState<Comparison | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState<string | null>(null),
    [query, setQuery] = useState(""),
    [selected, setSelected] = useState("");
  const [pollRevision, setPollRevision] = useState(0);
  const generation = useRef(0);
  useEffect(
    () => () => {
      generation.current++;
    },
    [],
  );
  useEffect(() => {
    if (!comparison || !["queued", "comparing"].includes(comparison.status))
      return;
    const id = comparison.id,
      version = generation.current;
    const controller = new AbortController();
    const timer = setTimeout(
      () =>
        request<Comparison>(`/api/comparisons/${id}`, {
          signal: controller.signal,
        })
          .then((value) => {
            if (version === generation.current) setComparison(value);
          })
          .catch((error) => {
            if (!controller.signal.aborted && version === generation.current)
              setError(
                error instanceof Error
                  ? error.message
                  : "Comparison status unavailable.",
              );
          }),
      700,
    );
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [comparison, pollRevision]);
  async function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    if (
      ![aStart, aEnd, bStart, bEnd].every((value) => /^\d+$/.test(value)) ||
      BigInt(aStart) > BigInt(aEnd) ||
      BigInt(bStart) > BigInt(bEnd)
    ) {
      setError(
        "Enter nonnegative integer step ranges with each start at or before its end.",
      );
      return;
    }
    const version = ++generation.current;
    setBusy(true);
    setComparison(null);
    setSelected("");
    try {
      const value = await post<Comparison>("/api/comparisons", {
        run_a: run.id,
        run_b: other || null,
        a: [aStart, aEnd],
        b: [bStart, bEnd],
      });
      if (version === generation.current) setComparison(value);
    } catch (error) {
      if (version === generation.current)
        setError(error instanceof Error ? error.message : "Comparison failed.");
    } finally {
      if (version === generation.current) setBusy(false);
    }
  }
  async function inspect(side: "a" | "b", feature: string, sampleId: string) {
    if (!comparison) return;
    const version = ++generation.current;
    setBusy(true);
    setError(null);
    try {
      const value = await post<Citation>(
        `/api/comparisons/${comparison.id}/inspect`,
        { side, feature, sample_id: sampleId },
      );
      if (version === generation.current) onCitation(value);
    } catch (error) {
      if (version === generation.current)
        setError(error instanceof Error ? error.message : "Inspection failed.");
    } finally {
      if (version === generation.current) setBusy(false);
    }
  }
  async function exportFile(format: string) {
    if (!comparison) return;
    try {
      await downloadReport(
        `/api/comparisons/${comparison.id}/export?format=${format}`,
        `comparison.${format}`,
      );
    } catch (error) {
      setError(error instanceof Error ? error.message : "Export failed.");
    }
  }
  const report = comparison?.report,
    feature = report?.features.find((item) => item.feature === selected);
  const filtered =
    report?.features.filter((item) =>
      item.feature.toLowerCase().includes(query.toLowerCase()),
    ) ?? [];
  const pending =
    busy ||
    comparison?.status === "queued" ||
    comparison?.status === "comparing";
  return (
    <section className="compare-view">
      <div className="section-heading">
        <div>
          <h2>Compare observed windows</h2>
          <p className="muted">
            Prompt-balanced estimates and source-linked citations.
          </p>
        </div>
      </div>
      <form className="comparison-form" onSubmit={submit}>
        <fieldset>
          <legend>A · {run.run}</legend>
          <div className="range-inputs">
            <label>
              A start
              <input
                inputMode="numeric"
                pattern="[0-9]+"
                value={aStart}
                onChange={(event) => setAStart(event.target.value)}
                required
              />
            </label>
            <label>
              A end
              <input
                inputMode="numeric"
                pattern="[0-9]+"
                value={aEnd}
                onChange={(event) => setAEnd(event.target.value)}
                required
              />
            </label>
          </div>
        </fieldset>
        <fieldset>
          <legend>B · comparison window</legend>
          <label>
            Comparison run
            <select
              value={other}
              onChange={(event) => setOther(event.target.value)}
            >
              <option value="">Same run: {run.run}</option>
              {runs
                .filter((item) => item.status === "ready" && item.id !== run.id)
                .map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.run}
                  </option>
                ))}
            </select>
          </label>
          <div className="range-inputs">
            <label>
              B start
              <input
                inputMode="numeric"
                pattern="[0-9]+"
                value={bStart}
                onChange={(event) => setBStart(event.target.value)}
                required
              />
            </label>
            <label>
              B end
              <input
                inputMode="numeric"
                pattern="[0-9]+"
                value={bEnd}
                onChange={(event) => setBEnd(event.target.value)}
                required
              />
            </label>
          </div>
        </fieldset>
        <button className="primary" disabled={pending}>
          Compare windows
        </button>
      </form>
      <p className="note">
        Ranges are inclusive. Shared-prompt estimates use exact prompt
        identities. Missing uncertainty and zero shared prompts remain
        unavailable.
      </p>
      <ErrorNotice
        message={error}
        retry={
          comparison && ["queued", "comparing"].includes(comparison.status)
            ? () => {
                setError(null);
                setPollRevision((value) => value + 1);
              }
            : undefined
        }
      />
      <p className="note">
        Pointwise 95% intervals use a normal approximation and measured standard
        errors. Intervals are unadjusted across features. Available estimates
        remain visible when intervals are unavailable. A zero measured SE does
        not establish zero uncertainty.
      </p>
      <ErrorNotice message={comparison?.error ?? null} />
      {pending && (
        <p className="notice" role="status">
          {comparison?.status === "comparing"
            ? "Computing comparison…"
            : "Comparison queued…"}
        </p>
      )}
      {comparison?.status === "ready" && report && (
        <>
          <div className="section-heading">
            <h3>Comparison evidence</h3>
            <div className="button-group">
              <button onClick={() => exportFile("json")}>
                Export comparison JSON
              </button>
              <button onClick={() => exportFile("html")}>
                Export comparison HTML
              </button>
            </div>
          </div>
          <DataTable
            caption="Compared windows"
            data={report.sides.map((side) => ({
              ...side,
              window: Array.isArray(side.window) ? side.window.join("–") : null,
            }))}
            columns={[
              "side",
              "run",
              "window",
              "n_rollouts",
              "n_prompts",
              "n_sampling_units",
            ]}
          />
          <section className="prompt-mix">
            <h3>Prompt overlap</h3>
            <Badge warn={report.prompt_mix.status === "no_shared_prompts"}>
              {label(text(report.prompt_mix.status))}
            </Badge>
            <Fields
              data={report.prompt_mix}
              fields={["shared_prompts", "only_a_prompts", "only_b_prompts"]}
            />
            <details>
              <summary>Prompt descriptor shifts</summary>
              <DataTable
                data={rows(report.prompt_mix.prompt_shift)}
                columns={[
                  "descriptor",
                  "statistic",
                  "before",
                  "after",
                  "delta",
                  "distance",
                  "n_before",
                  "n_after",
                  "missing_before",
                  "missing_after",
                ]}
                caption="Comparison prompt shifts"
              />
            </details>
          </section>
          <label className="comparison-search">
            Search comparison features
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <div className="table-scroll">
            <table>
              <caption>Feature comparisons</caption>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>A mean</th>
                  <th>B mean</th>
                  <th>Difference</th>
                  <th>Uncertainty</th>
                  <th>Paired estimate</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((item) => (
                  <tr key={item.feature}>
                    <th>
                      <button
                        className="text-button"
                        onClick={() => setSelected(item.feature)}
                      >
                        {item.feature}
                      </button>
                      <span className="muted table-subtitle">
                        {item.tier} · {item.direction}
                      </span>
                    </th>
                    <td>{text(item.a.mean)}</td>
                    <td>{text(item.b.mean)}</td>
                    <td>{text(item.difference.estimate)}</td>
                    <td>{label(item.difference.status)}</td>
                    <td>
                      {text(item.paired.estimate)}
                      <span className="muted table-subtitle">
                        {label(item.paired.status)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {filtered.length === 0 && (
            <p className="empty">No comparison features match.</p>
          )}
          {feature && (
            <article className="comparison-detail">
              <h3>{feature.feature}</h3>
              <div className="estimates">
                <EstimateView title="A estimate" value={feature.a} />
                <EstimateView title="B estimate" value={feature.b} />
                <EstimateView title="Difference" value={feature.difference} />
                <EstimateView
                  title="Shared-prompt estimate"
                  value={feature.paired}
                />
              </div>
              <p className="note">
                B - A interval method: {text(feature.difference.method)}. Reward
                associations and feature contrasts are descriptive.
              </p>
              <div className="evidence-grid">
                {(["a", "b"] as const).map((side) => (
                  <section key={side}>
                    <h4>Side {side.toUpperCase()} citations</h4>
                    <p className="muted">
                      {label(feature.citations[side].status)}
                    </p>
                    {feature.citations[side].items.map((item) => (
                      <button
                        key={String(item.sample_id)}
                        disabled={busy}
                        onClick={() =>
                          inspect(side, feature.feature, String(item.sample_id))
                        }
                      >
                        Inspect {item.sample_id} · step {text(item.step)}
                      </button>
                    ))}
                  </section>
                ))}
              </div>
            </article>
          )}
        </>
      )}
    </section>
  );
}
