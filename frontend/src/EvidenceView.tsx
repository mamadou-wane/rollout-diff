import { Badge, DataTable, Fields, label, text } from "./common";
import type { Evidence } from "./types";
function SavedReportSummary({ value }: { value: Evidence }) {
  const summary = value.summary;
  const mix = summary.prompt_mix;
  return (
    <section aria-label="Saved report summary">
      <h3>Saved report summary</h3>
      <p className="note">
        Load the original rollout dump to view feature curves or inspect source
        responses. These saved summaries do not provide verified original
        responses.
      </p>
      {value.kind === "analysis" ? (
        <>
          {(summary.runs ?? []).map((run) => (
            <article className="condition" key={run.run}>
              <div className="section-heading">
                <h4>{run.run}</h4>
                <Badge>{run.status}</Badge>
              </div>
              <Fields
                data={run.counts}
                fields={[
                  "rollouts",
                  "observed_steps",
                  "prompts",
                  "series",
                  "screened_series",
                  "sequential_alerts",
                  "unscorable_series",
                  "events",
                  "behavioral_events",
                  "contextual_events",
                  "other_events",
                ]}
              />
            </article>
          ))}
          {!summary.runs?.length && (
            <p className="muted">No saved run summaries available.</p>
          )}
        </>
      ) : (
        <>
          <Fields data={summary} fields={["mode", "feature_count"]} />
          <DataTable
            caption="Saved comparison windows"
            columns={[
              "side",
              "run",
              "window",
              "observed_steps",
              "n_rollouts",
              "n_prompts",
              "n_sampling_units",
            ]}
            data={(summary.sides ?? []).map((side) => ({
              side: side.side,
              run: side.run,
              window: side.window.map(String).join("–"),
              observed_steps: side.observed_steps.length,
              n_rollouts: side.n_rollouts,
              n_prompts: side.n_prompts,
              n_sampling_units: side.n_sampling_units,
            }))}
          />
          <h4>Saved prompt comparability</h4>
          {mix ? (
            <>
              <Badge warn={mix.status === "no_shared_prompts"}>
                {label(mix.status)}
              </Badge>
              <Fields
                data={mix}
                fields={["shared_prompts", "only_a_prompts", "only_b_prompts"]}
              />
              <p className="note">
                Only exact shared prompts support the paired comparison.
                Different prompt composition can contribute to a difference
                between windows.
              </p>
              <DataTable
                caption="Saved prompt descriptor shifts"
                data={mix.prompt_shift}
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
              />
            </>
          ) : (
            <p className="muted">Saved prompt comparability is unavailable.</p>
          )}
        </>
      )}
    </section>
  );
}
export function EvidenceView({
  value,
  onOpen,
}: {
  value: Evidence | null;
  onOpen: () => void;
}) {
  return (
    <section className="saved-view">
      <div className="section-heading">
        <div>
          <h2>Saved evidence</h2>
          <p className="muted">
            Read a saved replay, intervention, case or report.
          </p>
        </div>
        <button className="primary" onClick={onOpen}>
          Open saved evidence
        </button>
      </div>
      <p className="notice">
        Read only. Saved claims are historical evidence. No candidate execution,
        approval, intervention or replay is available here.
      </p>
      {!value ? (
        <div className="empty">
          <h3>Choose a saved evidence file</h3>
          <p>
            Source identity, response linkage and available task validation
            appear separately.
          </p>
        </div>
      ) : (
        <>
          <div className="detail-heading">
            <h3>{value.path}</h3>
            <Badge>{label(value.kind)}</Badge>
          </div>
          <p>{value.verification}</p>
          <div className="hash">
            <span>Saved file SHA-256</span>
            <code>{value.sha256}</code>
          </div>
          <section className="source-link">
            <h3>Source linkage</h3>
            <Badge warn={value.source_link.status !== "matched"}>
              {label(value.source_link.status)}
            </Badge>
            <p>{value.source_link.reason}</p>
            {value.identity && <Fields data={value.identity} />}
          </section>
          {value.kind === "analysis" || value.kind === "comparison" ? (
            <SavedReportSummary value={value} />
          ) : (
            <Fields data={value.summary} />
          )}

          <p className="notice warning">
            <strong>Mechanism unresolved.</strong> The saved evidence does not
            establish a causal mechanism.
          </p>
          {value.hypothesis && (
            <section>
              <h3>Recorded hypothesis</h3>
              <p className="literal">{value.hypothesis}</p>
            </section>
          )}
          {value.conditions.length > 0 && (
            <section>
              <h3>Condition evidence</h3>
              {value.conditions.map((condition, index) => (
                <article className="condition" key={index}>
                  <h4>{text(condition.name)}</h4>
                  <div className="evidence-grid">
                    <section>
                      <h4>Historical claim</h4>
                      <p className="note">
                        Recorded claim, untrusted as task validation.
                      </p>
                      <Fields
                        data={condition}
                        fields={[
                          "historical_status",
                          "historical_passed",
                          "historical_evaluated",
                          "historical_reason",
                        ]}
                      />
                    </section>
                    <section>
                      <h4>Saved protected task result</h4>
                      <Badge warn={condition.task_determinate !== true}>
                        {condition.task_determinate === true
                          ? "Determinate"
                          : "Unknown / unavailable"}
                      </Badge>
                      <Fields
                        data={{
                          ...condition,
                          task_reason:
                            condition.task_determinate === true &&
                            condition.task_reason === null
                              ? "Not applicable"
                              : condition.task_reason,
                        }}
                        fields={[
                          "task_status",
                          "task_determinate",
                          "task_passed",
                          "task_failed",
                          "task_completed",
                          "task_applicable",
                          "task_reason",
                        ]}
                      />
                    </section>
                  </div>
                  <Fields
                    data={condition}
                    fields={[
                      "discrepancy",
                      "cleanup_verified",
                      "answer_stream_sha256",
                    ]}
                  />
                </article>
              ))}
            </section>
          )}
          {value.predictions.length > 0 && (
            <DataTable
              caption="Recorded predictions"
              columns={["field", "expected", "observed", "status"]}
              data={value.predictions}
            />
          )}
          <section>
            <h3>Protected source and proposed edit</h3>
            <p className="note">
              Edits and source diffs are displayed as literal evidence.
              Protected task validation remains separate from historical reward
              claims.
            </p>
            {value.edit ? (
              <>
                <Fields data={{ occurrence: value.edit.occurrence }} />
                <h4>Original text</h4>
                <pre className="response">{value.edit.old}</pre>
                <h4>Proposed text</h4>
                <pre className="response">{value.edit.new}</pre>
              </>
            ) : (
              <p className="muted">No recorded literal edit.</p>
            )}
            {value.source_diff && (
              <details>
                <summary>Verified source diff</summary>
                <pre className="response">{value.source_diff}</pre>
              </details>
            )}
          </section>
        </>
      )}
    </section>
  );
}
