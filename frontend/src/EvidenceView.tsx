import type { ReactNode } from "react";
import {
  DataTable,
  Fields,
  known,
  label,
  PROMPT_SHIFT_COLUMNS,
  runCounts,
  text,
} from "./common";
import type { Evidence, Row, Step } from "./types";

interface Reading {
  value: string;
  missing: boolean;
}
function isCount(value: unknown) {
  return typeof value === "number" || typeof value === "string";
}
function isReport(value: Evidence) {
  return value.kind === "analysis" || value.kind === "comparison";
}
function withReason(summary: string, reason: unknown) {
  return typeof reason === "string" && reason
    ? `${summary}: ${reason}.`
    : summary;
}
function statusName(status: unknown) {
  const name = label(
    typeof status === "string" && status ? status : "unavailable",
  );
  return name[0].toUpperCase() + name.slice(1);
}
function historicalClaim(condition: Row): Reading {
  const {
    historical_passed: passed,
    historical_evaluated: evaluated,
    historical_status: status,
    historical_reason: reason,
  } = condition;
  if (isCount(passed) && isCount(evaluated))
    return {
      value: withReason(`${text(passed)} of ${text(evaluated)} passed`, reason),
      missing: false,
    };
  return {
    value: withReason(statusName(status), reason),
    missing: !status || status === "unavailable" || status === "unresolved",
  };
}
function taskResult(condition: Row): Reading {
  const {
    task_passed: passed,
    task_failed: failed,
    task_applicable: applicable,
    task_reason: reason,
  } = condition;
  if (
    condition.task_determinate === true &&
    isCount(passed) &&
    isCount(failed) &&
    isCount(applicable)
  )
    return {
      value: withReason(
        `${text(passed)} passed, ${text(failed)} failed, of ${text(applicable)} applicable`,
        reason,
      ),
      missing: false,
    };
  return {
    value: withReason(statusName(condition.task_status), reason),
    missing: true,
  };
}
function ReadingText({ reading }: { reading: Reading }) {
  return (
    <span className={reading.missing ? "unavailable" : undefined}>
      {reading.value}
    </span>
  );
}
function Value({ value, literal }: { value: unknown; literal?: boolean }) {
  if (!known(value)) return <span className="unavailable">Unavailable</span>;
  const shown = Array.isArray(value) ? JSON.stringify(value) : text(value);
  return literal ? <code>{shown}</code> : <>{shown}</>;
}
function ConditionRow({
  title,
  conditions,
  cell,
}: {
  title: string;
  conditions: Row[];
  cell: (condition: Row) => ReactNode;
}) {
  return (
    <tr>
      <th scope="row">{title}</th>
      {conditions.map((condition, index) => (
        <td key={index}>{cell(condition)}</td>
      ))}
    </tr>
  );
}
function ConditionTable({ conditions }: { conditions: Row[] }) {
  return (
    <div className="table-scroll" tabIndex={0}>
      <table className="saved-conditions" aria-label="Condition evidence">
        <thead>
          <tr>
            <th scope="col">Field</th>
            {conditions.map((condition, index) => (
              <th scope="col" key={index}>
                {text(condition.name)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <ConditionRow
            title="Historical claim"
            conditions={conditions}
            cell={(condition) => (
              <ReadingText reading={historicalClaim(condition)} />
            )}
          />
          <ConditionRow
            title="Saved protected task result"
            conditions={conditions}
            cell={(condition) => (
              <>
                <ReadingText reading={taskResult(condition)} />
                <span className="saved-aside">
                  {condition.task_determinate === true
                    ? "Determinate"
                    : "Unknown / unavailable"}
                </span>
              </>
            )}
          />
          <ConditionRow
            title="Discrepancy"
            conditions={conditions}
            cell={(condition) => <Value value={condition.discrepancy} />}
          />
          <ConditionRow
            title="Cleanup verified"
            conditions={conditions}
            cell={(condition) => <Value value={condition.cleanup_verified} />}
          />
          <ConditionRow
            title="Answer stream SHA-256"
            conditions={conditions}
            cell={(condition) => (
              <Value value={condition.answer_stream_sha256} literal />
            )}
          />
        </tbody>
      </table>
    </div>
  );
}
function RecordedFields({ value }: { value: Evidence }) {
  return (
    <details className="more">
      <summary>All recorded fields</summary>
      {value.conditions.map((condition, index) => (
        <div className="saved-recorded" key={index}>
          <h3>{text(condition.name)}</h3>
          <Fields
            data={condition}
            fields={[
              "historical_status",
              "historical_passed",
              "historical_evaluated",
              "historical_reason",
            ]}
          />
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
          <Fields
            data={condition}
            fields={[
              "discrepancy",
              "cleanup_verified",
              "answer_stream_sha256",
            ]}
          />
        </div>
      ))}
      {!isReport(value) && <Fields data={value.summary} />}
    </details>
  );
}
function RunCounts({ counts }: { counts: Record<string, Step | null> }) {
  return (
    <p className="section-note">
      {runCounts(counts)} {text(counts.screened_series)} of{" "}
      {text(counts.series)} series were screened;{" "}
      {text(counts.unscorable_series)} were unscorable.
    </p>
  );
}
function SavedReportSummary({ value }: { value: Evidence }) {
  const summary = value.summary;
  const mix = summary.prompt_mix;
  return (
    <section className="section" aria-label="Saved report summary">
      <h2>Saved report summary</h2>
      <p className="section-note">
        Load the original rollout dump to view feature curves or inspect source
        responses. These saved summaries do not provide verified original
        responses.
      </p>
      {value.kind === "analysis" ? (
        <>
          {(summary.runs ?? []).map((run) => (
            <div className="saved-run" key={run.run}>
              <div className="section-heading">
                <h3>{run.run}</h3>
                <span className="section-note">{label(run.status)}</span>
              </div>
              <RunCounts counts={run.counts} />
            </div>
          ))}
          {!summary.runs?.length && (
            <p className="unavailable">No saved run summaries available.</p>
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
          <div className="saved-run">
            <h3>Saved prompt comparability</h3>
            {mix ? (
              <>
                <p
                  className={
                    mix.status === "no_shared_prompts"
                      ? "section-note unavailable"
                      : "section-note"
                  }
                >
                  {label(mix.status)}
                </p>
                <Fields
                  data={mix}
                  fields={[
                    "shared_prompts",
                    "only_a_prompts",
                    "only_b_prompts",
                  ]}
                />
                <p className="section-note">
                  Only exact shared prompts support the paired comparison.
                  Different prompt composition can contribute to a difference
                  between windows.
                </p>
                <DataTable
                  caption="Saved prompt descriptor shifts"
                  data={mix.prompt_shift}
                  columns={PROMPT_SHIFT_COLUMNS}
                />
              </>
            ) : (
              <p className="unavailable">
                Saved prompt comparability is unavailable.
              </p>
            )}
          </div>
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
  if (!value)
    return (
      <div className="saved">
        <h1 className="headline">Saved evidence</h1>
        <p className="dek">
          Read a saved replay, intervention, case or report. Read only. Saved
          claims are historical evidence. No candidate execution, approval,
          intervention or replay is available here.
        </p>
        <section className="section">
          <h2>Choose a saved evidence file</h2>
          <p className="section-note">
            Source identity, response linkage and available task validation
            appear separately.
          </p>
          <p className="link-row">
            <button type="button" className="pill" onClick={onOpen}>
              Open saved evidence
            </button>
          </p>
        </section>
      </div>
    );
  const identity = value.identity;
  const conditionHeading =
    value.conditions.length > 1 ? "Control and variant" : "Recorded condition";
  return (
    <div className="saved">
      <p className="kicker">
        <span>
          <code>{value.path}</code>, read only, {label(value.kind)}
        </span>
        <button type="button" className="link" onClick={onOpen}>
          Open saved evidence
        </button>
      </p>
      <h1 className="headline">
        A recorded {label(value.kind)}.{" "}
        <span>Mechanism unresolved.</span>
      </h1>
      <p className="dek">
        <span>{value.verification}</span> The saved evidence does not establish
        a causal mechanism.
      </p>
      <section className="section" aria-label="Source">
        <h2>Source</h2>
        <p>{value.source_link.reason}</p>
        <p className="quiet-note">
          Status: {label(value.source_link.status)}.{" "}
          {identity ? (
            <>
              Run <code>{text(identity.run)}</code>, sample{" "}
              <code>{text(identity.sample_id)}</code>, step{" "}
              {text(identity.step)}, prompt{" "}
              <code>{text(identity.prompt_id)}</code>.
            </>
          ) : (
            <span className="unavailable">Canonical identity unavailable.</span>
          )}{" "}
          Saved file SHA-256 <code>{value.sha256}</code>.
        </p>
      </section>
      {isReport(value) && <SavedReportSummary value={value} />}
      {value.hypothesis && (
        <section className="section" aria-label="Recorded hypothesis">
          <h2>Recorded hypothesis</h2>
          <p className="saved-hypothesis">{value.hypothesis}</p>
          <p className="quiet-note">Saved text, shown literally.</p>
        </section>
      )}
      {value.conditions.length > 0 ? (
        <section className="section" aria-label={conditionHeading}>
          <h2>{conditionHeading}</h2>
          <p className="section-note">
            Historical claims are recorded and untrusted as task validation.
            Protected task results stay separate from them.
          </p>
          <ConditionTable conditions={value.conditions} />
          <p className="quiet-note">
            Discrepancy means the historical claim passed while the determinate
            protected task failed. It is recorded only when the historical run
            reports verified configuration and cleanup. Cleanup and the answer
            stream describe the protected task run.
          </p>
          <RecordedFields value={value} />
        </section>
      ) : (
        !isReport(value) && <RecordedFields value={value} />
      )}
      {value.predictions.length > 0 && (
        <section className="section" aria-label="Predictions">
          <h2>Predictions</h2>
          <div className="table-scroll" tabIndex={0}>
            <table aria-label="Recorded predictions">
              <thead>
                <tr>
                  <th scope="col">Field</th>
                  <th scope="col">Expected</th>
                  <th scope="col">Observed</th>
                  <th scope="col">Status</th>
                </tr>
              </thead>
              <tbody>
                {value.predictions.map((prediction, index) => (
                  <tr key={index}>
                    <th scope="row">
                      <code>{text(prediction.field)}</code>
                    </th>
                    <td>
                      <Value value={prediction.expected} />
                    </td>
                    <td>
                      <Value value={prediction.observed} />
                    </td>
                    <td
                      className={
                        prediction.status === "unavailable"
                          ? "unavailable"
                          : undefined
                      }
                    >
                      {label(text(prediction.status))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      <section className="section" aria-label="Proposed edit">
        <h2>Proposed edit</h2>
        <p className="section-note">
          Literal evidence. Nothing can be executed, approved or replayed from
          this page.
        </p>
        {value.edit ? (
          <>
            <p className="quiet-note">
              Applies to occurrence {text(value.edit.occurrence)} of the
              original text.
            </p>
            <div className="saved-edit">
              <section aria-label="Original text">
                <h3>Original</h3>
                <pre className="response">{value.edit.old}</pre>
              </section>
              <section aria-label="Proposed text">
                <h3>Proposed</h3>
                {value.edit.new ? (
                  <pre className="response">{value.edit.new}</pre>
                ) : (
                  <p className="quiet-note">Empty text.</p>
                )}
              </section>
            </div>
          </>
        ) : (
          <p className="unavailable">No recorded literal edit.</p>
        )}
        {value.source_diff && (
          <details className="more">
            <summary>Verified source diff</summary>
            <pre className="response">{value.source_diff}</pre>
          </details>
        )}
      </section>
    </div>
  );
}
