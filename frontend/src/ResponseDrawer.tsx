import { useEffect, useMemo, useRef, useState } from "react";
import {
  ErrorNotice,
  Fields,
  known,
  label,
  Modal,
  row,
  signed,
  text,
} from "./common";
import type { Citation, ReaderContext, Row } from "./types";

const SHADING_LIMIT = 4_000_000;

function stated(value: unknown, format: (value: unknown) => string = text) {
  return known(value) ? format(value) : "unavailable";
}
function differingLines(a: string[], b: string[]) {
  const width = b.length + 1;
  const table = new Uint32Array((a.length + 1) * width);
  for (let i = a.length - 1; i >= 0; i--)
    for (let j = b.length - 1; j >= 0; j--)
      table[i * width + j] =
        a[i] === b[j]
          ? table[(i + 1) * width + j + 1] + 1
          : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
  const inA = a.map(() => true),
    inB = b.map(() => true);
  let i = 0,
    j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      inA[i++] = false;
      inB[j++] = false;
    } else if (table[(i + 1) * width + j] >= table[i * width + j + 1]) i++;
    else j++;
  }
  return [inA, inB];
}
// reward_present and reward_absent are group means at this prompt and step, and
// difference is their difference. citation.reward is the one cited rollout.
function dek(citation: Row, comparison: Row) {
  const promptId = citation.prompt_id;
  const counts = [
      comparison.n_rollouts,
      comparison.n_present,
      comparison.n_absent,
    ],
    rewards = [
      comparison.reward_present,
      comparison.reward_absent,
      comparison.difference,
    ];
  const parts: string[] = [];
  if (counts.some(known))
    parts.push(
      `Prompt ${text(promptId)} has ${stated(counts[0])} ${String(counts[0]) === "1" ? "rollout" : "rollouts"} at this step: ${stated(counts[1])} with the feature and ${stated(counts[2])} without.`,
    );
  if (rewards.every(known))
    parts.push(
      `Present mean reward ${text(rewards[0])} against an absent mean of ${text(rewards[1])}, a difference of ${signed(rewards[2])}. The difference is descriptive.`,
    );
  else if (rewards.some(known))
    parts.push(
      `Present mean reward ${stated(rewards[0])}, absent mean ${stated(rewards[1])}, difference ${stated(rewards[2], signed)}.`,
    );
  if (typeof citation.selection_reason === "string")
    parts.push(
      `Selected for ${label(citation.selection_reason)}${rewards.some(known) ? "; the reward difference is conditioned on that choice" : ""}.`,
    );
  return parts.join(" ");
}
function Identity({ item }: { item: Row }) {
  return (
    <span className="reader-identity">
      <code>{text(item.sample_id)}</code>
      {typeof item.side === "string" && (
        <>
          , run <code>{text(item.run)}</code>, side {item.side.toUpperCase()}
        </>
      )}
      , reward {stated(item.reward)}
    </span>
  );
}
function Literal({
  response,
  differs,
}: {
  response: string;
  differs?: boolean[];
}) {
  const lines = response.split("\n");
  return (
    <>
      <pre className="response">
        {lines.map((line, index) => (
          <span
            key={index}
            className={differs?.[index] ? "reader-line differs" : "reader-line"}
          >
            {index < lines.length - 1 ? `${line}\n` : line}
          </span>
        ))}
      </pre>
      {response === "" && (
        <p className="quiet-note reader-aside">This response is empty.</p>
      )}
    </>
  );
}
export function ResponseDrawer({
  value,
  onClose,
  context,
}: {
  value: Citation;
  onClose: () => void;
  context: ReaderContext;
}) {
  const [shown, setShown] = useState(value),
    [busy, setBusy] = useState(false),
    [error, setError] = useState<string | null>(null),
    [shade, setShade] = useState(true);
  const root = useRef<HTMLDivElement>(null),
    generation = useRef(0),
    pressed = useRef<HTMLButtonElement | null>(null);
  useEffect(() => {
    setShown(value);
    setBusy(false);
    setError(null);
    return () => {
      generation.current++;
    };
  }, [value]);
  // Disabling a focused button drops focus out of the dialog's focus trap.
  useEffect(() => {
    const button = pressed.current;
    if (busy || !button) return;
    pressed.current = null;
    (button.disabled
      ? button.closest<HTMLElement>('[role="dialog"]')
      : button
    )?.focus();
  }, [busy]);
  const citation = shown.citation,
    comparison = row(citation.comparison),
    counterpart = row(comparison.counterpart);
  const compared = Object.keys(comparison).length > 0;
  const response = String(citation.response);
  const matched =
    typeof counterpart.response === "string" ? counterpart.response : null;
  const differs = useMemo(() => {
    if (matched === null) return null;
    const a = response.split("\n"),
      b = matched.split("\n");
    return a.length * b.length > SHADING_LIMIT ? null : differingLines(a, b);
  }, [response, matched]);
  const tooLong = matched !== null && !differs;
  const index = context.sampleIds.indexOf(String(citation.sample_id));
  const step = text(citation.step);
  const feature = <code>{context.feature}</code>;
  const summary = dek(citation, comparison);
  async function move(sampleId: string, button: HTMLButtonElement) {
    const requestId = ++generation.current;
    pressed.current = button;
    setBusy(true);
    setError(null);
    try {
      const next = await context.load(sampleId);
      if (requestId !== generation.current) return;
      setShown(next);
      const scroller = root.current?.closest(".scrim");
      if (scroller) scroller.scrollTop = 0;
    } catch (error) {
      if (requestId === generation.current)
        setError(error instanceof Error ? error.message : "Inspection failed.");
    } finally {
      if (requestId === generation.current) setBusy(false);
    }
  }
  return (
    <Modal title="Original response" onClose={onClose}>
      <div className="reader" ref={root}>
        <div className="kicker reader-top">
          <button
            type="button"
            className="link"
            aria-label={`Close original response, back to ${context.back}`}
            onClick={onClose}
          >
            <span aria-hidden="true">←</span> {context.back}
          </button>
          {index >= 0 && context.sampleIds.length > 1 && (
            <span className="reader-paging">
              <span role="status">
                {busy
                  ? "Verifying…"
                  : `Response ${index + 1} of ${context.sampleIds.length}`}
              </span>
              <button
                type="button"
                className="link"
                disabled={busy || index === 0}
                onClick={(event) =>
                  move(context.sampleIds[index - 1], event.currentTarget)
                }
              >
                Previous
              </button>
              <button
                type="button"
                className="link"
                disabled={busy || index === context.sampleIds.length - 1}
                onClick={(event) =>
                  move(context.sampleIds[index + 1], event.currentTarget)
                }
              >
                Next
              </button>
            </span>
          )}
        </div>
        <ErrorNotice message={error} />
        <h1 className="headline">
          {matched !== null ? (
            <>
              The same prompt at step {step}, with and without {feature}
            </>
          ) : compared ? (
            <>
              The same prompt at step {step}, with {feature} only
            </>
          ) : (
            <>
              The response at step {step}, with {feature}
            </>
          )}
        </h1>
        {summary && <p className="dek">{summary}</p>}
        <div className={compared ? "reader-columns" : "reader-columns single"}>
          <section aria-label="Response with the feature">
            <div className="reader-column-heading">
              <h2>With {feature}</h2>
              <Identity item={citation} />
            </div>
            <Literal
              response={response}
              differs={shade ? differs?.[0] : undefined}
            />
          </section>
          {compared && (
            <section aria-label="Matched response without the feature">
              <div className="reader-column-heading">
                <h2>Without</h2>
                {matched !== null && <Identity item={counterpart} />}
              </div>
              {matched !== null ? (
                <Literal
                  response={matched}
                  differs={shade ? differs?.[1] : undefined}
                />
              ) : (
                <p className="unavailable reader-aside">
                  No matched response is available.{" "}
                  {typeof comparison.reason === "string"
                    ? `Reason: ${label(comparison.reason)}.`
                    : "Reason unavailable."}{" "}
                  This is a missing comparison, not a negative result.
                </p>
              )}
            </section>
          )}
        </div>
        <div className="reader-provenance">
          <div className="reader-provenance-line">
            <p>
              Source verified at inspection. Shown as literal text, never
              rendered or executed.
              {tooLong &&
                " These responses are too long to compare line by line, so differing lines are not shaded."}
            </p>
            {differs && (
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={shade}
                  onChange={(event) => setShade(event.target.checked)}
                />
                Shade lines that differ
              </label>
            )}
          </div>
          <details className="more">
            <summary>Hashes, recorded fields and verified files</summary>
            <dl className="facts">
              <div>
                <dt>Response SHA-256</dt>
                <dd>
                  <code>{shown.response_sha256}</code>
                </dd>
              </div>
              <div>
                <dt>Canonical source SHA-256</dt>
                <dd>
                  <code>{shown.source.canonical_sha256}</code>
                </dd>
              </div>
            </dl>
            <h3>This response</h3>
            <Fields
              data={citation}
              fields={[
                ...("side" in citation ? ["side"] : []),
                "run",
                "sample_id",
                "step",
                "prompt_id",
                "feature",
                "feature_value",
                "reward",
                "selection_reason",
              ]}
            />
            {compared && (
              <>
                <h3>Matched comparison</h3>
                <p className="section-note">
                  Same exact prompt and step. Reward differences are
                  descriptive.
                </p>
                <Fields
                  data={{
                    ...comparison,
                    reason:
                      comparison.status === "available" &&
                      comparison.reason === null
                        ? "Not applicable"
                        : comparison.reason,
                  }}
                  fields={[
                    "status",
                    "reason",
                    "n_rollouts",
                    "n_present",
                    "n_absent",
                    "reward_present",
                    "reward_absent",
                    "difference",
                  ]}
                />
              </>
            )}
            {matched !== null && (
              <>
                <h3>Matched response</h3>
                <Fields
                  data={counterpart}
                  fields={[
                    "run",
                    "sample_id",
                    "step",
                    "prompt_id",
                    "reward",
                    "feature_value",
                  ]}
                />
              </>
            )}
            <h3>Verified source files</h3>
            <ul className="sources">
              {shown.source.files.map((file) => (
                <li key={file.path}>
                  <code>{file.path}</code>
                  <code>{file.sha256}</code>
                  <span>{String(file.size_bytes)} bytes</span>
                </li>
              ))}
            </ul>
          </details>
        </div>
      </div>
    </Modal>
  );
}
