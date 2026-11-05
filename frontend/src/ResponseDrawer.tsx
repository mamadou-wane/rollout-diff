import { Badge, Fields, Modal, row, text } from "./common";
import type { Citation } from "./types";
export function ResponseDrawer({
  value,
  onClose,
}: {
  value: Citation;
  onClose: () => void;
}) {
  const citation = value.citation,
    comparison = row(citation.comparison),
    counterpart = row(comparison.counterpart);
  return (
    <Modal title="Original response" onClose={onClose} drawer>
      <Badge>Source verified at inspection</Badge>
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
      <div className="hash">
        <span>Response SHA-256</span>
        <code>{value.response_sha256}</code>
      </div>
      <div className="hash">
        <span>Canonical source SHA-256</span>
        <code>{value.source.canonical_sha256}</code>
      </div>
      <h3>Literal response text</h3>
      <pre className="response">
        {typeof citation.response === "string"
          ? citation.response
          : "Response unavailable."}
      </pre>
      {Object.keys(comparison).length > 0 && (
        <section>
          <h3>Matched absent completion</h3>
          <p className="note">
            Same exact prompt and step. Reward differences are descriptive.
          </p>
          <Fields
            data={{
              ...comparison,
              reason:
                comparison.status === "available" && comparison.reason === null
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
          {typeof counterpart.response === "string" ? (
            <>
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
              <pre className="response">{counterpart.response}</pre>
            </>
          ) : (
            <p className="muted">{text(comparison.reason)}</p>
          )}
        </section>
      )}
      <details>
        <summary>Verified source files</summary>
        <ul className="sources">
          {value.source.files.map((file) => (
            <li key={file.path}>
              <code>{file.path}</code>
              <code>{file.sha256}</code>
              <span>{String(file.size_bytes)} bytes</span>
            </li>
          ))}
        </ul>
      </details>
    </Modal>
  );
}
