import { useEffect, useRef, useState } from "react";
import { downloadReport, post, useResource } from "./api";
import {
  Badge,
  DataTable,
  ErrorNotice,
  Fields,
  label,
  row,
  text,
} from "./common";
import { FeatureChart } from "./FeatureChart";
import type {
  Citation,
  Event,
  EventDetail,
  Member,
  Row,
  Run,
  Series,
} from "./types";

function MemberEvidence({ member, full }: { member: Member; full?: Row }) {
  const evidence = member.evidence;
  return (
    <>
      <div className="evidence-grid">
        {[
          ["Replication", "replication"],
          ["Persistence", "persistence"],
          ["Reward association", "reward_association"],
          ["Prompt breadth", "prompt_breadth"],
        ].map(([title, key]) => (
          <section key={key}>
            <h4>{title}</h4>
            <Fields data={row(evidence[key])} />
          </section>
        ))}
      </div>
      <section className="prompt-mix-evidence">
        <h4>Prompt mix and shared prompts</h4>
        <Fields
          data={full ?? {}}
          fields={[
            "promptmix_status",
            "prompt_shared_prompts",
            "prompt_shared_sampling_units",
            "prompt_shared_change",
            "prompt_shared_se",
            "prompt_length_delta",
            "prompt_length_distance",
            "prompt_before_start",
            "prompt_before_end",
            "prompt_after_start",
            "prompt_after_end",
          ]}
        />
      </section>
      <Fields
        data={evidence}
        fields={["status", "tier", "scale_source", "numerical_floor_dominant"]}
      />
      {full && (
        <details>
          <summary>Full recorded feature evidence</summary>
          <Fields data={full} />
        </details>
      )}
    </>
  );
}
export function EventsView({
  run,
  onCitation,
}: {
  run: Run;
  onCitation: (value: Citation) => void;
}) {
  const [query, setQuery] = useState(""),
    [kind, setKind] = useState("all"),
    [tier, setTier] = useState("all"),
    [floor, setFloor] = useState("all"),
    [offset, setOffset] = useState(0);
  const [rank, setRank] = useState<number | null>(null),
    [feature, setFeature] = useState(""),
    [featureQuery, setFeatureQuery] = useState("");
  const [inspectBusy, setInspectBusy] = useState<string | null>(null),
    [error, setError] = useState<string | null>(null),
    [absent, setAbsent] = useState(true);
  const generation = useRef(0);
  useEffect(
    () => () => {
      generation.current++;
    },
    [],
  );
  const params = new URLSearchParams({
    q: query,
    kind,
    tier,
    floor,
    offset: String(offset),
    limit: "50",
  });
  const events = useResource<{ total: number; events: Event[] }>(
    `/api/runs/${run.id}/events?${params}`,
  );
  const detail = useResource<EventDetail>(
    rank === null ? null : `/api/runs/${run.id}/events/${rank}`,
  );
  const features = useResource<{
    features: { feature: string; kind: string; tier: string }[];
  }>(
    `/api/runs/${run.id}/features?q=${encodeURIComponent(featureQuery)}&limit=100`,
  );
  const series = useResource<Series>(
    feature
      ? `/api/runs/${run.id}/series?feature=${encodeURIComponent(feature)}`
      : null,
  );
  function filter(set: (value: string) => void, value: string) {
    set(value);
    setOffset(0);
  }
  function select(event: Event) {
    generation.current++;
    setRank(event.rank);
    setFeature(event.representative_feature);
    setError(null);
    setInspectBusy(null);
  }
  async function inspect(member: Member, sampleId: string) {
    const requestId = ++generation.current;
    setInspectBusy(sampleId);
    setError(null);
    try {
      const value = await post<Citation>(`/api/runs/${run.id}/inspect`, {
        event_rank: rank,
        feature: member.feature,
        sample_id: sampleId,
        with_absent: absent,
      });
      if (requestId === generation.current) onCitation(value);
    } catch (error) {
      if (requestId === generation.current)
        setError(error instanceof Error ? error.message : "Inspection failed.");
    } finally {
      if (requestId === generation.current) setInspectBusy(null);
    }
  }
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
  const selectedMember = detail.data?.event.members.find(
    (member) => member.feature === feature,
  );
  const withAnchors = series.data
    ? {
        ...series.data,
        evidence: {
          ...series.data.evidence,
          onset_step:
            series.data.evidence?.onset_step ?? selectedMember?.onset_step,
          alert_step:
            series.data.evidence?.alert_step ?? selectedMember?.alert_step,
        },
      }
    : null;
  return (
    <>
      <div className="workspace-toolbar">
        <p className="muted">Ranked behavioral and contextual events</p>
        <div className="button-group">
          <button onClick={() => exportFile("json")}>Export JSON</button>
          <button onClick={() => exportFile("html")}>Export HTML</button>
        </div>
      </div>
      <ErrorNotice message={error} />
      <div className="investigation-layout">
        <aside className="event-pane" aria-label="Ranked events">
          <div className="event-filters">
            <label>
              Search events
              <input
                type="search"
                value={query}
                onChange={(event) => filter(setQuery, event.target.value)}
              />
            </label>
            <div className="filter-grid">
              <label>
                Evidence class
                <select
                  value={kind}
                  onChange={(event) => filter(setKind, event.target.value)}
                >
                  <option value="all">All classes</option>
                  <option value="behavioral">Behavioral</option>
                  <option value="contextual">Contextual</option>
                  <option value="unavailable">Unavailable</option>
                </select>
              </label>
              <label>
                Feature tier
                <select
                  value={tier}
                  onChange={(event) => filter(setTier, event.target.value)}
                >
                  <option value="all">All tiers</option>
                  <option value="structural">Structural</option>
                  <option value="prose">Prose</option>
                </select>
              </label>
              <label>
                Numerical floor
                <select
                  value={floor}
                  onChange={(event) => filter(setFloor, event.target.value)}
                >
                  <option value="all">Any floor status</option>
                  <option value="yes">Dominant</option>
                  <option value="no">Not dominant</option>
                </select>
              </label>
            </div>
          </div>
          <ErrorNotice message={events.error} retry={events.refresh} />
          {events.loading && (
            <p className="loading" role="status">
              Loading ranked events…
            </p>
          )}
          {events.data && (
            <>
              <p className="result-count">
                {events.data.total} events · original ranks preserved
              </p>
              <ol className="event-list">
                {events.data.events.map((event) => (
                  <li key={event.rank}>
                    <button
                      className={`event-row ${rank === event.rank ? "active" : ""}`}
                      aria-pressed={rank === event.rank}
                      aria-label={`Event ${event.rank}: ${event.representative_feature}`}
                      onClick={() => select(event)}
                    >
                      <span className="event-rank">
                        {event.rank.toString().padStart(2, "0")}
                      </span>
                      <span className="event-description">
                        <strong>{event.representative_feature}</strong>
                        <span>
                          Onset {text(event.onset_step)} · alert{" "}
                          {text(event.alert_step)}
                        </span>
                        <span>
                          {event.n_members}{" "}
                          {event.n_members === 1 ? "feature" : "features"}
                        </span>
                        <Badge warn={event.ranking_status === "unavailable"}>
                          {label(event.ranking_status)}
                        </Badge>
                      </span>
                      <span aria-hidden="true">›</span>
                    </button>
                  </li>
                ))}
              </ol>
              {events.data.events.length === 0 && (
                <p className="empty">No events match these filters.</p>
              )}
              <div className="pagination">
                <button
                  aria-label="Previous events"
                  disabled={offset === 0}
                  onClick={() => setOffset((value) => Math.max(0, value - 50))}
                >
                  Previous
                </button>
                <span>
                  {events.data.total
                    ? `${offset + 1}–${Math.min(offset + 50, events.data.total)}`
                    : "0"}{" "}
                  of {events.data.total}
                </span>
                <button
                  aria-label="Next events"
                  disabled={offset + 50 >= events.data.total}
                  onClick={() => setOffset((value) => value + 50)}
                >
                  Next
                </button>
              </div>
            </>
          )}
        </aside>
        <section className="detail-pane" aria-label="Investigation detail">
          <div className="feature-controls">
            <label>
              Find feature
              <input
                type="search"
                value={featureQuery}
                onChange={(event) => setFeatureQuery(event.target.value)}
                placeholder="Search all observed features"
              />
            </label>
            <label>
              Feature
              <select
                value={feature}
                onChange={(event) => setFeature(event.target.value)}
              >
                <option value="">Select a feature</option>
                {feature &&
                  !features.data?.features.some(
                    (item) => item.feature === feature,
                  ) && <option value={feature}>{feature}</option>}
                {features.data?.features.map((item) => (
                  <option key={item.feature} value={item.feature}>
                    {item.feature} · {item.tier}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <ErrorNotice message={features.error} retry={features.refresh} />
          {rank === null && !feature && (
            <div className="empty detail-empty">
              <h2>Choose an event to investigate</h2>
              <p>
                Open a ranked event, then inspect a citation to read the exact
                original response.
              </p>
              <p>You can also browse any observed feature, including reward.</p>
            </div>
          )}
          <ErrorNotice message={detail.error} retry={detail.refresh} />
          {detail.loading && <p role="status">Loading event evidence…</p>}
          {detail.data && (
            <>
              <div className="detail-heading">
                <div>
                  <span className="eyebrow">
                    Event {detail.data.event.rank}
                  </span>
                  <h2>{detail.data.event.representative_feature}</h2>
                </div>
                <Badge>{label(detail.data.event.ranking_status)}</Badge>
              </div>
              <div className="event-anchors">
                <span>
                  Onset <strong>{text(detail.data.event.onset_step)}</strong>
                </span>
                <span>
                  First alert{" "}
                  <strong>{text(detail.data.event.alert_step)}</strong>
                </span>
                <span>
                  Onset range{" "}
                  <strong>
                    {text(detail.data.event.onset_start)}–
                    {text(detail.data.event.onset_end)}
                  </strong>
                </span>
              </div>
              <p className="note">
                Sequential alerts, retrospective boundaries and corroboration
                are separate evidence. Feature names and reward associations do
                not establish exploit success or mechanism.
              </p>
              <details>
                <summary>Ranking availability and warnings</summary>
                <Fields data={detail.data.event.ranking} />
                {[
                  "missing_evidence_members",
                  "numerical_floor_members",
                  "numerical_floor_unavailable_members",
                ].map((key) =>
                  Array.isArray(detail.data!.event.ranking[key]) &&
                  (detail.data!.event.ranking[key] as Row[]).length > 0 ? (
                    <p className="notice warning" key={key}>
                      {label(key)}:{" "}
                      {(detail.data!.event.ranking[key] as Row[])
                        .map((item) => text(item.feature))
                        .join(", ")}
                    </p>
                  ) : null,
                )}
              </details>
            </>
          )}
          {series.loading && <p role="status">Loading feature observations…</p>}
          <ErrorNotice message={series.error} retry={series.refresh} />
          {withAnchors && (
            <>
              <div className="feature-title">
                <h3>{withAnchors.feature}</h3>
                <Badge>
                  {withAnchors.tier} · {withAnchors.kind}
                </Badge>
              </div>
              <FeatureChart series={withAnchors} />
              {!selectedMember && (
                <details>
                  <summary>Recorded feature evidence</summary>
                  <Fields data={withAnchors.evidence ?? {}} />
                </details>
              )}
              <details>
                <summary>Prompt mix and coverage</summary>
                <DataTable
                  data={withAnchors.prompt_shifts}
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
                  caption="Prompt shifts"
                />
                <DataTable
                  data={withAnchors.prompt_values}
                  columns={[
                    "prompt_id",
                    "before_mean",
                    "after_mean",
                    "change",
                    "before_n",
                    "after_n",
                  ]}
                  caption="Prompt values"
                />
              </details>
              <details>
                <summary>Persistence observations</summary>
                <DataTable
                  data={withAnchors.persistence}
                  columns={["step", "mean", "se", "evaluable", "retained"]}
                  caption="Persistence observations"
                />
              </details>
            </>
          )}
          {detail.data && (
            <section className="members">
              <div className="section-heading">
                <h3>Event members and citations</h3>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={absent}
                    onChange={(event) => setAbsent(event.target.checked)}
                  />
                  Include matched absent
                </label>
              </div>
              {detail.data.event.members.map((member) => (
                <article className="member" key={member.feature}>
                  <div className="section-heading">
                    <h4>{member.feature}</h4>
                    <button onClick={() => setFeature(member.feature)}>
                      View curve
                    </button>
                  </div>
                  <MemberEvidence
                    member={member}
                    full={detail.data?.evidence.find(
                      (item) => item.feature === member.feature,
                    )}
                  />
                  <div className="citations">
                    {member.exemplars.map((example, index) =>
                      example.status === "selected" &&
                      typeof example.sample_id === "string" ? (
                        <button
                          key={example.sample_id}
                          disabled={inspectBusy !== null}
                          onClick={() => inspect(member, example.sample_id!)}
                          aria-label={`Inspect ${example.sample_id}`}
                        >
                          <strong>
                            {inspectBusy === example.sample_id
                              ? "Verifying…"
                              : example.sample_id}
                          </strong>
                          <span>
                            Step {text(example.step)} · reward{" "}
                            {text(example.reward)}
                          </span>
                          <span>Read original response →</span>
                        </button>
                      ) : (
                        <p key={index} className="muted">
                          Citation {label(text(example.status))}
                        </p>
                      ),
                    )}
                    {member.exemplars.length === 0 && (
                      <p className="muted">No selected citations available.</p>
                    )}
                  </div>
                </article>
              ))}
            </section>
          )}
        </section>
      </div>
    </>
  );
}
