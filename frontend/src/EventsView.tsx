import { useEffect, useRef, useState } from "react";
import { post, useResource } from "./api";
import {
  DataTable,
  ErrorNotice,
  Fields,
  figure,
  figures,
  known,
  label,
  plural,
  PROMPT_SHIFT_COLUMNS,
  row,
  sameValue,
  signed,
  text,
} from "./common";
import { FeatureChart } from "./FeatureChart";
import type {
  Citation,
  CompareSeed,
  Event,
  EventDetail,
  Headline,
  Member,
  ReaderContext,
  Row,
  Run,
  Series,
} from "./types";

interface Fact {
  title: string;
  value: string;
  missing?: boolean;
}
function missing(title: string, reason?: unknown): Fact {
  const detail = typeof reason === "string" && reason ? label(reason) : "";
  return {
    title,
    value: detail ? `Unavailable: ${detail}.` : "Unavailable.",
    missing: true,
  };
}
function horizon(value: unknown) {
  return value === true
    ? "horizon complete"
    : value === false
      ? "horizon incomplete"
      : "horizon unavailable";
}
function memberFacts(member: Member, full?: Row): Fact[] {
  const evidence = member.evidence;
  const replication = row(evidence.replication),
    persistence = row(evidence.persistence),
    reward = row(evidence.reward_association),
    breadth = row(evidence.prompt_breadth);
  const facts: Fact[] = [];
  if (evidence.numerical_floor_dominant === true)
    facts.push({
      title: "Scale",
      value: "The numerical floor dominates this alert scale.",
    });
  facts.push(
    full?.retrospective_status === "change"
      ? {
          title: "Retrospective boundary",
          value: `Change point at step ${text(full.changepoint_step)}.`,
        }
      : full?.retrospective_status === "no_change"
        ? {
            title: "Retrospective boundary",
            value: "The retrospective detector selected no change point.",
          }
        : missing("Retrospective boundary", full?.retrospective_status),
  );
  const cutoff = `Cutoff step ${text(replication.cutoff_step)}, ${horizon(replication.horizon_complete)}.`;
  facts.push(
    replication.status === "replicated"
      ? {
          title: "Replication",
          value: `First alert replicated in both sampling-unit halves, same direction, within tolerance. ${cutoff}`,
        }
      : replication.status === "not_replicated"
        ? {
            title: "Replication",
            value: `First alert not replicated in the sampling-unit halves. ${cutoff}`,
          }
        : missing("Replication", replication.status),
  );
  const window = `Window ${text(full?.corroboration_window_start)} to ${text(full?.corroboration_window_end)}, ${horizon(full?.corroboration_horizon_complete)}.`;
  const reason = label(text(full?.corroboration_reason));
  facts.push(
    full?.corroboration_status === "corroborated"
      ? { title: "Corroboration", value: `Corroborated: ${reason}. ${window}` }
      : full?.corroboration_status === "contradicted"
        ? { title: "Corroboration", value: `Contradicted: ${reason}. ${window}` }
        : missing(
            "Corroboration",
            full?.corroboration_reason ?? full?.corroboration_status,
          ),
  );
  facts.push(
    persistence.status === "available" || persistence.status === "partial"
      ? {
          title: "Persistence",
          value: `Retained at ${text(persistence.n_retained)} of ${text(persistence.n_evaluable)} evaluable steps after the first alert. Coverage ${figure(persistence.coverage)}${
            persistence.status === "partial"
              ? `; ${plural(persistence.n_unavailable, "step")} without a measured SE`
              : ""
          }.`,
        }
      : missing("Persistence", persistence.status),
  );
  facts.push(
    reward.status === "available"
      ? {
          title: "Reward association",
          value: `${signed(reward.mean_difference)} mean difference across ${text(reward.n_mixed_groups)} mixed groups and ${text(reward.n_included_rollouts)} rollouts, steps ${text(reward.window_start)} to ${text(reward.window_end)}. Descriptive only.`,
        }
      : missing("Reward association", reward.status),
  );
  facts.push(
    known(breadth.positive_prompts)
      ? {
          title: "Prompt breadth",
          value: `Present in ${text(breadth.positive_prompts)} of ${text(breadth.total_prompts)} prompts observed in steps ${text(full?.prompt_after_start)} to ${text(full?.prompt_after_end)}.`,
        }
      : missing(
          "Prompt breadth",
          member.kind === "continuous"
            ? "breadth counts binary features only"
            : undefined,
        ),
  );
  const se = full?.prompt_shared_se;
  const zero =
    se === 0 ? " A zero measured SE does not establish zero uncertainty." : "";
  facts.push(
    full?.promptmix_status === "available" ||
      full?.promptmix_status === "shared_uncertainty_unavailable"
      ? {
          title: "Prompt mix",
          value: `${plural(full.prompt_shared_prompts, "shared prompt")}. Shared-prompt change ${signed(full.prompt_shared_change)}, SE ${text(se)}. Length distance ${text(full.prompt_length_distance)}.${zero}`,
        }
      : full?.promptmix_status === "no_shared_prompts"
        ? {
            title: "Prompt mix",
            value: "No exact prompts appear in both windows.",
            missing: true,
          }
        : missing("Prompt mix", full?.promptmix_status),
  );
  return facts;
}
function eventName(event: Event, headline?: Headline) {
  const name = `Event ${event.rank}: ${event.representative_feature}`;
  if (!headline) return name;
  const [before, after] = figures(headline.before, headline.after);
  return `${name}, ${before} to ${after}`;
}
/** Sign of the shared-prompt change, treating floating-point noise as none. */
function direction({ before, after }: Headline) {
  return sameValue(before, after) ? 0 : Math.sign(after - before);
}
function Change({ headline }: { headline?: Headline }) {
  if (!headline) return <span className="event-change" />;
  const [before, after] = figures(headline.before, headline.after);
  return (
    <span className="event-change">
      {before} → {after}
    </span>
  );
}
function Title({
  feature,
  headline,
  noShared,
}: {
  feature: string;
  headline?: Headline | null;
  noShared: boolean;
}) {
  const name = <code>{feature}</code>;
  if (!headline)
    return (
      <h1 className="headline">
        {name}
        {noShared && " has no prompts shared across its windows"}
      </h1>
    );
  const moved = direction(headline);
  const [before, after] = figures(headline.before, headline.after);
  return (
    <h1 className="headline">
      {name}{" "}
      {moved === 0
        ? `stayed at ${figure(headline.before)} across shared prompts`
        : `${moved > 0 ? "rose" : "fell"} from ${before} to ${after}`}
    </h1>
  );
}
function dek(
  headline: Headline | null | undefined,
  evidence: Row,
  member: Member | undefined,
  event: Event | undefined,
  alertDirection: string | null | undefined,
) {
  const parts: string[] = [];
  if (headline)
    parts.push(
      `Mean across ${plural(headline.n_prompts, "shared prompt")}: baseline steps ${text(evidence.prompt_before_start)} to ${text(evidence.prompt_before_end)} against onset to first alert, steps ${text(evidence.prompt_after_start)} to ${text(evidence.prompt_after_end)}.`,
    );
  else if (evidence.promptmix_status === "no_shared_prompts")
    parts.push("Before and after values are unavailable.");
  const onset = member?.onset_step ?? evidence.onset_step ?? event?.onset_step;
  const alert = member?.alert_step ?? evidence.alert_step ?? event?.alert_step;
  if (known(onset) && known(alert))
    parts.push(
      `Sequential streak began at step ${text(onset)}; first alert at step ${text(alert)}.`,
    );
  else if (evidence.sequential_status)
    parts.push(`Sequential status: ${label(text(evidence.sequential_status))}.`);
  if (event && String(event.onset_start) !== String(event.onset_end))
    parts.push(
      `Event members began between steps ${text(event.onset_start)} and ${text(event.onset_end)}.`,
    );
  if (headline && alertDirection) {
    const moved = direction(headline);
    if (moved !== (alertDirection === "up" ? 1 : -1))
      parts.push(
        `The sequential alert moved ${alertDirection} across all prompts at each step; ${moved === 0 ? "shared prompts show no net change" : "the shared-prompt mean moved the other way"}.`,
      );
  }
  return parts.join(" ");
}
export function EventsView({
  run,
  onCitation,
  onCompare,
}: {
  run: Run;
  onCitation: (value: Citation, context: ReaderContext) => void;
  onCompare: (seed: CompareSeed) => void;
}) {
  const [query, setQuery] = useState(""),
    [kind, setKind] = useState("all"),
    [tier, setTier] = useState("all"),
    [floor, setFloor] = useState("all"),
    [offset, setOffset] = useState(0);
  const [rank, setRank] = useState<number | null>(null),
    [feature, setFeature] = useState(""),
    [featureQuery, setFeatureQuery] = useState(""),
    [browsing, setBrowsing] = useState(false);
  const [inspectBusy, setInspectBusy] = useState<string | null>(null),
    [error, setError] = useState<string | null>(null),
    [absent, setAbsent] = useState(true);
  const generation = useRef(0);
  const started = useRef(false);
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
  const events = useResource<{
    total: number;
    events: Event[];
    headlines?: Record<string, Headline>;
  }>(`/api/runs/${run.id}/events?${params}`);
  const detail = useResource<EventDetail>(
    rank === null ? null : `/api/runs/${run.id}/events/${rank}`,
  );
  const features = useResource<{
    total: number;
    features: { feature: string; kind: string; tier: string }[];
  }>(
    browsing
      ? `/api/runs/${run.id}/features?q=${encodeURIComponent(featureQuery)}&limit=100`
      : null,
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
  useEffect(() => {
    if (started.current || feature) return;
    const firstEvent = events.data?.events[0];
    if (firstEvent) {
      started.current = true;
      select(firstEvent);
    }
  }, [events.data, feature]);
  function browse(name: string) {
    generation.current++;
    started.current = true;
    setRank(null);
    setFeature(name);
    setError(null);
    setInspectBusy(null);
  }
  function fetchCitation(member: Member, sampleId: string) {
    return post<Citation>(`/api/runs/${run.id}/inspect`, {
      event_rank: rank,
      feature: member.feature,
      sample_id: sampleId,
      with_absent: absent,
    });
  }
  async function inspect(member: Member, sampleId: string) {
    if (inspectBusy !== null) return;
    const requestId = ++generation.current;
    const event = detail.data?.event;
    setInspectBusy(sampleId);
    setError(null);
    try {
      const value = await fetchCitation(member, sampleId);
      if (requestId === generation.current)
        onCitation(value, {
          back: `Event ${event?.rank}, ${member.feature}`,
          feature: member.feature,
          sampleIds: member.exemplars
            .filter(
              (example) =>
                example.status === "selected" &&
                typeof example.sample_id === "string",
            )
            .map((example) => example.sample_id!),
          load: (id) => fetchCitation(member, id),
        });
    } catch (error) {
      if (requestId === generation.current)
        setError(error instanceof Error ? error.message : "Inspection failed.");
    } finally {
      if (requestId === generation.current) setInspectBusy(null);
    }
  }
  const event = rank === null ? undefined : detail.data?.event;
  const selectedMember = event?.members.find(
    (member) => member.feature === feature,
  );
  const fullEvidence = detail.data?.evidence.find(
    (item) => item.feature === feature,
  );
  const evidence: Row = fullEvidence ?? series.data?.evidence ?? {};
  const headline =
    detail.data?.headlines?.[feature] ?? series.data?.headline ?? null;
  const canCompare =
    known(evidence.prompt_before_start) && known(evidence.prompt_after_end);
  const others = event?.members.filter((member) => member.feature !== feature);
  const filtered =
    query !== "" || kind !== "all" || tier !== "all" || floor !== "all";
  const noEvents = events.data?.total === 0 && !filtered;
  return (
    <div className="events">
      <aside aria-label="Ranked events">
        <label className="visually-labelled">
          <span>Search events</span>
          <input
            className="field-line"
            type="search"
            placeholder="Search events"
            value={query}
            onChange={(event) => filter(setQuery, event.target.value)}
          />
        </label>
        <div className="filters" role="group" aria-label="Evidence class">
          {[
            ["all", "All"],
            ["behavioral", "Behavioral"],
            ["contextual", "Contextual"],
            ["unavailable", "Unavailable"],
          ].map(([value, name]) => (
            <button
              key={value}
              type="button"
              className="filter"
              aria-pressed={kind === value}
              onClick={() => filter(setKind, value)}
            >
              {name}
            </button>
          ))}
        </div>
        <ErrorNotice message={events.error} retry={events.refresh} />
        {events.loading && (
          <p className="loading" role="status">
            Loading ranked events…
          </p>
        )}
        {events.data && (
          <>
            <ol className="event-list">
              {events.data.events.map((item) => (
                <li key={item.rank}>
                  <button
                    type="button"
                    className={`event-row ${item.ranking_status}`}
                    aria-pressed={rank === item.rank}
                    aria-label={eventName(
                      item,
                      events.data?.headlines?.[item.representative_feature],
                    )}
                    title={`${item.representative_feature}, ${label(item.ranking_status)}`}
                    onClick={() => select(item)}
                  >
                    <span className="event-rank">{item.rank}</span>
                    <code className="event-feature">
                      {item.representative_feature}
                    </code>
                    <Change
                      headline={
                        events.data?.headlines?.[item.representative_feature]
                      }
                    />
                  </button>
                </li>
              ))}
            </ol>
            {filtered && events.data.events.length === 0 && (
              <p className="quiet-note">No events match these filters.</p>
            )}
            {!noEvents && (
              <p className="quiet-note">
                {plural(events.data.total, "event")}. Ranks stay fixed when you
                filter.
              </p>
            )}
            {events.data.total > 50 && (
              <p className="pagination">
                <button
                  type="button"
                  className="link"
                  aria-label="Previous events"
                  disabled={offset === 0}
                  onClick={() => setOffset((value) => Math.max(0, value - 50))}
                >
                  Previous
                </button>
                <span>
                  {offset + 1} to {Math.min(offset + 50, events.data.total)} of{" "}
                  {events.data.total}
                </span>
                <button
                  type="button"
                  className="link"
                  aria-label="Next events"
                  disabled={offset + 50 >= events.data.total}
                  onClick={() => setOffset((value) => value + 50)}
                >
                  Next
                </button>
              </p>
            )}
          </>
        )}
        <details className="more">
          <summary>More filters</summary>
          <div className="filter-fields">
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
        </details>
        <details
          className="more"
          open={browsing}
          onToggle={(event) => setBrowsing(event.currentTarget.open)}
        >
          <summary>Browse all features</summary>
          <label className="visually-labelled">
            <span>Find feature</span>
            <input
              className="field-line"
              type="search"
              placeholder="Search all observed features"
              value={featureQuery}
              onChange={(event) => setFeatureQuery(event.target.value)}
            />
          </label>
          <ErrorNotice message={features.error} retry={features.refresh} />
          {features.data &&
            features.data.total > features.data.features.length && (
              <p className="quiet-note">
                {features.data.features.length} of {features.data.total}{" "}
                features shown. Search to narrow.
              </p>
            )}
          <ul className="feature-list">
            {features.data?.features.map((item) => (
              <li key={item.feature}>
                <button
                  type="button"
                  className="event-row"
                  aria-pressed={rank === null && feature === item.feature}
                  aria-label={`Feature ${item.feature}`}
                  title={item.tier}
                  onClick={() => browse(item.feature)}
                >
                  <span />
                  <code className="event-feature">{item.feature}</code>
                  <span />
                </button>
              </li>
            ))}
          </ul>
        </details>
      </aside>
      <section className="event-detail" aria-label="Investigation detail">
        <ErrorNotice message={error} />
        {rank === null && !feature && !events.loading && (
          <div className="blank">
            <h1 className="headline">
              {noEvents
                ? "No events in this run"
                : "Choose an event to investigate"}
            </h1>
            <p className="dek">
              {noEvents ? (
                <>
                  {text(run.counts?.screened_series)} of{" "}
                  {text(run.counts?.series)} series were screened and{" "}
                  {text(run.counts?.unscorable_series)} were unscorable.
                  Unscorable series are missing evidence, not a negative result.
                  You can still browse any observed feature, including reward.
                </>
              ) : (
                "Open a ranked event, then read an exact original response. You can also browse any observed feature, including reward."
              )}
            </p>
          </div>
        )}
        <ErrorNotice message={detail.error} retry={detail.refresh} />
        {detail.loading && !event && (
          <p className="loading" role="status">
            Loading event evidence…
          </p>
        )}
        {feature && (rank === null || event) && (
          <>
            <p className="kicker">
              <span>
                {event
                  ? `Event ${event.rank} of ${text(run.counts?.events)}, ${label(event.ranking_status)}`
                  : `Feature, ${series.data ? `${series.data.tier}, ${series.data.kind}` : "loading"}`}
              </span>
              {canCompare && (
                <button
                  type="button"
                  className="link"
                  onClick={() =>
                    onCompare({
                      feature,
                      a: [
                        text(evidence.prompt_before_start),
                        text(evidence.prompt_before_end),
                      ],
                      b: [
                        text(evidence.prompt_after_start),
                        text(evidence.prompt_after_end),
                      ],
                    })
                  }
                >
                  Compare these windows
                </button>
              )}
            </p>
            <Title
              feature={feature}
              headline={headline}
              noShared={evidence.promptmix_status === "no_shared_prompts"}
            />
            <p className="dek">
              {dek(
                headline,
                evidence,
                selectedMember,
                event,
                selectedMember?.direction ?? series.data?.direction,
              )}
            </p>
            {others && others.length > 0 && (
              <p className="also">
                <span>Also in this event</span>
                {others.map((member) => (
                  <button
                    type="button"
                    key={member.feature}
                    className="link"
                    onClick={() => setFeature(member.feature)}
                  >
                    <code>{member.feature}</code>
                  </button>
                ))}
              </p>
            )}
            {series.loading && (
              <p className="loading" role="status">
                Loading feature observations…
              </p>
            )}
            <ErrorNotice message={series.error} retry={series.refresh} />
            {series.data && (
              <FeatureChart
                series={series.data}
                onsetRange={
                  event ? [event.onset_start, event.onset_end] : undefined
                }
                caption={
                  headline
                    ? "The headline averages only prompts shared by the baseline and onset-to-alert windows; the curve averages all prompts at each step."
                    : undefined
                }
              />
            )}
          </>
        )}
        {event && selectedMember && (
          <section className="section" aria-label="Responses">
            <div className="section-heading">
              <h2>Responses</h2>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={absent}
                  onChange={(event) => setAbsent(event.target.checked)}
                />
                Include a matched response without the feature
              </label>
            </div>
            <ul className="responses">
              {selectedMember.exemplars.map((example, index) =>
                example.status === "selected" &&
                typeof example.sample_id === "string" ? (
                  <li key={example.sample_id}>
                    <button
                      type="button"
                      className="response-row"
                      aria-disabled={inspectBusy !== null}
                      aria-label={`Read response ${example.sample_id}, step ${text(example.step)}, reward ${text(example.reward)}, ${label(text(example.selection_reason))}${inspectBusy === example.sample_id ? ", verifying" : ""}`}
                      onClick={() =>
                        inspect(selectedMember, example.sample_id!)
                      }
                    >
                      <code>{example.sample_id}</code>
                      <span>Step {text(example.step)}</span>
                      <span>Reward {text(example.reward)}</span>
                      <span>{label(text(example.selection_reason))}</span>
                      <span className="read">
                        {inspectBusy === example.sample_id
                          ? "Verifying…"
                          : "Read"}
                      </span>
                    </button>
                  </li>
                ) : (
                  <li key={index} className="unavailable response-missing">
                    Citation unavailable: {label(text(example.status))}.
                  </li>
                ),
              )}
              {selectedMember.exemplars.length === 0 && (
                <li className="unavailable response-missing">
                  No selected citations available.
                </li>
              )}
            </ul>
          </section>
        )}
        {event && selectedMember && (
          <section className="section" aria-label="Evidence">
            <h2>Evidence</h2>
            <p className="section-note">
              Sequential alerts, retrospective boundaries and corroboration are
              separate evidence. Feature names and reward associations are
              descriptive and do not establish mechanism or exploit success.
            </p>
            <dl className="facts">
              {memberFacts(selectedMember, fullEvidence).map((fact) => (
                <div key={fact.title}>
                  <dt>{fact.title}</dt>
                  <dd className={fact.missing ? "unavailable" : undefined}>
                    {fact.value}
                  </dd>
                </div>
              ))}
            </dl>
            <details className="more">
              <summary>Ranking availability and warnings</summary>
              <Fields data={event.ranking} />
              {[
                "missing_evidence_members",
                "numerical_floor_members",
                "numerical_floor_unavailable_members",
              ].map((key) =>
                Array.isArray(event.ranking[key]) &&
                (event.ranking[key] as Row[]).length > 0 ? (
                  <p className="section-note" key={key}>
                    {label(key)}:{" "}
                    {(event.ranking[key] as Row[])
                      .map((item) => text(item.feature))
                      .join(", ")}
                  </p>
                ) : null,
              )}
            </details>
          </section>
        )}
        {series.data && feature && (rank === null || event) && (
          <details className="more">
            <summary>All recorded fields</summary>
            <Fields data={evidence} />
            <DataTable
              data={series.data.prompt_shifts}
              columns={PROMPT_SHIFT_COLUMNS}
              caption="Prompt shifts"
            />
            <DataTable
              data={series.data.prompt_values}
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
            <DataTable
              data={series.data.persistence}
              columns={["step", "mean", "se", "status", "retained"]}
              caption="Persistence observations"
            />
          </details>
        )}
      </section>
    </div>
  );
}
