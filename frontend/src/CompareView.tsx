import {
  Fragment,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type PointerEvent,
  type ReactNode,
} from "react";
import { downloadReport, post, request, useResource } from "./api";
import {
  DataTable,
  ErrorNotice,
  Fields,
  figure,
  known,
  label,
  plural,
  PROMPT_SHIFT_COLUMNS,
  rows,
  signed,
  text,
} from "./common";
import type {
  Citation,
  CompareSeed,
  Comparison,
  ComparisonFeature,
  Estimate,
  ReaderContext,
  Row,
  Run,
  Step,
} from "./types";

type Side = "a" | "b";
type Range = [string, string];

const STEP = /^\d+$/;
const PREVIEW_ROWS = 20;
const COUNTS = ["n_prompts", "n_sampling_units", "n_rollouts"];

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
function ordered(steps: bigint[]) {
  return [...new Set(steps)].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}
function observed(data: { steps: Step[] } | null) {
  return (data?.steps ?? []).map((step) => BigInt(step));
}
function spans(steps: bigint[]) {
  let spacing = 0n;
  for (let index = 1; index < steps.length; index++) {
    const gap = steps[index] - steps[index - 1];
    if (!spacing || gap < spacing) spacing = gap;
  }
  const result: [bigint, bigint][] = [];
  steps.forEach((step, index) => {
    if (index && step - steps[index - 1] <= spacing)
      result[result.length - 1][1] = step;
    else result.push([step, step]);
  });
  return result;
}
function interval(value: Estimate): [number, number] | null {
  return finite(value.ci_low) && finite(value.ci_high)
    ? [value.ci_low, value.ci_high]
    : null;
}
function reach(value: Estimate) {
  return Math.max(
    0,
    ...[value.estimate, ...(interval(value) ?? [])]
      .filter(finite)
      .map((bound) => Math.abs(bound)),
  );
}
function scaleFor(largest: number) {
  if (largest <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(largest));
  return Math.ceil(largest / power) * power;
}

function StepInput({
  name,
  value,
  onChange,
}: {
  name: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <input
      className="compare-step"
      aria-label={name}
      inputMode="numeric"
      pattern="[0-9]+"
      autoComplete="off"
      required
      style={{ width: `${Math.max(value.length, 2) + 1}ch` }}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}
function Timeline({
  axis,
  aSteps,
  bSteps,
  twoRuns,
  a,
  b,
  side,
  onWindow,
}: {
  axis: bigint[];
  aSteps: bigint[];
  bSteps: bigint[];
  twoRuns: boolean;
  a: Range;
  b: Range;
  side: Side;
  onWindow: (start: string, end: string) => void;
}) {
  const anchor = useRef<bigint | null>(null);
  const first = axis[0],
    last = axis[axis.length - 1];
  const percent = (step: bigint) =>
    last === first
      ? 50
      : Number(((step - first) * 1000000n) / (last - first)) / 10000;
  const at = (step: bigint) => `${percent(step)}%`;
  function nearest(event: PointerEvent<SVGSVGElement>) {
    const steps = side === "a" ? aSteps : bSteps;
    const box = event.currentTarget.getBoundingClientRect();
    if (!box.width || !steps.length) return null;
    const ratio = Math.min(
      1,
      Math.max(0, (event.clientX - box.left) / box.width),
    );
    const target =
      first + ((last - first) * BigInt(Math.round(ratio * 1000000))) / 1000000n;
    const distance = (step: bigint) =>
      step < target ? target - step : step - target;
    return steps.reduce((best, step) =>
      distance(step) < distance(best) ? step : best,
    );
  }
  function start(event: PointerEvent<SVGSVGElement>) {
    if (event.button !== 0) return;
    const step = nearest(event);
    if (step === null) return;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    anchor.current = step;
    onWindow(String(step), String(step));
  }
  function move(event: PointerEvent<SVGSVGElement>) {
    if (anchor.current === null) return;
    const step = nearest(event);
    if (step === null) return;
    const [low, high] =
      step < anchor.current ? [step, anchor.current] : [anchor.current, step];
    onWindow(String(low), String(high));
  }
  function stop() {
    anchor.current = null;
  }
  function bar(name: Side, [from, to]: Range, y: number, labelY: number) {
    if (!STEP.test(from) || !STEP.test(to)) return null;
    const low = BigInt(from),
      high = BigInt(to);
    if (low > high || high < first || low > last) return null;
    const left = low < first ? first : low,
      right = high > last ? last : high;
    return (
      <g className={`compare-window ${name}`}>
        <line x1={at(left)} x2={at(right)} y1={y} y2={y} />
        <text
          x={at(left)}
          y={labelY}
          textAnchor={percent(left) > 97 ? "end" : "start"}
        >
          {name.toUpperCase()}
        </text>
      </g>
    );
  }
  const middle = first + (last - first) / 2n;
  // Two runs get a line each, so a gap in one is not filled by the other.
  const tracks = twoRuns
    ? [
        { steps: aSteps, y: 22 },
        { steps: bSteps, y: 28 },
      ]
    : [{ steps: aSteps, y: 25 }];
  return (
    <div className="compare-timeline">
      <svg
        aria-hidden="true"
        width="100%"
        height="50"
        onPointerDown={start}
        onPointerMove={move}
        onPointerUp={stop}
        onPointerCancel={stop}
      >
        {tracks.map(({ steps, y }) =>
          spans(steps).map(([from, to]) => (
            <line
              key={`${y}-${from}`}
              className="compare-observed"
              x1={at(from)}
              x2={at(to)}
              y1={y}
              y2={y}
            />
          )),
        )}
        {bar("a", a, 22, 13)}
        {bar("b", b, 28, 45)}
      </svg>
      <div className="compare-axis">
        <span>{first === last ? "" : String(first)}</span>
        <span>
          {first === last
            ? String(first)
            : middle > first && middle < last
              ? String(middle)
              : ""}
        </span>
        <span>{first === last ? "" : String(last)}</span>
      </div>
    </div>
  );
}
function Overlap({ mix }: { mix: Row }) {
  if (mix.status === "no_shared_prompts")
    return (
      <p className="compare-overlap">
        <strong>No prompts appear in both windows.</strong> Paired estimates are
        unavailable.
      </p>
    );
  const onlyA = text(mix.only_a_prompts);
  return (
    <p className="compare-overlap">
      <strong>
        {plural(mix.shared_prompts, "prompt")}{" "}
        {String(mix.shared_prompts) === "1" ? "appears" : "appear"} in both
        windows.
      </strong>{" "}
      {onlyA} {onlyA === "1" ? "appears" : "appear"} only in A and{" "}
      {text(mix.only_b_prompts)} only in B.
    </p>
  );
}
function IntervalPlot({
  paired,
  estimate,
  scale,
}: {
  paired: Estimate;
  estimate: number;
  scale: number;
}) {
  const bounds = interval(paired);
  const at = (value: number) => `${50 + (value / scale) * 50}%`;
  const words = bounds
    ? `95% normal interval ${signed(bounds[0])} to ${signed(bounds[1])}`
    : `interval unavailable, ${label(paired.status)}`;
  return (
    <svg
      className="compare-plot"
      role="img"
      aria-label={`Shared-prompt change ${signed(estimate)}, ${words}.${paired.se === 0 ? " A zero measured SE does not establish zero uncertainty." : ""}`}
      width="100%"
      height="20"
    >
      <line className="compare-zero" x1="50%" x2="50%" y1="0" y2="20" />
      {bounds && (
        <line
          className="error-bar"
          x1={at(bounds[0])}
          x2={at(bounds[1])}
          y1="10"
          y2="10"
        />
      )}
      <circle
        className={known(paired.se) ? "mean-point" : "uncertain-point"}
        cx={at(estimate)}
        cy="10"
        r="3.5"
      />
    </svg>
  );
}
function SideMean({ value }: { value: Estimate }) {
  return finite(value.mean) ? (
    <>{figure(value.mean)}</>
  ) : (
    <span className="unavailable">Unavailable: {label(value.status)}</span>
  );
}
function FeatureTable({
  binary,
  items,
  selected,
  onToggle,
  detail,
}: {
  binary: boolean;
  items: ComparisonFeature[];
  selected: string;
  onToggle: (feature: string) => void;
  detail: (item: ComparisonFeature) => ReactNode;
}) {
  if (!items.length) return null;
  const shared = scaleFor(Math.max(...items.map((item) => reach(item.paired))));
  return (
    <div className="table-scroll" tabIndex={0}>
      <table className="compare-table">
        <caption>{binary ? "Binary features" : "Continuous features"}</caption>
        <thead>
          <tr>
            <th scope="col">Feature</th>
            <th scope="col" className="num">
              A mean
            </th>
            <th scope="col" className="num">
              B mean
            </th>
            <th scope="col" className="num">
              B minus A
            </th>
            <th scope="col">
              {!binary ? (
                "Each row on its own scale"
              ) : items.some((item) => finite(item.paired.estimate)) ? (
                <span className="compare-scale">
                  <span>{signed(-shared)}</span> <span>0</span>{" "}
                  <span>{signed(shared)}</span>
                </span>
              ) : (
                "Interval"
              )}
            </th>
            <th scope="col" className="num">
              Shared-prompt change
            </th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <Fragment key={item.feature}>
              <tr>
                <th scope="row">
                  <button
                    type="button"
                    className="link"
                    aria-expanded={selected === item.feature}
                    onClick={() => onToggle(item.feature)}
                  >
                    <code>{item.feature}</code>
                  </button>
                </th>
                <td className="num">
                  <SideMean value={item.a} />
                </td>
                <td className="num">
                  <SideMean value={item.b} />
                </td>
                <td className="num">
                  {finite(item.difference.estimate) ? (
                    signed(item.difference.estimate)
                  ) : (
                    <span className="unavailable">Unavailable</span>
                  )}
                </td>
                {finite(item.paired.estimate) ? (
                  <>
                    <td>
                      <IntervalPlot
                        paired={item.paired}
                        estimate={item.paired.estimate}
                        scale={binary ? shared : scaleFor(reach(item.paired))}
                      />
                    </td>
                    <td className="num">
                      {signed(item.paired.estimate)}
                      {!interval(item.paired) && (
                        <span className="compare-no-interval">
                          No interval: {label(item.paired.status)}
                        </span>
                      )}
                    </td>
                  </>
                ) : (
                  <td colSpan={2} className="unavailable">
                    Unavailable: {label(item.paired.status)}
                  </td>
                )}
              </tr>
              {selected === item.feature && (
                <tr>
                  <td colSpan={6} className="compare-detail">
                    {detail(item)}
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function EstimateView({ title, value }: { title: string; value: Estimate }) {
  const change = value.mean === undefined;
  const format = change ? signed : text;
  const bounds = interval(value);
  const lines = [
    [change ? "Estimate" : "Mean", format(value.mean ?? value.estimate)],
    ["SE", text(value.se)],
    ["Status", label(value.status)],
    ...(bounds
      ? [["95% normal interval", `${format(bounds[0])} to ${format(bounds[1])}`]]
      : []),
    ...COUNTS.filter((key) => known(value[key])).map((key) => [
      label(key),
      text(value[key]),
    ]),
  ];
  return (
    <section className="compare-estimate">
      <h3>{title}</h3>
      <dl>
        {lines.map(([name, content]) => (
          <div key={name}>
            <dt>{name}</dt>
            <dd
              className={content === "Unavailable" ? "unavailable" : undefined}
            >
              {content}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
export function CompareView({
  run,
  runs,
  seed,
  onCitation,
}: {
  run: Run;
  runs: Run[];
  seed: CompareSeed | null;
  onCitation: (value: Citation, context: ReaderContext) => void;
}) {
  const [other, setOther] = useState(""),
    [aStart, setAStart] = useState(""),
    [aEnd, setAEnd] = useState(""),
    [bStart, setBStart] = useState(""),
    [bEnd, setBEnd] = useState(""),
    [side, setSide] = useState<Side>("a");
  const [comparison, setComparison] = useState<Comparison | null>(null),
    [submitted, setSubmitted] = useState(""),
    [busy, setBusy] = useState(false),
    [error, setError] = useState<string | null>(null),
    [query, setQuery] = useState(""),
    [selected, setSelected] = useState(""),
    [pinned, setPinned] = useState(""),
    [showAll, setShowAll] = useState(false);
  const [inspectBusy, setInspectBusy] = useState<string | null>(null),
    [inspectError, setInspectError] = useState<string | null>(null),
    [exportError, setExportError] = useState<string | null>(null);
  const [pollRevision, setPollRevision] = useState(0);
  const generation = useRef(0);
  // Separate from generation: StrictMode's simulated unmount must not discard
  // a seeded request, which the seed guard will not send a second time.
  const mounted = useRef(true);
  const seeded = useRef<CompareSeed | null>(null);
  const live = (version: number) =>
    mounted.current && version === generation.current;
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  const runSteps = useResource<{ steps: Step[] }>(`/api/runs/${run.id}/steps`);
  const otherSteps = useResource<{ steps: Step[] }>(
    other ? `/api/runs/${other}/steps` : null,
  );
  const aSteps = useMemo(() => observed(runSteps.data), [runSteps.data]);
  const chosenSteps = useMemo(
    () => observed(otherSteps.data),
    [otherSteps.data],
  );
  const bSteps = other ? chosenSteps : aSteps;
  const axis = useMemo(() => ordered([...aSteps, ...bSteps]), [aSteps, bSteps]);
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
  async function compare(comparisonRun: string, a: Range, b: Range) {
    setError(null);
    if (
      ![...a, ...b].every((value) => STEP.test(value)) ||
      BigInt(a[0]) > BigInt(a[1]) ||
      BigInt(b[0]) > BigInt(b[1])
    ) {
      setError(
        "Enter nonnegative integer step ranges with each start at or before its end.",
      );
      return;
    }
    const version = ++generation.current;
    setBusy(true);
    setComparison(null);
    setSubmitted([comparisonRun, ...a, ...b].join(" "));
    setSelected("");
    setPinned("");
    setShowAll(false);
    setInspectBusy(null);
    setInspectError(null);
    setExportError(null);
    try {
      const value = await post<Comparison>("/api/comparisons", {
        run_a: run.id,
        run_b: comparisonRun || null,
        a,
        b,
      });
      if (live(version)) setComparison(value);
    } catch (error) {
      if (live(version))
        setError(error instanceof Error ? error.message : "Comparison failed.");
    } finally {
      if (live(version)) setBusy(false);
    }
  }
  useEffect(() => {
    if (!seed || seeded.current === seed) return;
    seeded.current = seed;
    setOther("");
    setAStart(seed.a[0]);
    setAEnd(seed.a[1]);
    setBStart(seed.b[0]);
    setBEnd(seed.b[1]);
    compare("", seed.a, seed.b);
    setSelected(seed.feature);
    setPinned(seed.feature);
    setQuery("");
  }, [seed]);
  function submit(event: FormEvent) {
    event.preventDefault();
    compare(other, [aStart, aEnd], [bStart, bEnd]);
  }
  function setWindow(start: string, end: string) {
    (side === "a" ? setAStart : setBStart)(start);
    (side === "a" ? setAEnd : setBEnd)(end);
  }
  async function inspect(
    citationSide: Side,
    item: ComparisonFeature,
    sampleId: string,
  ) {
    if (!comparison || inspectBusy !== null) return;
    const id = comparison.id,
      feature = item.feature;
    const load = (value: string) =>
      post<Citation>(`/api/comparisons/${id}/inspect`, {
        side: citationSide,
        feature,
        sample_id: value,
      });
    const version = ++generation.current;
    setInspectBusy(sampleId);
    setInspectError(null);
    try {
      const value = await load(sampleId);
      if (live(version))
        onCitation(value, {
          back: `Compare, ${feature}`,
          feature,
          sampleIds: item.citations[citationSide].items.map((example) =>
            String(example.sample_id),
          ),
          load,
        });
    } catch (error) {
      if (live(version))
        setInspectError(
          error instanceof Error ? error.message : "Inspection failed.",
        );
    } finally {
      if (live(version)) setInspectBusy(null);
    }
  }
  async function exportFile(format: string) {
    if (!comparison) return;
    const version = generation.current;
    setExportError(null);
    try {
      await downloadReport(
        `/api/comparisons/${comparison.id}/export?format=${format}`,
        `comparison.${format}`,
      );
    } catch (error) {
      if (live(version))
        setExportError(
          error instanceof Error ? error.message : "Export failed.",
        );
    }
  }
  function toggle(feature: string) {
    setInspectError(null);
    setSelected((value) => (value === feature ? "" : feature));
  }
  const report = comparison?.status === "ready" ? comparison.report : null;
  const stale =
    !!report && submitted !== [other, aStart, aEnd, bStart, bEnd].join(" ");
  const pending =
    busy ||
    comparison?.status === "queued" ||
    comparison?.status === "comparing";
  const needle = query.toLowerCase();
  const matches =
    report?.features.filter((item) =>
      item.feature.toLowerCase().includes(needle),
    ) ?? [];
  const binary = matches.filter((item) => item.kind === "binary");
  const continuous = matches.filter((item) => item.kind !== "binary");
  const shownBinary =
    showAll || query
      ? binary
      : binary.filter(
          (item, index) => index < PREVIEW_ROWS || item.feature === pinned,
        );
  function detail(item: ComparisonFeature) {
    return (
      <>
        <p className="quiet-note">
          Tier {item.tier}, B minus A direction {label(item.direction)}.
        </p>
        <div className="compare-estimates">
          <EstimateView title="A estimate" value={item.a} />
          <EstimateView title="B estimate" value={item.b} />
          <EstimateView title="Difference" value={item.difference} />
          <EstimateView title="Shared-prompt estimate" value={item.paired} />
        </div>
        <p className="section-note">
          B minus A interval method: {label(text(item.difference.method))}.
          Reward associations and feature contrasts are descriptive.
        </p>
        <ErrorNotice message={inspectError} />
        <div className="compare-citations">
          {(["a", "b"] as const).map((citationSide) => {
            const citations = item.citations[citationSide];
            return (
              <section key={citationSide}>
                <h3>Side {citationSide.toUpperCase()} citations</h3>
                {citations.items.length ? (
                  <p className="section-note">
                    Status: {label(citations.status)}.
                  </p>
                ) : (
                  <p className="unavailable">
                    Unavailable: {label(citations.status)}.
                  </p>
                )}
                <ul className="compare-reads">
                  {citations.items.map((example) => (
                    <li key={String(example.sample_id)}>
                      <button
                        type="button"
                        className="link"
                        aria-disabled={inspectBusy !== null}
                        onClick={() =>
                          inspect(
                            citationSide,
                            item,
                            String(example.sample_id),
                          )
                        }
                      >
                        Read response <code>{example.sample_id}</code>
                      </button>
                      <span>
                        {inspectBusy === example.sample_id
                          ? "Verifying…"
                          : `Step ${text(example.step)}`}
                      </span>
                    </li>
                  ))}
                </ul>
              </section>
            );
          })}
        </div>
      </>
    );
  }
  return (
    <section className="compare">
      <form noValidate onSubmit={submit}>
        <h1 className="headline compare-title">
          Compare steps{" "}
          <StepInput name="A start" value={aStart} onChange={setAStart} /> to{" "}
          <StepInput name="A end" value={aEnd} onChange={setAEnd} /> with steps{" "}
          <StepInput name="B start" value={bStart} onChange={setBStart} /> to{" "}
          <StepInput name="B end" value={bEnd} onChange={setBEnd} />
        </h1>
        {axis.length > 0 && (
          <Timeline
            axis={axis}
            aSteps={aSteps}
            bSteps={bSteps}
            twoRuns={!!other}
            a={[aStart, aEnd]}
            b={[bStart, bEnd]}
            side={side}
            onWindow={setWindow}
          />
        )}
        <div className="compare-controls">
          {axis.length > 0 && (
            <span
              className="compare-drag"
              role="group"
              aria-label="Side to drag"
            >
              Drag the line to set
              {(["a", "b"] as const).map((name) => (
                <button
                  type="button"
                  key={name}
                  className="filter"
                  aria-pressed={side === name}
                  onClick={() => setSide(name)}
                >
                  {name.toUpperCase()}
                </button>
              ))}
            </span>
          )}
          <label className="compare-from">
            Window B from
            <select
              aria-label="Window B from comparison run"
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
          <button
            type="submit"
            className={report && !stale ? "pill quiet" : "pill"}
            disabled={pending}
          >
            Compare windows
          </button>
        </div>
      </form>
      {stale && (
        <p className="compare-stale" role="status">
          The windows changed. Results below are from the previous comparison.
        </p>
      )}
      <ErrorNotice message={runSteps.error} retry={runSteps.refresh} />
      <ErrorNotice message={otherSteps.error} retry={otherSteps.refresh} />
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
      <ErrorNotice message={comparison?.error ?? null} />
      {pending && (
        <p className="loading" role="status">
          {comparison?.status === "comparing"
            ? "Computing comparison…"
            : "Comparison queued…"}
        </p>
      )}
      {report && (
        <div className={stale ? "compare-results stale" : "compare-results"}>
          <Overlap mix={report.prompt_mix} />
          <p className="section-note compare-overlap-note">
            A and B means cover all prompts in each window. Only exact shared
            prompts support the shared-prompt change, and prompt composition can
            contribute to the overall difference.
          </p>
          <section className="section">
            <div className="section-heading">
              <h2>What changed</h2>
              <input
                className="field-line compare-search"
                type="search"
                aria-label="Search comparison features"
                placeholder="Search features"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </div>
            {matches.length === 0 ? (
              <p className="quiet-note">No comparison features match.</p>
            ) : (
              <figure className="figure compare-figure">
                <FeatureTable
                  binary
                  items={shownBinary}
                  selected={selected}
                  onToggle={toggle}
                  detail={detail}
                />
                <FeatureTable
                  binary={false}
                  items={continuous}
                  selected={selected}
                  onToggle={toggle}
                  detail={detail}
                />
                {shownBinary.length < binary.length && (
                  <p className="link-row compare-limit">
                    <span>
                      {shownBinary.length} of {binary.length} binary features
                      shown. Hidden rows can include unavailable estimates.
                    </span>
                    <button
                      type="button"
                      className="link"
                      onClick={() => setShowAll(true)}
                    >
                      Show all {binary.length} binary features
                    </button>
                  </p>
                )}
                <figcaption>
                  <strong>Figure 1.</strong> Shared-prompt change, B minus A,
                  with pointwise 95% normal intervals from measured standard
                  errors, unadjusted across features. An open point has no
                  measured SE. A zero measured SE does not establish zero
                  uncertainty. Binary features follow report order, largest
                  overall difference first. Continuous features are in name
                  order.
                </figcaption>
              </figure>
            )}
            <ErrorNotice message={exportError} />
            <p className="link-row">
              <button
                type="button"
                className="link"
                onClick={() => exportFile("json")}
              >
                Export comparison JSON
              </button>
              <button
                type="button"
                className="link"
                onClick={() => exportFile("html")}
              >
                Export comparison HTML
              </button>
            </p>
            <details className="more">
              <summary>Compared windows and prompt shifts</summary>
              <p className="quiet-note">Ranges are inclusive.</p>
              <DataTable
                caption="Compared windows"
                data={report.sides.map((item) => ({
                  ...item,
                  window: (item.window as Step[]).join(" to "),
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
              <Fields
                data={{
                  ...report.prompt_mix,
                  status: label(text(report.prompt_mix.status)),
                }}
                fields={[
                  "status",
                  "shared_prompts",
                  "only_a_prompts",
                  "only_b_prompts",
                ]}
              />
              <DataTable
                data={rows(report.prompt_mix.prompt_shift)}
                columns={PROMPT_SHIFT_COLUMNS}
                caption="Comparison prompt shifts"
              />
            </details>
          </section>
        </div>
      )}
    </section>
  );
}
