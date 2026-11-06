import { useEffect, useRef, useState } from "react";
import { DataTable, known, text } from "./common";
import type { Row, Series, Step } from "./types";

const HEIGHT = 320;
const TRACKS = 44;
const LEFT = 52;
const RIGHT = 12;
const TOP = 40;
const BOTTOM = 280;

function niceTop(value: number) {
  if (value <= 0) return 0;
  const power = 10 ** Math.floor(Math.log10(value));
  const fraction = value / power;
  const stops = [1, 1.5, 2, 3, 4, 5, 6, 8, 10];
  return (stops.find((stop) => fraction <= stop + 1e-9) ?? 10) * power;
}
function useWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    if (!ref.current || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) =>
      setWidth(Math.max(280, Math.round(entry.contentRect.width))),
    );
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}
export function FeatureChart({
  series,
  onsetRange,
  caption,
}: {
  series: Series;
  onsetRange?: [Step, Step];
  caption?: string;
}) {
  const [ref, width] = useWidth(750);
  const [tracks, setTracks] = useState(false);
  const points = series.points;
  const bounds = points
    .filter((point) => point.mean !== null && Number.isFinite(point.mean))
    .flatMap((point) => [
      point.mean! - (point.se ?? 0),
      point.mean! + (point.se ?? 0),
    ]);
  const high = bounds.length ? Math.max(0, ...bounds) : 1;
  const low = bounds.length ? Math.min(0, ...bounds) : 0;
  const top = niceTop(high);
  const bottom = low < 0 ? -niceTop(-low) : 0;
  const span = top - bottom || 1;
  const steps = points.map((point) => BigInt(String(point.step)));
  const first = steps[0] ?? 0n,
    last = steps[steps.length - 1] ?? first;
  const right = width - RIGHT;
  const position = (step: bigint) =>
    LEFT +
    (last === first
      ? 0.5
      : Number(((step - first) * 1000000n) / (last - first)) / 1000000) *
      (right - LEFT);
  const x = (index: number) => position(steps[index]);
  const y = (value: number) =>
    BOTTOM - ((value - bottom) / span) * (BOTTOM - TOP);
  const same = (left: unknown, step: Step) =>
    known(left) && String(left) === String(step);

  // A schedule can be coarser than one step, so only a jump wider than the
  // smallest spacing counts as missing observations.
  let spacing: bigint | null = null;
  steps.forEach((step, index) => {
    if (!index) return;
    const difference = step - steps[index - 1];
    if (spacing === null || difference < spacing) spacing = difference;
  });
  const groups: number[][] = [];
  let group: number[] = [];
  points.forEach((point, index) => {
    const jump =
      index > 0 && spacing !== null && steps[index] - steps[index - 1] > spacing;
    if (jump || point.mean === null) {
      if (group.length) groups.push(group);
      group = [];
    }
    if (point.mean !== null) group.push(index);
  });
  if (group.length) groups.push(group);
  const breaks = groups.slice(1).map((current, index) => {
    const previous = groups[index];
    return [previous[previous.length - 1], current[0]] as const;
  });

  const bands: string[] = [];
  const bars: number[] = [];
  for (const members of groups) {
    let run: number[] = [];
    const flush = () => {
      if (run.length === 1) bars.push(run[0]);
      if (run.length > 1) {
        const upper = run.map(
          (index) => `${x(index)} ${y(points[index].mean! + points[index].se!)}`,
        );
        const lower = [...run]
          .reverse()
          .map(
            (index) =>
              `${x(index)} ${y(points[index].mean! - points[index].se!)}`,
          );
        bands.push(`M ${upper.join(" L ")} L ${lower.join(" L ")} Z`);
      }
      run = [];
    };
    for (const index of members) {
      if (points[index].se === null) flush();
      else run.push(index);
    }
    flush();
  }

  const evidence = series.evidence;
  const onset = points.findIndex((point) =>
    same(evidence.onset_step, point.step),
  );
  const alert = points.findIndex((point) =>
    same(evidence.alert_step, point.step),
  );
  const onsetSide = onset >= 0 && x(onset) < 150 ? "start" : "end";
  const alertSide = alert >= 0 && x(alert) > right - 150 ? "end" : "start";
  const persistence = new Map(
    series.persistence.map((item) => [String(item.step), item]),
  );
  const coverage = Math.max(1, ...points.map((point) => Number(point.n_prompts)));
  const ticks = [
    ...new Set(
      bottom < 0 ? [bottom, 0, top] : [bottom, (bottom + top) / 2, top],
    ),
  ];
  const longSteps = points.some((point) => String(point.step).length > 10);
  const stride = Math.max(1, Math.floor((points.length - 1) / 4));
  const labelled = (index: number) =>
    index === 0 ||
    index === points.length - 1 ||
    (!longSteps &&
      width >= 480 &&
      index % stride === 0 &&
      points.length - 1 - index >= stride / 2);
  const range =
    onsetRange && String(onsetRange[0]) !== String(onsetRange[1])
      ? ([BigInt(onsetRange[0]), BigInt(onsetRange[1])] as const)
      : null;
  const shownBreaks = breaks
    .slice(0, 3)
    .map(([a, b]) => `${points[a].step} and ${points[b].step}`);
  return (
    <figure className="figure" aria-label="Feature curve">
      <div ref={ref}>
        {!points.length ? (
          <p className="unavailable">No observations for this feature.</p>
        ) : (
          <svg
            className="chart"
            width={width}
            height={HEIGHT + (tracks ? TRACKS : 0)}
            viewBox={`0 0 ${width} ${HEIGHT + (tracks ? TRACKS : 0)}`}
            role="img"
            aria-label="Feature mean and standard error over observed steps"
          >
            <title>
              {series.feature}: prompt-balanced means with missing gaps
            </title>
            <desc>
              Horizontal positions preserve exact training step differences.
              Jumps wider than the smallest step spacing break the curve. The
              table contains exact steps, uncertainty and sample counts.
            </desc>
            {breaks.map(([a, b]) => (
              <rect
                key={`${String(points[a].step)}-gap`}
                className="gap-band"
                x={x(a)}
                y={TOP}
                width={x(b) - x(a)}
                height={BOTTOM - TOP}
              >
                <title>
                  No observations between steps {points[a].step} and{" "}
                  {points[b].step}
                </title>
              </rect>
            ))}
            {range && (
              <rect
                className="onset-range"
                x={position(range[0])}
                y={TOP}
                width={Math.max(0, position(range[1]) - position(range[0]))}
                height={BOTTOM - TOP}
              >
                <title>
                  Event member onsets {String(range[0])} to {String(range[1])}
                </title>
              </rect>
            )}
            {ticks.map((value) => (
              <g key={value}>
                <line
                  x1={LEFT}
                  x2={right}
                  y1={y(value)}
                  y2={y(value)}
                  className="gridline"
                />
                <text x={LEFT - 10} y={y(value) + 4} textAnchor="end">
                  {text(value)}
                </text>
              </g>
            ))}
            {bands.map((path, index) => (
              <path key={index} d={path} className="error-band" />
            ))}
            {bars.map((index) => (
              <path
                key={index}
                className="error-bar"
                d={`M ${x(index)} ${y(points[index].mean! - points[index].se!)} V ${y(points[index].mean! + points[index].se!)}`}
              />
            ))}
            {groups.map((members) => (
              <path
                key={members[0]}
                className="mean-line"
                d={members
                  .map(
                    (index, order) =>
                      `${order ? "L" : "M"} ${x(index)} ${y(points[index].mean!)}`,
                  )
                  .join(" ")}
              />
            ))}
            {points.map((point, index) => (
              <g key={String(point.step)}>
                {point.mean !== null ? (
                  <circle
                    cx={x(index)}
                    cy={y(point.mean)}
                    r={point.se === null ? 3 : 1.75}
                    className={
                      point.se === null ? "uncertain-point" : "mean-point"
                    }
                  >
                    <title>
                      Step {point.step}: mean {text(point.mean)}, SE{" "}
                      {text(point.se)}
                    </title>
                  </circle>
                ) : (
                  <text
                    x={x(index)}
                    y={BOTTOM - 4}
                    textAnchor="middle"
                    className="missing-mark"
                  >
                    ×<title>Step {point.step}: no observation</title>
                  </text>
                )}
                {labelled(index) && (
                  <text
                    x={x(index)}
                    y={BOTTOM + 22 + (tracks ? TRACKS : 0)}
                    textAnchor={
                      index === 0
                        ? "start"
                        : index === points.length - 1
                          ? "end"
                          : "middle"
                    }
                  >
                    {String(point.step)}
                  </text>
                )}
                {tracks && (
                  <>
                    {persistence.get(String(point.step))?.retained === false ? (
                      <path
                        className="unretained"
                        d={`M ${x(index) - 2.5} ${BOTTOM + 11.5} l 5 5 m 0 -5 l -5 5`}
                      >
                        <title>Step {point.step}: not retained</title>
                      </path>
                    ) : (
                      persistence.has(String(point.step)) && (
                        <circle
                          cx={x(index)}
                          cy={BOTTOM + 14}
                          r="2.5"
                          className={
                            persistence.get(String(point.step))?.retained === true
                              ? "retained"
                              : "track-missing"
                          }
                        >
                          <title>
                            Step {point.step}: retained{" "}
                            {text(persistence.get(String(point.step))?.retained)}
                          </title>
                        </circle>
                      )
                    )}
                    <circle
                      cx={x(index)}
                      cy={BOTTOM + 32}
                      r="2.5"
                      className="coverage-point"
                      fillOpacity={Number(point.n_prompts) / coverage}
                    >
                      <title>
                        Step {point.step}: {String(point.n_prompts)} prompts
                      </title>
                    </circle>
                  </>
                )}
              </g>
            ))}
            {tracks && (
              <>
                {series.persistence.length > 0 && (
                  <text x={LEFT - 10} y={BOTTOM + 18} textAnchor="end">
                    Kept
                  </text>
                )}
                <text x={LEFT - 10} y={BOTTOM + 36} textAnchor="end">
                  Prompts
                </text>
              </>
            )}
            {onset >= 0 && (
              <g>
                <line
                  x1={x(onset)}
                  x2={x(onset)}
                  y1={TOP - 8}
                  y2={BOTTOM}
                  className="marker onset"
                />
                <text
                  className="marker-label"
                  x={x(onset) + (onsetSide === "end" ? -6 : 6)}
                  y={onsetSide === "start" && alertSide === "start" ? 12 : 26}
                  textAnchor={onsetSide}
                >
                  Onset {String(evidence.onset_step)}
                </text>
              </g>
            )}
            {alert >= 0 && (
              <g>
                <line
                  x1={x(alert)}
                  x2={x(alert)}
                  y1={TOP - 8}
                  y2={BOTTOM}
                  className="marker"
                />
                <text
                  className="marker-label strong"
                  x={x(alert) + (alertSide === "end" ? -6 : 6)}
                  y={onsetSide === "end" && alertSide === "end" ? 12 : 26}
                  textAnchor={alertSide}
                >
                  First alert {String(evidence.alert_step)}
                </text>
              </g>
            )}
          </svg>
        )}
      </div>
      <figcaption>
        <strong>Figure 1.</strong> Prompt-balanced mean ± 1 SE by training step,
        with exact prompts weighted equally. Gaps remain missing; an open point
        has no measured SE.
        {shownBreaks.length > 0 &&
          ` Breaks between steps ${shownBreaks.join(", ")}${breaks.length > 3 ? ` and ${breaks.length - 3} more` : ""} mark missing observations.`}
        {caption && ` ${caption}`}
        {tracks &&
          (series.persistence.length > 0
            ? ` In the Kept row, a filled point stayed at least ${text(series.persistence[0].effect_fraction)} of the alert-time effect from baseline, a cross did not, and grey has no measured SE.`
            : " No persistence observations follow an alert for this feature.")}
        {tracks &&
          " The Prompts row fills with the prompt count at each step, relative to this series' largest."}{" "}
        {points.length > 0 && (
          <button
            type="button"
            className="link"
            onClick={() => setTracks((value) => !value)}
          >
            {tracks
              ? "Hide persistence and prompts per step"
              : "Show persistence and prompts per step"}
          </button>
        )}
      </figcaption>
      <details className="more">
        <summary>Observation table and counts</summary>
        <DataTable
          caption="Feature observations"
          columns={[
            "step",
            "mean",
            "se",
            "n_rollouts",
            "n_prompts",
            "n_sampling_units",
            "retained",
          ]}
          data={points.map(
            (point) =>
              ({
                ...point,
                retained: persistence.get(String(point.step))?.retained ?? null,
              }) as Row,
          )}
        />
      </details>
    </figure>
  );
}
