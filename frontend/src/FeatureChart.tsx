import { DataTable, text } from "./common";
import type { Row, Series, Step } from "./types";
export function FeatureChart({ series }: { series: Series }) {
  const points = series.points;
  const valid = points.filter(
    (point) => point.mean !== null && Number.isFinite(point.mean),
  );
  const bounds = valid.flatMap((point) => [
    point.mean! - (point.se ?? 0),
    point.mean! + (point.se ?? 0),
  ]);
  const min = Math.min(0, ...bounds),
    max = Math.max(1, ...bounds),
    span = max - min || 1;
  const steps = points.map((point) => BigInt(String(point.step)));
  const first = steps[0] ?? 0n,
    last = steps[steps.length - 1] ?? first;
  const x = (index: number) =>
    56 +
    (last === first
      ? 0.5
      : Number(((steps[index] - first) * 1000000n) / (last - first)) /
        1000000) *
      650;
  const y = (value: number) => 210 - ((value - min) / span) * 172;
  const segments: string[] = [];
  let segment = "";
  points.forEach((point, index) => {
    if (index > 0 && steps[index] - steps[index - 1] > 1n) {
      if (segment) segments.push(segment);
      segment = "";
    }
    if (point.mean === null) {
      if (segment) segments.push(segment);
      segment = "";
    } else segment += `${segment ? " L" : "M"} ${x(index)} ${y(point.mean)}`;
  });
  if (segment) segments.push(segment);
  const evidence = series.evidence ?? {};
  const markers = [
    ["Onset", evidence.onset_step],
    ["Alert", evidence.alert_step],
  ] as const;
  const persistence = new Map(
    series.persistence.map((item) => [String(item.step), item]),
  );
  const same = (left: unknown, right: Step) =>
    left !== null && left !== undefined && String(left) === String(right);
  return (
    <section className="chart-section" aria-label="Feature curve">
      <div className="section-heading">
        <h3>Feature over observed steps</h3>
        <span className="muted">Mean ± 1 SE</span>
      </div>
      {!points.length ? (
        <p className="empty">No observations for this feature.</p>
      ) : (
        <svg
          viewBox="0 0 750 270"
          role="img"
          aria-label="Feature mean and standard error over observed steps"
        >
          <title>
            {series.feature}: prompt-balanced means with missing gaps
          </title>
          <desc>
            Horizontal positions preserve exact training step differences.
            Nonadjacent steps break the curve. The table contains exact steps,
            uncertainty and sample counts.
          </desc>
          {[0, 0.5, 1].map((t) => {
            const value = min + t * span;
            return (
              <g key={t}>
                <line
                  x1="56"
                  x2="706"
                  y1={y(value)}
                  y2={y(value)}
                  className="gridline"
                />
                <text x="46" y={y(value) + 4} textAnchor="end">
                  {text(value)}
                </text>
              </g>
            );
          })}
          {segments.map((path, index) => (
            <path key={index} d={path} className="mean-line" />
          ))}
          {points.map((point, index) => (
            <g key={String(point.step)}>
              {point.mean !== null ? (
                <>
                  {point.se !== null && (
                    <path
                      className="error-bar"
                      d={`M ${x(index)} ${y(point.mean - point.se)} V ${y(point.mean + point.se)} M ${x(index) - 4} ${y(point.mean - point.se)} H ${x(index) + 4} M ${x(index) - 4} ${y(point.mean + point.se)} H ${x(index) + 4}`}
                    />
                  )}
                  <circle
                    cx={x(index)}
                    cy={y(point.mean)}
                    r="3.5"
                    className={
                      point.se === null ? "uncertain-point" : "mean-point"
                    }
                  >
                    <title>
                      Step {point.step}: mean {text(point.mean)}, SE{" "}
                      {text(point.se)}
                    </title>
                  </circle>
                </>
              ) : (
                <text x={x(index)} y="225" textAnchor="middle">
                  ×
                </text>
              )}
              {(index === 0 ||
                index === points.length - 1 ||
                (points.length < 9 && String(point.step).length < 11)) && (
                <text
                  x={x(index)}
                  y="246"
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
              {persistence.has(String(point.step)) && (
                <rect
                  x={x(index) - 3}
                  y="258"
                  width="6"
                  height="5"
                  className={
                    persistence.get(String(point.step))?.retained === true
                      ? "retained"
                      : "unretained"
                  }
                />
              )}
            </g>
          ))}
          {markers.map(([name, step]) => {
            const index = points.findIndex((point) => same(step, point.step));
            return index < 0 ? null : (
              <g key={name}>
                <line
                  x1={x(index)}
                  x2={x(index)}
                  y1="28"
                  y2="222"
                  className={`marker ${name.toLowerCase()}`}
                />
                <text
                  x={Math.min(
                    String(step).length > 10 ? 520 : 660,
                    x(index) + 4,
                  )}
                  y={name === "Onset" ? 16 : 29}
                >
                  {name} {String(step)}
                </text>
              </g>
            );
          })}
        </svg>
      )}
      <p className="note">
        Exact prompts receive equal weight. Positions follow training steps.
        Gaps remain missing; unavailable SE is an open point. Persistence marks
        use recorded retained status.
      </p>
      <details>
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
    </section>
  );
}
