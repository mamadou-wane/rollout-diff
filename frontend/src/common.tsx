import { useEffect, useRef, type ReactNode } from "react";
import type { Row } from "./types";
export function text(value: unknown): string {
  if (value === null || value === undefined) return "Unavailable";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number")
    return Number.isInteger(value)
      ? String(value)
      : String(Number(value.toPrecision(4)));
  return String(value);
}
export function known(value: unknown) {
  return value !== null && value !== undefined;
}
export function plural(count: unknown, noun: string) {
  return `${text(count)} ${noun}${String(count) === "1" ? "" : "s"}`;
}
/** Display equality with a small rounding tolerance. */
export function sameValue(a: number, b: number) {
  return Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b));
}
export function figure(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return text(value);
  if (Math.abs(value) >= 100) return String(Math.round(value));
  // Rounding must not turn a prevalence below one into exactly one.
  let digits = 3;
  while (
    Number(value.toPrecision(digits)) === 1 &&
    !sameValue(value, 1) &&
    digits < 15
  )
    digits++;
  return String(Number(value.toPrecision(digits)));
}
/** Round two values for display, keeping unequal values visibly distinct. */
export function figures(a: number, b: number): [string, string] {
  if (sameValue(a - b, 0)) return [figure(a), figure(a)];
  if (figure(a) !== figure(b)) return [figure(a), figure(b)];
  let digits = 4;
  while (Number(a.toPrecision(digits)) === Number(b.toPrecision(digits)))
    digits++;
  return [
    String(Number(a.toPrecision(digits))),
    String(Number(b.toPrecision(digits))),
  ];
}
export const PROMPT_SHIFT_COLUMNS = [
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
];
export function runCounts(counts: Row) {
  const count = (key: string) => text(counts[key]);
  return `${count("rollouts")} rollouts across ${count("observed_steps")} observed steps and ${count("prompts")} prompts. ${count("sequential_alerts")} sequential alerts, grouped into ${count("events")} events: ${count("behavioral_events")} behavioral, ${count("contextual_events")} contextual, ${count("other_events")} other.`;
}
export function signed(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return text(value);
  const magnitude = text(Math.abs(value));
  return value > 0 ? `+${magnitude}` : value < 0 ? `−${magnitude}` : magnitude;
}
export function label(value: string) {
  return value.replaceAll("_", " ");
}
export function row(value: unknown): Row {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Row)
    : {};
}
export function rows(value: unknown): Row[] {
  return Array.isArray(value) ? value.map(row) : [];
}
export function ErrorNotice({
  message,
  retry,
}: {
  message: string | null;
  retry?: () => void;
}) {
  return message ? (
    <div role="alert" className="notice">
      {message}
      {retry && (
        <button type="button" className="link" onClick={retry}>
          Retry
        </button>
      )}
    </div>
  ) : null;
}
export function Fields({ data, fields }: { data: Row; fields?: string[] }) {
  const keys =
    fields ??
    Object.keys(data).filter(
      (key) =>
        data[key] === null ||
        ["string", "number", "boolean"].includes(typeof data[key]),
    );
  return keys.length ? (
    <dl className="fields">
      {keys.map((key) => (
        <div key={key}>
          <dt>{label(key)}</dt>
          <dd>{text(data[key])}</dd>
        </div>
      ))}
    </dl>
  ) : (
    <p className="muted">No recorded evidence.</p>
  );
}
export function DataTable({
  data,
  columns,
  caption,
}: {
  data: Row[];
  columns: string[];
  caption: string;
}) {
  if (!data.length)
    return <p className="muted">No {caption.toLowerCase()} available.</p>;
  return (
    <div className="table-scroll" tabIndex={0}>
      <table>
        <caption>{caption}</caption>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{label(column)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((item, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>{text(item[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function Modal({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    ref.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeRef.current();
      }
      if (event.key === "Tab") {
        const elements = Array.from(
          ref.current?.querySelectorAll<HTMLElement>(
            'button:not(:disabled),input:not(:disabled),select:not(:disabled),a[href],summary,[tabindex="0"]',
          ) ?? [],
        );
        if (!elements.length) {
          event.preventDefault();
          return;
        }
        const first = elements[0],
          last = elements[elements.length - 1],
          active = document.activeElement,
          outside = !ref.current?.contains(active);
        if (
          event.shiftKey &&
          (outside || active === first || active === ref.current)
        ) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && (outside || active === last)) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    document.body.classList.add("modal-open");
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.classList.remove("modal-open");
      // A trigger that unmounted or was disabled cannot take focus back.
      (previous?.isConnected && previous !== document.body
        ? previous
        : document.querySelector<HTMLElement>('[role="tabpanel"]')
      )?.focus();
    };
  }, []);
  return (
    <div className="scrim">
      <div
        ref={ref}
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
      >
        {children}
      </div>
    </div>
  );
}
