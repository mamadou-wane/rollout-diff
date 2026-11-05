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
    <div role="alert" className="notice error">
      {message}
      {retry && <button onClick={retry}>Retry</button>}
    </div>
  ) : null;
}
export function Badge({
  children,
  warn = false,
}: {
  children: ReactNode;
  warn?: boolean;
}) {
  return <span className={`badge ${warn ? "warning" : ""}`}>{children}</span>;
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
  drawer = false,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  drawer?: boolean;
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
          last = elements[elements.length - 1];
        if (
          event.shiftKey &&
          (document.activeElement === first ||
            document.activeElement === ref.current)
        ) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
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
      previous?.focus();
    };
  }, []);
  return (
    <div
      className="scrim"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={ref}
        className={`modal ${drawer ? "drawer" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
      >
        <div className="modal-heading">
          <h2>{title}</h2>
          <button onClick={onClose} aria-label={`Close ${title.toLowerCase()}`}>
            Close
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
