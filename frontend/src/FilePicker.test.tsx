import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { FilePicker } from "./FilePicker";

const ROOT = {
  path: ".",
  entries: [
    { name: "runs", path: "runs", kind: "directory" },
    { name: "a.parquet", path: "a.parquet", kind: "trl" },
    { name: "b.jsonl", path: "b.jsonl", kind: "verl" },
    { name: "saved.json", path: "saved.json", kind: "evidence" },
  ],
};
function serve(handler?: (url: string) => unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string) => {
      const url = String(input);
      const override = handler?.(url);
      if (override instanceof Promise || override instanceof Response)
        return override;
      const value =
        override ??
        (url === "/api/files?path=runs"
          ? { path: "runs", entries: [] }
          : url.startsWith("/api/files")
            ? ROOT
            : { id: "created" });
      return new Response(JSON.stringify(value), { status: 200 });
    }),
  );
}
function posted(url: string) {
  const call = vi.mocked(fetch).mock.calls.find(([item]) => item === url);
  return JSON.parse(String(call?.[1]?.body));
}
function open(mode: "run" | "evidence", runId?: string) {
  const handlers = {
    onClose: vi.fn(),
    onRun: vi.fn(),
    onEvidence: vi.fn(),
  };
  const view = render(<FilePicker mode={mode} runId={runId} {...handlers} />);
  return { ...handlers, ...view };
}
beforeEach(() => serve());

describe("run mode", () => {
  it("loads a run with an explicit format and name", async () => {
    const { onRun, onClose, onEvidence } = open("run");
    const user = userEvent.setup();
    expect(
      screen.getByRole("heading", { name: "Load a rollout run" }),
    ).toBeInTheDocument();
    await user.click(
      await screen.findByRole("button", { name: "Select a.parquet" }),
    );
    expect(screen.getByLabelText("Input format")).toHaveValue("trl");
    await user.selectOptions(screen.getByLabelText("Input format"), "verl");
    expect(screen.getByLabelText("Reward column (optional)")).toBeDisabled();
    await user.type(screen.getByLabelText("Run name"), "first");
    await user.click(screen.getByRole("button", { name: "Load selected run" }));
    await waitFor(() => expect(onRun).toHaveBeenCalledOnce());
    expect(posted("/api/runs")).toEqual({
      path: "a.parquet",
      format: "verl",
      run: "first",
      reward_column: null,
    });
    expect(onEvidence).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalledOnce();
  });
  it("infers the format from a selected file and sends the reward column", async () => {
    open("run");
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Select b.jsonl" }),
    );
    expect(screen.getByLabelText("Input format")).toHaveValue("verl");
    await user.click(screen.getByRole("button", { name: "Select a.parquet" }));
    expect(screen.getByLabelText("Input format")).toHaveValue("trl");
    expect(
      screen.getByRole("button", { name: "Select a.parquet" }),
    ).toHaveAttribute("aria-pressed", "true");
    await user.type(
      screen.getByLabelText("Reward column (optional)"),
      "reward",
    );
    await user.type(screen.getByLabelText("Run name"), "first");
    await user.click(screen.getByRole("button", { name: "Load selected run" }));
    await waitFor(() =>
      expect(posted("/api/runs").reward_column).toBe("reward"),
    );
  });
  it("chooses the current directory with Use this directory", async () => {
    const { onRun } = open("run");
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Use this directory" }),
    );
    expect(screen.getByText(/^Selected:/)).toHaveTextContent("Selected: .");
    await user.selectOptions(screen.getByLabelText("Input format"), "verl");
    await user.type(screen.getByLabelText("Run name"), "whole");
    await user.click(screen.getByRole("button", { name: "Load selected run" }));
    await waitFor(() => expect(onRun).toHaveBeenCalledOnce());
    expect(posted("/api/runs")).toEqual({
      path: ".",
      format: "verl",
      run: "whole",
      reward_column: null,
    });
  });
  it("selects a directory row and requires a selection and a name", async () => {
    open("run");
    const user = userEvent.setup();
    const submit = screen.getByRole("button", { name: "Load selected run" });
    expect(submit).toBeDisabled();
    expect(screen.getByText(/^Selected:/)).toHaveTextContent(
      "Choose a file or directory",
    );
    await user.click(
      await screen.findByRole("button", { name: "Select directory runs" }),
    );
    expect(screen.getByText(/^Selected:/)).toHaveTextContent("Selected: runs");
    expect(submit).toBeDisabled();
    await user.type(screen.getByLabelText("Run name"), "  ");
    expect(submit).toBeDisabled();
    await user.type(screen.getByLabelText("Run name"), "named");
    expect(submit).toBeEnabled();
  });
  it("disables evidence files and keeps run inputs selectable", async () => {
    open("run");
    expect(
      await screen.findByRole("button", { name: "Select saved.json" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Select a.parquet" }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Select b.jsonl" }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Select directory runs" }),
    ).toBeEnabled();
  });
});

describe("evidence mode", () => {
  it("disables run files and offers no directory selection", async () => {
    open("evidence");
    expect(
      await screen.findByRole("button", { name: "Select a.parquet" }),
    ).toBeDisabled();
    expect(screen.getByRole("button", { name: "Select b.jsonl" })).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Select saved.json" }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Open directory runs" }),
    ).toBeEnabled();
    expect(
      screen.queryByRole("button", { name: "Select directory runs" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Use this directory" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Run name")).not.toBeInTheDocument();
  });
  it("reads selected evidence bound to the loaded run", async () => {
    const { onEvidence, onClose, onRun } = open("evidence", "run-1");
    const user = userEvent.setup();
    expect(
      screen.getByRole("heading", { name: "Open saved evidence" }),
    ).toBeInTheDocument();
    await user.click(
      await screen.findByRole("button", { name: "Select saved.json" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Read selected evidence" }),
    );
    await waitFor(() => expect(onEvidence).toHaveBeenCalledOnce());
    expect(posted("/api/evidence")).toEqual({
      path: "saved.json",
      run_id: "run-1",
    });
    expect(onRun).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalledOnce();
  });
});

describe("navigation and failure", () => {
  it("opens a directory, clears the selection and returns to the parent", async () => {
    open("run");
    const user = userEvent.setup();
    expect(
      screen.getByRole("button", { name: "Parent directory" }),
    ).toBeDisabled();
    await user.click(
      await screen.findByRole("button", { name: "Select b.jsonl" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Open directory runs" }),
    );
    expect(
      await screen.findByText("No supported inputs in this directory."),
    ).toBeInTheDocument();
    expect(screen.getByText(/^Selected:/)).toHaveTextContent(
      "Choose a file or directory",
    );
    await user.click(screen.getByRole("button", { name: "Parent directory" }));
    expect(
      await screen.findByRole("button", { name: "Select b.jsonl" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Parent directory" }),
    ).toBeDisabled();
  });
  it("shows a failed read and allows another attempt", async () => {
    serve((url) =>
      url === "/api/evidence"
        ? new Response(JSON.stringify({ detail: "invalid evidence JSON" }), {
            status: 400,
          })
        : undefined,
    );
    const { onClose, onEvidence } = open("evidence");
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Select saved.json" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Read selected evidence" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "invalid evidence JSON",
    );
    expect(onEvidence).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: "Read selected evidence" }),
    ).toBeEnabled();
  });
  it("closes from the heading link and from Cancel", async () => {
    const { onClose } = open("run");
    const user = userEvent.setup();
    await user.click(
      screen.getByRole("button", { name: "Close load a rollout run" }),
    );
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onClose).toHaveBeenCalledTimes(2);
  });
  it("aborts a pending request on unmount and ignores its result", async () => {
    let finish: (response: Response) => void = () => {};
    const pending = new Promise<Response>((resolve) => {
      finish = resolve;
    });
    serve((url) => (url === "/api/evidence" ? pending : undefined));
    const { onEvidence, onClose, unmount } = open("evidence");
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Select saved.json" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Read selected evidence" }),
    );
    const call = vi
      .mocked(fetch)
      .mock.calls.find(([url]) => url === "/api/evidence");
    expect(call?.[1]?.signal?.aborted).toBe(false);
    unmount();
    expect(call?.[1]?.signal?.aborted).toBe(true);
    finish(new Response(JSON.stringify({ kind: "replay" })));
    await pending;
    await Promise.resolve();
    expect(onEvidence).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });
});
