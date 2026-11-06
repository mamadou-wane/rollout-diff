import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { EvidenceView } from "./EvidenceView";
import type { Evidence } from "./types";

const base: Evidence = {
  kind: "analysis",
  path: "report.json",
  sha256: "a".repeat(64),
  verification: "Validated recorded structure",
  source_link: {
    status: "not_loaded",
    reason: "Saved report has no verified response binding",
  },
  identity: null,
  summary: {},
  conditions: [],
  hypothesis: null,
  predictions: [],
  edit: null,
  source_diff: null,
  mechanism_status: "unresolved",
};

describe("saved report summaries", () => {
  it("shows every saved analysis run with status and analytical counts", () => {
    const value: Evidence = {
      ...base,
      summary: {
        runs: [
          {
            run: "saved-training-run",
            status: "analyzed",
            counts: {
              series: 91,
              screened_series: 85,
              sequential_alerts: 6,
              unscorable_series: 6,
              rollouts: 1234,
              observed_steps: 80,
              prompts: 50,
              events: 4,
              behavioral_events: 2,
              contextual_events: 1,
              other_events: 1,
            },
          },
          {
            run: "empty-run",
            status: "empty",
            counts: {
              rollouts: 0,
              observed_steps: 0,
              prompts: 0,
              series: 0,
              screened_series: 0,
              sequential_alerts: 0,
              unscorable_series: 0,
              events: 0,
              behavioral_events: 0,
              contextual_events: 0,
              other_events: 0,
            },
          },
        ],
      },
    };
    render(<EvidenceView value={value} onOpen={() => {}} />);
    const summary = screen.getByRole("region", {
      name: "Saved report summary",
    });
    expect(
      within(summary).getByRole("heading", { name: "saved-training-run" }),
    ).toBeInTheDocument();
    expect(within(summary).getByText(/1234 rollouts/)).toHaveTextContent(
      "1234 rollouts across 80 observed steps and 50 prompts. 6 sequential alerts, grouped into 4 events: 2 behavioral, 1 contextual, 1 other. 85 of 91 series were screened; 6 were unscorable.",
    );
    expect(within(summary).getByText(/^0 rollouts/)).toBeInTheDocument();
    expect(
      within(summary).getByRole("heading", { name: "empty-run" }),
    ).toBeInTheDocument();
    expect(within(summary).getByText("analyzed")).toBeInTheDocument();
    expect(within(summary).getByText("empty")).toBeInTheDocument();
    expect(
      within(summary).queryByText("No recorded evidence."),
    ).not.toBeInTheDocument();
    expect(
      within(summary).getByText(/Load the original rollout dump/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /inspect|curve/i }),
    ).not.toBeInTheDocument();
  });
  it("shows saved comparison windows and exact-prompt comparability", () => {
    const value: Evidence = {
      ...base,
      kind: "comparison",
      summary: {
        mode: "dumps",
        feature_count: 42,
        sides: [
          {
            side: "a",
            run: "first",
            window: ["9007199254740993", "9007199254740995"],
            observed_steps: ["9007199254740993", "9007199254740995"],
            n_rollouts: 120,
            n_prompts: 8,
            n_sampling_units: 4,
          },
          {
            side: "b",
            run: "second",
            window: [20, 30],
            observed_steps: [20, 30],
            n_rollouts: 90,
            n_prompts: 5,
            n_sampling_units: 3,
          },
        ],
        prompt_mix: {
          status: "no_shared_prompts",
          shared_prompts: 0,
          only_a_prompts: 8,
          only_b_prompts: 5,
          prompt_shift: [
            {
              descriptor: "prompt_chars",
              statistic: "ks",
              before: 12,
              after: 18,
              delta: 6,
              distance: 0.5,
              n_before: 8,
              n_after: 5,
              missing_before: 0,
              missing_after: 0,
            },
          ],
        },
      },
    };
    render(<EvidenceView value={value} onOpen={() => {}} />);
    const summary = screen.getByRole("region", {
      name: "Saved report summary",
    });
    expect(within(summary).getByText("42")).toBeInTheDocument();
    const sides = within(summary).getByRole("table", {
      name: "Saved comparison windows",
    });
    expect(
      within(sides).getByText("9007199254740993–9007199254740995"),
    ).toBeInTheDocument();
    expect(within(sides).getByText("first")).toBeInTheDocument();
    expect(within(sides).getByText("120")).toBeInTheDocument();
    expect(within(summary).getByText("no shared prompts")).toBeInTheDocument();
    expect(within(summary).getByText("shared prompts")).toBeInTheDocument();
    expect(
      within(summary).getByRole("table", {
        name: "Saved prompt descriptor shifts",
      }),
    ).toHaveTextContent("prompt_chars");
    expect(
      within(summary).getByText(/Load the original rollout dump/),
    ).toBeInTheDocument();
  });
});

it.each([
  [true, "Not applicable"],
  [false, "Unavailable"],
])(
  "displays null protected task reason when determinate is %s accurately",
  (determinate, expected) => {
    const value: Evidence = {
      ...base,
      kind: "replay",
      conditions: [
        {
          name: "control",
          task_determinate: determinate,
          task_reason: null,
          task_passed: null,
        },
      ],
    };
    render(<EvidenceView value={value} onOpen={() => {}} />);
    expect(screen.getByText("task reason").parentElement).toHaveTextContent(
      expected,
    );
    expect(screen.getByText("task passed").parentElement).toHaveTextContent(
      "Unavailable",
    );
    expect(value.conditions[0].task_reason).toBeNull();
  },
);

function cells(name: string) {
  return within(screen.getByRole("row", { name: new RegExp(`^${name}`) }))
    .getAllByRole("cell")
    .map((cell) => cell.textContent);
}
describe("empty state", () => {
  it("offers only the read-only open action", async () => {
    const onOpen = vi.fn();
    render(<EvidenceView value={null} onOpen={onOpen} />);
    expect(
      screen.getByRole("heading", { level: 1, name: "Saved evidence" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", {
        level: 2,
        name: "Choose a saved evidence file",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText(/No candidate execution/)).toBeInTheDocument();
    await userEvent.click(
      screen.getByRole("button", { name: "Open saved evidence" }),
    );
    expect(onOpen).toHaveBeenCalledOnce();
  });
});
describe("loaded evidence", () => {
  const intervention: Evidence = {
    ...base,
    kind: "intervention",
    path: "saved-intervention.json",
    identity: {
      run: "demo-a",
      sample_id: "116:412:0",
      step: "9007199254740993",
      prompt_id: "p412",
    },
    hypothesis: "<b>Literal</b> hypothesis",
    summary: {
      case_id: "aria-1",
      answer_stream_equal: null,
      historical_claims_authenticated: false,
    },
    conditions: [
      {
        name: "control",
        historical_status: "reported",
        historical_passed: 1,
        historical_evaluated: 1,
        historical_reason: null,
        task_status: "failed",
        task_determinate: true,
        task_applicable: 12,
        task_completed: 12,
        task_passed: 9,
        task_failed: 3,
        task_reason: null,
        discrepancy: true,
        answer_stream_sha256: "d".repeat(64),
        cleanup_verified: true,
      },
      {
        name: "variant",
        historical_status: "unavailable",
        historical_passed: null,
        historical_evaluated: null,
        historical_reason: "not_attempted",
        task_status: "unresolved",
        task_determinate: false,
        task_applicable: 12,
        task_completed: 4,
        task_passed: 4,
        task_failed: 0,
        task_reason: "execution_timeout",
        discrepancy: null,
        answer_stream_sha256: null,
        cleanup_verified: null,
      },
      {
        name: "third",
        historical_status: "reported",
        historical_passed: 0,
        historical_evaluated: 1,
        historical_reason: null,
        task_status: "passed",
        task_determinate: true,
        task_applicable: 2,
        task_completed: 2,
        task_passed: 2,
        task_failed: 0,
        task_reason: null,
        discrepancy: false,
        answer_stream_sha256: "e".repeat(64),
        cleanup_verified: false,
      },
    ],
    predictions: [
      {
        field: "exception_types",
        expected: [],
        observed: ["AssertionError"],
        status: "violated",
      },
      {
        field: "task_status",
        expected: "failed",
        observed: null,
        status: "unavailable",
      },
    ],
    edit: { old: "<script>old()</script>", new: "", occurrence: 1 },
    source_diff: "--- canonical-response\n+++ derived-response",
  };
  it("renders any number of conditions as columns", () => {
    render(<EvidenceView value={intervention} onOpen={() => {}} />);
    expect(
      screen.getByRole("heading", { level: 2, name: "Control and variant" }),
    ).toBeInTheDocument();
    const table = screen.getByRole("table", { name: "Condition evidence" });
    expect(
      within(table)
        .getAllByRole("columnheader")
        .map((header) => header.textContent),
    ).toEqual(["Field", "control", "variant", "third"]);
    expect(cells("Historical claim")).toEqual([
      "1 of 1 passed",
      "Unavailable: not_attempted.",
      "0 of 1 passed",
    ]);
    expect(cells("Saved protected task result")).toEqual([
      "9 passed, 3 failed, of 12 applicableDeterminate",
      "Unresolved: execution_timeout.Unknown / unavailable",
      "2 passed, 0 failed, of 2 applicableDeterminate",
    ]);
    expect(cells("Discrepancy")).toEqual(["Yes", "Unavailable", "No"]);
    expect(cells("Cleanup verified")).toEqual(["Yes", "Unavailable", "No"]);
    expect(cells("Answer stream SHA-256")).toEqual([
      "d".repeat(64),
      "Unavailable",
      "e".repeat(64),
    ]);
  });
  it("marks an indeterminate task result without presenting its counts", () => {
    render(<EvidenceView value={intervention} onOpen={() => {}} />);
    expect(screen.getByText("Unresolved: execution_timeout.")).toHaveClass(
      "unavailable",
    );
    expect(screen.queryByText(/4 passed/)).not.toBeInTheDocument();
    const recorded = screen
      .getByText("All recorded fields")
      .closest("details") as HTMLElement;
    expect(within(recorded).getAllByText("task completed")).toHaveLength(3);
    expect(within(recorded).getByText("execution_timeout")).toBeInTheDocument();
    expect(within(recorded).getByText("not_attempted")).toBeInTheDocument();
    expect(within(recorded).getByText("aria-1")).toBeInTheDocument();
    expect(within(recorded).getByText("answer stream equal")).toBeInTheDocument();
  });
  it("reads null values as Unavailable in the unavailable style", () => {
    render(
      <EvidenceView
        value={{
          ...base,
          kind: "replay",
          conditions: [
            {
              name: "recorded",
              historical_status: "unavailable",
              historical_passed: null,
              historical_evaluated: null,
              historical_reason: null,
              task_status: "unavailable",
              task_determinate: false,
              task_applicable: null,
              task_completed: null,
              task_passed: null,
              task_failed: null,
              task_reason: null,
              discrepancy: null,
              answer_stream_sha256: null,
              cleanup_verified: null,
            },
          ],
        }}
        onOpen={() => {}}
      />,
    );
    expect(
      screen.getByRole("heading", { level: 2, name: "Recorded condition" }),
    ).toBeInTheDocument();
    for (const name of [
      "Historical claim",
      "Discrepancy",
      "Cleanup verified",
      "Answer stream SHA-256",
    ]) {
      const row = screen.getByRole("row", { name: new RegExp(`^${name}`) });
      expect(within(row).getByText("Unavailable")).toHaveClass("unavailable");
    }
    const task = screen.getByRole("row", {
      name: /^Saved protected task result/,
    });
    expect(within(task).getByText("Unavailable")).toHaveClass("unavailable");
    expect(within(task).getByText("Unknown / unavailable")).toBeInTheDocument();
    expect(screen.getByText("Canonical identity unavailable.")).toHaveClass(
      "unavailable",
    );
    expect(screen.getByText("No recorded literal edit.")).toHaveClass(
      "unavailable",
    );
  });
  it("does not invent counts from a boolean historical result", () => {
    render(
      <EvidenceView
        value={{
          ...base,
          kind: "replay",
          conditions: [
            {
              name: "baseline",
              historical_status: "passed",
              historical_passed: true,
              task_status: "unavailable",
              task_determinate: false,
              task_reason: "Missing task counters",
            },
          ],
        }}
        onOpen={() => {}}
      />,
    );
    expect(cells("Historical claim")).toEqual(["Passed"]);
    expect(cells("Saved protected task result")).toEqual([
      "Unavailable: Missing task counters.Unknown / unavailable",
    ]);
    expect(screen.queryByText(/ of .* passed/)).not.toBeInTheDocument();
  });
  it("states the mechanism as unresolved", () => {
    render(<EvidenceView value={base} onOpen={() => {}} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "A recorded analysis. Mechanism unresolved.",
    );
    expect(screen.getByText(/does not establish/)).toHaveTextContent(
      "The saved evidence does not establish a causal mechanism.",
    );
  });
  it("shows source identity, the full saved hash and the verification verbatim", () => {
    render(<EvidenceView value={intervention} onOpen={() => {}} />);
    expect(screen.getByText("a".repeat(64)).tagName).toBe("CODE");
    expect(screen.getByText("Validated recorded structure")).toBeInTheDocument();
    expect(
      screen.getByText("Saved report has no verified response binding"),
    ).toBeInTheDocument();
    const source = screen.getByRole("region", { name: "Source" });
    expect(source).toHaveTextContent(
      "Status: not loaded. Run demo-a, sample 116:412:0, step 9007199254740993, prompt p412.",
    );
  });
  it("keeps the recorded hypothesis, edit and diff literal", () => {
    const { container } = render(
      <EvidenceView value={intervention} onOpen={() => {}} />,
    );
    expect(screen.getByText("<b>Literal</b> hypothesis")).toBeInTheDocument();
    expect(screen.getByText("Saved text, shown literally.")).toBeInTheDocument();
    const original = screen.getByRole("region", { name: "Original text" });
    expect(within(original).getByText("<script>old()</script>").tagName).toBe(
      "PRE",
    );
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("b")).toBeNull();
    expect(
      within(screen.getByRole("region", { name: "Proposed text" })).getByText(
        "Empty text.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText(/occurrence 1/)).toBeInTheDocument();
    expect(
      screen.getByText(
        "Literal evidence. Nothing can be executed, approved or replayed from this page.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("Verified source diff")).toBeInTheDocument();
    expect(screen.getByText(/canonical-response/).tagName).toBe("PRE");
  });
  it("lists predictions with labelled statuses and explicit gaps", () => {
    render(<EvidenceView value={intervention} onOpen={() => {}} />);
    const table = screen.getByRole("table", { name: "Recorded predictions" });
    const first = within(table).getByRole("row", { name: /^exception_types/ });
    expect(first).toHaveTextContent('[]["AssertionError"]violated');
    const second = within(table).getByRole("row", { name: /^task_status/ });
    expect(within(second).getAllByText(/unavailable/i)[0]).toHaveClass(
      "unavailable",
    );
  });
  it("offers no execute, approve or replay control and keeps the open link", () => {
    const { rerender } = render(
      <EvidenceView value={intervention} onOpen={() => {}} />,
    );
    const forbidden = { name: /execute|approve|replay/i };
    expect(screen.queryByRole("button", forbidden)).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Open saved evidence" }),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("button")).toHaveLength(1);
    rerender(<EvidenceView value={null} onOpen={() => {}} />);
    expect(screen.queryByRole("button", forbidden)).not.toBeInTheDocument();
  });
});
