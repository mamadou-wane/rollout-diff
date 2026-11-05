import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
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
            counts: { rollouts: 0, observed_steps: 0, prompts: 0, events: 0 },
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
    expect(within(summary).getByText("1234")).toBeInTheDocument();
    expect(within(summary).getAllByText("sequential alerts")).toHaveLength(2);
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
