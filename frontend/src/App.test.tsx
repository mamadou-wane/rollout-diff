import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { initializeToken } from "./api";
import { FeatureChart } from "./FeatureChart";

const event = {
  rank: 7,
  representative_feature: "call:check",
  ranking_status: "behavioral",
  onset_start: 3,
  onset_end: 3,
  onset_step: 3,
  alert_step: 5,
  n_members: 1,
  ranking: { reward_availability: "available" },
  members: [
    {
      feature: "call:check",
      kind: "binary",
      status: "alerted",
      evidence: {
        tier: "structural",
        replication: { status: "replicated" },
        persistence: { status: "available", fraction: 1 },
        reward_association: { status: "available" },
      },
      exemplars: [
        {
          status: "selected",
          sample_id: "sample-1",
          step: 5,
          prompt_id: "p1",
          reward: 1,
        },
      ],
    },
  ],
};
const run = {
  id: "r1",
  run: "fixture",
  status: "ready",
  counts: {
    rollouts: 30,
    observed_steps: 3,
    prompts: 2,
    series: 4,
    screened_series: 4,
    sequential_alerts: 1,
    unscorable_series: 0,
    events: 1,
    behavioral_events: 1,
    contextual_events: 0,
    other_events: 0,
  },
  source: {
    path: "run.jsonl",
    canonical_sha256: "a".repeat(64),
    format: "trl",
    files: [],
  },
};
const series = {
  feature: "call:check",
  kind: "binary",
  tier: "structural",
  evidence: { onset_step: 3, alert_step: 5 },
  points: [
    {
      step: 1,
      mean: 0,
      se: 0,
      n_rollouts: 4,
      n_prompts: 2,
      n_sampling_units: 2,
    },
    {
      step: 2,
      mean: null,
      se: null,
      n_rollouts: 0,
      n_prompts: 0,
      n_sampling_units: 0,
    },
    {
      step: 3,
      mean: 1,
      se: null,
      n_rollouts: 2,
      n_prompts: 1,
      n_sampling_units: 1,
    },
  ],
  persistence: [],
  prompt_values: [],
  prompt_shifts: [],
};
function serve(handler?: (url: string, init?: RequestInit) => unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string, init?: RequestInit) => {
      const url = String(input);
      const override = handler?.(url, init);
      if (override instanceof Promise) return override;
      if (override instanceof Response) return override;
      const value =
        override ??
        (url === "/api/session"
          ? { root_label: "fixtures", formats: ["trl", "verl"] }
          : url === "/api/runs"
            ? { runs: [run] }
            : url.includes("/features?")
              ? {
                  features: [
                    {
                      feature: "call:check",
                      tier: "structural",
                      kind: "binary",
                    },
                  ],
                }
              : url.includes("/series?")
                ? series
                : url.includes("/events/7")
                  ? { event, evidence: [], prompt_shifts: [] }
                  : url.includes("/events?")
                    ? { total: 1, events: [event] }
                    : url.includes("/inspect")
                      ? {
                          citation: {
                            run: "fixture",
                            sample_id: "sample-1",
                            step: 5,
                            response:
                              "<script>alert(1)</script>\nraw exact text",
                            comparison: {
                              status: "available",
                              counterpart: {
                                sample_id: "absent-1",
                                response: "absent response",
                              },
                            },
                          },
                          source: run.source,
                          response_sha256: "b".repeat(64),
                        }
                      : url.includes("/files?")
                        ? {
                            path: ".",
                            entries: [
                              {
                                name: "run.jsonl",
                                path: "run.jsonl",
                                kind: "trl",
                              },
                            ],
                          }
                        : run);
      return new Response(JSON.stringify(value), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
}
beforeEach(() => {
  sessionStorage.setItem("rolloutdiff.session.v1", "secret");
  serve();
});
describe("local investigation", () => {
  it("moves a fragment token to tab storage and removes it from history", () => {
    window.history.replaceState(null, "", "/#token=private-token");
    expect(initializeToken()).toBe("private-token");
    expect(window.location.hash).toBe("");
    expect(sessionStorage.getItem("rolloutdiff.session.v1")).toBe(
      "private-token",
    );
  });
  it("opens literal verified response text in two clicks and includes matched absent evidence", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: /Event 7/ }));
    await user.click(
      await screen.findByRole("button", { name: /Read response sample-1/ }),
    );
    const drawer = await screen.findByRole("dialog", {
      name: "Original response",
    });
    expect(drawer.querySelector("pre.response")?.textContent).toBe(
      "<script>alert(1)</script>\nraw exact text",
    );
    expect(drawer.querySelector("script")).toBeNull();
    expect(within(drawer).getByText("b".repeat(64))).toBeInTheDocument();
    expect(within(drawer).getByText("absent response")).toBeInTheDocument();
    const calls = vi.mocked(fetch).mock.calls;
    expect(
      calls.every(
        ([, init]) =>
          new Headers(init?.headers).get("Authorization") === "Bearer secret",
      ),
    ).toBe(true);
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
  it("filters and pages events without changing reported ranks", async () => {
    serve((url) =>
      url.includes("/events?") ? { total: 61, events: [event] } : undefined,
    );
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole("button", { name: /Event 7/ });
    await user.type(screen.getByLabelText("Search events"), "check");
    await user.click(
      within(screen.getByRole("group", { name: "Evidence class" })).getByRole(
        "button",
        { name: "Behavioral" },
      ),
    );
    await user.click(screen.getByRole("button", { name: "Next events" }));
    await waitFor(() =>
      expect(
        vi
          .mocked(fetch)
          .mock.calls.some(
            ([url]) =>
              String(url).includes("offset=50") &&
              String(url).includes("q=check") &&
              String(url).includes("kind=behavioral"),
          ),
      ).toBe(true),
    );
    expect(screen.getByRole("button", { name: /Event 7/ })).toBeInTheDocument();
  });
  it("loads a selected rooted file with explicit format and run name", async () => {
    serve((url, init) =>
      url === "/api/runs" && init?.method === "POST"
        ? { ...run, id: "r2", run: "second" }
        : undefined,
    );
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "Load run" }));
    await user.click(
      await screen.findByRole("button", { name: "Select run.jsonl" }),
    );
    await user.type(screen.getByLabelText("Run name"), "second");
    await user.click(screen.getByRole("button", { name: "Load selected run" }));
    await waitFor(() =>
      expect(
        vi
          .mocked(fetch)
          .mock.calls.some(
            ([url, init]) =>
              url === "/api/runs" &&
              init?.method === "POST" &&
              JSON.parse(String(init.body)).run === "second",
          ),
      ).toBe(true),
    );
    await user.click(
      await screen.findByRole("button", { name: "Run second, ready" }),
    );
    expect(screen.getByLabelText("Loaded run")).toHaveDisplayValue(
      "second, ready",
    );
  });
  it("opens the first event and states shared-prompt values in its headline", async () => {
    const headline = { before: 0.02, after: 0.41, n_prompts: 6 };
    const evidence = {
      feature: "call:check",
      promptmix_status: "available",
      prompt_before_start: 1,
      prompt_before_end: 2,
      prompt_after_start: 3,
      prompt_after_end: 5,
    };
    serve((url) =>
      url.includes("/events/7")
        ? {
            event,
            evidence: [evidence],
            prompt_shifts: [],
            headlines: { "call:check": headline },
          }
        : url.includes("/events?")
          ? { total: 1, events: [event], headlines: { "call:check": headline } }
          : undefined,
    );
    render(<App />);
    expect(
      await screen.findByRole("heading", {
        name: "call:check rose from 0.02 to 0.41",
        level: 1,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/baseline steps 1 to 2 against onset to first alert/),
    ).toBeInTheDocument();
    expect(screen.getByText("0.02 → 0.41")).toBeInTheDocument();
  });
  it("shows no values in the headline without shared prompts", async () => {
    serve((url) =>
      url.includes("/events/7")
        ? {
            event,
            evidence: [
              { feature: "call:check", promptmix_status: "no_shared_prompts" },
            ],
            prompt_shifts: [],
            headlines: {},
          }
        : undefined,
    );
    render(<App />);
    expect(
      await screen.findByRole("heading", {
        name: "call:check has no prompts shared across its windows",
        level: 1,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Before and after values are unavailable/),
    ).toBeInTheDocument();
  });
  it("reads floating-point noise as no change and scopes it to shared prompts", async () => {
    const headline = {
      before: 0.20000000000000004,
      after: 0.19999999999999998,
      n_prompts: 3,
    };
    serve((url) =>
      url.includes("/events/7")
        ? {
            event: {
              ...event,
              members: [{ ...event.members[0], direction: "up" }],
            },
            evidence: [{ feature: "call:check" }],
            prompt_shifts: [],
            headlines: { "call:check": headline },
          }
        : undefined,
    );
    render(<App />);
    expect(
      await screen.findByRole("heading", {
        name: "call:check stayed at 0.2 across shared prompts",
        level: 1,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/shared prompts show no net change/),
    ).toBeInTheDocument();
  });
  it("keeps close values distinct and never rounds a prevalence up to one", async () => {
    const { figure, figures } = await import("./common");
    expect(figure(0.9996)).toBe("0.9996");
    expect(figure(1)).toBe("1");
    expect(figure(0.9999999999999999)).toBe("1");
    expect(figure(1.0000000000000002)).toBe("1");
    expect(figures(12345.2, 12345.4)).toEqual(["12345.2", "12345.4"]);
    expect(figures(0.02, 0.41)).toEqual(["0.02", "0.41"]);
    expect(figures(0.20000000000000004, 0.19999999999999998)).toEqual([
      "0.2",
      "0.2",
    ]);
    expect(figures(0, 1.85e-17)).toEqual(["0", "0"]);
  });
  it("reports partial persistence with its numbers", async () => {
    serve((url) =>
      url.includes("/events/7")
        ? {
            event: {
              ...event,
              members: [
                {
                  ...event.members[0],
                  evidence: {
                    ...event.members[0].evidence,
                    persistence: {
                      status: "partial",
                      n_retained: 4,
                      n_evaluable: 5,
                      n_unavailable: 1,
                      coverage: 0.83,
                    },
                  },
                },
              ],
            },
            evidence: [
              {
                feature: "call:check",
                retrospective_status: "no_change",
              },
            ],
            prompt_shifts: [],
          }
        : undefined,
    );
    render(<App />);
    expect(
      await screen.findByText(
        "Retained at 4 of 5 evaluable steps after the first alert. Coverage 0.83; 1 step without a measured SE.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText("The retrospective detector selected no change point."),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Retrospective onset/)).not.toBeInTheDocument();
  });
  it("discards a citation that resolves after the user changed tabs", async () => {
    let finish: (response: Response) => void = () => {};
    const pending = new Promise<Response>((resolve) => {
      finish = resolve;
    });
    serve((url) => (url.includes("/inspect") ? pending : undefined));
    const user = userEvent.setup();
    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: /^Read response sample-1/ }),
    );
    await user.click(screen.getByRole("tab", { name: "Compare" }));
    await act(async () =>
      finish(
        new Response(
          JSON.stringify({
            citation: { sample_id: "sample-1", step: 5, response: "late" },
            source: run.source,
            response_sha256: "b".repeat(64),
          }),
        ),
      ),
    );
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
  it("closes the run menu on Escape and returns focus to its trigger", async () => {
    const user = userEvent.setup();
    render(<App />);
    const trigger = await screen.findByRole("button", {
      name: "Run fixture, ready",
    });
    await user.click(trigger);
    const unload = await screen.findByRole("button", { name: "Unload run" });
    unload.focus();
    await user.keyboard("{Escape}");
    expect(
      screen.queryByRole("region", { name: "Run summary" }),
    ).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });
  it("unloads a run from the run menu", async () => {
    serve((url, init) =>
      url === "/api/runs/r1" && init?.method === "DELETE"
        ? new Response(null, { status: 204 })
        : undefined,
    );
    const user = userEvent.setup();
    render(<App />);
    await user.click(
      await screen.findByRole("button", { name: "Run fixture, ready" }),
    );
    await user.click(await screen.findByRole("button", { name: "Unload run" }));
    expect(
      await screen.findByRole("heading", { name: "Load a rollout run to begin" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Load run" })).toHaveFocus();
  });
  it("states when no persistence observations follow an alert", async () => {
    const user = userEvent.setup();
    render(<FeatureChart series={series} />);
    await user.click(
      screen.getByRole("button", {
        name: "Show persistence and prompts per step",
      }),
    );
    expect(
      screen.getByText(/No persistence observations follow an alert/),
    ).toBeInTheDocument();
    expect(screen.queryByText("Kept")).not.toBeInTheDocument();
  });
  it("reconciles a browsed feature with its alert direction", async () => {
    serve((url) =>
      url.includes("/series?")
        ? {
            ...series,
            direction: "up",
            headline: { before: 0.4, after: 0.1, n_prompts: 1 },
          }
        : url.includes("/events?")
          ? { total: 0, events: [] }
          : undefined,
    );
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByText("Browse all features"));
    await user.click(
      await screen.findByRole("button", { name: "Feature call:check" }),
    );
    expect(
      await screen.findByText(/the shared-prompt mean moved the other way/),
    ).toHaveTextContent("Sequential streak began at step 3; first alert at step 5.");
  });
  it("explains a run without events without calling it a negative result", async () => {
    serve((url) =>
      url.includes("/events?") ? { total: 0, events: [] } : undefined,
    );
    render(<App />);
    expect(
      await screen.findByRole("heading", { name: "No events in this run" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Unscorable series are missing evidence/),
    ).toBeInTheDocument();
  });
  it("shows explicit API failure without a false empty result", async () => {
    serve((url) =>
      url.includes("/events?")
        ? new Response(
            JSON.stringify({ detail: "Source changed. Reload the run." }),
            { status: 409 },
          )
        : undefined,
    );
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Source changed",
    );
    expect(
      screen.queryByText("No events match these filters."),
    ).not.toBeInTheDocument();
  });
  it("shows missing series observations and unavailable uncertainty in an accessible table", () => {
    render(<FeatureChart series={series} />);
    expect(
      screen.getByRole("img", { name: /Feature mean/ }),
    ).toBeInTheDocument();
    const table = screen.getByRole("table", { name: "Feature observations" });
    expect(within(table).getAllByText("Unavailable").length).toBeGreaterThan(1);
    expect(screen.getByText(/Gaps remain missing/)).toBeInTheDocument();
  });
});

describe("comparison and saved evidence", () => {
  it("compares decimal step windows and shows absent paired evidence explicitly", async () => {
    serve((url, init) =>
      url === "/api/comparisons" && init?.method === "POST"
        ? {
            id: "c1",
            status: "ready",
            report: {
              mode: "windows",
              sides: [
                {
                  side: "a",
                  run: "fixture",
                  window: [1, 2],
                  n_rollouts: 10,
                  n_prompts: 2,
                  n_sampling_units: 2,
                },
                {
                  side: "b",
                  run: "fixture",
                  window: [3, 5],
                  n_rollouts: 10,
                  n_prompts: 2,
                  n_sampling_units: 2,
                },
              ],
              prompt_mix: {
                status: "no_shared_prompts",
                shared_prompts: 0,
                only_a_prompts: 2,
                only_b_prompts: 2,
                prompt_shift: [],
              },
              features: [
                {
                  feature: "call:check",
                  tier: "structural",
                  kind: "binary",
                  direction: "increase",
                  a: { mean: 0, se: 0, status: "available" },
                  b: { mean: 1, se: 0, status: "available" },
                  difference: {
                    estimate: 1,
                    se: 0,
                    status: "available",
                    method: "independent",
                  },
                  paired: {
                    estimate: null,
                    se: null,
                    status: "no_shared_prompts",
                    n_prompts: 0,
                  },
                  citations: {
                    a: { status: "no_positive_rollouts", items: [] },
                    b: {
                      status: "selected",
                      items: [{ sample_id: "sample-1", step: 5 }],
                    },
                  },
                },
              ],
            },
          }
        : undefined,
    );
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("tab", { name: "Compare" }));
    for (const [name, value] of [
      ["A start", "1"],
      ["A end", "2"],
      ["B start", "3"],
      ["B end", "5"],
    ])
      await user.type(screen.getByLabelText(name), value);
    await user.click(screen.getByRole("button", { name: "Compare windows" }));
    expect(
      await screen.findByText("Unavailable: no shared prompts"),
    ).toBeInTheDocument();
    expect(screen.getByText("Shared-prompt change")).toBeInTheDocument();
    expect(
      vi
        .mocked(fetch)
        .mock.calls.some(
          ([url, init]) =>
            url === "/api/comparisons" &&
            JSON.parse(String(init?.body)).a[0] === "1",
        ),
    ).toBe(true);
  });
  it("separates historical reward claims from task validation in saved evidence", async () => {
    serve((url) =>
      url === "/api/evidence"
        ? {
            kind: "replay",
            path: "saved.json",
            sha256: "c".repeat(64),
            verification: "Validated saved schema",
            source_link: { status: "not_loaded", reason: "Source not loaded" },
            identity: null,
            summary: {},
            conditions: [
              {
                name: "baseline",
                historical_status: "passed",
                historical_passed: true,
                task_status: "unavailable",
                task_determinate: false,
                task_reason: "Missing task counters",
                discrepancy: null,
                cleanup_verified: null,
              },
            ],
            hypothesis: null,
            predictions: [],
            edit: null,
            source_diff: null,
            mechanism_status: "unresolved",
          }
        : url.includes("/files?")
          ? {
              path: ".",
              entries: [
                { name: "saved.json", path: "saved.json", kind: "evidence" },
              ],
            }
          : undefined,
    );
    const user = userEvent.setup();
    render(<App />);
    await user.click(
      await screen.findByRole("tab", { name: "Saved evidence" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Open saved evidence" }),
    );
    await user.click(
      await screen.findByRole("button", { name: "Select saved.json" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Read selected evidence" }),
    );
    expect(await screen.findByText("Historical claim")).toBeInTheDocument();
    expect(screen.getByText("Saved protected task result")).toBeInTheDocument();
    expect(screen.getByText("Missing task counters")).toBeInTheDocument();
    expect(screen.getByText(/Mechanism unresolved/)).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /execute|approve|replay/i }),
    ).not.toBeInTheDocument();
  });
});

describe("stale responses and exact chart evidence", () => {
  it("ignores delayed event details after a newer event is selected", async () => {
    let resolveOld: (response: Response) => void = () => {};
    const old = new Promise<Response>((resolve) => {
      resolveOld = resolve;
    });
    const next = {
      ...event,
      rank: 8,
      representative_feature: "call:new",
      members: [{ ...event.members[0], feature: "call:new", exemplars: [] }],
    };
    serve((url) =>
      url.includes("/events?")
        ? { total: 2, events: [event, next] }
        : url.includes("/events/7")
          ? old
          : url.includes("/events/8")
            ? { event: next, evidence: [], prompt_shifts: [] }
            : undefined,
    );
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: /Event 7/ }));
    await user.click(screen.getByRole("button", { name: /Event 8/ }));
    expect(
      await screen.findByRole("heading", { name: /call:new/, level: 1 }),
    ).toBeInTheDocument();
    resolveOld(
      new Response(JSON.stringify({ event, evidence: [], prompt_shifts: [] })),
    );
    await waitFor(() =>
      expect(
        screen.queryByRole("heading", { name: /call:check/, level: 1 }),
      ).not.toBeInTheDocument(),
    );
  });
  it("keeps adjacent large decimal steps distinct and breaks missing-step lines", () => {
    const exact = {
      ...series,
      points: [
        { ...series.points[0], step: "9007199254740993", mean: 0.2, se: 0.1 },
        { ...series.points[0], step: "9007199254740994", mean: 0.4, se: null },
        { ...series.points[0], step: "9007199254741000", mean: 0.8, se: 0.2 },
      ],
      evidence: {
        onset_step: "9007199254740994",
        alert_step: "9007199254741000",
      },
    };
    const { container } = render(<FeatureChart series={exact} />);
    expect(screen.getByText("Onset 9007199254740994")).toBeInTheDocument();
    expect(
      screen.getByText("First alert 9007199254741000"),
    ).toBeInTheDocument();
    const paths = container.querySelectorAll(".mean-line");
    expect(paths).toHaveLength(2);
    const points = container.querySelectorAll("circle");
    expect(points[0].getAttribute("cx")).not.toEqual(
      points[1].getAttribute("cx"),
    );
    expect(container.querySelectorAll(".error-bar")).toHaveLength(2);
    expect(container.querySelectorAll(".uncertain-point")).toHaveLength(1);
  });
});

describe("pending input cancellation and display precision", () => {
  it("does not apply saved evidence after its picker was closed", async () => {
    let finish: (response: Response) => void = () => {};
    const pending = new Promise<Response>((resolve) => {
      finish = resolve;
    });
    serve((url) =>
      url === "/api/evidence"
        ? pending
        : url.includes("/files?")
          ? {
              path: ".",
              entries: [
                { name: "saved.json", path: "saved.json", kind: "evidence" },
              ],
            }
          : undefined,
    );
    const user = userEvent.setup();
    render(<App />);
    await user.click(
      await screen.findByRole("tab", { name: "Saved evidence" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Open saved evidence" }),
    );
    await user.click(
      await screen.findByRole("button", { name: "Select saved.json" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Read selected evidence" }),
    );
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    const call = vi
      .mocked(fetch)
      .mock.calls.find(([url]) => url === "/api/evidence");
    expect(call?.[1]?.signal?.aborted).toBe(true);
    finish(
      new Response(
        JSON.stringify({
          kind: "replay",
          path: "old-source.json",
          sha256: "c".repeat(64),
          verification: "Old source verified",
          source_link: { status: "matched", reason: "Old run source" },
          identity: null,
          summary: {},
          conditions: [],
          predictions: [],
          mechanism_status: "unresolved",
        }),
      ),
    );
    await waitFor(() =>
      expect(screen.queryByText("Old source verified")).not.toBeInTheDocument(),
    );
  });
  it("preserves scientific exponents when formatting small uncertainty", async () => {
    const { text } = await import("./common");
    expect(Number(text(1.25e-10))).toBe(1.25e-10);
    expect(Number(text(1.25e-20))).toBe(1.25e-20);
  });
});

it("can retry a failed background comparison status request", async () => {
  let attempts = 0;
  serve((url, init) =>
    url === "/api/comparisons" && init?.method === "POST"
      ? { id: "retry-comparison", status: "queued" }
      : url === "/api/comparisons/retry-comparison"
        ? ++attempts === 1
          ? new Response(
              JSON.stringify({ detail: "Status temporarily unavailable" }),
              { status: 503 },
            )
          : {
              id: "retry-comparison",
              status: "ready",
              report: {
                mode: "windows",
                sides: [],
                prompt_mix: { status: "no_shared_prompts" },
                features: [],
              },
            }
        : undefined,
  );
  const user = userEvent.setup();
  render(<App />);
  await user.click(await screen.findByRole("tab", { name: "Compare" }));
  for (const [name, value] of [
    ["A start", "1"],
    ["A end", "2"],
    ["B start", "3"],
    ["B end", "4"],
  ])
    await user.type(screen.getByLabelText(name), value);
  await user.click(screen.getByRole("button", { name: "Compare windows" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Status temporarily unavailable",
  );
  await user.click(screen.getByRole("button", { name: "Retry" }));
  expect(
    await screen.findByRole("heading", { name: "What changed" }),
  ).toBeInTheDocument();
});

describe("same-document session launch", () => {
  it("accepts a new fragment token after the missing-token screen", async () => {
    sessionStorage.clear();
    render(<App />);
    expect(screen.getByRole("alert")).toHaveTextContent(
      "No local session token",
    );
    await act(async () => {
      window.history.replaceState(null, "", "/#token=new-session");
      window.dispatchEvent(new HashChangeEvent("hashchange"));
    });
    expect(
      await screen.findByRole("button", { name: /Event 7/ }),
    ).toBeInTheDocument();
    expect(window.location.hash).toBe("");
    expect(sessionStorage.getItem("rolloutdiff.session.v1")).toBe(
      "new-session",
    );
    expect(
      vi
        .mocked(fetch)
        .mock.calls.every(
          ([, init]) =>
            new Headers(init?.headers).get("Authorization") ===
            "Bearer new-session",
        ),
    ).toBe(true);
  });
  it("clears original response and run state while connecting with replacement credentials", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: /Event 7/ }));
    await user.click(
      await screen.findByRole("button", { name: /^Read response sample-1/ }),
    );
    expect(
      await screen.findByRole("dialog", { name: "Original response" }),
    ).toBeInTheDocument();
    let finishSession: (value: Response) => void = () => {};
    serve((url) =>
      url === "/api/session"
        ? new Promise<Response>((resolve) => {
            finishSession = resolve;
          })
        : url === "/api/runs"
          ? { runs: [] }
          : undefined,
    );
    await act(async () => {
      window.history.replaceState(null, "", "/#token=replacement");
      window.dispatchEvent(new HashChangeEvent("hashchange"));
    });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Run summary" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Event 7/ }),
    ).not.toBeInTheDocument();
    await act(async () =>
      finishSession(
        new Response(JSON.stringify({ root_label: "replacement-root" })),
      ),
    );
    expect(await screen.findByText("replacement-root")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Load a rollout run to begin" }),
    ).toBeInTheDocument();
  });
  it("retries an expired session on a fragment navigation without a page reload", async () => {
    serve((_url, init) =>
      new Headers(init?.headers).get("Authorization") === "Bearer secret"
        ? new Response("{}", { status: 401 })
        : undefined,
    );
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Session authorization expired",
    );
    await act(async () => {
      window.history.replaceState(null, "", "/#token=reissued");
      window.dispatchEvent(new HashChangeEvent("hashchange"));
    });
    expect(
      await screen.findByRole("button", { name: /Event 7/ }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("root selection and run-bound evidence", () => {
  it("loads the launch root with the explicitly selected adapter", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "Load run" }));
    await user.selectOptions(screen.getByLabelText("Input format"), "verl");
    await user.click(
      screen.getByRole("button", { name: "Use this directory" }),
    );
    await user.type(screen.getByLabelText("Run name"), "whole-dump");
    await user.click(screen.getByRole("button", { name: "Load selected run" }));
    await waitFor(() => {
      const call = vi
        .mocked(fetch)
        .mock.calls.find(
          ([url, init]) => url === "/api/runs" && init?.method === "POST",
        );
      expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({
        path: ".",
        format: "verl",
        run: "whole-dump",
      });
    });
  });
  it("clears saved evidence linked to the previous run when a new run is loaded", async () => {
    const saved = {
      kind: "replay",
      path: "old-run-evidence.json",
      sha256: "c".repeat(64),
      verification: "Old source saved evidence",
      source_link: { status: "matched", reason: "Matched to previous run" },
      identity: null,
      summary: {},
      conditions: [],
      predictions: [],
      mechanism_status: "unresolved",
    };
    serve((url, init) =>
      url === "/api/evidence"
        ? saved
        : url === "/api/runs" && init?.method === "POST"
          ? { ...run, id: "r2", run: "new-run" }
          : url.includes("/files?")
            ? {
                path: ".",
                entries: [
                  {
                    name: "old-run-evidence.json",
                    path: "old-run-evidence.json",
                    kind: "evidence",
                  },
                  {
                    name: "new-run.jsonl",
                    path: "new-run.jsonl",
                    kind: "verl",
                  },
                ],
              }
            : undefined,
    );
    const user = userEvent.setup();
    render(<App />);
    await user.click(
      await screen.findByRole("tab", { name: "Saved evidence" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Open saved evidence" }),
    );
    await user.click(
      await screen.findByRole("button", {
        name: "Select old-run-evidence.json",
      }),
    );
    await user.click(
      screen.getByRole("button", { name: "Read selected evidence" }),
    );
    expect(
      await screen.findByText("Old source saved evidence"),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Load run" }));
    await user.click(
      await screen.findByRole("button", { name: "Select new-run.jsonl" }),
    );
    await user.type(screen.getByLabelText("Run name"), "new-run");
    await user.click(screen.getByRole("button", { name: "Load selected run" }));
    await waitFor(() =>
      expect(screen.getByRole("tab", { name: "Events" })).toHaveAttribute(
        "aria-selected",
        "true",
      ),
    );
    await user.click(screen.getByRole("tab", { name: "Saved evidence" }));
    expect(
      screen.queryByText("Old source saved evidence"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Choose a saved evidence file" }),
    ).toBeInTheDocument();
  });
});
