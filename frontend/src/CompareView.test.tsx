import { StrictMode } from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CompareView } from "./CompareView";
import type { CompareSeed, ComparisonFeature, Run } from "./types";

const run: Run = { id: "r1", run: "demo-a", status: "ready" };
const control: Run = { id: "r2", run: "demo-b", status: "ready" };

function feature(
  name: string,
  overrides: Partial<ComparisonFeature> = {},
): ComparisonFeature {
  return {
    feature: name,
    tier: "structural",
    kind: "binary",
    direction: "increase",
    a: {
      mean: 0.1,
      se: 0.01,
      ci_low: 0.08,
      ci_high: 0.12,
      status: "available",
      n_rollouts: 180,
      n_prompts: 6,
      n_sampling_units: 6,
    },
    b: { mean: 0.4, se: 0.02, status: "available", n_prompts: 6 },
    difference: {
      estimate: 0.3,
      se: 0.02,
      ci_low: 0.26,
      ci_high: 0.34,
      status: "available",
      method: "paired",
    },
    paired: {
      estimate: 0.3,
      se: 0.02,
      ci_low: 0.26,
      ci_high: 0.34,
      status: "available",
      n_prompts: 6,
      n_sampling_units: 6,
    },
    citations: {
      a: { status: "no_positive_rollouts", items: [] },
      b: {
        status: "selected",
        items: [
          { sample_id: "16:0:0", step: 16 },
          { sample_id: "16:1:0", step: 17 },
        ],
      },
    },
    ...overrides,
  };
}
function ready(features: ComparisonFeature[], prompt_mix?: object) {
  return {
    id: "c1",
    status: "ready",
    report: {
      mode: "windows",
      sides: [
        {
          side: "a",
          run: "demo-a",
          window: [1, 15],
          n_rollouts: 180,
          n_prompts: 6,
          n_sampling_units: 6,
        },
      ],
      prompt_mix: prompt_mix ?? {
        status: "shared_prompts",
        shared_prompts: 6,
        only_a_prompts: 2,
        only_b_prompts: 1,
        prompt_shift: [],
      },
      features,
    },
  };
}
type Handler = (url: string, init?: RequestInit) => unknown;
function serve(handler: Handler = () => undefined) {
  const mock = vi.fn(async (input: string, init?: RequestInit) => {
    const url = String(input);
    const value = await handler(url, init);
    if (value instanceof Response) return value;
    return new Response(
      JSON.stringify(
        value ?? (url.endsWith("/steps") ? { steps: [1, 2, 3, 4] } : {}),
      ),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  });
  vi.stubGlobal("fetch", mock);
  return mock;
}
function posts(mock: ReturnType<typeof serve>) {
  return mock.mock.calls
    .filter(
      ([url, init]) => url === "/api/comparisons" && init?.method === "POST",
    )
    .map(([, init]) => JSON.parse(String(init?.body)));
}
function view(seed: CompareSeed | null = null, runs: Run[] = [run]) {
  const onCitation = vi.fn();
  return {
    onCitation,
    ...render(
      <CompareView run={run} runs={runs} seed={seed} onCitation={onCitation} />,
    ),
  };
}
const respond =
  (value: object): Handler =>
  (url, init) =>
    url === "/api/comparisons" && init?.method === "POST" ? value : undefined;
async function fill(
  user: ReturnType<typeof userEvent.setup>,
  values: [string, string, string, string] = ["1", "2", "3", "5"],
) {
  for (const [name, value] of [
    ["A start", values[0]],
    ["A end", values[1]],
    ["B start", values[2]],
    ["B end", values[3]],
  ])
    await user.type(screen.getByLabelText(name), value);
}
async function submitted(
  user: ReturnType<typeof userEvent.setup>,
  values?: [string, string, string, string],
) {
  await fill(user, values);
  await user.click(screen.getByRole("button", { name: "Compare windows" }));
}
function timeline(container: HTMLElement) {
  const svg = container.querySelector<SVGSVGElement>(".compare-timeline svg");
  if (svg)
    vi.spyOn(svg, "getBoundingClientRect").mockReturnValue({
      left: 0,
      width: 100,
    } as DOMRect);
  return svg;
}

beforeEach(() => {
  sessionStorage.setItem("rolloutdiff.session.v1", "secret");
});

describe("submitting windows", () => {
  it("posts decimal strings", async () => {
    const mock = serve(respond(ready([feature("def:solve")])));
    const user = userEvent.setup();
    view();
    await submitted(user);
    await screen.findByRole("heading", { name: "What changed" });
    expect(posts(mock)).toEqual([
      { run_a: "r1", run_b: null, a: ["1", "2"], b: ["3", "5"] },
    ]);
  });
  it("keeps large decimal steps unchanged", async () => {
    const mock = serve(respond(ready([])));
    const user = userEvent.setup();
    view();
    await submitted(user, [
      "9007199254740993",
      "9007199254740995",
      "9007199254740997",
      "12345678901234567890",
    ]);
    await screen.findByRole("heading", { name: "What changed" });
    expect(posts(mock)[0].a).toEqual(["9007199254740993", "9007199254740995"]);
    expect(posts(mock)[0].b).toEqual([
      "9007199254740997",
      "12345678901234567890",
    ]);
  });
  it("submits on Enter", async () => {
    const mock = serve(respond(ready([])));
    const user = userEvent.setup();
    view();
    await fill(user);
    await user.type(screen.getByLabelText("B end"), "{Enter}");
    await screen.findByRole("heading", { name: "What changed" });
    expect(posts(mock)).toHaveLength(1);
  });
  it("gives the range message for empty and reversed windows", async () => {
    const mock = serve();
    const user = userEvent.setup();
    view();
    await user.click(screen.getByRole("button", { name: "Compare windows" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Enter nonnegative integer step ranges with each start at or before its end.",
    );
    await submitted(user, ["5", "2", "3", "5"]);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(posts(mock)).toHaveLength(0);
  });
  it("sends the chosen comparison run", async () => {
    const mock = serve(respond(ready([])));
    const user = userEvent.setup();
    view(null, [run, control]);
    await user.selectOptions(screen.getByLabelText(/comparison run/i), "demo-b");
    expect(
      screen.getByRole("option", { name: "Same run: demo-a" }),
    ).toBeInTheDocument();
    await submitted(user);
    await screen.findByRole("heading", { name: "What changed" });
    expect(posts(mock)[0].run_b).toBe("r2");
  });
  it("retries after a failed status poll", async () => {
    let attempts = 0;
    serve((url, init) =>
      url === "/api/comparisons" && init?.method === "POST"
        ? { id: "c1", status: "queued" }
        : url === "/api/comparisons/c1"
          ? ++attempts === 1
            ? new Response(
                JSON.stringify({ detail: "Status temporarily unavailable" }),
                { status: 503 },
              )
            : ready([])
          : undefined,
    );
    const user = userEvent.setup();
    view();
    await submitted(user);
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Comparison queued…",
    );
    expect(
      screen.queryByRole("heading", { name: "What changed" }),
    ).not.toBeInTheDocument();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Status temporarily unavailable",
    );
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(
      await screen.findByRole("heading", { name: "What changed" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
  it("shows a failed comparison's error", async () => {
    serve(respond({ id: "c1", status: "failed", error: "Could not compare." }));
    const user = userEvent.setup();
    view();
    await submitted(user);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not compare.",
    );
  });
});

describe("seed", () => {
  const seed: CompareSeed = {
    feature: "def:solve",
    a: ["9007199254740993", "9007199254740995"],
    b: ["9007199254740997", "9007199254740999"],
  };
  it("fills the inputs and posts once under StrictMode", async () => {
    const mock = serve(respond(ready([feature("def:solve")])));
    render(
      <StrictMode>
        <CompareView run={run} runs={[run]} seed={seed} onCitation={vi.fn()} />
      </StrictMode>,
    );
    expect(screen.getByLabelText("A start")).toHaveValue("9007199254740993");
    expect(screen.getByLabelText("B end")).toHaveValue("9007199254740999");
    await screen.findByRole("heading", { name: "What changed" });
    expect(posts(mock)).toEqual([
      { run_a: "r1", run_b: null, a: seed.a, b: seed.b },
    ]);
    expect(
      screen.getByRole("button", { name: "Compare windows" }),
    ).toBeEnabled();
  });
  it("resets the comparison run when a new seed arrives", async () => {
    const mock = serve(respond(ready([])));
    const user = userEvent.setup();
    const { rerender } = view(null, [run, control]);
    await user.selectOptions(screen.getByLabelText(/comparison run/i), "demo-b");
    expect(posts(mock)).toHaveLength(0);
    const props = { run, runs: [run, control], onCitation: vi.fn() };
    rerender(<CompareView {...props} seed={seed} />);
    await screen.findByRole("heading", { name: "What changed" });
    expect(screen.getByLabelText(/comparison run/i)).toHaveValue("");
    expect(posts(mock)[0].run_b).toBeNull();
    rerender(<CompareView {...props} seed={seed} />);
    expect(posts(mock)).toHaveLength(1);
    rerender(<CompareView {...props} seed={{ ...seed }} />);
    expect(posts(mock)).toHaveLength(2);
    await screen.findByRole("heading", { name: "What changed" });
  });
});

describe("timeline", () => {
  it("snaps a drag to observed steps and writes exact decimal strings", async () => {
    serve((url) =>
      url.endsWith("/steps")
        ? {
            steps: [
              "9007199254740993",
              "9007199254740995",
              "9007199254740997",
              "9007199254741993",
            ],
          }
        : undefined,
    );
    const { container } = view();
    await screen.findByText("Drag the line to set");
    const svg = timeline(container)!;
    expect(svg).toHaveAttribute("aria-hidden", "true");
    expect(svg.getAttribute("viewBox")).toBeNull();
    fireEvent.pointerDown(svg, { clientX: 0, pointerId: 1 });
    expect(screen.getByLabelText("A start")).toHaveValue("9007199254740993");
    expect(screen.getByLabelText("A end")).toHaveValue("9007199254740993");
    fireEvent.pointerMove(svg, { clientX: 1, pointerId: 1 });
    expect(screen.getByLabelText("A end")).toHaveValue("9007199254740997");
    fireEvent.pointerMove(svg, { clientX: 90, pointerId: 1 });
    expect(screen.getByLabelText("A end")).toHaveValue("9007199254741993");
    fireEvent.pointerUp(svg, { clientX: 90, pointerId: 1 });
    fireEvent.pointerMove(svg, { clientX: 0, pointerId: 1 });
    expect(screen.getByLabelText("A end")).toHaveValue("9007199254741993");
    expect(screen.getByRole("button", { name: "A" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });
  it("breaks the observed line at gaps wider than the smallest spacing", async () => {
    serve((url) =>
      url.endsWith("/steps") ? { steps: [4, 8, 12, 40, 44, 100] } : undefined,
    );
    const { container } = view();
    await screen.findByText("Drag the line to set");
    const lines = [...container.querySelectorAll(".compare-observed")];
    expect(lines.map((line) => line.getAttribute("x1"))).toEqual([
      "0%",
      "37.5%",
      "100%",
    ]);
    expect(lines[0].getAttribute("x2")).toBe("8.3333%");
    expect(screen.getByText("52")).toBeInTheDocument();
  });
  it("uses the union axis and snaps side B to the comparison run", async () => {
    serve((url) =>
      url === "/api/runs/r1/steps"
        ? { steps: [0, 10, 20] }
        : url === "/api/runs/r2/steps"
          ? { steps: [50, 100] }
          : undefined,
    );
    const user = userEvent.setup();
    const { container } = view(null, [run, control]);
    await screen.findByText("Drag the line to set");
    await user.selectOptions(screen.getByLabelText(/comparison run/i), "demo-b");
    await screen.findByText("100");
    await user.click(screen.getByRole("button", { name: "B" }));
    const svg = timeline(container)!;
    fireEvent.pointerDown(svg, { clientX: 5, pointerId: 1 });
    expect(screen.getByLabelText("B start")).toHaveValue("50");
    expect(screen.getByLabelText("A start")).toHaveValue("");
    const lines = [...container.querySelectorAll(".compare-observed")];
    expect(lines.map((line) => line.getAttribute("x1"))).toEqual(["0%", "50%"]);
    expect(lines[0].getAttribute("y1")).not.toBe(lines[1].getAttribute("y1"));
  });
  it("draws window bars only for valid windows on the axis", async () => {
    serve((url) => (url.endsWith("/steps") ? { steps: [0, 10] } : undefined));
    const user = userEvent.setup();
    const { container } = view();
    await screen.findByText("Drag the line to set");
    await fill(user, ["0", "5", "90", "99"]);
    expect(container.querySelector(".compare-window.a line")).toHaveAttribute(
      "x2",
      "50%",
    );
    expect(container.querySelector(".compare-window.b")).toBeNull();
  });
});

describe("results", () => {
  it("states the overlap counts", async () => {
    serve(respond(ready([feature("def:solve")])));
    const user = userEvent.setup();
    view();
    await submitted(user);
    expect(
      await screen.findByText("6 prompts appear in both windows."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/2 appear only in A and 1 only in B\./),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Only exact shared prompts support the shared-prompt change/),
    ).toBeInTheDocument();
  });
  it("shows explicit unavailable text and no point without shared prompts", async () => {
    serve(
      respond(
        ready(
          [
            feature("call:check", {
              difference: {
                estimate: 1,
                se: 0,
                status: "partial_overlap",
                method: "independent",
              },
              paired: {
                estimate: null,
                se: null,
                status: "no_shared_prompts",
                n_prompts: 0,
              },
            }),
          ],
          {
            status: "no_shared_prompts",
            shared_prompts: 0,
            only_a_prompts: 2,
            only_b_prompts: 2,
            prompt_shift: [],
          },
        ),
      ),
    );
    const user = userEvent.setup();
    const { container } = view();
    await submitted(user);
    expect(
      await screen.findByText("No prompts appear in both windows."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Paired estimates are unavailable\./),
    ).toBeInTheDocument();
    expect(screen.getByText("Unavailable: no shared prompts")).toHaveClass(
      "unavailable",
    );
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(container.querySelector("circle")).toBeNull();
    expect(screen.getByText("Interval")).toBeInTheDocument();
    const counts = screen.getByText("only a prompts").parentElement!;
    expect(within(counts).getByText("2")).toBeInTheDocument();
  });
  it("draws an open point and a visible status when no SE was measured", async () => {
    serve(
      respond(
        ready([
          feature("zero:se", {
            paired: {
              estimate: 0.5,
              se: 0,
              ci_low: 0.5,
              ci_high: 0.5,
              status: "available",
            },
          }),
          feature("measured:se"),
          feature("no:interval", {
            paired: {
              estimate: -0.2,
              se: null,
              ci_low: null,
              ci_high: null,
              status: "insufficient_units",
            },
          }),
        ]),
      ),
    );
    const user = userEvent.setup();
    view();
    await submitted(user);
    const [zero, measured, noInterval] = await screen.findAllByRole("img");
    expect(zero.querySelector("circle")).toHaveClass("mean-point");
    expect(zero).toHaveAccessibleName(
      "Shared-prompt change +0.5, 95% normal interval +0.5 to +0.5. A zero measured SE does not establish zero uncertainty.",
    );
    expect(measured.querySelector("circle")).toHaveClass("mean-point");
    expect(measured.querySelector(".error-bar")).toHaveAttribute("x1", "76%");
    expect(measured).toHaveAccessibleName(
      "Shared-prompt change +0.3, 95% normal interval +0.26 to +0.34.",
    );
    expect(noInterval.querySelector("circle")).toHaveClass("uncertain-point");
    expect(
      screen.getByText("No interval: insufficient units"),
    ).toBeInTheDocument();
    expect(noInterval.querySelector(".error-bar")).toBeNull();
    expect(noInterval).toHaveAccessibleName(
      "Shared-prompt change −0.2, interval unavailable, insufficient units.",
    );
    expect(
      screen.getByRole("columnheader", { name: "−0.5 0 +0.5" }),
    ).toBeInTheDocument();
  });
  it("plots the shared-prompt estimate, never the overall difference", async () => {
    serve(
      respond(
        ready([
          feature("kept:apart", {
            difference: {
              estimate: 0.9,
              se: 0.01,
              ci_low: 0.88,
              ci_high: 0.92,
              status: "available",
              method: "independent",
            },
            paired: {
              estimate: 0.1,
              se: 0.01,
              ci_low: 0.08,
              ci_high: 0.12,
              status: "available",
            },
          }),
        ]),
      ),
    );
    const user = userEvent.setup();
    view();
    await submitted(user);
    const plot = await screen.findByRole("img");
    expect(plot.querySelector("circle")).toHaveAttribute("cx", "75%");
    expect(screen.getByText("+0.1")).toBeInTheDocument();
  });
  it("keeps report order within each group", async () => {
    const paired = (estimate: number | null) => ({
      estimate,
      se: estimate === null ? null : 1,
      status: estimate === null ? "numeric_unavailable" : "available",
    });
    serve(
      respond(
        ready([
          feature("b:none", { paired: paired(null) }),
          feature("b:positive", { paired: paired(0.1) }),
          feature("b:negative", { paired: paired(-0.4) }),
          feature("c:large", { kind: "continuous", paired: paired(-90) }),
          feature("c:small", { kind: "continuous", paired: paired(2) }),
        ]),
      ),
    );
    const user = userEvent.setup();
    view();
    await submitted(user);
    const tables = await screen.findAllByRole("table");
    const names = (table: HTMLElement) =>
      within(table)
        .getAllByRole("button")
        .map((button) => button.textContent);
    expect(names(tables[0])).toEqual(["b:none", "b:positive", "b:negative"]);
    expect(names(tables[1])).toEqual(["c:large", "c:small"]);
    expect(within(tables[0]).getByRole("caption")).toHaveTextContent(
      "Binary features",
    );
    expect(
      within(tables[1]).getByText("Each row on its own scale"),
    ).toBeInTheDocument();
    expect(within(tables[1]).getByRole("caption")).toHaveTextContent(
      "Continuous features",
    );
    expect(screen.getByText(/Binary features follow report order/)).toBeInTheDocument();
  });
  it("limits binary rows until asked, never continuous rows", async () => {
    const binary = Array.from({ length: 25 }, (_, index) =>
      feature(`b:${index}`),
    );
    const continuous = Array.from({ length: 15 }, (_, index) =>
      feature(`c:${index}`, { kind: "continuous" }),
    );
    serve(respond(ready([...binary, ...continuous])));
    const user = userEvent.setup();
    view();
    await submitted(user);
    await screen.findByRole("heading", { name: "What changed" });
    expect(screen.getAllByRole("button", { expanded: false })).toHaveLength(35);
    expect(screen.getByText(/^20 of 25 binary features shown\./)).toBeInTheDocument();
    await user.type(screen.getByLabelText("Search comparison features"), "b:");
    expect(screen.getAllByRole("button", { expanded: false })).toHaveLength(25);
    await user.clear(screen.getByLabelText("Search comparison features"));
    await user.click(
      screen.getByRole("button", { name: "Show all 25 binary features" }),
    );
    expect(screen.getAllByRole("button", { expanded: false })).toHaveLength(40);
    await user.type(
      screen.getByLabelText("Search comparison features"),
      "absent",
    );
    expect(
      screen.getByText("No comparison features match."),
    ).toBeInTheDocument();
    expect(screen.queryByText(/^Figure 1\./)).not.toBeInTheDocument();
  });
  it("expands a seeded feature past the row limit and clears a prior search", async () => {
    const binary = Array.from({ length: 25 }, (_, index) =>
      feature(`b:${String(index).padStart(2, "0")}`),
    );
    serve(respond(ready(binary)));
    const seed = { feature: "b:24", a: ["1", "2"], b: ["3", "4"] } as CompareSeed;
    const { rerender } = view();
    const user = userEvent.setup();
    await submitted(user);
    await user.type(
      await screen.findByLabelText("Search comparison features"),
      "b:01",
    );
    rerender(
      <CompareView run={run} runs={[run]} seed={seed} onCitation={vi.fn()} />,
    );
    expect(
      await screen.findByRole("button", { name: "b:24", expanded: true }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Search comparison features")).toHaveValue("");
    await user.click(screen.getByRole("button", { name: "b:24", expanded: true }));
    expect(
      screen.getByRole("button", { name: "b:24", expanded: false }),
    ).toBeInTheDocument();
  });
  it("dims results after an edit and clears on submit", async () => {
    const mock = serve(respond(ready([feature("def:solve")])));
    const user = userEvent.setup();
    const { container } = view(null, [run, control]);
    await submitted(user);
    await screen.findByRole("heading", { name: "What changed" });
    const button = screen.getByRole("button", { name: "Compare windows" });
    expect(button).toHaveClass("quiet");
    expect(container.querySelector(".compare-results")).not.toHaveClass(
      "stale",
    );
    const message =
      "The windows changed. Results below are from the previous comparison.";
    expect(screen.queryByText(message)).not.toBeInTheDocument();
    await user.type(screen.getByLabelText("A end"), "0");
    expect(screen.getByText(message)).toBeInTheDocument();
    expect(container.querySelector(".compare-results")).toHaveClass("stale");
    expect(button).not.toHaveClass("quiet");
    await user.click(button);
    await screen.findByRole("heading", { name: "What changed" });
    expect(posts(mock)).toHaveLength(2);
    expect(screen.queryByText(message)).not.toBeInTheDocument();
    expect(container.querySelector(".compare-results")).not.toHaveClass(
      "stale",
    );
    await user.selectOptions(screen.getByLabelText(/comparison run/i), "demo-b");
    expect(container.querySelector(".compare-results")).toHaveClass("stale");
  });
  it("marks results stale when the timeline changes a window", async () => {
    serve(respond(ready([feature("def:solve")])));
    const user = userEvent.setup();
    const { container } = view();
    await screen.findByText("Drag the line to set");
    await submitted(user);
    await screen.findByRole("heading", { name: "What changed" });
    fireEvent.pointerDown(timeline(container)!, { clientX: 100, pointerId: 1 });
    expect(screen.getByLabelText("A start")).toHaveValue("4");
    expect(container.querySelector(".compare-results")).toHaveClass("stale");
  });
  it("expands a row with estimates and citations and passes a reader context", async () => {
    const citation = {
      citation: { sample_id: "16:1:0" },
      source: {},
      response_sha256: "a".repeat(64),
    };
    const mock = serve((url, init) =>
      url.endsWith("/inspect")
        ? citation
        : respond(ready([feature("def:solve")]))(url, init),
    );
    const user = userEvent.setup();
    const { onCitation } = view();
    await submitted(user);
    const row = await screen.findByRole("button", { name: "def:solve" });
    expect(row).toHaveAttribute("aria-expanded", "false");
    await user.click(row);
    expect(row).toHaveAttribute("aria-expanded", "true");
    for (const name of [
      "A estimate",
      "B estimate",
      "Difference",
      "Shared-prompt estimate",
      "Side A citations",
      "Side B citations",
    ])
      expect(screen.getByRole("heading", { name })).toBeInTheDocument();
    expect(
      screen.getByText(/B minus A interval method: paired\./),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Tier structural, B minus A direction increase/),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Unavailable: no positive rollouts."),
    ).toBeInTheDocument();
    expect(screen.getByText("Step 17")).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "Read response 16:1:0" }),
    );
    expect(onCitation).toHaveBeenCalledTimes(1);
    const [value, context] = onCitation.mock.calls[0];
    expect(value).toEqual(citation);
    expect(context).toMatchObject({
      back: "Compare, def:solve",
      feature: "def:solve",
      sampleIds: ["16:0:0", "16:1:0"],
    });
    await context.load("16:0:0");
    const bodies = mock.mock.calls
      .filter(([url]) => url === "/api/comparisons/c1/inspect")
      .map(([, init]) => JSON.parse(String(init?.body)));
    expect(bodies).toEqual([
      { side: "b", feature: "def:solve", sample_id: "16:1:0" },
      { side: "b", feature: "def:solve", sample_id: "16:0:0" },
    ]);
    await user.click(row);
    expect(
      screen.queryByRole("heading", { name: "A estimate" }),
    ).not.toBeInTheDocument();
  });
  it("shows an inspection failure beside the citations and allows another try", async () => {
    let attempts = 0;
    serve((url, init) =>
      url.endsWith("/inspect")
        ? ++attempts === 1
          ? new Response(JSON.stringify({ detail: "Citation changed" }), {
              status: 400,
            })
          : { citation: {}, source: {}, response_sha256: "" }
        : respond(ready([feature("def:solve")]))(url, init),
    );
    const user = userEvent.setup();
    const { onCitation } = view();
    await submitted(user);
    await user.click(await screen.findByRole("button", { name: "def:solve" }));
    const read = screen.getByRole("button", { name: "Read response 16:0:0" });
    await user.click(read);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Citation changed",
    );
    expect(onCitation).not.toHaveBeenCalled();
    await user.click(read);
    expect(onCitation).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
  it("does not open a citation after unmount", async () => {
    let release: (value: Response) => void = () => {};
    serve((url, init) =>
      url.endsWith("/inspect")
        ? new Promise<Response>((resolve) => {
            release = resolve;
          })
        : respond(ready([feature("def:solve")]))(url, init),
    );
    const user = userEvent.setup();
    const { onCitation, unmount } = view();
    await submitted(user);
    await user.click(await screen.findByRole("button", { name: "def:solve" }));
    await user.click(
      screen.getByRole("button", { name: "Read response 16:0:0" }),
    );
    unmount();
    release(new Response(JSON.stringify({ citation: {} }), { status: 200 }));
    await new Promise((resolve) => setTimeout(resolve));
    expect(onCitation).not.toHaveBeenCalled();
  });
  it("exports through the comparison routes and keeps the existing tables", async () => {
    const mock = serve((url, init) =>
      url.includes("/export")
        ? new Response(JSON.stringify({ detail: "Export refused" }), {
            status: 409,
          })
        : respond(ready([feature("def:solve")]))(url, init),
    );
    const user = userEvent.setup();
    view();
    await submitted(user);
    await user.click(
      await screen.findByRole("button", { name: "Export comparison JSON" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Export refused",
    );
    await user.click(
      screen.getByRole("button", { name: "Export comparison HTML" }),
    );
    expect(
      mock.mock.calls
        .map(([url]) => url)
        .filter((url) => url.includes("export")),
    ).toEqual([
      "/api/comparisons/c1/export?format=json",
      "/api/comparisons/c1/export?format=html",
    ]);
    expect(
      screen.getByText("Compared windows and prompt shifts"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("table", { name: "Compared windows" }),
    ).toHaveTextContent("1 to 15");
    expect(
      screen.getByText("No comparison prompt shifts available."),
    ).toBeInTheDocument();
  });
});
