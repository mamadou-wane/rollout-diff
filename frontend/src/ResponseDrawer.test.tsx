import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ResponseDrawer } from "./ResponseDrawer";
import type { Citation, ReaderContext, Row } from "./types";

const source = {
  path: "run.jsonl",
  format: "verl",
  canonical_sha256: "a".repeat(64),
  files: [{ path: "run/1.jsonl", sha256: "c".repeat(64), size_bytes: 12 }],
};
const context: ReaderContext = {
  back: "Event 1, def:solve",
  feature: "def:solve",
  sampleIds: [],
  load: () => Promise.reject(new Error("Not used.")),
};
function cite(citation: Row, response_sha256 = "b".repeat(64)): Citation {
  return { citation, source, response_sha256 };
}
function matched(response: string, overrides: Row = {}) {
  return {
    status: "available",
    reason: null,
    n_rollouts: 2,
    n_present: 1,
    n_absent: 1,
    reward_present: 2,
    reward_absent: 0,
    difference: 2,
    counterpart: {
      run: "r",
      sample_id: "16:0:1",
      step: 16,
      prompt_id: "p0",
      reward: 0,
      response,
      feature_value: 0,
    },
    ...overrides,
  };
}
const noAbsent = {
  status: "unavailable",
  reason: "no_absent_completion",
  n_rollouts: 4,
  n_present: 4,
  n_absent: 0,
  reward_present: 3.5,
  reward_absent: null,
  difference: null,
  counterpart: null,
};
function pair(present: string, absent: string): Citation {
  return cite({
    sample_id: "16:0:0",
    step: 16,
    prompt_id: "p0",
    feature: "def:solve",
    reward: 2,
    response: present,
    comparison: matched(absent),
  });
}
function blocks() {
  return Array.from(document.querySelectorAll<HTMLElement>("pre.response"));
}
function shaded(block: HTMLElement) {
  return Array.from(block.querySelectorAll(".differs")).map(
    (line) => line.textContent,
  );
}
function deferred<T>() {
  let resolve!: (value: T) => void, reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
function paged(load: ReaderContext["load"]): ReaderContext {
  return {
    back: "Event 1, def:solve",
    feature: "def:solve",
    sampleIds: ["s1", "s2", "s3"],
    load,
  };
}
function sample(id: string, response = `text of ${id}`): Citation {
  return cite(
    { sample_id: id, step: 16, reward: 1, response },
    `${id}-digest`,
  );
}

describe("matched comparison reason", () => {
  it.each([
    ["available", "Not applicable"],
    ["unavailable", "Unavailable"],
  ])("displays null reason for %s evidence accurately", (status, expected) => {
    const value: Citation = {
      citation: {
        side: "b",
        feature: "def:solve",
        response: "literal",
        comparison: {
          status,
          reason: null,
          n_rollouts: null,
          counterpart: { response: "matched literal" },
        },
      },
      source: {
        path: "run.jsonl",
        format: "verl",
        canonical_sha256: "a".repeat(64),
        files: [],
      },
      response_sha256: "b".repeat(64),
    };
    render(<ResponseDrawer value={value} onClose={() => {}}
        context={context}
      />);
    expect(screen.getByText("side").parentElement).toHaveTextContent("b");
    expect(screen.getByText("feature").parentElement).toHaveTextContent("def:solve");
    expect(screen.getByText("reason").parentElement).toHaveTextContent(
      expected,
    );
    expect(screen.getByText("n rollouts").parentElement).toHaveTextContent(
      "Unavailable",
    );
    expect(value.citation.comparison).toMatchObject({
      reason: null,
      n_rollouts: null,
    });
  });
});

describe("literal text", () => {
  it.each([
    ["trailing newline", "```python\ndef solve():\n    pass\n```\n"],
    ["no trailing newline", "plain words."],
    ["blank lines", "\n\nfirst\n\n\nlast\n\n"],
    ["carriage returns and indentation", "a\r\n\tb  \r\n"],
    ["empty", ""],
  ])("keeps textContent exact with %s", (_, response) => {
    render(
      <ResponseDrawer value={pair(response, response)} onClose={() => {}}
        context={context}
      />,
    );
    const [present, absent] = blocks();
    expect(present.textContent).toBe(response);
    expect(absent.textContent).toBe(response);
    expect(present.querySelectorAll("span")).toHaveLength(
      response.split("\n").length,
    );
  });
  it("keeps HTML-looking text literal", () => {
    const response = '<img src=x onerror="alert(1)">\n<script>alert(2)</script>\n**bold**';
    render(
      <ResponseDrawer value={pair(response, "<b>x</b>")} onClose={() => {}}
        context={context}
      />,
    );
    const dialog = screen.getByRole("dialog", { name: "Original response" });
    expect(blocks()[0].textContent).toBe(response);
    expect(dialog.querySelector("img, script, b, strong")).toBeNull();
    expect(blocks()[1].textContent).toBe("<b>x</b>");
  });
  it("names an empty response so a blank column is not read as a failure", () => {
    render(
      <ResponseDrawer
        value={cite({ sample_id: "s", step: 1, response: "" })}
        onClose={() => {}}
        context={context}
      />,
    );
    expect(screen.getByText("This response is empty.")).toBeInTheDocument();
  });
});

describe("line shading", () => {
  it("marks only differing lines and can be turned off", async () => {
    const user = userEvent.setup();
    render(
      <ResponseDrawer
        value={pair("same\nonly present\nend\n", "same\nonly absent\nmore\nend\n")}
        onClose={() => {}}
        context={context}
      />,
    );
    const [present, absent] = blocks();
    expect(shaded(present)).toEqual(["only present\n"]);
    expect(shaded(absent)).toEqual(["only absent\n", "more\n"]);
    await user.click(screen.getByLabelText("Shade lines that differ"));
    expect(shaded(present)).toEqual([]);
    expect(shaded(absent)).toEqual([]);
    expect(present.textContent).toBe("same\nonly present\nend\n");
  });
  it("marks a difference that is only a trailing newline", () => {
    render(<ResponseDrawer value={pair("same\n", "same")} onClose={() => {}}
        context={context}
      />);
    const [present, absent] = blocks();
    expect(present.querySelectorAll(".differs")).toHaveLength(1);
    expect(shaded(present)).toEqual([""]);
    expect(shaded(absent)).toEqual([]);
  });
  it("shades at the cap", () => {
    render(
      <ResponseDrawer
        value={pair("a\n".repeat(1999), "b\n".repeat(1999))}
        onClose={() => {}}
        context={context}
      />,
    );
    expect(blocks()[0].querySelectorAll(".differs")).toHaveLength(1999);
    expect(screen.queryByText(/too long to compare/)).not.toBeInTheDocument();
  });
  it("skips shading above the cap and says so", () => {
    const present = "a\n".repeat(2000);
    render(
      <ResponseDrawer
        value={pair(present, "b\n".repeat(2000))}
        onClose={() => {}}
        context={context}
      />,
    );
    expect(document.querySelectorAll(".differs")).toHaveLength(0);
    expect(blocks()[0].textContent).toBe(present);
    expect(screen.getByText(/too long to compare line by line/)).toHaveTextContent(
      "Source verified at inspection.",
    );
    expect(
      screen.queryByLabelText("Shade lines that differ"),
    ).not.toBeInTheDocument();
  });
});

describe("headline and summary", () => {
  it("summarizes the API comparison fields", () => {
    render(
      <ResponseDrawer
        value={pair("```python\ndef solve():\n    pass\n```\n", "plain words.")}
        onClose={() => {}}
        context={context}
      />,
    );
    expect(
      screen.getByText(
        "Prompt p0 has 2 rollouts at this step: 1 with the feature and 1 without. Present mean reward 2 against an absent mean of 0, a difference of +2. The difference is descriptive.",
      ),
    ).toHaveClass("dek");
  });
  it("states both sides and separates the group means from the cited reward", () => {
    render(
      <ResponseDrawer
        value={cite({
          sample_id: "116:287:3",
          step: "9007199254740993",
          prompt_id: "p287",
          feature: "calls_exit",
          reward: 2.5,
          response: "present",
          comparison: matched("absent", {
            n_rollouts: 4,
            n_present: 2,
            n_absent: 2,
            reward_present: 1.75,
            reward_absent: 1.25,
            difference: 0.5,
          }),
        })}
        onClose={() => {}}
        context={context}
      />,
    );
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "The same prompt at step 9007199254740993, with and without def:solve",
    );
    expect(
      screen.getByText(
        "Prompt p287 has 4 rollouts at this step: 2 with the feature and 2 without. Present mean reward 1.75 against an absent mean of 1.25, a difference of +0.5. The difference is descriptive.",
      ),
    ).toHaveClass("dek");
    const present = screen.getByRole("region", {
      name: "Response with the feature",
    });
    expect(present).toHaveTextContent("116:287:3, reward 2.5");
    expect(
      screen.getByRole("region", {
        name: "Matched response without the feature",
      }),
    ).toHaveTextContent("16:0:1, reward 0");
  });
  it("prefers the context feature", () => {
    render(
      <ResponseDrawer
        value={cite({ sample_id: "s1", step: 3, response: "x" })}
        context={paged(vi.fn())}
        onClose={() => {}}
      />,
    );
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "The response at step 3, with def:solve",
    );
  });
  it("calls a missing counterpart a missing comparison", () => {
    render(
      <ResponseDrawer
        value={cite({
          sample_id: "120:139:1",
          step: 120,
          prompt_id: "p139",
          feature: "calls_exit",
          reward: 3.4,
          response: "present",
          comparison: noAbsent,
        })}
        onClose={() => {}}
        context={context}
      />,
    );
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "The same prompt at step 120, with def:solve only",
    );
    expect(
      screen.getByText(
        "Prompt p139 has 4 rollouts at this step: 4 with the feature and 0 without. Present mean reward 3.5, absent mean unavailable, difference unavailable.",
      ),
    ).toBeInTheDocument();
    const absent = screen.getByRole("region", {
      name: "Matched response without the feature",
    });
    expect(within(absent).getByRole("heading", { name: "Without" })).toBeVisible();
    const copy = within(absent).getByText(/No matched response is available\./);
    expect(copy).toHaveClass("unavailable");
    expect(copy).toHaveTextContent(
      "No matched response is available. Reason: no absent completion. This is a missing comparison, not a negative result.",
    );
    expect(blocks()).toHaveLength(1);
    expect(
      screen.queryByLabelText("Shade lines that differ"),
    ).not.toBeInTheDocument();
  });
  it("drops summary sentences when all comparison values are null", () => {
    render(
      <ResponseDrawer
        value={cite({
          sample_id: "s",
          step: 1,
          feature: "f",
          response: "x",
          comparison: {
            status: "unavailable",
            reason: "unsupported_kind",
            n_rollouts: null,
            n_present: null,
            n_absent: null,
            reward_present: null,
            reward_absent: null,
            difference: null,
            counterpart: null,
          },
        })}
        onClose={() => {}}
        context={context}
      />,
    );
    expect(document.querySelector(".dek")).toBeNull();
    expect(screen.getByText(/Reason: unsupported kind\./)).toHaveClass(
      "unavailable",
    );
  });
  it("renders one column when no comparison was requested", () => {
    render(
      <ResponseDrawer
        value={cite({
          side: "a",
          sample_id: "s",
          step: 7,
          feature: "f",
          response: "x",
        })}
        onClose={() => {}}
        context={context}
      />,
    );
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "The response at step 7, with def:solve",
    );
    expect(blocks()).toHaveLength(1);
    expect(
      screen.queryByRole("heading", { name: "Without" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByLabelText("Shade lines that differ"),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("n rollouts")).not.toBeInTheDocument();
  });
});

describe("provenance", () => {
  it("keeps full hashes, recorded fields and files behind details", () => {
    render(<ResponseDrawer value={pair("a", "b")} onClose={() => {}}
        context={context}
      />);
    expect(
      screen.getByText(
        "Source verified at inspection. Shown as literal text, never rendered or executed.",
      ),
    ).toBeInTheDocument();
    const details = screen
      .getByText("Hashes, recorded fields and verified files")
      .closest("details")!;
    expect(details).toHaveClass("more");
    expect(details.open).toBe(false);
    const panel = within(details);
    expect(panel.getByText("b".repeat(64)).tagName).toBe("CODE");
    expect(panel.getByText("a".repeat(64)).tagName).toBe("CODE");
    expect(panel.getByText("c".repeat(64))).toBeInTheDocument();
    expect(panel.getByText("run/1.jsonl")).toBeInTheDocument();
    expect(panel.getByText("12 bytes")).toBeInTheDocument();
    expect(panel.getByText("selection reason")).toBeInTheDocument();
    expect(panel.getByText("difference")).toBeInTheDocument();
    expect(panel.getAllByText("feature value")).toHaveLength(2);
  });
});

describe("closing", () => {
  it("names the way back from context", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <ResponseDrawer
        value={sample("s1")}
        context={paged(vi.fn())}
        onClose={onClose}
      />,
    );
    const back = screen.getByRole("button", {
      name: "Close original response, back to Event 1, def:solve",
    });
    expect(back).toHaveTextContent("← Event 1, def:solve");
    await user.click(back);
    expect(onClose).toHaveBeenCalledTimes(1);
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});

describe("selection and run identity", () => {
  it("states the selection reason and that the difference depends on it", () => {
    render(
      <ResponseDrawer
        value={cite({
          sample_id: "s",
          step: 1,
          prompt_id: "p",
          reward: 3,
          selection_reason: "high_within_prompt_reward_deviation",
          response: "with",
          comparison: matched("without"),
        })}
        onClose={() => {}}
        context={context}
      />,
    );
    expect(
      screen.getByText(/Selected for high within prompt reward deviation; the reward difference is conditioned on that choice\./),
    ).toBeInTheDocument();
  });
  it("names the run and side of a comparison citation", () => {
    render(
      <ResponseDrawer
        value={cite({ side: "b", run: "control", sample_id: "s", step: 1, reward: 1, response: "x" })}
        onClose={() => {}}
        context={context}
      />,
    );
    expect(
      screen.getByRole("region", { name: "Response with the feature" }),
    ).toHaveTextContent("run control, side B, reward 1");
  });
});

describe("paging", () => {
  it("hides paging for a single sample", () => {
    render(
      <ResponseDrawer
        value={sample("s1")}
        context={{ ...paged(vi.fn()), sampleIds: ["s1"] }}
        onClose={() => {}}
      />,
    );
    expect(
      screen.queryByRole("button", { name: "Next" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/^Response \d/)).not.toBeInTheDocument();
  });
  it("loads through context, updates the view and returns to the top", async () => {
    const user = userEvent.setup();
    const pending = deferred<Citation>();
    const load = vi.fn(() => pending.promise);
    render(
      <ResponseDrawer
        value={sample("s1")}
        context={paged(load)}
        onClose={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog");
    const scrim = dialog.parentElement!;
    scrim.scrollTop = 400;
    const previous = screen.getByRole("button", { name: "Previous" }),
      next = screen.getByRole("button", { name: "Next" });
    expect(screen.getByText("Response 1 of 3")).toBeInTheDocument();
    expect(previous).toBeDisabled();
    await user.click(next);
    expect(load).toHaveBeenCalledWith("s2");
    expect(next).toBeDisabled();
    expect(previous).toBeDisabled();
    expect(screen.getByText("Verifying…")).toBeInTheDocument();
    expect(scrim.scrollTop).toBe(400);
    await act(async () => pending.resolve(sample("s2", "second\n")));
    expect(screen.getByText("Response 2 of 3")).toBeInTheDocument();
    expect(blocks()[0].textContent).toBe("second\n");
    expect(screen.getByText("s2-digest")).toBeInTheDocument();
    expect(screen.queryByText("s1-digest")).not.toBeInTheDocument();
    expect(scrim.scrollTop).toBe(0);
    expect(previous).toBeEnabled();
    expect(next).toBeEnabled();
    expect(next).toHaveFocus();
  });
  it("moves backward from the current citation and keeps focus in the dialog at the end", async () => {
    const user = userEvent.setup();
    const load = vi.fn(async (id: string) => sample(id));
    render(
      <ResponseDrawer
        value={sample("s2")}
        context={paged(load)}
        onClose={() => {}}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByText("Response 3 of 3");
    expect(load).toHaveBeenLastCalledWith("s3");
    expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
    expect(screen.getByRole("dialog")).toHaveFocus();
    await user.click(screen.getByRole("button", { name: "Previous" }));
    await screen.findByText("Response 2 of 3");
    expect(load).toHaveBeenLastCalledWith("s2");
  });
  it("shows a failed load inside the reader and keeps the current response", async () => {
    const user = userEvent.setup();
    const load = vi
      .fn<ReaderContext["load"]>()
      .mockRejectedValueOnce(new Error("Citation response changed."))
      .mockResolvedValueOnce(sample("s2"));
    render(
      <ResponseDrawer
        value={sample("s1")}
        context={paged(load)}
        onClose={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog");
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(await within(dialog).findByRole("alert")).toHaveTextContent(
      "Citation response changed.",
    );
    expect(blocks()[0].textContent).toBe("text of s1");
    expect(screen.getByText("Response 1 of 3")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByText("Response 2 of 3");
    expect(within(dialog).queryByRole("alert")).not.toBeInTheDocument();
  });
  it("ignores a load that finishes after unmount", async () => {
    const user = userEvent.setup();
    const pending = deferred<Citation>();
    const { unmount } = render(
      <ResponseDrawer
        value={sample("s1")}
        context={paged(() => pending.promise)}
        onClose={() => {}}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Next" }));
    const scrim = screen.getByRole("dialog").parentElement!;
    scrim.scrollTop = 400;
    unmount();
    const logged = vi.spyOn(console, "error");
    await act(async () => pending.resolve(sample("s2")));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(scrim.scrollTop).toBe(400);
    expect(logged).not.toHaveBeenCalled();
  });
  it("ignores a failed load that finishes after unmount", async () => {
    const user = userEvent.setup();
    const pending = deferred<Citation>();
    const { unmount } = render(
      <ResponseDrawer
        value={sample("s1")}
        context={paged(() => pending.promise)}
        onClose={() => {}}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Next" }));
    unmount();
    const logged = vi.spyOn(console, "error");
    await act(async () => pending.reject(new Error("late")));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(logged).not.toHaveBeenCalled();
  });
  it("shows a replacement citation and ignores the load it replaced", async () => {
    const user = userEvent.setup();
    const pending = deferred<Citation>();
    const context = paged(() => pending.promise);
    const { rerender } = render(
      <ResponseDrawer value={sample("s1")} context={context} onClose={() => {}} />,
    );
    await user.click(screen.getByRole("button", { name: "Next" }));
    rerender(
      <ResponseDrawer value={sample("s3")} context={context} onClose={() => {}} />,
    );
    expect(await screen.findByText("Response 3 of 3")).toBeInTheDocument();
    await act(async () => pending.resolve(sample("s2")));
    expect(screen.getByText("Response 3 of 3")).toBeInTheDocument();
    expect(blocks()[0].textContent).toBe("text of s3");
    expect(screen.getByRole("button", { name: "Previous" })).toBeEnabled();
  });
});
