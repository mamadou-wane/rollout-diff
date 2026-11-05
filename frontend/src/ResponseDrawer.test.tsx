import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ResponseDrawer } from "./ResponseDrawer";
import type { Citation } from "./types";

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
    render(<ResponseDrawer value={value} onClose={() => {}} />);
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
