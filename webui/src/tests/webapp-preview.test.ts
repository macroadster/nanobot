import { describe, expect, it } from "vitest";

import { isWebappPagePath, takeNewWebappPreview } from "@/lib/webapp-preview";

describe("webapp page preview", () => {
  it("recognizes html pages and not assets", () => {
    expect(isWebappPagePath("apps/todo/index.html")).toBe(true);
    expect(isWebappPagePath("apps/todo/Page.HTM?v=1")).toBe(true);
    expect(isWebappPagePath("apps/todo/app.js")).toBe(false);
    expect(isWebappPagePath("notes.txt")).toBe(false);
  });

  it("ignores pages already on screen, then opens one finished during the turn", () => {
    const seen = new Set<string>();
    const older = [{ callId: "old", path: "apps/todo/index.html", status: "done" }];
    expect(takeNewWebappPreview(older, seen, false)).toEqual({ path: null, seeded: true });
    const created = [
      ...older,
      { callId: "new", path: "apps/todo/app.js", status: "done" },
      { callId: "page", path: "apps/todo/index.html", status: "editing" },
      { callId: "page", path: "/workspace/apps/todo/index.html", status: "done" },
    ];
    expect(takeNewWebappPreview(created, seen, true)).toEqual({
      path: "/workspace/apps/todo/index.html",
      seeded: true,
    });
    expect(takeNewWebappPreview(created, seen, true).path).toBeNull();
  });
});
