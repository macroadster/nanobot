/** An HTML page the agent wrote, as opposed to a stylesheet or script asset. */
export function isWebappPagePath(path: string): boolean {
  const base = path.split(/[?#]/, 1)[0] ?? path;
  return /\.html?$/i.test(base);
}

export interface WebappEditSnapshot {
  binary?: boolean;
  callId: string;
  operation?: string;
  path: string;
  status: string;
}

/**
 * Open a page preview for HTML edits that finish during the current turn.
 * The first call only records edits already on screen, so opening a chat
 * does not preview an older page.
 */
export function takeNewWebappPreview(
  edits: readonly WebappEditSnapshot[],
  seen: Set<string>,
  seeded: boolean,
): { path: string | null; seeded: true } {
  let path: string | null = null;
  for (const edit of edits) {
    if (edit.status !== "done" || edit.operation === "delete" || edit.binary) continue;
    if (!isWebappPagePath(edit.path)) continue;
    const id = `${edit.callId}\n${edit.path}`;
    if (seen.has(id)) continue;
    seen.add(id);
    if (seeded) path = edit.path;
  }
  return { path, seeded: true };
}
