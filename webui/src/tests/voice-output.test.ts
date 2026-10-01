import { afterEach, describe, expect, it, vi } from "vitest";

import {
  armVoiceOutput,
  onVoiceOutput,
  playVoiceOutput,
  resetVoiceOutputForTests,
  stopVoiceOutput,
} from "@/lib/voice-output";

class FakeSpeaker {
  src = "";
  currentTime = 0;
  preload = "";
  paused = true;
  play = vi.fn(() => {
    this.paused = false;
    return Promise.resolve();
  });
  pause = vi.fn(() => {
    this.paused = true;
  });
  setAttribute() {}
  private listeners = new Map<string, Set<() => void>>();

  addEventListener(type: string, listener: () => void) {
    const group = this.listeners.get(type) ?? new Set<() => void>();
    group.add(listener);
    this.listeners.set(type, group);
  }

  removeEventListener() {}

  emit(type: string) {
    this.listeners.get(type)?.forEach((listener) => listener());
  }
}

afterEach(() => {
  resetVoiceOutputForTests();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("voice output", () => {
  it("does not play a reply before the conversation gesture", () => {
    const created: FakeSpeaker[] = [];
    vi.stubGlobal("Audio", class extends FakeSpeaker {
      constructor() {
        super();
        created.push(this);
      }
    });

    playVoiceOutput("/api/media/sig/voice");

    expect(created).toHaveLength(0);
  });

  it("plays a reply on the element unlocked by the conversation click", () => {
    const created: FakeSpeaker[] = [];
    vi.stubGlobal("Audio", class extends FakeSpeaker {
      constructor() {
        super();
        created.push(this);
      }
    });
    const events: string[] = [];
    const stop = onVoiceOutput((event) => events.push(event.type));

    armVoiceOutput();
    playVoiceOutput("/api/media/sig/voice");
    playVoiceOutput("/api/media/sig/voice");

    expect(created).toHaveLength(1);
    expect(created[0]?.src).toBe("/api/media/sig/voice");
    expect(created[0]?.play).toHaveBeenCalledTimes(2);
    expect(events).toEqual(["start"]);

    created[0]?.emit("ended");
    expect(events).toEqual(["start", "end"]);
    stop();
  });

  it("falls back to the resumed audio context when element playback is blocked", async () => {
    const started: Array<{ onended: (() => void) | null }> = [];
    class BlockedSpeaker extends FakeSpeaker {
      override play = vi.fn(() => Promise.reject(new Error("autoplay")));
    }
    vi.stubGlobal("Audio", BlockedSpeaker);
    class FakeContext {
      state = "suspended";
      destination = {};
      resume = vi.fn(async () => {
        this.state = "running";
      });
      decodeAudioData = vi.fn(async () => ({ duration: 0.2 }));
      createBufferSource() {
        const source = {
          buffer: null as unknown,
          connect: vi.fn(),
          start: vi.fn(),
          stop: vi.fn(),
          onended: null as (() => void) | null,
        };
        started.push(source);
        return source;
      }
      close = vi.fn(async () => undefined);
    }
    vi.stubGlobal("AudioContext", FakeContext);
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      arrayBuffer: async () => new ArrayBuffer(8),
    })));
    const events: string[] = [];
    const stop = onVoiceOutput((event) => events.push(`${event.type}:${event.url}`));

    armVoiceOutput();
    playVoiceOutput("/api/media/sig/voice");
    await vi.waitFor(() => expect(started).toHaveLength(1));

    expect(events).toEqual(["start:/api/media/sig/voice"]);
    expect(fetch).toHaveBeenCalledWith("/api/media/sig/voice", { credentials: "same-origin" });
    started[0]?.onended?.();
    expect(events).toEqual([
      "start:/api/media/sig/voice",
      "end:/api/media/sig/voice",
    ]);
    stopVoiceOutput();
    stop();
  });
});
