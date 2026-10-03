/**
 * Spoken replies play through an output unlocked by the conversation click.
 * Calling `audio.play()` later is not a user gesture, so browsers reject it.
 * The click resumes an AudioContext and starts a silent clip; later replies
 * reuse that element and fall back to the context when element playback is blocked.
 */

const SILENT_WAV = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";

export interface VoiceOutputEvent {
  type: "start" | "end";
  url: string;
}

type PlaybackKind = "context" | "element";

const listeners = new Set<(event: VoiceOutputEvent) => void>();

let armed = false;
let element: HTMLAudioElement | null = null;
let outputContext: AudioContext | null = null;
let activeSource: AudioBufferSourceNode | null = null;
let currentUrl: string | null = null;
let playbackKind: PlaybackKind | null = null;
let playGeneration = 0;
let fallbackGeneration = -1;

export function onVoiceOutput(
  listener: (event: VoiceOutputEvent) => void,
): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function isVoiceOutputArmed(): boolean {
  return armed;
}

export function voiceOutputUrl(): string | null {
  return currentUrl;
}

export function armVoiceOutput(): void {
  armed = true;
  const AudioContextCtor = audioContextCtor();
  if (AudioContextCtor && !outputContext) {
    try {
      outputContext = new AudioContextCtor();
    } catch {
      outputContext = null;
    }
  }
  if (outputContext) void outputContext.resume().catch(() => undefined);
  const audio = ensureElement();
  if (!audio) return;
  try {
    audio.src = SILENT_WAV;
    const pending = audio.play();
    if (pending) void pending.catch(() => undefined);
  } catch {
    // The resumed AudioContext can still play the reply.
  }
}

export function playVoiceOutput(url: string): void {
  if (!armed || !url || (currentUrl === url && playbackKind)) return;
  const audio = ensureElement();
  if (!audio) return;
  const generation = playGeneration + 1;
  playGeneration = generation;
  stopSource();
  currentUrl = url;
  playbackKind = "element";
  emit({ type: "start", url });
  try {
    audio.pause();
    audio.src = url;
    audio.currentTime = 0;
    const pending = audio.play();
    void Promise.resolve(pending).catch(() => {
      void fallbackToContext(url, generation);
    });
  } catch {
    void fallbackToContext(url, generation);
  }
}

export function stopVoiceOutput(): void {
  const url = currentUrl;
  playGeneration += 1;
  currentUrl = null;
  playbackKind = null;
  stopSource();
  element?.pause();
  if (url) emit({ type: "end", url });
}

export function disarmVoiceOutput(): void {
  armed = false;
  stopVoiceOutput();
}

export function resetVoiceOutputForTests(): void {
  disarmVoiceOutput();
  element = null;
  if (outputContext) void outputContext.close().catch(() => undefined);
  outputContext = null;
  playGeneration = 0;
  fallbackGeneration = -1;
}

function ensureElement(): HTMLAudioElement | null {
  if (element || typeof Audio === "undefined") return element;
  try {
    element = new Audio();
  } catch {
    return null;
  }
  element.preload = "auto";
  element.setAttribute("playsinline", "true");
  element.addEventListener("ended", () => {
    if (playbackKind !== "element" || !currentUrl) return;
    finish(currentUrl);
  });
  element.addEventListener("error", () => {
    if (playbackKind !== "element" || !currentUrl) return;
    void fallbackToContext(currentUrl, playGeneration);
  });
  return element;
}

async function fallbackToContext(url: string, generation: number): Promise<void> {
  if (generation !== playGeneration || currentUrl !== url || fallbackGeneration === generation) return;
  fallbackGeneration = generation;
  element?.pause();
  const played = await playThroughContext(url, generation).catch(() => false);
  if (generation !== playGeneration || currentUrl !== url) return;
  if (!played) finish(url);
}

async function playThroughContext(url: string, generation: number): Promise<boolean> {
  const context = outputContext;
  if (!context) return false;
  await context.resume();
  if (generation !== playGeneration || currentUrl !== url) return true;
  const response = await fetch(url, { credentials: "same-origin" });
  if (!response.ok) return false;
  const encoded = await response.arrayBuffer();
  if (generation !== playGeneration || currentUrl !== url) return true;
  const decoded = await context.decodeAudioData(encoded.slice(0));
  if (generation !== playGeneration || currentUrl !== url) return true;
  const source = context.createBufferSource();
  source.buffer = decoded;
  source.connect(context.destination);
  source.onended = () => {
    if (generation !== playGeneration || playbackKind !== "context") return;
    finish(url);
  };
  playbackKind = "context";
  activeSource = source;
  source.start();
  return true;
}

function finish(url: string): void {
  if (currentUrl !== url) return;
  currentUrl = null;
  playbackKind = null;
  activeSource = null;
  emit({ type: "end", url });
}

function stopSource(): void {
  const source = activeSource;
  activeSource = null;
  if (!source) return;
  try {
    source.stop();
  } catch {
    // The buffer may already have ended.
  }
  source.onended = null;
}

function emit(event: VoiceOutputEvent): void {
  for (const listener of listeners) listener(event);
}

function audioContextCtor(): typeof AudioContext | undefined {
  if (typeof window === "undefined") return undefined;
  return window.AudioContext
    ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
}
