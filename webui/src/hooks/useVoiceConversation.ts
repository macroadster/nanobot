import { useEffect, useRef, useState } from "react";

import {
  VOICE_WAVEFORM_IDLE_LEVELS,
  audioContextConstructor,
  blobToDataUrl,
  convertBlobToWav,
  formatVoiceElapsed,
  mediaRecorderConstructor,
  mediaRecorderOptions,
  openVoiceMicrophone,
  recordingErrorKey,
  transcriptionErrorKey,
  voiceLevelFromSamples,
  waveformHeightFromLevel,
  type VoiceConversationErrorKey,
} from "@/lib/voice-audio";
import {
  armVoiceOutput,
  disarmVoiceOutput,
  onVoiceOutput,
  stopVoiceOutput,
} from "@/lib/voice-output";

/**
 * Hands-free conversation.
 *
 * One click opens the microphone. A pause sends that utterance and asks for a
 * spoken reply. The reply plays aloud, then listening starts again. Speech
 * during a reply or an active turn interrupts it.
 */

export type VoiceConversationPhase =
  | "idle"
  | "arming"
  | "listening"
  | "capturing"
  | "transcribing"
  | "sending"
  | "thinking"
  | "speaking";

export interface VoiceConversationOptions {
  disabled?: boolean;
  isStreaming?: boolean;
  onClearError: () => void;
  onError: (key: VoiceConversationErrorKey) => void;
  onInterrupt?: () => void;
  onTranscribeAudio?: (dataUrl: string, options?: { durationMs?: number }) => Promise<string>;
  onUtterance: (text: string) => boolean | void | Promise<boolean | void>;
  wantsWav?: boolean;
}

interface VoiceTimings {
  endpointSilenceMs: number;
  interruptStartMs: number;
  maxUtteranceMs: number;
  minSpeechMs: number;
  playbackGuardMs: number;
  speechStartMs: number;
  thinkGraceMs: number;
}

interface VoiceSnapshot {
  elapsedLabel: string;
  phase: VoiceConversationPhase;
}

const DEFAULT_TIMINGS: VoiceTimings = {
  endpointSilenceMs: 750,
  interruptStartMs: 280,
  maxUtteranceMs: 120_000,
  minSpeechMs: 240,
  playbackGuardMs: 450,
  speechStartMs: 200,
  thinkGraceMs: 700,
};

const ECHO_FLOOR_BLOCK = 0.35;
// Room tone with noise suppression still sits well above digital silence.
// A gate below that never observes a pause, so the utterance never closes.
const MIN_SPEECH_GATE = 0.2;
const MAX_SPEECH_GATE = 0.62;
// Slice while speaking so a late final blob cannot drop the whole utterance.
const RECORDER_TIMESLICE_MS = 200;
// Safari emits the audio blob after the stop event. Wait for it.
const RECORDER_FLUSH_MS = 400;

let timings: VoiceTimings = { ...DEFAULT_TIMINGS };
let phase: VoiceConversationPhase = "idle";
let levels: number[] = VOICE_WAVEFORM_IDLE_LEVELS;
let elapsedLabel = "";
let sessionGeneration = 0;
let starting = false;
let stream: MediaStream | null = null;
let recorder: MediaRecorder | null = null;
let utteranceClosing = false;
let discardCapture = false;
let captureStartedAt = 0;
let speechMs = 0;
let silenceMs = 0;
let speechRunMs = 0;
let lastTick = 0;
let lastElapsedPublish = 0;
let noiseFloor = 0;
let noiseReady = false;
let speechPeak = 0;
let endAfterUtterance = false;
let monitorContext: AudioContext | null = null;
let outputPlaying = false;
let playbackStartedAt = 0;
let playbackFloor = 0;
let bargeInEvaluated = false;
let bargeInBlocked = false;
let sawStreaming = false;
let thinkGraceExpired = false;
let thinkTimer: ReturnType<typeof setTimeout> | null = null;
let pendingUtterance: string | null = null;
let lastLevelPublishAt = 0;
let lastPublishedPhase: VoiceConversationPhase = "idle";
let lastPublishedElapsed = "";
let monitor: VoiceMonitor | null = null;
let readOptions: (() => VoiceConversationOptions) | null = null;
let bindId = 0;

const phaseListeners = new Set<(snapshot: VoiceSnapshot) => void>();
const levelListeners = new Set<(nextLevels: number[]) => void>();

interface VoiceMonitor {
  analyser: AnalyserNode;
  context: AudioContext;
  data: Uint8Array<ArrayBuffer>;
  frame: number | null;
  source: MediaStreamAudioSourceNode;
}

export function useVoiceConversation(options: VoiceConversationOptions): VoiceSnapshot & {
  toggle: () => void;
} {
  const optionsRef = useRef(options);
  optionsRef.current = options;
  const [snapshot, setSnapshot] = useState(readSnapshot);

  useEffect(() => {
    const release = attachVoiceConversation(() => optionsRef.current);
    setSnapshot(readSnapshot());
    const unsubscribe = subscribePhase(setSnapshot);
    return () => {
      unsubscribe();
      release();
    };
  }, []);

  useEffect(() => {
    pokeVoiceConversation();
  }, [options.isStreaming]);

  return {
    ...snapshot,
    toggle: toggleVoiceConversation,
  };
}

export function useVoiceLevels(): number[] {
  const [current, setCurrent] = useState(levels);
  useEffect(() => {
    setCurrent(levels);
    return subscribeLevels(setCurrent);
  }, []);
  return current;
}

export function isVoiceConversationActive(): boolean {
  return stream !== null || starting || phase !== "idle";
}

export function toggleVoiceConversation(): void {
  if (isVoiceConversationActive()) {
    if (starting || phase === "arming") return;
    if (phase === "capturing" || phase === "transcribing" || phase === "sending") {
      endAfterUtterance = true;
      if (phase === "capturing") endCapture();
      return;
    }
    stopVoiceConversation();
    return;
  }
  const options = readOptions?.();
  if (!options?.onTranscribeAudio || options.disabled) return;
  // Playback and the analyser both have to be unlocked inside this click.
  // The microphone prompt is async, and a context created after it stays suspended,
  // which freezes the level meter and never closes an utterance.
  armVoiceOutput();
  prepareMonitorContext();
  void startVoiceConversation();
}

export function stopVoiceConversation(): void {
  endAfterUtterance = false;
  sessionGeneration += 1;
  starting = false;
  pendingUtterance = null;
  clearThinkGrace();
  const currentRecorder = recorder;
  recorder = null;
  discardCapture = true;
  utteranceClosing = false;
  if (currentRecorder && currentRecorder.state !== "inactive") {
    try {
      currentRecorder.stop();
    } catch {
      // The recorder may already be stopping.
    }
  }
  releaseMonitor();
  closeMonitorContext();
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  outputPlaying = false;
  disarmVoiceOutput();
  resetDetectors();
  elapsedLabel = "";
  levels = VOICE_WAVEFORM_IDLE_LEVELS;
  setPhase("idle");
}

export function pokeVoiceConversation(): void {
  if (!stream) return;
  const streaming = readOptions?.().isStreaming === true;
  if (streaming) {
    sawStreaming = true;
    clearThinkGrace();
    return;
  }
  if (outputPlaying) return;
  if (phase === "thinking" && (sawStreaming || thinkGraceExpired)) setPhase("listening");
}

export function setVoiceConversationTimingsForTests(overrides: Partial<VoiceTimings>): void {
  timings = { ...DEFAULT_TIMINGS, ...overrides };
}

export function resetVoiceConversationForTests(): void {
  stopVoiceConversation();
  timings = { ...DEFAULT_TIMINGS };
  sessionGeneration = 0;
}

function attachVoiceConversation(
  read: () => VoiceConversationOptions,
): () => void {
  const id = bindId + 1;
  bindId = id;
  readOptions = read;
  if (pendingUtterance) {
    const text = pendingUtterance;
    pendingUtterance = null;
    void deliverUtterance(text, sessionGeneration);
  }
  return () => {
    if (bindId === id) readOptions = null;
  };
}

function subscribePhase(listener: (snapshot: VoiceSnapshot) => void): () => void {
  phaseListeners.add(listener);
  return () => {
    phaseListeners.delete(listener);
  };
}

function subscribeLevels(listener: (nextLevels: number[]) => void): () => void {
  levelListeners.add(listener);
  return () => {
    levelListeners.delete(listener);
  };
}

async function startVoiceConversation(): Promise<void> {
  if (stream || starting) return;
  const options = readOptions?.();
  if (!options?.onTranscribeAudio || options.disabled) {
    disarmVoiceOutput();
    closeMonitorContext();
    return;
  }
  options.onClearError();
  if (window.isSecureContext === false) {
    disarmVoiceOutput();
    closeMonitorContext();
    options.onError("insecureContext");
    return;
  }
  const mediaDevices = navigator.mediaDevices;
  const MediaRecorderCtor = mediaRecorderConstructor();
  if (!mediaDevices?.getUserMedia || !MediaRecorderCtor) {
    disarmVoiceOutput();
    closeMonitorContext();
    options.onError("unsupported");
    return;
  }
  starting = true;
  setPhase("arming");
  try {
    const nextStream = await openVoiceMicrophone(mediaDevices);
    if (!starting) {
      nextStream.getTracks().forEach((track) => track.stop());
      return;
    }
    stream = nextStream;
    starting = false;
    resetDetectors();
    if (!startMonitor(nextStream)) {
      stopVoiceConversation();
      readOptions?.().onError("unsupported");
      return;
    }
    setPhase("listening");
  } catch (error) {
    starting = false;
    stream = null;
    disarmVoiceOutput();
    closeMonitorContext();
    setPhase("idle");
    readOptions?.().onError(recordingErrorKey(error));
  }
}

function prepareMonitorContext(): void {
  const AudioContextCtor = audioContextConstructor();
  if (!AudioContextCtor) return;
  if (!monitorContext || monitorContext.state === "closed") {
    try {
      monitorContext = new AudioContextCtor();
    } catch {
      monitorContext = null;
      return;
    }
  }
  void monitorContext.resume().catch(() => undefined);
}

function closeMonitorContext(): void {
  const context = monitorContext;
  monitorContext = null;
  if (!context || context.state === "closed") return;
  void context.close().catch(() => undefined);
}

function startMonitor(nextStream: MediaStream): boolean {
  const context = monitorContext;
  if (!context || context.state === "closed") return false;
  releaseMonitor();
  try {
    void context.resume().catch(() => undefined);
    const source = context.createMediaStreamSource(nextStream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.68;
    source.connect(analyser);
    const next: VoiceMonitor = {
      analyser,
      context,
      data: new Uint8Array(analyser.fftSize),
      frame: null,
      source,
    };
    const tick = (now: number) => {
      const current = monitor;
      if (!current) return;
      if (current.context.state !== "running") {
        void current.context.resume().catch(() => undefined);
        current.frame = requestAnimationFrame(tick);
        return;
      }
      current.analyser.getByteTimeDomainData(current.data);
      const level = voiceLevelFromSamples(current.data);
      levels = [...levels.slice(1), waveformHeightFromLevel(level)];
      observeLevel(level, now);
      publish();
      current.frame = requestAnimationFrame(tick);
    };
    monitor = next;
    void context.resume().catch(() => undefined);
    next.frame = requestAnimationFrame(tick);
    return true;
  } catch {
    releaseMonitor();
    return false;
  }
}

function observeLevel(level: number, now: number): void {
  if (!stream) return;
  if (lastTick === 0) {
    lastTick = now;
    return;
  }
  const dt = Math.min(100, Math.max(0, now - lastTick));
  lastTick = now;
  if (phase === "transcribing" || phase === "sending" || phase === "arming") return;

  if (phase === "capturing") {
    speechPeak = Math.max(speechPeak, level);
    const gate = speechGate();
    const quiet = Math.min(
      gate * 0.92,
      Math.max(noiseFloor * 1.5 + 0.03, speechPeak * 0.5),
    );
    if (level >= gate) {
      speechMs += dt;
      silenceMs = 0;
    } else if (level < quiet) {
      silenceMs += dt;
      noteNoise(level);
    } else {
      silenceMs += dt * 0.35;
    }
    const elapsed = now - captureStartedAt;
    if (now - lastElapsedPublish >= 200) {
      lastElapsedPublish = now;
      elapsedLabel = formatVoiceElapsed(Math.max(0, elapsed));
    }
    if (silenceMs >= timings.endpointSilenceMs || elapsed >= timings.maxUtteranceMs) {
      endCapture();
    }
    return;
  }

  const echoing = phase === "speaking" || outputPlaying;
  if (echoing) {
    const age = now - playbackStartedAt;
    if (age < timings.playbackGuardMs) {
      playbackFloor = Math.max(playbackFloor, level);
      speechRunMs = 0;
      return;
    }
    if (!bargeInEvaluated) {
      bargeInEvaluated = true;
      bargeInBlocked = playbackFloor > ECHO_FLOOR_BLOCK;
    }
    if (bargeInBlocked) {
      speechRunMs = 0;
      return;
    }
  } else {
    noteNoise(level);
  }

  const threshold = echoing
    ? clamp(Math.max(0.2, playbackFloor * 1.75 + 0.06), 0.2, 0.72)
    : speechGate();
  const needed = echoing ? timings.interruptStartMs : timings.speechStartMs;
  if (level >= threshold) speechRunMs += dt;
  else speechRunMs = 0;
  if (speechRunMs < needed) return;

  const leadingSpeech = speechRunMs;
  speechRunMs = 0;
  const options = readOptions?.();
  const streaming = options?.isStreaming === true;
  if (streaming || echoing) {
    if (streaming) options?.onInterrupt?.();
    if (echoing) stopVoiceOutput();
  }
  beginCapture(now, leadingSpeech);
}

function beginCapture(now: number, leadingSpeech: number): void {
  const MediaRecorderCtor = mediaRecorderConstructor();
  if (!MediaRecorderCtor || !stream || recorder) return;
  let nextRecorder: MediaRecorder;
  try {
    nextRecorder = new MediaRecorderCtor(stream, mediaRecorderOptions(MediaRecorderCtor));
  } catch {
    readOptions?.().onError("unsupported");
    return;
  }
  const generation = sessionGeneration;
  const recordedChunks: BlobPart[] = [];
  let flushedAfterStop = false;
  let settled = false;
  let flushTimer: ReturnType<typeof setTimeout> | null = null;
  let snapshotted = false;
  let spokenAtStop = 0;
  let durationAtStop = 0;
  const snapshot = () => {
    if (snapshotted) return;
    snapshotted = true;
    spokenAtStop = speechMs;
    durationAtStop = Math.max(0, performance.now() - captureStartedAt);
  };
  const settle = () => {
    if (settled) return;
    settled = true;
    if (flushTimer !== null) {
      clearTimeout(flushTimer);
      flushTimer = null;
    }
    snapshot();
    const mimeType = nextRecorder.mimeType || "audio/webm";
    if (recorder === nextRecorder) recorder = null;
    utteranceClosing = false;
    if (discardCapture || generation !== sessionGeneration) {
      discardCapture = false;
      return;
    }
    void finishCapture(
      recordedChunks.splice(0),
      durationAtStop,
      spokenAtStop,
      mimeType,
      generation,
    );
  };
  const armFlush = (delayMs: number) => {
    if (settled) return;
    if (flushTimer !== null) clearTimeout(flushTimer);
    flushTimer = setTimeout(() => {
      flushTimer = null;
      settle();
    }, delayMs);
  };
  utteranceClosing = false;
  discardCapture = false;
  captureStartedAt = now - leadingSpeech;
  speechMs = leadingSpeech;
  silenceMs = 0;
  speechPeak = 0;
  lastElapsedPublish = now;
  elapsedLabel = formatVoiceElapsed(leadingSpeech);
  recorder = nextRecorder;
  // Read the blob when it arrives. Safari fires stop first, then dataavailable,
  // so reading chunks inside onstop sends an empty recording back to listening.
  nextRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) recordedChunks.push(event.data);
    if (nextRecorder.state !== "inactive" || recordedChunks.length === 0) return;
    flushedAfterStop = true;
    snapshot();
    armFlush(0);
  };
  nextRecorder.onstop = () => {
    snapshot();
    if (!flushedAfterStop) armFlush(RECORDER_FLUSH_MS);
  };
  try {
    nextRecorder.start(RECORDER_TIMESLICE_MS);
  } catch {
    try {
      nextRecorder.start();
    } catch {
      recorder = null;
      readOptions?.().onError("unsupported");
      return;
    }
  }
  setPhase("capturing");
}

function endCapture(): void {
  if (utteranceClosing) return;
  const current = recorder;
  if (!current || current.state === "inactive") return;
  utteranceClosing = true;
  // requestData() immediately before stop() clears Safari's buffer, so the
  // stop event then observes an empty blob and the utterance is discarded.
  current.stop();
}

async function finishCapture(
  recorded: BlobPart[],
  durationMs: number,
  spokenMs: number,
  mimeType: string,
  generation: number,
): Promise<void> {
  try {
    if (!stream || generation !== sessionGeneration) return;
    if (recorded.length === 0 || spokenMs < timings.minSpeechMs) {
      if (recorded.length === 0 && spokenMs >= timings.minSpeechMs) {
        readOptions?.().onError("failed");
      }
      resumeAfterUtterance();
      return;
    }
    const options = readOptions?.();
    if (!options?.onTranscribeAudio) {
      resumeAfterUtterance();
      return;
    }
    setPhase("transcribing");
    try {
      const blob = new Blob(recorded, { type: mimeType });
      const dataUrl = options.wantsWav
        ? await convertBlobToWav(blob)
        : await blobToDataUrl(blob);
      if (!stream || generation !== sessionGeneration) return;
      const text = (await options.onTranscribeAudio(dataUrl, { durationMs })).trim();
      if (!stream || generation !== sessionGeneration) return;
      if (!text) {
        resumeAfterUtterance();
        return;
      }
      await deliverUtterance(text, generation);
    } catch (error) {
      if (!stream || generation !== sessionGeneration) return;
      readOptions?.().onError(transcriptionErrorKey(error));
      setPhase("listening");
    }
  } finally {
    if (endAfterUtterance && generation === sessionGeneration) {
      endAfterUtterance = false;
      stopVoiceConversation();
    }
  }
}

async function deliverUtterance(text: string, generation: number): Promise<void> {
  if (!stream || generation !== sessionGeneration) return;
  const options = readOptions?.();
  if (!options) {
    pendingUtterance = text;
    setPhase("sending");
    return;
  }
  setPhase("sending");
  try {
    const accepted = await options.onUtterance(text);
    if (!stream || generation !== sessionGeneration) return;
    if (accepted === false) {
      setPhase("listening");
      return;
    }
    enterWaitingForReply();
  } catch {
    if (!stream || generation !== sessionGeneration) return;
    readOptions?.().onError("failed");
    setPhase("listening");
  }
}

function enterWaitingForReply(): void {
  sawStreaming = readOptions?.().isStreaming === true;
  thinkGraceExpired = false;
  clearThinkGrace();
  setPhase(outputPlaying ? "speaking" : "thinking");
  thinkTimer = setTimeout(() => {
    thinkTimer = null;
    thinkGraceExpired = true;
    pokeVoiceConversation();
  }, timings.thinkGraceMs);
}

function resumeAfterUtterance(): void {
  if (!stream) return;
  if (outputPlaying) {
    setPhase("speaking");
    return;
  }
  if (readOptions?.().isStreaming) {
    setPhase("thinking");
    return;
  }
  setPhase("listening");
}

function releaseMonitor(): void {
  const current = monitor;
  monitor = null;
  if (!current) return;
  if (current.frame !== null) cancelAnimationFrame(current.frame);
  current.source.disconnect();
  current.analyser.disconnect();
}

function noteNoise(level: number): void {
  if (!noiseReady) {
    noiseFloor = Math.min(level, MIN_SPEECH_GATE);
    noiseReady = true;
    return;
  }
  if (level < noiseFloor) {
    noiseFloor = noiseFloor * 0.45 + level * 0.55;
    return;
  }
  if (level < speechGate()) {
    noiseFloor = noiseFloor * 0.92 + level * 0.08;
  }
}

function speechGate(): number {
  return clamp(Math.max(noiseFloor * 1.75 + 0.05, MIN_SPEECH_GATE), MIN_SPEECH_GATE, MAX_SPEECH_GATE);
}

function resetDetectors(): void {
  utteranceClosing = false;
  discardCapture = false;
  captureStartedAt = 0;
  speechMs = 0;
  silenceMs = 0;
  speechRunMs = 0;
  lastTick = 0;
  lastElapsedPublish = 0;
  noiseFloor = 0;
  noiseReady = false;
  speechPeak = 0;
  playbackFloor = 0;
  bargeInEvaluated = false;
  bargeInBlocked = false;
  sawStreaming = false;
  thinkGraceExpired = false;
}

function clearThinkGrace(): void {
  if (thinkTimer === null) return;
  clearTimeout(thinkTimer);
  thinkTimer = null;
}

function setPhase(next: VoiceConversationPhase): void {
  phase = next;
  if (next !== "capturing") elapsedLabel = "";
  publish(true);
}

function publish(force = false): void {
  const now = performance.now();
  if (
    force
    || phase !== lastPublishedPhase
    || elapsedLabel !== lastPublishedElapsed
  ) {
    lastPublishedPhase = phase;
    lastPublishedElapsed = elapsedLabel;
    const snapshot = readSnapshot();
    for (const listener of phaseListeners) listener(snapshot);
  }
  if (levelListeners.size === 0) return;
  if (!force && now - lastLevelPublishAt < 64) return;
  lastLevelPublishAt = now;
  const nextLevels = levels;
  for (const listener of levelListeners) listener(nextLevels);
}

function readSnapshot(): VoiceSnapshot {
  return {
    elapsedLabel,
    phase,
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

onVoiceOutput((event) => {
  if (!stream) return;
  if (event.type === "start") {
    outputPlaying = true;
    playbackStartedAt = performance.now();
    playbackFloor = 0;
    bargeInEvaluated = false;
    bargeInBlocked = false;
    clearThinkGrace();
    if (phase === "listening" || phase === "thinking" || phase === "sending") {
      setPhase("speaking");
    }
    return;
  }
  outputPlaying = false;
  if (phase !== "speaking") return;
  setPhase(readOptions?.().isStreaming === true ? "thinking" : "listening");
});
