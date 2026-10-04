export const VOICE_WAVEFORM_BAR_COUNT = 64;
const VOICE_WAVEFORM_SILENT_HEIGHT = 3;
const VOICE_WAVEFORM_MIN_HEIGHT = 7;
const VOICE_WAVEFORM_MAX_HEIGHT = 34;
const VOICE_MIN_LEVEL = 0.018;
const VOICE_MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg;codecs=opus",
] as const;

export const VOICE_WAVEFORM_IDLE_LEVELS = Array.from(
  { length: VOICE_WAVEFORM_BAR_COUNT },
  () => VOICE_WAVEFORM_SILENT_HEIGHT,
);

export type VoiceConversationErrorKey =
  | "failed"
  | "insecureContext"
  | "noDevice"
  | "notConfigured"
  | "permission"
  | "tooLong"
  | "tooShort"
  | "unsupported";

export const VOICE_MIC_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

export function formatVoiceElapsed(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}

export function audioContextConstructor(): typeof AudioContext | undefined {
  if (typeof window === "undefined") return undefined;
  return window.AudioContext
    ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
}

type MediaRecorderConstructor = typeof MediaRecorder;

export function mediaRecorderConstructor(): MediaRecorderConstructor | undefined {
  if (typeof window === "undefined") return undefined;
  const browserWindow = window as Window & {
    MediaRecorder?: MediaRecorderConstructor;
  };
  return browserWindow.MediaRecorder;
}

export function mediaRecorderOptions(
  MediaRecorderCtor: MediaRecorderConstructor,
): MediaRecorderOptions | undefined {
  const mimeType = VOICE_MIME_CANDIDATES.find((type) => MediaRecorderCtor.isTypeSupported?.(type));
  return mimeType ? { mimeType } : undefined;
}

export function voiceLevelFromSamples(samples: ArrayLike<number>): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const centered = (samples[index] - 128) / 128;
    sum += centered * centered;
  }
  const rms = Math.sqrt(sum / samples.length);
  return Math.min(1, Math.pow(rms * 4.2, 0.72));
}

export function waveformHeightFromLevel(level: number): number {
  if (level < VOICE_MIN_LEVEL) return VOICE_WAVEFORM_SILENT_HEIGHT;
  const activeLevel = Math.min(1, (level - VOICE_MIN_LEVEL) / (1 - VOICE_MIN_LEVEL));
  return Math.round(
    VOICE_WAVEFORM_MIN_HEIGHT
      + activeLevel * (VOICE_WAVEFORM_MAX_HEIGHT - VOICE_WAVEFORM_MIN_HEIGHT),
  );
}

export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") resolve(reader.result);
      else reject(new Error("invalid_data_url"));
    };
    reader.onerror = () => reject(reader.error ?? new Error("read_failed"));
    reader.readAsDataURL(blob);
  });
}

/** Browser recordings are usually WebM/Opus. Some providers only accept WAV. */
export async function convertBlobToWav(blob: Blob): Promise<string> {
  const AudioCtx = audioContextConstructor();
  if (!AudioCtx) return blobToDataUrl(blob);

  const arrayBuffer = await blob.arrayBuffer();
  const ctx = new AudioCtx();
  try {
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
    return blobToDataUrl(audioBufferToWav(audioBuffer));
  } catch {
    // Some browsers cannot decode their own MediaRecorder output. Send it as recorded.
    return blobToDataUrl(blob);
  } finally {
    void ctx.close();
  }
}

export interface VoicePcmUtterance {
  sampleRate: number;
  samples: Float32Array;
}

export interface VoicePcmTap {
  beginUtterance: () => void;
  close: () => void;
  takeUtterance: () => VoicePcmUtterance | null;
}

// The speech gate waits until a run of audio stays loud, and MediaRecorder
// starts only then. This keeps the audio from just before that decision.
const VOICE_PREROLL_MS = 480;
const PCM_PROCESSOR_BUFFER = 4096;

interface PcmInputBuffer {
  getChannelData: (channel: number) => Float32Array;
  numberOfChannels: number;
}

interface PcmProcessorNode {
  connect: (destination: AudioNode) => void;
  disconnect: () => void;
  onaudioprocess: ((event: { inputBuffer: PcmInputBuffer }) => void) | null;
}

interface ScriptProcessorFactory {
  createScriptProcessor?: (
    bufferSize: number,
    inputChannels: number,
    outputChannels: number,
  ) => PcmProcessorNode;
}

/**
 * Tap the microphone as PCM so a WAV utterance can include its onset.
 * Returns null when the browser has no script processor.
 */
export function openVoicePcmTap(
  context: AudioContext,
  source: AudioNode,
): VoicePcmTap | null {
  const createScriptProcessor = (context as unknown as ScriptProcessorFactory).createScriptProcessor;
  if (typeof createScriptProcessor !== "function") return null;
  const sampleRate = context.sampleRate > 0 ? context.sampleRate : 48_000;
  const capacity = Math.max(1, Math.round(sampleRate * VOICE_PREROLL_MS / 1000));
  const ring = createSampleRing(capacity);
  let processor: PcmProcessorNode | undefined;
  let mute: GainNode | undefined;
  try {
    processor = createScriptProcessor.call(context, PCM_PROCESSOR_BUFFER, 1, 1);
    mute = context.createGain();
    mute.gain.value = 0;
    source.connect(processor as unknown as AudioNode);
    processor.connect(mute);
    mute.connect(context.destination);
  } catch {
    try {
      processor?.disconnect();
      mute?.disconnect();
    } catch {
      // The graph may already be closing.
    }
    return null;
  }
  if (!processor || !mute) return null;

  let capturing = false;
  let chunks: Float32Array[] = [];
  const push = (input: Float32Array) => {
    ring.push(input);
    if (!capturing || input.length === 0) return;
    const copy = new Float32Array(input.length);
    copy.set(input);
    chunks.push(copy);
  };
  processor.onaudioprocess = (event) => {
    push(mixdown(event.inputBuffer));
  };

  return {
    beginUtterance() {
      capturing = true;
      chunks = [ring.snapshot()];
    },
    takeUtterance() {
      if (!capturing) return null;
      capturing = false;
      const samples = concatSamples(chunks);
      chunks = [];
      return { sampleRate, samples };
    },
    close() {
      capturing = false;
      chunks = [];
      processor.onaudioprocess = null;
      try {
        processor.disconnect();
        mute.disconnect();
      } catch {
        // The graph may already be closing.
      }
    },
  };
}

export function voiceSamplesToWavDataUrl(samples: Float32Array, sampleRate: number): Promise<string> {
  return blobToDataUrl(floatChannelsToWav([samples], sampleRate > 0 ? sampleRate : 48_000));
}

function mixdown(buffer: PcmInputBuffer): Float32Array {
  const channels = Math.max(1, buffer.numberOfChannels || 1);
  const first = buffer.getChannelData(0);
  if (channels === 1) return first;
  const mixed = new Float32Array(first.length);
  mixed.set(first);
  for (let channel = 1; channel < channels; channel += 1) {
    const data = buffer.getChannelData(channel);
    const length = Math.min(mixed.length, data.length);
    for (let index = 0; index < length; index += 1) mixed[index] += data[index] ?? 0;
  }
  for (let index = 0; index < mixed.length; index += 1) mixed[index] /= channels;
  return mixed;
}

function createSampleRing(capacity: number) {
  const data = new Float32Array(capacity);
  let write = 0;
  let filled = 0;
  return {
    push(input: Float32Array) {
      if (capacity === 0 || input.length === 0) return;
      let offset = 0;
      let length = input.length;
      if (length >= capacity) {
        offset = length - capacity;
        length = capacity;
        data.set(input.subarray(offset));
        write = 0;
        filled = capacity;
        return;
      }
      const end = write + length;
      if (end <= capacity) {
        data.set(input.subarray(offset, offset + length), write);
      } else {
        const first = capacity - write;
        data.set(input.subarray(offset, offset + first), write);
        data.set(input.subarray(offset + first, offset + length), 0);
      }
      write = (write + length) % capacity;
      filled = Math.min(capacity, filled + length);
    },
    snapshot(): Float32Array {
      const out = new Float32Array(filled);
      if (filled === 0) return out;
      const start = (write - filled + capacity) % capacity;
      if (start + filled <= capacity) {
        out.set(data.subarray(start, start + filled));
      } else {
        const first = capacity - start;
        out.set(data.subarray(start), 0);
        out.set(data.subarray(0, filled - first), first);
      }
      return out;
    },
  };
}

function concatSamples(chunks: Float32Array[]): Float32Array {
  let total = 0;
  for (const chunk of chunks) total += chunk.length;
  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

function audioBufferToWav(buffer: AudioBuffer): Blob {
  const channels: Float32Array[] = [];
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    channels.push(buffer.getChannelData(channel));
  }
  return floatChannelsToWav(channels, buffer.sampleRate);
}

function floatChannelsToWav(channels: Float32Array[], sampleRate: number): Blob {
  const numChannels = Math.max(1, channels.length);
  const bitsPerSample = 16;
  const length = channels[0]?.length ?? 0;
  const interleaved = new Int16Array(length * numChannels);
  for (let index = 0; index < length; index += 1) {
    for (let channel = 0; channel < numChannels; channel += 1) {
      const sample = Math.max(-1, Math.min(1, channels[channel]?.[index] ?? 0));
      interleaved[index * numChannels + channel] = sample < 0
        ? sample * 0x8000
        : sample * 0x7FFF;
    }
  }

  const dataSize = interleaved.byteLength;
  const totalSize = 44 + dataSize;
  const bytes = new ArrayBuffer(totalSize);
  const view = new DataView(bytes);
  writeString(view, 0, "RIFF");
  view.setUint32(4, totalSize - 8, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * (bitsPerSample / 8), true);
  view.setUint16(32, numChannels * (bitsPerSample / 8), true);
  view.setUint16(34, bitsPerSample, true);
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);
  new Int16Array(bytes, 44).set(interleaved);
  return new Blob([bytes], { type: "audio/wav" });
}

function writeString(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

export function transcriptionErrorKey(error: unknown): VoiceConversationErrorKey {
  const detail = error instanceof Error ? error.message : "";
  if (detail === "not_configured") return "notConfigured";
  if (detail === "duration") return "tooLong";
  return "failed";
}

export function recordingErrorKey(error: unknown): VoiceConversationErrorKey {
  const name = error instanceof Error ? error.name : "";
  if (name === "NotFoundError") return "noDevice";
  return "permission";
}

export async function openVoiceMicrophone(mediaDevices: MediaDevices): Promise<MediaStream> {
  try {
    return await mediaDevices.getUserMedia({ audio: VOICE_MIC_CONSTRAINTS });
  } catch (error) {
    const name = error instanceof Error ? error.name : "";
    if (name !== "OverconstrainedError") throw error;
    return mediaDevices.getUserMedia({ audio: true });
  }
}
