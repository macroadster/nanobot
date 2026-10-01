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

function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const bitsPerSample = 16;
  const channels: Float32Array[] = [];
  for (let channel = 0; channel < numChannels; channel += 1) {
    channels.push(buffer.getChannelData(channel));
  }
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
