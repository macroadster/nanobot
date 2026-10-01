import type { UIMediaAttachment, UIMediaKind } from "@/lib/types";

const IMAGE_EXTENSIONS = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".webp",
  ".bmp",
  ".ico",
  ".svg",
  ".tif",
  ".tiff",
]);

const VIDEO_EXTENSIONS = new Set([
  ".mp4",
  ".webm",
  ".mov",
  ".m4v",
  ".avi",
  ".mkv",
  ".3gp",
]);

const AUDIO_EXTENSIONS = new Set([
  ".aac",
  ".flac",
  ".m4a",
  ".mp3",
  ".ogg",
  ".opus",
  ".wav",
  ".weba",
]);

function cleanPath(value: string): string {
  return value.split(/[?#]/, 1)[0]?.toLowerCase() ?? "";
}

function extensionOf(value?: string): string {
  if (!value) return "";
  const path = cleanPath(value);
  const dot = path.lastIndexOf(".");
  if (dot < 0) return "";
  return path.slice(dot);
}

function explicitMediaKind(media: { url?: string; name?: string }): UIMediaKind | null {
  const url = media.url ?? "";
  if (url.startsWith("data:image/")) return "image";
  if (url.startsWith("data:video/")) return "video";
  if (url.startsWith("data:audio/")) return "audio";

  const ext = extensionOf(media.name) || extensionOf(url);
  if (!ext) return null;
  if (IMAGE_EXTENSIONS.has(ext)) return "image";
  if (VIDEO_EXTENSIONS.has(ext)) return "video";
  if (AUDIO_EXTENSIONS.has(ext)) return "audio";
  return "file";
}

export function inferMediaKind(media: { url?: string; name?: string }): UIMediaKind {
  return explicitMediaKind(media) ?? "file";
}

export function toMediaAttachment(media: {
  url?: string;
  name?: string;
  kind?: UIMediaKind;
}): UIMediaAttachment {
  return {
    kind: explicitMediaKind(media) ?? media.kind ?? "file",
    url: media.url,
    name: media.name,
  };
}
