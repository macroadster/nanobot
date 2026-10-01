import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  DEFAULT_TRANSCRIPTION_FORM,
  DEFAULT_TRANSCRIPTION_SETTINGS,
  TranscriptionSettings,
} from "@/components/settings/capabilities/TranscriptionSettings";
import type { SettingsPayload } from "@/lib/types";

describe("TranscriptionSettings", () => {
  it("shows Grok Voice reply controls", () => {
    const settings = {
      transcription: {
        ...DEFAULT_TRANSCRIPTION_SETTINGS,
        provider: "grok",
        voice_configured: true,
        providers: [{ name: "grok", label: "Grok (xAI)", configured: true }],
      },
    } as SettingsPayload;

    render(
      <TranscriptionSettings
        settings={settings}
        form={{ ...DEFAULT_TRANSCRIPTION_FORM, provider: "grok", voiceId: "eve" }}
        dirty={false}
        saving={false}
        onChangeForm={() => undefined}
        onSave={() => undefined}
        onOpenProviders={() => undefined}
        showBrandLogos={false}
        requiresRestartPending={false}
      />,
    );

    expect(screen.getByRole("switch", { name: "Speak replies" })).toBeInTheDocument();
    expect(screen.getByText("Grok ready")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("eve")).toHaveValue("eve");
  });
});
