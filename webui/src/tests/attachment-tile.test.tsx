import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AttachmentTile } from "@/components/AttachmentTile";

describe("AttachmentTile", () => {
  it("renders a voice reply as an audio player", () => {
    render(
      <AttachmentTile
        attachment={{ kind: "audio", url: "/api/media/sig/voice", name: "reply.mp3" }}
      />,
    );

    const player = screen.getByLabelText("Voice reply: reply.mp3");
    expect(player.tagName).toBe("AUDIO");
    expect(player).toHaveAttribute("src", "/api/media/sig/voice");
    expect(player).toHaveAttribute("controls");
  });
});
