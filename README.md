<p align="center">
  <a target="_blank" href="https://maxmelichov.github.io/MamboRambo-site/">
    <img width="240" alt="MamboTTS logo" src="./mambotts-desktop/src/assets/mambotts-logo.png" />
  </a>
</p>

<h1 align="center">MamboTTS</h1>

<p align="center">
  <strong>Native offline BlueTTS for desktop</strong>
</p>

<p align="center">
  <a target="_blank" href="https://maxmelichov.github.io/MamboRambo-site/">
    🔗 Download MamboTTS
  </a>
  &nbsp; | &nbsp; Give it a Star ⭐ | &nbsp;
  <a target="_blank" href="https://github.com/sponsors/maxmelichov">Support the project 🤝</a>
</p>

<hr />

<p align="center">
  <a target="_blank" href="https://maxmelichov.github.io/MamboRambo-site/">
    <img width="800" alt="MamboTTS desktop screenshot" src="./docs/images/mambotts-studio.png" />
  </a>
</p>

## Features

- Local text-to-speech with BlueTTS
- Fully offline generation after the model is downloaded
- Saved voices: Noa, Lily, Daniel, and Adam
- Supported languages: Hebrew, English, Spanish, German, and Italian
- Audio preview after creation
- 💻 Desktop apps for `macOS` on Apple Silicon, `Linux` on x86_64, and `Windows` on x86_64
- Local HTTP API with Swagger docs for tools and automation
- Agent-ready `/skill` instructions for AI workflows

## Install

One command per platform. Each script is plain text at the URL it is fetched
from, so you can read it first if you would rather see what it does before it
does it.

**macOS on Apple Silicon**

```sh
curl -fsSL https://github.com/maxmelichov/MamboTTS/releases/latest/download/install.sh | sh
```

This downloads the DMG, copies `MamboTTS.app` into `/Applications`, and then
removes the quarantine attribute with `xattr -dr com.apple.quarantine`. That
last step is the reason the script exists. The macOS build is ad-hoc signed
rather than Developer ID signed, because Developer ID signing needs a paid
Apple Developer account, so Gatekeeper refuses the first launch and tells you
nothing useful about why. Clearing the quarantine flag is the same approval as
right-clicking the app and choosing Open, granted once, out in the open, to an
app you asked for by name.

**Linux on x86_64**

```sh
curl -fsSL https://github.com/maxmelichov/MamboTTS/releases/latest/download/install.sh | sh
```

This puts the AppImage in `~/.local/bin` and writes a launcher entry into
`~/.local/share/applications`, so MamboTTS appears in your applications menu.
Nothing here asks for sudo. If your distribution has dropped `libfuse2` the
AppImage will complain, and running it with `--appimage-extract-and-run` gets
around that without installing anything.

Debian, Ubuntu, Fedora, and RHEL users who would rather use their package
manager can take the `.deb` or `.rpm` from the
[release](https://github.com/maxmelichov/MamboTTS/releases/latest) instead.

**Windows on x86_64**

```powershell
irm https://github.com/maxmelichov/MamboTTS/releases/latest/download/install.ps1 | iex
```

This downloads the desktop installer and runs it. The NSIS window is left
visible on purpose: the command above is pasted from the internet, and the
installer is the one place you get to see where the app is going and say no.
Pass `-Silent` if you are installing unattended.

The app installs for the current user, so nothing asks for an administrator.
On first launch the onboarding screen downloads the BlueTTS model and the
Renikud phonemizer, around 1.5 GB, into the app data folder. That is the only
part that needs the network; generation after it is offline.

If you want the HTTP API without the window, `-Server` installs the headless
server into `%LOCALAPPDATA%\MamboTTS` with a `MamboTTS-Server.cmd` launcher
that starts it and opens the Swagger page. From there `/docs` is a working
console over `/v1/audio/speech`, `/v1/voices`, `/v1/phonemize`, and `/skill`,
which is everything the desktop app uses to make audio.

## Models and phonemizers

MamboTTS builds on these open-source projects:

- [BlueTTS](https://github.com/maxmelichov/BlueTTS) is the local ONNX text-to-speech runtime, shipped by default
- [RenikudPlus](https://github.com/maxmelichov/RenikudPlus) handles Hebrew grapheme-to-IPA conversion with speaker conditioning
- [Phonikud](https://github.com/phonikud/phonikud) supplies Hebrew vocalization and diacritics-aware IPA tools

## Python

`crates/mambotts-py` exposes the same engine to Python and ships a FastAPI
server over it, so the HTTP API is available on the platforms the desktop app
does not ship an installer for, such as Intel macOS and Arm Linux. See its
[README](crates/mambotts-py/README.md).

## Build

See [BUILDING.md](docs/BUILDING.md).

## Adding models

Want to ship another open-source TTS model or voice bundle with MamboTTS? See [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) for licensing, registry wiring, server/desktop steps, and the PR checklist.

---

This code is taken from [Chirp](https://github.com/thewh1teagle/chirp).
