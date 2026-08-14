export interface ReleaseAsset {
  url: string
  name: string
  platform: "macos" | "windows" | "linux"
  arch: "darwin-aarch64" | "darwin-x86_64" | "windows-x86_64" | "linux-x86_64"
  kind: "dmg" | "exe" | "msi" | "zip" | "appimage" | "deb" | "rpm"
}

export interface LatestRelease {
  version: string
  url: string
  publishedAt: string
  assets: ReleaseAsset[]
}
