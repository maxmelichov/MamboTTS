<#
.SYNOPSIS
    Installs the MamboTTS local server and its HTTP API on Windows.

.DESCRIPTION
    There is no MamboTTS desktop app for Windows. What Windows gets is the
    piece underneath it: the local server, which holds the BlueTTS runtime and
    exposes it over HTTP with Swagger docs at /docs. Everything the desktop app
    does with speech goes through that same API, so a Windows machine can
    generate audio, list voices, and phonemize text without the window.

    The script downloads the portable server zip from the GitHub release,
    unpacks it into %LOCALAPPDATA%\MamboTTS, downloads the BlueTTS model files
    and the Renikud phonemizer next to it, and writes a launcher that starts the
    server and opens the Swagger page in your browser.

    Run it with:
        irm https://github.com/maxmelichov/MamboTTS/releases/latest/download/install.ps1 | iex

    Running it a second time replaces the server in place and keeps any model
    file that already downloaded correctly.

.PARAMETER Version
    Install a specific release, for example 1.1.2. Defaults to the newest
    desktop release published on GitHub.

.PARAMETER Port
    The port the launcher starts the server on. Defaults to 8899.

.PARAMETER InstallDir
    Where everything lands. Defaults to %LOCALAPPDATA%\MamboTTS.
#>
[CmdletBinding()]
param(
    [string] $Version = '',
    [int]    $Port = 8899,
    [string] $InstallDir = (Join-Path $env:LOCALAPPDATA 'MamboTTS')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Repo = 'maxmelichov/MamboTTS'
$TagPrefix = 'mambotts-desktop-'
# Used when the GitHub API cannot be reached, which is common enough from a
# shared address that a hard failure would be the wrong default. Keep this in
# step with the newest published desktop release.
$FallbackVersion = '1.1.2'
$ModelSubDir = 'models\bluetts-2.5'

function Write-Step { param([string] $Message) Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Detail { param([string] $Message) Write-Host "    $Message" }
function Write-Warn { param([string] $Message) Write-Warning $Message }

# The registry in crates/mambotts-registry/src/lib.rs is the single source of
# truth for these URLs, and the running server publishes it at
# /v1/models/sources. The script asks the server it just unpacked rather than
# keeping a second copy of the list. This array is only the parachute for when
# that query fails, so a network hiccup does not leave a half installed tree.
$FallbackModelFiles = @(
    @{ name = 'duration_predictor.onnx';       url = 'https://huggingface.co/notmax123/BlueTTS2.5-onnx/resolve/main/duration_predictor.onnx' },
    @{ name = 'duration_predictor_style.onnx'; url = 'https://huggingface.co/notmax123/BlueTTS2.5-onnx/resolve/main/duration_predictor_style.onnx' },
    @{ name = 'text_encoder.onnx';             url = 'https://huggingface.co/notmax123/BlueTTS2.5-onnx/resolve/main/text_encoder.onnx' },
    @{ name = 'vector_estimator.onnx';         url = 'https://huggingface.co/notmax123/BlueTTS2.5-onnx/resolve/main/vector_estimator.onnx' },
    @{ name = 'vocoder.onnx';                  url = 'https://huggingface.co/notmax123/BlueTTS2.5-onnx/resolve/main/vocoder.onnx' },
    @{ name = 'stats.npz';                     url = 'https://huggingface.co/notmax123/BlueTTS2.5-onnx/resolve/main/stats.npz' },
    @{ name = 'uncond.npz';                    url = 'https://huggingface.co/notmax123/BlueTTS2.5-onnx/resolve/main/uncond.npz' },
    @{ name = 'vocab.json';                    url = 'https://huggingface.co/notmax123/BlueTTS2.5-onnx/resolve/main/vocab.json' },
    @{ name = 'tts.json';                      url = 'https://huggingface.co/notmax123/BlueTTS2.5-onnx/resolve/main/tts.json' },
    @{ name = 'voices/libri_female_1088.json'; url = 'https://huggingface.co/notmax123/BlueTTS2.5-onnx/resolve/main/voices/libri_female_1088.json' },
    @{ name = 'voices/libri_female_6147.json'; url = 'https://huggingface.co/notmax123/BlueTTS2.5-onnx/resolve/main/voices/libri_female_6147.json' },
    @{ name = 'voices/libri_male_6209.json';   url = 'https://huggingface.co/notmax123/BlueTTS2.5-onnx/resolve/main/voices/libri_male_6209.json' },
    @{ name = 'voices/libri_male_8088.json';   url = 'https://huggingface.co/notmax123/BlueTTS2.5-onnx/resolve/main/voices/libri_male_8088.json' },
    @{ name = 'renikud-plus.onnx';             url = 'https://huggingface.co/notmax123/RenikudPlus/resolve/main/model.onnx' }
)

function Assert-Supported {
    if (-not $env:LOCALAPPDATA) {
        throw 'LOCALAPPDATA is not set, so this does not look like a Windows session. Use install.sh on macOS and Linux.'
    }
    $arch = $env:PROCESSOR_ARCHITECTURE
    if ($arch -and $arch -ne 'AMD64' -and $arch -ne 'x86') {
        Write-Warn "This machine reports $arch. MamboTTS ships an x86_64 server, which runs on Arm Windows only through emulation and will be slow."
    }
}

function Resolve-Version {
    if ($Version) {
        Write-Detail "Using the requested version $Version."
        return $Version
    }
    Write-Step 'Looking up the newest MamboTTS release'
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases?per_page=30" `
            -Headers @{ 'Accept' = 'application/vnd.github+json'; 'User-Agent' = 'mambotts-installer' } `
            -TimeoutSec 20
        $tag = ($releases | Where-Object { $_.tag_name -like "$TagPrefix*" } | Select-Object -First 1).tag_name
        if ($tag) {
            $resolved = $tag -replace "^$TagPrefix", '' -replace '^v', ''
            Write-Detail "Newest release is $tag."
            return $resolved
        }
    } catch {
        Write-Warn "Could not read the GitHub releases API, which usually means a rate limit. Falling back to the pinned version $FallbackVersion."
    }
    return $FallbackVersion
}

# Downloads a file and refuses anything that is not a real body of a plausible
# size. Without this an HTML error page installs itself as a zero byte model.
function Get-Verified {
    param(
        [string] $Uri,
        [string] $OutFile,
        [long]   $MinimumBytes = 1
    )
    $dir = Split-Path -Parent $OutFile
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $temp = "$OutFile.partial"
    if (Test-Path $temp) { Remove-Item $temp -Force }

    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $Uri -OutFile $temp -UseBasicParsing -TimeoutSec 600
    } catch {
        if (Test-Path $temp) { Remove-Item $temp -Force }
        throw "Download failed for $Uri : $($_.Exception.Message)"
    }

    if (-not (Test-Path $temp)) { throw "Download produced no file for $Uri" }
    $size = (Get-Item $temp).Length
    if ($size -lt $MinimumBytes) {
        Remove-Item $temp -Force
        throw "The download from $Uri is only $size bytes, which is too small to be the real file."
    }
    Move-Item -Path $temp -Destination $OutFile -Force
    return $size
}

function Install-Server {
    param([string] $ReleaseVersion)

    $asset = "MamboTTS-Server_${ReleaseVersion}_x86_64-windows.zip"
    $url = "https://github.com/$Repo/releases/download/$TagPrefix" + "v$ReleaseVersion/$asset"
    $temp = Join-Path ([IO.Path]::GetTempPath()) $asset

    Write-Step "Downloading $asset"
    Write-Detail "from $url"
    $size = Get-Verified -Uri $url -OutFile $temp -MinimumBytes 1000000
    Write-Detail "$size bytes"

    Write-Step "Unpacking into $InstallDir"
    if (-not (Test-Path $InstallDir)) { New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null }
    # Only the program files are replaced. The models directory sits alongside
    # them and must survive a reinstall, because it is the slow half.
    foreach ($name in @('mambotts-server.exe', 'onnxruntime.dll', 'onnxruntime_providers_shared.dll')) {
        $existing = Join-Path $InstallDir $name
        if (Test-Path $existing) { Remove-Item $existing -Force -ErrorAction SilentlyContinue }
    }
    Expand-Archive -Path $temp -DestinationPath $InstallDir -Force
    Remove-Item $temp -Force -ErrorAction SilentlyContinue

    $exe = Join-Path $InstallDir 'mambotts-server.exe'
    if (-not (Test-Path $exe)) {
        throw "The archive did not contain mambotts-server.exe. Look inside $InstallDir and report this."
    }
    Write-Detail "mambotts-server.exe and the two ONNX Runtime DLLs are in place."
    return $exe
}

# Starts the freshly unpacked server with no model loaded, asks it for the
# download list it was compiled with, and stops it again.
function Get-ModelFileList {
    param([string] $Exe)

    Write-Step 'Asking the server which model files it needs'
    $probePort = 8912
    $process = $null
    try {
        $process = Start-Process -FilePath $Exe `
            -ArgumentList @('serve', '--host', '127.0.0.1', '--port', "$probePort", '--exit-with-parent', 'false') `
            -PassThru -WindowStyle Hidden
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Milliseconds 500
            try {
                $sources = Invoke-RestMethod -Uri "http://127.0.0.1:$probePort/v1/models/sources" -TimeoutSec 5
                $blue = $sources.runtimes | Where-Object { $_.id -eq 'blue' } | Select-Object -First 1
                if ($blue -and $blue.files) {
                    Write-Detail "$($blue.files.Count) files listed by the server itself."
                    return $blue.files
                }
            } catch {
                # Not up yet. Keep waiting.
            }
        }
        Write-Warn 'The server did not answer in time, so the built in file list is used instead.'
    } catch {
        Write-Warn "Could not query the server ($($_.Exception.Message)), so the built in file list is used instead."
    } finally {
        if ($process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
    return $FallbackModelFiles
}

function Install-Models {
    param($Files)

    $modelDir = Join-Path $InstallDir $ModelSubDir
    Write-Step "Downloading the BlueTTS model into $modelDir"
    Write-Detail 'This is around 570 MB and only happens once.'
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null

    $index = 0
    $total = @($Files).Count
    foreach ($file in $Files) {
        $index++
        $relative = ($file.name -replace '/', '\')
        $dest = Join-Path $modelDir $relative
        if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 0)) {
            Write-Detail "[$index/$total] $($file.name) is already here, keeping it."
            continue
        }
        Write-Detail "[$index/$total] $($file.name)"
        Get-Verified -Uri $file.url -OutFile $dest -MinimumBytes 1 | Out-Null
    }

    $renikud = Join-Path $modelDir 'renikud-plus.onnx'
    if (-not (Test-Path $renikud)) {
        throw "renikud-plus.onnx is missing from $modelDir, so Hebrew phonemization would fail."
    }
    return $modelDir
}

function Write-Launcher {
    param([string] $ModelDir)

    $launcher = Join-Path $InstallDir 'MamboTTS-Server.cmd'
    Write-Step "Writing the launcher at $launcher"

    # exit_with_parent defaults to true on the server, which would kill it the
    # moment this .cmd window closes. Passing false explicitly is what lets the
    # server outlive the shell that started it.
    $lines = @(
        '@echo off',
        'setlocal',
        'set "ROOT=%~dp0"',
        "set `"PORT=$Port`"",
        'echo Starting the MamboTTS server on http://127.0.0.1:%PORT%',
        'echo Loading the BlueTTS model takes a few seconds on first start.',
        'start "MamboTTS server" /min "%ROOT%mambotts-server.exe" serve --host 127.0.0.1 --port %PORT% --model-dir "%ROOT%models\bluetts-2.5" --renikud "%ROOT%models\bluetts-2.5\renikud-plus.onnx" --exit-with-parent false',
        'set "READY="',
        'for /l %%i in (1,1,60) do (',
        '  if not defined READY (',
        '    curl.exe -s -o nul "http://127.0.0.1:%PORT%/v1/voices" && set "READY=1"',
        '    if not defined READY ping -n 2 127.0.0.1 >nul',
        '  )',
        ')',
        'start "" "http://127.0.0.1:%PORT%/docs"',
        'echo.',
        'echo Swagger docs:  http://127.0.0.1:%PORT%/docs',
        'echo Speech:        POST http://127.0.0.1:%PORT%/v1/audio/speech',
        'echo Voices:        GET  http://127.0.0.1:%PORT%/v1/voices',
        'echo Phonemize:     POST http://127.0.0.1:%PORT%/v1/phonemize',
        'echo Agent skill:   GET  http://127.0.0.1:%PORT%/skill',
        'echo.',
        'echo The server keeps running in its own window. Close that window to stop it.',
        'endlocal'
    )
    Set-Content -Path $launcher -Value $lines -Encoding ASCII
    return $launcher
}

function Add-StartMenuShortcut {
    param([string] $Launcher)
    try {
        $startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
        if (-not (Test-Path $startMenu)) { return $null }
        $shortcut = Join-Path $startMenu 'MamboTTS Server.lnk'
        $shell = New-Object -ComObject WScript.Shell
        $link = $shell.CreateShortcut($shortcut)
        $link.TargetPath = $Launcher
        $link.WorkingDirectory = $InstallDir
        $link.Description = 'Start the MamboTTS local server and open its Swagger docs'
        $link.Save()
        return $shortcut
    } catch {
        Write-Warn "Could not write the Start Menu shortcut: $($_.Exception.Message)"
        return $null
    }
}

Write-Host 'MamboTTS server installer for Windows'
Write-Host ''
Write-Host 'This installs the local MamboTTS server and its HTTP API. There is no'
Write-Host 'MamboTTS desktop window on Windows. The API is the product here, and it'
Write-Host 'does everything the desktop app does with speech.'
Write-Host ''

Assert-Supported
$resolvedVersion = Resolve-Version
$exe = Install-Server -ReleaseVersion $resolvedVersion
$files = Get-ModelFileList -Exe $exe
$modelDir = Install-Models -Files $files
$launcher = Write-Launcher -ModelDir $modelDir
$shortcut = Add-StartMenuShortcut -Launcher $launcher

Write-Host ''
Write-Host "MamboTTS server $resolvedVersion is installed." -ForegroundColor Green
Write-Host ''
Write-Host 'What landed where:'
Write-Host "  Server:    $exe"
Write-Host "  Runtime:   $(Join-Path $InstallDir 'onnxruntime.dll')"
Write-Host "  Model:     $modelDir"
Write-Host "  Launcher:  $launcher"
if ($shortcut) { Write-Host "  Shortcut:  $shortcut" }
Write-Host ''
Write-Host 'Start it by running the launcher, or from the Start Menu:'
Write-Host "  `"$launcher`""
Write-Host ''
Write-Host "It opens http://127.0.0.1:$Port/docs once the model is loaded. The API is:"
Write-Host "  POST http://127.0.0.1:$Port/v1/audio/speech"
Write-Host "  GET  http://127.0.0.1:$Port/v1/voices"
Write-Host "  POST http://127.0.0.1:$Port/v1/phonemize"
Write-Host "  GET  http://127.0.0.1:$Port/skill"
