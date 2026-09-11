<#
.SYNOPSIS
    Installs the MamboTTS desktop app on Windows, or the headless server on request.

.DESCRIPTION
    Windows gets the real desktop app. The script downloads the signed-by-nobody
    but perfectly ordinary NSIS installer from the GitHub release, checks that
    what arrived is actually a Windows executable of a plausible size, and then
    runs it. The installer is the same one you would get by clicking the download
    link on the website, so it puts MamboTTS in the Start Menu and registers an
    entry under Installed apps that removes it again.

    Models are not downloaded here. The desktop app fetches them itself the first
    time you open it, through its onboarding screen, into its own application
    data directory. That is around 1.5 GB and it only happens once. Downloading
    them from this script as well would put a second copy of the same gigabyte
    and a half somewhere the app never looks.

    With -Server the script does something different and older: it installs the
    headless server instead, which is the HTTP API on its own with Swagger docs
    at /docs, no window, and the model files downloaded next to it. That mode
    exists for machines that want the API and nothing else.

    Run it with:
        irm https://github.com/maxmelichov/MamboTTS/releases/latest/download/install.ps1 | iex

    A piped script cannot take parameters, so to pass any of the switches below
    you have to hand the downloaded text to a script block instead:
        & ([scriptblock]::Create((irm https://github.com/maxmelichov/MamboTTS/releases/latest/download/install.ps1))) -Server

    Running it a second time is safe. In desktop mode the NSIS installer upgrades
    the existing install in place; in server mode the server is replaced and any
    model file that already downloaded correctly is kept.

.PARAMETER Version
    Install a specific release, for example 1.3.0. Defaults to the newest
    desktop release published on GitHub.

.PARAMETER Server
    Install the headless server instead of the desktop app: the HTTP API, the
    BlueTTS model files, and a launcher that starts the server and opens its
    Swagger page. No window, no Tauri app, no onboarding.

.PARAMETER Silent
    Desktop mode only. Runs the NSIS installer with /S, so it installs without
    showing its window and without asking you anything. Off by default, because
    a script you piped in from the internet should not install software behind
    your back. Turn it on for unattended machines and CI.

.PARAMETER Port
    Server mode only. The port the launcher starts the server on. Defaults
    to 8899.

.PARAMETER InstallDir
    Server mode only. Where the server and its models land. Defaults to
    %LOCALAPPDATA%\MamboTTS. The desktop app's location is chosen by its own
    installer, which you can change in the installer window when it is not
    running silently.
#>
[CmdletBinding()]
param(
    [string] $Version = '',
    [switch] $Server,
    [switch] $Silent,
    [int]    $Port = 8899,
    [string] $InstallDir = (Join-Path $env:LOCALAPPDATA 'MamboTTS')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# $PSBoundParameters is per function, so a function asking it what the caller
# passed would only ever learn about its own arguments. The script level copy
# has to be taken out here, once.
$ScriptArgs = $PSBoundParameters

$Repo = 'maxmelichov/MamboTTS'
$TagPrefix = 'mambotts-desktop-'
# Used when the GitHub API cannot be reached, which is common enough from a
# shared address that a hard failure would be the wrong default. Keep this in
# step with the newest published desktop release.
$FallbackVersion = '1.3.0'
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
        Write-Warn "This machine reports $arch. MamboTTS ships x86_64 builds, which run on Arm Windows only through emulation and will be slow."
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

# A proxy login page or a GitHub error page can be large enough to sail past a
# size check, and the next thing this script does with a downloaded installer is
# execute it. Reading the DOS and PE headers is the cheap way to know that the
# file really is a Windows executable before that happens.
function Test-PortableExecutable {
    param([string] $Path)

    $stream = $null
    try {
        $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        if ($stream.Length -lt 512) { return $false }
        $reader = New-Object IO.BinaryReader($stream)
        if ($reader.ReadUInt16() -ne 0x5A4D) { return $false }   # 'MZ'
        # e_lfanew, at offset 0x3C, points at the PE signature.
        $stream.Position = 0x3C
        $peOffset = $reader.ReadInt32()
        if ($peOffset -le 0 -or ($peOffset + 4) -gt $stream.Length) { return $false }
        $stream.Position = $peOffset
        return ($reader.ReadUInt32() -eq 0x00004550)             # 'PE\0\0'
    } catch {
        return $false
    } finally {
        if ($stream) { $stream.Dispose() }
    }
}

# Downloads a file and refuses anything that is not a real body of a plausible
# size. Without this an HTML error page installs itself as a zero byte model.
# Anything that fails a check is deleted while it is still called .partial, so
# a rejected download never exists under the name the caller would go on to use.
function Get-Verified {
    param(
        [string] $Uri,
        [string] $OutFile,
        [long]   $MinimumBytes = 1,
        [switch] $MustBeExecutable
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
    if ($MustBeExecutable -and -not (Test-PortableExecutable -Path $temp)) {
        Remove-Item $temp -Force
        throw "The download from $Uri is $size bytes but has no PE header, so it is not a Windows executable. Something between here and GitHub answered with a page instead of the file, and it will not be run."
    }
    Move-Item -Path $temp -Destination $OutFile -Force
    return $size
}

# Where the NSIS installer puts MamboTTS.exe depends on whether it installed for
# you or for the machine, and the script does not get to choose in interactive
# mode, so both are worth a look before reporting a path.
function Find-DesktopApp {
    $candidates = @()
    if ($env:LOCALAPPDATA) { $candidates += (Join-Path $env:LOCALAPPDATA 'MamboTTS\MamboTTS.exe') }
    if ($env:ProgramFiles) { $candidates += (Join-Path $env:ProgramFiles 'MamboTTS\MamboTTS.exe') }
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

# Returns $true when the installer ran to completion, $false when the person in
# front of it decided otherwise. Anything else throws.
function Install-Desktop {
    param([string] $ReleaseVersion)

    $asset = "MamboTTS_${ReleaseVersion}_x64-setup.exe"
    $url = "https://github.com/$Repo/releases/download/$TagPrefix" + "v$ReleaseVersion/$asset"
    $temp = Join-Path ([IO.Path]::GetTempPath()) $asset

    Write-Step "Downloading $asset"
    Write-Detail "from $url"
    try {
        # The bundle carries the app, the server sidecar and the ONNX Runtime,
        # so it is tens of megabytes. Five is already far below anything real.
        $size = Get-Verified -Uri $url -OutFile $temp -MinimumBytes 5000000 -MustBeExecutable
    } catch {
        # Windows desktop bundles only exist from the release that added the
        # windows-latest build onwards. Older tags carry the server zip and
        # nothing else, and a bare 404 tells nobody that.
        throw "$($_.Exception.Message)`n`nIf that was a 404, release $ReleaseVersion probably predates the Windows desktop build. Install the newest release by leaving -Version off, or pass -Server to get the HTTP API from this one."
    }
    Write-Detail "$([math]::Round($size / 1MB)) MB, and the PE header checks out."

    try {
        $process = $null
        if ($Silent) {
            Write-Step 'Running the installer with no window, because you asked for -Silent'
            $process = Start-Process -FilePath $temp -ArgumentList @('/S') -PassThru -Wait
        } else {
            Write-Step 'Starting the MamboTTS installer'
            Write-Detail 'Its window is opening now. Nothing else happens here until you finish or cancel it.'
            $process = Start-Process -FilePath $temp -PassThru -Wait
        }

        $code = 0
        if ($process) { $code = $process.ExitCode }
        if ($null -eq $code) { $code = 0 }

        # NSIS uses 1 for a cancelled install and 2 for an install that broke.
        # Cancelling is a decision, not a failure, so it does not throw.
        if ($code -eq 1) {
            Write-Warn 'The installer was cancelled, so nothing was installed.'
            return $false
        }
        if ($code -ne 0) {
            throw "The installer exited with code $code. Run $asset by hand to see what it says."
        }
        return $true
    } finally {
        Remove-Item $temp -Force -ErrorAction SilentlyContinue
    }
}

# Looks inside the zip without unpacking it, so the script can tell a build that
# ships espeak-ng-data from one that does not before it deletes anything.
function Test-ArchiveHasEspeakData {
    param([string] $Path)

    $archive = $null
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
        $archive = [IO.Compression.ZipFile]::OpenRead($Path)
        foreach ($entry in $archive.Entries) {
            # Zip entries are meant to use forward slashes and plenty of writers
            # ignore that, Compress-Archive included, so both characters have to
            # count as the separator here.
            $normalized = $entry.FullName.Replace('\', '/')
            if ($normalized -like 'espeak-ng-data/*' -or $normalized -like '*/espeak-ng-data/*') {
                return $true
            }
        }
        return $false
    } catch {
        # A zip this cannot read is a zip Expand-Archive is about to complain
        # about anyway, and answering no here only means a stale directory
        # survives one more install rather than something being destroyed.
        return $false
    } finally {
        if ($archive) { $archive.Dispose() }
    }
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

    # The desktop app ships its own copy of mambotts-server.exe as a sidecar and
    # its installer defaults to this same folder, so the two installs can end up
    # writing the same file name at different versions. That only ever surfaces
    # later as a mystery, so say it out loud now.
    if (Test-Path (Join-Path $InstallDir 'MamboTTS.exe')) {
        Write-Warn "The MamboTTS desktop app is installed in $InstallDir and carries its own copy of mambotts-server.exe. This install will replace that copy. Pass -InstallDir to keep the two apart."
    }

    # Only the program files are replaced. The models directory sits alongside
    # them and must survive a reinstall, because it is the slow half.
    foreach ($name in @('mambotts-server.exe', 'onnxruntime.dll', 'onnxruntime_providers_shared.dll')) {
        $existing = Join-Path $InstallDir $name
        if (Test-Path $existing) { Remove-Item $existing -Force -ErrorAction SilentlyContinue }
    }
    # Expand-Archive overwrites the files that the zip contains and leaves
    # everything else alone, so an espeak-ng-data directory from an older build
    # would half survive a reinstall: new files written into it, files that were
    # dropped still sitting there, and no way to tell which is which. Clearing
    # it first is what makes the replacement complete.
    #
    # It is only cleared when the archive actually carries a replacement. The
    # desktop app puts its own espeak-ng-data in this same folder, and deleting
    # that in exchange for nothing would take four languages away from an app
    # this script was not even asked to touch.
    $espeakDir = Join-Path $InstallDir 'espeak-ng-data'
    $archiveHasEspeak = Test-ArchiveHasEspeakData -Path $temp
    if ($archiveHasEspeak -and (Test-Path $espeakDir)) {
        Write-Detail 'Clearing the old espeak-ng-data directory so the new one replaces it whole.'
        Remove-Item $espeakDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    Expand-Archive -Path $temp -DestinationPath $InstallDir -Force
    Remove-Item $temp -Force -ErrorAction SilentlyContinue

    $exe = Join-Path $InstallDir 'mambotts-server.exe'
    if (-not (Test-Path $exe)) {
        throw "The archive did not contain mambotts-server.exe. Look inside $InstallDir and report this."
    }

    # espeak-ng turns English, Spanish, German and Italian into phonemes and
    # reads its rules out of espeak-ng-data next to the executable. Without that
    # directory those four languages come back as HTTP 500 while Hebrew keeps
    # working, which reads like a language bug rather than a missing file. Catch
    # it here, before the model download spends twenty minutes.
    if (-not (Test-Path $espeakDir)) {
        throw "espeak-ng-data is missing from $InstallDir. Without it only Hebrew works and English, Spanish, German and Italian all fail with HTTP 500. This build of $asset predates the fix that bundles it: install the newest release by running the installer without -Version, or copy an espeak-ng-data directory next to mambotts-server.exe yourself."
    }
    $espeakFiles = @(Get-ChildItem -Path $espeakDir -Recurse -File -ErrorAction SilentlyContinue)
    if ($espeakFiles.Count -lt 1) {
        throw "espeak-ng-data in $InstallDir is empty, so English, Spanish, German and Italian would fail with HTTP 500. Delete $espeakDir and run this installer again."
    }
    if (-not $archiveHasEspeak) {
        Write-Warn "$asset does not carry espeak-ng-data, so the directory already in $InstallDir is what English, Spanish, German and Italian will use. That is usually fine, since the phoneme data rarely changes, but a newer release ships its own copy."
    }

    Write-Detail "mambotts-server.exe, the two ONNX Runtime DLLs and $($espeakFiles.Count) espeak-ng-data files are in place."
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
    # renikud-plus.onnx is 1.23 GB of that on its own; the BlueTTS ONNX files
    # account for most of the rest.
    Write-Detail 'This is around 1.5 GB and only happens once.'
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
        'REM curl.exe ships with Windows 10 1803 and later. Where it is missing,',
        'REM fall back to a flat wait rather than looping on a command that is',
        'REM never going to succeed.',
        'where curl.exe >nul 2>&1 || goto :flatwait',
        'for /l %%i in (1,1,60) do (',
        '  if not defined READY (',
        '    curl.exe -s -o nul "http://127.0.0.1:%PORT%/v1/voices" && set "READY=1"',
        '    if not defined READY ping -n 2 127.0.0.1 >nul',
        '  )',
        ')',
        'goto :ready',
        ':flatwait',
        'ping -n 16 127.0.0.1 >nul',
        ':ready',
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

function Install-DesktopMode {
    # -Port and -InstallDir belong to the server. The desktop installer decides
    # its own location, so silently accepting them would be a lie.
    foreach ($name in @('Port', 'InstallDir')) {
        if ($ScriptArgs.ContainsKey($name)) {
            Write-Warn "-$name only means something with -Server, so it is ignored here."
        }
    }

    Write-Host 'MamboTTS installer for Windows'
    Write-Host ''
    Write-Host 'This downloads the MamboTTS desktop app and runs its installer. The app'
    Write-Host 'downloads its voices and models itself the first time you open it, which'
    Write-Host 'is around 1.5 GB, so this part is quick.'
    Write-Host ''
    Write-Host 'If you only want the HTTP API and no window, run this again with -Server.'
    Write-Host ''

    Assert-Supported
    $resolvedVersion = Resolve-Version
    $installed = Install-Desktop -ReleaseVersion $resolvedVersion
    if (-not $installed) {
        Write-Host ''
        Write-Host 'Nothing was installed. Run the same command again whenever you want to.'
        return
    }

    $app = Find-DesktopApp

    Write-Host ''
    Write-Host "MamboTTS $resolvedVersion is installed." -ForegroundColor Green
    Write-Host ''
    if ($app) {
        Write-Host 'What landed where:'
        Write-Host "  App:       $app"
        Write-Host '  Start Menu: MamboTTS'
    } else {
        Write-Host 'Open it from the Start Menu, under MamboTTS.'
    }
    Write-Host ''
    Write-Host 'The first launch opens an onboarding screen that downloads the BlueTTS'
    Write-Host 'model and the Hebrew phonemizer, around 1.5 GB, into the app data folder.'
    Write-Host 'Leave it running until it finishes; after that it starts straight up.'
    Write-Host ''
    Write-Host 'To remove it later: Settings, Installed apps, MamboTTS.'
}

function Install-ServerMode {
    Write-Host 'MamboTTS server installer for Windows'
    Write-Host ''
    Write-Host 'This installs the local MamboTTS server and its HTTP API: no window, no'
    Write-Host 'onboarding, everything through the API. Drop -Server if you wanted the'
    Write-Host 'desktop app instead.'
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
    Write-Host "  espeak:    $(Join-Path $InstallDir 'espeak-ng-data')"
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
}

if ($Server) {
    if ($Silent) {
        Write-Warn '-Silent only means something for the desktop installer. The server install never shows a window anyway.'
    }
    Install-ServerMode
} else {
    Install-DesktopMode
}
