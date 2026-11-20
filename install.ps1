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
    data directory: the BlueV3 bundle (about 740 MB) and the Hebrew G2P
    RenikudPlus, 1.2 GB in full precision on most computers or 312 MB as int8 on
    machines with 4 GB of memory or less. It only happens once. Downloading them
    from this script as well would put a second copy somewhere the app never
    looks.

    With -Server the script does something different and older: it installs the
    headless server instead, which is the HTTP API on its own with Swagger docs
    at /docs, no window, and the model files downloaded next to it. That mode
    exists for machines that want the API and nothing else. The files and the
    RenikudPlus choice are exactly the desktop app's: the registry compiled into
    the server lists them (with their sizes and SHA-256), fp32 RenikudPlus
    unless the machine has 4 GB of memory or less, and MAMBOTTS_HEBREW_G2P=fp32
    or int8 overrides that.

    Run it with:
        irm https://github.com/maxmelichov/MamboTTS/releases/latest/download/install.ps1 | iex

    A piped script cannot take parameters, so to pass any of the switches below
    you have to hand the downloaded text to a script block instead:
        & ([scriptblock]::Create((irm https://github.com/maxmelichov/MamboTTS/releases/latest/download/install.ps1))) -Server

    Running it a second time is safe. In desktop mode the NSIS installer upgrades
    the existing install in place; in server mode the server is replaced and any
    model file that already downloaded correctly (right size and SHA-256) is
    kept.

.PARAMETER Version
    Install a specific release, for example 1.3.1. Defaults to the newest
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
$FallbackVersion = '1.4.2'
# The registry's directory for this bundle (RuntimeManifest.directory).
$ModelSubDir = 'models\bluev3-3.0.1'

function Write-Step { param([string] $Message) Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Detail { param([string] $Message) Write-Host "    $Message" }
function Write-Warn { param([string] $Message) Write-Warning $Message }

# The registry in crates/mambotts-registry/src/lib.rs is the single source of
# truth for these files, and the running server publishes it at
# /v1/models/sources. The script asks the server it just unpacked rather than
# keeping a second copy of the list. These arrays are only the parachute for
# when that query fails, so a network hiccup does not leave a half installed
# tree. They are the registry's BlueV3 v3.0.1 files at BlueV3-onnx d0a0270e (the
# revision its MANIFEST.json and BlueV3's download_models.py name) and
# RenikudPlus 679c56c, with the registry's sizes and SHA-256; keep them in step
# with it.
$BlueBase = 'https://huggingface.co/notmax123/BlueV3-onnx/resolve/d0a0270e60b96d5066cfe7ebed1f7c2872361c43'
$FallbackModelFiles = @(
    @{ name = 'duration_predictor_style.onnx'; size =   1524267; sha256 = '46197a2c10f06e151adc6dd109082889e6aaf903c46ce25d798eebaa7b07e313'; url = "$BlueBase/onnx/duration_predictor_style.onnx" },
    @{ name = 'text_encoder.onnx';             size =  27951369; sha256 = '9dd84651a48048dce980d9d54c11fd3c04170602fb4f9e66957c08eeee2e55ba'; url = "$BlueBase/onnx/text_encoder.onnx" },
    @{ name = 'vector_estimator.onnx';         size = 256556761; sha256 = '2f0e4af007f166afdc020331b863b1ea6b93c29258ae3b649f81234c1d7ad7c1'; url = "$BlueBase/onnx/vector_estimator.onnx" },
    @{ name = 'vocoder.onnx';                  size = 101428551; sha256 = '054a700e9ef17d43de203212dca335fa2cd000cd9ca7701df49fcc544b49ccbe'; url = "$BlueBase/onnx/vocoder.onnx" },
    @{ name = 'vocab.json';                    size =      3382; sha256 = '9cf96e595f62ea8a36d3be6724b06fb821614c2938d0724852d1a25feff5bad5'; url = "$BlueBase/onnx/vocab.json" },
    @{ name = 'tts.json';                      size =      8654; sha256 = '38c65e35451577954dd94eaf7fc619b8231cc37b3328ffb95556fa821983ef48'; url = "$BlueBase/onnx/tts.json" },
    @{ name = 'voices/noa.json';               size =    287992; sha256 = '34fea6e7746dc16399640604c8de87564bd759659ee91c536c1d30a3e9bab49d'; url = "$BlueBase/voices/noa.json" },
    @{ name = 'voices/lily.json';              size =    288400; sha256 = '8e3d320955ff8faf9ee5363111f9883ad476dd60d9919137d2851dd8c1ecefc1'; url = "$BlueBase/voices/lily.json" },
    @{ name = 'voices/maya.json';              size =    288319; sha256 = 'e0e1ab7de876c6097b2c3e378ed4add4aef2744d98a7755c38e684f2d3760c41'; url = "$BlueBase/voices/maya.json" },
    @{ name = 'voices/emma.json';              size =    288500; sha256 = 'e547c10547a306d3b81448c56e49a6ab03fababce23f15dc121eee19f4a1e9c4'; url = "$BlueBase/voices/emma.json" },
    @{ name = 'voices/sara.json';              size =    288412; sha256 = '373ff9b480cc74961c0df98f9c7cfad9feff000b5ab08237d0e22844546c97a4'; url = "$BlueBase/voices/sara.json" },
    @{ name = 'voices/shira.json';             size =    288188; sha256 = 'b42d26f1cb8d6ae58b0df074eba753442b56291aaf8d6e5ad2ac45ab66efcd53'; url = "$BlueBase/voices/shira.json" },
    @{ name = 'voices/tamar.json';             size =    288221; sha256 = '2ee2bcce78444da3f29417812695ae54d075b95fafaaa05eab40bb1e7ed19766'; url = "$BlueBase/voices/tamar.json" },
    @{ name = 'voices/yael.json';              size =    288284; sha256 = 'd8019210bc57798ae03448ecf71e58538d23a69c0c120f7989fa851fc99d1d5a'; url = "$BlueBase/voices/yael.json" },
    @{ name = 'voices/adam.json';              size =    288306; sha256 = '6b296fe8cb515aca19c5e6accdfc755fe9450867cdd2c4a223414b8e7a734d03'; url = "$BlueBase/voices/adam.json" },
    @{ name = 'voices/daniel.json';            size =    288934; sha256 = '4114b4183caa2f6336f575d380b877ea23015adc92f8d5184c9fef1b031c143a'; url = "$BlueBase/voices/daniel.json" },
    @{ name = 'voices/ariel.json';             size =    288782; sha256 = 'ac793ec51a7c4cf9fe4980fe254cbdb04f42ac2fbf37007b6e714a9348b9d6f2'; url = "$BlueBase/voices/ariel.json" },
    @{ name = 'voices/david.json';             size =    288731; sha256 = '07a2f81ac84142602c1c90dcea254af5c8497c4f116f71b2aaa403764c795945'; url = "$BlueBase/voices/david.json" },
    @{ name = 'voices/ethan.json';             size =    288722; sha256 = '441e9ed43f163784d435353fe8baca5fb90a978450542eb993d6e656ceef3ab0'; url = "$BlueBase/voices/ethan.json" },
    @{ name = 'voices/jonathan.json';          size =    288527; sha256 = '90f38ae4f18f585419c2705ed5ce0099a4e591258f75d5e872536318ff16bcd9'; url = "$BlueBase/voices/jonathan.json" },
    @{ name = 'voices/michael.json';           size =    288321; sha256 = '6ddf319b702b438cc500bee7b85be47cc42d1259282b91428679636e37b2cd91'; url = "$BlueBase/voices/michael.json" },
    @{ name = 'polyg2p/en.lex';                size =   2759393; sha256 = '713aaed9ce1891e111eab2d213df560f043947b9268c6411b26a8c756ea99d32'; url = "$BlueBase/frontend/polyg2p_data/en.lex" },
    @{ name = 'polyg2p/en-large.g2p';          size =   4880559; sha256 = 'a119ddaabc98ceb2696f2d9af52cfc0733388286d489b175c43129365ea15c4b'; url = "$BlueBase/frontend/polyg2p_data/en-large.g2p" },
    @{ name = 'polyg2p/en.g2p';                size =   1780155; sha256 = '2f638c5358a2a90075a575024d747931d1d92160e18ce0f9ed7176894a366808'; url = "$BlueBase/frontend/polyg2p_data/en.g2p" },
    @{ name = 'polyg2p/de.lex';                size =  17815772; sha256 = 'eeb7df564ef1bc894bf7cc39f0f054f5c516d38c2119922ac69ff8c31681594c'; url = "$BlueBase/frontend/polyg2p_data/de.lex" },
    @{ name = 'polyg2p/it.lex';                size =    516990; sha256 = '9f7e3363abe1beaeb6eab1d5393388d9f91b877491ade9841f47a27a871554f7'; url = "$BlueBase/frontend/polyg2p_data/it.lex" },
    @{ name = 'polyg2p/ru_accents.lex';        size =   5529194; sha256 = '505e3235f41a61fb54af70fb0a426f6ae68bb75e4d422841f4063a08865c8b07'; url = "$BlueBase/frontend/polyg2p_data/ru_accents.lex" },
    @{ name = 'polyg2p/ru_homographs.lex';     size =     82364; sha256 = '27a3bb226121130717c7e44fb21a25b826e1e5b77900f0a151bdf560bbaaa9e0'; url = "$BlueBase/frontend/polyg2p_data/ru_homographs.lex" },
    @{ name = 'polyg2p/ru_suffix.lex';         size =    399562; sha256 = '139cf5af5174d9408b27c428be5b341aaeeac144f3dd23bed0ab740a2b871ea6'; url = "$BlueBase/frontend/polyg2p_data/ru_suffix.lex" },
    @{ name = 'polyg2p/ru_yo.lex';             size =   2163336; sha256 = '16dbb14ea719a5f8562f64a41873e701c6163737cb03af55efa038c07b112165'; url = "$BlueBase/frontend/polyg2p_data/ru_yo.lex" },
    @{ name = 'polyg2p/ru_yo_homographs.lex';  size =     19043; sha256 = '057cddf02713732706127274c796a6f418ad7096679ac532f7c31cf7c9266064'; url = "$BlueBase/frontend/polyg2p_data/ru_yo_homographs.lex" },
    @{ name = 'polyg2p/LICENSE';               size =     11357; sha256 = 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'; url = "$BlueBase/frontend/polyg2p_data/LICENSE" },
    @{ name = 'polyg2p/NOTICE';                size =      2734; sha256 = '1a5544c762f18316b934e03db8ced24931f752f5332aa70c3bb15ca1ba92a114'; url = "$BlueBase/frontend/polyg2p_data/NOTICE" }
)
# Optional: the int8 vector estimator (registry optional_files), faster on
# ONNX Runtime's CPU path, which Windows runs. The server runs float32
# without it, so a failed download is only a warning.
$FallbackOptionalFiles = @(
    @{ name = 'vector_estimator.int8.onnx'; size = 66080072; sha256 = '0f9ac63e1e501a1b4996d508b2c3b1aeb5d4ff4b4e91aa5e46fdbe9dc967f61e'; url = 'https://huggingface.co/notmax123/BlueV3-onnx/resolve/f0db5595f83dc3df0e46eb6d74f083a2b5ccd74d/onnx/vector_estimator.int8.onnx' }
)
$RenikudBase = 'https://huggingface.co/notmax123/RenikudPlus/resolve/679c56ca449d41873fb8ff7711ddf7563d28198f'
$FallbackHebrewG2p = @(
    @{ variant = 'fp32'; name = 'renikud-plus-fp32.onnx'; size = 1234209530; sha256 = 'c3a2bdc35afbf53f28cdece4672dcca9b530dfa43a0d65b4c5bbc47150b32d0d'; url = "$RenikudBase/model.onnx" },
    @{ variant = 'int8'; name = 'renikud-plus.onnx';      size =  311605741; sha256 = '0337468fad56eadf5ecc7becfdf21648a4d668de41de5bc27e860141b5afcadd'; url = "$RenikudBase/model_int8.onnx" }
)
# HEBREW_G2P_INT8_MAX_MEMORY: at or below this much memory, int8.
$FallbackInt8MaxMemory = [long]4 * 1024 * 1024 * 1024

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
        # Skip a release whose Windows build has not been uploaded yet, so a
        # release published from another platform first never breaks this install.
        $tag = ($releases | Where-Object {
                $_.tag_name -like "$TagPrefix*" -and ($_.assets | Where-Object { $_.name -like '*_x64-setup.exe' })
            } | Select-Object -First 1).tag_name
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
        [long]   $ExpectedBytes = 0,
        [string] $Sha256 = '',
        # 0 is no limit: the fp32 RenikudPlus is 1.2 GB, and on PowerShell 7
        # the timeout covers the whole transfer.
        [int]    $TimeoutSec = 600,
        [switch] $MustBeExecutable
    )
    $dir = Split-Path -Parent $OutFile
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $temp = "$OutFile.partial"
    if (Test-Path $temp) { Remove-Item $temp -Force }

    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $Uri -OutFile $temp -UseBasicParsing -TimeoutSec $TimeoutSec
    } catch {
        if (Test-Path $temp) { Remove-Item $temp -Force }
        $status = 0
        try { $status = [int]$_.Exception.Response.StatusCode } catch { $status = 0 }
        if ($status -in 401, 403, 404, 410) {
            throw "$Uri is no longer available (HTTP $status): the pinned model revision is gone or refused. Install the newest MamboTTS release, whose file list points at the current files."
        }
        throw "Download failed for $Uri : $($_.Exception.Message)"
    }

    if (-not (Test-Path $temp)) { throw "Download produced no file for $Uri" }
    $size = (Get-Item $temp).Length
    if ($size -lt $MinimumBytes) {
        Remove-Item $temp -Force
        throw "The download from $Uri is only $size bytes, which is too small to be the real file."
    }
    if ($ExpectedBytes -gt 0 -and $size -ne $ExpectedBytes) {
        Remove-Item $temp -Force
        throw "The download from $Uri is $size bytes, but the registry says $ExpectedBytes."
    }
    if ($Sha256) {
        $actual = (Get-FileHash -Path $temp -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $Sha256.ToLowerInvariant()) {
            Remove-Item $temp -Force
            throw "The download from $Uri failed verification: SHA-256 $actual, expected $Sha256."
        }
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
    # Releases before BlueV3 shipped an espeak-ng-data directory beside the
    # executable. Nothing reads it any more, so an upgrade just removes it.
    $staleEspeakDir = Join-Path $InstallDir 'espeak-ng-data'
    if (Test-Path $staleEspeakDir) {
        Remove-Item $staleEspeakDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    Expand-Archive -Path $temp -DestinationPath $InstallDir -Force
    Remove-Item $temp -Force -ErrorAction SilentlyContinue

    $exe = Join-Path $InstallDir 'mambotts-server.exe'
    if (-not (Test-Path $exe)) {
        throw "The archive did not contain mambotts-server.exe. Look inside $InstallDir and report this."
    }

    Write-Detail 'mambotts-server.exe and the two ONNX Runtime DLLs are in place.'
    return $exe
}

# Starts the freshly unpacked server with no model loaded, asks it for the
# download list it was compiled with, and stops it again. Returns the bundle's
# files, the RenikudPlus variants and the memory threshold between them.
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
                # A server from before the registry carried sizes, hashes and
                # the RenikudPlus variants cannot describe this bundle.
                $hasG2p = $blue -and ($blue.PSObject.Properties.Name -contains 'hebrew_g2p') -and @($blue.hebrew_g2p).Count -gt 0
                if ($blue -and $blue.files -and $hasG2p -and $blue.directory -eq ($ModelSubDir -replace '^models\\', '')) {
                    Write-Detail "$(@($blue.files).Count) files and $(@($blue.hebrew_g2p).Count) RenikudPlus variants listed by the server itself."
                    # The files the server carries itself (git-stored in the
                    # model repository, gone from the Hub after a squash) come
                    # from it, so it stays up until Install-Models is done.
                    $files = @($blue.files | ForEach-Object {
                            if ($_.PSObject.Properties.Name -contains 'shipped' -and $_.shipped) {
                                $_.url = "http://127.0.0.1:$probePort/v1/models/files/$($_.name)"
                            }
                            $_
                        })
                    $keep = $process
                    $process = $null
                    # A server from before optional files lists none: use the
                    # built in list then.
                    $optional = $FallbackOptionalFiles
                    if ($blue.PSObject.Properties.Name -contains 'optional_files') { $optional = @($blue.optional_files) }
                    return @{
                        Files = $files
                        OptionalFiles = $optional
                        HebrewG2p = @($blue.hebrew_g2p)
                        Int8MaxMemory = [long]$blue.hebrew_g2p_policy.int8_max_memory
                        Server = $keep
                    }
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
    return @{ Files = $FallbackModelFiles; OptionalFiles = $FallbackOptionalFiles; HebrewG2p = $FallbackHebrewG2p; Int8MaxMemory = $FallbackInt8MaxMemory }
}

# The desktop app's choice (registry hebrew_g2p_variant, bluetts.com's
# hebrewG2PVariant): fp32 unless the machine has the threshold or less of
# memory; MAMBOTTS_HEBREW_G2P=fp32|int8 wins, and an unknown size is fp32.
function Select-HebrewG2pVariant {
    param([long] $Int8MaxMemory)

    switch -Regex (("$env:MAMBOTTS_HEBREW_G2P").Trim().ToLowerInvariant()) {
        '^(fp32|float|float32)$' { return 'fp32' }
        '^(int8|quantized)$' { return 'int8' }
    }
    $memory = [long]0
    try {
        $memory = [long](Get-CimInstance -ClassName Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory
    } catch {
        $memory = 0
    }
    if ($memory -gt 0 -and $memory -le $Int8MaxMemory) { return 'int8' }
    return 'fp32'
}

function Test-ModelFile {
    param([string] $Path, $File)
    if (-not (Test-Path $Path)) { return $false }
    if ((Get-Item $Path).Length -ne [long]$File.size) { return $false }
    $actual = (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    return ($actual -eq ([string]$File.sha256).ToLowerInvariant())
}

function Install-Models {
    param($List)

    $modelDir = Join-Path $InstallDir $ModelSubDir
    $variant = Select-HebrewG2pVariant -Int8MaxMemory $List.Int8MaxMemory
    $g2p = $List.HebrewG2p | Where-Object { $_.variant -eq $variant } | Select-Object -First 1
    if (-not $g2p) { throw "No RenikudPlus file for the $variant variant in the file list." }
    $files = @($List.Files) + @($g2p)

    Write-Step "Downloading the BlueTTS model into $modelDir"
    $bytes = [long]0
    foreach ($file in $files) { $bytes += [long]$file.size }
    Write-Detail "RenikudPlus $variant ($($g2p.name)). About $([math]::Round($bytes / 1MB)) MB in all; it only happens once."
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null

    $index = 0
    $total = $files.Count
    foreach ($file in $files) {
        $index++
        $relative = ($file.name -replace '/', '\')
        $dest = Join-Path $modelDir $relative
        if (Test-ModelFile -Path $dest -File $file) {
            Write-Detail "[$index/$total] $($file.name) is already here and verified, keeping it."
            continue
        }
        Write-Detail "[$index/$total] $($file.name)"
        Get-Verified -Uri $file.url -OutFile $dest -ExpectedBytes ([long]$file.size) -Sha256 $file.sha256 -TimeoutSec 0 | Out-Null
    }

    foreach ($file in @($List.OptionalFiles)) {
        if (-not $file) { continue }
        $dest = Join-Path $modelDir ($file.name -replace '/', '\')
        if (Test-ModelFile -Path $dest -File $file) {
            Write-Detail "$($file.name) (optional) is already here and verified, keeping it."
            continue
        }
        Write-Detail "$($file.name) (optional, faster on the CPU)"
        try {
            Get-Verified -Uri $file.url -OutFile $dest -ExpectedBytes ([long]$file.size) -Sha256 $file.sha256 -TimeoutSec 0 | Out-Null
        } catch {
            Write-Warn "Could not download $($file.name) ($($_.Exception.Message)). The server runs without it, only slower."
        }
    }

    # One RenikudPlus on disk, as in the desktop app: the variant this machine
    # does not run goes once its own is in place.
    foreach ($other in $List.HebrewG2p) {
        if ($other.name -ne $g2p.name) {
            Remove-Item (Join-Path $modelDir $other.name) -Force -ErrorAction SilentlyContinue
        }
    }
    # The 2.5 bundle's directory: nothing reads it any more.
    $stale = Join-Path $InstallDir 'models\bluetts-2.5'
    if (Test-Path $stale) {
        Write-Detail 'Removing the BlueTTS 2.5 model directory, which this release no longer reads.'
        Remove-Item $stale -Recurse -Force -ErrorAction SilentlyContinue
    }
    return @{ ModelDir = $modelDir; Renikud = $g2p.name }
}

function Write-Launcher {
    param([string] $Renikud)

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
        "start `"MamboTTS server`" /min `"%ROOT%mambotts-server.exe`" serve --host 127.0.0.1 --port %PORT% --model-dir `"%ROOT%$ModelSubDir`" --renikud `"%ROOT%$ModelSubDir\$Renikud`" --exit-with-parent false",
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
        # A shortcut to a .cmd inherits the generic Windows script icon, which is
        # how the Start Menu entry ended up looking like nothing in particular.
        # The server archive ships mambotts.ico beside the binary; older archives
        # do not, so the icon stays optional.
        $icon = Join-Path $InstallDir 'mambotts.ico'
        if (Test-Path $icon) { $link.IconLocation = "$icon,0" }
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
    Write-Host 'downloads its voices and models itself the first time you open it (about'
    Write-Host '1.7 GB, or 1 GB on machines with 4 GB of memory or less), so this part is quick.'
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
    Write-Host 'model and the Hebrew phonemizer, about 1.7 GB, into the app data folder.'
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
    $list = Get-ModelFileList -Exe $exe
    try {
        $models = Install-Models -List $list
    } finally {
        if ($list.Server -and -not $list.Server.HasExited) {
            Stop-Process -Id $list.Server.Id -Force -ErrorAction SilentlyContinue
        }
    }
    $modelDir = $models.ModelDir
    $launcher = Write-Launcher -Renikud $models.Renikud
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
}

if ($Server) {
    if ($Silent) {
        Write-Warn '-Silent only means something for the desktop installer. The server install never shows a window anyway.'
    }
    Install-ServerMode
} else {
    Install-DesktopMode
}
