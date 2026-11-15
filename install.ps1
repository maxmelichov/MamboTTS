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
$FallbackVersion = '1.3.7'
# The registry's directory for this bundle (RuntimeManifest.directory).
$ModelSubDir = 'models\bluev3-3.0.0'

function Write-Step { param([string] $Message) Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Detail { param([string] $Message) Write-Host "    $Message" }
function Write-Warn { param([string] $Message) Write-Warning $Message }

# The registry in crates/mambotts-registry/src/lib.rs is the single source of
# truth for these files, and the running server publishes it at
# /v1/models/sources. The script asks the server it just unpacked rather than
# keeping a second copy of the list. These arrays are only the parachute for
# when that query fails, so a network hiccup does not leave a half installed
# tree. They are the registry's BlueV3 v3.0.0 files at BlueV3-onnx fb3dc38 (the
# commit the v3.0.0 tag names) and RenikudPlus 679c56c, with the registry's
# sizes and SHA-256; keep them in step with it.
$BlueBase = 'https://huggingface.co/notmax123/BlueV3-onnx/resolve/fb3dc3862557b8fad913a5c66ec9dfb5c9bbbf86'
$FallbackModelFiles = @(
    @{ name = 'duration_predictor_style.onnx'; size =   1524267; sha256 = '9aa8951637c7de4c2f6d43296af6394f4527a810fe8ba56df5aeedc2cec7247b'; url = "$BlueBase/onnx/duration_predictor_style.onnx" },
    @{ name = 'text_encoder.onnx';             size =  27951369; sha256 = '288d48053d82f12fb4b7319ea0a5c8c669c4e87ade983c695666fa9e346408c1'; url = "$BlueBase/onnx/text_encoder.onnx" },
    @{ name = 'vector_estimator.onnx';         size = 256556761; sha256 = '46abc33ac65e24cb5331b6d29ec4520b9a33a4a86a7723b5e44677af10d5b584'; url = "$BlueBase/onnx/vector_estimator.onnx" },
    @{ name = 'vocoder.onnx';                  size = 101428551; sha256 = '054a700e9ef17d43de203212dca335fa2cd000cd9ca7701df49fcc544b49ccbe'; url = "$BlueBase/onnx/vocoder.onnx" },
    @{ name = 'vocab.json';                    size =      3382; sha256 = '9cf96e595f62ea8a36d3be6724b06fb821614c2938d0724852d1a25feff5bad5'; url = "$BlueBase/onnx/vocab.json" },
    @{ name = 'tts.json';                      size =      8509; sha256 = '35af77b0634c2d4be4eb2faeeabdb364ad6bf80302202f6b5952e83984326eea'; url = "$BlueBase/onnx/tts.json" },
    @{ name = 'voices/noa.json';               size =    287857; sha256 = 'd392adf5542220ca57841308ffd7ac76f013025b624efd8f43a8f53a799e4638'; url = "$BlueBase/voices/noa.json" },
    @{ name = 'voices/lily.json';              size =    288281; sha256 = '7d1b9f54d4a6797b31e997f59a55399e3daf73782aa2a477721afd963ca59969'; url = "$BlueBase/voices/lily.json" },
    @{ name = 'voices/maya.json';              size =    288203; sha256 = '83a9f7303c954951677fc0db17e58bfeee8064553f67f3991b7c7d0ed4c8f8ce'; url = "$BlueBase/voices/maya.json" },
    @{ name = 'voices/emma.json';              size =    288397; sha256 = '46cedaaa14f68dc9017e8c0fac94b21ba615845c1f7981cb5bfe3740cf4a2365'; url = "$BlueBase/voices/emma.json" },
    @{ name = 'voices/sara.json';              size =    288277; sha256 = 'dc43188552dfede47f9450104cce931ae356e8896f1b9b093317d4dff4fc6863'; url = "$BlueBase/voices/sara.json" },
    @{ name = 'voices/shira.json';             size =    288019; sha256 = '19611940a2cdc1b90ce70b13b12682c3c2b1ef6be14a78907c4fb8228a913ca3'; url = "$BlueBase/voices/shira.json" },
    @{ name = 'voices/tamar.json';             size =    288114; sha256 = '09026d10ef83b9e7e490b68f17ef7fb54e80940bf6321990b53a5a7340c79987'; url = "$BlueBase/voices/tamar.json" },
    @{ name = 'voices/yael.json';              size =    288145; sha256 = '3c82bf087ce6831b75c2755741dc519ebe87a58cc1e02e36f4cfa7ca35650501'; url = "$BlueBase/voices/yael.json" },
    @{ name = 'voices/adam.json';              size =    288210; sha256 = '9daa9a234c5a4a36cab1d02c527c307a8a0058a3f806b316af01150a98604d22'; url = "$BlueBase/voices/adam.json" },
    @{ name = 'voices/daniel.json';            size =    288828; sha256 = 'c8a6dd2a63259f686230cf3fd55e28906a158c9363cd33cb374a02201b862f5f'; url = "$BlueBase/voices/daniel.json" },
    @{ name = 'voices/ariel.json';             size =    288672; sha256 = '67cc2e5c2f5c582d120aee4c5cd65b4ab21c4a4d89a0fe81ef1938a1d9f266cc'; url = "$BlueBase/voices/ariel.json" },
    @{ name = 'voices/david.json';             size =    288598; sha256 = '6e51be7341233ce360659031f0ab4c01e6829de845cd3ccf1e9a1a5cde370109'; url = "$BlueBase/voices/david.json" },
    @{ name = 'voices/ethan.json';             size =    288591; sha256 = '802e09b837dc8ff8bf2023ea0d9a89b845b5385b5312d00d95734ca40c840b20'; url = "$BlueBase/voices/ethan.json" },
    @{ name = 'voices/jonathan.json';          size =    288406; sha256 = '7af846f12fb30ba29dcf5047e04ce2f9d0f6d4164cd31109220cb03d744a52ff'; url = "$BlueBase/voices/jonathan.json" },
    @{ name = 'voices/michael.json';           size =    288215; sha256 = '2018e228025313923abec2bc6b468db2e4a98ef916fea40d5501c0ec6fb3f335'; url = "$BlueBase/voices/michael.json" },
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
                    return @{
                        Files = @($blue.files)
                        HebrewG2p = @($blue.hebrew_g2p)
                        Int8MaxMemory = [long]$blue.hebrew_g2p_policy.int8_max_memory
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
    return @{ Files = $FallbackModelFiles; HebrewG2p = $FallbackHebrewG2p; Int8MaxMemory = $FallbackInt8MaxMemory }
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
    $models = Install-Models -List $list
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
