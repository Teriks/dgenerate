# dgenerate installer bootstrap (Windows)
#
# Usage (PowerShell):
#   irm https://github.com/Teriks/dgenerate/releases/latest/download/install.ps1 | iex
#
# Downloads the installer attached to this GitHub release and launches it.
# In the installer UI, choose "Latest Release" for the current stable build.
#
# Arguments are forwarded to the installer when this file is executed directly:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -s -v 5.0.0
#
# The release workflow replaces the tag placeholder on the $Tag assignment
# below before uploading this file. Do not repeat that placeholder earlier.

& {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$InstallerArgs
    )

    $ErrorActionPreference = 'Stop'
    if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
        $PSNativeCommandUseErrorActionPreference = $false
    }
    $ProgressPreference = 'SilentlyContinue'

    $tmp = $null
    try {
        try {
            $tls12 = [Net.SecurityProtocolType]::Tls12
            $current = [Net.ServicePointManager]::SecurityProtocol
            if (($current -band $tls12) -eq 0) {
                [Net.ServicePointManager]::SecurityProtocol = $current -bor $tls12
            }
        } catch {
        }

        if (-not $InstallerArgs) {
            $InstallerArgs = @()
        }

        $Repo = 'Teriks/dgenerate'
        $Tag = '@DGENERATE_RELEASE_TAG@'
        if ($Tag -eq '@DGENERATE_RELEASE_TAG@') {
            throw 'this installer script was not published with a release tag.'
        }

        $ApiUrl = "https://api.github.com/repos/$Repo/releases/tags/$([uri]::EscapeDataString($Tag))"

        if ($env:OS -ne 'Windows_NT') {
            throw "this script supports Windows only. On Linux or macOS: curl -fsSL https://github.com/$Repo/releases/download/$Tag/install.sh | sh"
        }

        $arch = $env:PROCESSOR_ARCHITEW6432
        if (-not $arch) {
            $arch = $env:PROCESSOR_ARCHITECTURE
        }
        if ($arch -and $arch -ne 'AMD64') {
            Write-Host "warning: the Windows installer is built for x64; detected '$arch'. Continuing anyway."
        }

        Write-Host "Looking up the $Tag installer..."
        $headers = @{
            'User-Agent' = 'dgenerate-install'
            'Accept'     = 'application/vnd.github+json'
        }
        try {
            $release = Invoke-RestMethod -Uri $ApiUrl -Headers $headers
        } catch {
            throw "could not fetch release info from $ApiUrl (GitHub API rate limit?)"
        }

        if (-not $release -or -not $release.assets) {
            throw "could not fetch release info from $ApiUrl (GitHub API rate limit?)"
        }

        $asset = @($release.assets) | Where-Object {
            $_.name -like 'dgenerate-installer-win-*.exe'
        } | Select-Object -First 1

        if (-not $asset -or -not $asset.browser_download_url) {
            throw "no Windows installer found in release '$Tag'. See https://github.com/$Repo/releases/tag/$Tag"
        }

        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("dgenerate-install-" + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $tmp -Force | Out-Null
        $installer = Join-Path $tmp $asset.name

        Write-Host "Downloading $($asset.name)..."
        $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
        if ($curl) {
            & $curl.Source -fL --progress-bar -o $installer $asset.browser_download_url
            if ($LASTEXITCODE -ne 0) {
                throw "download failed."
            }
        } else {
            $client = New-Object System.Net.WebClient
            try {
                $client.Headers['User-Agent'] = 'dgenerate-install'
                $client.DownloadFile($asset.browser_download_url, $installer)
            } finally {
                $client.Dispose()
            }
        }

        Unblock-File -LiteralPath $installer -ErrorAction SilentlyContinue

        Write-Host ""
        Write-Host "Launching the installer."
        Write-Host '  -> Select "Latest Release" to install the current stable version.'
        Write-Host "If Windows SmartScreen appears, choose More info, then Run anyway."
        Write-Host ""

        $startParams = @{
            FilePath         = $installer
            WorkingDirectory = $env:USERPROFILE
            Wait             = $true
            PassThru         = $true
        }
        if ($InstallerArgs.Count -gt 0) {
            $startParams.ArgumentList = $InstallerArgs
        }
        $proc = Start-Process @startParams
        if ($null -ne $proc.ExitCode -and $proc.ExitCode -ne 0) {
            throw "installer exited with code $($proc.ExitCode)."
        }
        $global:LASTEXITCODE = 0
    } catch {
        $message = $_.Exception.Message
        if (-not $message) {
            $message = "$_"
        }
        Write-Host "error: $message" -ForegroundColor Red
        $global:LASTEXITCODE = 1

        $shouldExit = [bool]$PSCommandPath
        if (-not $shouldExit) {
            try {
                $shouldExit = [Console]::IsInputRedirected
            } catch {
                $shouldExit = $true
            }
        }
        if ($shouldExit) {
            exit 1
        }
    } finally {
        if ($tmp -and (Test-Path -LiteralPath $tmp)) {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
} @args
