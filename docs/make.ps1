# Sphinx documentation build for PowerShell.
#
#   .\make.ps1
#   .\make.ps1 html
#
# SPHINXBUILD overrides the sphinx-build executable. SPHINXOPTS is passed
# through to sphinx-build.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot

$sourceDir = '.'
$buildDir = '_build'
$target = 'help'
$extra = @()
if ($args.Count -gt 0) {
    $target = $args[0]
}
if ($args.Count -gt 1) {
    $extra = @($args[1..($args.Count - 1)])
}

$sphinx = $env:SPHINXBUILD
if (-not $sphinx) {
    $sphinx = 'sphinx-build'
}

if (-not (Get-Command $sphinx -ErrorAction SilentlyContinue)) {
    Write-Host ''
    Write-Host "The 'sphinx-build' command was not found. Make sure you have Sphinx"
    Write-Host 'installed, then set the SPHINXBUILD environment variable to point'
    Write-Host "to the full path of the 'sphinx-build' executable. Alternatively you"
    Write-Host 'may add the Sphinx directory to PATH.'
    Write-Host ''
    Write-Host "If you don't have Sphinx installed, grab it from"
    Write-Host 'https://www.sphinx-doc.org/'
    exit 1
}

$sphinxOpts = @()
if ($env:SPHINXOPTS) {
    $sphinxOpts = @($env:SPHINXOPTS -split '\s+' | Where-Object { $_ })
}

& $sphinx -M $target $sourceDir $buildDir @sphinxOpts @extra
exit $LASTEXITCODE
