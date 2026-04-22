param(
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DistDir = Join-Path $RootDir "dist/windows"
$MainClass = "atri.palaash.jforge.app.JForgeApp"
$IconPath = Join-Path $RootDir "assets/icons/jforge.ico"
$RuntimeImage = $env:JFORGE_RUNTIME_IMAGE

Push-Location $RootDir

if (-not (Test-Path "target/jforge-1.0.0-SNAPSHOT.jar")) {
    mvn clean -DskipTests package
}

New-Item -Path $DistDir -ItemType Directory -Force | Out-Null

$jpackageArgs = @(
  "--type", "exe",
  "--name", "JForge",
  "--dest", $DistDir,
  "--input", "target",
  "--main-jar", "jforge-1.0.0-SNAPSHOT.jar",
  "--main-class", $MainClass,
  "--app-version", $Version,
  "--vendor", "JForge",
  "--win-menu",
  "--win-shortcut",
  "--win-dir-chooser",
  "--win-per-user-install"
)

if (Test-Path $IconPath) {
    $jpackageArgs += @("--icon", $IconPath)
}

if ($RuntimeImage -and (Test-Path $RuntimeImage)) {
    $jpackageArgs += @("--runtime-image", $RuntimeImage)
}

jpackage @jpackageArgs

Pop-Location

Write-Host "Windows EXE created in: $DistDir"
