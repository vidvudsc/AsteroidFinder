from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import os
import shutil
import stat
import sys


def main() -> int:
    if sys.platform != "darwin":
        raise SystemExit("The AsteroidFinder.app launcher is only for macOS.")
    repo_root = Path(__file__).resolve().parents[1]
    app_root = repo_root / "dist" / "AsteroidFinder.app"
    contents = app_root / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    icon_source = Path(str(files("asteroidfinder_desktop").joinpath("assets/icon.png")))
    shutil.copy2(icon_source, resources / "icon.png")
    (contents / "Info.plist").write_text(_info_plist(), encoding="utf-8")
    launcher = macos / "AsteroidFinder"
    launcher.write_text(_launcher_script(repo_root), encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(app_root)
    return 0


def _launcher_script(repo_root: Path) -> str:
    python = Path(sys.executable)
    return f"""#!/bin/zsh
cd {sh_quote(repo_root)}
export PYTHONPATH={sh_quote(repo_root)}:${{PYTHONPATH:-}}
exec {sh_quote(python)} -m asteroidfinder_desktop
"""


def _info_plist() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "https://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>AsteroidFinder</string>
  <key>CFBundleIdentifier</key>
  <string>com.asteroidfinder.desktop</string>
  <key>CFBundleName</key>
  <string>AsteroidFinder</string>
  <key>CFBundleDisplayName</key>
  <string>AsteroidFinder</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
"""


def sh_quote(path: Path) -> str:
    return "'" + os.fspath(path).replace("'", "'\\''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
