# scripts/render_demo_gif.tape — hero GIF recording (T029, US1, FR-022)
#
# Records the reviewer-facing 5-command flow from
# specs/002-portfolio-polish/quickstart.md as an animated terminal GIF
# embedded at the top of README.md. The viewer sees what `bin-packer`
# does in ~30 seconds without leaving the README.
#
# RUN:
#     vhs scripts/render_demo_gif.tape -o docs/assets/hero.gif
#
# Prerequisites:
#   - vhs >= 0.11 (`brew install vhs ttyd` on the maintainer machine —
#     not currently in CI because vhs needs a TTY).
#   - bin-packer-3d installed in the active env BEFORE running this
#     tape. The recording assumes `bin-packer` is on PATH; the
#     "install" command in the tape is shown as a typed-then-cleared
#     line (Ctrl+U) so the viewer sees the intent without a real
#     30-second pip resolution blowing the 30-second budget.
#     Recommended setup before recording:
#         python -m venv /tmp/hero-venv
#         source /tmp/hero-venv/bin/activate
#         pip install -e '.[viz]'   # local editable so the registry is fresh
#   - The repo's `examples/headline.csv` is reachable from CWD when
#     vhs runs (relative path used in the pack command below).
#
# Tape parameters (per spec-02 T029):
#   - 800×500 viewport
#   - 8 fps
#   - ~30 seconds total runtime
#
# Theme: Catppuccin Mocha (dark, colourblind-friendly, brand-cohesive
# with bin-packer-3d's BIN_PACKER_3D_DARK visualisation theme from
# the US2 plotter rewrite).

Output docs/assets/hero.gif

Require bin-packer

Set Shell "bash"
Set Width 800
Set Height 500
Set FontSize 14
Set FontFamily "JetBrains Mono"
Set Theme "Catppuccin Mocha"
Set Padding 18
Set Framerate 8
Set TypingSpeed 45ms

# === Setup (hidden) — fresh prompt, clean working directory =================
Hide
Type "cd $(mktemp -d -t bin-packer-hero-XXXX)"
Enter
Type "clear"
Enter
Show

# === Command 1: install (typed-then-cleared per header note) ===============
# Visually presents the install one-liner that ships in the README
# text adjacent to this GIF. Ctrl+U clears the line without invoking
# pip — the tape's host env is assumed pre-installed (see header).
Type "pip install 'bin-packer-3d[viz]'"
Sleep 1500ms
Ctrl+u
Sleep 400ms

# === Command 2: info — registered algorithms ===============================
Type "bin-packer info"
Sleep 300ms
Enter
Sleep 2500ms

# === Command 3: pack the bundled headline dataset with BFD =================
Type "bin-packer pack examples/headline.csv --strategy bfd --visualize -o out/"
Sleep 300ms
Enter
Sleep 3500ms

# === Command 4: demo — every registered algorithm side by side =============
Type "bin-packer demo -o demo/"
Sleep 300ms
Enter
Sleep 3500ms

# === Command 5: open — show the artefacts the demo produced ================
# `xdg-open` would launch the system browser, which vhs cannot
# meaningfully capture. We substitute `ls` of the produced files so
# the viewer sees the concrete output paths and can recognise the
# bin_<N>.html files referenced in the README narrative.
Type "ls -1 demo/bfd/"
Sleep 300ms
Enter
Sleep 2500ms

# Final pause — let the viewer take in the last frame before loop.
Sleep 1500ms
