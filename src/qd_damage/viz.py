"""Repertoire visuals: figures, videos and interactive 3D pages, laid out for reuse (portfolio, report).

Usage:
    python -m qd_damage.viz --runs results/repertoires/*_seed? --out media/final/repertoires --title "..."

Layout written in --out:
    index.html      overview page (light: it links to the files below)
    README.md       file catalogue, provenance and the command that regenerates everything
    figures/        each figure as PNG (web, slides) and vector PDF (LaTeX)
    videos/         H.264 MP4 and GIF
    interactive/    self-contained 3D pages (Brax viewer + three.js inlined), work offline
    data/           displayed numbers (summary.json), trajectories, copies of the metrics.csv / run.json used

