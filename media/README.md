# Media

Videos, interactive 3D pages and figures. Each visual folder is produced by `python -m qd_damage.viz` or
`python -m qd_damage.viz_ite` and holds its own `README.md` (file list, provenance, command to regenerate it)
and an `index.html` overview page. Page texts and the figures in `figures/` are in French; the figures in
`figures_en/` are the English versions used by the reports.

| Folder | Content | Status |
|---|---|---|
| [`final/ite_main/`](final/ite_main/) | Adaptation on a DCRL-ME repertoire (seed 0), front-right leg paralysed. All five ITE repetitions recommend the same gait (after 6 or 7 trials); repetition 2 is shown. Means over 8 episodes: intact 30.6 m, damaged with the same gait 2.8 m, after ITE 9.4 m | Final results |
| [`final/ite_pgame/`](final/ite_pgame/) | Same on a PGA-ME repertoire (seed 0), median repetition (4 trials): 43.0 m, 10.3 m, 12.1 m | Final results |
| [`final/figures/`](final/figures/), [`final/figures_en/`](final/figures_en/) | Analysis figures of the full campaign | Final results |
| [`final/repertoires/`](final/repertoires/) | The 9 final repertoires (3 algorithms x 3 seeds, 2000 iterations, T4 GPU); best stored gait (PGA-ME, seed 1): a bounding gait, 39.7 m in 12.5 s intact and 12.9 m with leg 4 paralysed (one episode each) | Final results, before adaptation |
| [`checks/leg_mapping/`](checks/leg_mapping/) | Video checking which action drives which leg | Check |

## Layout of a visual folder

```
<folder>/
├── index.html      overview page (open in a browser)
├── README.md       catalogue and provenance
├── figures/        PNG (200 dpi) and vector PDF
├── videos/         real-time H.264 MP4 and GIF
├── interactive/    self-contained 3D pages (drag to rotate, scroll to zoom), work offline
└── data/           summary.json (numbers shown), run_*.json, metrics_*.csv, trajectories
```

The 3D pages inline the Brax viewer and three.js (Apache 2.0 / MIT, see `src/qd_damage/assets/`), so a single
HTML file can be embedded with an `<iframe>`. Numbers quoted elsewhere come from the `data/summary.json` files.
The repertoires themselves (`results/repertoires/`) are too large for Git; they are kept as outputs of the private Kaggle kernels.
