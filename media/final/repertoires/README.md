# Répertoires finaux — 3 algos × 3 seeds, 2000 itérations

Généré le 2026-10-01T02:33:42 (code `cbec89b`). Runs complets (2000 itérations par run, 3 algorithmes × 3 seeds) : résultats finaux, avant blessure.

Régénérer : `python -m qd_damage.viz --runs results/repertoires/me_seed0 results/repertoires/me_seed1 results/repertoires/me_seed2 results/repertoires/pgame_seed0 results/repertoires/pgame_seed1 results/repertoires/pgame_seed2 results/repertoires/dcrlme_seed0 results/repertoires/dcrlme_seed1 results/repertoires/dcrlme_seed2 --out media/final/repertoires --title "Répertoires finaux — 3 algos × 3 seeds, 2000 itérations"`

Meilleure démarche : PGA-ME, case 581.
Intact : 39.75 m vers l'avant en 12.5 s.
Patte 4 paralysée (actions [6, 7] à 0) : 12.86 m.

## Pour quoi faire

| Usage | Fichier |
|---|---|
| README GitHub | `videos/intact_vs_patte4_camera_fixe.gif` |
| LinkedIn, slides | `videos/*.mp4` (H.264, temps réel, 20 i/s) |
| Portfolio web (3D à la souris) | `interactive/*.html` : pages autonomes, à mettre en `<iframe>` ou en lien |
| Rapport LaTeX | `figures/*.pdf` (vectoriel) |
| Web, slides | `figures/*.png` (200 dpi) |
| Chiffres et provenance | `data/summary.json`, `data/run_*.json`, `data/metrics_*.csv` |

## Fichiers

### figures/
- `figures/contacts_pieds.pdf`
- `figures/contacts_pieds.png`
- `figures/progression_qd.pdf`
- `figures/progression_qd.png`
- `figures/repertoire_dcrlme.pdf`
- `figures/repertoire_dcrlme.png`
- `figures/repertoire_dcrlme_seed0.pdf`
- `figures/repertoire_dcrlme_seed0.png`
- `figures/repertoire_dcrlme_seed1.pdf`
- `figures/repertoire_dcrlme_seed1.png`
- `figures/repertoire_dcrlme_seed2.pdf`
- `figures/repertoire_dcrlme_seed2.png`
- `figures/repertoire_me.pdf`
- `figures/repertoire_me.png`
- `figures/repertoire_me_seed0.pdf`
- `figures/repertoire_me_seed0.png`
- `figures/repertoire_me_seed1.pdf`
- `figures/repertoire_me_seed1.png`
- `figures/repertoire_me_seed2.pdf`
- `figures/repertoire_me_seed2.png`
- `figures/repertoire_pgame.pdf`
- `figures/repertoire_pgame.png`
- `figures/repertoire_pgame_seed0.pdf`
- `figures/repertoire_pgame_seed0.png`
- `figures/repertoire_pgame_seed1.pdf`
- `figures/repertoire_pgame_seed1.png`
- `figures/repertoire_pgame_seed2.pdf`
- `figures/repertoire_pgame_seed2.png`
- `figures/trajets_vus_de_dessus.pdf`
- `figures/trajets_vus_de_dessus.png`

### videos/
- `videos/intact_camera_suiveuse.mp4`
- `videos/intact_vs_patte4_camera_fixe.gif`
- `videos/intact_vs_patte4_camera_fixe.mp4`
- `videos/patte4_paralysee_camera_suiveuse.mp4`

### interactive/
- `interactive/intact.html`
- `interactive/patte4_paralysee.html`

### data/
- `data/metrics_dcrlme_seed0.csv`
- `data/metrics_dcrlme_seed1.csv`
- `data/metrics_dcrlme_seed2.csv`
- `data/metrics_me_seed0.csv`
- `data/metrics_me_seed1.csv`
- `data/metrics_me_seed2.csv`
- `data/metrics_pgame_seed0.csv`
- `data/metrics_pgame_seed1.csv`
- `data/metrics_pgame_seed2.csv`
- `data/rollouts.npz`
- `data/run_dcrlme_seed0.json`
- `data/run_dcrlme_seed1.json`
- `data/run_dcrlme_seed2.json`
- `data/run_me_seed0.json`
- `data/run_me_seed1.json`
- `data/run_me_seed2.json`
- `data/run_pgame_seed0.json`
- `data/run_pgame_seed1.json`
- `data/run_pgame_seed2.json`
- `data/summary.json`
