# Bundled 3D viewer

`brax_viewer_v0.1.0_three_r135.js` bundles into a single script (IIFE, global `window.BraxViewer`):

- the Brax HTML viewer, `js/{viewer,animator,selector,system}.js` at tag `google/brax@v0.1.0` (the one loaded by
  `brax.v1.io.html`), Apache 2.0 licence (`LICENSE-brax.txt`);
- three.js r135 (`three@0.135.0` on npm) with OrbitControls, lil-gui and ParametricGeometry, MIT licence
  (`LICENSE-three.txt`).

It is inlined in every simulation page written by `qd_damage.viz`, so the pages work offline without a CDN.

To rebuild it:

```bash
npm install three@0.135.0 esbuild@0.24.0
# download the 4 js files from https://cdn.jsdelivr.net/gh/google/brax@v0.1.0/js/
# replace the https://cdn.jsdelivr.net/gh/mrdoob/three.js@r135/... imports with 'three' / 'three/examples/jsm/...'
printf "import {Viewer} from './viewer.js';\nwindow.BraxViewer = Viewer;\n" > entry.js
npx esbuild entry.js --bundle --minify --format=iife --legal-comments=inline --outfile=brax_viewer_v0.1.0_three_r135.js
```
