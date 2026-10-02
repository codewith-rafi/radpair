# RadPair Project Page

Static GitHub Pages site for **RadPair**: cross-modal alignment and unified representation methods for chest X-ray images and clinical text.

Content mirrors the repository README and `experiments/README.md`. Every number on the page comes from a run's `metrics.json`; nothing is estimated or reworded for effect.

Live URL: <https://codewith-rafi.github.io/radpair/>

## Layout

```
index.html                 page content, meta tags, JSON-LD
.nojekyll                  keeps GitHub Pages from filtering assets
static/css/                bulma + index.css
static/js/index.js         BibTeX copy, scroll-to-top, related-work dropdown
static/images/*.svg        figures, generated from the reported metrics
```

## Figures

All figures are hand-written SVG, so they stay crisp and add no raster weight. To regenerate one after a new run, edit the bar heights in the corresponding file; every figure carries a `<title>` and `<desc>` for accessibility and is also described in the page caption.

| File | Shows |
|---|---|
| `fig_architecture.svg` | Dual encoder, shared 256-D space, three heads |
| `fig_protocol.svg` | Licensed dump to sealed test, with safeguards |
| `fig_validation_retrieval.svg` | Three-way validation Recall@1 with chance line |
| `fig_train_val_gap.svg` | Train versus validation at best epoch |
| `fig_test_retrieval.svg` | Sealed-test Recall@1/5/10, both directions |
| `fig_disease_itm.svg` | Disease image-only vs text-only, and ITM |
| `fig_ablations.svg` | A2 adaptation on/off, A1 text section |
| `fig_embed_api.svg` | Embedding API interface contract |
| `social_preview.svg` | 1200x630 social card |
| `favicon.svg` | Site icon |

## Deploy (GitHub Pages)

This folder is published as the **site root** by `.github/workflows/pages.yml` (`path: docs/report`). Do **not** use Settings → Pages → “Deploy from a branch” with `/docs` — that would publish all of `docs/` (including `embed_api.md`) instead of this page.

### One-time setup

1. Open the repo on GitHub → **Settings → Pages**
2. Under **Build and deployment → Source**, choose **GitHub Actions**
3. Push `docs/report/` and the workflow (or run **Actions → Deploy report to Pages → Run workflow**)
4. Confirm the workflow is green, then open <https://codewith-rafi.github.io/radpair/>

If the workflow is green but the site 404s, Pages Source is still set to branch deploy.

### Local preview

Open `docs/report/index.html` in a browser, or:

```bash
npx --yes serve docs/report
```
