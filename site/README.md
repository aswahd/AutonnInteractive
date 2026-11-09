# Auto nnInteractive project page

A self-contained adaptation of [Roman Hauksson’s Astro academic project template](https://github.com/RomanHauksson/academic-project-astro-template). Builds to static HTML, CSS, JavaScript, and preview images. No Python, model server, or parent-repository files are needed.

## Local development

Use Node.js 24 or newer. Inside this directory, run:

```sh
npm ci
npm run dev
```

Build with `npm run build`; preview with `npm run preview`. The complete deployable site is in `dist/`. Move this entire directory wherever you like and use the same commands. Dependencies and build outputs are ignored.

## GitHub Pages

For https://aswahd.github.io/AutonnInteractive/:

```sh
SITE_URL=https://aswahd.github.io BASE_PATH=/AutonnInteractive/ npm run build
```

Change these values for another host or repository. The default base is `/`; all demo assets respect the configured base.

An optional workflow is in `deployment/github-pages.yml`. Copy it into the hosting repository’s `.github/workflows/`, set `BUILD_PATH` to the site's directory (`site` here, `.` for a standalone repository), and select **Settings → Pages → Source → GitHub Actions**. No workflow is installed in the parent repository automatically.

## Editing

- `src/paper.mdx`: content, project links, and template header.
- `src/components/Workflow.astro`: recorded workflow display.
- `src/scripts/demo.js`: organ selection, playback, and mask controls.
- `src/styles/workflow.css`: styling, animation, and mobile layout.
- `public/assets/`: AbdomenAtlas previews for liver, spleen, left kidney, and pancreas, each with four supports and eight queries.

The demo plays saved model outputs and supports pause, replay, stage selection, keyboard navigation, and reduced-motion preferences. Each card pairs an axial CT slice and mask overlay with a rendered 3D segmentation preview. Original volumes, checkpoints, and generation scripts are not needed to build or serve it.

Template attribution and adaptation details are in `TEMPLATE.md` and the page footer.
