# Frontend layout fix prompt

You are a senior frontend engineer.

Repository:
https://github.com/askarov32/ai-directional-prediction

Task:
Fix the frontend spacing and layout issues in the `frontend` directory.

Context:
The UI is a vanilla HTML/CSS/JavaScript frontend. Do not introduce React, Tailwind, build tools, or new dependencies. Keep the current visual style: dark scientific dashboard, 2D directional prediction workspace.

Current visual problems from screenshots:
1. The page has weak horizontal spacing. The main hero and workspace are too close to the viewport edges.
2. The `Prediction result` area looks cramped.
3. Several labels inside the planar SVG preview overlap each other near the top-left area:
   - model label
   - domain label
   - axis label
   - source/probe metadata
4. After running prediction, the “Ready for inference” empty state still appears above populated result cards. It must disappear after a result is available.
5. Result cards should not overlap or visually collapse when values are long, especially:
   - model version
   - request id
   - displacement values
   - vector values

Files to inspect and modify:
- `frontend/index.html`
- `frontend/assets/styles/layout.css`
- `frontend/assets/styles/components.css`
- `frontend/assets/styles/base.css`
- `frontend/assets/scripts/charts.js`
- `frontend/assets/scripts/ui.js`

---

## 1. Add stable page side spacing

Find the outer page/container class used by the main UI. It may be named something like:
- `.page-shell`
- `.app-shell`
- `.workspace-shell`
- `.container`

Make sure the main page content has consistent left/right padding:

```css
.page-shell {
  width: min(100% - 48px, 1680px);
  margin-inline: auto;
  padding-block: clamp(32px, 4vw, 72px);
}

@media (max-width: 768px) {
  .page-shell {
    width: min(100% - 24px, 1680px);
    padding-block: 24px 40px;
  }
}
```

If the actual class is different, apply the same logic to the real outer wrapper. Do not create duplicate unused classes.

---

## 2. Fix workspace two-column layout

The setup panel and result panel should have a stable two-column layout on desktop and collapse on smaller screens.

Use this layout behavior:

```css
.workspace,
.workspace-grid,
.prediction-workspace {
  display: grid;
  grid-template-columns: minmax(360px, 540px) minmax(0, 1fr);
  gap: clamp(24px, 3vw, 44px);
  align-items: start;
}

@media (max-width: 1100px) {
  .workspace,
  .workspace-grid,
  .prediction-workspace {
    grid-template-columns: 1fr;
  }
}
```

Use the real class name from the HTML. Do not blindly add all three selectors if only one exists.

Important:
- Add `min-width: 0` to the right/result column.
- Add `min-width: 0` to cards that contain long values.

Example:

```css
.card,
.result-panel,
.result-card,
.metric-card {
  min-width: 0;
}
```

---

## 3. Hide empty result state correctly

There is a bug where the empty state “Ready for inference” is still shown after prediction.

Add a generic hidden utility:

```css
.is-hidden {
  display: none !important;
}
```

Or, if the project avoids global utilities, explicitly support:

```css
.empty-state.is-hidden,
.result-content.is-hidden,
.error-banner.is-hidden {
  display: none !important;
}
```

Then verify in `frontend/assets/scripts/ui.js` that:
- `renderResult()` adds `is-hidden` to `resultEmpty`
- `renderResult()` removes `is-hidden` from `resultContent`
- `renderIdle()` removes `is-hidden` from `resultEmpty`
- `renderIdle()` adds `is-hidden` to `resultContent`

Expected behavior:
- Before prediction: show “Ready for inference”, hide result cards.
- After successful prediction: hide “Ready for inference”, show result cards.
- After reset/error: behavior must stay consistent.

---

## 4. Fix Prediction result card grid

Result cards should be visually separated and should not collapse.

Add or adjust result content layout:

```css
.result-content {
  display: grid;
  gap: 1rem;
}

.result-metrics,
.result-grid,
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}

@media (max-width: 900px) {
  .result-metrics,
  .result-grid,
  .metrics-grid {
    grid-template-columns: 1fr;
  }
}
```

Use only the real class names from the HTML.

For long values:

```css
.metric-card,
.metadata-card {
  overflow: hidden;
}

.metric-card strong,
.metadata-card strong,
.mono-value {
  overflow-wrap: anywhere;
  word-break: break-word;
}
```

---

## 5. Fix SVG preview label overlap in `charts.js`

Open:

```text
frontend/assets/scripts/charts.js
```

The SVG preview currently places multiple text labels too close to each other. Refactor the label positions so the chart has reserved header space.

Required behavior:
- Domain rectangle should start lower, leaving enough room above it.
- Model label should be above the domain rectangle.
- Domain summary should be below the model label.
- Axis labels should not overlap the model/domain labels.
- Source/probe labels should be offset from points.
- Distance/azimuth label should not cover the source/probe labels.

Suggested constants:

```js
const box = {
  width: 1000,
  height: 620,
  marginLeft: 90,
  marginRight: 70,
  marginTop: 105,
  marginBottom: 70,
};
```

If the current code uses a single `margin`, replace it with separate margins:
- `marginLeft`
- `marginRight`
- `marginTop`
- `marginBottom`

Then update projection logic so the drawable domain uses:

```js
const innerWidth = width - marginLeft - marginRight;
const innerHeight = height - marginTop - marginBottom;
```

For 2D domain projection:
- `x = marginLeft + normalizedX * innerWidth`
- `y = marginTop + (1 - normalizedY) * innerHeight`

Place labels approximately like this:

```js
const titleY = 45;
const subtitleY = 70;
const domainTopY = box.marginTop;
```

Model label:

```js
x = box.marginLeft;
y = titleY;
```

Domain summary:

```js
x = box.marginLeft;
y = subtitleY;
```

Y-axis label:

```js
x = box.marginLeft - 55;
y = box.marginTop - 12;
```

X-axis label:

```js
x = box.width - box.marginRight + 10;
y = box.height - box.marginBottom + 28;
```

Do not place the y-axis label at the same vertical position as the model label.

---

## 6. Make visual stage responsive

In CSS:

```css
.visual-stage {
  width: 100%;
  overflow: hidden;
}

.visual-stage svg {
  display: block;
  width: 100%;
  height: auto;
  min-height: 420px;
  max-height: 560px;
}
```

If labels still get clipped, use a larger SVG viewBox instead of absolute CSS scaling.

---

## 7. Improve header spacing in Prediction result

The result card header should not squeeze title, subtitle, model badge, and latency badge.

```css
.card__header--result {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}

.result-badges {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  justify-content: flex-end;
}

@media (max-width: 700px) {
  .card__header--result {
    flex-direction: column;
  }

  .result-badges {
    justify-content: flex-start;
  }
}
```

---

## 8. Do not change backend/API/model logic

Do not modify:
- backend prediction contract
- model services
- API response format
- physical values
- prediction calculations

This task is only frontend layout, spacing, SVG label positioning, and visibility state.

---

## Acceptance criteria

1. Page has clear left/right spacing on desktop and mobile.
2. Setup panel and Prediction result panel are aligned with a consistent gap.
3. `Prediction result` no longer has overlapping elements.
4. “Ready for inference” is hidden after prediction result appears.
5. SVG labels do not overlap near the top-left area.
6. Long metadata values wrap inside cards instead of breaking layout.
7. Mobile layout remains readable.

---

## Important wording rules

- Do not describe the product as a complete field-validated thermoelastic simulator.
- Keep the positioning as a research prototype / AI-assisted directional prediction workspace.
- Keep the UI 2D only.
- Do not reintroduce 3D controls, 3D wording, or 3D visualization.
- Do not change the backend contract.
- Do not fabricate new physical outputs.
- Do not hide important values only to make the layout look cleaner.
- Prefer wrapping, spacing, and grid fixes over deleting UI content.
