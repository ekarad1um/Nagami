# AcousticsLab Frontend — Design-System Notes

A running record of the non-obvious design and engineering decisions taken
while iterating on the Dashboard surface.  Companion to
[`ARCHITECTURE.md`](./ARCHITECTURE.md) and [`PLAN.md`](./PLAN.md).

## Type & casing hierarchy

The app uses a four-tier casing hierarchy.  Mixing tiers inside one surface
is a smell.

| Tier                  | Style                                  | Examples                                          |
| --------------------- | -------------------------------------- | ------------------------------------------------- |
| Brand                 | PascalCase                             | `AcousticsLab`                                    |
| Top-level navigation  | Title Case                             | `Dashboard`, `Workspace`, `Converter`             |
| Panel titles (`h2`)   | Title Case                             | `Visualization`, `Inference`, `Configuration`     |
| Section labels (`h3`) | `UPPERCASE` + `tracking-wider` + 11px  | `MICROPHONE`, `INFERENCE CADENCE`, `ACTIVE HEAD`  |
| Form labels           | Sentence case                          | `Source`, `Channel`, `Hop samples`, `Top-K`       |
| Body / status         | lowercase                              | `live`, `connecting`, `healthy`, `default`        |
| Pills                 | source lowercase + CSS `capitalize`    | renders as `Live`, `Healthy`, `Default`           |

**Pills are special**: the source data stays lowercase (`'live'`,
`'default'`) and `text-transform: capitalize` does the rendering.  This
avoids duplicating string-formatting logic on every site that emits a
status.

## Label normalization

Backend labels follow the Speech-Commands convention of marking synthetic
classes with surrounding underscores: `_unknown_`, `_background_noise_`.
The wire format is treated as data; the display layer maps it via
`prettyLabel()` in
[`TopKMeter.svelte`](../../web/src/lib/components/dashboard/TopKMeter.svelte):

```ts
raw.replace(/^_+/, '').replace(/_+$/, '').replace(/_+/g, ' ')
// _unknown_         -> "unknown"
// _background_noise_ -> "background noise"
// stop              -> "stop"
```

The raw label still lives in the `title` attribute for engineers
inspecting with devtools.

## Corner radius and padding scale

A coherent scale (every value is a multiple of 4px) keeps the optical
rhythm consistent across nested surfaces:

| Surface              | Radius     | Padding   | Ratio  | Notes                                  |
| -------------------- | ---------- | --------- | ------ | -------------------------------------- |
| Top-level panel      | `rounded-xl` (12px) | `p-5` (20px) | 1.67×  | Visualization, Inference, Configuration |
| Nested card          | `rounded-lg` (8px)  | `p-3.5` (14px) | 1.75×  | Active Head card                       |
| Form control         | `rounded-md` (6px)  | `px-2.5 py-1.5` | 1.67H / 1.0V | Selects                                |
| Action button        | `rounded-md` (6px)  | `px-3.5 py-1.5` | 2.33H / 1.0V | Apply buttons                          |
| Pill                 | `rounded-full`      | `px-2 py-0.5`   | —      | Status, origin                         |
| Outer rhythm         | —                   | `gap-5` / `space-y-5` | 20px gap | Matches panel interior padding         |

**Why padding ≥ ~1.5× radius matters**: with `p-4` (16px) + `rounded-xl`
(12px), the corner curve consumes 12 of the 16px padding area, leaving
only 4px of straight padding before the curve starts.  Content near the
corner reads as *tighter* than content near the edges — uneven optical
density.  Pushing the ratio to 1.5–2.0× gives the curve room to sit
*within* the padding rather than dominating it, so all four sides of a
content corner feel evenly cushioned.

**Why the cross-card padding ratio is identical (1.67×)**: the three
top-level panels live side-by-side; matching their interior breathing
room makes the row read as one composition instead of three slightly
different boxes.

**Why nested cards use a slightly higher ratio (1.75×)**: a deeper
hierarchy level should feel *slightly* more contained than its parent.
Pushing nested ratio higher than the parent reads as "this is a card
inside a card" without explicit borders shouting about it.

**Why outer whitespace == panel padding (`gap-5` == `p-5`)**: the eye
reads consistent whitespace as continuous; mismatched gaps make the
layout feel grid-defined rather than card-defined.

## Squircle corners, selectively

`*, *::before, *::after { corner-shape: squircle }` is applied
globally as a progressive enhancement (Chrome 139+, Firefox in progress
as of May 2026; older browsers silently fall back to circular curves).

Pills are exempted: `corner-shape: round` on `.rounded-full` and on the
slider track/thumb pseudo-elements.  At max-radius, a squircle's
slightly-flatter cap subtly breaks the established "pill" identity.
The same exception applies to the Top-K bar's outer container.

The visible-but-not-shouting payoff is on top-level panels:
`border-radius: 12px` + `corner-shape: squircle` gives an iOS-style
soft corner instead of a circular arc.

## Sliders

A custom-rendered control replaces the native chrome to keep
Chromium/Firefox identical:

- **Height** `1.875rem` (30px) — matches the height of a
  `text-xs / py-1.5 / 1px-border` select so a slider and a select on
  adjacent rows share baselines.
- **Track** 6px tall, zinc-200, `rounded-full`.
- **Filled portion** blue-500, painted by a two-stop linear gradient on
  `::-webkit-slider-runnable-track` keyed off a CSS custom property
  `--slider-percent` that the Svelte component computes from `(value -
  min) / (max - min)`.  Firefox uses the dedicated
  `::-moz-range-progress` pseudo and ignores the variable.
- **Thumb** 16px diameter, blue-500, 2px white border, soft shadow.
  Centered on the track via `margin-top: -(thumb_h - track_h) / 2`,
  which by coincidence is `-5px` at both the 4/14 and 6/16 pairings —
  if the ratio changes the formula travels.
- **Hover** scales the thumb to 1.08; **focus** adds a 4px translucent
  blue ring.

The Svelte side just sets `style="--slider-percent: {pct}%"` inline.
No JS listeners on input — Svelte reactivity covers it.

## Scroll-aware fade edges

Top-K can carry up to 20 rows; the panel caps its visible region
with `max-h-44` and scrolls inside.  A 28px linear-gradient mask is
toggled per direction:

```svelte
class:fade-edge-top={canScrollUp}
class:fade-edge-bottom={canScrollDown}
```

`canScrollUp = scrollTop > 0` and `canScrollDown = scrollTop +
clientHeight < scrollHeight - 1`.  The owner re-measures on `scroll`,
on `ResizeObserver` of the container, and inside an `$effect` that
re-runs when `streams.latestTopK` changes shape (so a config-driven
Top-K change updates the fade state immediately).

The 28px gradient was chosen empirically — at 18px the fade reads as
a hairline rather than an affordance; 28px is large enough that the
"more available" cue lands without consuming a full row of content.

## Layout pitfalls (lessons learned)

### Equal-height panels with internal scroll

The natural reflex is `grid-template-rows: min-content` on the parent
grid to force the row height to track only Visualization's content.
**This doesn't work**: an item's `min-content` contribution includes
its full intrinsic content size when it contains explicit-height
descendants (like the Top-K's children).  The row sizes to the larger
item regardless.

The reliable fix is to cap the *content* directly: `max-h-44` on the
Top-K wrapper bounds the Inference panel's intrinsic height under
Visualization's natural height, so the grid auto-sizing chooses
Visualization.  `overflow-y-auto` on the same wrapper handles overflow
inside the cap.  `mt-auto` pins the Active Head card to the panel
bottom so the spacer absorbs any leftover height.

### CSS Grid arbitrary `min-content`

Tailwind v4's `grid-rows-[min-content]` doesn't generate the expected
CSS — `grid-template-rows` gets `repeat(min-content, minmax(0, 1fr))`,
which the browser silently discards.  Use the arbitrary property syntax
`[grid-template-rows:min-content]` or a hand-rolled CSS class.  We
ultimately didn't need either (see above), but this gotcha is worth
remembering.

### Conditional form rows resize their parent

Earlier versions of the Microphone column rendered the Device row only
when Source was "fixed device" and the Channel-index row only when
Channel was "fixed".  Toggling either dropdown made the column jump in
height, throwing off the cross-column alignment.  The fix was to
collapse each two-control pair into a single dropdown:

- `Source`: `auto · first available` or each candidate.
- `Channel`: `auto` or each integer.

The policy's two-field shape (`{ mic: { kind, id? }, channel: { kind,
channel? } }`) is reconstructed at submit time from the single string
value.  Layout is now intrinsically stable; no rows appear or vanish.

### Cross-column row alignment

To make `Source ↔ Hop samples` and `Channel ↔ Top-K` align row-by-row,
both rows must have the same total height *including* the label.  The
key insight: a `<select>`'s height is `text-xs (16) + 2×py-1.5 (12) +
2×1px (border) = 30px`.  Match it on the slider side with
`height: 1.875rem` (30px).  DOM measurements after the fix:
Channel-select bottom and Top-K-slider bottom land within 1px of each
other.

### Visualization bottom padding

A `<footer>` (sample rate + window length metadata) below the
spectrogram pushed the panel's effective bottom padding to ~48px
(`mt-3` + footer line + `p-5`), while the sides stayed at 20px.  The
corner asymmetry was the perceived problem; the footer was the cause.

Resolution: fold the metadata into the panel header as a small
subtitle next to the `h2`, drop the footer.  Now the spectrogram ends
exactly `p-5` (20px) above the card edge, matching the sides.

### Three-tier elevation system

The dashboard uses exactly three surface tones to create perceived depth.
Within each tier every surface is identical; the eye therefore learns
to read tone changes as "this is at a different elevation" rather than
"this is a different type of card".

| Tier              | Color   | Hex       | Examples                                                                 |
| ----------------- | ------- | --------- | ------------------------------------------------------------------------ |
| Page (outer)      | zinc-50 | `#fafafa` | `<body>` background                                                       |
| Card (elevated)   | white   | `#ffffff` | Visualization / Inference / Configuration panels; Health-badge popover; select control |
| Nested data       | zinc-50 | `#fafafa` | Active Head card; WaveformCanvas viewport                                |

The page-tier and nested-data-tier deliberately share `zinc-50`.  Cards
sit between them at white, reading as "raised" surfaces that the
nested-data wells recede back from.  Active Head and the waveform area
visually pair within the white panels — both are `zinc-50` — so the eye
groups them as "this is data living inside the card".

Tones outside this tier system are reserved for *state* and for
*visualization-context* roles, not for hierarchy:

| Purpose                          | Color                                |
| -------------------------------- | ------------------------------------ |
| Status / origin pills            | `bg-emerald-100` / `bg-blue-100` / `bg-zinc-200` |
| Progress-bar tracks (Top-K)      | `bg-zinc-100`                        |
| Slider unfilled track            | `#e4e4e7` (zinc-200)                 |
| Spectrogram canvas (dark heatmap)| `bg-zinc-950`                        |
| Disabled controls                | `bg-zinc-50` / `bg-zinc-100`         |
| Orphaned-head warning            | `bg-amber-50`                        |
| Form error rows                  | `bg-rose-50`                         |

The user learns "tone change = something to notice or a different
elevation", not "tone change = sub-card I should mentally classify".

The waveform canvas's internal fill is wired through a prop
(`background`) defaulting to `#fafafa` so the elevation tier can be
overridden per-instance (e.g., on a future Tiny Dashboard floating
overlay where the parent surface differs).

### Canvas-to-padding visual balance

Numerically-equal padding (`p-5` on all four sides) isn't enough — the
panel can still *look* top-heavy if the header (text + status pill +
`mb-3`) takes ~54px while the bottom only has the bare `p-5` (20px).
The fix isn't more bottom padding (that breaks corner symmetry); it's
giving the *content* enough vertical presence that the bottom
whitespace reads as proportional to the content above it.

Final canvas sizing for the Visualization panel:

- Waveform `h-32` (128px)
- Spectrogram `h-56` (224px)

At those sizes the dominant spec canvas anchors the visual weight, so
the 20px bottom strip stops feeling like a remnant.  An earlier attempt
with `flex-1 min-h-44` on the spectrogram is **not** the answer — it
introduces a flex-grow chain that inflates the grid row size unpredictably
when paired with Inference's `mt-auto` spacer (both wrappers ended up at
570px instead of ~390px because `flex-1` interacted with `align-items:
stretch`).  Fixed heights keep the section's intrinsic size predictable.

## Active Head card

The card carries five fields, of which the `id` is by far the longest
(a 36-char UUID).  Constraints:

1. The `id` must fit on a single line at the typical viewport width.
   Wrapping a UUID looks broken.
2. All five values (`id`, `version`, `classes`, `workspace`,
   `revision`) should share one type size — mixed sizes inside a card
   read as "design accident", not hierarchy.
3. The label column has to be wide enough for the longest label
   (`version`, ~49px at text-xs).

Final form:

- Label column `3.5rem` (56px), value column `1fr`.
- Both `dt` and `dd` cells `text-xs` for the labels and
  `font-mono text-[10px]` for the values.
- `truncate` on every `dd` — even though at typical widths nothing
  truncates, it's the safety net for narrow viewports.  Anything that
  ever overflows shows `…` rather than wrapping.
- `title={fullValue}` on the `id` and `workspace` cells so devtools
  hover surfaces the un-truncated string.

`text-[10px]` is the conventional size for technical hashes/UUIDs in
modern dashboards (Stripe, Vercel) and it earns enough column width
back that the parent panel can keep its 1.67× padding ratio without
forcing a UUID wrap.

## Health badge state colors

| State        | Dot         | Meaning                                |
| ------------ | ----------- | -------------------------------------- |
| `connecting` | zinc-400    | Daemon reachable but no snapshot yet   |
| `ok`         | emerald-500 | All subsystems healthy + fresh         |
| `degraded`   | amber-500   | Any subsystem stale or `degraded_reason` set, or metrics stale |
| `down`       | rose-500    | Daemon unreachable, or any subsystem reports `healthy: false` |

The dot animates with `animate-ping` only in `ok` — degradation and
errors are static so they don't compete with the "everything is fine"
pulse.

## Pitfalls outside the design layer

### `corner-shape: squircle` is universal-selector-safe

Applying via `*, *::before, *::after { corner-shape: squircle }` has
no perceptible perf impact and the property is silently ignored on
elements without a `border-radius`.  Don't try to be clever with
narrow selectors.

### `appearance: none` disables `accent-color`

We strip the native select chrome via `.select-chevron`'s
`appearance: none` and paint our own SVG chevron.  This also disables
`accent-color`, which is why slider track fill is done via custom
gradient + CSS variable instead.

### Tailwind v4 + svelte-prettier + tailwind-prettier-plugin

`prettier-plugin-tailwindcss@0.6.14` crashes `prettier-plugin-svelte`
on `.svelte` files without a `<script>` block (`TypeError:
getVisitorKeys is not a function`).  Mitigation: don't load the
Tailwind class-sorter plugin in `.prettierrc.json` — Svelte's plugin
preserves class order well enough, and ESLint catches non-canonical
class spellings.

## File map

Surfaces touched during this iteration:

- [`web/src/app.css`](../../web/src/app.css) — global tokens, slider, chevron, fade utilities
- [`web/src/routes/+layout.svelte`](../../web/src/routes/+layout.svelte) — tab nav, brand, header
- [`web/src/routes/+page.svelte`](../../web/src/routes/+page.svelte) — Dashboard composition
- [`web/src/lib/components/HealthBadge.svelte`](../../web/src/lib/components/HealthBadge.svelte)
- [`web/src/lib/components/dashboard/VisualizationPanel.svelte`](../../web/src/lib/components/dashboard/VisualizationPanel.svelte)
- [`web/src/lib/components/dashboard/InferencePanel.svelte`](../../web/src/lib/components/dashboard/InferencePanel.svelte)
- [`web/src/lib/components/dashboard/ConfigurationPanel.svelte`](../../web/src/lib/components/dashboard/ConfigurationPanel.svelte)
- [`web/src/lib/components/dashboard/ActiveHeadCard.svelte`](../../web/src/lib/components/dashboard/ActiveHeadCard.svelte)
- [`web/src/lib/components/dashboard/TopKMeter.svelte`](../../web/src/lib/components/dashboard/TopKMeter.svelte)
- [`web/src/lib/components/dashboard/WaveformCanvas.svelte`](../../web/src/lib/components/dashboard/WaveformCanvas.svelte)
- [`web/src/lib/components/dashboard/SpectrogramCanvas.svelte`](../../web/src/lib/components/dashboard/SpectrogramCanvas.svelte)
