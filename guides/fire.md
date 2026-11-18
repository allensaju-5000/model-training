# Labelling guide: fire and smoke

Classes (ids): `smoke` = 0, `fire` = 1 (same as D-FIRE).

## fire
- Visible flames: the bright yellow/orange/white burning region, including the
  glow core. Not reflections, embers on the ground, or orange sky/sunset.
- Separate burning regions (different windows, separate fires on one building)
  get separate boxes. A continuous wall of flame is one box.
- Scenes engulfed in flame (wildfire walls, lava-like fields, flashover rooms) get
  one box over the whole burning area, even if that is most of the frame.

## smoke
- In Smoke images: one box per plume or smoke column, covering the dense,
  clearly visible part; stop where it fades into haze or sky.
- In fire images: add smoke boxes only when the smoke is clearly distinct from the
  flames (a separate dark plume above a car or building fire), not for haze
  around the flames.

## NEG
- Everything in `No_Fire/Safe_Fire` and `No_Fire/Artificial_Fire` (candles,
  stoves, fireplaces, screens, lamps): nothing to box. These are the negatives
  that teach the model what is not a dangerous fire.

## SKIP
- Images outside the domain: e.g. tornado or dust funnels in the Smoke folder
  (looks like smoke, is not fire smoke). Labelling them as smoke would teach
  false alarms; labelling them NEG would contradict similar real smoke.

## Edge cases decided so far
- Pale, washed-out flashover frames (room fully alight, low contrast) are fire.
- Boxes may run to the image edge; do not stop a box short of the edge for
  flames that continue out of frame.
- A long row of many small burning windows on one vehicle or facade (e.g. a train
  carriage) gets one box over the burning row, not one box per window.
- Volcanic eruption plumes where the volcano/cone is visible: SKIP (ash, not fire
  smoke; same reasoning as tornadoes). Huge plumes with no visible source stay
  `smoke` but are listed as uncertain.
