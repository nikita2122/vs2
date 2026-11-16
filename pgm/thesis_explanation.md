# How to read the thesis

This note explains the thesis *Research on an Automated Box-Counting Method Using a Deep Instance Segmentation Model*. It is a guide to the argument, the two technical updates, and the numbers. It is not a second copy of the thesis, and it is not a field report. The thesis itself is the LaTeX document in the `main` folder. The two short papers in `paper1` and `paper2` each develop one of the updates on its own.

## What problem the thesis is solving

At the end of a brewery packaging line the product is no longer a bottle on a conveyor. It is a plastic crate, the crates are stacked on a pallet, and a forklift is driving that pallet into the warehouse. That short passage is the last moment the load is presented, face-on, to a camera the building can own. A count taken earlier, on the conveyor, misses stacking mistakes. A count taken later, from the aisle, sees the long side of many pallets and a great deal of occlusion between them.

The camera looks at the side of the load. It sees the front face only. Crates behind that face change the inventory and do not appear in the image. The system is asked for two numbers:

- how many crates of each body color passed (amber, green, red, blue, yellow, and white in the design palette);
- the total, which is the sum of those color counts.

Color is not a cosmetic label. Two colors on one pallet are normal, and a swap of amber and yellow leaves the total unchanged while the stock record is wrong. A missed cell hurts both the total and the color line.

The geometry used throughout is a design choice, not a tape measurement of one brewery's crate: a front face 0.40 m wide and 0.28 m tall, a crate depth of 0.30 m, stacks of a few crates in each direction, and a pallet depth of one, two, or three crates. The face is what the image contains. The depth is what the image does not contain.

## Why a plain segmenter is not an answer

An instance segmenter, including the YOLO family, proposes a mask and a center for each visible object. That is necessary and not sufficient. The thesis lists four failures.

1. **Merged seams.** Two crates of the same color touch along a thin horizontal lip. If the segmenter keeps one box, the tally is short by one crate.
2. **Color under the wrong lamp.** A classifier trained on RGB in one aisle has learned the product of the lamp and the plastic. Another door, or another lamp, moves the decision. Amber and yellow are only about 9.3 degrees apart in the design palette, so they collide first. White sits near neutral and drifts toward several classes rather than toward one neighbor.
3. **The invisible rank.** A perfect mask of the front face, multiplied by nothing, is the front count. If the pallet is three crates deep, that perfect mask is wrong by a factor of three. No improvement in mask accuracy repairs a factor of three.
4. **Position-dependent misses.** The forks shadow the bottom row. The top row is clipped by the frame and smeared if the shutter is slow. These misses are not independent coin flips.

Published neighbors each solve a piece. Mask R-CNN, YOLACT, and YOLO-seg produce instances. Warehouse YOLO variants and lightweight rotated box detectors run at line rate. Area-ratio counting recovers partial goods when there is no regular pitch. A recent 3D counting method estimates volume times occupancy for jumbled identical objects seen from several views. Crowd density maps integrate a blob count. Global color constancy estimates one gain for a whole photograph. None of them both names body color under a changing lamp and turns a side view of a regular crate stack into a count that includes an integer depth. The thesis does not replace the segmenter. It adds the two missing stages.

## The baseline the thesis updates

YOLO-vf-seg is the project's baseline, not a new claim. It is a one-stage segmenter in the YOLOv8-seg line: a cross-stage backbone, a feature pyramid, a decoupled detection head, and a mask built the way YOLACT builds a mask. A dictionary of prototype masks covers the image, each detection predicts a short vector of coefficients, and the instance mask is the linear combination of the prototypes, cropped to the box.

The letters "vf" name a vertical-feature gate in the neck. On a side view, the edges that separate stacked crates are nearly horizontal lips, periodic down the image. The gate averages each feature map across the width, mixes that column with a short vertical convolution, and multiplies the map by a sigmoid of the result. Rows that contain a lip can be kept; flat rows can be suppressed. The gate does not know the crate size in metres, it does not name a color, and it does not count. It only makes the lips easier for the mask head to see.

The centers passed downstream are the first moment of each mask, mapped into metres on the front face by a homography measured once at installation.

## How the thesis is organized

The document has three chapters, plus front matter and a bibliography of 102 items.

**Chapter 1** is the survey and the problem statement. It walks through image formation, detection versus semantic versus instance segmentation, the YOLO and YOLACT line, density-based counting, classical and learned color constancy, and occlusion-aware segmentation. It then reviews carton datasets, warehouse segmenters, area-ratio inventory, volumetric counting of stacked objects, and rotated logistics-box detectors. The third section draws the gate, defines the passage count formally, lists requirements, and states what is out of scope. Bottle inspection, lot codes, broken-crate detection, and cartons of unknown size are out of scope. The lattice pitch is an input.

**Chapter 2** derives the two updates. Section 2.1 is the color head. Section 2.2 is the lattice counter. They share the mask and nothing else, on purpose, so a color error cannot create a cell and a filled hole cannot change a prototype.

**Chapter 3** specifies the gate system and then estimates the two modules. The estimate is a Monte Carlo instrument (`simulate_estimation.py` in `main`), seeded, writing `estimation_results.json`. It does not read a photograph. The thesis says this in the abstract, again beside the tables, and again in the conclusion. The short papers repeat it.

## Innovation 1 — the chromatic prototype head

The short paper in `paper1` is this module on its own. The full derivation is Chapter 2, Section 2.1 of the thesis.

### The decision

For each mask, output one of six color classes and a margin between the best score and the second best. Do not output a recolored image, and do not output one illuminant for the whole frame. The mask is the smallest region on which "this crate has one body color" is even approximately true. A box contains gaps, forks, and the crate behind. The whole frame mixes every crate with the floor.

### Why ordinary color constancy destroys the class

The working image model is diagonal: observed color equals illuminant times reflectance, times a shade. On a perfectly uniform face, a Minkowski norm (gray-world at order 1, shades-of-gray at order 6) estimates the product of illuminant and reflectance, not the illuminant alone. Dividing by that estimate pushes the face to gray. The lamp is gone and so is the plastic. That is a theorem in the thesis, not an implementation bug. It is why the experiments show gray-world and shades-of-gray stuck near one-in-six accuracy no matter how mild the lamp is.

The head therefore refuses to classify the corrected color by itself. It builds an eight-number feature:

- three numbers: chromaticity of the corrected face (informative mainly when the mask is dirty);
- two numbers: a simple opponent pair, red-minus-green and yellow-versus-blue;
- three numbers: chromaticity of the illuminant estimate itself.

On a clean face the first five numbers collapse toward zero and the last three still point at "lamp times plastic." A linear classifier is allowed to read that direction. Training sees many random lamps, so the weights can learn which coordinates survive.

### Training distribution inside the simulator

Each synthetic face picks a class, multiplies by a random positive gain whose strength is the "severity," applies a shadow and a smooth shade ramp, and with probability one quarter pastes a strip of a different crate color along the edge. That strip is important. Without it the classifier could ignore the corrected color and read the class from the illuminant feature alone. With it, the feature has to cope with a messy boundary, which is what a real mask error looks like. Six thousand faces train a linear softmax for five hundred steps. Test faces are fresh draws at fixed severities 0.05, 0.15, 0.30, 0.45, and 0.60. Severity 0.05 is a couple of degrees from white. Severity 0.30 is on the order of fourteen degrees. Severity 0.60 is a real lamp change, not a tint.

### What the color table means

At severity 0.05, raw nearest-prototype matching is best (about 98.6 percent) and the new head is just behind (about 97.2 percent). Correcting a nearly white lamp throws away a clean signal. From severity 0.15 upward the head leads, and the lead grows only modestly: about 83 percent versus 78 percent at 0.30, and about 61 percent versus 53 percent at 0.60. Classical corrections stay between roughly 17 and 22 percent everywhere, which is chance.

The harsh confusion matrix is the result that a mean would hide. Amber recall is about 0.47 and yellow about 0.41, and they swap with each other. Red and blue hold. White's mistakes spread across every chromatic class. The thesis therefore does not say "the color problem is solved." It says the head is the better of the two learned-versus-raw options once the lamp is not mild, and that amber and yellow must be allowed to abstain.

A second run drops coordinates. Corrected chromaticity alone is weak. The illuminant block alone is strong in mild light and collapses when the lamp is harsh. Chromaticity plus illuminant matches the full eight-dimensional feature. The opponent pair adds almost nothing under the weight penalty. It is kept because it is cheap and because amber versus yellow is an opponent direction.

### What is specified but not trained

A deployment network would also pool the segmenter's neck features inside the mask and add a 32-dimensional projection. The thesis counts those weights: 9,760 parameters, about 0.036 percent of a medium YOLO segmenter. The simulator does not train that path, because there are no labeled gate images. Reporting the chromatic-path accuracy as if the neck path had been trained would be a misstatement. The thesis keeps the two claims apart.

### How a building would use the margin

The margin is a rank, not a probability. Once a week, known-color pallets set an abstention cut at the tenth percentile of margins seen that week. Uncertain faces are listed for a person and left out of the automatic per-color tally. The geometric total is still booked, because geometry does not depend on color. The camera's automatic white balance is locked. A firmware white balance is exactly the global gain the argument rejected, and it would move whenever the forklift mast entered the frame.

## Innovation 2 — perspective lattice completion

The short paper in `paper2` is this module on its own. The full derivation is Chapter 2, Section 2.2.

### The quantity being estimated

If the front face has occupied cells and the pallet is `d` crates deep, the true count is `d` times the number of occupied front cells, assuming a hole in the front means the ranks behind that hole are empty too. The image supplies noisy centers. A separate reading supplies depth, correct with probability 0.90 and otherwise off by one in the simulator. The counter's job is to decide which front cells are occupied and then multiply.

### The steps

1. Map mask centers into metres on the front plane with the installation homography.
2. Estimate the lattice origin as the median residual of those centers modulo the known pitch. The median ignores a minority of false detections. The truck does not have to stop on a painted mark.
3. Snap each center to an integer cell. Several centers in one cell still count once. Noise of a few centimetres does not cross a 20 cm half-cell; missed detections do.
4. One left-to-right pass fills an empty cell that already has two orthogonal neighbors. One neighbor is the outside of a solid wall and must not grow a rind. Two neighbors is the corner of a hole. The same rule also fills a true step in the outline, which is the source of a small positive bias.
5. Multiply the number of occupied cells by the depth reading.

Nothing in those steps is learned. The only geometric parameters are the nominal width and height.

### Why the obvious baselines lose

- **Count the masks and stop.** This reports the front, not the pallet. Mean absolute error is about 13.6 crates, and the bias is equally negative. The error is almost a pure under-count.
- **Count the masks and multiply by depth.** Error falls to about 4.9 crates, bias about −4.3. Depth is the largest single correction in the whole estimate. Any discussion that skips this comparison will give the lattice credit that belongs to the integer depth.
- **Area ratio.** Each detection contributes a noisy visible fraction whose mean is 0.86, the fractions are summed and rounded, then multiplied by depth. Because the fraction sits below one, the counter is short even when the detector was only a little short. Error is about 7.3 crates, worse than raw masks times depth. That does not refute area-ratio papers on irregular cargo. It says a fraction is the wrong summary of a missed crate whose size is already known.
- **Span.** Take the robust width and height of the point cloud, divide by the pitch, and fill that rectangle solid. This already uses the pitch. Error is about 3.3 crates, bias about +1.6. It is the serious baseline.
- **PLC on one frame.** Error about 3.0 crates, bias about +0.83, exact on about 34 percent of trials. The gain over the span is a few tenths of a crate. The bias drop is the real signature of the neighbor test: both methods invent cells, and the test invents fewer than a solid rectangle.
- **PLC on a union of five frames, keeping any cell that appeared once.** Error about 5.8 crates, bias about +5.3. This is a nearly pure over-count. False cells that never coexist in one frame become neighbors in the union, and the fill rule then adds more. The system specification therefore runs the lattice once, on the sharpest frame. A second frame is not useless in principle. A vote of "one" is the thing the estimate rejected.

The same ordering holds at every depth. At depth one, where there is no hidden rank, even the plain mask count is competitive. At depth three it is an inventory error of a different kind. PLC remains the best row in each column, and the five-frame union remains worse in each column.

### A stack you can count by hand

One simulated load is four rows by three columns, but the top row has only the right-hand crate. Ten front cells, depth two, truth 20. The first frame sees nine of the ten. The fill rule puts the missed cell back and also fills the empty cell in the middle of the top step, because that cell has a neighbor below and a neighbor to the right. The corner of the top row is visited first and stays empty. Eleven cells times depth two is 22. The span fills the whole top row and books 24. The five-frame union books 28. The example is there so the positive bias is a picture, not only a column in a table.

### What the operator is supposed to see

The total alone would hide the difference between a stack that was seen and a stack that was inferred. The record therefore includes:

- the total and the per-color counts;
- the depth reading, and whether it was a default rather than a pallet-build record;
- how many cells were filled rather than detected;
- the origin residual in centimetres;
- the weakest color margin on the chosen frame.

An exact total with several filled cells can be a cancellation: one real miss left unrestored, one phantom invented. The exact-match rate must not be briefed as "one passage in three is clean." The mean absolute error of about three crates on a load of about twenty-three, and the bias of under one crate in the long direction, are the numbers that match the arithmetic.

Over a hypothetical shift of forty pallets, the bias adds to roughly thirty crates long, while the sum of absolute errors is on the order of a hundred crates. A daily reconciliation against production sees something like the bias, not the absolute error. Turning the fill rule off and going back to masks times depth would reconcile short by roughly four crates per pallet, which is both larger and the more dangerous sign.

## How the two innovations meet

They share the mask. The mask's pixels go to the color feature. The mask's center goes to the lattice. After that split, a wrong color and a wrong count are different events.

A completed cell that no detection owns inherits the majority color of its orthogonal neighbors, with margin as the tie break. That inheritance rule is specified and not estimated, because the simulator draws color trials and geometry trials separately. A loose bound says the per-color error is at most the total-count error plus the crates whose type was swapped. The point of the bound is that the errors add. They are not assumed to cancel. The display keeps them on separate lines so a correct total cannot hide an amber/yellow swap.

A building that controls its lamps and not its stacking would keep the lattice and could fall back to raw color. A building with irregular cartons and stable lamps would do the opposite and should use an area ratio instead of a pitch it does not have. The brewery passage, as defined, enables both modules, which is why both are in the updated model.

## The system around them

The camera is on the building, not on the truck: about 2.5 m from the face plane, optical axis horizontal at about 1.4 m height, exposure at or under 2 ms, frame rate about 15 Hz, white balance locked. At that standoff a pixel on the face is a couple of millimetres. The lattice is not limited by sensor resolution. It is limited by which crates are detected and by whether the homography scale is right.

A motion trigger fills a short buffer. The segmenter runs on each kept frame. The color head scores every mask. When the truck has passed, the frame with the strongest median color margin is the one frame sent to the lattice, together with the depth reading. The preferred depth source is the pallet build record. A shift default is allowed only if it is labeled as a default. There is deliberately no code path from mask area to depth.

The segmenter is the only stage that can miss the timing budget. The color head and the lattice are arithmetic on a few dozen objects. If a deployment is late, the specified downgrade is a smaller backbone, not the removal of the lattice. Removing the lattice reintroduces the factor-of-depth error, which is larger than the accuracy gap between a small and a medium backbone.

Three calibrations are required. A planar target fixes the camera matrix and the homography, in metres. A tape check confirms the pitch, with a second pitch allowed as a second hypothesis chosen by the smaller origin residual. The weekly known-color pallets set the abstention threshold and do not refit the color weights unless the abstention rate has been bad for a week.

The system refuses to book a total when the buffer is empty, when the origin residual exceeds half a cell, or when depth is missing and no default exists. It does not refuse merely because a color margin is low. Images of a refusal are kept. A booked passage keeps the chosen frame, the masks, the centers, both cell sets (before and after filling), and the displayed fields. Unused frames are dropped so that a later "improvement" cannot quietly union them.

## How to read the estimate without over-claiming it

The thesis supports these statements, and not a field accuracy.

- Per-mask gray-world and shades-of-gray are the wrong correction on a nearly uniform crate. That is the proposition plus the flat curves near chance.
- The linear head degrades more slowly than raw color as the lamp strengthens, and the remaining errors concentrate on amber, yellow, and white. The crossover at very mild light is part of the result.
- Multiplying a front count by a depth reading is the dominant counting gain inside this model.
- Given that reading, the lattice is a small improvement on the span and a clear improvement on the area ratio used here.
- Unioning five frames at vote one increases positive bias. One frame is the recommendation because of that row, not in spite of it.

It does not support a claim about cardboard of unknown size, a different depth sensor, a trained neck path, or a measured seam-merge rate. The center noise in the counting model is a stand-in for many ways a center can move, including a merged seam. If real frames merge seams so badly that one center lands between two crates, the lattice will book one cell and the filled-cell counter will not rise, because the second crate never arrived as a detection. The vertical gate is the specified mitigation, and it is untested. A field trial that labels seams is the trial that can confirm or retire that part of the architecture.

A trial that could replace the tables would record, at one gate, the images, the depth reading, a person-confirmed count per color, and enough lamp variety that "severity" means something measured. It would keep the one-frame lattice and the five-frame union as separate rows. It would publish the confusion matrix. Until then, Chapter 3 is a specification constrained by an explicit generator: one frame, not five; a chromatic feature, not a divided-out color; depth as an input, not as a fraction of mask area.

## Where the files are

- `main/thesis.pdf` is the thesis. `main/thesis.txt` is its extracted text. `main/simulate_estimation.py` is the instrument that produced the tables.
- `paper1` is the short paper on the color head.
- `paper2` is the short paper on the lattice counter.
- This document is the guide to both.

## A compact map from requirement to evidence

| Need | Where it is handled | What would overturn it |
| --- | --- | --- |
| A count per color and a total | Color head plus lattice | A joint trial where per-color error behaves unlike the separate estimates |
| One answer per passage | Lattice on one frame; the five-frame row is the rejected alternative | A vote rule that beats one frame on real passages |
| Side view only | Depth is an external reading | A side-image depth estimate better than the pallet record |
| Lamps not controlled | Feature design and the color table; firmware white balance forbidden | A dock where raw color wins at every real severity |
| Adjacent crates of one color stay two cells | Vertical gate in the baseline; not numerically estimated | Labeled frames where merged seams dominate |
| Must run while the truck passes | Head is a few thousand weights; lattice is a small loop | A profiled segmenter that misses the frame time after the head is added |
| Errors must keep their sign | Bias is reported next to absolute error | A site where over-counts and under-counts have the same cost and the same cause |
