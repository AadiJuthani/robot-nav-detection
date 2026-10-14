# Extra credit: disparity to depth (epipolar geometry)

The single-camera distance in this project needs a guess about how tall each
object is. A stereo pair doesn't, and the reason comes down to the same similar
triangles idea, just done twice.

## Setup

Two identical pinhole cameras, focal length `f` (pixels), side by side and
pointing the same way. Their optical centers are `B` meters apart (the baseline).
After rectification the image planes are coplanar and the rows line up, so the
epipolar lines are horizontal: a point in the left image can only show up on
the same row in the right image. That's what turns the matching problem into a
1D search along a row.

Put the left camera at the origin and the right one at `(B, 0, 0)`. A point in
the world sits at `P = (X, Y, Z)` in left-camera coordinates.

```
                 P (X, Y, Z)
                 /\
                /  \
               /    \            Z = depth
              /      \
   ----------x_L------x_R------  image planes (distance f in front)
            /          \
     O_L  ●────── B ─────●  O_R
```

## Derivation

Projecting `P` into each camera (pinhole model, x measured from each image's
principal point):

- left camera:  `x_L = f * X / Z`
- right camera: in its own frame the point is at `X - B`, so `x_R = f * (X - B) / Z`

Subtract the two:

```
x_L - x_R = f * X / Z - f * (X - B) / Z = f * B / Z
```

The left-minus-right shift is the disparity `d = x_L - x_R`, so

```
d = f * B / Z        =>        Z = f * B / d
```

Depth is inversely proportional to disparity. Something close shifts a lot
between the two images, something far away barely moves, and something at
infinity has zero disparity.

## What it means in practice

- No assumption about object size anymore. The baseline `B` is known exactly
  because we built the rig, so it plays the role `H_real` plays in the mono
  formula (`Z = f * H_real / h_px`).
- Error grows with distance. Differentiate `Z = fB/d`:
  `dZ/dd = -fB/d^2 = -Z^2/(fB)`, so a fixed 1-pixel matching error causes an
  error that grows with the square of the depth. A wider baseline or longer
  focal length helps.
- For our pipeline: run the detector on the left image, take the stereo
  disparity inside each box (median, to ignore background pixels leaking into
  the box), and `Z = f * B / d_median` replaces the height-based guess.

Same idea as `navdet/distance.py`, just with a baseline we measured instead of
an object height we guessed.
