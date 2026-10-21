# Method

## 1. The FloWM memory, read as a phase-space field

FloWM keeps a hidden state $h_t(x,\nu)$ indexed by a position $x$ on a grid
around the agent and by a velocity $\nu$ from a finite set $V$. At every step
the content of channel $\nu$ is rolled by $\nu$ pixels and the whole state is
rolled by the agent's own motion. A convolution and a nonlinearity then update
the state, and the current observation is written into all channels at the
viewport. The decoder reads the maximum over $\nu$ inside the viewport.

We read this state as a field on phase space, with $x$ the position and $\nu$
the velocity. The per-step roll is the free transport of such a field, and the
velocity set $V$ is a quadrature of velocity space. The same structure appears
in lattice Boltzmann methods, where a distribution over discrete velocities is
streamed by one cell per step and then mixed locally. The two-dimensional
Galilean group acts on this field by shifting $x$ (translations) and by
shifting $\nu$ (boosts). FloWM is equivariant to both, but only on the lattice.
A translation by a non-integer number of pixels or a boost to a velocity off
the lattice has no representation.

This reading predicts how FloWM fails. A channel accumulates an object only
when the roll matches the object's motion. A digit moving at 0.7 pixels per
frame is misaligned by 0.3 pixels per frame in the nearest channel, which is
nine pixels after 30 observed frames. No channel stays coherent, and the
memory of the digit is lost. Section 1 of RESULTS.md confirms this with the published FloWM weights.
Section 2 confirms it with an oracle that sets the velocity nodes to the true
velocities and recovers the integer-data accuracy.

## 2. Spectral transport

For a periodic map, a shift by a real vector $d$ is a multiplication of the
Fourier coefficients by $e^{-2\pi i\,k\cdot d}$. We replace both rolls by this
operation. The node velocity $\nu$, the agent motion $a_t$ and the time step
$\Delta t$ then take real values. The shift is exact for band-limited maps.
$T$ consecutive shifts compose into a single phase, so no error accumulates.
On a 50x50 map the cost is two FFTs per channel per step. Everything else in
the cell is unchanged, so on integer data this model and FloWM reach the same
accuracy. We call it the spectral model below.

## 3. Adaptive velocity nodes

Spectral transport alone does not help with off-lattice object velocities,
because the node velocities are still integers. We therefore let each node
track the velocity of the content it holds. The node velocity of sequence $b$
is $\nu_i(t)=\bar\nu_i+\rho_i(t)$, with $\bar\nu_i$ a lattice point and
$\rho_i$ a bounded offset. Adjacent cells overlap, so a velocity near a cell
boundary can be held by either neighbour.

The offset is estimated from the observations, without learned parameters,
in three steps.

1. Each node carries an observation imprint, a low-dimensional projection of
   the encoder output written at the viewport and transported exactly like the
   memory. The imprint decays by a factor 0.9 per step. Registration between
   the transported imprint and the new encoder output compares two copies of
   the same signal, so brightness constancy holds.
2. Inside the viewport, a Lucas-Kanade step gives a sub-pixel displacement per
   pixel. These are averaged over the channel with weights that favour pixels
   where the imprint and the new observation agree. The average over an
   object's support reduces the noise of the per-pixel estimates by about an
   order of magnitude.
3. The offset is updated with an evidence-weighted running mean, with the gain
   $w_t/\sum_{s\le t}w_s$. The first strong observation moves the node almost
   all the way, later observations refine it. A single learned scalar
   multiplies the gain.

The transport remains the exact spectral shift of Section 2 with per-sequence
node velocities. No warping or resampling is involved, so nothing blurs and
nothing can accumulate.

Under a boost of the whole scene by a real vector $u$, every node velocity
shifts by $u$ and the lattice part of the shift permutes the channels. The
model is therefore equivariant to continuous boosts up to the quality of the
registration, which we measure directly (RESULTS.md, Section 5).

## 4. What we tried and dropped

- A per-pixel residual velocity field, moved by bilinear warping. Training
  suppressed the learned residual estimator, whose gain fell to 0.15. A
  bilinear warp of 0.5 pixel per step also blurs a blob to half its peak in 30
  steps. With an explicit Lucas-Kanade estimator the model matched the spectral
  model and did not beat it, because a single-step per-pixel estimate has an error as large
  as the residual it estimates.
- A diffusion-free version of the same idea that moves content by whole
  pixels once a sub-pixel accumulator passes 0.5. It trains unstably, with
  loss spikes within the first 3k steps under every variant we tried.
- A denser velocity lattice (81 nodes, spacing 0.5). It helps a little (0.062
  against 0.079 at 10k steps) at three times the cost.
- Learned static node velocities. Superseded by the adaptive nodes.

## 5. Related ideas

The construction is close to a k-means in velocity space. Each node is
attracted to the velocity of the coherent content in its channel. The registration step
is the local half of loop closure in SLAM, applied in the feature space of the
world model. We do not yet use the global half, which would synchronise
estimates across time and across nodes.
