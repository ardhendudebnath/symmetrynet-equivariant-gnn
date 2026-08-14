# Building symmetry into a neural network instead of teaching it

*A write-up of SymmetryNet: an E(3)-equivariant graph neural network for molecular
property prediction, with the representation theory implemented from scratch.*

---

## The problem, and why it is not really a machine learning problem

Suppose you want to predict a molecule's HOMO-LUMO gap — the energy difference between
its highest occupied and lowest unoccupied molecular orbital, which largely determines
how it absorbs light and how it reacts. The input is a list of atoms and their 3D
coordinates. The output is one number in electron-volts.

Here is the awkward part. If I hand you the same molecule twice, but the second time I
have rotated it by 40 degrees, every single input number changes. The output must not.
Not "should mostly not" — must not, exactly, because the orientation was never a property
of the molecule in the first place. It was an artifact of how someone chose to write down
the coordinates.

A standard neural network has no idea about any of this. Feed it raw coordinates and it
will happily learn a function whose value drifts as you spin the input. The usual
mitigation is data augmentation: show it many rotated copies and hope it infers the
symmetry. This works tolerably, and it is deeply unsatisfying. You are spending model
capacity, training time, and data to teach the network something you already knew with
certainty before training began. And you still only get approximate invariance, valid
near your data and nowhere else.

The alternative is to build the symmetry into the architecture so that a
symmetry-violating function is *not expressible*. Then invariance holds exactly, for every
setting of the weights, including random initialisation — which is a strange and pleasant
thing to be able to say about a neural network. The picture at the top of the README shows
exactly that: untrained models, one flat to 10⁻¹⁵ eV and one swinging by 10⁻² eV.

## Invariant is not enough

The obvious fix is to feed the network only quantities that do not change under rotation.
Interatomic distances are the natural choice, and that is precisely what SchNet-style
models do. It works, it is simple, and it is genuinely rotation-invariant.

It also throws information away. A set of distances from one atom to its neighbours does
not determine the bond *angles* between them. Two chemically different local geometries
can produce identical distance inputs. Deeper message passing recovers some of this
indirectly — angle information leaks in through shared neighbours over several hops — but
it is being reconstructed rather than represented.

The distinction that unlocks the better architecture is between *invariance* and
*equivariance*. An invariant quantity does not change under rotation. An equivariant
quantity changes in a **known, prescribed way**. A force vector is equivariant: rotate the
molecule and the force rotates with it. That is not a loss of information, it is
information carried in a structured form.

So the design becomes: keep everything equivariant through the depth of the network, so
directional information survives, and collapse to invariance only in the final readout.

## What representation theory contributes

To say "this feature transforms in a known way" you need to say precisely which way. That
is what representation theory provides.

A *representation* assigns to each rotation R a matrix D(R) acting on your feature space,
consistent with composition. Representations decompose into *irreducible* pieces, indexed
for SO(3) by a non-negative integer ℓ, with the degree-ℓ piece having dimension 2ℓ+1.
ℓ=0 is a scalar (invariant — energy, charge). ℓ=1 is a vector (force, dipole). ℓ=2 is a
symmetric traceless rank-2 tensor (quadrupole). Every feature in the network carries an ℓ
label, and that label is a checkable commitment about how it must behave.

Three ingredients then build a layer:

**Spherical harmonics** turn a direction into a degree-ℓ feature. They are not an
arbitrary choice: the Laplacian commutes with rotations, so its eigenspaces are
rotation-invariant subspaces, and those eigenspaces are exactly the spherical harmonics.
Equivalently, degree-ℓ harmonic polynomials restricted to the sphere form a
(2ℓ+1)-dimensional irreducible space — the 2ℓ+1 arising because harmonicity strips out the
rotation-invariant factors of r² that make the raw polynomial space reducible. The
[full derivation is here](math_foundations.md).

**Clebsch-Gordan tensor products** combine features. This is the part with no analogue in
ordinary deep learning. You cannot simply matrix-multiply an ℓ=1 feature into an ℓ=2
feature; that would destroy the transformation law. Instead you take the outer product and
decompose it back into irreps. The Clebsch-Gordan coefficients are the change of basis
that does this, and they satisfy the intertwining identity
(D^ℓ¹ ⊗ D^ℓ²) C = C D^ℓ — transform-then-combine equals combine-then-transform. This is
the equivariant analogue of a weight matrix, and it is the only operation in the network
that can *create* new angular information.

It is more familiar than it sounds. For two vectors, 1 ⊗ 1 = 0 ⊕ 1 ⊕ 2 with dimensions
9 = 1 + 3 + 5, and the three pieces are the dot product, the cross product, and the
symmetric traceless outer product. Clebsch-Gordan coefficients are the systematic
generalisation of "the ways two vectors can be multiplied".

**Gated nonlinearities** provide the activation. Applying ReLU componentwise to an ℓ=1
feature is not equivariant — clipping components individually depends on the frame you
chose. But multiplying a degree-ℓ feature by an *invariant* scalar commutes with D^ℓ
trivially. So gates are computed from the ℓ=0 channels and used to scale everything else.

Distance enters through a small MLP whose output weights the tensor product. A distance is
invariant, so using it to parameterise an equivariant operation is safe — and this is what
lets the interaction depend on geometry without breaking symmetry.

## Implementing it rather than importing it

The production model uses `e3nn`. But `e3nn` hides exactly the machinery that matters, so
the project first implements all of it by hand, and keeps that code in the repository:
real spherical harmonics in closed form, Clebsch-Gordan coefficients from the Racah
formula transformed into the real basis, and Wigner-D matrices built recursively from
those coefficients.

The Wigner-D construction is worth a note on test design. It would be circular to fit D^ℓ
from spherical harmonics and then use it to verify those harmonics. Instead D^ℓ is built
algebraically — D⁰ = 1, D¹ is the rotation matrix reordered, and D^ℓ = Cᵀ(D^{ℓ-1} ⊗ D¹)C
— so no spherical harmonic appears anywhere in it. Checking
Y(Rr) = D(R)Y(r) is then a real cross-check of two independent derivations. It passes at
10⁻¹⁶, as does agreement with `e3nn`.

## Two things I only learned by measuring

**`e3nn` bakes its Wigner-3j constants at construction dtype.** Build a model under the
default float32, call `.double()`, and the constants compiled into each `TensorProduct`
stay float32 — `.double()` widens already-truncated numbers. The equivariance residual
jumps from 1.3e-15 to 2.1e-09 for ℓ_max=2 and 8.0e-09 for ℓ_max=3, while ℓ_max=1 is
unaffected. That pattern reads exactly like "my higher-degree paths are buggy", and it
would be very easy to spend a day chasing a bug that does not exist. Constructing under
float64 fixes it completely.

**The natural tensor product does not fit in memory.** `FullyConnectedTensorProduct`
learns a weight per (input channel × output channel) per path, *per edge*. At
multiplicity 64 that is ~65,000 weights per edge, and a batch of 96 QM9 molecules has
~27,000 edges — a 7.1 GiB tensor for one forward pass, which is how I found out. The fix
is `uvu` connection mode, where weights are per-channel rather than per channel-pair,
followed by an `o3.Linear` to restore cross-channel mixing at shared-weight cost. About
100× less memory for the same expressiveness. This is what NequIP does; the reason is not
obvious until you hit the wall.

**Exact equivariance says nothing about good conditioning.** This one cost the most time
and taught the most. The first full comparison had the equivariant model *losing* to the
distance-only baseline — 137 meV against 96 meV at the same epoch. Everything about the
symmetry was fine: equivariance held at 1e-15, training was stable, gradients were
finite, no warnings. The model just underfit, with training error tracking validation
error the whole way down.

Instrumenting activation magnitudes found two scale errors stacked on top of each other.
The Bessel radial basis takes values of order 0.3 rather than 1, and the MLP consuming it
is initialised for unit-variance inputs, so the tensor-product weights it emitted had
standard deviation 0.12 — while e3nn's tensor product normalises on the assumption that
those weights have unit variance. The message pathway was therefore attenuated roughly
eightfold, and progressively worse with depth: the ratio of message magnitude to skip
magnitude fell from 0.27 at the first layer to 0.115 at the fourth.

Correcting the basis scale immediately exposed the opposite error, which the first one had
been hiding. Dividing the neighbour sum by √N assumes the incoming messages are
independent; they are not, since they all involve the same central atom and its local
chemistry, so the sum grows like N and leaves a systematic gain of about √N ≈ 4 per
layer. Activations compounded 1.5 → 4.7 → 11.7 → 47.9 across four layers.

Equivariant `BatchNorm` fixes the residual stream properly. It rescales each irrep by the
*norm* of its components, which is a rotation-invariant quantity, so the symmetry
guarantee is untouched — the equivariance suite still passes at 1e-15 — while the layer
standard deviations flatten to 1.00, 1.00, 1.00, 1.00.

The general lesson is worth stating plainly, because the symmetry framing actively
encourages the mistake: proving a network is equivariant proves nothing about whether it
is trainable. The two properties are independent, and the failure mode of the second is
silent. A model that is provably correct and quietly underfitting looks exactly like a
model that simply needs more data.

A smaller point in the same spirit: an equivariance error is only meaningful relative to
the model's own reproducibility floor. CUDA's `scatter_add` uses atomics and
floating-point addition is not associative, so the same input twice gives ~2e-15 of
variation. Any tolerance has to sit above that, and the test suite measures it rather than
assuming.

## Results

The experimental design matters as much as the numbers. Both models share the same
training loop, optimiser, schedule, radial basis, cutoff, aggregation normalisation,
readout structure and epoch budget. Only `--model` changes. Splits are fixed and seeded;
target standardisation is fitted on the training split alone. The point is that any
difference is attributable to the angular machinery rather than to tuning.

Three things were measured. One worked perfectly, one worked exactly as the theory
predicts, and one did not work at all.

**Equivariance holds exactly.** Relative deviation under 200 random rotations is 1.3e-15
for the equivariant model against 1.2e-2 for the raw-coordinate control — thirteen orders
of magnitude, on untrained weights. This was never really in doubt once the tests passed,
but it is the property the whole architecture exists to provide.

**Angular information is worth something, and the ablation isolates how much.** ℓ_max=0
is a fully equivariant architecture in which the selection rule permits only 0 ⊗ 0 → 0,
so no angular path exists; ℓ_max=1 adds vectors, ℓ_max=2 adds rank-2 tensors. Test MAE
falls monotonically 113.6 → 89.6 → 78.6 meV, a 31% reduction, with every other
architectural variable held fixed. That is a clean confirmation of the central
representational claim.

**Equivariance wins decisively — but only after changing which equivariant architecture is
used.** At a matched 200-epoch budget:

| model | equivariance via | params | test MAE | time |
|---|---|---|---|---|
| PaiNN (`l<=1`) | vector algebra | 576k | **43.43 meV** | 81 min |
| distance-only baseline | discards direction | 277k | 56.03 meV | 37 min |
| TFN (`l_max=2`) | Clebsch-Gordan | 573k | 59.43 meV | 277 min |

Three claims stack here, and the order they arrived in is the point of the whole project.

The Tensor Field Network — the architecture the brief specifies, and the one whose
mathematics this repository implements from scratch — is exactly equivariant and *loses*
to a distance-only baseline that reproduces published SchNet. So equivariance by itself
buys nothing. That was the state of the project for most of its life, and reporting it was
uncomfortable but correct.

PaiNN then beats that baseline by 22.5% and the TFN by 26.9%, with the same parameter
count as the TFN and under a third of its compute. The symmetry group is identical, the
target is identical, the training loop is identical. Only the mechanism differs.

And the mechanism that wins has *less* angular resolution than the one that loses. The
ablation shows `l_max=2` beating `l_max=1` by 12% inside the TFN family — a clean
confirmation of the representational theory. Yet PaiNN, structurally incapable of
representing anything above `l=1`, beats TFN at `l_max=2` by 27%. "More angular
resolution is better" holds within an architecture and does not transfer across
architectures. The binding constraint was never angular resolution. It was that
Clebsch-Gordan tensor products cost 5x more per gradient step, and that compute buys more
when spent on width and depth.

This is the most useful thing the project produced, and it is not what I expected going
in. The interesting question about equivariant networks is not *whether* to impose the
symmetry — it is what you are willing to pay to impose it, because the cheapest sufficient
construction beat the most general one.

**Neither equivariant model is more data-efficient, and this is the result I trust most,
because it survived the architecture change that overturned everything else.** Every point
is trained to convergence rather than to a fixed epoch count:

| training molecules | baseline | PaiNN | TFN | PaiNN ratio | TFN ratio |
|---|---|---|---|---|---|
| 11,000 | 142.03 | **122.28** | 182.66 | 1.16 | 0.78 |
| 27,500 | 98.06 | **84.12** | 117.31 | 1.17 | 0.84 |
| 55,000 | 71.31 | **58.09** | 80.47 | 1.23 | 0.89 |
| 110,000 | 56.03 | **43.43** | 59.43 | 1.29 | 0.94 |

PaiNN beats the baseline at every size, which is a genuine architecture-level win. But the
prediction under test was about the *shape* of these curves: symmetry should matter most
when data is scarce, so the equivariant advantage should be largest on the left and shrink
to the right. Both curves slope the other way. PaiNN wins everywhere but by less when data
is scarce; the TFN loses everywhere and by more when data is scarce.

Two architectures, opposite outcomes, identical trend. On this target the benefit of a
built-in symmetry grows with data rather than shrinking.

**Replicated over three seeds.** A result that contradicts a widely-repeated claim should
not rest on one run, so the whole grid was repeated with three seeds, each driving a
different train/val/test split as well as a different initialisation:

| training molecules | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| 11,000 | 1.167 | 1.119 | 1.081 | 1.122 ± 0.043 |
| 27,500 | 1.177 | 1.149 | 1.140 | 1.155 ± 0.019 |
| 55,000 | 1.228 | 1.235 | 1.196 | 1.220 ± 0.021 |
| 110,000 | 1.290 | 1.286 | 1.240 | 1.272 ± 0.028 |

The mean is monotone across all four sizes, the direction is unanimous in 3/3 seeds, and
the gap between the extremes (+0.150) is roughly three times the pooled standard deviation
(0.051). Within a seed the two models see byte-identical data, so each ratio is a paired
comparison; across seeds the split changes, so surviving all three means the effect is not
an artifact of one partition.

I deliberately did not compute a p-value. With three seeds a t-test would manufacture
precision that the sample size cannot support. Unanimity of direction plus non-overlapping
spread at the extremes is the strongest honest statement available, and the analysis script
is written to report exactly that — including a branch that prints "seeds disagree; no
directional claim is supported" had the replication failed.

One protocol detail worth recording: seed 0's two smallest baseline runs were originally
trained with a shorter early-stopping patience than everything else. Early stopping only
truncates *after* the best checkpoint, so the effect is second-order, but it would have
biased the baseline slightly worse at small data — inflating the equivariant ratio exactly
where the finding claims it is smallest, i.e. in the direction of the conclusion. Both were
retrained under the common protocol before any of this was reported.

This also falsifies the explanation I offered when only the TFN had been measured. I had
attributed its backwards trend to optimisation difficulty: the tensor-product model needed
roughly four times the baseline's schedule to converge, and I argued that cost swamped the
symmetry benefit in the low-data regime. PaiNN is a direct test of that claim, since it
keeps the symmetry and removes most of the optimisation burden. The explanation accounts
for the *level* — PaiNN sits above 1.0 where the TFN sits below — and fails for the
*slope*, which does not change at all. Whatever produces the trend is not optimisation
difficulty.

**And then fitting the scaling law explained it.** Error curves here follow a power law
tightly, `MAE(N) ≈ A·N^(−b)`, with R² at or above 0.997 on every fit across both models and
all three seeds — so the exponent is a real quantity, not a line forced through scatter.

| model | exponent `b` |
|---|---|
| distance-only baseline | 0.406 ± 0.006 |
| equivariant PaiNN | 0.462 ± 0.006 |
| difference | +0.056 ± 0.010, steeper in 3/3 seeds (≈5 sd) |

That closes the loop algebraically. The ratio of two power laws is
`(A_base/A_equi)·N^(b_equi − b_base)`, so it can only increase with `N` if the equivariant
exponent is steeper. It is, by 14% relative. The trend I could not explain was not a
mystery; it was the visible signature of a better exponent.

More importantly, it sharpens what the whole result says. "Data efficiency" as normally
claimed is a statement about the *offset*: you need fewer samples to reach a given error, so
the advantage is largest where data is scarce. That is not what equivariance provides here.
It provides a better *exponent* — the rate at which error falls with added data improves.
Those two mechanisms have opposite signatures in the ratio curve, and this measurement
distinguishes them cleanly.

So the honest conclusion is not that equivariance failed to deliver. It is that
**equivariance improves scaling rather than sample efficiency**, which is a more specific
claim than the one I set out to test and, if anything, a more useful one — an offset
advantage is a fixed discount, while an exponent advantage compounds with every molecule
added.

I had earlier guessed that QM9's molecules might be repetitive enough for a distance-only
model to recover angular structure from data once it had enough. The scaling fit says the
opposite: the baseline's disadvantage *grows* with data. That guess was wrong, and the
measurement was what settled it.

Getting to a curve I trust took two corrections worth recording. The first version used a
fixed epoch budget per fraction, which converges the small-data points while starving the
large-data ones, because a 10% split has a fifth as many gradient steps per epoch. The
second version fixed the budgets but I nearly repeated the mistake in reverse: re-running
the 50% point at 200 epochs would have given it roughly half the updates the full split
needed, since half the data means half the steps per epoch. The correct budget is set by
gradient steps, not epochs — 370 rather than 200. The convergence check is now automated
so this cannot recur silently.

A note on how this conclusion moved. For most of the project the honest summary was
"equivariance does not help here", and I wrote it up that way rather than tuning until it
flipped. That was right at the time and would have been wrong to soften. But it was a
statement about one architecture, and I should have been more careful to say so — the
result generalised much less than the framing implied. The correction came from testing a
second equivariant model, not from adjusting the first.

Two things make me confident this is a real result rather than a broken setup. The
baseline lands at 63.8 meV where published SchNet on this exact target sits at ~63 meV —
reproducing the literature is what certifies the control as a genuine opponent rather than
a strawman. And the equivariant model is not obviously mis-specified: it is exactly
equivariant, well-conditioned after the fix above, and its own ablation behaves precisely
as the representation theory says it should.

The honest reading is that "equivariant" is not by itself a guarantee of winning. TFN is a
2018 architecture, and the models that actually beat SchNet on this target — PaiNN at
45.7 meV, DimeNet++ at 32.6 meV — did so by adding mechanisms beyond raw equivariance.
Symmetry buys you a smaller, physically correct hypothesis space; it does not
automatically buy you accuracy on a scalar target where a strong invariant model already
does well.

The compute budget explains part but not all of the gap. The equivariant model's best
epoch was the *final* epoch at both 50% and 100% data, so it was still improving when the
budget ended and those points understate it. But at 25% and 10% it converged well before
the cap and still lost by 16–29% — and small-data is exactly where the data-efficiency
argument should have been strongest.

## What I would do next

**Find out what actually drives the data-efficiency trend.** Both equivariant models show
the advantage growing with data, and my optimisation-difficulty explanation was falsified
by PaiNN. The cheapest discriminating test is a harder dataset: QM9 molecules are small and
chemically repetitive, so a distance-only model may simply be recovering angular structure
from data once it has enough. A benchmark with more geometric diversity would separate
"symmetry is not worth much here" from "QM9 is too easy to show it".

**Predict forces rather than a scalar.** Forces are ℓ=1, so the output itself is
equivariant rather than invariant, and the advantage should be considerably larger: a
distance-only model has to reconstruct a vector field it cannot directly represent. PaiNN
is well suited to this, since its vector channels already carry exactly the right object.
MD17 is the natural benchmark.

**Test where PaiNN's `l<=1` ceiling actually binds.** The comparison here says PaiNN's
cheaper mechanism beats the TFN's more general one on a scalar target. That should not hold
forever — properties genuinely requiring rank-2 structure, such as polarisability or
quadrupole moments, are where the tensor-product machinery should earn its cost back.
Finding the crossover would say something more useful than either result alone.

Beyond that, the same representation-theoretic ideas apply to 2D image rotations through
`escnn`, and writing up how the machinery differs between continuous 3D rotations and
planar ones would sharpen the underlying intuition.
