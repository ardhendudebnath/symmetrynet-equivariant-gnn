# The representation theory behind an equivariant network

*Phase 0 deliverable. This is the derivation I worked through before writing any model
code, in my own words. The two questions it answers are the ones you have to answer to
implement — rather than merely invoke — an E(3)-equivariant network:*

1. *Why are spherical harmonics the natural basis for the irreducible representations of
   SO(3)?*
2. *What does a Clebsch-Gordan tensor product actually compute?*

Everything derived here is implemented from scratch in
[`src/symmetrynet/scratch/`](../src/symmetrynet/scratch/) and checked numerically in
[`tests/`](../tests/).

---

## 1. Equivariance, and why it is a constraint worth having

A group $G$ acts on inputs and outputs. A function $f$ is **equivariant** when

$$f(g \cdot x) = g \cdot f(x) \qquad \text{for all } g \in G,$$

and **invariant** when the action on the output is trivial, $f(g\cdot x) = f(x)$.

For molecules the relevant group is $E(3)$: rotations, reflections, and translations of
3D space. A molecule's energy, dipole magnitude, or HOMO-LUMO gap does not depend on how
you happened to orient the molecule when you wrote down its coordinates. That is not an
approximation or a statistical regularity — it is exact physics.

There are two ways to get a network to respect this.

**Learn it.** Feed in raw coordinates, augment the training data with random rotations,
and hope the network infers the symmetry. This spends capacity on re-learning something
you already knew, and the result holds only approximately and only near the data.

**Build it in.** Constrain every layer so that it *cannot* represent a symmetry-violating
function. The hypothesis space shrinks to exactly the physically meaningful functions.
Nothing is learned about rotations because nothing needs to be.

The second is what this project does, and the difference is measurable: the equivariant
model's prediction changes by $\sim 10^{-15}$ eV under rotation, the raw-coordinate
control by $\sim 10^{-2}$ eV — thirteen orders of magnitude, with no training involved
in either case.

The subtlety worth stating: we do **not** want every layer to be *invariant*. An
invariant-at-every-layer network throws away directional information immediately and can
never recover it. We want each layer to be *equivariant* — to carry directional
information in a form that transforms predictably — and collapse to invariance only at
the very last step. That is the entire architectural idea, and it is why we need to know
*how* things rotate, which is what representation theory tells us.

---

## 2. Representations and irreducibility

A **representation** of $G$ on a vector space $V$ is a map $D: G \to GL(V)$ satisfying
$D(g_1 g_2) = D(g_1) D(g_2)$. It is the concrete answer to "if I rotate the world, what
matrix hits my feature vector?"

A representation is **reducible** if $V$ has a proper subspace preserved by every
$D(g)$ — you can split your features into independent pieces that never mix. Otherwise
it is **irreducible** (an *irrep*).

Irreps matter because of the standard decomposition theorem: every finite-dimensional
representation of a compact group is a direct sum of irreps. Choosing a basis adapted to
that decomposition makes $D(g)$ block-diagonal. Instead of tracking how an arbitrary
144-dimensional feature vector transforms, you track a handful of small independent
blocks with known behaviour.

For SO(3) the irreps are indexed by a non-negative integer $\ell$, and the irrep of
degree $\ell$ has dimension $2\ell + 1$:

| $\ell$ | dim | transforms like | example |
|--------|-----|-----------------|---------|
| 0 | 1 | scalar (invariant) | energy, mass, charge |
| 1 | 3 | vector | force, dipole moment, velocity |
| 2 | 5 | symmetric traceless rank-2 tensor | quadrupole moment, strain |

So "this feature is $\ell = 1$" is a precise, checkable claim: under a rotation $R$ it
must be hit by a specific $3\times3$ matrix. Not "it is vector-ish".

---

## 3. Why spherical harmonics

Here is the question sharply: we need a rotation-equivariant way to encode a direction
$\hat r$. Which functions on the sphere should we use?

### 3.1 The action on functions, and the Laplacian

SO(3) acts on functions on the sphere by $(R \cdot f)(\hat r) = f(R^{-1} \hat r)$. This
makes $L^2(S^2)$ an (infinite-dimensional) representation. Decomposing it into irreps
tells us the natural basis.

The key observation is that the **Laplacian commutes with rotations**. Rotations are
isometries; $\Delta$ is built only from the metric, so $\Delta(R \cdot f) = R \cdot (\Delta f)$.

That has an immediate consequence. If $\Delta f = \lambda f$, then
$\Delta (R \cdot f) = R \cdot \Delta f = \lambda (R\cdot f)$, so $R \cdot f$ has the same
eigenvalue. **Each eigenspace of the Laplacian is a rotation-invariant subspace** — i.e. a
representation in its own right. The eigenspaces of the spherical Laplacian are precisely
the spaces spanned by the degree-$\ell$ spherical harmonics.

So spherical harmonics are not a convenient choice among many. They are what you get by
decomposing functions on the sphere according to a rotation-commuting operator. This is
the same reason Fourier modes are the natural basis for translation-equivariant problems:
diagonalise an operator that commutes with the group.

### 3.2 The polynomial derivation, and where $2\ell+1$ comes from

The eigenvalue argument shows the eigenspaces are representations, but not that they are
*irreducible*, and it does not obviously give the dimension. The polynomial route gives
both, and it is the one I find most convincing.

Let $P_\ell$ be the homogeneous polynomials of degree $\ell$ in $(x,y,z)$. Rotations
preserve degree, so $P_\ell$ is a representation. Its dimension is the number of
monomials $x^a y^b z^c$ with $a+b+c=\ell$:

$$\dim P_\ell = \binom{\ell+2}{2} = \frac{(\ell+1)(\ell+2)}{2}.$$

For $\ell = 2$ that is 6 — but we said the $\ell=2$ irrep has dimension 5. So $P_2$ is
reducible, and it is easy to see why: $r^2 = x^2+y^2+z^2$ is rotation **invariant**, so
the multiples of $r^2$ inside $P_2$ (a 1-dimensional subspace, spanned by $r^2$ itself)
form their own invariant subspace. Generally $r^2 P_{\ell-2} \subset P_\ell$ is invariant.

Strip it out. Define the **harmonic** polynomials $H_\ell = \{p \in P_\ell : \Delta p = 0\}$.
The classical decomposition is

$$P_\ell = H_\ell \oplus r^2 P_{\ell-2},$$

and therefore

$$\dim H_\ell = \frac{(\ell+1)(\ell+2)}{2} - \frac{(\ell-1)\ell}{2} = 2\ell + 1 .$$

**There is the $2\ell+1$.** It is the count of degree-$\ell$ polynomials left after
removing everything that a rotation-invariant factor of $r^2$ could account for. And
$H_\ell$ is irreducible under SO(3).

Restricting $H_\ell$ to the unit sphere gives exactly the degree-$\ell$ real spherical
harmonics. Concretely, for $\ell=1$ the harmonic polynomials are spanned by $x, y, z$ —
the harmonics of degree 1 *are* the direction vector. For $\ell = 2$ they are spanned by
$xy,\ yz,\ xz,\ x^2-y^2,\ 3z^2-r^2$: five functions, the traceless quadratic forms. The
tracelessness is the harmonicity condition, and it is what removes the sixth, redundant,
rotation-invariant combination.

So $Y_\ell(\hat r)$ is the canonical way to turn a direction into a degree-$\ell$
equivariant feature, and

$$Y_\ell(R\hat r) = D^{\ell}(R)\, Y_\ell(\hat r)$$

is a definition-level identity, not something to be hoped for.
[`test_spherical_harmonics.py`](../tests/test_spherical_harmonics.py) verifies it to
$\sim 10^{-16}$ against a Wigner-D matrix derived independently.

### 3.3 A convention warning worth its own paragraph

Real spherical harmonics are conventionally ordered $m = -\ell \dots +\ell$, which makes
$Y_1 \propto (y, z, x)$ — not $(x,y,z)$. Worse, **`e3nn` relabels the axes** so that its
second coordinate is the polar axis. The two agree only under
$(x,y,z)_{\text{standard}} = (z,x,y)_{\text{e3nn}}$.

This is the kind of detail that silently produces a model that trains to mediocre
accuracy with no error message. This project implements the standard convention and pins
the relationship down with a test rather than a comment.

---

## 4. Tensor products and Clebsch-Gordan coefficients

Spherical harmonics let us *encode* geometry. To *combine* features we need one more
ingredient, and it is the mathematical heart of the architecture.

### 4.1 The problem

In an ordinary network, combining features is a matrix multiply. Here that is illegal:
mixing an $\ell=1$ component into an $\ell=2$ component with an arbitrary weight would
destroy the transformation law that makes the whole scheme work. Layers may only mix
features *within* a degree (channel mixing, which commutes with $D^\ell$), which by
itself can never create angular information.

So how do we produce genuinely new geometric features?

### 4.2 The answer

Take the outer product. If $u$ lives in irrep $\ell_1$ and $v$ in irrep $\ell_2$, then
$u \otimes v$ lives in a $(2\ell_1+1)(2\ell_2+1)$-dimensional space with representation
$D^{\ell_1} \otimes D^{\ell_2}$. That representation is reducible, and it decomposes as

$$\ell_1 \otimes \ell_2 \;=\; \bigoplus_{\ell = |\ell_1 - \ell_2|}^{\ell_1+\ell_2} \ell .$$

The range is the **selection rule**. The **Clebsch-Gordan coefficients**
$C^{(\ell_1\ell_2\ell)}_{m_1 m_2 m}$ are the change of basis that performs this
decomposition — the projector picking out the $\ell$ piece:

$$w_m \;=\; \sum_{m_1 m_2} C^{(\ell_1 \ell_2 \ell)}_{m_1 m_2 m}\; u_{m_1} v_{m_2}.$$

Viewing $C$ as a matrix of shape $\big((2\ell_1{+}1)(2\ell_2{+}1),\, 2\ell{+}1\big)$, the
defining property is that it **intertwines** the representations:

$$\big(D^{\ell_1}(R) \otimes D^{\ell_2}(R)\big)\, C \;=\; C\, D^{\ell}(R) \quad \forall R .$$

Read left to right: transforming the inputs and then combining gives the same answer as
combining and then transforming the output. That single identity is why the tensor
product is the equivariant analogue of a weight matrix, and
[`test_clebsch_gordan.py`](../tests/test_clebsch_gordan.py) asserts it numerically for
every path used.

### 4.3 It is more familiar than it looks

The $\ell_1 = \ell_2 = 1$ case is entirely recognisable. Two vectors, $3 \times 3 = 9$
components, decomposing as $1 \otimes 1 = 0 \oplus 1 \oplus 2$ with dimensions
$9 = 1 + 3 + 5$:

- the $\ell = 0$ part is the **dot product** $u \cdot v$ (one number, invariant);
- the $\ell = 1$ part is the **cross product** $u \times v$ (three components, a vector);
- the $\ell = 2$ part is the symmetric traceless outer product (five components).

So the Clebsch-Gordan machinery is the systematic generalisation of "the ways two vectors
can be multiplied to give something with definite rotational character". Both special
cases are asserted directly in the test suite.

The selection rule also explains the project's central experiment. With $\ell_{\max}=0$
every feature is a scalar, the only available path is $0 \otimes 0 \to 0$, and no amount
of depth creates angular sensitivity — the model is architecturally equivalent to a
distance-only GNN. Raising $\ell_{\max}$ opens paths that can represent bond angles and
dihedrals directly. That is a claim about representational capacity, and it is what the
ablation measures.

### 4.4 Computing them

The complex-basis coefficients follow from the **Racah formula**, an alternating factorial
sum. Real spherical harmonics relate to complex ones by a fixed unitary $U^\ell$, so the
real-basis coefficients are

$$C_{\text{real}} = (U^{\ell_1} \otimes U^{\ell_2})\, C_{\text{complex}}\, (U^{\ell})^{\dagger},$$

which comes out real up to a global phase (physically meaningless — the learned weights
absorb it). Implemented in
[`clebsch_gordan.py`](../src/symmetrynet/scratch/clebsch_gordan.py); it agrees with
`e3nn`'s `wigner_3j` to $10^{-16}$ in float64.

---

## 5. Wigner-D matrices, and how to test any of this

$D^\ell(R)$ is the explicit $(2\ell+1) \times (2\ell+1)$ matrix by which a degree-$\ell$
feature rotates. It turns "this feature lives in irrep $\ell$" into something a unit test
can check.

A tempting but circular approach is to *fit* $D^\ell$ from spherical harmonics and then
use it to test those harmonics. Instead this project builds it algebraically:

- $D^0(R) = [1]$;
- $D^1(R) = P R P^\top$, the rotation matrix itself in the $(y,z,x)$ ordering;
- $D^{\ell}(R) = C^\top \big(D^{\ell-1}(R) \otimes D^{1}(R)\big) C$ for $\ell \ge 2$,
  which is the intertwining identity read backwards (valid because $C$ has orthonormal
  columns).

No spherical harmonic appears anywhere in that construction, so checking
$Y_\ell(R\hat r) = D^\ell(R) Y_\ell(\hat r)$ is a genuine cross-check of two independent
derivations. It passes at $\sim 10^{-16}$.

---

## 6. Assembling a layer

The pieces now fit together, and each is equivariant for its own reason:

| Step | Operation | Why it is equivariant |
|------|-----------|----------------------|
| Geometry | $r_{ij} = x_j - x_i$ | relative positions are unchanged by translation |
| Direction | $Y_\ell(\hat r_{ij})$ | transforms by $D^\ell$ by construction |
| Distance | $\text{MLP}(\lVert r_{ij}\rVert)$ | a length is invariant, so its output may be used as weights |
| Combine | Clebsch-Gordan tensor product | intertwines, by the identity in §4.2 |
| Aggregate | $\sum_{j \in \mathcal{N}(i)}$ | a sum of equivariant terms is equivariant |
| Activate | gated nonlinearity | higher-$\ell$ features are only ever *scaled* by invariants |
| Readout | take $\ell = 0$ channels | already invariant |

The nonlinearity deserves a note. Applying ReLU componentwise to an $\ell=1$ feature is
**not** equivariant: $\mathrm{ReLU}(Dv) \neq D\,\mathrm{ReLU}(v)$, because a rotation mixes
components and clipping them individually depends on the frame. The fix is that a
higher-$\ell$ feature may be multiplied by any invariant scalar, since a scalar commutes
with $D^\ell$ trivially. Gated nonlinearities compute gates from the $\ell=0$ channels and
scale the rest.

Composition of equivariant maps is equivariant, so the network is equivariant end to end.
Reading only the $\ell=0$ channels at the end converts that into exact invariance of the
prediction — for *every* setting of the weights, including random initialisation. The
demo figure in the README is a picture of that fact.

---

## References

- Bronstein, Bruna, Cohen, Veličković — *Geometric Deep Learning: Grids, Groups, Graphs,
  Geodesics, and Gauges*
- Thomas et al. — *Tensor Field Networks: Rotation- and Translation-Equivariant Neural
  Networks for 3D Point Clouds*
- Batzner et al. — *E(3)-equivariant graph neural networks for data-efficient and accurate
  interatomic potentials* (NequIP)
- Geiger & Smidt — *e3nn: Euclidean Neural Networks*
- Sakurai, *Modern Quantum Mechanics*, ch. 3 — angular momentum, Wigner-D, and
  Clebsch-Gordan from the physics side
