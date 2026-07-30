# Grimlock 6.0 — The Open Problems

*A formal statement of the problems that survive a working pipeline. Companion to `GRIMLOCK_6.0_DESIGN_DECISIONS.md`. Written after an extended output-quality push (notation, over-detection) hit walls that are structural, not tuning. The point of this document is to name each wall precisely — in prose, in math, and in computer-science terms — so future work attacks the real problem instead of re-deriving a filter that can't win.*

---

## 0. Preliminaries: the pipeline as a composition of operators

Let the input be a recorded mixture, a real-valued signal

$$ x(t) = \sum_{i=1}^{S} s_i(t) + \eta(t), \qquad t \in [0, T], $$

where $s_i$ are the $S$ true instrument sources and $\eta$ is recording/room noise. The mixture is captured in $C \in \{1, 2\}$ channels (mono/stereo). Note immediately that $S \gg C$: this is an **underdetermined** system from the first line.

Grimlock is a composition of four operators:

$$ x \;\xrightarrow{\;D\;}\; (\hat{s}_1, \dots, \hat{s}_S) \;\xrightarrow{\;T\;}\; \mathcal{N}=\{n_k\} \;\xrightarrow{\;A\;}\; \tilde{\mathcal{N}} \;\xrightarrow{\;E\;}\; \text{output} $$

- $D$ — **separation** (Demucs `htdemucs_6s`): mixture → per-stem estimates $\hat{s}_i$.
- $T$ — **transcription** (Basic Pitch / CREPE per stem): audio → detected notes $n_k = (\text{pitch}, t^{\text{on}}, t^{\text{off}}, v)$.
- $A$ — **analysis + annotation** (rhythm, key, meter, acoustic witnesses): attaches evidence, never mutates notes.
- $E$ — **engraving/export** (Scribe): notes + findings → MIDI (or, hypothetically, a notation format).

Each operator is imperfect, and — crucially — **their errors have different mathematical characters and live at different points in the chain.** The rest of this document is one section per structural problem, ordered by where it enters.

---

## I. Source-Separation Leakage — "the bleed problem"

### Prose
The separator does not return the true stems. It returns estimates that contain energy leaked from other instruments. When Basic Pitch honestly transcribes the guitar stem, it also transcribes the piano that bled into it. Those leaked notes are *real music in the wrong file* — harmonically valid, confidently detected — so nothing downstream that asks "is this a legitimate tone?" can reject them. This is the root cause of most over-detection, and it is **upstream of everything we control at the note level.**

### Math
Write the separator's output as the true source plus a leakage term:

$$ \hat{s}_i = s_i + \underbrace{\sum_{j \neq i} (\mathcal{L}_{ij} \, s_j)}_{\text{leakage}} + \underbrace{r_i}_{\text{artifact}}, $$

where $\mathcal{L}_{ij}$ is a frequency-dependent leakage operator (in the STFT domain, $\hat{S}_i(f,\tau) \approx M_i(f,\tau)\,X(f,\tau)$ for a soft mask $M_i$; leakage is exactly where $M_i$ fails to be an indicator of source $i$'s support). Define per-stem **leakage ratio**

$$ \rho_i = \frac{\big\| \sum_{j\neq i} \mathcal{L}_{ij} s_j \big\|^2}{\|s_i\|^2}. $$

A perfect separator has $\rho_i = 0$. Real neural separators have $\rho_i > 0$, worst where sources overlap in pitch and time (guitar/piano comping the same chord).

Empirically (3-song, 3468-note diagnostic): **≈16.5% of detected notes have near-silent energy at their own fundamental in their own stem** — i.e. the note's evidence lives in a *different* stem. Formally, for note $n$ in stem $i$ with fundamental $f_0(n)$, define own-band energy $E^{\text{own}}(n) = \int_{|f-f_0|<\delta} |\hat{S}_i(f)|^2\,df$; the suspect population is $\{n : E^{\text{own}}(n) \ll \text{median}_i E^{\text{own}}\}$.

### CS framing
$D$ is an **underdetermined inverse problem** ($S$ sources from $C\le2$ channels) made solvable only by a learned prior (the network's training distribution). Leakage is the irreducible error of that prior on out-of-distribution material. **No post-hoc, note-level operator can invert $D$'s errors** — it can only *mask* them by deletion, trading false-positive notes for false-negative ones. The only true fixes are (a) a better $D$ (we already use the best available checkpoint) or (b) accepting leakage as a modeled nuisance and estimating note ownership across stems jointly, which is a **cross-stem assignment problem** (bipartite matching of a detected note to the stem that most plausibly owns it), not a per-note classification.

---

## II. Over-Detection as Unsupervised Legitimacy Classification

### Prose
Given the detected notes, some are real and some are spurious (leakage or hallucination). We want to label them — but we have **no ground truth**. So it is unsupervised: we can only ask which measurable features *separate* the population into two clusters, and whether that separation is clean enough to threshold.

### Math
Each note gets a feature vector $\phi(n) \in \mathbb{R}^d$. Candidate features and their measured discriminating power, quantified by the **Fisher ratio**

$$ J(\phi_k) = \frac{(\mu_1 - \mu_0)^2}{\sigma_1^2 + \sigma_0^2} $$

(ratio of between-class to within-class variance for feature $k$, splitting on the weak-own-energy suspect set):

| feature | suspect mean | clean mean | verdict |
|---|---|---|---|
| `active_material` (AnechoicMa) | 0.165 | 0.316 | **discriminates** ($J$ large) |
| own-$f_0$ band energy (÷ stem median) | ≈0 | ≈1 | **discriminates** (definitional) |
| `harmonic_match` (Schoenberg) | 0.872 | 0.911 | numerator ≈0 → $J\approx0$ |
| Basic Pitch confidence | 0.446 | 0.494 | $J\approx0$ |

The two useful signals were combined with a logical AND (precision guard):

$$ \text{drop}(n) = \big[\,E^{\text{own}}_{\text{ratio}}(n) < \tau_E\,\big] \;\wedge\; \big[\,\text{active}(n) < \tau_A\,\big]. $$

The AND protects single-signal cases (energy-weak-but-active = quiet real passage; strong-but-inactive = sustained soft note).

### The fragility, formalized
The classifier is a threshold on `active`. Let $p(a)$ be the density of `active` over the suspect population. The fraction of labels that flip under a small threshold or distribution perturbation $\Delta$ is

$$ \Delta N \;\approx\; p(\tau_A)\,\Delta. $$

We set $\tau_A = 0.15$, but the suspect population's **mode sits at $\approx 0.165$** — i.e. $\tau_A$ is placed essentially *at the peak of the density*, where $p(\tau_A)$ is maximal, so $\partial(\text{labels})/\partial\tau$ is maximal. Any run-to-run shift in the distribution (see Problem III) sweeps a large mass across the boundary. On the frozen diagnostic the filter dropped ~6.6%; on a fresh separation it dropped **0%**. This is not a bug in the logic — it is a **decision boundary placed in the highest-density region of the feature**, which is the textbook definition of an unstable classifier.

### CS framing
This is **unsupervised binary classification with no labels and heavy class overlap.** Because there is no ground truth, we cannot even measure precision/recall directly — only surrogate separation (Fisher ratios, cluster distance). The honest models are: (a) place the threshold in a *low-density valley* of the feature (if one exists), not at the mode; (b) replace the hard threshold with a graded, distribution-relative score (percentile within the run) so the operating point is invariant to run-to-run scale shifts; or (c) reframe as the cross-stem *assignment* problem from Problem I, which turns "is this note spurious?" (unanswerable in isolation) into "which stem owns this note?" (answerable relative to the other stems).

---

## III. Pipeline Stochasticity and Non-Reproducibility

### Prose
Demucs runs with a random time-shift trick (`shifts=1`, unseeded). So separation — and therefore *everything downstream* — is a different result every run on the same file. This silently poisons every experiment: you cannot tell whether a metric changed because of your code or because of the dice.

### Math
The pipeline is not a function but a **random function** of an internal seed $\omega$:

$$ \Pi(x, \omega), \qquad \omega \sim \text{Uniform}. $$

For any scalar metric $m$, the variance observed across runs decomposes as

$$ \mathrm{Var}(m) = \underbrace{\mathrm{Var}_{\text{code}}(m)}_{\text{what a change actually did}} + \underbrace{\mathrm{Var}_{\omega}(m)}_{\text{separation dice}} + \text{(interaction)}. $$

To detect a true code effect of size $\delta$ against the noise floor requires

$$ |\delta| \;\gtrsim\; \sqrt{\mathrm{Var}_\omega(m)} \qquad\text{or}\qquad n \gtrsim \frac{\mathrm{Var}_\omega(m)}{\delta^2}\ \text{repeats to average out } \omega. $$

When $\mathrm{Var}_\omega \gtrsim \mathrm{Var}_{\text{code}}$ (as it is here — it is exactly what made the Problem II filter untestable), a single A/B run is **uninformative**.

### CS framing
This is a **reproducibility / determinism defect**, and it is the cheapest high-value fix on the board: seed $\omega$ (or set `shifts=0`, collapsing $\mathrm{Var}_\omega \to 0$). Until it is fixed, no threshold tuned on one run generalizes, no regression test is stable, and no A/B comparison is trustworthy. **Every other experiment in this document is confounded by it.** It should arguably be fixed before anything else, precisely because it is boring.

---

## IV. Rhythmic Quantization as an Ill-Posed Inverse Problem

### Prose
A human plays with expressive timing; the *symbolic rhythm* they intended is a hidden variable. Recovering it from the performed millisecond onsets is an inverse problem, and it is ill-posed: several different notations produce nearly the same timing (triplet vs. swung eighths vs. rubato). You cannot choose among them from the timing alone; you need a prior for what a musician would actually *write*.

### Math
Generative (forward) model. Let symbolic onsets lie on a rational grid $\beta_k \in \frac{1}{q}\mathbb{Z}$ (quarter-note units). The performed onset is the grid position mapped through tempo $\tau(\cdot)$ plus expressive noise:

$$ t_k = \tau(\beta_k) + \varepsilon_k, \qquad \varepsilon_k \sim \mathcal{N}(0, \sigma^2(\text{style})), $$

with $\sigma$ larger for swung/rubato material. Inversion is Bayesian:

$$ \hat{S} = \arg\max_{S}\; P(S \mid t) \;=\; \arg\max_S\; \underbrace{\prod_k \mathcal{N}\!\big(t_k;\, \tau(\beta_k),\, \sigma^2\big)}_{\text{likelihood } P(t\mid S)} \;\cdot\; \underbrace{e^{-\lambda\,C(S)}}_{\text{readability prior } P(S)}, $$

where $C(S)$ is a cognitive-load cost (ties, tuplets, off-beat starts, fine subdivisions). This is exactly `rhythm_inference.py`.

### Why it is ill-posed
The forward map $S \mapsto t$ is **many-to-one after convolution with $\varepsilon$**: distinct $S_1 \neq S_2$ produce overlapping likelihoods, so $P(t\mid S)$ alone does not identify $S$. This is the signature of an ill-posed inverse problem (à la Hadamard: the solution is not unique / not stable). The readability prior $P(S)$ is the **regularizer** that restores well-posedness — it is not decoration, it is what makes the problem solvable at all. The style-dependent $\sigma$ (swing) is why the same $0.0/0.68$ onset pair is a dotted figure under a straight prior and two straight eighths under a swing prior.

### CS framing
Bayesian MAP inference / regularized inverse problem, solved per beat by a small **beam/exhaustive search** over a discrete hypothesis space (the beat-filling vocabulary). It is genuinely tractable and largely built. Its honest limits: the prior weights $\lambda, C(\cdot)$ are hand-authored (not learned from an engraved-score corpus), and it decides symbolic onset+duration but **not** ties/beams/voices — which brings us to the ceiling.

---

## V. The Representational Ceiling — MIDI is a Lossy Quotient of Notation

### Prose
Even with perfect rhythm inference, the output cannot *look* like clean sheet music if it is MIDI, because MIDI has no symbol for a tie, a tuplet bracket, a voice, a beam, or an enharmonic accidental. Two visually different notations that sound identical are the *same* MIDI file. This is not a limitation of our code; it is a property of the format, and it is provable.

### Math
Let $\mathcal{N}$ be the space of notation objects (pitch-spelling, ties, dots, tuplets, voices, beams, …) and $\mathcal{M}$ the space of MIDI event streams (pitch, on, off, velocity, tempo, meter, key). Playback is a rendering map

$$ \rho : \mathcal{N} \longrightarrow \mathcal{M}. $$

$\rho$ is **not injective**: e.g. two tied eighth-notes and one quarter-note render to the identical MIDI note, so $\rho(N_1) = \rho(N_2)$ for $N_1 \neq N_2$. Define the equivalence $N_1 \sim N_2 \iff \rho(N_1)=\rho(N_2)$ ("sounds the same"). Then

$$ \mathcal{M} \;\cong\; \mathcal{N} / \!\sim, $$

MIDI is the **quotient of notation by aural equivalence.** A map has a left inverse **iff** it is injective; $\rho$ is not injective, therefore

$$ \nexists\; \rho^{-1}_{\text{left}} : \mathcal{M} \to \mathcal{N} \quad\text{with}\quad \rho^{-1}_{\text{left}} \circ \rho = \mathrm{id}_{\mathcal{N}}. $$

**You cannot recover notation from MIDI.** Any "read the sheet music back from the MIDI" step is provably lossy — the discarded information (which enharmonic, which voice, where the ties go) is gone at the moment of rendering.

### CS framing
This is a **lossy encoding / non-invertible quotient map.** The engineering consequence is exact: to *emit* notation you must **compute in $\mathcal{N}$ and serialize $\mathcal{N}$ directly** (MusicXML / MEI), never route the symbolic content through $\mathcal{M}$. The current pipeline's export target is $\mathcal{M}$; therefore "clean sheet music" is unreachable on the current target *by construction*, independent of any algorithm quality. Crossing this ceiling is a **format decision** (add a `music21`/MusicXML back-end), not an algorithm improvement.

---

## VI. The Evaluation Gap — No Ground Truth, and Goodhart's Shadow

### Prose
We have no reference transcription for these songs. So there is no automatic way to score "is this transcription good." Every judgment of quality has come from the user's ear. This means our diagnostics can measure *properties* (note counts, energy ratios, readability indices) but cannot measure *quality*, and optimizing the properties can silently diverge from quality.

### Math
Let $Q : \mathcal{T} \to \mathbb{R}$ be true perceptual transcription quality over the space of transcriptions $\mathcal{T}$. $Q$ is only evaluable by a human oracle. Every automated metric $m_i : \mathcal{T} \to \mathbb{R}$ is a **surrogate**, and we have **no validated correlation** $\mathrm{corr}(m_i, Q)$. Optimizing a surrogate,

$$ \hat{\mathcal{T}} = \arg\max_{\mathcal{T}} m_i(\mathcal{T}), $$

is only safe insofar as $m_i$ tracks $Q$; where it doesn't, we get **Goodhart's law** — "when a measure becomes a target, it ceases to be a good measure." (Concrete instance: the note-count-÷-onset-count "over-detection" metric is confounded by polyphony, so minimizing it would penalize correct chords.)

### CS framing
This is the **objective-specification problem**: we are optimizing without an accessible loss function. The honest consequences are: (a) keep a human in the loop as the oracle of last resort — the ear is not a fallback, it is the *only* validated evaluator; (b) treat every automatic metric as a hypothesis about $Q$ to be checked against listening, never as the target itself; (c) when possible, obtain a *small* labeled anchor (one hand-verified passage) to calibrate a surrogate before trusting it at scale.

---

## VII. How the problems compose

The problems are not independent; they form a dependency chain, and the errors **propagate and interact**:

```
   III (stochastic D)  ──amplifies──►  II (unstable classifier threshold)
        │
        ▼
   I (leakage in ŝ) ──creates──► II (spurious notes)
        │                            │
        │                            ▼
        │                     VI (can't even measure whether a fix helped)
        ▼
   T (honest transcription of the wrong stem)
        │
        ▼
   IV (quantize whatever notes survive) ──blocked by──► V (MIDI can't show it)
```

Read top to bottom: **III makes II untestable**; **I feeds II its spurious notes**; **VI means we can't score any of it automatically**; and even a perfect IV is **capped by V** on a MIDI target. This is why chasing a better output-side filter kept failing — the leverage is not there.

---

## VIII. Implications — where the leverage actually is

Ranked by (value $\times$ tractability), honestly:

1. **Fix III first (determinism).** Seed or disable Demucs's random shift. Near-zero cost; unblocks *every* other measurement. Nothing else is trustworthy until this is done.
2. **Reframe II as cross-stem assignment (I).** Stop asking "is this note real?" in isolation (unanswerable, unstable). Ask "which stem owns this note?" — a bipartite matching against per-stem own-$f_0$ energy — which protects genuine doublings and directly attacks leakage. Graded/percentile scoring, not a mode-adjacent hard threshold.
3. **Make the V decision explicitly.** If clean notation is a real goal, MIDI is a dead end by the quotient argument; commit to a MusicXML/MEI back-end (a format decision) or explicitly scope the deliverable to "a good MIDI a human finishes in a DAW" and stop fighting the ceiling.
4. **Respect VI throughout.** Anchor every automatic metric to at least one hand-verified listen before trusting it; treat the ear as the oracle, not the fallback.

The transcription *core* (separation now genuine 6-stem, drums, bass, tempo, key) is in good shape. The unsolved problems all live at the **edges**: the separation front (I, III) and the notation back (IV capped by V), with II and VI as the cross-cutting difficulties of doing anything measurable in between.

---

## IX. The Problems in Other Clothes — Cross-Domain Transfers (possible paths forward)

Every problem above has a *named, decades-old twin* in a field with bigger budgets and higher stakes — radar, sonar, astronomy, particle physics, seismic imaging, CT/MRI. All of those fields solve the same sentence: **recover latent structure from indirect, noisy, underdetermined measurements of a wave field.** In several cases the twin comes with an algorithm that transfers directly. These are candidate paths forward, not commitments.

### IX.1 — CFAR detection (radar) → the §II threshold fragility

Radar solved our exact problem in the 1960s: a fixed detection threshold fails because clutter (sea state, rain) shifts the noise floor scene-to-scene — precisely "Demucs stochasticity shifts the AnechoicMa distribution run-to-run, threshold at the mode, 6.6% becomes 0%." **Constant False Alarm Rate** detection makes the threshold adaptive:

$$ \tau = \alpha \cdot \hat{\mu}_{\text{local}}, $$

where $\hat{\mu}_{\text{local}}$ is estimated from the surrounding cells *of the same scene* (for us: the same run's own distribution) and $\alpha$ is derived analytically from a chosen false-alarm probability $P_{FA}$. The operating point lives in **probability space, not feature space**, so it is invariant to run-to-run scale shifts *by construction*. This is the rigorous form of the "distribution-relative threshold" fix note_support needs — with theory for choosing $\alpha$ instead of a guess. Companion tools from the same family: the **look-elsewhere effect / trials factor** (particle physics) and **Benjamini–Hochberg FDR** — testing 3468 notes at per-note rate $p$ guarantees $\approx 3468p$ false alarms; control the family-wise rate, not the per-note one.

### IX.2 — Injection–recovery testing (astronomy/LIGO) → the §VI evaluation gap

None of these fields has ground truth on real events either — nobody hands astronomers a labeled exoplanet. Their answer: **inject synthetic signals into real data and measure recovery.** Musical version: render a known MIDI through a soundfont, mix it into a real recording or stem, run the full pipeline, score precision/recall against the injected truth. This *manufactures a calibrated loss function* — completeness and contamination as functions of SNR, polyphony, register — without a single hand label. It converts §VI from "no loss function" to "a loss function on demand." The ear remains the oracle for *musicality*; injection–recovery becomes the oracle for *detection statistics*.

### IX.3 — Time-slide coincidence backgrounds (LIGO) → doubling vs. bleed (§I/§II)

To know how many two-detector coincidences are *accidental*, LIGO shifts one detector's data stream by a non-physical offset and counts coincidences — an empirical chance baseline. Musical version: time-shift one stem's notes by e.g. +5 s and count cross-stem coincidences → the accidental-doubling rate for *this song*, measured rather than assumed. (`TimeSignatureDetector`'s shuffle baseline independently invented this trick; LIGO shows it generalizes.)

### IX.4 — Conservation / closure test (particle physics) → cross-stem assignment (§I)

Physics hands us a constraint we have never used: the stems must sum back to the mixture, $\sum_i \hat{s}_i \approx x$. Therefore the mixture's energy at a note's $f_0$ is a **budget**, and stems claiming that note must share it — a note whose claimed energy exceeds the mixture's budget in that band is *physically impossible*, no threshold needed. Energy allocation across stems becomes a constrained assignment problem rather than six independent opinions (see §X.2 for the optimal-transport formalization). This is also the crowded-field-photometry lesson (DAOPHOT): in a crowded field you never photometer stars independently — you **jointly fit** all overlapping sources and allocate the observed light once. Our mix is *always* a crowded field.

### IX.5 — Pulsar timing / PLL (astronomy, comms) → tempo and drift (§IV)

Tempo estimation from sparse noisy onsets **is** pulsar timing: recover period + phase + drift from irregular times-of-arrival. Mature tools: **epoch folding**, the **Lomb–Scargle periodogram** (period detection on unevenly sampled point processes — exactly an onset train), phase-coherent timing solutions, formal *glitch* detectors (= tempo changes). A beat tracker is a **phase-locked loop**; rubato is oscillator drift. Also: fitting a rational lattice to observed points (ReverseGeoCrypt's job) is diffraction-pattern **indexing** in crystallography — solved there.

### IX.6 — Limited-angle CT and compressed sensing (MRI) → sharpening §V

Limited-angle tomography confirms the quotient argument: the forward operator has a genuine null space. But compressed sensing (Candès–Romberg–Tao) adds the twist: formally insufficient measurements *can* identify a signal given a strong enough prior. Translated: $\rho$ has no left inverse, but the **fiber** $\rho^{-1}(M)$ supports a posterior, and a music prior (key → spelling, meter → ties, voice-leading → voices) concentrates it. The precise statement becomes: notation from MIDI is MAP decoding through a lossy channel — possible but fighting the format — while notation emitted **upstream from the analysis layer** decodes with side information that narrows the fiber *before the information is destroyed*. Same conclusion as §V (emit from $A$, not from $\mathcal{M}$), now with an information-theoretic reason.

### IX.7 — Metrology (all of physics) → §III

A number without an error bar is not a measurement. Every field here either freezes the randomness (calibrated instrument = seed the pipeline) or runs repeats and quotes $\pm\sigma$ (Monte Carlo error propagation, MRI signal averaging). Single-shot A/B comparison on an unseeded instrument is a practice none of these fields would accept. Nothing to steal algorithmically; the *standard* is the transfer.

### IX.8 — Honest split: transfers vs. decoration

| Transfers (comes with an algorithm) | Decorative (metaphor only, or inapplicable) |
|---|---|
| CFAR → note_support threshold (§II) | Beamforming/DOA — we have no microphone array |
| Injection–recovery → calibrated evaluation (§VI) | Full-waveform inversion — would mean rebuilding Demucs |
| Time-slide backgrounds → doubling-vs-bleed baseline (§I/II) | Detector-response unfolding — needs a response matrix we don't have |
| Conservation budget + joint fitting → cross-stem assignment (§I) | CLEAN deconvolution — approximately what Basic Pitch's training already does |
| Lomb–Scargle / epoch folding / PLL → tempo & drift (§IV) | Quantum-measurement irreversibility — §V already has the proof |
| Fiber-posterior view → sharpens §V's "emit upstream" | |

The pattern worth noticing: the two biggest transfers (CFAR, injection–recovery) land exactly on the two problems §VII calls *cross-cutting* — the unstable classifier and the missing loss function. These fields did not beat us with cleverer features; they beat us with **better statistical procedure**. That is consistent with the diagnostic: our features (`active_material`, own-$f_0$ energy) were already discriminating — the thresholding and evaluation methodology were the amateur parts.

---

## X. The State-Estimator Horizon — an external proposal, triaged

*An external AI proposal ("Musical State Estimator / factor graph") arrived after §IX was drafted. Per standing practice it is triaged here — kept ideas grounded, duplicates named, violations flagged — rather than adopted wholesale. Notably, much of what it presents as novel is a restatement of what 6.0 already is.*

### X.1 — What the proposal re-derives (already built)

- **"Keep alternatives alive; high entropy means don't commit; decide later"** — this *is* the annotation-not-mutation law. The `AnnotationStore` accumulates typed evidence against frozen Notes; nothing commits until `engrave()`. The proposal describes our architecture back to us as a novelty.
- **"Notation as energy minimization / Ising / minimum free energy"** — exactly `rhythm_inference`'s $P(S)\propto e^{-\lambda C(S)}$ readability prior. Same mathematics, statistical-physics vocabulary.
- **"Never classify, estimate probability" for stem ownership** — the graded support score already proposed for note_support (task #268), now strengthened by CFAR (§IX.1).
- **Graph community detection for instrument/voice assignment** — re-derives `voice_continuity` and instrument_attribution.

### X.2 — What is genuinely worth keeping (new, concrete, scoped)

1. **Damped-oscillator evidence for bleed (§I/§II).** A played note is locally $x(t) = A e^{-\lambda t}\cos(2\pi f t + \phi)$. Bleed inherits the *source* instrument's excitation physics — attack sharpness, decay rate $\lambda$, Q — even after the separator's mask reshapes its spectrum. Fitting decay and phase coherence at a note's $f_0$ (matching-pursuit-style atomic decomposition, or a simple exponential fit to the band envelope) gives a physically richer ownership feature than band energy alone: *does this note's envelope match the physics of the instrument claiming it?* Slots into note_support as a better $E^{\text{own}}$; touches nothing upstream. **This is the one real new signal-processing idea in the proposal.**
2. **Optimal transport as the formalism for the conservation budget (§IX.4).** Given the closure constraint $\sum_i \hat{s}_i \approx x$, allocating the mixture's energy at $(f,\tau)$ across stems *is* a transportation problem: move observed energy mass to instrument claims at minimum cost, total mass conserved. OT is the mature machinery for exactly that; it upgrades §IX.4 from "constraint" to "solvable program."
3. **CASA / Gestalt grouping as an evidence layer.** Auditory Scene Analysis (Bregman): humans group sound by harmonicity, common onset, common fate, spectral continuity. Reframing ownership as "what auditory object does this event belong to" — with common-onset/common-fate as measured features — is a legitimate, literature-backed companion to the time-slide baseline (§IX.3). Evidence layer only; annotations, not decisions.

### X.3 — The big idea, and the required constraint on it

The proposal's centerpiece: replace the pipeline with a **factor graph** — one hidden musical state (tempo, harmony, voices, instruments, rhythm), every subsystem contributing probabilistic constraints, the score emerging as the joint MAP solution. Honest assessment:

- It is a real research direction — it is how modern SLAM and speech decoding work, and §0's operator-composition framing points at it.
- **The annotation architecture is already the tractable first-order version of it.** Evidence accumulates against shared state; commitment is deferred to the last responsible moment. What 6.0 lacks relative to a full factor graph is *message passing between evidence types* (loopy belief propagation), which buys joint consistency at the price of convergence risk, compute cost, and — the decisive one — **undebuggability**. 6.0's layered, locally-testable design exists precisely because monolithic joint systems can't be verified one claim at a time.
- **A hard scope constraint, non-negotiable:** the proposal's diagram has engraving contributing constraints *into* the musical state — i.e., notation preferences influencing transcription. That is exactly the violation this project has already rejected (standing rule: notation/export never touches transcription or analysis). Any factor graph ever built here must be **directed at the notation boundary**: notation factors *receive* messages and never send them upstream. Readability is a prior over how to *write* what happened, never over *what happened*.

Verdict: file under **Grimlock 7.0 horizon**. Not a 6.0 work item. If 6.0's edges (I, II via §IX methods) are solved and the remaining errors look like *coordination failures between subsystems* — each locally right, jointly inconsistent — that is the signal the factor graph is the next architecture. Until then it is premature generalization.

### X.4 — Cut

- **Persistent homology / TDA** — no stated musical payoff; "underexplored in MIR" is a warning, not a credential.
- **Modal analysis** — duplicate of X.2's oscillator idea, mechanical-engineering dialect.
- **HMM/Kalman/particle-filter/WFST/SLAM name lists** — a bibliography, not a design. (The one concrete instance already lives in §IX.5: beat tracking as a PLL/timing solution.)
- **"Stop thinking about MIDI"** — the deliverable is still a score a musician can read; §V/§IX.6 already state precisely where MIDI's ceiling is and what crossing it costs. Abstraction is a means here, not the product.

---

## XI. Updated leverage ranking (supersedes §VIII where they differ)

1. **Determinism first (§III / IX.7)** — seed or disable Demucs's random shift. Unblocks all measurement. Unchanged, still first, still boring.
2. **Injection–recovery harness (§IX.2)** — build the synthetic-truth evaluator *before* the next filter, so the next filter is the first one we can actually score.
3. **CFAR-style note_support (§IX.1)** — re-cut the §II classifier with a $P_{FA}$-derived, run-relative threshold; add the damped-oscillator feature (§X.2.1) if band energy alone under-discriminates.
4. **Conservation/OT cross-stem assignment (§IX.4 + §X.2.2)** — the structural attack on bleed, replacing per-note verdicts with joint allocation.
5. **The §V format decision** — unchanged: MusicXML/MEI back-end or explicitly scope to "good MIDI a human finishes"; now with §IX.6's sharper "emit upstream, narrow the fiber early" argument.
6. **Factor graph (§X.3)** — horizon only; revisit when the remaining errors look like coordination failures, not evidence failures.

---

## XII. What we tried, what broke, what's new (2026-07-17/18)

*A working session that resolved two problems, reversed two of this document's own conclusions, and surfaced four genuinely new ones. Recorded here because several of the reversals came from the user's ear beating a metric — which is itself the §VI evaluation problem, observed live.*

### XII.0 Resolved

- **§III Stochasticity — FIXED.** `apply_model(shifts=1)` drew an unseeded random time-shift. `separation_engine/demucs_engine.py` now seeds `random`/`numpy`/`torch` (default `seed=0`). Verified three ways: byte-identical stems (seed=0) vs. differing (seed=None, PIANO Δ0.107); **byte-identical full-pipeline MIDI** across two Hopeful runs; and an exhaustive RNG audit proving Demucs was the *sole* non-deterministic source (`meter.py`'s shuffle was already locally seeded). Measurement is now unblocked.
- **Tempo mis-resolution — FIXED (a real bug, not a domain limit).** On Hopeful the anchor witness (madmom) read **146.3** — essentially correct — and the referee *degraded it to 157* by averaging in two witnesses reading ~19% high that did not fold to any clean ratio of the anchor. `resolve_tempo` now (a) excludes witnesses whose best fold lands beyond 8% of the anchor (implementing behaviour its own docstring already claimed) and (b) leans on the anchor in proportion to the excluded weight. Hopeful 157→148; validated on No Pasarán (excluded the same offender, resolved a clean 132). **This is the clearest instance yet of the triptych thesis: the reconciliation layer took a correct answer and made it worse.**

### XII.1 NEW — Separator label instability, and why transcription does not distribute over it

**The biggest finding of the session, and it re-explains §II.** The user, listening to isolated stems: *"it fades in and out and I can't tell you why piano is split up the way it is… it doesn't seem to be by range or consistent."* htdemucs_6s's guitar/piano heads are weakly trained relative to drums/bass/vocals, and they **reassign the same content between guitar, piano, and other over time.**

The consequence is arithmetic. We transcribe each stem independently and union the results. But if the separator's assignment is an unstable soft one that we treat as a hard partition, then

$$ \Big|\bigcup_i T(\hat s_i)\Big| \;\neq\; \Big|T\Big(\sum_i \hat s_i\Big)\Big| $$

— **transcription does not distribute over an unstable partition.** Measured directly on Hopeful:

| | notes |
|---|---|
| guitar transcribed alone | 1,549 |
| piano transcribed alone | 1,564 |
| other transcribed alone | 1,532 |
| **Σ of the three** | **4,645** |
| **the three summed to one stem, transcribed once** | **2,183** |

**53% duplication.** More than half those notes were the same music counted repeatedly because the separator could not decide which stem owned it. This also closes the gap that never added up: Basic Pitch on the bare mix finds **3,260** notes; the 6-stem pipeline reported **7,822**.

**So §II's "over-detection" is substantially over-*counting*.** Across this session that diagnosis moved three times — (1) spurious/hallucinated notes → (2) real polyphony flattened onto one track → (3) *separator duplication*. Only (3) is supported by a controlled measurement.

**CS framing:** a labeling/clustering instability. The model emits a soft, time-varying assignment; the pipeline consumes it as an exclusive partition. **Architectural consequence:** htdemucs_6s's extra heads cost more than they give on this material — either run plain `htdemucs` (4 well-trained targets) or merge guitar+piano+other immediately after separation and transcribe once. *Not yet adopted; needs verification on a second song.*

### XII.2 NEW — Separation destroys the evidence that sustain questions need

Attempting to replace a magic gap threshold with real acoustics, we asked **AnechoicMa** per note-gap: is the note still ringing (resonance) or is this real silence (void)? It answered **REST for 201 of 202 gaps** — zero discriminating power.

The reason is structural, not a tuning failure. Separation works by masking: $\hat s_i = M_i \odot X$. A decaying tail is exactly the low-confidence, ambiguous energy a mask suppresses. So the separated stem genuinely *is* near-silent between notes, and a witness measuring resonance on $\hat s_i$ is measuring **the mask, not the instrument**.

$$ \textbf{You cannot ask a gated signal about sustain — the separator removed the answer before the question.} $$

Generalizes beyond this decision: **any acoustic witness that depends on decay, resonance, or release is unreliable when run on separated stems.** (AnechoicMa remains valid for the activity/void judgments it was ported for; it is the *sustain* question it cannot serve.)

### XII.3 NEW — Detector truncation is the root of the "notation confetti"

The readable-rhythm problem has a single upstream cause. Using Basic Pitch's own posteriorgram, comparing per-pitch activation *during the gap after a note* against activation *during the note itself*:

$$ \text{median}\ \frac{\overline{A_{\text{gap}}}}{\overline{A_{\text{note}}}} \;=\; 1.29 $$

**The pitch is more active after the model's declared note-off than during the note.** Basic Pitch systematically ends notes early. That explains bass durations with a median of 153 ms — which at 146 BPM *is* a sixteenth — and therefore the 16th/32nd clutter that made the score unreadable. The durations were never real; they were truncated.

Corroborated independently: Basic Pitch on the bare mix yields **54% sixteenth-or-finer**, *identical* to the full pipeline's output. Neither our quantization nor our separation caused the confetti, and neither fixed it. Only duration correction moved it (54% → **40%**).

**This validates SustainRecovery's premise** — it already extends note-ends from this same posteriorgram evidence. The notation layer should *read* those annotations rather than re-derive the rule.

### XII.4 NEW — Loudness-confounded evaluation (a methodological trap that bit twice)

Two "is this note real?" tests failed the same way. Both compared a note's own-stem energy against a reference that was not loudness-normalized — typically $\max_j e_j$ over other stems. With per-stem gains $g_i$, the statistic

$$ r = \frac{e_{\text{own}}}{\max_j e_j} \;\propto\; \frac{g_{\text{own}}}{\max_j g_j} $$

scales with *relative loudness*, not authenticity — so **a quiet but genuine instrument is structurally guilty.** Consequences observed:

- The bleed measurement reported the piano stem **86% "bleed."** Raw RMS then showed piano at **0.0146 with peak 0.464** — quieter than the others but unmistakably real audio. The user's ear ("piano seemed okay") was right; the metric was wrong.
- Vocals ranked #1 "bleed source" for *every* stem — an artifact of being broadband and loud, hence winning any max-comparison.
- `note_support` (§II, still parked) has the same shape.

**Rule going forward:** normalize *within* a stem before comparing *across* stems, and never use max-over-others as the reference. Add to §VI's ledger: this is the evaluation gap manifesting as false confidence in a plausible-looking statistic.

### XII.5 §V revised — the MIDI ceiling is real, but it was never the bottleneck

MusicXML was tested for the first time (music21 installed). It delivered exactly what §V predicted MIDI could not: real measures, rests, ties, beams, correct clefs, key signature, and the guitar split into three independent readable voices.

**But it fixed nothing rhythmically on its own.** The same notes exported to MusicXML with literal durations were still **62% sixteenths**. The cleanup came from two decisions *we* control:

| policy | 16th-or-finer |
|---|---|
| literal durations, 16th grid | 62% |
| fill-to-next-note, 16th grid | 37% |
| fill-to-next, eighth grid | **0%** |
| posteriorgram-driven durations, 16th grid | **40%** *(detail preserved)* |

So the honest restatement of §V: **the format was the ceiling, not the cause.** The correct framing is *"make good symbolic decisions **and** use a format that can express them"* — MIDI blocked the second half; our duration policy blocked the first. A large share of the readability win was available in MIDI all along.

A further finding: **no single global grid can be both smooth and detailed** — a coarse grid erases genuine sixteenth motion, a fine grid clutters. The subdivision must be chosen *per beat*, which is precisely what `quantization/rhythm_inference.py` was built to do and remains unwired.

### XII.6 Tried and rejected (recorded so we don't retry)

- **A guitar↔piano bleed sieve** — measured *before* building: piano accounts for only 11% of guitar's cross-stem energy and guitar 17% of piano's. They are not primarily contaminating each other. The grounding measurement prevented building the wrong thing.
- **AnechoicMa as the fill-vs-rest oracle** — XII.2; zero discrimination.
- **A global quantization grid** — XII.5; forces a smooth-vs-detail trade that per-beat inference avoids.
- **Deleting "over-detected" notes** — XII.1 shows most were real music counted twice. Deletion would remove genuine content; **separation and de-duplication, not filtering.**

### XII.7 Revised leverage ranking (supersedes §XI)

1. **De-duplicate the harmonic stems** (XII.1) — merge guitar+piano+other before transcription, or drop to 4-stem. Largest measured effect of anything found: −53% notes with no loss of real content. Verify on a second song first.
2. **Duration correction from the posteriorgram / SustainRecovery** (XII.3) — the root cause of unreadable rhythm; 54%→40% sixteenths, and it's already computed.
3. **Per-beat adaptive quantization** via `rhythm_inference` (XII.5) — replaces the global grid; the remaining smooth-vs-detail gap.
4. **A real `musicxml_export`** fed by the pipeline's own annotations (notation timing, voices, ties, key) rather than a scratchpad re-derivation.
5. **Meter** — still the weakest scalar: Hopeful resolved 4/4 against a confirmed 6/4; No Pasarán moved 12/4→4/4 between runs, contended both times (#255).
6. **Injection–recovery harness** (§IX.2) — now more valuable than ever, because *four separate conclusions this session were overturned by a single measurement or by the user's ear.* Nothing here is trustworthy without a scoreable ground truth.

**The through-line of this session:** every real gain came from *measuring before building* — and every reversal came from a metric that looked principled but wasn't grounded. The engine's detection layer keeps proving sound; the failures keep turning out to be in how components hand off to each other, and in how we evaluate them.

---

## XIII. The merge confirmed, and what two external audits were worth (2026-07-18)

### XIII.0 RESOLVED — the harmonic-stem merge, end to end

XII.1 predicted the merge would cut Hopeful from 7,822 notes to roughly 5,300 with no loss of real content. Full pipeline run, `merge_harmonic_stems=True`, seed 0:

| stem | baseline | merged | Δ |
|---|---|---|---|
| drums | 662 | 662 | 0 |
| bass | 1099 | 1099 | 0 |
| vocals | 1416 | 1416 | 0 |
| guitar + piano + other | 4645 | 2191 | **−2454** |
| **total** | **7822** | **5368** | **−2454** |

Two independent confirmations that this is the *right* −2454 and not collateral damage:

1. **The delta is exactly the harmonic delta.** Drums, bass and vocals are unchanged to the note, so nothing outside the merged stems was touched.
2. **In-pipeline agrees with offline.** Transcribing the reference merged `.wav` standalone gave 2,183 notes; the pipeline's merged stem gives 2,191 — 0.4% apart. The merge inside the Conductor reproduces the measurement that motivated it.

Tempo held at 148.0 BPM (the #270 fix), key Bm (0.83). Meter still resolves 4/4 against a confirmed 6/4 — **unchanged, still #255.** This is now the top open scalar.

**Unexplained, logged for later:** MuseScore renders the earlier export in Gb (6 flats); we detect Bm (2 sharps). MuseScore infers key from accidentals so it is not authoritative, but the two readings are not reconcilable and nobody has looked at why.

### XIII.1 NEW — Correlated evidence counted as independent votes

The sharpest finding to come out of the external audits, and it is true of us.

> Ask ten doctors about the same MRI. If all ten agree, that is not ten independent opinions. It is one MRI.

Many of our witnesses derive from **the same STFT**: spectral flatness, centroid, band energy, harmonic-series match, HPS. We combine them by confidence-weighted averaging, which is only sound for *independent* evidence. For correlated evidence it systematically **overstates** the resulting confidence — agreement is partly guaranteed by shared input, not by corroboration.

This is a mechanism for something already observed but unexplained: witness agreement has been a weak predictor of correctness. Formally, with correlation matrix Σ, the effective number of independent observations is

    N_eff = N² / (1ᵀ Σ 1)

which for strongly correlated witnesses is far below N. We currently behave as if N_eff = N everywhere.

**Not a call to merge the witnesses** (see XIII.7) — they must stay separable to stay individually falsifiable. The fix is to know Σ and discount accordingly.

### XIII.2 NEW — Per-note uncertainty has no referee

`epistemic/referee.py` is a real fusion point, but only for **scalars**: tempo, meter, key. For **per-note** decisions there is no referee at all. Annotations accumulate against a note id and the engraver consumes them ad hoc.

Worse, the numbers are not commensurable: a 0.7 from AnechoicMa and a 0.7 from the note-onset witness are not the same quantity, and nothing in the system says so. Most of these are heuristic scores wearing the word "confidence." Reserving *probability* for normalized distributions, and naming the rest honestly (support, reliability, likelihood, weight), is a prerequisite for any principled fusion.

This and XIII.1 are the same debt seen from two sides.

### XIII.3 NEW — Notation needs a level of representation above the note

A trill is not a note. A glissando is not one note. A cymbal roll is not one note. Each is **many detections mapping to one notated symbol** — and we have no object that can hold that relation. `Note` is (correctly) frozen at the detection floor; nothing above it groups.

This is a **notation-layer** gap, not an argument against the frozen note. It becomes a requirement on the `NotationScore` object (XIII.6): a grouping level that owns ornaments, gestures, and phrase-level symbols, sitting above immutable notes.

Adjacent and equally real: **voice ownership** is not modelled. Bass line, left hand, melody, counter-melody — who owns a note is undefined, and it matters precisely at export, where voice assignment determines stems, beams and staff placement.

And the **notation intent layer** is missing outright: ties, tuplets, grace notes, beams, pickup, swing rendering. These are currently *inferred by MuseScore*, which is why the page is a guess. They should be decided by us and serialized, not left to the reader to reconstruct.

### XIII.4 NEW — Evaluation is two axes, not one

§VI treats evaluation as a single ill-posed problem. It is two, and they behave differently:

- **Perceptual correctness** — does the MIDI *sound* like the recording?
- **Notation correctness** — does the page *read* like the music?

Confirmed by ear: Hopeful is close to 100% on the first and roughly 20% on the second. A single blended metric would have hidden that entirely, and would have sent us to fix the detector — which is not broken.

This split also **falsifies the "premature commitment" diagnosis** (XIII.7): if committing to notes too early were the root disease, perceptual correctness would be damaged too. It isn't.

### XIII.5 NEW — Multi-target tracking as a §IX transfer (extends §IX.8)

**IX.9 — Multi-target tracking / data association (radar, air-traffic control) → voice assignment (§I).** JPDA, MHT and track-before-detect are the mature literature for "which observation belongs to which object, over time, under uncertainty." That is exactly the voice-assignment problem, and `voice_continuity`'s greedy pitch-proximity streaming is a crude first-order version of it.

Note the retargeting: this was proposed for *stem ownership*, but XIII.0 largely dissolved that problem by merging. The live instance is **tracking voices within a stem**, which XIII.3 shows is also blocking notation.

### XIII.6 The `NotationScore` object — one symbolic truth, many renderings

Both external audits converged on this independently, and it matches the user's own "post-production translator" framing. Today MIDI and MusicXML each re-derive measures, voices, durations and ties separately. They should serialize from one object:

    annotations → NotationScore → { MIDI, MusicXML, LilyPond }

`NotationScore` owns what XIII.3 says is missing: measures, voices, ties, tuplets, ornament/gesture groups, spelling. MIDI stops being the truth and becomes one exporter.

**`Timeline`, scoped honestly.** A single object owning sample/frame/ms/beat/tick/measure conversion is right in principle. Measured scale before committing: **21 conversion sites across 11 files**, with `core/tempo_types.py` already a partial home, and PPQN already centralized as one constant in `output/scribe_engraver.py`. This is a half-day consolidation, not a version-defining rewrite — worth doing alongside `NotationScore`, not instead of it.

### XIII.7 External audits — rejected, with reasons (recorded so we don't retry)

Two long external architecture audits were triaged this session. Both were useful as idea generators and unreliable as verdicts. Recording the rejections *and why*, because these proposals will recur:

- **"Premature commitment is the root disease; un-freeze the note; make pitch a hypothesis."** We ran this experiment. 5.x gated everything through the epistemic layer and fidelity went **88% → 22%** (bible §8). And XIII.4 refutes it directly: the notes are *right*. A disease that leaves the output correct is not the disease.
- **"The engines know too much about each other; replace with a blackboard."** Verified false: **zero cross-package imports** exist between `pitch_engine`, `rhythm_engine`, `acoustic_witness`, `key_intelligence`, `instrument_attribution` (one type-only edge, `quantization` → `pitch_engine`). We also already *have* the blackboard — `MusicalFindingsMap` + `AnnotationStore` + `MusicBox`. The everyone-messages-everyone version is 5.x, and it is what made 5.x unfalsifiable.
- **"`Note` is a Christmas tree of confidence/support/fingerprint/metadata fields."** False. `core/note_types.py:42` — nine fields, frozen. That describes Symphony 5.x's `NoteEvent`.
- **"Add a `note.evidence[...]` registry."** That is the `AnnotationStore`, keyed by `note.id`. Already the central law (§2.2).
- **"MusicBox owns notes/tempo/audio; grow it into world state."** MusicBox is an append-only forensic ledger and owns no notes.
- **"Merge Groove/Pulse/Beat/Tempo/Lattice into one engine; merge the acoustic witnesses."** They are already one package each. They are separate *files* because they must be able to disagree — which is how #271 (note-onset witness reading 19% high) was caught at all. Merging destroys the diagnostic. The dedup concern underneath is already handled by `TransformCache`.
- **"Retire the word *witness*; it implies certainty."** Read backwards. A witness is precisely an account that is *contested and cross-examined*; the courtroom frame (testimony, referee, veto, contention) is load-bearing.
- **"Confidence should exist once, at the final decision."** Would break the referee, which needs per-witness confidence to weight. The real problem is XIII.2 — incommensurability, not multiplicity.
- **"Genre calibration profiles (`profile = JazzTrio()`)."** Centralizing constants: yes. Genre presets: no — selecting a profile *is* a confident global claim about the music, made invisibly, with no way to validate it. That is the inherited reflex (§10) with a nicer API.

**Methodological note.** Both audits scored a codebase they had only read as a flat text dump and never run. Several load-bearing claims were checkable in one grep and false. Their *prescriptions* (notation intent layer, `NotationScore`, voice ownership, correlated evidence) were consistently better than their *diagnoses* — and every one of their diagnoses was beaten by the user's ear. Treat this genre as a source of hypotheses, never of verdicts, and ground each one before acting.

### XIII.8 Revised leverage ranking (supersedes XII.7)

1. ~~De-duplicate the harmonic stems~~ — **DONE, XIII.0.** −2454 notes, fully accounted for.
2. **`NotationScore` + notation intent layer** (XIII.3, XIII.6) — now the single largest lever. It is where ties, tuplets, voices, ornaments and beaming stop being MuseScore's guesses. Absorbs the MusicXML export item.
3. **Duration correction from the posteriorgram** (XII.3) — still the root cause of unreadable rhythm, still already computed.
4. **Per-beat adaptive quantization** via `rhythm_inference` (XII.5).
5. **Meter** (#255) — the weakest scalar and now the only one still wrong on a song we know the answer to.
6. **Injection–recovery harness** (§IX.2) — unchanged in importance and unbuilt; the split metric of XIII.4 tells us it must score the two axes *separately*.
7. **Evidence correlation / commensurable confidence** (XIII.1, XIII.2) — real debt, but diagnostic before it is corrective. Measure Σ before designing any fusion.

## XIV. The rebuild session — solved, walled, and the wiring audit (2026-07-27/28)

An extended build+measure pass took the 8-item working list (1 consolidation, 2 tempo-octave, 3 notation, 4 timbre, 5 drums, 6 stem-merge, 7 downbeat, 8 reconciliation) and resolved or *precisely bounded* every one. All commits on `grimlock-6.0-rebuild`. The through-line: the tractable slices are now either done, or measured as walls, and what genuinely remains is large (upstream voice separation) or risky (note rewrite).

### XIV.1 Solved and committed
- **Consolidation** (the keystone; the "notation confetti" root of XII.3). `quantization/note_consolidation.py` merges Basic Pitch's fragmented same-pitch runs into sustained notes — opt-in, annotation-only (§2), pitch-equality a hard gate + connected-component grouping (ported from Symphony `velocity_merge`). You Say pitched 4847→2547, durations ~doubled. `4225746`.
- **Drums** (was a 3-rule stub). `rhythm_engine/drums.py` rebuilt multiband multi-voice — kick/snare/hat detected independently per band and coexisting — then an over-detection refinement (kick must dominate low-band; snare on a narrow 2-5kHz "crack" band; per-band energy gate). You Say 2912→1504 hits at musical per-bar rates. `f063d92`+`a81b3d2`.
- **Distributed downbeat (#7).** `rhythm_engine/downbeat_witness.py` = madmom RNN+DBN bar tracker as a meter witness, after measuring that every hand-rollable source is a weak cue (beat-sync harmonic change 1.04-1.23× chance; kick-on-1 1.10-1.38×, weakest on syncopation). Fixes Hopeful→6/4, No Pasarán→4/4; voted into `resolve_meter` (helper, not fighter). `0ca24e3`.
- **Tempo octave (#2).** `epistemic/referee.py::arbitrate_tempo_octave` — a *confident* downbeat witness reading a clean 2:1 above the resolved tempo un-halves it (No Pasarán 66→130); a confidence floor protects a genuinely-slow triplet-feel song (Gospel stays 67; its witness is only 0.30 confident). `0ca24e3`. **Recorded negative:** harmonic rhythm is octave-INVARIANT in absolute time, so "let harmonic rhythm vote on the octave" (XI-era idea) cannot work — disproved by measurement.
- **Notation slice (#3).** MusicXML export wired opt-in (`transcribe_file(output_musicxml_path=...)`); the page now reads rhythm_inference's beat-relative symbolic timing + consolidation, and tuplets are gated at the LATTICE level — the exporter honors rhythm_inference's per-beat `is_tuplet` verdict rather than re-guessing a triplet from a rounded duration (that per-note guessing produced 3210 tuplets > 4971 noteheads). Hopeful 6/4 page: rests 1.51→1.10, ties 2099→1092, tuplets 0→177, 32nds 849→2. `b7fb984`.
- **Check reconciliation, safe half (#8).** A `repeat_drift` annotation localizes disagreement to the OUTLIER repeat instance (No Pasarán: 209 notes across A/B/C; consistent D/E untouched). Annotation-only; the note-rewrite half stays deferred. `a3e50d3`.

### XIV.2 Measured negative results — walls, recorded so we don't retry
- **Instrument timbre ID (#4).** Brass vs piano is not separable by blind per-note DSP: attack (58 vs 45 ms), decay slope, bandwidth all overlap, and spectral centroid puts brass (2145) DEAD BETWEEN the two pianos (2121, 2712). Un-brassed the patch map to neutral keyboards (`5a6cf00`). Genuine family ID needs reference-template matching, not DSP.
- **Stem similarity-gated merge (#6).** No cheap pre-transcription signal separates "one instrument scattered" from "distinct instruments": mean-MFCC timbre-sim is +0.96..1.00 for ALL songs (Gospel is all-brass, so ~1.0 is even correct); onset-overlap is ~0.18 for ALL (scattered content is mutually EXCLUSIVE across stems, not duplicated). Kept always-merge (−53% dedup, validated). The only reliable signal is post-transcription note-count duplication, which needs the transcription the merge exists to avoid.
- **Global ternary/swing detection (#1).** The KDE ratio-cluster analyzer is ALREADY ported (`lattice_witness.find_ratio_clusters`) and returns `binary` for every song — pure triplets and pure binary both give IOI ratio 1.0, so ratios can't discriminate; a within-beat subdivision-occupancy test is equally flat (~chance, incl. the binary control). BUT the tuplet gate does not need it: rhythm_inference's per-beat `is_tuplet` is well-behaved (Hopeful 177 triplets, binary You Say 0.5%). If a global family is ever needed (compound meter in `resolve_denominator`), AGGREGATE the per-beat verdicts — do not build a DSP detector.

**Pattern (four negatives, same direction): demucs's separated harmonic stems do not carry cheap discriminating features — for instrument identity (#4), merge decisions (#6), or rhythmic family (#1).** The separation front is where the cheap signals die; progress there needs learned models / reference templates, not more DSP.

### XIV.3 Voice separation is upstream — confirmed by regression
The page is still gappy (Hopeful voices ~50% rests, overflow staves). Fixing it at the exporter — register-continuity voice assignment + sparse-voice merge — was MEASURED WORSE (rest ratio 1.10→1.24) and reverted: the merged harmonic stem has genuine 5+-deep polyphony, so no legal voice merge exists and the exporter can only render the polyphony it is handed. This confirms §XIII.5 empirically: voice tracking is an upstream **data-association** problem (JPDA/MHT), not a rendering fix. The ~50%-rest voices reflect REAL overlapping content. Real levers (both large): proper data-association, or a notation-specific de-merge (keep guitar/piano/other as separate thinner staves rather than the brightness-rebucketed merged families).

### XIV.4 The data-flow audit — signals computed and consumed by nobody
A producer/consumer trace of the `AnnotationStore` found analysis with no downstream:
- **`tie_candidate`** (tie_reconstruction) → carries `{tied_to_note_id, boundary_ms}`, exactly what the page needs for ties; the exporter never reads it. ★ highest-value cheap wiring.
- **`section` / `repeat_group` / `motif` / `repeat_drift`** (Check) → the entire form/structure layer dead-ends; nothing consumes song structure.
- **`duration_hypothesis`** (temporal_lattice) → symbolic durations, unread (engraver uses `notation_timing`).
- **`wobble_group`** (pitch_wobble_collapse) → unconsumed (0 on tested songs anyway).
- **`key_fit` / `acoustic_activity`** → redundant per-note copies of data that already flows via `findings.key` / the `AnechoicReport` object.
Lesson: consumption is almost entirely the **timing → engraver** path; the SYMBOLIC/STRUCTURAL signals (ties, sections, repeats) never reach the page. Part of why the page is still poor is a **wiring gap, not a missing algorithm.**

### XIV.5 Revised leverage ranking (supersedes XIII.8)
1. **Wire `tie_candidate` → the MusicXML page** — cheap, safe, already computed; real ties on the page.
2. **Voice separation via upstream data-association** (JPDA/MHT) — now confirmed as the single largest page lever and the ONLY place it can be fixed. Research-grade; its own effort.
3. **Wire Check's structure (section/repeat) into notation** — rehearsal marks, section/barline resets, and the substrate for #8's rewrite.
4. **#8 note-rewrite reconciliation (opt-in)** — consume `repeat_drift` to prefer the consensus reading; risky (real music varies), user-gated.
5. **Retire-or-wire the redundant dead-ends** — stop producing data nothing reads.
The blind-DSP items (timbre #4, merge-gate #6, global ternary #1) are **CLOSED as walls** — do not re-chase; genuine progress needs learned models / reference templates, not more DSP.

**Through-line, extended:** the detection layer has now survived a full pipeline change (the merge) without regression, and survived two hostile architecture reviews with its core laws intact. Everything still open is downstream of the notes and upstream of the page.
