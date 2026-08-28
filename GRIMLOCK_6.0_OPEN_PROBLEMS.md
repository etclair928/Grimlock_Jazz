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

## XV. Voice separation — the graph proposal, tested and triaged (2026-07-28)

An external proposal (ChatGPT) argued to replace `VoiceLine` with a `VoiceAssociationGraph`: notes as nodes, *pairwise continuation edges* carrying evidence from every witness (pitch, register, timbre, phrase, harmony, stem), voices emerging only when a consumer runs a global solver (min-cost flow / path cover / belief propagation) at query time — "delay commitment; the primitive is the relationship, not the voice label." Philosophically it's the frozen-detection/annotation law applied to a *relation*. We tested it against measurement rather than argument.

### XV.1 The experiment (Hopeful harmonic stems, 2337 chord-events)
Three-way: merge policy × algorithm, plus ChatGPT's edge-entropy diagnostic.
- **greedy first-free (current):** 9 voices, mean intra-voice pitch-jump **8.2** semitones — incoherent "voices" (whatever slot was free).
- **greedy REGISTER-CONTINUITY** (pick the closest-register free voice): 9 voices, jump **2.5** — *same voice count, coherent lines, one-line change, no solver.*
- **global min-cost path-cover:** floors at **~177 voices** no matter the terminal cost (it snaps a voice at every rest-gap >650 ms, so it physically cannot make few voices), best coherence 4.6. **Dominated on both axes by the one-liner.**
- **edge entropy 0.68** (high): even with pitch+register, most notes have several near-equal continuations — **voice is genuinely under-determined by the strong cues.** That is why no method is clean.
- **un-merging** (stem-of-origin as a prior) separates *instruments* but not voices *within* a stem (per-stem entropy 0.63-0.67).

### XV.2 Verdict
The graph is **elegant infrastructure for a problem a trivial greedy fix solves better.** ChatGPT's *coherence intuition* was right (global beats first-free on line continuity) and his *edge-entropy diagnostic* is a keeper; his *specific solver* (continuation path-cover) is refuted — it can't produce few voices — and the **"generalize to a universal Musical Relationship Graph"** step is the exact model-everything-jointly scope-creep §X/§XIII.7 reject. **Not building the graph rewrite.**

### XV.3 What was adopted
- **Register-continuity in `_events_to_voices`** (the one-liner). Coherence 8.2→2.5. HONEST TRADE-OFF, measured at the render: it **raises the rest ratio 1.10→1.24 (+~666 rests)** on Hopeful, because coherent voices honestly *rest* when their line is silent, whereas first-free crams unrelated notes into a dense staff (fewer rests, but not real voices). The rest-ratio metric penalizes honest voicing; coherence is the better readability signal — but this is a your-eye call and fully reversible.
- **`tie_candidate` wired at last** — but it turned out **superseded by consolidation**: all 39 Hopeful tie candidates (same-pitch, gap ≤40 ms) fall inside consolidation's 60 ms merge window, so consolidation already fuses them into one sustained note (better than two tied notes). Wired correctly anyway: ties are drawn only when both ends survive, so it's a **correct no-op on the consolidated page (0 ties) and a real edge in the faithful/non-consolidated view (39 ties).** No longer dead-end data; music21 also still auto-renders ~1082 cross-barline ties independently.

### XV.4 What remains for voice separation
The ceiling is **XV.1's entropy 0.68** — voice is under-determined by the cheap cues, so *no* solver (greedy, graph, or flow) gets a clean answer. Real progress needs either far stronger cues (a learned voice model / transformer) or accepting that genuinely dense polyphony is dense on the page. The graph is not the unlock; the missing information is.

**Through-line, extended:** the detection layer has now survived a full pipeline change (the merge) without regression, and survived two hostile architecture reviews with its core laws intact. Everything still open is downstream of the notes and upstream of the page.

---

## XVI. The page session — engraving, the University, and the Klangio benchmark (2026-07-30 → 2026-08-06)

*The first session aimed squarely at §XIII.4's second axis (the page). It ended
by inverting the diagnosis: the notes were never the problem.*

### XVI.0 Built and verified

- **`output/playability.py`** — two-hand physical feasibility per instant. Pure,
  reusable as an objective function.
- **`output/piano_reduction.py`** — hysteresis-Viterbi register split (treble/bass
  boundary with a switch penalty), a **≤4 rhythmic-independence voicer**, and
  gesture cohesion. Same-rhythm notes become chord tones; divergent rhythm
  becomes a voice; >4 overlap merges rather than spilling a staff.
- **Per-stem staff routing** (`build_routed_score`) — each stem its own staff;
  grand staff only where earned (keyboard/guitar timbre or the merged `other`);
  bass capped at 2 voices; drums their own staff.
- **`university/`** — Grimlock University: read-only pattern study, null-model
  grading, purpose-built melodic streaming, opt-in APPLY. OFF is byte-identical
  to Jazz.
- **Intermediate pickle + `tools/reexport_notation.py`** — notation iteration went
  from ~80 minutes to seconds. This is what made everything else here measurable.

### XVI.1 Measured negatives — myths killed, recorded so we don't retry

- **Playability is NOT the bottleneck.** Harmonic content is **86–99% two-hand
  playable**; true >10-finger impossibility is **0–1.2%**; mean simultaneity ~3.3.
  The "dense unplayable cluster dump" does not exist. **The mush is HORIZONTAL**
  (fragmentation), not vertical (density). This killed the note-dropping
  reduction the engraving design was originally built around.
- **Unbounded legato is over-sustain.** Filling every gap drove rest/note to 0.02
  but playability **97% → 33%**. Bounded legato (≤1 beat) is the keeper.
- **`pedal_point` grades NOISE** on all three songs (lift 0.34/0.72/1.3) —
  demoted. Same-pitch adjacency arises by chance in a small pitch vocabulary.
- **`rearticulation` finds 0 runs with consolidation ON, 63 with it OFF** —
  `note_consolidation` already absorbs exactly what it targets. Retained as a
  probe for whether consolidation is working.
- **Voice cohesion helps layout, not content**: gestures kept intact 34→53%
  (Hopeful), 45→56%, 34→48%; note counts unchanged ±4.

### XVI.2 NEW — the null model is a partial answer to §VI

> A detector that fires as often on **shuffled** notes as on real music is
> detecting nothing.

Destroy the structure a detector claims (pitch-shuffle for melodic, time-shuffle
for rhythmic), keep everything else, compare firing rates. **No ground truth, no
labels, ~10× cheaper than the injection–recovery harness (§IX.2)** — and it
directly decides dead / noise / real / too-loose.

It earned its place immediately by catching four things, **three of them ours**:

1. **A bug in the harness itself** — grading a *melodic* detector against a
   *time*-shuffle is meaningless (the pitch sequence survives), which masked
   `scale_run`'s real 8.65× lift as "NOISE". Each detector now declares which
   nulls actually destroy its structure.
2. **A bad detector** — `arpeggio` v1 fired **more** on shuffled pitches (0.153)
   than on real music (0.111): any three leaping notes often form some triad
   under some rotation. Tightened → REAL on all three songs.
3. **A useless detector** — `pedal_point`, demoted.
4. **Its own blind spot** — the first vertical detector (`chord`) fired 29/52/12
   times and received **no grade at all**, because the model only ran detectors
   on melodic lines. An ungraded detector had silently entered the curriculum.

**Tiering is now a measurement, not an opinion:** GRADUATED (unanimous REAL) may
alter the page; PROVISIONAL (mixed) may publish but **not move a notehead**;
FAILED is demoted.

### XVI.3 NEW — streaming was the real ceiling on pattern discovery

The curriculum read lines from `voice_continuity.stream_into_lines`, which is a
**consolidator for instrument identity, not a melodic-line finder**: any sustain
overlap starts a new line, and its 300 ms gap is absolute (a beat at 148bpm is
405 ms). Lines averaged **1.8–2.2 notes**, so any ≥4-note detector was nearly
unreachable.

A purpose-built reader (onset succession, sustain overlap tolerated,
beat-relative gap, **45 ms chord guard so block chords never read as arpeggios**)
raised notes-in-a-usable-line from **29–43% → 68–81%**, mean line 2.19 → 4.89.
**Coverage roughly doubled before a single new detector was added** — and every
detector still graded REAL, so longer lines revealed patterns rather than
manufacturing them. `voice_continuity` is untouched.

**Generalizable lesson:** we were measuring detectors against an artificially
starved input. Fix the input's *reachability* before tuning its consumers.

### XVI.4 NEW — the Klangio benchmark (the most valuable external datum to date)

Four-way comparison on *Heavy Rotation Vibez* (user-confirmed ground truth:
**piano, drums, bass, trumpet**, plus soft background horns).

**Where Klangio wins, decisively:**
- **rest/note 0.08 vs our 0.79** — a 10× cleaner page.
- **Proper unpitched percussion** (1,045 hits, `display-step`, rest/note 0.02).
  Ours emits *pitched* MIDI 36–42 on a normal staff — wrong notation, and our
  worst staff.
- A real 2-staff grand staff for piano.

**Where Klangio fails:**
- **19% of its output is duplicated** — `Guitar P1 ≡ P2` (766 notes) and
  `Bass P3 ≡ P4` (305) are byte-identical part copies.
- **Invented instruments.** It reports Guitar, Violin and Wind; none exist. Its
  "Violin" spans **MIDI 28–107 — seven octaves**, impossible for any wind or
  string instrument: a *contaminated bin* holding the trumpet plus ~112 notes
  that cannot belong to it. Its confidently-named part is the messy one; its
  vaguely-named "Wind" is the clean horn line (97.5% within trumpet range).
  **This is §XII.1's label instability, shipped as a product feature.**

### XVI.5 NEW — the baseline-choice error (the methodological trap, third bite)

Scored against **BP-on-the-mix**, our pipeline looked like 2.07× over-detection.
Scored against the **stem-sum baseline** — the standing law, *raw stems → Basic
Pitch* — the picture inverts:

| | notes vs stem-sum | sounding-time |
|---|---|---|
| **Jazz** | **1.00×** (4,500 / 4,516) | 0.87× |
| **Klangio** | **0.76×** (drops ~24%) | 1.15× |

§XII.4 recorded loudness-confounded evaluation as "a trap that bit twice." This
is the third bite in a new costume: **the baseline you choose determines the
verdict.** An arbitrary threshold compounded it — a bounded-legato variant was
flagged "OFF" at sndT/BP 1.69–1.95 while **Klangio ships at 2.68×**; our fidelity
bar was stricter than the commercial product's.

*(Honest caveat: the 1.00× is offsetting differences, not clean equivalence — the
merge removes ~1,183 harmonic notes while our drum engine adds ~600 over BP's
113, plus gains on vocals/bass.)*

### XVI.6 NEW — rest clutter is a LAYOUT property, not a transcription property

Measured across both systems, rest/note is essentially a function of
**voices-per-staff**:

| voices/staff | rest/note |
|---|---|
| 1 (Klangio) | **0.02–0.12** |
| 2 (our bass line) | 0.37 |
| 4 (our staves) | **0.79–1.15** |

Every voice must account for *all* time on its staff, so three of four voices are
resting at any instant. The linked mechanism is **chord-tone share — Klangio
31–57%, us 10–15%**: a chord tone stacks onto an existing notehead, consuming no
rhythmic slot and generating no rests. **We split into voices what Klangio stacks
into chords.**

This explains the entire 10× rest gap **without reference to note quality**, and
means the largest available win is fidelity-free: **chord-stack on the quantized
grid rather than on raw-ms proximity** (we group within 30–45 ms *then* quantize;
notes landing in the same 16th slot are already co-located on the page).

### XVI.7 NEW — our over-detection filters are inert

On *Heavy Rotation Vibez* (4,500 notes): `micro_note_purge` flagged **0**,
`schoenberg` noise/hallucination **0**, `note_support` **3.5%**. Three quality
gates, all built and wired, finding essentially nothing; dropping every flagged
note still leaves 4,343.

Per §XVI.5 we may not *need* them for fidelity — but if we ever want to reduce
for **readability**, we currently have **no working mechanism**. This is
`pedal_point`'s disease at pipeline scale: sophisticated machinery calibrated too
conservatively to ever act.

### XVI.8 The reframe — a truth machine judged as a notation engine

> Every reflex that makes us faithful — freeze the note, don't drop, don't
> invent, demand evidence before acting — is exactly the reflex that makes the
> page cluttered.

We hold ~4,500 baseline-accurate notes and render them badly. Klangio holds
~3,400 (having editorially discarded a quarter) and renders them beautifully.
**Klangio is a worse transcriber and a better engraver.** The gap is not
perception, not detection, not even voicing algorithms — it is **editorial
nerve**: the willingness to decide that a *correct* note does not belong on the
page. The frozen-note law governs the detection floor; we let it leak into
notation policy, where it does not apply and never should have.

### XVI.9 NEW — there is still no readability objective

Everything we optimize is fidelity (baseline ratios) or validity (null models).
§XIII.4 said evaluation is two axes; **we only ever built metrics for one.** No
page-quality target has ever been written down, so nothing has ever moved toward
one. Candidate axes, all cheap and already computable: rest/note, noteheads per
beat per staff, voices per staff, chord-tone share, tie count, clef/split churn.
The ear remains the oracle for musicality — but readability is measurable, and
Klangio establishes what the achievable range looks like.

### XVI.10 Revised leverage ranking (supersedes XIV.5)

1. **Chord-stack on the quantized grid** (§XVI.6) — the single largest page win,
   **zero fidelity cost**. Target chord share 15% → ~40%, cascading into fewer
   voices and fewer rests.
2. **Proper unpitched percussion notation** (§XVI.4) — free, unambiguous, and
   currently our worst staff.
3. **Define the readability objective** (§XVI.9) — without it, 1 and 4 cannot be
   scored, and we will keep optimizing correctness by default.
4. **Voices-per-staff policy: cap at 2, spill to more staves** (§XVI.6) — an
   honest trade (taller score), now priceable against a benchmark.
5. **Recalibrate the fidelity guardrail** (§XVI.5) — thresholds must be
   benchmarked, not invented; there is measured headroom for moderate sustain.
6. **Meter correctness as a cleanliness lever** — we read HRV as 107bpm 6/4,
   Klangio as 100bpm 4/4 (far more plausible for the material). Wrong barlines
   manufacture ties and rests, so a meter error costs the page twice.
7. **Instrument naming without double-counting** — transcribe the merged stem (no
   duplication), then *label* notes by per-stem energy at their own f0. This
   would beat Klangio outright: names, with no phantom Violin and no duplicate
   parts.
8. **Voice separation via stronger cues** — unchanged from §XV.4: entropy 0.68 is
   the ceiling; the missing information, not the solver, is the blocker.
9. **Injection–recovery harness** (§IX.2) — still unbuilt, but §XVI.2 shows a
   cheaper falsification instrument exists for anything detector-shaped.

**Through-line, extended:** the detection layer has now survived the merge, two
hostile audits, an engraving rebuild, and a head-to-head against a commercial
product — and against the correct baseline it sits **at parity**. Everything
still open is downstream of the notes and upstream of the page, and this session
finally established which of those two the remaining work belongs to: **the page,
and specifically the willingness to edit it.**

---

## XVII. Same-recording head-to-head vs Klangio — and a key-detection bug (2026-08-06)

*§XVI compared us to Klangio on one song we had transcribed and it had not. Then
four more Klangio exports arrived, two of them **the identical source recordings
we run** (Hopeful, prospering) and one a **different performance of a song we
had transcribed** (Prospering (Cover)). That combination is a genuinely new
evaluation instrument, and it produced the first hard bug this whole comparison
has found — in us.*

### XVII.0 NEW — two ground-truth-free evaluation instruments

1. **Same-recording head-to-head.** Both systems consume identical audio, so
   *every* difference is attributable to the system. No performance variance, no
   separation variance.
2. **Cross-version agreement.** Two different *performances* of the same song,
   transcribed by two different *systems*. Convergence on key, progression and
   pitch-class profile is evidence both are right — with no labels.

Together with the null model (§XVI.2), that is now **three** partial answers to
§VI's "no accessible loss function," none of which needs ground truth. The null
model falsifies *detectors*; these two corroborate *readings*.

### XVII.1 Content agreement is HIGH — the systems validate each other

| comparison | pitch-class correlation |
|---|---|
| Hopeful — **same recording**, ours vs Klangio | **r = +0.975** |
| Prospering — **different recording**, ours vs Klangio's cover | **r = +0.948** at zero transposition |

The cover comparison peaks at **+0.948 with no transposition** (runner-up +0.806
at +5 semitones, which is only the subdominant sharing 6 of 7 tones). Bass roots
match too — ours A♭/F/E♭/D♭, the cover's E♭/A♭/F/D♭: the same **I–vi–V–IV** in
A♭ major. **Our transcription captures the song's identity, not merely the
audio** — an independent performance transcribed by an independent system lands
on the same harmonic DNA.

### XVII.2 NEW BUG — key detection is wrong on both songs, and our own notes know better

Krumhansl key-fitting run on each system's **own transcribed notes**:

| song | our reported key | our NOTES say | Klangio reported | Klangio's NOTES say |
|---|---|---|---|---|
| prospering | **E** | **G♯/A♭ major** (r=+0.888) | A♭ major | A♭ major |
| Hopeful | **Bm** | **A♯/B♭ minor** (r=+0.80) | A♭ major | A♯/B♭ major/minor |

- **prospering:** 98.9% of our sounding time lies inside the A♭-major scale, every
  non-scale tone under 0.6% — and we labelled it **E**. Klangio and our own notes
  agree on A♭; only our *label* dissents.
- **Hopeful:** both systems' notes point to **B♭ minor** (G♭ outweighs G in both
  profiles, the B♭-minor signature). We reported **B minor — a semitone sharp.**
  Klangio reported A♭ major, which shares 6 of 7 tones but misses root and mode.

**Mechanism:** `analyze_key` reads **chroma from the full mix**, where drums and
percussion smear the profile. Our *transcribed notes* are 90–99% diatonic — a far
cleaner signal that we already compute and then ignore. **Detect key from our own
notes** (or fuse notes-key as a witness against mix-chroma-key). Small, and now
validated on two songs against an independent system.

This is §XVI's thesis again in a new place: **we compute better evidence than we
consume.**

### XVII.3 Meter — Hopeful CORROBORATED, Heavy Rotation Vibez is ours to fix

- **Hopeful:** we read **148bpm 6/4**; Klangio reads **3/4**. These are the *same
  pulse* — 6/4 is two 3/4 bars. Our meter is **corroborated**, and the §XVI
  suspicion that our 6/4 readings are systematically wrong is **withdrawn**.
- **Heavy Rotation Vibez:** we read **6/4**, Klangio **4/4**. Those are *not*
  compatible, and 4/4 is far more plausible for the material. That one is a
  genuine error, and it stays on the leverage list (§XVI.10 item 6).

The lesson for the referee: a meter "disagreement" must be checked for
*metrical equivalence* before being scored as a conflict.

### XVII.4 DECISIVE — rest clutter is layout, proven on identical audio

On **Hopeful**, from the same recording:

| | unique noteheads | rest/note | staves × voices | chord-tone share |
|---|---|---|---|---|
| **Jazz** | 5,495 | **0.79** | 5 staves × up to 4 voices | 10–15% |
| **Klangio** | **~8,880 (1.6× MORE)** | **0.10** | 10 staves × mostly 1 voice | 41–54% |

**Klangio emits 1.6× more notes than we do and still has eight times fewer rests
per note.** This eliminates every remaining explanation involving note quality,
over-detection or density. Rest clutter is **purely a layout policy**: voices per
staff, and whether simultaneities are stacked as chord tones or split into
voices. §XVI.6 hypothesised this; this measurement proves it on identical input.

### XVII.5 Klangio's failure mode is systematic — four files, one pattern

Across all four exports (Heavy Rotation Vibez, Hopeful, prospering, Prospering
(Cover)):

- **Duplicate parts, every single time** — `Guitar P≡P` and `Bass P≡P`,
  byte-identical note sequences. 19% of its output on HRV.
- **Phantom instruments, every time.** On prospering (ground truth **Bass, Piano,
  Trumpet, Drums**) it emits **Piano, Guitar, Violin, Wind, Synth** — four of five
  do not exist. The single trumpet is **fragmented across Violin (652 notes,
  midi 35–89, 77% coverage), Wind (696, 41–94, 81%) and Synth (420, 41–94, 74%)**:
  three parts covering nearly the same register, each partial.
- **This is why the trumpet "feels dropped"** to a reader — not because content is
  missing (Klangio's harmonic total is **3,661 vs our 1,879, 1.95× MORE**), but
  because no single staff holds a coherent trumpet line.
- On HRV its "Violin" spans **seven octaves (midi 28–107)** — a contaminated bin,
  physically impossible for any one instrument.

**This is exactly §XII.1's separator label instability, shipped as a product
feature.** Klangio did not solve the identity problem we merge stems to avoid; it
committed to confident names on top of it. Our refusal to name is *more honest*
— and §XVI.10 item 7 (transcribe merged, then label by per-stem energy at each
note's own f0) would let us name defensibly and beat it outright.

### XVII.6 Bass — they drop 38%, we over-range

Same recording, prospering:

| | notes | range | time coverage |
|---|---|---|---|
| **Jazz** | **605** | midi **27–77** | **100%** |
| **Klangio** | 376 (**keeps 62%**) | midi **34–55** | 94% |

Klangio drops **38%** of the bass, and its 3 missing windows are all in the first
15 s (plausibly a genuine bass-free intro). But its range is **tight and
realistic** (B♭1–G3) while **8 of our bass notes sit above G4** — implausible for
a bass instrument.

So: **we are more complete, they are more disciplined.** Symphony carried a
`FAMILY_PITCH_RANGE_MIDI` plausibility check for exactly this; **Jazz has no
equivalent.** Porting it is cheap, is annotation-only, and closes a defect this
comparison made visible.

### XVII.7 What the head-to-head establishes

Stated plainly, because it settles what the remaining work actually is:

> **We are the better transcriber. Klangio is the better engraver.**

- **Content:** we are at parity with the stem baseline (§XVI.5, 1.00×), we agree
  with an independent system at r=0.975 on identical audio, and we agree with an
  independent *performance* at r=0.948. Our content is sound.
- **Identity:** Klangio invents instruments and duplicates parts on every file;
  we invent nothing. Our restraint is correct, but it currently costs us all
  labelling — which is a solvable problem, not an inherent trade.
- **Page:** Klangio wins decisively and for reasons that have **nothing to do
  with hearing** — more staves, fewer voices per staff, aggressive chord
  stacking, and proper unpitched percussion.
- **Labels:** both systems get key wrong on the same material while both systems'
  *notes* are right. Nobody's reported key should be trusted over its own notes.

**Everything Jazz still needs is downstream of the notes.** The detection layer
has now been externally corroborated twice; the remaining work is the page, the
labels, and the discipline to use evidence we already produce.

### XVII.8 Additions to the leverage ranking (extends XVI.10)

- **NEW 2a. Detect key from our own notes** (§XVII.2) — a real bug, wrong on 2/2
  songs, with the better signal already computed. Cheap and immediately testable
  against both prospering recordings and Hopeful.
- **NEW 6a. Port `FAMILY_PITCH_RANGE_MIDI` range plausibility** (§XVII.6) —
  annotation-only, catches our out-of-range bass notes.
- **Item 6 refined:** meter errors must first be tested for *metrical
  equivalence* (6/4 vs 3/4 is agreement, not conflict). HRV's 6/4-vs-4/4 remains
  a genuine error.
- **Item 7 strengthened:** naming is no longer a "nice to have." Klangio proves
  the market expects instrument names *and* that naming badly (phantom Violin,
  a trumpet split three ways) is worse than not naming. Labelling merged content
  by per-stem energy is the one place we could straightforwardly surpass it.

### XVII.9 FIXED (2026-08-06) — the key-detection bug was worse than diagnosed

§XVII.2 blamed the *input* (mix chroma). Building the notes-based path
disproved that: it returned the **same wrong answers**, which meant the fault
was in `detect_key` itself. Two independent bugs, both now fixed:

1. **Index-vs-pitch-class.** `KEYS_MAJOR`/`KEYS_MINOR` are in circle-of-fifths
   order (deliberately - index *i* is a relative major/minor pair), but
   `_correlate_with_key` did `np.roll(profile, -key_idx)`, a *chromatic* shift.
   Passing the list index tested the wrong profile whenever index != pitch
   class: **6 of 12 major keys were mislabeled by a TRITONE** (G, A, B, Db, Eb,
   F) and **all 12 minor keys were wrong** (off by 3 or 9 semitones).
2. **Rotation sign.** The profiles are tonic-first and chroma index 0 = C, so
   the tonic must be rolled TO the pitch class (`+pc`); `-pc` placed it at
   `12-pc`, correct only for pc 0 and 6. Caught by a synthetic test: a *pure
   A-flat-major profile* was reported as "E".

**Verification.** A 24-key synthetic round-trip (build each key's pure profile,
require `detect_key` to name it back) now passes **24/24**; before the fix it
passed 1. On real material, against the independent Klangio reading:

| song | before | after | independent check |
|---|---|---|---|
| Hopeful | Bm | **Bbm** | notes + Klangio |
| prospering | E | **Ab** | notes + Klangio |
| Heavy Rotation Vibez | B | Em | (Klangio says Ab - unresolved) |

**Also wired:** `analyze_key_from_notes` / `chroma_from_notes` build a real
12 x N chromagram from transcribed notes (drums excluded), so `detect_key`'s
edge-weighting and relative-key tie-breaker keep working. The Conductor now runs
it as a **second witness** after transcription and takes the more confident
reading, logging both. Guided key remains a hard lock (§2.7).

**Lesson recorded:** a synthetic round-trip test would have caught this the day
the module was written. `detect_key` was described in its own docstring as "the
pure, testable core" - and had no test.

### XVII.10 FIXED (2026-08-06) — pitch-range plausibility ported

`instrument_attribution/range_check.py`, annotation-only, keyed by **stem** (not
the brightness-bucket family, which cannot support a range claim). Generous
bounds: bass **23-67**, vocals **36-88**; the merged harmonic stem is left
unconstrained on purpose.

Measured: **3 / 70 / 100** implausible notes on Hopeful / prospering / HRV. The
bass flags are the ones §XVII.6 predicted (8 on prospering, 34 on HRV, up to
midi 83). The vocals flags turned out to be **low**, not high - 51 and 61 notes
**below C2**, with minima at **midi 21 and 27 (A0, D#1)**. No human voice
produces those: that is bass bleed into the vocals stem or an octave error, and
it is a second defect this check surfaces for free. The high vocal bound was
loosened 84 -> 88 after measuring that only 1-2 notes per song exceed it (real
falsetto/harmony, not artifacts).

Nothing is dropped - the verdict exists for the engraver or a future reducer,
which is precisely the "working mechanism" §XVI.7 said we lacked.

### XVII.11 BUILT (2026-08-06) — the readability objective finally exists

`output/readability.py` scores an emitted MusicXML on six axes, each with a
reference value measured from the Klangio head-to-head rather than invented:
rest ratio, chord share, max voices/staff, tempo marks, unpitched drums,
overfull measures. §XIII.4 named this second axis; §XVI.9 recorded that we had
never built a metric for it. It exists now, and it earned its keep within the
hour by falsifying the very next thing we tried.

It also caught that **Klangio emits 7-9 tempo marks per score** - the same
system-level-object-per-part sloppiness we had, so that defect is industry-wide,
not ours alone. (Ours is fixed: 5 -> 1.)

### XVII.12 MEASURED NEGATIVE — chord-stacking on the quantized grid does NOT work

§XVI.10 ranked "chord-stack on the quantized grid" as leverage **#1**, "the
single largest page win, zero fidelity cost," predicting chord share 15% -> 40%.
Implemented at BOTH ends - `musicxml_exporter._chord_events_gridded` and the
voicer's own `piano_reduction._chord_events` (grouping by snapped onset+end
instead of raw 40 ms / 90 ms tolerances). Measured on prospering:

| variant | rest/note | chord share |
|---|---|---|
| raw-ms grouping (before) | 0.876 | 0.180 |
| grid grouping, exporter | 0.876 | 0.181 |
| grid grouping, voicer | 0.884 | **0.172** |

**No improvement; marginally worse.** The prediction was wrong. Grouping
tolerance was never the binding constraint - by the time the exporter groups,
`piano_reduction.assign_voices` has already committed each note to a voice, and
the exporter can only chord-group *within* a voice that is already decided.

The code is kept (it is more principled than raw-ms proximity and is a no-op
when tempo is unknown), but it is **not** the lever, and leverage item #1 is
hereby demoted.

### XVII.13 CONFIRMED — voices-per-staff IS the lever (leverage #4 promoted to #1)

The same mechanism §XVI.6 identified, attacked from the correct end. Sweeping
`max_voices` on prospering:

| cap | rest/note | chord share | notes |
|---|---|---|---|
| 4 (current default) | 0.884 | 0.172 | 3,626 |
| 3 | 0.748 | 0.177 | 3,637 |
| **2** | **0.560** | 0.202 | 3,679 |
| 1 | 0.345 | 0.269 | 3,631 |

Monotonic, and it generalises - cap 4 -> 2 across all three songs:

| song | before | after | change |
|---|---|---|---|
| prospering | 0.884 | 0.560 | **-37%** |
| Hopeful | 0.953 | 0.666 | **-30%** |
| Heavy Rotation Vibez | 0.793 | 0.489 | **-38%** |

Note counts stay flat (within 1%): **no content is lost, only re-laid-out.**
Chord share *rises* as the cap falls, because fewer voices forces simultaneities
to merge into chords - so the chord-share goal of §XVII.12 is achieved as a
by-product of the voice cap, not by tuning grouping windows. **The two levers
were always one lever.**

**Recommended default is cap = 2, NOT the best-scoring cap = 1.** Cap 1 scores
better on every axis but forces every simultaneity into a block chord, which
destroys the melody-over-held-chord independence that makes syncopation
readable - the exact failure mode rejected in the engraving doc §13.2. Cap 2
halves the rests while preserving genuine two-voice piano writing. This is a
visible change to every page, so it is left as an explicit parameter pending an
eye/ear check rather than silently switched.

**Method note worth keeping:** the readability metric was built *before* the
change it was meant to score, and it immediately falsified the author's own
top-ranked hypothesis. That is the null model's lesson (§XVI.2) applied to
engineering priorities rather than detectors.

### XVII.14 The page-fix session (2026-08-06) — five wins, four negatives

Driven almost entirely by the user's eye and ear on real exports. Every item
below is measured on prospering unless noted.

**WINS (shipped):**

| fix | before | after |
|---|---|---|
| key detection (2 real bugs, §XVII.9) | Bm / E | **B-flat m / A-flat** - both now match independent evidence |
| voices per staff (§XVII.13) | 4 | **2** |
| tempo marks (system object emitted per part) | 5 | **1** |
| drum notation | pitched MIDI 36-42 | **`<unpitched>` + percussion clef**, kick F4 / snare C5 / hi-hat G5 + x noteheads |
| sustain (user chose "C_beat" by ear) | dry, fill=0 | **fill=1.0 beat** |
| **rest/note (combined)** | **0.876** | **~0.41 (-53%)** |

`output/readability.py` (§XVII.11) is what made all of this scoreable.

**NEGATIVES (recorded so they are not retried):**

1. **Chord-stacking on the quantized grid does nothing** (§XVII.12). Predicted
   as leverage #1; delivered 0.876 -> 0.884.
2. **"One grid per measure" for tuplets is a REGRESSION** - reverted. Forcing a
   measure containing any tuplet beat wholly onto thirds pushed binary material
   onto k/3: prospering nonsense ratios 4 -> 7 and bad measures 0 -> 3; Hopeful
   exploded to 1502 triplets, nine distinct nonsense ratios, 22 bad measures.
3. **Metric-boundary fill is metric-negative.** The user's rule ("a figure
   landing on the AND shouldn't leave a rest to the barline - write the 8th
   out") is musically right and IS implemented, but rest COUNT rose 1790 ->
   1837 and sustain 1.61x -> 1.74x raw. Mechanism: a longer note holds its
   voice longer, so the next note cannot reuse that voice -> more voices ->
   more rests. **Kept anyway**, because rest-count is the wrong metric for the
   request - it measures how MANY rests exist, while the complaint is about
   rest DURATION after a note. Left ON pending the ear.
4. **Chord collapse has no measurable effect.** Requiring an exact end-cell
   match really did tear ragged block chords into separate voices, and the fix
   is unit-tested both ways (ragged chord -> 1 slot; 16th-vs-whole -> 2 slots).
   But on real data chord share moved 0.180 -> 0.182: by the time the voicer
   runs, `assign_hands` has already split chords across the two staves and
   consolidation has merged what it could. Correct code, wrong assumption about
   where theproblem lived.

**Side effect worth knowing:** the chosen sustain raises notehead count 3,627 ->
4,519 (+25%) because longer notes cross barlines and split into tied notes. The
page is smoother and *denser* at once.

**Residual tuplet nonsense (4 notes of 2,773 on prospering: 12:11, 12:7).**
Three exporter-side attempts failed. 12 = lcm(3,4), so these arise where
rhythm_inference's PER-BEAT tuplet verdicts put k/3 and k/4 positions in one
measure. **The fix belongs upstream in tuplet detection, not in the exporter** -
stop patching the symptom.

**Method note.** Four of the user's page observations this session ("choppy",
"events don't line up", "why the red boxes", "block chords should be one voice")
each turned out to name a REAL defect - two of them bugs no metric had flagged
(the per-part tempo marks; the two different sustain policies applied to
different staves in the same score). The eye is finding defects faster than the
metrics are. The metrics' job is to stop us shipping a fix that does not work -
which they did, four times today.

### XVII.15 ROOT CAUSE — why triplets read as 16th pickups, and what ReverseGeoCrypt can actually see

*User observation (2026-08-06): "tuplets and triplets get interpreted as 16th
note pickups... ReverseGeoCrypt was designed to figure these sorta things out so
I thought." Traced. The instinct is right that this is where the answer lives;
the conclusion is that **ReverseGeoCrypt is structurally incapable of detecting
an evenly-played tuplet**, and has never been the thing producing our triplets.*

**What it actually measures.** `lattice_witness._compute_ratios` compares
**consecutive IOIs to each other** (`ioi[i+1] / ioi[i]`, plus the i+2 skip),
KDE-peaks the resulting ratios, and matches peaks against family ideals:

```
binary:     [1.0, 2.0, 0.5, 4.0, 0.25]
ternary:    [1.3333, 0.6667, 0.3333, 2.6667]
swing:      [1.5, 0.6667, 3.0]
quintuplet: [1.25, 0.8, 2.5]
septuplet:  [1.1429, 0.875, 2.2857]
```

**The structural blindness.** Three *evenly played* triplets have IOI ratios of
**1.0, 1.0** - each note equally spaced from the last. And `1.0` is the FIRST
ideal under **binary**. So an even triplet is mathematically indistinguishable
from even eighths under this test. A consecutive-IOI-ratio measure can only
detect **uneven** figures (swing 1.5, dotted 3.0); it is blind to even tuplets by
construction - and even tuplets are most of the triplets in this repertoire.

**Two aggravating implementation details:**

1. The scoring loop `break`s on the first matching family and `binary` is first
   in the dict, so binary gets first claim on every peak near 1.0 / 2.0 / 0.5 -
   the most common ratios in any music.
2. `0.6667` appears in **both** ternary and swing, but ternary is checked first,
   so **swing can essentially never win on that ratio**.

**The evidence matches exactly.** Across the seven saved intermediates,
`ratio_family` is **`binary` on six** and **`quintuplet` on one**
(`A_Strangers_Smile`, a 62bpm 6/4 worship track - almost certainly wrong;
quintuplet's 1.25 and ternary's 1.3333 differ by less than
`RATIO_MATCH_TOLERANCE = 0.1`, so they collide). **Never once ternary or swing.**

**How that becomes "16th note pickups" on the page.** Three evenly-spaced onsets
in a beat -> ratios 1.0 -> classified binary -> the notation quantizer snaps them
to the nearest **16th** grid -> 0, 1/3, 2/3 lands as 0, 0.25, 0.5/0.75. That IS a
16th-note pickup figure. The user is hearing the classifier's blind spot
rendered as notation.

**Where ratio_family is actually consumed** (modestly, and never destructively):
- `resolve_denominator(numerator, ratio_family)` and `estimate_time_signature` -
  simple vs compound meter.
- `musicxml_exporter`: `ratio_family in {ternary, swing}` can only **ADD**
  triplet permission, never remove it (that gate was deliberately left un-ANDed,
  §XIV, precisely because it reads binary everywhere).

So for tuplets it is effectively **inert**. Every triplet we currently emit comes
from `quantization.rhythm_inference`, which does the structurally right thing:
it evaluates a **whole beat** against filling templates - including
`BeatFilling("eighth_triplet", (0.0, 1/3, 2/3), is_tuplet=True)` - with a +3
complexity penalty so a tuplet must clearly out-explain the binary fillings.

**The reframe.** Tuplet detection must be **beat-relative, not ratio-relative**:
the question is *"how many onsets fall inside this beat, and at what phase?"* -
three evenly-spaced onsets in a beat is a triplet regardless of what consecutive
ratios say. ReverseGeoCrypt should be understood and used as an **uneven/swing
detector and a lattice-period witness**, NOT as a tuplet detector. Its
`ratio_family` output should not be treated as tuplet evidence at all.

**This also explains the residual nonsense tuplet ratios** (§XVII.14: 12:11,
12:7, 4 notes of 2,773). They appear exactly where `rhythm_inference`'s per-beat
verdict says triplet while neighbouring beats say binary, putting k/3 and k/4
positions in one measure (12 = lcm(3,4)). Three exporter-side attempts to fix
that failed because **the exporter is the wrong layer** - it is faithfully
rendering a mixed-grid decision made upstream.

**Consequences for the leverage ranking:**
- **Do not** gate anything further on `ratio_family` as a tuplet signal; it
  cannot carry that load. (Its meter use is fine and stays.)
- The honest tuplet lever is **strengthening `rhythm_inference`'s per-beat
  verdict** (its confidence, and consistency between adjacent beats), not
  post-hoc repair in the exporter.
- A swing/uneven detector is the one thing ReverseGeoCrypt's ratio test *is*
  suited to - and `swing_ratio` is already measured independently by
  `rhythm_engine.estimate_groove`, so the two should be cross-checked rather
  than both trusted blindly.

---

## XVIII. Project audit — what is resolved, what is dead, where the synergy is (2026-08-06)

*A real audit: import graph from the Conductor, function-level call analysis,
and an annotation write/read trace across all 95 modules.*

### XVIII.0 RETIRED — problems that are now actually resolved

| was | status |
|---|---|
| **§XVI.9** "no readability objective exists" - the §XIII.4 second axis, unbuilt for months | **RESOLVED.** `output/readability.py`: 6 axes (rest ratio, chord share, max voices/staff, tempo marks, unpitched drums, overfull measures), each benchmarked against Klangio rather than invented. |
| **§XVII.2** key detection wrong on 2/2 songs | **RESOLVED.** Two real bugs in `detect_key`: the profile was rotated by LIST INDEX instead of pitch class (6/12 major keys off by a tritone, all 12 minor keys wrong) and the rotation SIGN was inverted. Synthetic round-trip 1/24 -> **24/24**. Hopeful Bm -> **B-flat m**, prospering E -> **A-flat**, both matching Klangio and our own note content. |
| **§XVI.10 item 2** drums notated as pitched MIDI 36-42 | **RESOLVED.** `<unpitched>` + percussion clef, kick F4 / snare C5 / hi-hat G5, x noteheads - the same placement Klangio produced on identical audio. |
| **§XVI.10 item 4** voices-per-staff policy | **RESOLVED.** Default `max_voices=2`. rest/note **0.876 -> ~0.41 (-53%)** with note counts flat. |
| tempo marks (system object emitted per part) | **RESOLVED.** 5 -> 1. (Klangio emits 7-9; the defect is industry-wide.) |
| **§XVI.10 item 1** chord-stack on the quantized grid | **RETIRED AS REFUTED**, not solved (§XVII.12). Implemented at both ends, delivered 0.876 -> 0.884. Demoted off the leverage list. |
| **§XVII.6** instrument-range plausibility absent in Jazz | **PORTED** (`range_plausibility`) - but see XVIII.2: it is written and never read. |

### XVIII.1 DEAD CODE: essentially none

Function-level analysis flagged 20 public functions as "never called outside
their own module." **19 are false positives:**

- **Same-file internal APIs** - `build_music21_score`, `assign_hands`,
  `two_hand_feasible`, `build_timeline`, `search_lattice`, and all six
  `schoenberg_mirror.analyze_*` helpers (each called by `audit_note`).
- **Dynamic dispatch** - the 10 `university/detectors.py` entries are invoked
  through the `DETECTORS` registry dict, which no regex can see.

**Exactly one genuinely unused function: `university/streams.py::line_stats`** -
a diagnostic helper written during the streaming measurement. Harmless; keep or
delete.

**No unreachable modules.** All 95 are reachable from `orchestration/conductor.py`.

This is worth stating plainly because it contradicts the usual expectation for a
codebase this size: **Jazz has no dead-code problem.** What it has is the
opposite - live code whose *output* nobody consumes.

### XVIII.2 THE REAL FINDING — six annotation kinds are WRITTEN AND NEVER READ

This is §XIV.4's "signals computed and consumed by nobody," now quantified by
tracing every `ANNOTATION_KIND` from writer to reader:

| annotation | written by | read by | note |
|---|---|---|---|
| `acoustic_activity` | conductor (AnechoicMa) | **nobody** | frame-level silence / resonance / activity probability |
| `key_fit` | conductor (Key Intelligence) | **nobody** | per-note in-key membership + weight |
| `duration_hypothesis` | conductor (duration_witness) | **nobody** | symbolic duration + probability |
| `range_plausibility` | conductor | **nobody** | just ported; MuseScore already draws these red for us |
| `wobble_group` | conductor (PitchWobbleCollapse) | **nobody** | vocals-only, deliberately parked |
| `pattern_study` | University | **nobody** | by design - it is the corpus |

Everything else (`instrument_family`, `consolidation`, `notation_timing`,
`onset_refinement`, `sustain_extension`, `tie_candidate`, `note_support`,
`harmonic_legitimacy`, `legitimacy_verdict`, `quantization`, `voice`,
`pattern_voice_cohesion`, `pattern_page_suppress`) has a real consumer.

So: **five live evidence streams, produced every run, that change nothing.**

### XVIII.3 WHERE THE SYNERGY IS — three connections worth making

Ranked by how directly they solve a problem we already have.

**1. `acoustic_activity` -> gate the legato fill.** *(highest value)*
This session set `fill_max_beats` **by ear**, blanket-filling every gap under one
beat. But AnechoicMa already measures, per time window, whether the audio is
genuinely silent or still resonating. That is exactly the missing evidence: fill
a gap when the stem is still ringing, leave it as a rest when it is truly
silent. It converts our one remaining hand-tuned aesthetic threshold into a
measured decision, and it is the honest version of the "more sustain" fix the
user asked for (§XVII.14 negative #3) - the same goal, evidence-driven instead
of blanket.

**2. `range_plausibility` -> stem re-routing, not just flagging.**
The check flags 61 out-of-range notes on the vocals staff and 10 on bass. The
vocals outliers are **sub-C2, down to A0** - that is not a vocalist, it is bass
bleed sitting in the wrong stem. MuseScore already colours these red on the page,
so an external tool is doing our validation for us. Routing a range-implausible
note to the staff whose range it *does* fit would fix a real misattribution
rather than merely annotating it.

**3. `key_fit` -> enharmonic spelling.**
Spelling (F-sharp vs G-flat) was identified as a required sub-problem in the very
first engraving design and never built; music21 currently guesses. `key_fit`
already computes per-note key membership against a now-CORRECT key (XVIII.0), so
the input exists. This is the cheapest path to accidentals that read properly.

*(`duration_hypothesis` could cross-check the notation quantizer; `wobble_group`
stays parked per standing user decision.)*

### XVIII.4 State of Grimlock Jazz

**Detection layer: externally corroborated, at parity.** Against the correct
stem-sum baseline we sit at **1.00x** while Klangio is 0.76x; on identical audio
the two systems agree at **r=0.975** on pitch-class content, and across two
different performances of the same song at **r=0.948**. The frozen-note law and
the merge have survived every audit. *Nothing in detection is the bottleneck.*

**Page layer: much improved, still behind.** rest/note **0.876 -> ~0.41**, max
voices 4 -> 2, real percussion notation, one tempo mark, 0 overfull measures on
prospering. Klangio still reads cleaner (0.06-0.10) - and §XVII.4 established
that the remaining gap is **staff count** (they use 9-10 single-voice staves; we
use 5), which is a structural choice about page shape, not a threshold.

**Labels: one bug fixed, one gap open.** Key is now correct. Instrument naming
remains deliberately absent - and the Klangio comparison vindicated that
restraint (it invents Guitar/Violin/Wind/Synth on every file and duplicates two
parts every time), while also showing the opportunity: label the merged content
by per-stem energy and we would name defensibly where they cannot.

**Tuplets: root-caused, not fixed** (§XVII.15). ReverseGeoCrypt structurally
cannot see an even tuplet; `rhythm_inference`'s per-beat verdict is the only real
tuplet evidence and is where the work belongs.

**Tooling.** `tools/` has grown 10 scripts with real overlap -
`run_hopeful_full.py` is now subsumed by `run_full.py`, and
`run_hrv_analysis.py` / `run_hrv_pipeline_only.py` are single-song scaffolding.
Consolidating to `run_full.py` + `reexport_notation.py` + the three measurement
tools (`readability`/`playability`/`university_report`) would cut the surface
without losing capability.

### XVIII.5 Revised leverage ranking (supersedes XVI.10 / XVII.8)

1. **Wire `acoustic_activity` into the legato gate** (XVIII.3 #1) - replaces the
   last hand-tuned aesthetic threshold with measured evidence.
2. **Wire `range_plausibility` into stem routing** (XVIII.3 #2) - fixes real
   misattribution; MuseScore is already showing us the errors.
3. **Strengthen `rhythm_inference`'s per-beat tuplet verdict** (§XVII.15) - the
   only honest tuplet lever; stop patching the exporter.
4. **More staves for harmonic content** - the entire remaining rest gap
   (§XVII.4). A page-shape decision that needs the user's eye, not a metric.
5. **`key_fit` -> enharmonic spelling** (XVIII.3 #3) - now unblocked by the key fix.
6. **Instrument naming by per-stem energy** - the one place we could clearly
   surpass Klangio.
7. **Injection-recovery harness** (§IX.2) - still unbuilt; the null model
   (§XVI.2) covers anything detector-shaped more cheaply.

**Through-line:** the codebase is not carrying dead weight - it is carrying
**unconsumed evidence**. Five live signals change nothing today, and three of
them map directly onto problems we are currently solving by hand-tuning. The
next real gains are wiring, not building.

---

## XIX. Retrospective — the proposal that started this arc, measured against what happened (2026-08-07)

*The whole engraving arc began with a pasted essay arguing that Grimlock's
problem was no longer DSP but **musical organization** - that we had left Music
Information Retrieval and entered **Computational Music Engraving**, and that the
missing piece was a "Musical Structure Discovery" pass between quantization and
voice separation, whose job was **semantic compression**: notes -> patterns ->
voices, optimizing not "recover every note" but "produce the score a copyist
would write."*

*We then spent the arc actually testing it. Recording the result, because the
scorecard is unusual: the diagnosis was right, the prescription was not, and the
one claim it made most confidently is the one we falsified hardest.*

### XIX.1 What it got right - confirmed independently

- **"The problem is organization, not separation."** Quantified: against the
  stem-sum baseline we sit at **1.00x** while Klangio is 0.76x, and on identical
  audio the two systems agree at **r=0.975** on pitch-class content. The
  detection layer is externally corroborated. §XVI.8 reached the same sentence
  from the opposite direction - *a truth machine judged as a notation engine*.
- **"You need an explicit cost function for engraving."** Built
  (`output/readability.py`, §XVII.11), benchmarked against a shipping product
  rather than invented. It was the single most useful artifact of the arc,
  because it began **falsifying the author's own hypotheses within the hour**.
- **"Three different optimization problems have been conflated"** (acoustic
  accuracy / musical analysis / engraving compression). This framing is sound
  and is now how the leverage list is organized.

### XIX.2 What we measured that CORRECTS it

**1. The prescribed cure underdelivered.** The Musical Structure Discovery pass
was built - it is Grimlock University. It discovers scale runs, arpeggios,
sequences, neighbour tones and ostinati with null-model-validated detectors, at
22-25% coverage. In APPLY mode it moved **7 noteheads out of ~4,000**. Gesture
cohesion improved intactness 34% -> 53%: real, but local.

Meanwhile **rest/note fell 0.876 -> ~0.41** from three blunt, non-musical
changes: cap voices per staff at 2, notate drums as unpitched percussion, and
set a sustain policy.

Most pointedly, the essay's own example - *"instead of four voices, ask: can this
be one chord?"* - was implemented **twice** (exporter and voicer) and delivered
**0.876 -> 0.884, i.e. nothing** (§XVII.12). What actually merged chords back
together was capping the voice count, which forced merging as a side effect.

> **The pattern layer is real and worth keeping. It was not the breakthrough.
> Layout policy was.**

**2. Two terms in its proposed cost function are in direct opposition.** It
rewards *"long continuous melodic streams"* while penalizing *"excessive
rests."* Measured (§XV.3): coherent voices **honestly rest** when their line is
silent, so register-continuity voicing RAISED the rest ratio 1.10 -> 1.24. And
§XVII.13 showed rest count is largely a *function of voices-per-staff*, not an
independent axis. These cannot both be maximized; the objective has to choose.

**3. The claim we falsified hardest:** *"Notice what's missing from that list:
audio. At this stage you're no longer doing signal processing."*

The arc's best single fix was wiring audio back **in**. The choppiness the user
heard was attacked twice with thresholds and got wrong both times; the actual fix
was consulting AnechoicMa's resonance-vs-silence evidence to decide whether a gap
is a real rest or a cut-off artifact (§XVIII.3 #1). The engraving layer needed
*more* acoustic evidence, not less. Signal processing had not stopped being
relevant - it had stopped being **consulted** (five annotations written and never
read, §XVIII.2).

### XIX.3 The natural experiment nobody planned: Klangio

Klangio is this essay's thesis shipped as a product - aggressive simplification,
confident musical objects, semantic compression. Measured against it on identical
audio (§XVII): it reads **10x cleaner** (rest/note 0.06-0.10 vs our 0.79 at the
time) **and** invents Guitar/Violin/Wind/Synth on a track containing
piano/bass/trumpet/drums, duplicates 19% of its output verbatim, drops 24% of
detected content, and files a trumpet into a seven-octave "Violin" bin.

> **"Produce the score a copyist would write" is the right objective - but a
> copyist compresses from KNOWLEDGE of what the music is. Simulate that
> confidence without the knowledge and the failure mode is not a worse page; it
> is a beautiful page that lies.**

This is the strongest argument for our restraint on instrument naming, and
simultaneously the clearest picture of what we are still losing on the page.

### XIX.4 The lesson worth carrying forward

**A sophisticated diagnosis does not imply a sophisticated cure.** The essay
diagnosed the disease correctly and named the field correctly, then assumed the
remedy had to match the diagnosis in sophistication. Measured, the effect
ordering was the reverse:

1. **Layout policy** (voices/staff, percussion notation, staff count) - cheap,
   blunt, largest wins.
2. **Evidence wiring** (audio consulted at the page layer) - medium, and the fix
   for the defect the user's ear actually caught.
3. **Pattern discovery** - real, null-validated, and so far the *smallest*
   effect.

And one thing neither party anticipated: **four of the defects fixed in this arc
were found by the user's eye and ear, not by any metric** - per-part tempo marks,
two different sustain policies applied to different staves of one score, nonsense
tuplet ratios (24:13, 12:7), and block chords torn into separate voices over
inaudible release differences. The metrics' role turned out to be *stopping bad
fixes from shipping* - which they did four times - not finding the defects.

**Standing rule this suggests for the next big proposal:** rank candidate fixes
by measured effect before by conceptual depth, and build the scoreboard before
the fix it is meant to score.

---

## XX. THE READABILITY GAP - the #1 goal, the ten hypotheses, and why the University cannot close it (2026-08-08/09)

*Standing directive from this session: **closing the readability gap with
Klangio is now the first priority, above everything else.** Detection is
finished as a problem. We hear what is there. We cannot organise it on a page.*

*This section states the problem precisely, records the measured gap across four
songs, gives ten hypotheses for closing it, and answers a question asked
directly: why has Grimlock University contributed so little to this?*

### XX.1 The gap, measured on four songs

Every number from `output/readability.py` on emitted MusicXML - the artifact a
human opens - with the coherence axes added 2026-08-08.

| axis | Klangio | ours | our best song | our worst |
|---|---|---|---|---|
| rest / note | **0.032 - 0.10** | 0.122 - 0.522 | Nov19 0.122 | End Transmission 0.522 |
| top-line stability | **0.663 - 0.701** | 0.339 - 0.579 | Nov19 0.579 | End Transmission 0.339 |
| voice 1 is the tune | **0.539 - 0.776** | 0.403 - 0.517 | Nov19 0.517 | Federal 0.453 |
| max voices | **2** | 3 | - | - |
| beams emitted | **2226** | 741 | - | - |
| chord share | 0.225 - 0.45 | 0.135 - 0.364 | - | - |

And what it costs Klangio to get there - also measured, also on four songs:

| defect | Klangio | ours |
|---|---|---|
| duplicated output | **15.3%, 19%, 20.1%** | **0%** |
| invented instruments | Violin, Wind, Synth, Guitar x2, Bass x2 | none |
| tempo marks | 8 - 15 | **1** |
| overfull (corrupt) measures | 1 - 12 | **0** |
| content vs stem-sum baseline | 0.76x on one song, **2.07x** on another | 1.00x |

### XX.2 The problem, stated exactly

**We notate the PERFORMANCE. A copyist notates the INTENT.**

Every structural choice we made is correct for a truth machine and wrong for an
engraver:

- A note ends where its ENERGY DECAYS (Basic Pitch), not where the player
  stopped. Measured: **63.6% of consecutive notes overlap the next one**, median
  104ms. An interval partition cannot put two overlapping events in one voice,
  so a legato line is FORCED to alternate voices. This single fact caused the
  fragmented melody we chased all session.
- Onsets are where energy rose, not where the beat is. Measured with a
  phase-invariant statistic: **onset alignment to the tempo grid is R = 0.03**,
  which is essentially none. We are notating un-quantised performance time and
  asking the page to make sense of it.
- Every detected note is written. Klangio drops 24% on one song and
  more than doubles another; we are at 1.00x on all of them. Faithful, cluttered.
- One tempo and one time signature for the whole song. Klangio emits 8-15 tempo
  marks. Measured on End Transmission: two contiguous 8-bar regions
  (21.4-34.6s, 108.1-121.5s) are NOT in 4 - corroborated across the drum stem and
  the mix independently, plus a shared bar of 3 at 47.5s. We bar all of it as 4/4.

**Klangio reads better because it COMMITS.** It quantises hard, caps at two
voices, extends notes to the next onset, names instruments, and drops what does
not fit. Its failure mode is the price of that: a beautiful page that lies.
Ours is the mirror - an honest page nobody can read.

Â§XVI.8 already drew the boundary this needs: the frozen-note law governs the
DETECTION FLOOR. The page is a VIEW and is allowed editorial judgement, provided
the performance clock stays untouched. **We wrote that down and then kept
notating like a truth machine anyway.** The gap is the distance between that
sentence and the code.

### XX.3 Ten hypotheses to close the gap

Ranked by measured evidence first, conceptual appeal last - the standing lesson
of Â§XIX.

**1. Cap voices per staff at 2.** MEASURED on two songs, both directions: cap 2
beats cap 3 on rest ratio, top-line stability AND voice1-is-top, losing only
mean_voice_jump (End Transmission 0.518 -> 0.418 rest, 0.370 -> 0.432 stability;
Federal 0.198 -> 0.157, 0.419 -> 0.501). We REJECTED cap 2 previously on
`mean_voice_jump` alone - the metric now proven blind to melodic coherence.
Klangio's max_voices is 2 on every song. Cheapest real win available.

**2. Notate durations from the GRID, not from the audio.** A note lasts until
the next note in its voice unless evidence says otherwise. The legato fill and
the 0.5-beat overlap tolerance are partial versions; the general policy is that
note ENDS are an engraving decision, not a measurement. This is the single
biggest lever on rest ratio, which is our worst axis.

**3. Quantise onsets before the page sees them.** R = 0.03 means we hand the
notation layer un-gridded time. No layout policy can fix rhythm that was never
snapped. This is upstream of 1, 2 and 9 and may be the true root cause.

**4. Per-measure meter and tempo.** `TempoMeter` carries ONE numerator;
`resolve_meter` returns one. End Transmission demonstrably needs at least three
regions. Klangio's 8-15 tempo marks are not a defect to beat - they are it
tracking something real that we flatten.

**5. An explicit EDITORIAL PASS with a drop budget.** The page is permitted to
omit. Give it a budget (start at 5%) and spend it on the least-corroborated
notes. Â§XX.5 below supplies exactly the evidence needed to choose which.

**6. Phrase-scoped voice assignment.** Voices are currently assigned greedily
across the whole staff, so a voice's identity must survive every rest in the
song. Reset at phrase boundaries and a voice only has to be coherent within a
phrase - which is all a reader needs.

**7. Beaming as a first-class decision.** We emit 741 beams to Klangio's 2226.
Beams are how a reader parses rhythm; a third as many is a third of the rhythmic
signposting. Currently whatever music21 does by default.

**8. Enharmonic spelling from the key.** `key_fit` is computed every run and read
by nobody. Both systems show prominent B-naturals in flat keys. Cheap, visible.

**9. Chord recognition over vertical stacks.** `chord_share` 0.135-0.364 against
Klangio's 0.225-0.45. Simultaneous notes that share a rhythm should be ONE
chord in ONE voice, not n voices. Already partly done by `_chord_events`;
the tolerance (40ms onset / 90ms offset) is a guess never swept.

**10. Instrument naming by defensible evidence.** The one axis where we can
beat Klangio outright rather than catch up: it invents Violin and Wind on a
trumpet record. Name only what per-stem energy and timbre support, and say
"unknown" otherwise.

### XX.4 Why Grimlock University comes up short

Asked directly, and the honest answer is structural rather than a matter of
tuning. In APPLY mode the University moved **7 noteheads out of ~4,000**.

**It operates on the wrong object.** It finds patterns in the NOTE STREAM. Every
measured page defect is about DURATION, VOICE ASSIGNMENT and RHYTHMIC SPELLING.
Knowing that five notes form a scale run does not tell you whether to write them
as sixteenths or triplets, where the note ends, or which voice they belong to.

**It is descriptive where engraving is prescriptive.** It labels what is there.
A copyist decides what to WRITE - which is a different verb, and the only one
that moves a page.

**Its unit is below the level where page decisions are made.** Detectors work
over 4-6 note motifs. Voices, staves, beams, rests and barlines are decided over
measures and phrases. There is no operator connecting a motif to a layout
choice, and adding one is the actual work.

**It has no vocabulary for the page.** No staff, voice, beam, tie or rest exists
in `university/`. It cannot express an engraving preference even when it is
right, so its output has to be translated by a layer that is itself the problem.

**Read-only by contract - correctly, and fatally for this purpose.** The
contract that made it safe also guarantees it can never be the mechanism.

**What it is actually good for, and should be kept for:** the null-model
falsification harness is the most rigorous instrument in the codebase. It caught
`arpeggio` firing MORE on shuffled pitches. Keep it as a MEASURING tool. Stop
expecting it to be a TRANSFORMING one. Default it OFF.

### XX.5 A new instrument: augmentation consensus (2026-08-09)

Tested this session and it works, which gives hypothesis 5 the evidence it needs.

Transcribe the same audio at three operating points - as-is, and resampled by
+/- 5 semitones - then map every detection back to the original key and tempo.
Both inverses are EXACT (`t * ratio`, `pitch - 12*log2(ratio)`), and resampling
is artifact-free because we only re-declare the sample rate; no phase vocoder
touches the transients. Verified: pitch-class profiles correlate at 0.977-0.997
after inversion.

Measured on End Transmission:

| bucket | other | median duration | in key |
|---|---|---|---|
| all three views agree | 380 | 291 ms | **99.5%** |
| baseline ONLY | 35 | 232 ms | **85.7%** |
| both shifted views, NOT baseline | 24 | 277 ms | 100% |
| one shifted view only | 174 | 148 ms | 96.6% |

**89-95% of our baseline is independently corroborated.** The un-corroborated
remainder is measurably less musical - in-key drops 99.5% -> 85.7% - which is a
hallucination signal derived without any ground truth. And 24 long notes have
two-witness support while the baseline misses them entirely.

This gives the editorial pass a defensible ranking: **drop the un-corroborated,
keep the confirmed, and consider adopting the twice-witnessed.** It also answers
the question that opened the augmentation experiment - slowed audio produces
longer melodic LINES (notes living in phrases of 8+ rose 45.7% -> 56.6%), though
its rhythmic claim did not survive (grid alignment did not improve).

### XX.6 What this section changes

1. **Readability is the objective now.** Detection is at 1.00x the stem-sum
   baseline and corroborated by Klangio at r = 0.927-0.983 across four songs. It
   should not be optimised further without new evidence.
2. **`mean_voice_jump` may not be used alone to reject a layout change.** It
   blessed a shredded melody at 4.49 while the top line changed voice on 52% of
   onsets. It cost us cap 2 for weeks.
3. **Hypotheses 1-3 are the near-term programme.** 1 is measured and ready. 2 and
   3 are upstream of everything else on the list.

### XX.7 CORRECTION - clutter is not a note-count problem (2026-08-09)

*Three external reviews were solicited on §XX. Two returned things already
built. The third returned one observation that corrects this section, so it is
recorded here rather than folded silently into the list.*

**The observation: Klangio has MORE notes than us and FEWER rests.** Verified
against our own measurements:

| song | Klangio notes | our notes | Klangio rest/note | our rest/note |
|---|---|---|---|---|
| Federal Blvd | 18,446 | 9,983 | 0.032 | 0.198 |
| End Transmission | 4,923 | 4,123 | 0.057 | 0.496 |

**1.19x - 1.85x more material, 6x - 9x fewer rests.**

This kills a hypothesis that was implicit in most of this document and explicit
in one item of it: *the page is cluttered because we write too much.* We do not
write too much. We write a comparable or smaller amount of music and spend far
more of the page on rests.

Rest ratio is therefore **a function of representation efficiency, not of note
count** - consistent with what §XVII.13 measured from the other side (rest count
is largely a function of voices-per-staff) and with the cap-2 result.

**Consequences for §XX.3:**

- **Hypothesis 5 (editorial pass with a drop budget) is DEMOTED.** Dropping the
  least-corroborated 5% attacks a variable that demonstrably is not the one
  driving the gap. The augmentation-consensus instrument (§XX.5) remains valuable
  - as a confidence signal, and for deciding which of two conflicting readings to
  believe - but "drop notes to clean the page" is now contraindicated by
  measurement. Klangio proves a page can be dense AND clean.
- **Hypotheses 1, 2 and 3 are unchanged and reinforced.** All three are about
  how material is REPRESENTED (voices per staff, where a note ends, where an
  onset sits), not how much of it there is.

**What the two other reviews returned.** Recorded because the pattern is now
consistent enough to be worth naming: of nine proposals in the first review,
eight were already shipping - seeded determinism, harmonic stem merge, native
MusicXML NotationScore, per-beat adaptive quantisation, null-model detector
testing, two-axis metrics, and posteriorgram sustain recovery
(`quantization/sustain_recovery.py`, wired at `conductor.py:316`). Its ninth,
adaptive CFAR thresholding, is the same fix made this session to the acoustic
gate: **never compare a score against a constant you did not derive from that
score's own distribution.** That rule is worth generalising even though the
proposal was not new. The second review reproduced §XX.6's ordering
independently, which is mild corroboration of the ordering and nothing more.

**The standing lesson, sharpened:** an external reviewer reading these documents
will return the documents. The value came from the one review that questioned a
premise rather than restating the plan.

---

## XXI. The readability push - what shipped, what was falsified, and the measurement that reframed it (2026-08-08 → 08-10)

*Worked §XX's list. Two changes shipped and are measured on the whole library.
Ten experiments were falsified, three of them because the METRIC was invalid
rather than the idea. And one number found at the end reframes most of the
work: the voice layer is not where our information is being lost.*

### XXI.1 Shipped, measured on 9 songs

**Voice cap 2** (`build_routed_score`, default changed from 3).
`tools/voice_cap_sweep.py` over every saved intermediate - **unanimous, no
exceptions**:

| | cap 2 | cap 3 |
|---|---|---|
| rest / note | **0.342** | 0.432 |
| top-line stability | **0.518** | 0.446 |
| voice 1 is the tune | **0.532** | 0.427 |
| mean voice jump | 4.85 | **4.08** |
| overfull measures | 0 | 0 |

Note counts move **<0.5%**: cap 2 does not discard music, it represents the same
music with fewer independent rhythmic layers. Cap 2 had been REJECTED before on
`mean_voice_jump` alone - the one axis it loses - which is the metric §XX.6
forbids using as a sole veto.

**Voice overlap tolerance + acoustic-gate repair** (`output/piano_reduction.py`).
Federal Blvd rest/note **0.305 → 0.191**, SUNO cut **0.209 → 0.122**; top-line
stability **0.416 → 0.545** and **0.396 → 0.579**; zero overfull measures.

**Two new metrics** (`output/readability.py`): `top_line_stability` and
`voice1_is_top`. They exist because `mean_voice_jump` read 4.49 - PASSING - while
the melody changed voice on 52% of onsets. An aggregate over voices structurally
cannot see WHICH voice carries the line.

### XXI.2 The two real defects behind the fragmented melody

Neither was a missing feature. Both were bugs in shipped code.

**1. A `continue` was shadowing the legato fill.** The acoustic gate branch
returned early for every note the witness had measured - 73% of them - so those
gaps got no sustain at all. Wiring the evidence in had made sustain WORSE than
the blanket threshold it replaced.

**2. 63.6% of consecutive notes OVERLAP the next one**, median 104ms, because
Basic Pitch ends a note where its energy decays rather than where the player
stopped. An interval partition cannot put two overlapping events in one voice,
so a legato line is FORCED to alternate voices. No smarter voice-CHOICE rule can
fix this; the constraint is occupancy, not preference.

### XXI.3 Falsified - ideas

1. **Register-rank voice assignment.** Musically correct account of a brass
   section (lead on top, parts holding position). Measured: top-line stability
   unchanged 0.419 → 0.420, voice1_is_top WORSE 0.413 → 0.299, jump worse.
   *A correct account of the MUSIC is not automatically a correct account of the
   BUG.*
2. **Relabelling voices by mean register.** Also backwards - and instructive:
   background figures sit HIGHER on average than a lead trumpet, so the
   highest-mean voice is not the melody. Ranking by top-note share works
   (+0.02-0.03 v1top), barely.
3. **The acoustic gate itself.** Properly calibrated it beats "blind flat window"
   by ~0.005 rest/note. §XIX called wiring audio in the best fix of the engraving
   arc; measured, that wiring was inert AND harmful. Kept only because it is a
   consistent hair better and costs nothing.
4. **Texture classifier v1.** Pitch-shuffling the input changed NOTHING - not one
   label. Its features were onset-only: a simultaneity-density detector wearing
   the word "texture".
5. **Five of six texture dimensions.** After the rebuild, per-dimension
   replication across songs: `chordality` +1.11/+0.35/(informative on all three);
   `melodic_motion` FLIPS SIGN (+0.60 on one song, -0.48 on another);
   `accomp_regularity`, `independence`, `sustain`, `ostinato` inconsistent.
   Chordality ALONE separates 9-12x better than the six-vector, because five
   noisy dimensions dilute one good one. **More dimensions is not more
   information.**
6. **`motif` as ground truth for "same voice".** Measured: median gap between
   consecutive same-motif notes is **15,074 ms = 14.6 beats**, and only 11% are
   within 2 beats. The annotation marks ONE ANCHOR PER OCCURRENCE, not the notes
   comprising the figure. 159 tagged notes over 19 motifs ≈ 8 occurrences each.
7. **Test-time augmentation for rhythm.** Resampling to 0.749x (which is exactly
   -5 semitones, so "slow it down" and "drop a fourth" are ONE artifact-free
   operation) finds 8-42% more notes and longer melodic LINES (notes in phrases
   of 8+ rose 45.7% → 56.6%). But phase-invariant onset alignment did not improve
   - the rhythmic claim failed. The extra notes are 40-50% SHORTER, mechanically
   explained: at 0.749x speed Basic Pitch's fixed minimum note length corresponds
   to a 0.749x shorter musical duration.

### XXI.4 Falsified - MEASUREMENTS. Three invalid metrics in one session.

This is the more important list, because each looked reasonable and each was
caught only by its own numbers.

1. **CV of downbeat spacing, to choose a meter.** Forcing ANY fixed bar length
   onto steady-tempo music yields evenly spaced downbeats. A 7-beat grid over
   4/4 scores CV 0.006. The statistic cannot distinguish a correct meter from an
   incorrect but evenly-divided one.
2. **An onset-PERMUTING null model.** Permuting onsets among notes preserves the
   onset MULTISET exactly, so every stack kept its size and the two features
   under test were bit-identical. The null was invariant to the thing it existed
   to destroy. Drawing fresh onsets fixed it - and then the null bit.
3. **A one-sided voice-cap objective.** rest/note falls monotonically as the cap
   falls (measured on all 23 regions: 0.059 → 0.117 → 0.185 → 0.215 for caps
   1-4), so cap 1 "won" 23 of 23 regions. That is arithmetic. An objective with
   no penalty for UNDER-representation always selects the smallest representation.

Also: **churn is the wrong statistic for form-derived regions.** A section
boundary is by definition where the music changes, so high churn between
adjacent sections is expected. It was correct for fixed windows and became
meaningless the moment regions came from `detect_form`.

> **The pattern: it is very easy to build a metric that answers a NEIGHBOURING
> question.** All three passed casual inspection. Before trusting a new metric,
> ask what it does on degenerate input - the smallest representation, a rigid
> grid, a shuffle that preserves the quantity under test.

### XXI.5 The measurement that reframes the rest

`tools/representation_frontier.py` compares notated events against the
CONSOLIDATED source events across caps 1-4, per region, on five distortion
channels. On Federal Blvd:

```
region A#0    onset   duration  overlap  ordering  dropped
  cap 1      63.215     1.391    0.136     0.000    0.007
  cap 2      63.215     1.391    0.136     0.000    0.007   <- identical
  cap 3      63.215     1.493    0.136     0.000    0.007
```

**Duration distortion is 1.391 - 139%.** Our notated durations differ from the
consolidated source durations by MORE THAN THE DURATIONS THEMSELVES, and the
figure barely moves across caps. Onset distortion is a flat 63ms regardless of
cap. Caps 1 and 2 are frequently bit-identical.

> **The information loss in our pages is dominated by how we notate TIME, not by
> how many voices we allow.** The whole voice-cap and texture programme is
> tuning a second-order perturbation on top of a large constant.

Pareto membership (duration/overlap/dropped): cap 1 93.3%, cap 2 86.7%, cap 3
66.7%, cap 4 53.3%. No cap is dominated everywhere - so a local budget is not
dead - but with duration distortion at 139% swamping the axes, any frontier
drawn here is measuring quantisation noise more than representation.

This is direct, quantified support for §XX hypotheses **2 (durations from the
grid)** and **3 (quantise onsets before the page)** being ranked above the voice
work, and it is the first hard evidence for WHY.

### XXI.6 What survived, and what it is actually called

The rebuilt probe (`tools/texture_probe.py`) is a CONSUMER: it reads `section`
for regions, `consolidation` to drop fragments (**a third of every song** -
36%/33%/28%), `notation_timing` for symbolic simultaneity, `legitimacy_verdict`,
`instrument_family` and `acoustic_activity`. Version 1 read three fields and
invented its own windows while 20 annotation streams and ~78,000 annotations sat
unread in the same pickle - §XVIII.2 with this tool as the offender.

What survives every null we could point at it:

> **Regions of the same section agree on how chord-like their simultaneities
> are, ~1 SD better than unrelated sections do, and that agreement collapses
> when pitches are shuffled.**

Federal Blvd +1.11 SD, End Transmission +1.06, Nov19 +0.35. That is real and
replicated - and it is one dimension, not a texture model. The honest name is
**chordality persistence**, not texture classification.

### XXI.7 New problems this opened

1. **Duration distortion of 139% is unexplained.** Everything downstream is being
   judged through that noise. This is now the highest-value measurement
   available and it needs no new machinery.
2. **There is NO ground truth for "same voice", and `motif` cannot supply it.**
   The pairwise-graph / JPDA direction therefore cannot be TESTED, let alone
   built - per-witness discrimination has no positive set to score against.
   `tools/texture_probe2.py` implements the edge probe and runs; it simply
   cannot be scored, and is left in that honest state.
3. **Mixed meter is unrepresentable.** `TempoMeter` carries one numerator.
   Measured on End Transmission: two contiguous 8-bar regions (21.4-34.6s,
   108.1-121.5s) are not in 4, corroborated independently by the drum stem and
   the mix, plus a shared bar of 3 at 47.5s. The drum stem took 5 only twice in
   82 bars - and both times inside those windows. We bar all of it 4/4.
   Separately: `DEFAULT_BEATS_PER_BAR = (3, 4, 6)` never OFFERS 5 or 7.
4. **Two songs are locked out of every region-based experiment.** `Hopeful` and
   `prospering` pickles predate section instance identity and store a bare label
   string. The corpus for anything region-shaped is 3 songs, not 9.
5. **`instrument_family` is claimed but unused** by the texture vector - which is
   why family-shuffle moves nothing. Either give it a dimension or stop listing
   it as consumed.
6. **Onset alignment is R = 0.03** against the tempo grid, in every augmentation
   condition including baseline. We hand the notation layer un-gridded time.

### XXI.8 Two instruments worth keeping

- **Augmentation consensus** (`tools/augmented_transcription.py`). Resample by
  ±5 semitones, transcribe, invert exactly. 89-95% of the baseline is
  corroborated by at least one shifted view, and the un-corroborated remainder is
  measurably less musical (in-key **99.5% → 85.7%**) - a hallucination signal
  derived with no ground truth. Note §XX.7 demotes its original purpose: Klangio
  carries MORE notes than us with 6-9x fewer rests, so dropping notes is not the
  lever.
- **The library-wide sweep** (`tools/voice_cap_sweep.py`). Deciding a layout
  default on 9 songs instead of the one that happened to be open is what made
  cap 2 defensible, and would have prevented the original cap-2 rejection.

---

## XXII. Why Klangio reads better - the four mechanisms, measured (2026-08-17)

*Four songs of head-to-head plus three answer keys make this answerable rather
than impressionistic. Klangio is NOT hearing better: on identical audio our
pitch-class agreement runs r=0.957-0.992, and on Federal Blvd we sat at 1.00x
the stem-sum baseline while it dropped 24%. It reads better for four specific
reasons, in descending order of measured size.*

### XXII.1 It tracks a tempo curve; we emit one number

Klangio writes **8-15 tempo marks per song**. We write 1.

This is the dominant cause. A grid that follows the performance puts each note
ON a position with a clean duration that tiles the bar. A rigid grid puts notes
BETWEEN positions, and the engraver fills the leftovers with rests and ties.

MEASURED: 87-97% of beats sit >0.35s from an isochronous grid on the rubato
pieces, drifting up to 7s. Even HRV - metronomic at beat-interval CV 0.030 -
drifts 2.0s over four minutes and is >0.35s off for 91% of beats. So this is not
a rubato-only concern.

Half-addressed: MusicalTime now carries the curve and all three clock<->position
conversions go through it (SS XXI). NOT addressed: we still EMIT one tempo mark,
so the curve is invisible to a reader and to playback.

### XXII.2 Its durations reach the next note; ours stop where the sound decays

Basic Pitch ends a note where energy falls off. A NOTATED duration is not an
acoustic fact - a quarter is a quarter whether played staccato or held under
pedal. MEASURED: 139% duration distortion against the consolidated source, and
it barely moves across voice caps, so it is not a layout artifact.

Short durations leave gaps between consecutive notes; gaps become rests.

### XXII.3 Its lattice can express what the music does; ours cannot

`notation_quantizer._subdivisions_per_beat` returns 3 only for 6/8, 9/8 and
12/8, and 4 for everything else. So OUTSIDE COMPOUND METER A TRIPLET IS
STRUCTURALLY IMPOSSIBLE. The Ellington reference carries 135 tuplet events and
31 transitions at 3:2 or 2:3; our barred output produces effectively none, while
over-producing 1:1 (288 vs 193). Each rounded tuplet leaves a remainder that
becomes a rest or a tie.

The fix is a lattice that can express both families (12 = lcm(3,4)), NOT
per-note tuplet guessing - that manufactured 3,210 tuplets against 4,971
noteheads on an early pass, which is why the ratio_family gate exists.

### XXII.4 Counter-intuitively, its HIGHER note count LOWERS its rest ratio

Klangio emits 8,014 pitched noteheads to our 3,079 on Old Soul's Patience -
same 283 seconds. Denser staves have less empty time to fill. More notes means
FEWER rests.

So "our page is cluttered because we detect too much" is backwards: we detect
LESS and write MORE rests. This is SS XX.7 confirmed on a fourth song.

### XXII.5 What it pays for this

Klangio commits, and the commitments cost it - consistently, across every song
we have compared:

| song | duplicated verbatim | invented parts | overfull measures |
|---|---|---|---|
| Federal Blvd | 20.1% | Guitar x2, Bass x2, Violin, Wind | 12 |
| End Transmission | 15.3% | Guitar x2, Bass x2, Violin, Wind | 1 |
| Heavy Rotation Vibez | 19% | Guitar, Violin, Wind, Synth | - |
| Old Soul's Patience | (Piano x2, Guitar x2, Bass x2) | Violin, Wind | 1 |

SS XIX said it precisely: *"produce the score a copyist would write" is the
right objective, but a copyist compresses from KNOWLEDGE of what the music is.
Simulate that confidence without the knowledge and the failure mode is not a
worse page - it is a beautiful page that lies.*

### XXII.6 A second confirmation of the bar-doubling bug

Old Soul's Patience: Klangio reads 3/4, we read **6/4 at 54.0 BPM**. That is the
same doubling HRV's answer key proved (107 BPM 3/4 against our 107.0 BPM 6/4,
the pulse correct to 0.2%). Two independent songs, same signature - and if 54 in
6/4 is really 108 in 3/4, the bar LENGTH is identical and only the label is
wrong. Cheap to carry, expensive to "fix" by flattening the tempo curve.

### XXII.7 Baseline discipline held

Jazz beat naked Basic Pitch on the same audio - recall 0.153 vs 0.127, F1 0.220
vs 0.201 (naked wins precision, as it should with 2,125 notes against our
3,079). The pipeline is adding, not subtracting.

### XXII.8 Next work, sized

1. **Emit the tempo curve** as tempo marks. We already compute it; a reader and
   a playback engine currently cannot see it. Purely additive.
2. **Durations to the next event**, in beat space. Attacks the 139%.
3. **Subdivision 12** where the evidence supports it, gated on ratio_family so
   per-note tuplet guessing cannot return.
4. **The meter vote** - now with TWO answer-key confirmations of bar doubling.
   Worth ~51 points of recall on band material (HRV: 93.3% detected, 42.2%
   delivered).
5. **Piano detection** - the ceiling, and the only item outside the notation
   layer (Chopin 69.1%, Ellington 62.1% recoverable from raw detection).

---

## XXIII. The audit session — nine defects, the tuplet unblock, and the measurement that moved the target (2026-08-18/19)

*A code audit that turned into a measurement session. Nine reproduced defects
fixed, tuplets unblocked outside compound meter, the Beat Hierarchy made a
tested module - and then two instruments (`event_survival_audit`, a new
injection self-test) established that neither the notation layer nor the timing
layer is where the answer-key gap lives, and that one of our two headline
timing statistics is not observable at all.*

### XXIII.0 The nine defects — all reproduced by running the code, all fixed

Each was reproduced against `../Symphony/.venv` before and after; the console
excerpts live in the commit and in `tests/test_audit_fixes.py`.

| # | defect | evidence |
|---|---|---|
| 1 | `_key_pitch_classes` transposed a minor key's set down 3 semitones, landing on the PARALLEL major | in A minor, `key_fit` called C♮ out-of-key and C♯ in-key |
| 2 | `Key(key.replace("m",""))` handed music21 the parallel major | `Am` engraved with 3 sharps; MIDI and MusicXML disagreed about the same transcription's key |
| 3 | notation durations capped at ONE BEAT | a 4-beat whole note exported as a quarter + 3 beats of manufactured rest |
| 4 | the groove quantizer's grid had no PHASE (`k * period_ms` from t=0) | a note landing exactly on tracked beat 1 (800.0ms) "snapped" to 789.5ms |
| 5 | the resolved meter never reached anything that quantizes | `resolve_tempo` defaults to (4,4) and meter resolves 40 lines later; with 6/8 resolved, `build_lattice` still used 4 subdivisions |
| 6 | MIDI beat grid misaligned for pickups of 50ms–half a beat | beat 0 landed at tick 168 of a 480-tick beat, displacing every beat in the file |
| 7 | the gap clamp assigned a raw `gap` to `quarterLength`, bypassing `_representable` | 1/3 + 1/4 = 0.5833 — the 24:13 shape |
| 8 | `allow_triplet` REPLACED the binary grid rather than widening it | a plain eighth inside a tuplet beat snapped 0.5 → 0.667 |
| 9 | `_normalize_voice_numbers` stripped the MusicXML DOCTYPE | ElementTree round-trip does not preserve it |

**Structural findings alongside them:**

- `MusicalTime` was "the only currency" in name only — `notation_quantizer`
  carried a SECOND implementation of `to_beats`/`to_ms` with the same
  interpolation and none of the glitch-confidence logic. Deleted; the map is now
  threaded into the lattice judge, the rhythm inference and both score builders.
- The **downbeat phase was computed in three places and consumed in none**:
  `estimate_time_signature` unpacked it into `_phase` and dropped it,
  `detect_pickup` was exported and never called, madmom's downbeats stopped at
  the octave arbiter. Two downstream modules each invented their own barline
  origin. Now resolved into `TempoMeter.downbeat_times_ms` and out to both.
- `resolve_tempo` returns a weighted `tempo_bpm` but the ANCHOR witness's
  `beat_times_ms` — two facts that can describe different pulses inside the one
  type §9 created to stop exactly that.
- `save_intermediate_path` does **not** store `beat_times_ms`, so a saved run
  cannot re-export its own notation. Found when a Hopeful re-export could not
  reconstruct its grid. Still open.

**One finding WITHDRAWN.** `core/window_pane.py` was flagged as dead by an
import-graph scan. Its header says it is deliberately staged for a live
dashboard and pre-empts this exact finding. The scan was not a substitute for
reading the module. Left untouched.

### XXIII.1 Meter — FIXED, and the root cause was the grid we sample on

Two changes, measured across the library by the new `tools/meter_harness.py`.

**The meter test was being sampled on a grid this project already documents as
wrong.** `build_phase_locked_grid` rebuilds an isochronous grid from one scalar
tempo. MEASURED on Chopin (beat CV 0.168, local tempo 63–102):

- the isochronous grid sits a median **194ms** from the tracked beats, up to
  394ms, with **48% of beats more than 200ms away**;
- sampled on it, **no meter candidate is significant at all** and 4/4 collapses
  to score 0.0217, so the estimator returns its (4, 4, 0.3) fallback;
- sampled on the **tracked beats**, the same estimator returns 4/4 at score
  0.159–0.186, significant, phase 0 — which is what the published edition says.

**madmom's downbeat witness has an uninformative confidence for its numerator.**
It answers **6 on four songs of five** whatever the truth is, and its
`mode_share` term is **1.00 everywhere**. The spacing-CV half of its confidence
IS informative across songs: where it is right it is confident (No Pasarán
0.920, HRV 0.877, Hopeful 0.533) and on the one song it is badly wrong it is not
(Chopin **0.384**). A floor at 0.50 lets two agreeing calibrated witnesses carry
a beat the trained one cannot see.

Scored under §XVII.3's metrical-equivalence rule:

| rule | Chopin (4) | HRV (3) | Hopeful (6) | No Pasarán (4) | |
|---|---|---|---|---|---|
| isochronous + sum *(shipped)* | 6 ✗ | 6 ✓ | 6 ✓ | 4 ✓ | 3/4 |
| tracked + sum | 6 ✗ | 6 ✓ | 6 ✓ | 4 ✓ | 3/4 |
| tracked + agreement-first | 4 ✓ | 4 ✗ | 3 ✓ | 4 ✓ | 3/4 |
| **tracked + sum + downbeat floor 0.50** | **4 ✓** | **6 ✓** | **6 ✓** | **4 ✓** | **4/4** |

Verified end to end: **Chopin now resolves 4/4** (was 6/4, against a published
edition), **Hopeful holds at 6/4**, and Hopeful's key is now **B♭ minor** where
every previous version wrote 5 sharps.

**HONEST ABOUT THE THRESHOLD.** 0.50 is a round number in the single gap between
0.384 and 0.533 in a four-song sample. The RULE is principled — an uncalibrated
witness should not outweigh two calibrated ones, the same shape as
`OCTAVE_ARBITER_MIN_CONFIDENCE` — but the VALUE is fitted to very little.

### XXIII.2 §XVII.3 and §XXII.6 contradict each other, and it changes that table

§XVII.3 says Hopeful's 6/4 against Klangio's 3/4 is *"the same pulse — 6/4 is
two 3/4 bars"* and **withdraws** the suspicion. §XXII.6 calls the identical
relationship on Old Soul's Patience and HRV *"the bar-doubling bug"*, confirmed
by an answer key.

The harness above scores HRV's 6 as **correct** under the first reading. Under
the second it is a **failure** and the score is 3/4, not 4/4. Same data,
opposite verdicts, and no way to settle it from inside the codebase. **This
needs a ruling before any meter score is quoted.**

### XXIII.3 Tuplets — the structural block, found and removed

**Do we have real triplets?** Yes — 3:2 was always the majority of our tuplets.
**Could we have anything else?** No, and for three reasons in series:

1. the notatable set was binary + exactly four triplet values, so a sextuplet's
   1/6 flattened to 1/8, a quintuplet's 1/5 to 1/4, a 12-tuplet's 1/12 to 1/8;
2. `tuplet_divisor` was written by the model and **never read** by the page —
   `notation_score` took only the `is_tuplet` bool, so the NUMBER was dropped at
   the page boundary;
3. given only a quarter-length, music21 renders 1/6 as two **3:2** groups, not
   one **6:4** — a different-but-equivalent reading it picks on its own.

**The Chopin edition is 6:4 ×297 of 507 tuplets.** The dominant tuplet in the
piece was precisely the one we could not write. Fixed by making the notatable
set divisor-aware and built from exact `Fraction`s, threading the divisor
through to the exporter, and STATING the bracket instead of letting it be
inferred. Verified in **4/4** — nothing in that path consults the time
signature, which answers §XXII.3 directly:

```
tuplet ratios emitted: {'3:2': 3, '5:4': 5, '6:4': 6, '7:4': 7}   junk: none
```

Exact fractions also remove the accumulated-float-error source: `24:13`,
`48:43` and `192:127` are not tuplets anyone detected, they are binary-float
sixths being reconciled.

### XXIII.4 The beat vocabulary — a bigger hole than the tuplet ceiling

`_BEAT_VOCABULARY` held nine hand-authored fillings and **every one of them
began at 0.0**, so a beat that does not start with a note — one beginning with a
rest, or under a note held over — was inexpressible. MEASURED on Chopin:
**75% of beats carrying a single onset have it at or past 0.125 of the way
through**, median **0.452**.

Replaced with a **subdivision-subset model**: divide the beat into `d` equal
parts, strike any subset, assign onsets to slots by exact monotone alignment
(a DP, not independent rounding — two onsets must not collapse onto one slot).
It subsumes the entire old vocabulary by construction (`dotted_eighth_sixteenth`
is `{0, 3/4}` on d=4; `eighth_triplet` is the full d=3 grid).

A second defect fell out: **13.9% of onset groups were assigned to the previous
beat**, a spike of 163 in the final sixteenth mirroring 185 in the first — the
same musical event split by the boundary. Onsets within 1/16 of the next beat
now belong to it.

**Beat-level rhythmic coverage: 18.8% → 23.8% → 97.3% → 98.5%** (fixed
vocabulary → even divisions → subset model → edge-snap).

### XXIII.5 Beat Hierarchy — the user's specification, as a tested module

`output/beat_hierarchy.py` + `tests/test_beat_hierarchy.py` (14 tests). The
four stated levels collapse to **one rule**: *a note may cross a boundary only
if it starts on a position at least as strong as that boundary* — with the
barline as an absolute exception nothing crosses, including a note that started
on one. (Writing that test caught the bug: barline-vs-barline was not
"stronger", so a note starting on a downbeat could run through the next.)

Compound meters group in threes; **odd meters get no invented half-measure** —
3/4, 5/4 and 7/8 have no unambiguous middle and a boundary where no reader
expects one is worse than none. 7/8 grouping (2+2+3 vs 3+2+2) remains an open
musical decision.

**Tuplet rules, per user directive:** the FRAME is anchored, not the first note
— a tuplet rest may hold slot 0, so a figure beginning on "ple" or "let" is
writable. A tuplet never leaves its beat, hence never crosses a barline. A
displaced or half-empty grid is not a tuplet and falls back to binary.

### XXIII.6 Shuffle — the prior was overriding the evidence

*User: "a lot of rhythms in the song are Triplets. As it is a shuffle. Written
and transcribed as 16th notes but they should be even 8th note triplets."*

MEASURED on a textbook shuffle beat (onsets at 0.0 and 0.667):

```
d=3  2_of_3[0,2]   sse 0.0000  PERFECT fit   cost 4.90   posterior -2.448
d=4  2_of_4[0,3]   sse 0.0069  worse fit     cost 3.50   posterior -2.288   WON
```

A perfectly-fitting triplet lost to a worse-fitting binary reading on the flat
+2.0 tuplet penalty alone. That penalty describes a pop page, not a shuffle. The
correction needs no new constant: `swing_ratio` is already measured per track by
`estimate_groove`, so the penalty on divisors of three now fades as measured
swing approaches a true triplet feel. Hopeful measured **0.5864** (and the raw
in-gap onset distribution peaks squarely in the 0.60–0.70 bucket, 51 of 187).

**Result: 3:2 went 258 → 753, tuplet share 3.55% → 10.26%** — now 3× Klangio's
3.45% on the same song. Straight tracks are unaffected and binary 16ths on a
swung track stay binary.

**THE COST, PREDICTED BY §XVII.15 TWELVE DAYS EARLIER.** Junk ratios **29 → 93**
and sub-32nd share **0.30% → 1.77%**. §XVII.15: *"They appear exactly where
rhythm_inference's per-beat verdict says triplet while neighbouring beats say
binary, putting k/3 and k/4 positions in one measure (12 = lcm(3,4))."* Making
triplets cheaper made more mixed-grid measures. The lever is beat-to-beat grid
CONSISTENCY upstream — and §XVII.15 also warns that three exporter-side attempts
have already failed. A fourth was attempted this session and is part of the 93.

**Also open:** the pickup. `_origin_before` now keeps the resolved downbeat and
steps back whole bars (verified on a synthetic 6/4 with a 3-beat anacrusis: the
big ONE lands on measure 2 beat 1), but on Hopeful measure 1 is still a full
6.000 because **the phase detection returned 0** — the accent evidence did not
clear `PICKUP_MIN_CONFIDENCE`. The plumbing works; the detection does not.
`detect_pickup` remains the unwired lever, and the user has supplied the answer
(3 beats in 6/4), which makes it directly testable.

### XXIII.7 DECISIVE — the notation layer is not where the notes are lost

`tools/event_survival_audit.py` on Chopin, answer-key recall at every stage:

```
1. Basic Pitch (raw)              62.9%
2. + onset refinement             63.0%   +0.1
3. + sustain recovery             63.0%   +0.0
4. + consolidation                62.7%   -0.3
5. + legitimacy filter            62.7%   +0.0
6. + notation quantization        62.6%   -0.1
```

**The entire pipeline costs 0.3 points.** This retires an argument rather than
advancing it: no amount of notation work can raise answer-key recall, and
§XVI.7's "our over-detection filters are inert" is confirmed from the other side
— the legitimacy filter contributes exactly +0.0.

### XXIII.8 REFRAME — it is a PRECISION problem, not a detection-capability one

At a generous 0.75s tolerance, Chopin:

```
recall 0.903   precision 0.429    false positives 1096    misses 89

FALSE POSITIVES BY SHAPE            BY STEM (solo piano!)
  same pitch class, octave off  61.2%    bass    354 of  463  (76.5%)
  unison duplicate              21.3%    vocals  129 of  200  (64.5%)
  unrelated pitch                7.5%    other   669 of 1257  (53.2%)
  no key note sounding           5.1%
  +19/+24 harmonic               4.9%
```

**We hear 90% of the edition and emit twice too much.** Only 89 notes are
genuinely missed. So the earlier framing — "detection is the ceiling, we need a
better model" — was wrong: a piano-specific model would fix the 89, not the
1096.

Two-thirds of the errors are *octave copies of notes actually sounding*, which
is what transcribing separated stems produces: each stem carries harmonics of
the same piano note, so one note is counted three times at three octaves. And on
a **solo piano recording** the bass and vocals stems are 76% and 64% wrong —
the same disease as the 1546 phantom drum hits this run produced on a nocturne.

**We have filters for this and none of them targets an octave copy.**
`check_range` exists but is annotation-only with `drop_purge_candidates` off.

### XXIII.9 Timing — not lag, not drift, not snapping

```
                      within 0.10s   within 0.25s     std
RAW Basic Pitch          32.3%          69.2%       0.2654
NOTATION-QUANTIZED       34.1%          71.4%       0.2590
```

- **Not a systematic lag**: mean −24ms, median −40ms, 42% late / 58% early.
- **Not drift**: linear trend **+0.84 ms per second** of audio, and the
  per-window means do not march in one direction.
- **Not snapping**: quantization slightly IMPROVES the distribution. A clean
  negative — the grid is not throwing notes off.

It is symmetric jitter, and a third of it was the ruler. Sweeping the scorer's
alignment resolution with the transcription held fixed:

```
 steps   s/step     std    <0.10s   <0.25s
   450    0.397   0.3154    29.2%    59.8%
   900    0.199   0.2654    32.3%    69.2%
  1800    0.099   0.2260    49.6%    79.1%
  3600    0.050   0.2163    57.4%    80.8%
```

The default 900 gave 0.199 s/step on a metric graded at ±0.25s. Raised to 1800.
On Chopin's MIDI this alone moved **recall 0.645 → 0.776 and F1 0.427 → 0.514
with no change to the transcription.** Every answer-key number quoted before
this fix was depressed by the instrument.

### XXIII.10 DECISIVE NEGATIVE — the scorer cannot see a timing bias at all

`tools/scorer_selftest.py` injects a known offset and asks the scorer to recover
it. On Chopin:

```
injected   recovered        injected   recovered
   -400ms      44.0ms          +100ms      39.6ms
   -100ms      47.4ms          +400ms      42.3ms
      0ms      46.6ms

recovered = -0.005 x injected + 44.5 ms      dtw cost 0.2141 -> 0.2169
```

Shifting **every note by ±400ms** moves the recovered bias by **less than 8ms**.
Chroma-DTW with an open end slides its path to absorb a global translation; the
alignment cost barely registers it.

**Consequences:**

- Every δ this tool has reported is an artifact of the alignment's freedom to
  slide, not pipeline latency. The −24ms and +46.6ms measured this session are
  both meaningless AS BIAS.
- Phase calibration against this reference is not merely low-value (δ/σ = 0.13);
  it is **unmeasurable**. Fitting δ̂ here fits the reference's noise.
- σ, by contrast, holds at 366–386ms across every injection — **spread is
  observable, bias is not**, which is the expected signature of a monotone-path
  alignment. The variance-dominated conclusion stands.
- A globally shifted score is still a correct score, so for NOTATION this blind
  spot is tolerable. For any claim about latency accuracy it is not.

### XXIII.11 Instruments built, and one that had to be calibrated against truth

- `tools/meter_harness.py` — meter only, across the library, against known
  answers. Both grids × both vote rules, so the grid question and the vote
  question stop being confounded.
- `tools/tuplet_audit.py` — barline crossings, frame anchoring, ratio census.
- `tools/compare_versions.py` — every export of one song side by side on
  rest/note, tie/note, chord/note, tuplet%, junk, sub-32nd share.
- `tools/scorer_selftest.py` — injection-recovery for the scorer itself.
- `tools/score_vs_answer_key.py` — now reports RMSE, a Gaussian soft-recall
  scored over EVERY reference note (a miss costs a real zero, so no threshold
  can quietly exclude it), and δ/σ with a plain verdict on whether calibration
  would pay.

**`tuplet_audit` FAILED the published Chopin edition three times before it was
right**, and each failure was the tool's:

1. it called `14:2` un-notatable because 14 > 12 — Chopin writes 14-tuplets.
   Junk is a nonsensical NORMAL count (13, 19, 43), not a large actual one.
2. it counted every note INSIDE a tuplet as a group start (music21 leaves
   `type=None` on middle notes) — 415 phantom violations in an edition with none.
3. it demanded whole-beat anchoring; the edition anchors `4:3` and `8:2` groups
   to the second EIGHTH of a beat, which is standard.

**Run an auditor against known-good ground truth before trusting it on your own
output.** The edition file itself then showed 2 tuplet notes crossing a barline
in bars 50/52 — an encoding artifact of that MusicXML, a reminder that the
answer key is ground truth for CONTENT, not for byte-perfect encoding.

**Testing:** 1 non-discoverable script → **53 tests** with a `pytest.ini`.

### XXIII.12 Process failures worth recording

- **Stale numbers reported three times.** A re-export crashed after a patch
  (two `_apply_tuplet` definitions, the old shadowing the new and returning
  `None`), the audit kept reading a file from an earlier run, and three
  "different" fixes produced byte-identical output before the log was opened.
  Identical output across different changes is a signal, not a coincidence.
- **Blind patching.** Several fixes were applied by string-replacement without
  reading the surrounding code; one deleted the drum-position table along with a
  duplicate function. Restored from git.
- **A fix that made things worse before better.** Rejecting non-anchored tuplets
  and falling back to the binary grid ROUNDED sub-32nd scraps UP to 0.125, and
  lengthening notes pushed barline crossings from 1 to 50. Shortening is always
  safe; lengthening never is.

### XXIII.13 Where we stand

**Two different projects have been conflated, and they have different verdicts.**

*The page* is measurably better: `rest/note` 0.447 → 0.280 (best ever, against
Klangio's 0.089), tuplets structurally possible outside compound meter for the
first time, meter right on Chopin and held on Hopeful, key right on Hopeful, the
hierarchy enforced and tested.

*The answer-key number* did not move for any notation reason, and §XXIII.7 shows
it cannot. What moved it was fixing the instrument (0.427 → 0.514).

### XXIII.14 Revised leverage ranking (supersedes XVIII.5)

1. **Gate separation on whether the stems are real.** Kills ~44% of false
   positives outright and stops the drum staff on a nocturne. Cheapest item on
   this list by a wide margin, and it is a precision AND a timing fix — the
   bass/vocals stems also carry the only systematic onset biases (−61ms, −98ms).
2. **An octave-collapse pass** on co-occurring same-pitch-class notes, deciding
   the fundamental from the stem's own spectrum. Targets 61% of the remaining
   errors — the filter we do not have.
3. **Beat-to-beat grid consistency** in `rhythm_inference` (§XVII.15's lever,
   now urgent because the shuffle fix fed the mechanism). NOT another exporter
   patch; four have failed.
4. **Wire `detect_pickup`** — written, exported, still called from nowhere, and
   the one thing standing between us and correct barlines on a song with an
   anacrusis.
5. **Deduplicate unisons** — 21.3% of false positives, mechanical.
6. **Emit the tempo curve to the page** (§XXII.8 #1) — still purely additive,
   still uncomputed on the MusicXML side. Note the MIDI already carries a
   multi-segment map (187 segments on Chopin); only the page shows one mark.
7. **Store `beat_times_ms` in the intermediate** — one line; without it a saved
   run cannot reproduce its own notation.
8. **Settle §XXIII.2** — the metrical-equivalence contradiction. A ruling, not
   an implementation.

**Through-line, updated.** §XVIII.5 said the codebase was carrying *unconsumed
evidence*. It still is — `detect_pickup`, the downbeat phase until this session,
`tuplet_divisor` until this session. But the larger finding is that we have been
**optimising the half of the pipeline that cannot move the number we are
grading ourselves on**, while the other half emits two wrong notes for every
right one and nobody has built the filter that would catch them.

---

## XXIV. The measurement session — a GUI, the fragmentation regression, and thirteen wrong hypotheses (2026-08-19 → 08-23)

The headline is not a fix. It is a ratio: **thirteen times in this session a
finding turned out to be a bad measurement rather than a bad implementation** —
including two findings from this session's own audit, retracted a day after
being written. That number is the most transferable thing here, and every claim
below therefore names how it was measured.

### XXIV.1 The largest single gain came from NOT running a model

Every Chopin file in the project was the same 180-second excerpt. The complete
411-second Rubinstein recording — the performance the answer key and the
Klangio transcription were both made from — was sitting in `Downloads`.

Run as one harmonic stem, with Demucs skipped:

| | 180s clip, 6 stems | 411s full, solo piano |
|---|---|---|
| notes | 3466 | 3204 |
| phantom drums/bass/vocals | 1546 / 463 / 200 | **0 / 0 / 0** |
| precision | 0.374 | **0.512** |
| recall | 0.785 | 0.735 |
| **F1** | 0.507 | **0.603** |
| key | D#m | **B major** (correct) |

**+0.096 F1 in one step**, and the answer key now aligns all 2231 of its notes
instead of 915. None of it came from a filter. It came from not inventing 2209
notes from instruments that are not playing.

The key came out right for a structural reason worth keeping: B major is
established by the return and the final cadence, and the excerpt contained
neither. Windowed key detection reads
`B D#m D#m D#m Ab Eb Eb D#m B B B B` — opens in B, wanders, returns.

`separation_engine/solo_detector.py` now answers "is this one instrument"
before separation, from three physical facts: a kit makes broadband noise above
8kHz repeatedly; **a piano cannot crescendo on a held note**, so sustained
frames that RISE prove something that is not a piano (this is the witness that
catches piano-plus-singer, which no percussion test can see); a guitar has no
string below 82Hz. Zero ensembles were called solo across the corpus — the only
error that silently deletes instruments.

### XXIV.2 Separation should describe notes, not generate them

The deeper version of XXIV.1, measured on Burden of Sentiment against Klangio:

| candidate | notes | precision | recall | F1 |
|---|---|---|---|---|
| raw Basic Pitch, **no separation** | 1602 | 0.202 | 0.236 | **0.218** |
| our `other` stem alone | 1977 | 0.168 | 0.243 | 0.199 |
| our full six-stem pipeline | 3425 | 0.147 | **0.369** | 0.211 |

The entire pipeline adds 1823 notes and does not beat feeding the raw mix to
Basic Pitch. Read it honestly: separation genuinely **helps recall** (0.369 vs
0.236) — it surfaces quiet notes buried in the mix — and badly hurts precision.

The asymmetry that matters: **a stage that GENERATES can invent; a stage that
DESCRIBES can only mislabel.** Separation is fallible either way, so it belongs
where its errors are cheap. `separation_engine/stem_attribution.py` prototypes
that: measure a note's f0 energy across stems, label it by the stem holding
most. Validated by asking whether it recovers a label the pipeline already
assigned — vocals 97%, bass 64%, harmonic family 59%, **72% overall, 78% among
uncontested notes**. Not yet good enough to replace per-stem detection; the
weak spot is the harmonic family, which is exactly where htdemucs_6s is weakest
and why those three stems are merged already.

### XXIV.3 The fragmentation regression — a gate blocking notes nobody could play

Consolidation absorbs a fraction of what it is eligible to absorb, and the
split is by DATE:

| run | eligible | absorbed | caught |
|---|---|---|---|
| `Hopeful_UNI` (08-06) | 1376 | 1376 | **100%** |
| `HRV_GATE` (08-07) | 857 | 857 | **100%** |
| `Hopeful_TUP` (08-23) | 1376 | 878 | 64% |
| `HRV_FULLRUN` (08-19) | 864 | 310 | 36% |
| `Chopin` (08-18+) | 882 | 216 | **24%** |

Same song, identical eligible count, different outcome. The attack gate landed
2026-08-14 (`486db37`). It was right to exist — before it, over-merging deleted
31% of Ellington's notes and 43% of its 4+ note chords — but it is **binary**,
and we swung from over-merging to under-merging. Both extremes are documented
failures.

Of the merges it blocked, **96-99% are notes that end and restart within 15ms**
(642 of Chopin's 666, 471 of Burden's 478). Fifteen milliseconds is a
sixty-fourth note at 240bpm. Nobody re-articulates that fast, so the onset at
that boundary is the note's OWN attack found a second time.

Fixed by `MIN_REARTICULATION_GAP_MS = 20` — a statement about what a player can
physically do, not a tuned threshold. Verified at the boundary: gap 0 and 10ms
merge, gap 25 and 40ms stay blocked.

**Confirmed on three songs never run before**, all back to 100% caught. The
decisive one is `You_Say_God_Says`, which the consolidation module cites BY
NAME as having returned 33.5% sub-32nd notes when the gate misbehaved: it now
runs at **1.4%**. The merges were recovered without reopening the over-merge.

### XXIV.4 Tuplets — the number that was computed and dropped

`rhythm_inference` decided a per-beat `tuplet_divisor`, the annotation writer
carried `is_tuplet` and dropped the number, and both exporter functions are
gated on `if divisor`. **The page emitted no deliberate tuplets at all** —
every bracket on every page was music21 inferring one from a duration it could
not express, which is simultaneously the junk-ratio source and the off-beat
source. One field.

With it restored, Hopeful's divisors read 3x541 and 6x182 — triplets and
sextuplets, which is what a shuffle is, and what the user said this song was
from the beginning.

**The vocabulary is now short, by directive:** no 5, 7, 9, 10, 12, and nothing
finer than a sixteenth ever written. The evidence backed the directive — with
the wide vocabulary available, Chopin read **146 of its 314 tuplet notes as
nonuplets and 10-tuplets**. That is not ornamentation; it is the beat fitter
given enough rope to explain onsets the grid did not fit. **The weird tuplet is
the symptom, not the disease**, and leaving it available lets a grid error hide
as a notation choice.

### XXIV.5 k/24 — a ceiling that could be exceeded

`notatable_at_most` returned `min(allowed)` when nothing fit under its ceiling —
a value *longer* than the bound it existed to enforce. Fifteen notes were asked
for <= 1/12 of a beat and given 1/8, overrunning by exactly 1/24 each and
displacing every onset after them. That single line produced the whole k/24
family.

Generalised: **any function whose name promises a bound must be checked for the
branch where the bound cannot be met.** Returning 0 and letting the caller
decide is correct; returning something out of bounds is not.

The same error recurred once more in this session, in code written to fix it —
`_notatable_chain` folded a sub-notatable residue into the previous note to
keep a chain summing exactly. Growing a note to absorb a leftover pushes the
next onset. **Shortening is always safe; lengthening never is.**

### XXIV.6 Two remaining rule violations are the serializer's, not ours

Measured, one object, one serialization apart: in memory the score has **2326
onsets and none off-grid**; written and read back it has **2327, one at 17/24**.
music21's writer adds a note and places it off the grid. Same author as the
barline-crossing rests — full-bar rests it writes mid-bar when a voice runs
short.

`output/musicxml_repair.py` now works on the bytes, because that is the only
place left: an in-memory clamp reported zero elements touched while the written
file still had thirteen crossings, `makeNotation=False` raises on complex
durations, and `splitAtDurations()` does not clear it. It rewrites the
**ambiguous whole rest** — a whole rest IS the conventional bar rest in any
meter, so writing one for four beats of a six-beat bar makes MuseScore draw it
filling the bar. 148 fixed on Hopeful, 77 on HRV, and rest-crossings 4 -> 0.

### XXIV.7 A verdict is only useful if it discriminates

The counterweight to §XVIII.5's "unconsumed evidence" theme. Not every unread
verdict deserves wiring, and the test is whether it fires *differently* on
material we handled well versus badly:

| witness | Hopeful vocals (good) | Burden vocals (bad) | Chopin piano (our best) | usable? |
|---|---|---|---|---|
| range implausible | **0%** | **8%** | **0%** | **yes** |
| Schoenberg "uncertain" | 16% | 93% | **63%** | no |
| key-fit out of key | 9% | 8% | **15%** | no |

`RANGE_ANNOTATION_KIND` — written every run since 2026-08-06, read by nothing,
and its own header records this same finding on a different song — is now
wired. It drops 2.8% on Burden, 2.6% on HRV, **0.1% on Hopeful and 0% on
Chopin**, only ever from bass and vocals. That distribution is the point: it is
silent on material we handle well.

The other two must stay explanatory. Chopin is chromatic and modulating, so
out-of-key notes there are real music.

**Also answered, in the negative:** we are *not* broadly mistaking partials for
pitches. Notes sitting on a harmonic of a lower struck note run at 32.5% for us
and 32.1% for Klangio — a ratio of 1.01. The UPPER partials (5f/6f/8f) do run
at 3x Klangio's rate, and they concentrate in the two stems the range check
flags: vocals 9.2%, bass 4.5%, merged harmonic 2.0%.

### XXIV.8 The octave filter, built and then refuted

`acoustic_witness/octave_stack.py` flags the interior of a 3+-octave stack. On
the six-stem clip it looked good: 87.2% of what it removed was unmatched by the
answer key, +0.0035 F1. **On the full solo-piano run it is net-negative** —
precision 54.4%, F1 -0.0033, with 26 of 57 flagged notes real.

It was never an octave detector. The phantom bass and vocal notes were stacking
octaves against real piano notes, so it was catching **stem bleed**. On clean
input what remains is Chopin's own octave doubling. Kept, off by default,
headers corrected in all three places.

This is the clearest instance of the session's pattern: a filter measured on
contaminated input, looking useful, and dissolving when the input was fixed.

### XXIV.9 Chopin is a stress test, not a target

Its local tempo spans **44-68 bpm (1.53x)**, and only **33% of the piece sits
within 15% of any single tempo we could report**. There is no tempo there to
get right. A single accuracy number on a rubato performance mostly measures how
the aligner felt that day.

But as a bug-finder it has been the most productive file in the project. It
found the octave filter and then refuted it, the barline crossings, k/24, the
`split_for_hierarchy` links no notehead can carry, the whole rest that reads as
a bar rest, the dropped `tuplet_divisor`, `_content_hash` copying a whole track
to read 64KB of it, and six of the thirteen measurement errors.

**The steady-pulse material is the real target.** On Burden the grid is already
right — 65.0 bpm against Klangio's 65.4, same meter, same 63 bars.

### XXIV.10 The front end

`app/` is a library layer with no UI imports (`probe`, `diagnostics`, `runner`,
`progress`, `lab`) plus a tkinter window. Notation is handed off to MuseScore,
so no renderer was needed and toolkit choice stopped mattering.

Three things in it worth keeping:

- **Probe is its own step.** It costs seconds and informs a decision that costs
  half an hour, and it shows the three numbers behind its verdict.
- **The clock never stops.** The pipeline is silent for most of its wall clock —
  thirty decisions across thirty to seventy minutes — so a bar driven by events
  freezes exactly when reassurance is wanted. The clock runs on wall time; the
  bar counts milestones and is labelled as that, not as a time estimate; and a
  quiet timer says "working, nothing reported for 4m 12s" rather than looking
  hung.
- **Window Pane finally has its consumer.** Its header has said since it was
  written that the resolution was "build the frontend, not delete this."
  Adopted by a TEE (`PaneMusicBox`) rather than forty-five edits to the
  conductor: Music_Box keeps everything forever, Window_Pane keeps a bounded
  window for whoever is watching now.

### XXIV.11 Items from §XXIII, settled

| §XXIII item | status |
|---|---|
| #1 stem-reality gate | **partly** — solo detector + range check, both wired |
| #2 octave-collapse pass | **built and refuted** on clean input (XXIV.8) |
| #3 beat-to-beat grid consistency | **superseded** — the actual mechanism was k/24 (XXIV.5), fixed |
| #4 wire `detect_pickup` | **still open** |
| #5 deduplicate unisons | **still open** |
| #6 tempo curve to the page | **still open** |
| #7 store `beat_times_ms` | **done** — and it mattered: without it a re-export engraved 2602 onsets where the pipeline wrote 2310 |
| #8 settle metrical equivalence | **still open** — needs a ruling, not an implementation |

### XXIV.12 Leverage ranking (supersedes §XXIII where they differ)

1. **Detect once, attribute after.** The only change that could move F1 by a
   step rather than a fraction. Prototyped at 72%; the end-to-end path has not
   been run.
2. **Chopin's tempo is 26% fast** — 70 bpm where the edition implies 55.4, and
   not a clean octave error. Witness contention is now persisted so the
   disagreement is visible; nothing acts on it yet. This is upstream of the
   tuplet mess: a wrong grid makes every real value land between positions, and
   the fitter reaches for a quintuplet to cover it.
3. **Burden's note count** — 3425 against Klangio's 1368. Fragmentation was the
   wrong suspect (our tie behaviour is already correct at 1.22 noteheads per
   sounding note against Klangio's 1.20); this is over-detection, which is what
   #1 addresses.
4. **Wire `detect_pickup`** — written, exported, still called from nowhere.
5. **Settle §XXIII.2** — a ruling.

**Through-line, updated.** §XVIII.5 said the codebase carries unconsumed
evidence; §XXIII said we were optimising the half of the pipeline that cannot
move the number. Both still hold. What this session adds is a third:
**roughly half of what a mechanical scan flags dissolves when measured.** The
scans found `tuplet_divisor` and the range check, both real and both valuable —
and also flagged `piano_reduction` and the wobble pass, both of which turned out
to be working correctly. The scan is a candidate generator, not a verdict, and
the discipline that separates the two is the only reason the fixes in this
section are trustworthy.

---

# XXV. The Clocks session — the page became readable, and four verdicts were being thrown away

Measured end to end on the full pipeline, matched detection thresholds
(onset 0.6 / frame 0.45 / min_note 107ms) so the only variable is the work.

| | human | before | **after** |
|---|---|---|---|
| eighth notes | 77.6% | 32.2% | **63.0%** |
| 16th notes | 0.0% | 41.7% | **7.4%** |
| 16th rests | 0 | 1347 | **228** |
| total rests | 1037 | 2767 | **1443** |
| 32nd rests | 0 | 6 | **0** |
| illegal tuplets | 0 | 1 | **0** |
| key signature | −4 (Ab) | −5 (Bbm) | **−4 (Ab)** |
| bass octave, as engraved | — | 63.5% | **94.6%** |

Tempo 130.0 exact and 4/4 throughout, unchanged.

## XXV.1 What was actually wrong

**The kit, not the harmony.** 79% of our sixteenth notes were followed
immediately by a sixteenth rest — 1120 pairs, the most common adjacency on the
page — and **1069 of them were drums**. A drum hit is an impulse: the notehead
marks an attack and its written length is nominal, so it is engraved out to the
next attack. Ours sat at the grid floor with the remainder of the beat left as a
rest. The kit is now 100% eighths against the published transcription's 100%
eighths.

**The subdivision fix from the previous session had never run.** `subdivision`
reached only `_chord_events_gridded`; both `_offset_quarter_length` calls on the
path that writes every onset and duration used the default sixteenth lattice.
The same two-site trap that cost a 39-minute run, in a different file.

**Anechoic Ma was gating one staff.** It measures resonance vs genuine silence
on the trailing gap of every pitched note of every run, and only
`piano_reduction` read it. Extracted to `output/ma_legato.py`; it now gates
every part, keeping the percentile calibration that module had already paid for
(resonance occupies ~[0.23, 0.72], so absolute thresholds are inert).

## XXV.2 The register of verdicts computed and never read

This session added **four** entries, bringing the register to seven:

| verdict | written | read by | fixed |
|---|---|---|---|
| `tuplet_divisor` | every run | nothing | earlier |
| `RANGE_ANNOTATION_KIND` | since 2026-08-06 | nothing | earlier |
| CREPE's bass read | every run | counted, discarded | **yes** |
| Ma's trailing gap | every run | piano staff only | **yes** |
| `stability.key` | every run, with reasoning | confidence only | **yes** |
| repair-pass write condition | — | named its keys | **yes** |
| `DURATION_ANNOTATION_KIND` | every run | nothing | open |

The key one is the sharpest. `analyze_key_stability` wrote *"Bbm spells notes
this recording does not use — it covers 92% of what was played against Ab's
97%, so Ab is reported"* into the trail on every single run, and the Conductor
took its confidence and dropped its key. The page said Bbm throughout.

## XXV.3 The bass, resolved after three wrong answers

| attempt | claim | outcome |
|---|---|---|
| 1 | reads 12 semitones flat | **wrong** — compared a transposing part's *written* pitch to our *sounding* pitch |
| 2 | no octave problem at all | **wrong** — true of the floor, false of the notes |
| 3 | 27.6% wrong octave, running *upward* | **confirmed** against the human bass |

The error is Basic Pitch locking onto the second harmonic, which is precisely
what a monophonic f0 tracker is positioned to catch. Time-aligned per note:
72.4% → 91.1%; as engraved, 63.5% → 94.6%.

**Why it took three tries is the durable finding.** Every aggregate metric tried
first was structurally unable to answer: notes-in-range moved 0.8 points,
impossible-note counts moved zero, and pitch-class distance was *identical
before and after by construction*, because an octave correction preserves pitch
class. That number was reported as evidence. See audit lesson 12.

## XXV.4 Still open

1. **Pitched parts at 7.4% sixteenths** against the human's 0%, and 1443 rests
   against 1037. The kit fix does not touch this. 273 Ma fills are still
   declined for reasons not yet diagnosed.
2. **Unrepresentable gaps.** A confounded run produced 33 rests music21
   expressed as 12:7 and 12:11. `musicxml_repair` now rewrites such rests, but
   only when the duration decomposes exactly into ordinary values — it refuses
   to approximate, so these survive. The real fix is upstream: stop creating a
   gap whose length cannot be written.
3. **Detect-once-attribute-after** — still the largest available change, still
   not run end to end.
4. **Chopin's tempo** 26% fast (70 vs 55.4 implied).
5. **`detect_pickup`** written, exported, never called — an eighth register
   entry waiting to happen.
