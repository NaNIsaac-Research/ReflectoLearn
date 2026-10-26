# H-cal2 results: count dispersion in 100 repeated public XRR scans

xrr-policy lane, 2026-10-25. Registration: `PREREG-HF-HW2.md` section 6 (ReflectoLearn
`research/fisher-global` d1f164c) and its amendment 1 (5d3a9f5, written after the pre-checks and
before the dispersion statistic was computed). Code and output: same branch, 5bf5a9d,
`experiments/005_fisher_global/hcal2.py` (`precheck`, `dispersion`), `hcal2_extra.py` (post hoc);
`docs/fisher_global/hcal2_precheck.json`, `hcal2.json`, `hcal2_extra.json`.

Data: Zenodo 10.5281/zenodo.18548583 (Haberl et al., J. Appl. Cryst. 59, 2026), one file,
`9.95 ms_50 Hz triangle.ort`, 17 396 652 bytes, md5 43c35b4c… (matches the record), stored read-only at
`$DATA/xrr-public/zenodo-18548583/`. Scan 734: Au 300 Å / Cr 80 Å / Si, PETRA III P08 at 18 keV,
Eiger2 X 1M, sample on a galvanometer at 50 Hz (triangle), 100 scans of 9.95 ms. No fit, no model.

## 1. Pre-checks (condition 4), before any dispersion number

| check | outcome [fact] |
|---|---|
| 1 counts | 100 data sets of 839 points; `raw` integer at every point (largest non-integer part 0), minimum 1 |
| 2 grid | Q identical across all 100 scans at every index (spread 0); Q 0.039–0.409 Å⁻¹, median step 4.4 × 10⁻⁴ |
| 3 normalisation, exposure | `af` and `etfp` the same across scans at every point (relative spread ≤ 10⁻⁶). `N` constant within each scan, different between scans (mean 2.14 × 10⁶ photons/s, relative SD 5.8 %, range 1.83–2.42 × 10⁶); correlation of `N` with the scan's first-10-point raw sum 0.9998. `bg` differs across scans (median relative spread 2.5 %) |
| 4 drift | scan totals: mean 1.15 × 10⁵ counts, slope 7.7 ± 7.2 counts per scan (0.7 % over 100 scans, within 1.1 SE), lag-1 autocorrelation 0.05; dispersion index of the 100 totals 37 |

- [fact] Read literally, registered pre-check 3 excludes every point (because `N` differs across
  scans) and the primary statistic would be **not testable**. Amendment 1 (before any φ was computed)
  limits the exclusion to `af` and `etfp`: by the header (R "normalized to square mean of the first
  10 data points") and the 0.9998 correlation, `N` is computed from each scan's own counts and is not
  applied to `raw`. The verdict below is under amendment 1; under the literal rule there is none.
- [fact] No trend over the 100 scans; the scans are not correlated with their neighbours. But the
  total counts vary from scan to scan 37 times more than Poisson allows: the incident intensity, or
  the angular coverage of a scan, fluctuates between scans of 9.95 ms.

## 2. Registered verdict

All 839 points have a mean of at least 10 counts (10 to about 10³), none excluded.

| statistic | value | Poisson expectation (n = 100) |
|---|---|---|
| fraction φ > 1.2 | **0.642** | 0.085 |
| fraction φ in [0.8, 1.2] | 0.347 | 0.843 |
| median φ | **1.34** | 0.993 |
| fraction φ < 0.8 | 0.011 | 0.071 |
| fraction with one-sided χ² p < 0.01 | 0.478 | 0.01 |

**H-cal2 supported** (rule: φ > 1.2 at ≥ 50 % of eligible points). The raw counts of this file are
overdispersed relative to Poisson over most of the curve.

## 3. Secondary and post-hoc descriptions

- [fact, registered secondary] By mean count: 10–100 counts (449 points) median φ 1.16 [IQR 1.04–1.36],
  44 % above 1.2; 100–1000 counts (390 points) median φ 1.62 [1.35–1.89], 88 % above 1.2.
- [fact, registered secondary] With a per-scan factor from the totals divided out (Pearson form), the
  median is 1.31 and 62 % are above 1.2: removing the scan-to-scan intensity variation changes little.
- [fact, post hoc, `hcal2_extra.json`] Fitting the excess variance s² − x̄ as a x̄² + b (dx̄/dq)²
  (non-negative, weighted by the sampling spread of s²): the x̄² term alone gives a relative noise SD
  of 4.6 %, the slope term alone fits worse, the two together hardly better than x̄² alone, and none of
  the fits describes the pattern well (weighted χ² per point 2.7, weighted R² below 0). The rank
  correlation between the relative excess (s² − x̄)/x̄² and (d ln x̄/dq)² is 0.27.
- [not checked] Whether Haberl et al. report this dispersion: the article was not reachable
  (RESEARCH-DESIGN section 8).

## 4. Interpretation [inference], kept apart from the counts

1. **A Poisson likelihood with one scale per scan does not describe these counts.** The excess grows
   with the count level (φ − 1 is about 0.16 at tens of counts and 0.6 at hundreds), which is the
   signature of a noise source proportional to the signal rather than of counting noise. At 10³
   counts a φ of 1.6 widens the true spread by √1.6 ≈ 1.27 against Poisson; a Poisson deviance would
   then read about 1.6 times too large per point, and Poisson intervals would be too narrow.
2. **The source is not identified.** Scan-to-scan intensity variation is present (totals) but
   dividing it out leaves the per-point excess almost unchanged, so most of the excess varies within
   a scan, between frames. Candidates: intensity fluctuation at the frame time scale (the Eiger ran at
   up to 3800 frames/s), jitter of the galvanometer angle assigned to a frame, and detector count-rate
   correction. The post-hoc decomposition does not separate them (neither model fits), and the weak
   correlation with the curve's slope leaves angle jitter possible but not shown.
3. **Scope.** One sample, one exposure (9.95 ms), one fast-scan instrument. This is not a laboratory
   diffractometer with step scans and seconds per point, and it says nothing about such data. It does
   show that "Poisson counts" is a property to check per instrument before a likelihood calibration
   study (question B) treats it as given. In R2a and X2e the counts are Poisson by construction
   (inverse crime), so their deviance scale is the ideal one.

## 5. Limits

- One file of six exposures; the shorter exposures (213 µs–2.45 ms, lower counts) were not
  downloaded (budget 10–17 MB).
- Counts are summed over a detector ROI; summing keeps Poisson statistics, but pixel-level effects
  (charge sharing, count-rate correction) are not visible in the file.
- The registered rule fixed thresholds on φ only; it does not say which noise model is right.
- Amendment 1 was made after the pre-checks were seen; it was necessary for any primary verdict and
  is reported with the literal outcome.
