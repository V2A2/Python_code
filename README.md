# Generalized ProbStar: Distribution-Aware Probabilistic Verification

This repository contains the experimental code for our work on
**Generalized ProbStar (GPS)**, a distribution-aware extension of
ProbStar for probabilistic neural-network verification under
non-Gaussian and multimodal input uncertainty.

The experiments use the **ACAS Xu** benchmark and compare three
predicate-distribution models:

-   **True distribution:** a truncated two-component 5D Student-(t)
    mixture on (\[-1,1\]\^5).
-   **ProbStar (PS):** a single truncated 5D Gaussian approximation.
-   **Generalized ProbStar (GPS):** a truncated 10-component Gaussian
    mixture model (GMM) fitted using expectation-maximization (EM).

The experimental pipeline separates exact reachability from probability
computation:

``` text
ACAS Xu network
      |
      v
Exact-Star propagation
      |
      v
Saved reachable Stars
      |
      v
Unsafe-region intersection
      |
      +------------------+------------------+
      |                  |                  |
      v                  v                  v
 True distribution    ProbStar          Generalized
 Student-t mixture    Gaussian           ProbStar GMM
      |                  |                  |
      +------------------+------------------+
                         |
                         v
        Unsafe probability / error / runtime / memory
```

## Repository Structure

### Distribution modeling

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `em_algorithm.py`                   Generates samples from the true
                                      truncated Student-(t) mixture, fits
                                      the single-Gaussian ProbStar
                                      approximation and the 10-component
                                      GMM used by Generalized ProbStar,
                                      and evaluates distribution
                                      mismatch.

  `distribution_mismatch.py`          Produces the marginal-distribution
                                      comparison used to visualize the
                                      mismatch between the true
                                      distribution, ProbStar, and
                                      Generalized ProbStar.

  `distribution_marginals_3x2.png`    Marginal-density comparison for the
                                      five predicate variables.

  `accuracy_result.py`                Generates the verification-error
                                      comparison figure for the selected
                                      Property-2 networks.

  `accuracy.png`                      Comparison of the absolute
                                      unsafe-probability errors of
                                      ProbStar and Generalized ProbStar.
  -----------------------------------------------------------------------

### ACAS Xu reachability

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `P2_propagation.py`                 Exact-Star propagation for the six
                                      selected ACAS Xu Property-2
                                      networks.

  `P3_propagation.py`                 Exact-Star propagation for the
                                      Property-3 ACAS Xu experiments.

  `P4_propagation.py`                 Exact-Star propagation for the
                                      Property-4 ACAS Xu experiments.

  `ACASXU/`                           ACAS Xu neural-network benchmark
                                      files used by the propagation
                                      scripts.
  -----------------------------------------------------------------------

The propagation stage is purely geometric. It propagates exact Stars
through the affine and ReLU layers and saves the resulting reachable
Stars. No probability distribution is required during this stage.

### Unsafe-probability computation

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `P2_unsafe_compute.py`              Computes unsafe probabilities for
                                      the six selected Property-2
                                      networks using the true, PS, and
                                      GPS predicate distributions.

  `P3_unsafe_compute.py`              Computes Property-3 unsafe
                                      probabilities from saved exact
                                      reachable Stars.

  `P4_unsafe_compute.py`              Computes Property-4 unsafe
                                      probabilities from saved exact
                                      reachable Stars.
  -----------------------------------------------------------------------

These scripts do **not** repeat network propagation. They load the saved
reachable Stars, intersect them with the corresponding unsafe output
region, and evaluate the probability under each of the three fixed
predicate distributions.

## Distribution Approximation Experiment

The ground-truth uncertainty model is a truncated two-component 5D
Student-(t) mixture. The main Generalized ProbStar approximation uses a
**10-component GMM** fitted by EM.

For the reported fitting run:

  Quantity                                          Result
  ------------------------------------------ -------------
  Number of samples                                 10,000
  GMM components                                        10
  EM converged                                         Yes
  EM iterations                                        875
  ProbStar fitting time                         0.000340 s
  ProbStar peak fitting memory                 0.445190 MB
  Generalized ProbStar fitting time            53.280812 s
  Generalized ProbStar peak fitting memory     5.187165 MB
  Reported statistical-test (p)-value           **0.0006**

The distribution experiment shows that a single Gaussian does not
capture the multimodal structure of the true input distribution well,
whereas the GMM provides a substantially closer approximation.

## Distribution Marginals

The following figure compares the one-dimensional marginal distributions
of the true Student-(t) mixture, the single-Gaussian ProbStar
approximation, and the GMM-based Generalized ProbStar approximation.

![Marginal distribution comparison](distribution_marginals_3x2.png)

The figure is intended as a visual comparison of the five
one-dimensional marginals. The verification experiment itself is
performed in the full five-dimensional predicate space.

## ACAS Xu Verification Results

We evaluate ProbStar and Generalized ProbStar on ACAS Xu Properties 2,
3, and 4. The table below reports the number of final reachable Stars,
total runtime, memory usage, unsafe probabilities, and absolute
verification errors.

The unsafe-probability error is

\[ e\_{`\mathrm{ver}`{=tex}} = \|p - p\_{`\mathrm{true}`{=tex}}\|, \]

where (p\_{`\mathrm{true}`{=tex}}) is evaluated under the true truncated
Student-(t) mixture.

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Property     Network      #Stars    Time PS   Time GPS  Memory PS Memory GPS   (p\_{`\mathrm{PS}`{=tex}})   (p\_{`\mathrm{GPS}`{=tex}})   (p\_{`\mathrm{true}`{=tex}}) PS Error      GPS
                                          (s)        (s)       (MB)       (MB)                                                                                                       Error
  ---------- --------- ----------- ---------- ---------- ---------- ---------- ---------------------------- ----------------------------- ------------------------------ -------- --------
  P2               2-2     472,257    1.640e5    1.641e5    9739.40    9786.32                       2.655%                        3.458%                         3.445%   0.790%   0.014%

  P2               3-5     252,793    5.764e4    5.769e4    4605.07    4651.99                       3.060%                        2.431%                         2.473%   0.587%   0.042%

  P2               4-5   1,003,886    2.880e5    2.881e5   21023.84   21070.76                       2.291%                        3.704%                         3.555%   1.265%   0.149%

  P2               5-1     475,299    1.671e5    1.672e5   10198.36   10245.28                       0.297%                        0.228%                         0.240%   0.057%   0.012%

  P2               5-4     402,853    1.301e5    1.301e5    8191.54    8238.46                       0.101%                        0.028%                         0.030%   0.071%   0.002%

  P2               6-5     652,417    2.193e5    2.193e5   13580.24   13627.16                       2.837%                        2.889%                         2.947%   0.110%   0.058%

  P3               1-1      70,631   8098.693   8151.974    1123.77    1128.51                           0%                            0%                             0%       0%       0%

  P3               1-2      39,529   8533.474   8586.755     686.03     690.77                           0%                            0%                             0%       0%       0%

  P3               4-3      20,746   4748.192   4801.473     397.38     402.12                           0%                            0%                             0%       0%       0%

  P3               3-6       1,857    759.773    813.054      39.49      44.23                           0%                            0%                             0%       0%       0%

  P3               1-7         500    208.800    262.089      62.39      67.43                         100%                          100%                           100%       0%       0%

  P3               1-8         393    172.555    225.895      64.82      69.56                         100%                          100%                           100%       0%       0%

  P3               1-9         290    111.301    164.443      58.42      63.16                         100%                          100%                           100%       0%       0%

  P4               1-4       1,184    285.757    338.026      21.83      67.43                           0%                            0%                             0%       0%       0%

  P4               1-5       6,608   1819.437   1869.768     133.67     181.42                           0%                            0%                             0%       0%       0%

  P4               1-6       4,443    853.274    908.955      79.15     124.27                           0%                            0%                             0%       0%       0%

  P4               1-7         642    128.514    185.715      11.34      61.26                         100%                          100%                           100%       0%       0%

  P4               1-8         397    166.478    213.729       7.51      51.43                         100%                          100%                           100%       0%       0%

  P4               1-9         471    124.378    179.259       8.43      57.15                         100%                          100%                           100%       0%       0%
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Accuracy on the nontrivial Property-2 cases

For all six selected P2 networks, Generalized ProbStar gives a smaller
absolute unsafe-probability error than the original single-Gaussian
ProbStar model.

![Verification error comparison](accuracy.png)

The largest reduction occurs for network 4-5, where the absolute error
decreases from **1.265%** to **0.149%**. For network 2-2, the error
decreases from **0.790%** to **0.014%**. The remaining selected networks
show the same trend.

For P3 and P4, the reported cases have unsafe probabilities of either 0%
or 100% under all three distributions. These cases therefore confirm the
same qualitative verification result across the tested uncertainty
models, but they are not used to quantify the distribution-modeling
accuracy improvement.

## Scalability

The reachability geometry is identical for PS and GPS. The difference
appears in distribution fitting and probability evaluation.

For large reachability problems, exact ReLU reachability dominates the
total runtime and memory cost. This is visible in the large P2 cases.
For example, network 4-5 generates more than one million Stars, and the
total runtimes are (2.880`\times10`{=tex}\^5) s for PS and
(2.881`\times10`{=tex}\^5) s for GPS. The corresponding memory
measurements are 21023.84 MB and 21070.76 MB.

For smaller problems, the additional probability-modeling cost of GPS is
more visible. The results therefore support **similar scalability on
reachability-dominated verification problems**, rather than claiming
identical computational cost.

## Running the Experiments

Install the Python dependencies used by the scripts, including NumPy,
SciPy, scikit-learn, Matplotlib, and psutil.

A typical workflow is:

``` bash
# 1. Fit/evaluate the uncertainty models
python3 em_algorithm.py

# 2. Generate the distribution-comparison figure
python3 distribution_mismatch.py

# 3. Exact-Star propagation
python3 P2_propagation.py
python3 P3_propagation.py
python3 P4_propagation.py

# 4. Compute unsafe probabilities from the saved reachable Stars
python3 P2_unsafe_compute.py
python3 P3_unsafe_compute.py
python3 P4_unsafe_compute.py

# 5. Generate the verification-error figure
python3 accuracy_result.py
```

The propagation scripts can require substantial computation time and
memory because exact ReLU reachability may generate hundreds of
thousands or more than one million Stars.

## Notes on Reproducibility

-   Reachability and probability evaluation are intentionally separated
    so that the same exact reachable sets can be evaluated under
    different predicate distributions without repeating neural-network
    propagation.
-   The ACAS Xu network files used by the experiments are included under
    `ACASXU/`.
-   The reported probabilities are computed directly on the saved
    exact-Star geometry; the P3/P4 probability scripts use deterministic
    numerical quadrature rather than Monte Carlo sampling.
-   Runtime and memory measurements depend on hardware and software
    configuration, so exact resource measurements may differ across
    systems.

## Citation

If you use this code, please cite the associated paper. The final paper
citation will be added here after publication.

``` bibtex
@inproceedings{generalizedprobstar,
  title     = {Distribution-Aware Probabilistic Verification of Neural Networks with Generalized ProbStars},
  author    = {Zhang, Yizhong and Tran, Dung},
  note      = {Citation information to be updated after publication}
}
```
