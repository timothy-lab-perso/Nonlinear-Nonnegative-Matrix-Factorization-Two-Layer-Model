# Nonlinear-Nonnegative-Matrix-Factorization-Two-Layer-Model
This repository contains the code and experiments for my M1 research internship on Nonlinear Nonnegative Matrix Factorization (NNMF). 

Code and experiments:

The first four showcase the algorithms and provide the graphs depicting the synthetic test I made:

-testH2.py simulates the algorithm in part 2.1.1 where only H2 is updated

-testW2.py works similarly for W2

-testW2H2.py simulates the algorithm in part 2.2 where W2 and H2 are updated simultaneously / alternatingly

-testWW2H2.py simulates the full algorithm of part 2.3 where the three factors are updated alternatingly

The last provides the reconstructed faces of CBCL:

-testWW2WH2-CBCL.py simulates classical NMF using a block coordinate descent algorithm, as well as the naive two step approach mentionned in the last.
It also contains an implementation of the aternating NNLS-subgradient descent algorithm, but won't execute it due to issues mentionned in the report


Dataset:
- cbcl-faces.zip is a compressed version of the CBCL dataset a standard benchmark for NMF:
- 19×19 grayscale face images (361 pixels per image).
- Flattened column-wise for NMF: X ∈ ℝ^{361×n}.
Source: http://cbcl.mit.edu/software-datasets/FaceData2.html
