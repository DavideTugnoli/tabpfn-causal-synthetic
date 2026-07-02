# TabularARGN Five-Dataset Results

## Status: final

This bundle preserves the paired analysis reported in the paper appendix. For
`csuite_mixed_confounding`, `N=20`, the 36 historical non-finite CMD values
caused by constant generated columns were recomputed from the exact same saved
synthetic datasets using the documented deterministic constant-correlation
handling. No model was retrained, no seed or synthetic dataset was replaced,
and the historical sources remain untouched.

The regenerated analysis now has 100 finite paired metrics in every cell.
Every CMD, kMTVD, and NNAA comparison remains Holm-significant in TabPFN's
favor in all 25 dataset/training-size cells.

This is the final five-dataset TabularARGN analysis:

- five datasets;
- five comparison training sizes (`N <= 500`);
- 100 paired seeds per dataset/training-size cell;
- CMD, kMTVD, and NNAA comparisons against both DAG-aware and
  vanilla-original TabPFN.

The primary saved analysis used distance-to-0.5 NNAA. The separate
`raw_nnaa_pairwise_audit.csv` verifies that the reported `25/25` significant
TabPFN wins also hold for raw NNAA.
