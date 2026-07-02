# DECAF Custom SCM Results

This final lightweight bundle contains the 500 validated DECAF result rows for
Custom SCM: 100 paired seeds at each training size in
`20,50,100,200,500`.

Validation guarantees:

- the seed set exactly matches the canonical cleaned TabPFN Custom SCM seed set
  at every training size;
- zero duplicate keys;
- zero failed rows;
- finite CMD, kMTVD, and raw NNAA values.

Raw chunk CSVs, synthetic NPZ files, and trained artifacts remain outside Git.
