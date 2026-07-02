# DECAF configuration provenance

Official repository:
`https://github.com/vanderschaarlab/DECAF`

Pinned upstream commit:
`3fbf35369641f87afde7999b932291a5d30ccd7b`

The adapter follows the DECAF paper implementation details:

- all continuous variables are standardized;
- generator/discriminator hidden width is `2d`, where `d` is the number of
  columns;
- learning rate is `0.001`;
- training runs for 50 epochs;
- the known Custom SCM DAG is supplied to the generator.

The current official repository updates the generator and discriminator once
per training batch. This differs from the paper text, which describes one
generator update per ten discriminator updates. The adapter does not patch the
upstream training loop; this discrepancy is recorded in run metadata.

