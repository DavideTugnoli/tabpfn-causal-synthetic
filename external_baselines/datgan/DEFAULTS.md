# DATGAN configuration provenance

Official repository:
`https://github.com/glederrey/DATGAN`

Pinned upstream commit:
`103929336703b91794f85c6160a8338b30eb158a`

The DATGAN paper trains every model for 1,000 epochs with batch size 500. The
official implementation uses `drop_remainder=True`, so batch size 500 performs
zero training steps when `N < 500`.

For the low-sample protocol, the adapter applies the minimal explicit
feasibility rule `batch_size = min(500, N)`. At `N=500` this exactly matches the
paper configuration. All other constructor parameters remain at official
repository defaults, and the known Custom SCM DAG is supplied.

