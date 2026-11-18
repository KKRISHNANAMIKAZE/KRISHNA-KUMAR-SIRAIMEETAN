# Data

`elec.csv` is the **Electricity Market** data set (often called *elec2*), a
standard benchmark for learning from data streams.

* 45,312 rows recorded every 30 minutes from May 1996 to December 1998 in the
  New South Wales (Australia) electricity market.
* 6 numeric features, already scaled to [0, 1]: `period` (time of day),
  `nswprice`, `nswdemand`, `vicprice`, `vicdemand`, `transfer`.
* Label `class`: 1 if the price went **up** compared with the last 24 hours,
  0 if it went **down** (57.5% down, 42.5% up).
* The relationship between features and label changes over time (concept
  drift), which makes it a realistic test for streaming models.

Source: the `scikit-multiflow/streaming-datasets` repository on GitHub
(this version leaves out the `date` and `day` columns).
Original reference: M. Harries, *Splice-2 comparative evaluation: Electricity
pricing*, University of New South Wales, 1999.
