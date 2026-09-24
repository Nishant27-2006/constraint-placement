# Data provenance

Every byte the benchmark depends on is public, key-free, and re-downloadable by
`bash scripts/download_data.sh`. No account, no registration, no email approval.

---

## Sources actually used

| Source | URL pattern | Observed | Licence |
|---|---|---|---|
| **EIA-930 Hourly Electric Grid Monitor, BALANCE** | `https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{YEAR}_{Jan_Jun\|Jul_Dec}.csv` | 12 files, **502 MB**, all HTTP 200 | US Government work — public domain |
| **Power Grid Lib — OPF** `pglib_opf_case118_ieee` | `https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/pglib_opf_case118_ieee.m` | 80,336 B, HTTP 200 | **CC-BY-4.0** |

**Why pglib rather than MATPOWER's own `case118.m`.** MATPOWER's licence
explicitly disclaims its case data files, whereas pglib releases its copies
under CC-BY-4.0. It also matters empirically: MATPOWER's `case118.m` has
**0 / 186** finite branch thermal ratings, so line-limit constraints would be
vacuous. pglib's copy has **186 / 186**.

## Sources verified as working but not used

Checked and reachable; held in reserve for an out-of-distribution suite or a
spatially-resolved solar upgrade. Details in `research/data_sources.md`.

* **Open Power System Data** 60-min time series (CC-BY-4.0, ~130 MB, European,
  ENTSO-E-derived without needing an ENTSO-E token). *Our copy is currently
  truncated at 9.5 MB — the server returned HTTP 200 for a partial body. Re-run
  the download before using it.*
* **NREL SIND** per-plant 5-minute solar with lat/lon and nameplate MW, via the
  Wayback raw mirror. Would let Suite B use genuinely spatially-diverse solar
  injections instead of Dirichlet-shared profiles.
* GEFCom2014 (126 MB, all four tracks), NYISO, CAISO OASIS, ERCOT MIS, Elia,
  NESO, Energinet, UCI 235/321, PVDAQ, ResStock/ComStock.

**Operational note:** `nrel.gov` and `developer.nrel.gov` were **unresolvable in
DNS** during verification (NOERROR / ANSWER: 0 from 8.8.8.8, 1.1.1.1, 9.9.9.9).
No `nrel.gov` hostname appears in the pipeline; NREL-derived data is reached via
the AWS OEDI buckets or a Wayback mirror, both verified.

---

## Decisions taken about the record, and why

### 1. Window: 2019-01-02 → 2024-06-30
EIA changed the BALANCE schema in 2024 H2, splitting solar and wind into
`... without Integrated Battery Storage` and `... with Integrated Battery
Storage` (44 → 65 columns). Splicing two definitions of the same channel would
put a documented discontinuity inside the training set. We use the homogeneous
record instead and state the cost: half a year of data discarded.

### 2. Missing hours are dropped, never imputed
A day with any missing hour is discarded. Imputation would manufacture values
that then appear to satisfy — or violate — the very identities under test.
Yield: 1,990 common complete days across all six balancing authorities.

### 3. Structurally absent channels are dropped per-area, not zero-filled
MISO never reports petroleum; BPAT reports no coal; CISO's hydro and `other`
series are >15 % missing. These are dropped for that area rather than filled
with zeros, which would inject a false "measurement" into an aggregation
identity. Channel counts therefore differ by area (7–9), which the constraint
builder handles per-area.

---

## Two priors the original draft asserted that the data **falsifies**

Measured with `hfm/data/eia930.py::verify_identities`, adjusted columns, 2023:

### (a) EIA's published Net Generation ≠ its published fuel sum

| BA | max abs residual (MW) | % of hours exactly consistent |
|---|---|---|
| PJM | **32,800** | 0.0 % |
| SWPP | 383 | 48.2 % |
| BPAT | 88 | 7.0 % |
| ERCO | 50 | 29.3 % |
| CISO | 3 | 47.0 % |
| MISO | 3 | 44.9 % |

The demand/net-generation/interchange balance is no better: CISO max residual
**6,910 MW**, exactly consistent in **0.0 %** of hours.

### (b) Solar is not zero at night
CISO 2023, local time: **3,326 negative solar values**, minimum **−87 MW**, and
*no* hour of the day at which solar is exactly zero across every day of the
year. ERCOT reports up to 24 MW at local midnight.

**Consequence.** Neither prior is imposed. Every equality constraint in the
benchmark is either an aggregation identity over measured components — exact by
construction, and labelled as such — or a law of the DC/AC network model. The
falsified priors are reported in the paper as a finding: physical intuition
about what "must" hold is not a substitute for measuring the record.

---

## What is real and what is stylised

**Real:** measured hourly demand, solar and wind for six US balancing
authorities over five and a half years, with their true cross-area correlation
structure preserved because every channel is read from the same rows of the same
file (stitching sources is exactly what destroys that structure).

**Stylised, and stated as such in the paper:** the transmission network is a
standard test case, not a reconstruction of any real system; system demand is
allocated across buses in proportion to the case's nominal demand; renewable
siting uses designated buses with Dirichlet-drawn shares. Suite B's *constraints*
are exact laws of the stated DC model, and the *stochasticity* is measured, but
the topology is a benchmark network.
