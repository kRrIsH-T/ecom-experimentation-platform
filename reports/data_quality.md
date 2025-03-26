# Data quality report

_Source: **kaggle** · generated 2026-11-01 15:47 UTC by `ecom ingest`_

- Rows: **109,950,743** over **61** days (2019-10-01 00:00:00 UTC → 2019-11-30 23:59:59 UTC)
- Approx. distinct users / sessions / products: 5,927,026 / 22,303,598 / 207,969
- Exact duplicate rows: **130,739** (0.119%) - dropped in silver
- Negative prices: **0**; zero prices: **256,761** - dropped in silver
- Price p1 / p50 / p99 / max: 5.79 / 164.93 / 1,688.59 / 2,574.07
- Unknown event types: 0

## Event type distribution

| event_type | rows | share |
|---|---:|---:|
| view | 104,335,509 | 94.89% |
| cart | 3,955,446 | 3.60% |
| purchase | 1,659,788 | 1.51% |

## Null rates

| column | null rate |
|---|---:|
| event_time | 0.000% |
| event_type | 0.000% |
| product_id | 0.000% |
| category_id | 0.000% |
| category_code | 32.209% |
| brand | 13.944% |
| price | 0.000% |
| user_id | 0.000% |
| user_session | 0.000% |
| event_date | 0.000% |

Missing `category_code` and `brand` are expected in this dataset (many products are uncategorised); they are kept and mapped to `unknown`. Rows missing `user_session` cannot be sessionised and are dropped in silver.
