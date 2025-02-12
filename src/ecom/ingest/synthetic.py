"""Synthetic clickstream generator that mirrors the Kaggle multi-category store dataset.

The real dataset (``mkechinov/ecommerce-behavior-data-from-multi-category-store``) needs a
Kaggle account and 14 GB of disk. This generator writes CSVs with the exact same schema and
row format (gzip-compressed, one file per month and user chunk, e.g. ``2019-Oct-000.csv.gz``)
so the whole pipeline runs anywhere, and it
plants realistic structure the downstream analysis can recover:

* a persistent per-user purchase intent plus a weekly AR(1) "in-market" state, which drives
  both activity and conversion (so recent behaviour predicts next-week purchases),
* day-of-week and hour-of-day patterns, a mid-November sale with a traffic and conversion spike,
* per-user price sensitivity (price-sensitive users cart the cheapest item they viewed),
* data-quality defects: exact duplicates, zero prices, missing brand / category_code / session.

Everything is vectorised with NumPy and seeded, so a given config always yields the same data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ecom.config import Config

log = logging.getLogger(__name__)

CSV_COLUMNS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]

# category_code, traffic share, median price, price log-sd, conversion log-multiplier, brands
CATEGORIES: list[tuple[str, float, float, float, float, list[str]]] = [
    (
        "electronics.smartphone",
        0.22,
        260,
        0.65,
        0.25,
        ["samsung", "apple", "xiaomi", "huawei", "oppo", "vivo", "realme"],
    ),
    (
        "electronics.audio.headphone",
        0.05,
        55,
        0.80,
        0.10,
        ["apple", "samsung", "jbl", "sony", "xiaomi", "sennheiser"],
    ),
    ("electronics.video.tv", 0.04, 420, 0.55, -0.15, ["samsung", "lg", "sony", "philips", "tcl"]),
    (
        "electronics.clocks",
        0.04,
        140,
        0.85,
        -0.05,
        ["apple", "samsung", "garmin", "casio", "xiaomi"],
    ),
    (
        "computers.notebook",
        0.06,
        680,
        0.45,
        -0.30,
        ["lenovo", "hp", "asus", "acer", "apple", "dell"],
    ),
    ("computers.desktop", 0.02, 520, 0.50, -0.45, ["hp", "lenovo", "dell", "acer"]),
    (
        "appliances.kitchen.washer",
        0.04,
        310,
        0.40,
        0.00,
        ["samsung", "lg", "bosch", "indesit", "beko"],
    ),
    (
        "appliances.kitchen.refrigerators",
        0.04,
        470,
        0.45,
        -0.10,
        ["samsung", "lg", "bosch", "atlant", "beko"],
    ),
    (
        "appliances.environment.vacuum",
        0.04,
        140,
        0.60,
        0.05,
        ["samsung", "philips", "xiaomi", "bosch", "karcher"],
    ),
    ("apparel.shoes", 0.05, 70, 0.60, -0.40, ["nike", "adidas", "puma", "reebok", "skechers"]),
    ("furniture.living_room.sofa", 0.02, 360, 0.60, -0.60, ["ikea", "divano", "lazurit"]),
    ("kids.toys", 0.02, 30, 0.80, -0.15, ["lego", "hasbro", "mattel", "chicco"]),
    ("sport.bicycle", 0.02, 310, 0.60, -0.40, ["stels", "forward", "merida", "trek"]),
    ("construction.tools.drill", 0.02, 90, 0.60, -0.05, ["bosch", "makita", "dewalt", "metabo"]),
    ("auto.accessories.player", 0.02, 85, 0.60, -0.20, ["pioneer", "kenwood", "sony", "jvc"]),
    ("accessories.bag", 0.02, 40, 0.70, -0.45, ["samsonite", "xiaomi", "wenger", "pierre"]),
    ("", 0.28, 45, 1.00, -0.20, []),  # products without a category_code (common in the real data)
]
UNCODED_CATEGORY_IDS = 40
DOW_TRAFFIC = np.array([0.95, 0.95, 0.97, 1.00, 1.02, 1.08, 1.10])  # Mon..Sun
SECONDS_PER_DAY = 86_400


@dataclass
class GenerationSummary:
    n_users: int
    n_products: int
    n_events: int
    n_sessions: int
    events_by_type: dict[str, int]
    files: list[str]
    seed: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _long_tail_brands(rng: np.random.Generator, n: int) -> list[str]:
    syll = ["lu", "ve", "ra", "to", "ki", "mo", "sa", "ne", "do", "ri", "ta", "lo", "vi", "ka"]
    names = {"".join(rng.choice(syll, size=rng.integers(2, 4))) for _ in range(n * 2)}
    return sorted(names)[:n]


@dataclass
class Catalog:
    product_id: np.ndarray
    category: np.ndarray  # index into CATEGORIES
    category_id: np.ndarray
    brand: np.ndarray  # object array, "" = missing
    price: np.ndarray
    cdf_by_category: list[np.ndarray]  # cumulative popularity per category
    index_by_category: list[np.ndarray]
    median_price: np.ndarray  # per category


def build_catalog(rng: np.random.Generator, n_products: int) -> Catalog:
    shares = np.array([c[1] for c in CATEGORIES])
    # Product counts are flatter than traffic: niche categories still carry many SKUs.
    counts = np.maximum(50, (shares**0.7 / (shares**0.7).sum() * n_products).astype(int))
    category = np.repeat(np.arange(len(CATEGORIES)), counts)
    n = len(category)
    product_id = 1_000_000 + rng.choice(99_000_000, size=n, replace=False)
    tail = _long_tail_brands(rng, 120)
    brand = np.empty(n, dtype=object)
    price = np.empty(n)
    category_id = np.empty(n, dtype=np.int64)
    cdfs, indices = [], []
    base_cat_id = 2_053_013_552_000_000_000
    for c, (_, _, median, sd, _, brands) in enumerate(CATEGORIES):
        idx = np.nonzero(category == c)[0]
        indices.append(idx)
        k = len(idx)
        pool = brands + tail[c * 7 : c * 7 + 7]
        weights = 1.0 / np.arange(1, len(pool) + 1) ** 1.1
        chosen = rng.choice(len(pool), size=k, p=weights / weights.sum())
        b = np.array(pool, dtype=object)[chosen]
        b[rng.random(k) < 0.13] = ""  # missing brand
        brand[idx] = b
        premium = np.where(b == "apple", 1.45, 1.0)
        price[idx] = np.round(np.exp(rng.normal(np.log(median), sd, size=k)) * premium, 2)
        if CATEGORIES[c][0]:
            category_id[idx] = base_cat_id + c * 1_000_003
        else:
            category_id[idx] = base_cat_id + 900_000_000 + rng.integers(0, UNCODED_CATEGORY_IDS, k)
        pop = rng.permutation(1.0 / np.arange(1, k + 1) ** 1.05)
        cdfs.append(np.cumsum(pop) / pop.sum())
    median_price = np.array(
        [np.median(price[category == c]) for c in range(len(CATEGORIES))], dtype=float
    )
    return Catalog(product_id, category, category_id, brand, price, cdfs, indices, median_price)


@dataclass
class Users:
    user_id: np.ndarray
    base_logit: np.ndarray
    activity: np.ndarray  # expected sessions per week when in an average state
    price_sens: np.ndarray
    evening_pref: np.ndarray
    primary_cat: np.ndarray
    secondary_cat: np.ndarray
    first_day: np.ndarray
    last_day: np.ndarray


def build_users(rng: np.random.Generator, n_users: int, n_days: int) -> Users:
    shares = np.array([c[1] for c in CATEGORIES])
    p = shares / shares.sum()
    first_day = rng.integers(-60, n_days, n_users)
    lifetime = rng.geometric(1 / 35, n_users)  # mean ~5 weeks active
    return Users(
        user_id=512_000_000 + rng.choice(90_000_000, size=n_users, replace=False),
        base_logit=rng.normal(-4.0, 0.9, n_users),
        activity=rng.gamma(0.6, 0.9, n_users),
        price_sens=rng.beta(2, 2, n_users),
        evening_pref=rng.beta(2, 3, n_users),
        primary_cat=rng.choice(len(CATEGORIES), size=n_users, p=p),
        secondary_cat=rng.choice(len(CATEGORIES), size=n_users, p=p),
        first_day=np.maximum(first_day, 0),
        last_day=np.minimum(first_day + lifetime, n_days - 1),
    )


def _day_profile(dates: pd.DatetimeIndex, sale_days: set[pd.Timestamp]):
    dow = dates.dayofweek.to_numpy()
    sale = np.array([d in sale_days for d in dates])
    trend = np.linspace(1.0, 1.15, len(dates))  # mild November growth
    traffic = DOW_TRAFFIC[dow] * trend * np.where(sale, 2.2, 1.0)
    weekend = dow >= 5
    return traffic, weekend.astype(float), sale.astype(float)


def _uuid_strings(rng: np.random.Generator, n: int) -> np.ndarray:
    hi = rng.integers(0, 2**63, size=n, dtype=np.int64)
    lo = rng.integers(0, 2**63, size=n, dtype=np.int64)
    out = np.empty(n, dtype=object)
    for i, (a, b) in enumerate(zip(hi.tolist(), lo.tolist(), strict=True)):
        h = f"{a:016x}{b:016x}"
        out[i] = f"{h[:8]}-{h[8:12]}-4{h[13:16]}-a{h[17:20]}-{h[20:32]}"
    return out


def _simulate_chunk(
    rng: np.random.Generator,
    users: Users,
    sl: slice,
    catalog: Catalog,
    traffic: np.ndarray,
    weekend: np.ndarray,
    sale: np.ndarray,
    missing_session_rate: float,
) -> pd.DataFrame:
    n_days = len(traffic)
    n_weeks = (n_days + 6) // 7
    n = sl.stop - sl.start
    week_of_day = np.arange(n_days) // 7

    # Weekly AR(1) in-market state: persistent, so this week's behaviour predicts next week's.
    phi, sigma = 0.6, 0.9
    z = np.empty((n, n_weeks))
    z[:, 0] = rng.normal(0, sigma / np.sqrt(1 - phi**2), n)
    for w in range(1, n_weeks):
        z[:, w] = phi * z[:, w - 1] + rng.normal(0, sigma, n)

    days = np.arange(n_days)
    active = (days[None, :] >= users.first_day[sl, None]) & (
        days[None, :] <= users.last_day[sl, None]
    )
    lam = (users.activity[sl, None] / 7.0) * np.exp(0.7 * z[:, week_of_day]) * traffic[None, :]
    n_sess = rng.poisson(np.where(active, lam, 0.0))
    u_loc, d_idx = np.nonzero(n_sess)
    reps = n_sess[u_loc, d_idx]
    s_u = np.repeat(u_loc, reps)  # local user index
    s_d = np.repeat(d_idx, reps)
    s_count = len(s_u)
    if s_count == 0:
        return pd.DataFrame(columns=[*CSV_COLUMNS, "_t"])
    g_u = s_u + sl.start  # global user index
    s_z = z[s_u, week_of_day[s_d]]

    # Session start time
    evening = rng.random(s_count) < users.evening_pref[g_u]
    hour = np.where(evening, rng.normal(19, 2, s_count), rng.normal(12.5, 3.5, s_count)) % 24
    s_start = s_d * SECONDS_PER_DAY + hour * 3600 + rng.uniform(0, 3600, s_count) % 3600

    # Session category
    r = rng.random(s_count)
    shares = np.array([c[1] for c in CATEGORIES])
    random_cat = rng.choice(len(CATEGORIES), size=s_count, p=shares / shares.sum())
    s_cat = np.where(
        r < 0.65, users.primary_cat[g_u], np.where(r < 0.80, users.secondary_cat[g_u], random_cat)
    )
    cat_conv = np.array([c[4] for c in CATEGORIES])
    s_logit = (
        users.base_logit[g_u] + 0.9 * s_z + 0.12 * weekend[s_d] + 0.35 * sale[s_d] + cat_conv[s_cat]
    )

    # Views
    n_views = 1 + rng.poisson(3.5 * np.exp(0.25 * s_z))
    v_s = np.repeat(np.arange(s_count), n_views)
    v_total = len(v_s)
    v_prod = np.empty(v_total, dtype=np.int64)
    v_cat = s_cat[v_s]
    for c in range(len(CATEGORIES)):
        mask = v_cat == c
        if mask.any():
            pick = np.searchsorted(catalog.cdf_by_category[c], rng.random(mask.sum()))
            v_prod[mask] = catalog.index_by_category[c][
                np.minimum(pick, len(catalog.index_by_category[c]) - 1)
            ]
    gaps = rng.exponential(45.0, v_total)
    csum = np.cumsum(gaps)
    starts = np.concatenate([[0], np.cumsum(n_views)[:-1]])
    v_offset = csum - np.repeat(csum[starts] - gaps[starts], n_views)
    v_time = s_start[v_s] + v_offset

    day_factor = np.where(sale[s_d] > 0, 0.9, 1.0)  # sale-day discount
    v_price = np.round(
        catalog.price[v_prod] * day_factor[v_s] * (1 + rng.normal(0, 0.01, v_total)), 2
    )
    rel_logp = np.log(np.maximum(v_price, 0.5) / catalog.median_price[v_cat])
    s_rel = np.add.reduceat(rel_logp, starts) / n_views

    # Carts: price-sensitive users are put off by expensive sessions and cart the cheapest item.
    price_term = 0.8 * users.price_sens[g_u] * s_rel
    does_cart = rng.random(s_count) < _sigmoid(s_logit + 0.9 - price_term)
    order = np.lexsort((v_price, v_s))  # sorted by session, then price
    cheapest = order[starts]  # first row of each session after sorting = cheapest view
    random_view = starts + (rng.random(s_count) * n_views).astype(int)
    pick_cheapest = rng.random(s_count) < users.price_sens[g_u]
    cart_view_1 = np.where(pick_cheapest, cheapest, random_view)
    second = does_cart & (rng.random(s_count) < 0.2) & (n_views > 1)
    cart_view_2 = starts + (rng.random(s_count) * n_views).astype(int)
    cart_sessions = np.concatenate([np.nonzero(does_cart)[0], np.nonzero(second)[0]])
    cart_views = np.concatenate([cart_view_1[does_cart], cart_view_2[second]])
    c_time = v_time[cart_views] + rng.exponential(20.0, len(cart_views))

    # Purchases (always preceded by a cart, so funnel stages nest)
    buy = rng.random(len(cart_views)) < _sigmoid(
        s_logit[cart_sessions] + 1.7 + 0.3 * sale[s_d[cart_sessions]]
    )
    session_end = s_start + np.add.reduceat(gaps, starts)
    p_sessions = cart_sessions[buy]
    p_views = cart_views[buy]
    p_time = np.maximum(session_end[p_sessions], c_time[buy]) + rng.exponential(90.0, len(p_views))

    session_ids = _uuid_strings(rng, s_count)
    missing = rng.random(s_count) < missing_session_rate
    session_ids[missing] = ""

    def frame(event_type: str, views: np.ndarray, sessions: np.ndarray, t: np.ndarray):
        prod = v_prod[views]
        return pd.DataFrame(
            {
                "_t": t,
                "event_type": event_type,
                "product_id": catalog.product_id[prod],
                "category_id": catalog.category_id[prod],
                "category_code": np.array([c[0] for c in CATEGORIES], dtype=object)[
                    catalog.category[prod]
                ],
                "brand": catalog.brand[prod],
                "price": v_price[views],
                "user_id": users.user_id[g_u[sessions]],
                "user_session": session_ids[sessions],
            }
        )

    return pd.concat(
        [
            frame("view", np.arange(v_total), v_s, v_time),
            frame("cart", cart_views, cart_sessions, c_time),
            frame("purchase", p_views, p_sessions, p_time),
        ],
        ignore_index=True,
    )


def _format_time(start: np.datetime64, seconds: np.ndarray) -> np.ndarray:
    ts = start + seconds.astype("timedelta64[s]")
    s = np.datetime_as_string(ts, unit="s")
    return np.char.add(np.char.replace(s, "T", " "), " UTC")


def generate(
    cfg: Config, out_dir: Path | None = None, n_users: int | None = None, chunk_users: int = 50_000
) -> GenerationSummary:
    syn = cfg["synthetic"]
    n_users = int(n_users or syn["n_users"])
    out_dir = out_dir or cfg.paths.raw_source("synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    dates = pd.date_range(syn["start_date"], syn["end_date"], freq="D")
    sale_days = {pd.Timestamp(d) for d in syn["sale_days"]}
    traffic, weekend, sale = _day_profile(dates, sale_days)
    catalog = build_catalog(rng, int(syn["n_products"]))
    users = build_users(rng, n_users, len(dates))
    start = np.datetime64(f"{syn['start_date']}T00:00:00")
    horizon = len(dates) * SECONDS_PER_DAY

    months = sorted({(d.year, d.month) for d in dates})
    prefix = {m: f"{pd.Timestamp(year=m[0], month=m[1], day=1):%Y-%b}" for m in months}
    for old in [*out_dir.glob("*.csv"), *out_dir.glob("*.csv.gz")]:
        old.unlink()
    files: list[Path] = []
    month_bounds = {
        m: (
            (pd.Timestamp(year=m[0], month=m[1], day=1) - dates[0]).total_seconds(),
            (
                (pd.Timestamp(year=m[0], month=m[1], day=1) + pd.offsets.MonthBegin(1)) - dates[0]
            ).total_seconds(),
        )
        for m in months
    }
    totals: dict[str, int] = {"view": 0, "cart": 0, "purchase": 0}
    n_events = n_sessions = 0

    for chunk, lo in enumerate(range(0, n_users, chunk_users)):
        sl = slice(lo, min(lo + chunk_users, n_users))
        df = _simulate_chunk(
            rng, users, sl, catalog, traffic, weekend, sale, float(syn["missing_session_rate"])
        )
        df = df[df["_t"] < horizon]
        n_sessions += df["user_session"].nunique()
        # Defects the data-quality step must catch
        dup = df.sample(frac=float(syn["duplicate_rate"]), random_state=int(rng.integers(1 << 31)))
        df = pd.concat([df, dup], ignore_index=True)
        zero = rng.random(len(df)) < float(syn["zero_price_rate"])
        df.loc[zero, "price"] = 0.0
        df = df.sort_values("_t", kind="stable")
        secs = np.floor(df["_t"].to_numpy())
        df["event_time"] = _format_time(start, secs)
        for et, cnt in df["event_type"].value_counts().items():
            totals[str(et)] += int(cnt)
        n_events += len(df)
        # One gzip file per month and user chunk: ~5x smaller than plain CSV, and several files
        # per month keep Spark's read parallel (a single gzip file is not splittable).
        for m, (a, b) in month_bounds.items():
            part = df[(secs >= a) & (secs < b)]
            if part.empty:
                continue
            path = out_dir / f"{prefix[m]}-{chunk:03d}.csv.gz"
            part[CSV_COLUMNS].to_csv(
                path, index=False, float_format="%.2f", compression={"method": "gzip", "mtime": 0}
            )
            files.append(path)
        log.info("Generated users %d-%d (%d events so far)", sl.start, sl.stop, n_events)

    summary = GenerationSummary(
        n_users=n_users,
        n_products=len(catalog.product_id),
        n_events=n_events,
        n_sessions=n_sessions,
        events_by_type=totals,
        files=sorted(p.name for p in files),
        seed=cfg.seed,
    )
    (out_dir / "_generation.json").write_text(json.dumps(asdict(summary), indent=2))
    return summary
