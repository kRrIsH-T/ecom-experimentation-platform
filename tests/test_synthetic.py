import pandas as pd

from ecom.ingest.load_raw import _csv_files
from ecom.ingest.synthetic import CSV_COLUMNS, generate


def _read(out) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(f, dtype={"category_code": str, "brand": str}) for f in _csv_files(out)]
    )


def test_generator_matches_kaggle_format(cfg, tmp_path):
    out = tmp_path / "raw"
    summary = generate(cfg, out, n_users=3000, chunk_users=1000)
    names = [f.name for f in _csv_files(out)]
    assert names == summary.files
    assert {n[:8] for n in names} == {"2019-Oct", "2019-Nov"}
    assert all(n.endswith(".csv.gz") for n in names)
    df = _read(out)
    assert list(df.columns) == CSV_COLUMNS
    assert len(df) == summary.n_events
    assert df["event_time"].str.match(r"^2019-1[01]-\d\d \d\d:\d\d:\d\d UTC$").all()
    assert set(df["event_type"]) == {"view", "cart", "purchase"}
    assert df["event_type"].value_counts(normalize=True)["view"] > 0.85
    assert df["category_code"].isna().mean() > 0.1  # uncategorised products like the real data
    assert df.duplicated().sum() > 0  # injected duplicates


def test_purchases_follow_carts_within_session(cfg, tmp_path):
    out = tmp_path / "raw"
    generate(cfg, out, n_users=2000)
    df = _read(out).dropna(subset=["user_session"])
    by_session = df.groupby("user_session")["event_type"].agg(set)
    with_purchase = by_session[by_session.apply(lambda s: "purchase" in s)]
    assert len(with_purchase) > 0
    assert with_purchase.apply(lambda s: "cart" in s).all()


def test_generation_is_reproducible(cfg, tmp_path):
    a = generate(cfg, tmp_path / "a", n_users=1000)
    b = generate(cfg, tmp_path / "b", n_users=1000)
    assert a.n_events == b.n_events
    for name in a.files:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
