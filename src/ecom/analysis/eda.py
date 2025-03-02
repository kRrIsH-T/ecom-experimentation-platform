"""Exploratory analysis: distribution, time-pattern, funnel, cohort and Pareto figures plus
a list of quantified insights for the report and dashboard."""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from ecom.config import Config
from ecom.layout import figures_dir, outputs_dir
from ecom.stats.hypothesis_tests import connect
from ecom.viz import BLUE, INK_2, MUTED, NEUTRAL_MID, ORANGE, SEQ_BLUE, SEQ_CMAP

log = logging.getLogger(__name__)

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MIN_CELL_SESSIONS = 200
THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}")


def _read(out, name: str) -> pd.DataFrame:
    return pd.read_csv(out / f"{name}.csv")


def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_distributions(con, figs) -> dict:
    prices = con.execute("SELECT event_type, price FROM events_clean USING SAMPLE 300000 ROWS").df()
    sessions = con.execute("SELECT duration_s, n_events FROM sessions").df()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    bins = np.logspace(0, np.log10(max(prices.price.max(), 10)), 60)
    for et, color in (("view", BLUE), ("purchase", ORANGE)):
        axes[0].hist(
            prices.loc[prices.event_type == et, "price"],
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            color=color,
            label=et,
        )
    axes[0].set(
        xscale="log", title="Price distribution (log scale)", xlabel="price", ylabel="density"
    )
    axes[0].legend()
    dur = sessions.loc[sessions.duration_s > 0, "duration_s"]
    axes[1].hist(dur, bins=np.logspace(0, np.log10(dur.max()), 60), color=BLUE)
    axes[1].set(
        xscale="log",
        title="Session duration (multi-event sessions, log)",
        xlabel="seconds",
        ylabel="sessions",
    )
    ev = sessions["n_events"].clip(upper=40)
    axes[2].hist(ev, bins=np.arange(0.5, 41.5), color=BLUE)
    axes[2].set(
        yscale="log",
        title="Events per session (capped at 40)",
        xlabel="events",
        ylabel="sessions (log)",
    )
    _save(fig, figs / "eda_distributions.png")
    return {
        "median_price_viewed": float(prices.loc[prices.event_type == "view", "price"].median()),
        "median_price_purchased": float(
            prices.loc[prices.event_type == "purchase", "price"].median()
        ),
        "median_session_s": float(dur.median()),
        "single_event_session_share": float((sessions.n_events == 1).mean()),
    }


def plot_time_patterns(out, figs) -> dict:
    hd = _read(out, "hour_dow")
    sess = hd.pivot(index="dow", columns="start_hour", values="sessions").reindex(range(1, 8))
    conv = hd.pivot(index="dow", columns="start_hour", values="conversion").reindex(range(1, 8))
    # Cells resting on few sessions are noise, not signal: grey them out.
    conv = conv.where(sess >= MIN_CELL_SESSIONS)
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.2))
    for ax, data, title, fmt in (
        (axes[0], sess, "Sessions by day x hour (UTC)", "{:,.0f}"),
        (
            axes[1],
            conv,
            f"Session conversion by day x hour (grey: < {MIN_CELL_SESSIONS} sessions)",
            "{:.1%}",
        ),
    ):
        cmap = SEQ_CMAP.with_extremes(bad=NEUTRAL_MID)
        im = ax.imshow(np.ma.masked_invalid(data.to_numpy(dtype=float)), aspect="auto", cmap=cmap)
        ax.set(
            yticks=range(7),
            yticklabels=DOW,
            xticks=range(0, 24, 2),
            title=title,
            xlabel="hour of day",
        )
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.03)
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, f=fmt: f.format(v)))
        cb.outline.set_visible(False)
    _save(fig, figs / "eda_hour_dow.png")

    daily = _read(out, "daily_kpis")
    daily["session_date"] = pd.to_datetime(daily["session_date"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
    axes[0].plot(daily.session_date, daily.sessions, color=BLUE)
    axes[0].set(title="Daily sessions", ylabel="sessions")
    axes[0].yaxis.set_major_formatter(THOUSANDS)
    axes[1].plot(daily.session_date, daily.conversion, color=BLUE)
    axes[1].set(title="Daily session conversion", ylabel="conversion")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    for ax in axes:
        ax.axvspan(
            pd.Timestamp("2019-11-15"),
            pd.Timestamp("2019-11-18"),
            color=SEQ_BLUE[0],
            lw=0,
            zorder=0,
        )
    axes[0].annotate(
        "Nov 15-17 sale",
        (pd.Timestamp("2019-11-15"), daily.sessions.max()),
        xytext=(-90, -5),
        textcoords="offset points",
        color=INK_2,
    )
    _save(fig, figs / "eda_daily.png")

    sale = daily.session_date.between("2019-11-15", "2019-11-17")
    peak = hd.groupby("start_hour")["sessions"].sum().idxmax()
    return {
        "sale_traffic_multiplier": float(
            daily.loc[sale, "sessions"].mean() / daily.loc[~sale, "sessions"].mean()
        ),
        "sale_conversion": float(
            daily.loc[sale, "purchasing_sessions"].sum() / daily.loc[sale, "sessions"].sum()
        ),
        "non_sale_conversion": float(
            daily.loc[~sale, "purchasing_sessions"].sum() / daily.loc[~sale, "sessions"].sum()
        ),
        "peak_hour_utc": int(peak),
    }


def plot_funnel(out, figs) -> dict:
    users = _read(out, "funnel_overall")
    sess = _read(out, "funnel_sessions")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, df, col, title in (
        (axes[0], users, "users", "User funnel"),
        (axes[1], sess, "sessions", "Session funnel"),
    ):
        ax.barh(df["stage"][::-1], df[col][::-1], color=SEQ_BLUE[3:10:3][::-1], height=0.6)
        for i, (v, pct) in enumerate(zip(df[col][::-1], df.iloc[::-1, 3], strict=True)):
            ax.text(v, i, f"  {v:,.0f} ({pct:.1%})", va="center", color=INK_2)
        ax.set(title=title, xlim=(0, df[col].max() * 1.35))
        ax.xaxis.set_major_formatter(THOUSANDS)
        ax.grid(axis="y", visible=False)
    _save(fig, figs / "eda_funnel.png")
    return {
        "view_to_cart_users": float(users.loc[1, "step_conversion"]),
        "cart_to_purchase_users": float(users.loc[2, "step_conversion"]),
        "view_to_purchase_users": float(users.loc[2, "pct_of_viewers"]),
        "session_conversion": float(sess.loc[2, "pct_of_sessions"]),
    }


def plot_cohorts(out, figs) -> dict:
    co = _read(out, "cohort_retention")
    mat = co.pivot(index="cohort_week", columns="week_number", values="retention")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mat.to_numpy(), cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(mat.to_numpy()):
        if np.isfinite(v):
            ax.text(
                j,
                i,
                f"{v:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if v > 0.55 else INK_2,
            )
    ax.set(
        xticks=range(mat.shape[1]),
        yticks=range(mat.shape[0]),
        yticklabels=mat.index,
        xlabel="weeks since first seen",
        title="Weekly cohort retention (share of cohort active)",
    )
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03).outline.set_visible(False)
    _save(fig, figs / "eda_cohorts.png")
    week1 = co[co.week_number == 1]
    clean = week1[week1.cohort_week != week1.cohort_week.min()]  # first cohort is left-censored
    return {"week1_retention_mean": float(clean["retention"].mean())}


def plot_pareto(out, figs) -> dict:
    brands = _read(out, "brand_pareto")
    cats = _read(out, "category_stats")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    top = brands.head(15)
    axes[0].barh(top["brand"][::-1], top["revenue"][::-1], color=BLUE, height=0.65)
    axes[0].set(title="Top 15 brands by revenue", xlabel="revenue")
    axes[0].xaxis.set_major_formatter(THOUSANDS)
    axes[0].grid(axis="y", visible=False)
    axes[1].plot(
        np.arange(1, len(brands) + 1) / len(brands), brands["cumulative_share"], color=BLUE
    )
    axes[1].axhline(0.8, color=MUTED, ls="--", lw=1)
    n80 = int((brands["cumulative_share"] < 0.8).sum()) + 1
    axes[1].annotate(
        f"{n80} of {len(brands)} brands = 80% of revenue",
        (n80 / len(brands), 0.8),
        xytext=(20, -30),
        textcoords="offset points",
        color=INK_2,
        arrowprops={"arrowstyle": "-", "color": MUTED},
    )
    axes[1].set(
        title="Revenue Pareto curve (brands)",
        xlabel="share of brands (ranked)",
        ylabel="cumulative share of revenue",
    )
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    _save(fig, figs / "eda_pareto.png")

    by_top = cats.groupby("category_top").agg(revenue=("revenue", "sum"), orders=("orders", "sum"))
    by_top = by_top.sort_values("revenue", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(by_top.index[::-1], by_top["revenue"][::-1], color=BLUE, height=0.65)
    ax.set(title="Revenue by top-level category", xlabel="revenue")
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.grid(axis="y", visible=False)
    _save(fig, figs / "eda_categories.png")
    leader = cats.sort_values("aov", ascending=False).iloc[0]
    return {
        "brands_for_80pct": n80,
        "n_brands": len(brands),
        "brand_share_for_80pct": n80 / len(brands),
        "top_category": str(by_top.index[0]),
        "top_category_revenue_share": float(by_top["revenue"].iloc[0] / by_top["revenue"].sum()),
        "highest_aov_category": str(leader["category_code"]),
        "highest_aov": float(leader["aov"]),
    }


def run_eda(cfg: Config) -> list[str]:
    out, figs = outputs_dir(cfg), figures_dir(cfg)
    con = connect(cfg)
    facts: dict = {}
    facts |= plot_distributions(con, figs)
    facts |= plot_time_patterns(out, figs)
    facts |= plot_funnel(out, figs)
    facts |= plot_cohorts(out, figs)
    facts |= plot_pareto(out, figs)
    conc = _read(out, "revenue_concentration").iloc[0]
    overview = _read(out, "overview").iloc[0]
    facts |= {
        "top_1pct_revenue_share": float(conc.top_1pct_share),
        "top_10pct_revenue_share": float(conc.top_10pct_share),
        "buyer_share": float(conc.buyer_share),
        "users": int(overview.users),
        "events": int(overview.events),
        "revenue": float(overview.revenue),
    }
    insights = [
        f"The top 1% of users generate {facts['top_1pct_revenue_share']:.0%} of revenue and the "
        f"top 10% generate {facts['top_10pct_revenue_share']:.0%}; only "
        f"{facts['buyer_share']:.1%} of visitors buy anything in two months.",
        f"{facts['view_to_cart_users']:.0%} of viewers ever add to cart, and "
        f"{facts['cart_to_purchase_users']:.0%} of those go on to buy - the view-to-cart step "
        f"is where most users are lost.",
        f"Sessions convert at {facts['session_conversion']:.1%}; half of all multi-event "
        f"sessions last under {facts['median_session_s'] / 60:.1f} minutes and "
        f"{facts['single_event_session_share']:.0%} of sessions are a single page view.",
        f"The Nov 15-17 sale lifted daily traffic {facts['sale_traffic_multiplier']:.1f}x and "
        f"conversion from {facts['non_sale_conversion']:.1%} to {facts['sale_conversion']:.1%}.",
        f"{facts['top_category'].title()} drives {facts['top_category_revenue_share']:.0%} of "
        f"revenue; {facts['highest_aov_category']} has the highest average order value "
        f"({facts['highest_aov']:,.0f}).",
        f"Revenue is concentrated in brands too: {facts['brands_for_80pct']} of "
        f"{facts['n_brands']} brands ({facts['brand_share_for_80pct']:.0%}) make up 80% of it.",
        f"Buyers pay a median {facts['median_price_purchased']:,.0f} per item vs a median "
        f"{facts['median_price_viewed']:,.0f} for viewed items, so cheaper items convert better.",
        f"Only {facts['week1_retention_mean']:.0%} of a weekly cohort comes back the next week; "
        f"traffic peaks at {facts['peak_hour_utc']:02d}:00 UTC.",
    ]
    (out / "eda_insights.json").write_text(
        json.dumps({"facts": facts, "insights": insights}, indent=2)
    )
    for line in insights:
        log.info("insight: %s", line)
    return insights
