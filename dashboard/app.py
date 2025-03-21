"""Streamlit dashboard: KPIs, funnel & cohorts, statistics, model, experiment, user lookup.

Run with ``ecom dashboard`` (or ``streamlit run dashboard/app.py``). Reads the small result
files the pipeline writes to ``data/outputs/<source>/<sample>/``; the user-lookup tab calls the
FastAPI service (``ecom serve``).
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from ecom.config import get_config
from ecom.viz import BLUE, INK_2, MUTED, NEUTRAL_MID, SEQ_BLUE, SERIES, plotly_template

st.set_page_config(page_title="E-commerce Experimentation", page_icon="🛒", layout="wide")

CFG = get_config()
TEMPLATE = plotly_template()
MODEL_ORDER = ["logreg", "xgboost", "spark_gbt", "h2o_automl"]  # colour follows the model
MODEL_NAMES = {
    "logreg": "Logistic regression",
    "xgboost": "XGBoost + Optuna",
    "spark_gbt": "Spark MLlib GBT",
    "h2o_automl": "H2O AutoML",
}
MODEL_COLORS = dict(zip(MODEL_ORDER, SERIES, strict=False))


# --- data access --------------------------------------------------------------------------


def available_runs() -> list[Path]:
    base = CFG.paths.outputs
    return sorted(p for p in base.glob("*/*") if p.is_dir() and any(p.glob("*.csv")))


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def read_json(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def fig_layout(fig: go.Figure, title: str, height: int = 360, **kw) -> go.Figure:
    fig.update_layout(template=TEMPLATE, title=title, height=height, **kw)
    return fig


def missing(what: str, step: str) -> None:
    st.info(f"No {what} yet - run `ecom {step}`.")


# --- sidebar ------------------------------------------------------------------------------

runs = available_runs()
st.sidebar.title("🛒 Experimentation platform")
if not runs:
    st.warning("No pipeline outputs found. Run `ecom all` first.")
    st.stop()
labels = [f"{r.parent.name} / {r.name}" for r in runs]
default = (
    labels.index(f"{CFG.source} / s{round(CFG.sample_rate * 10_000):04d}")
    if f"{CFG.source} / s{round(CFG.sample_rate * 10_000):04d}" in labels
    else 0
)
choice = st.sidebar.selectbox("Run (source / user sample)", labels, index=default)
OUT = runs[labels.index(choice)]
SRC = OUT.parent
api_url = st.sidebar.text_input("Scoring API", CFG.get("api.url", "http://localhost:8000"))
if OUT.parent.name == "synthetic":
    st.sidebar.caption("Synthetic Kaggle-format data: methods are real, numbers are simulated.")


def o(name: str) -> str:
    return str(OUT / name)


tabs = st.tabs(["Overview", "Funnel & cohorts", "Statistics", "Model", "Experiment", "User lookup"])

# --- overview -----------------------------------------------------------------------------

with tabs[0]:
    ov = read_csv(o("overview.csv"))
    if ov is None:
        missing("KPIs", "sql")
    else:
        r = ov.iloc[0]
        cols = st.columns(6)
        for col, (label, value, note) in zip(
            cols,
            [
                ("Events", f"{r.events:,.0f}", None),
                ("Users", f"{r.users:,.0f}", None),
                ("Sessions", f"{r.sessions:,.0f}", None),
                ("Revenue", f"{r.revenue:,.0f}", None),
                ("Session conversion", f"{r.session_conversion:.2%}", None),
                ("Buyers", f"{r.buyers:,.0f}", f"{r.user_conversion:.1%} of users"),
            ],
            strict=True,
        ):
            col.metric(label, value, delta=note, delta_color="off")
        daily = read_csv(o("daily_kpis.csv"))
        if daily is not None:
            c1, c2 = st.columns(2)
            f1 = go.Figure(
                go.Scatter(
                    x=daily.session_date,
                    y=daily.sessions,
                    mode="lines",
                    line={"color": BLUE, "width": 2},
                    name="sessions",
                    hovertemplate="%{x}<br>%{y:,.0f} sessions<extra></extra>",
                )
            )
            c1.plotly_chart(
                fig_layout(f1, "Daily sessions", showlegend=False), use_container_width=True
            )
            f2 = go.Figure(
                go.Scatter(
                    x=daily.session_date,
                    y=daily.conversion,
                    mode="lines",
                    line={"color": BLUE, "width": 2},
                    name="conversion",
                    hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
                )
            )
            f2.update_yaxes(tickformat=".1%")
            c2.plotly_chart(
                fig_layout(f2, "Daily session conversion", showlegend=False),
                use_container_width=True,
            )
        eda = read_json(o("eda_insights.json"))
        if eda:
            st.subheader("Insights")
            for line in eda["insights"]:
                st.markdown(f"- {line}")
        dq = read_json(str(SRC / "data_quality.json"))
        if dq:
            with st.expander("Data quality (full source)"):
                st.write(
                    f"{dq['rows']:,} rows · {dq['duplicate_rows']:,} duplicates "
                    f"({dq['duplicate_rate']:.2%}) · {dq['zero_prices']:,} zero prices · "
                    f"{dq['min_event_time']} → {dq['max_event_time']}"
                )
                st.dataframe(
                    pd.Series(dq["null_rate"], name="null rate").map("{:.3%}".format),
                    use_container_width=True,
                )

# --- funnel & cohorts ---------------------------------------------------------------------

with tabs[1]:
    funnel = read_csv(o("funnel_overall.csv"))
    if funnel is None:
        missing("funnel", "sql")
    else:
        c1, c2 = st.columns(2)
        f = go.Figure(
            go.Funnel(
                y=funnel.stage,
                x=funnel.users,
                texttemplate="%{value:,} (%{percentInitial:.1%})",
                marker={"color": [SEQ_BLUE[3], SEQ_BLUE[6], SEQ_BLUE[9]]},
                connector={"fillcolor": NEUTRAL_MID, "line": {"width": 0}},
                hovertemplate="%{y}: %{value:,} users<extra></extra>",
            )
        )
        c1.plotly_chart(
            fig_layout(f, "User funnel (each stage nested in the previous)"),
            use_container_width=True,
        )
        cats = read_csv(o("funnel_by_category.csv"))
        if cats is not None:
            cats = cats.sort_values("conversion")
            f = go.Figure(
                go.Bar(
                    x=cats.conversion,
                    y=cats.category_top,
                    orientation="h",
                    marker_color=BLUE,
                    customdata=cats[["view_sessions", "purchase_sessions"]],
                    hovertemplate="%{y}: %{x:.2%}<br>%{customdata[1]:,} of "
                    "%{customdata[0]:,} sessions<extra></extra>",
                )
            )
            f.update_xaxes(tickformat=".0%")
            c2.plotly_chart(
                fig_layout(f, "Session conversion by category", showlegend=False),
                use_container_width=True,
            )
        co = read_csv(o("cohort_retention.csv"))
        if co is not None:
            mat = co.pivot(index="cohort_week", columns="week_number", values="retention")
            f = go.Figure(
                go.Heatmap(
                    z=mat.values,
                    x=[f"week {c}" for c in mat.columns],
                    y=mat.index.astype(str),
                    zmin=0,
                    zmax=1,
                    colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
                    text=[[f"{v:.0%}" if pd.notna(v) else "" for v in row] for row in mat.values],
                    texttemplate="%{text}",
                    hovertemplate="cohort %{y}, %{x}: %{z:.1%}<extra></extra>",
                    colorbar={"tickformat": ".0%"},
                )
            )
            f.update_yaxes(autorange="reversed", title="first-seen week")  # oldest cohort on top
            st.plotly_chart(
                fig_layout(f, "Weekly cohort retention", height=460), use_container_width=True
            )
        rfm = read_csv(o("rfm_summary.csv"))
        if rfm is not None:
            c1, c2 = st.columns(2)
            f = go.Figure(
                go.Bar(
                    x=rfm.segment,
                    y=rfm.user_share,
                    marker_color=BLUE,
                    hovertemplate="%{x}: %{y:.1%} of buyers<extra></extra>",
                )
            )
            f.update_yaxes(tickformat=".0%")
            c1.plotly_chart(
                fig_layout(f, "RFM segments: share of buyers"), use_container_width=True
            )
            f = go.Figure(
                go.Bar(
                    x=rfm.segment,
                    y=rfm.revenue_share,
                    marker_color=BLUE,
                    hovertemplate="%{x}: %{y:.1%} of revenue<extra></extra>",
                )
            )
            f.update_yaxes(tickformat=".0%")
            c2.plotly_chart(
                fig_layout(f, "RFM segments: share of revenue"), use_container_width=True
            )

# --- statistics ---------------------------------------------------------------------------

with tabs[2]:
    stats = read_json(o("stats_tests.json"))
    if not stats:
        missing("statistical tests", "stats")
    else:
        rows = [
            {
                "Question": t["name"],
                "Test": t["test"],
                "Effect": f"{t['effect_size_name']} = {t['effect_size']:.3f}",
                "95% CI": (f"{t['ci_low']:.3f} to {t['ci_high']:.3f}" if t["ci_of"] else "-"),
                "p": f"{t['p_value']:.2g}",
                "p (BH)": f"{t['p_adjusted']:.2g}",
                "Significant": "yes" if t["significant"] else "no",
            }
            for t in stats["tests"]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        for t in stats["tests"]:
            with st.expander(t["name"]):
                st.markdown(f"**H0:** {t['h0']}  \n**H1:** {t['h1']}  \n**Method:** {t['test']}")
                st.json({"assumptions": t["assumptions"], "details": t["details"]}, expanded=False)
        boot = pd.DataFrame(stats["bootstrap_median_aov"])
        f = go.Figure(
            go.Scatter(
                x=boot["median"],
                y=boot.segment,
                mode="markers",
                marker={"color": BLUE, "size": 10},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": boot.ci_high - boot["median"],
                    "arrayminus": boot["median"] - boot.ci_low,
                    "color": INK_2,
                },
                hovertemplate="%{y}: median %{x:,.1f}<extra></extra>",
            )
        )
        st.plotly_chart(
            fig_layout(f, "Median order value with bootstrap (BCa) 95% CI", showlegend=False),
            use_container_width=True,
        )

# --- model --------------------------------------------------------------------------------

with tabs[3]:
    comp = read_csv(o("model_comparison.csv"))
    curves = read_json(o("model_curves.json"))
    if comp is None or not curves:
        missing("model results", "train")
    else:
        champ = read_json(o("champion.json")) or {}
        st.markdown(
            f"**Champion:** {MODEL_NAMES.get(champ.get('model_family'), '?')} - "
            f"registered as `{champ.get('name')}` v{champ.get('version')} "
            f"(selected on validation PR-AUC)"
        )
        headers = {
            "model": "Model",
            "valid_pr_auc": "Valid PR-AUC",
            "test_pr_auc": "Test PR-AUC",
            "test_roc_auc": "Test ROC-AUC",
            "test_brier": "Brier",
            "test_top_decile_precision": "Top-decile precision",
            "test_top_decile_lift": "Top-decile lift",
            "train_seconds": "Train time (s)",
        }
        show = comp[list(headers)].rename(columns=headers)
        st.dataframe(
            show.style.format(
                {c: "{:.3f}" for c in show.columns if c != "Model"}
                | {
                    "Top-decile precision": "{:.1%}",
                    "Top-decile lift": "{:.2f}x",
                    "Train time (s)": "{:.1f}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )
        keys = [k for k in MODEL_ORDER if k in curves]
        c1, c2, c3 = st.columns(3)
        f_pr, f_roc, f_cal = go.Figure(), go.Figure(), go.Figure()
        for k in keys:
            c = curves[k]
            color = MODEL_COLORS[k]
            f_pr.add_scatter(
                x=c["pr"]["recall"],
                y=c["pr"]["precision"],
                mode="lines",
                name=MODEL_NAMES[k],
                line={"color": color, "width": 2},
            )
            f_roc.add_scatter(
                x=c["roc"]["fpr"],
                y=c["roc"]["tpr"],
                mode="lines",
                name=MODEL_NAMES[k],
                line={"color": color, "width": 2},
            )
            f_cal.add_scatter(
                x=c["calibration"]["mean_predicted"],
                y=c["calibration"]["fraction_positive"],
                mode="lines+markers",
                name=MODEL_NAMES[k],
                line={"color": color, "width": 2},
                marker={"size": 8},
            )
        base = float(comp["test_base_rate"].iloc[0])
        f_pr.add_hline(
            y=base,
            line_dash="dash",
            line_color=MUTED,
            annotation_text=f"base rate {base:.3f}",
            annotation_position="top left",
        )
        f_roc.add_scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="random",
            line={"color": MUTED, "dash": "dash", "width": 1},
        )
        lim = max(max(curves[k]["calibration"]["mean_predicted"]) for k in keys)
        f_cal.add_scatter(
            x=[0, lim],
            y=[0, lim],
            mode="lines",
            name="perfect",
            line={"color": MUTED, "dash": "dash", "width": 1},
        )
        c1.plotly_chart(
            fig_layout(
                f_pr,
                "Precision-recall (test)",
                height=460,
                xaxis_title="recall",
                yaxis_title="precision",
            ),
            use_container_width=True,
        )
        c2.plotly_chart(
            fig_layout(
                f_roc,
                "ROC (test)",
                height=460,
                xaxis_title="false positive rate",
                yaxis_title="true positive rate",
            ),
            use_container_width=True,
        )
        c3.plotly_chart(
            fig_layout(
                f_cal,
                "Calibration (test)",
                height=460,
                xaxis_title="predicted",
                yaxis_title="observed",
            ),
            use_container_width=True,
        )
        champ_key = champ.get("model_family", keys[0])
        gains = pd.DataFrame(curves[champ_key]["gains"])
        c1, c2 = st.columns(2)
        f = go.Figure(
            go.Bar(
                x=gains.decile,
                y=gains.lift,
                marker_color=BLUE,
                hovertemplate="decile %{x}: %{y:.2f}x lift<extra></extra>",
            )
        )
        f.add_hline(y=1, line_dash="dash", line_color=MUTED)
        f.update_xaxes(dtick=1, title="score decile (1 = highest)")
        c1.plotly_chart(
            fig_layout(f, f"Lift by decile - {MODEL_NAMES[champ_key]}", showlegend=False),
            use_container_width=True,
        )
        shap_imp = read_csv(o("shap_importance.csv"))
        if shap_imp is not None:
            top = shap_imp.head(12).iloc[::-1]
            f = go.Figure(
                go.Bar(
                    x=top.mean_abs_shap,
                    y=top.feature,
                    orientation="h",
                    marker_color=BLUE,
                    hovertemplate="%{y}: %{x:.3f}<extra></extra>",
                )
            )
            c2.plotly_chart(
                fig_layout(f, "Mean |SHAP| (log-odds), XGBoost", height=400, showlegend=False),
                use_container_width=True,
            )
        drivers = read_json(o("model_drivers.json")) or []
        if drivers:
            st.subheader("Top drivers in plain English")
            for i, d in enumerate(drivers, 1):
                st.markdown(f"{i}. {d['text']}")
        if Path(o("shap_summary.png")).exists():  # per run (reports/figures is shared)
            with st.expander("SHAP summary plot"):
                st.image(o("shap_summary.png"))
        lb = read_csv(o("h2o_leaderboard.csv"))
        if lb is not None:
            with st.expander("H2O AutoML leaderboard (validation)"):
                st.dataframe(lb.head(15), use_container_width=True, hide_index=True)

# --- experiment ---------------------------------------------------------------------------

with tabs[4]:
    ab = read_json(o("ab_results.json"))
    if not ab:
        missing("experiment results", "abtest")
    else:
        dec = ab["decision"]
        icon = "✅" if dec["verdict"] == "SHIP" else "⛔"
        st.markdown(f"### {icon} Decision: {dec['verdict']}")
        for r in dec["reasons"]:
            st.markdown(f"- {r}")
        d = ab["design"]
        cols = st.columns(4)
        cols[0].metric("Baseline conversion", f"{d['baseline_rate']:.2%}")
        cols[1].metric(f"Users/arm for {d['mde_relative']:.0%} MDE", f"{d['n_per_arm']:,}")
        cols[2].metric("Users/arm available", f"{d['n_per_arm_available']:,}")
        cols[3].metric("Power at available n", f"{d['power_at_available_n']:.0%}")
        conv = ab["conversion"]
        segs = pd.DataFrame(ab["segments"])
        forest = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "segment": "overall",
                            "diff": conv["details"]["diff"],
                            "ci_low": conv["ci_low"],
                            "ci_high": conv["ci_high"],
                        }
                    ]
                ),
                segs[["segment", "diff", "ci_low", "ci_high"]] if len(segs) else None,
            ]
        ).iloc[::-1]
        c1, c2 = st.columns(2)
        f = go.Figure(
            go.Scatter(
                x=forest["diff"],
                y=forest.segment,
                mode="markers",
                marker={"color": BLUE, "size": 10},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": forest.ci_high - forest["diff"],
                    "arrayminus": forest["diff"] - forest.ci_low,
                    "color": INK_2,
                },
                hovertemplate="%{y}: %{x:+.2%}<extra></extra>",
            )
        )
        f.add_vline(x=0, line_color=MUTED)
        f.update_xaxes(tickformat="+.1%", title="treatment - control (conversion)")
        c1.plotly_chart(
            fig_layout(f, "Conversion difference with 95% CI", height=420, showlegend=False),
            use_container_width=True,
        )
        rev = ab["revenue"]
        f = go.Figure(
            go.Bar(
                x=["raw", "CUPED"],
                y=[rev["ci_width_raw"], rev["ci_width_cuped"]],
                marker_color=[MUTED, BLUE],
                hovertemplate="%{x}: CI width %{y:.2f}<extra></extra>",
            )
        )
        c2.plotly_chart(
            fig_layout(
                f,
                f"Revenue/user CI width: CUPED cuts variance "
                f"{rev['cuped_stats']['variance_reduction']:.0%}",
                height=420,
                showlegend=False,
            ),
            use_container_width=True,
        )
        seq = read_csv(o("ab_sequential.csv"))
        c1, c2 = st.columns(2)
        if seq is not None:
            f = go.Figure(
                go.Scatter(
                    x=seq.day,
                    y=seq.p_value,
                    mode="lines+markers",
                    line={"color": BLUE, "width": 2},
                    marker={"size": 8},
                    name="p-value",
                )
            )
            f.add_hline(y=0.05, line_dash="dash", line_color=INK_2, annotation_text="alpha 0.05")
            f.update_yaxes(type="log", title="p-value")
            f.update_xaxes(title="day of test", dtick=1)
            c1.plotly_chart(
                fig_layout(f, "Sequential p-value (why we don't peek)", showlegend=False),
                use_container_width=True,
            )
        bayes = ab["bayesian"]
        edges = bayes["lift_bin_edges"]
        mids = [(a + b) / 2 for a, b in pairwise(edges)]
        f = go.Figure(
            go.Bar(
                x=mids,
                y=bayes["lift_histogram"],
                marker_color=BLUE,
                hovertemplate="lift %{x:+.1%}<extra></extra>",
            )
        )
        f.add_vline(x=0, line_color=MUTED)
        f.update_xaxes(tickformat="+.0%", title="relative lift in conversion")
        c2.plotly_chart(
            fig_layout(
                f,
                f"Bayesian posterior: P(treatment better) = {bayes['prob_treatment_better']:.1%}",
                showlegend=False,
            ),
            use_container_width=True,
        )
        pk = ab["peeking"]
        st.caption(
            f"A/A peeking simulation ({pk['n_simulations']} runs, {pk['looks']} daily "
            f"looks): false-positive rate {pk['fpr_any_look']:.1%} when stopping at the "
            f"first p < 0.05 vs {pk['fpr_final_look_only']:.1%} with one look at the end."
        )
        arms = read_csv(o("ab_arms.csv"))
        if arms is not None:
            st.dataframe(arms, use_container_width=True, hide_index=True)

# --- user lookup --------------------------------------------------------------------------

with tabs[5]:
    st.markdown(
        "Scores come from MongoDB through the FastAPI service; **live** rescoring "
        "uses the MLflow champion model on the stored features."
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        with st.form("lookup"):
            uid = st.text_input("User id", placeholder="e.g. 512345678")
            if st.form_submit_button("Look up", type="primary") and uid.strip():
                # Keep the answer in session state so other widgets' reruns don't wipe it.
                try:
                    resp = requests.get(
                        f"{api_url}/users/{int(uid)}/score", params={"live": True}, timeout=10
                    )
                    st.session_state["lookup_result"] = (
                        ("ok", resp.json())
                        if resp.status_code == 200
                        else ("error", f"{resp.status_code}: {resp.json().get('detail')}")
                    )
                except ValueError:
                    st.session_state["lookup_result"] = ("error", "User id must be a number.")
                except requests.RequestException as exc:
                    st.session_state["lookup_result"] = (
                        "error",
                        f"API not reachable at {api_url} ({exc.__class__.__name__}). "
                        "Start it with `ecom serve`.",
                    )
        status, payload = st.session_state.get("lookup_result", (None, None))
        if status == "ok":
            st.metric(
                f"User {payload['user_id']}: propensity (next 7 days)",
                f"{payload['propensity']:.1%}",
            )
            st.write(
                f"Decile **{payload['decile']}** · tier **{payload['propensity_tier']}** "
                f"· segment **{payload['segment']}**"
            )
            st.caption(
                f"Model {payload['model']['name']} v{payload['model']['version']} · "
                f"live score matches stored: {payload['live_matches_stored']}"
            )
        elif status == "error":
            st.error(payload)
    with c2:
        try:
            segs = requests.get(f"{api_url}/segments", timeout=5).json()
            names = [s["segment"] for s in segs]
            seg = st.selectbox("Segment", names)
            k = st.slider("Top k", 5, 100, 10)
            top = requests.get(f"{api_url}/segments/{seg}/top", params={"k": k}, timeout=5).json()
            st.dataframe(pd.DataFrame(top["users"]), use_container_width=True, hide_index=True)
        except requests.RequestException:
            st.info(f"Segment browser needs the API at {api_url} (`ecom serve`).")

st.sidebar.markdown("---")
st.sidebar.caption(f"Run folder: `{OUT.relative_to(CFG.root)}`")
