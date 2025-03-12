"""SHAP explanations for the tree model, translated into plain-English drivers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.pipeline import Pipeline

FEATURE_LABELS = {
    "recency_days": "days since the user's last visit",
    "tenure_days": "days since the user was first seen in the window",
    "days_active": "number of active days",
    "n_sessions": "number of sessions",
    "n_events": "number of events",
    "n_views": "number of product views",
    "n_carts": "number of add-to-carts",
    "n_purchases": "number of past purchases",
    "monetary": "past spend",
    "avg_order_value": "average past order value",
    "view_to_cart": "view-to-cart rate",
    "cart_to_purchase": "cart-to-purchase rate",
    "cart_abandonment_rate": "cart abandonment rate",
    "distinct_categories": "number of categories browsed",
    "distinct_brands": "number of brands browsed",
    "distinct_products": "number of distinct products viewed",
    "avg_price_viewed": "average price of viewed products",
    "max_price_viewed": "most expensive product viewed",
    "relative_price_level": "price level viewed relative to the category median",
    "price_sensitivity": "price sensitivity (carts cheaper items than it views)",
    "weekend_share": "share of activity on weekends",
    "evening_share": "share of activity in the evening",
    "avg_session_duration_s": "average session length",
    "events_per_session": "events per session",
    "sessions_last_7d": "sessions in the last 7 days",
    "views_last_7d": "views in the last 7 days",
    "carts_last_7d": "add-to-carts in the last 7 days",
    "purchases_last_7d": "purchases in the last 7 days",
    "activity_trend": "recent activity vs the user's 3-week average",
}


def label_for(feature: str) -> str:
    if feature.startswith("top_category_"):
        return f"mostly browsing '{feature.removeprefix('top_category_')}'"
    return FEATURE_LABELS.get(feature, feature)


def explain_tree_model(
    pipe: Pipeline, X: pd.DataFrame, fig_dir, seed: int = 0, n: int = 3000, top_k: int = 5
) -> dict:
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xs = X.sample(min(n, len(X)), random_state=seed)
    Xt = pd.DataFrame(prep.transform(Xs), columns=prep.get_feature_names_out())
    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(Xt))
    mean_abs = np.abs(sv).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": Xt.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    shap.summary_plot(sv, Xt, show=False, max_display=15, plot_size=(9, 6))
    plt.title("SHAP summary - drivers of next-7-day purchase (log-odds)")
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", dpi=130)
    plt.close("all")
    shap.summary_plot(sv, Xt, plot_type="bar", show=False, max_display=15, plot_size=(9, 6))
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_importance.png", dpi=130)
    plt.close("all")
    for feat in importance["feature"].head(3):
        shap.dependence_plot(feat, sv, Xt, show=False, interaction_index="auto")
        plt.title(f"SHAP dependence: {feat}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"shap_dependence_{feat}.png", dpi=130)
        plt.close("all")

    drivers = []
    for feat in importance["feature"].head(top_k):
        j = list(Xt.columns).index(feat)
        x, s = Xt[feat].to_numpy(dtype=float), sv[:, j]
        rho = float(stats.spearmanr(x, s).statistic) if np.std(x) > 0 else 0.0
        if feat.startswith("top_category_"):
            hi, lo = s[x == 1], s[x == 0]
        else:
            q1, q3 = np.quantile(x, [0.25, 0.75])
            hi, lo = s[x >= q3], s[x <= q1]
        gap = float(hi.mean() - lo.mean()) if len(hi) and len(lo) else 0.0
        odds = float(np.exp(gap))
        label = label_for(feat)
        if feat.startswith("top_category_"):
            text = (
                f"Users {label} have {odds:.2f}x the purchase odds of other users, all else equal."
            )
        else:
            direction = "higher" if rho > 0 else "lower"
            text = (
                f"{label[0].upper()}{label[1:]}: the more of it, the {direction} the chance "
                f"of buying next week. Users in the top quartile have {odds:.2f}x the "
                f"purchase odds of users in the bottom quartile."
            )
        drivers.append(
            {
                "feature": feat,
                "label": label_for(feat),
                "mean_abs_shap": float(mean_abs[j]),
                "spearman_value_vs_shap": rho,
                "odds_ratio_top_vs_bottom_quartile": odds,
                "text": text,
            }
        )
    return {"importance": importance, "drivers": drivers}
