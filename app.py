"""Profit & Loss dashboard.

Run:   streamlit run app.py
Needs: streamlit pandas plotly openpyxl matplotlib   (prophet is optional)
Keep pl_loader.py in the same folder.
"""
from __future__ import annotations

import importlib.util
import inspect
import io
import itertools
import logging
import re
import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pl_analysis import (BUCKETS, CAT_LABEL, CURRENCY, EXPENSE_LINES, LINES, add_entity, build_report,
                         classify_cost, compact, favourability, fpct, mlabel, monthly_pnl, pct, statement)
from pl_export import build_excel as _build_excel
from pl_export import build_pdf as _build_pdf
from pl_loader import check, check_monthly, leaves, load_pl

HAS_PROPHET = importlib.util.find_spec("prophet") is not None
FORECAST_METRICS = ["Revenue", "Gross Profit", "Total Expenses", "Net Profit"]


# ===========================================================================
# Small UI helpers (work across Streamlit versions)
# ===========================================================================
def _width_kw(fn):
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


def show_df(df, **kw):
    st.dataframe(df, **_width_kw(st.dataframe), **kw)


def show_fig(fig, key=None):
    st.plotly_chart(fig, key=key, **_width_kw(st.plotly_chart))


def fmt_table(df, int_cols=(), pct_cols=(), dec_cols=(), signed_cols=()):
    def f_int(v):
        return "–" if pd.isna(v) else f"{v:,.0f}"

    def f_dec(v):
        return "–" if pd.isna(v) else f"{v:,.2f}"

    def f_pct(signed):
        return lambda v: "–" if pd.isna(v) else fpct(v, signed=signed)

    fm = {c: f_int for c in int_cols}
    fm.update({c: f_dec for c in dec_cols})
    fm.update({c: f_pct(c in signed_cols) for c in pct_cols})
    return df.style.format({c: f for c, f in fm.items() if c in df.columns}, na_rep="–")


def kpi_cards(items):
    """Responsive KPI cards that wrap instead of truncating long numbers."""
    import html
    cards = "".join(
        f'<div class="kpi"><div class="kpi-l">{html.escape(l)}</div><div class="kpi-v">{html.escape(v)}</div>'
        f'<div class="kpi-s">{html.escape(sub)}</div></div>' for l, v, sub in items)
    css = ("<style>.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px;margin:6px 0 16px}"
           ".kpi{border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:12px 16px;background:rgba(128,128,128,.07)}"
           ".kpi-l{font-size:.85rem;opacity:.75}.kpi-v{font-size:1.7rem;font-weight:700;line-height:1.3;word-break:break-word}"
           ".kpi-s{font-size:.8rem;opacity:.7;min-height:1.1em}</style>")
    st.markdown(css + f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def bullets(items):
    if items:
        st.markdown("\n".join(f"- {x}" for x in items))


def tbl(df, ints=(), pcts=(), signed=(), decs=(), **kw):
    show_df(fmt_table(df, int_cols=ints, pct_cols=pcts, dec_cols=decs, signed_cols=signed), hide_index=True, **kw)


@st.cache_data(show_spinner="Building report…")
def cached_report(leaf, partial_labels, recon_ok, semi_share, overrides):
    return build_report(leaf, partial_labels, recon_ok, semi_share, dict(overrides))


@st.cache_data(show_spinner="Building PDF…")
def pdf_bytes(R, fc_pack):
    return _build_pdf(R, fc_pack)


@st.cache_data(show_spinner=False)
def excel_bytes(R, recon, clean):
    return _build_excel(R, recon, clean)


# ===========================================================================
# Data preparation
# ===========================================================================
@st.cache_data(show_spinner="Reading file…")
def parse_upload(name: str, content: bytes):
    return load_pl(io.BytesIO(content), name)


def make_label(name, company, branch):
    m = re.search(r"(\d{3,})", name)
    return f"{company} · {branch} ({m.group(1) if m else name})"


def partial_months(df):
    """Months that are incomplete because the report period ends mid-month."""
    out = set()
    for pe in pd.to_datetime(df["PeriodEnd"]).drop_duplicates():
        if pe.normalize() != (pe + pd.offsets.MonthEnd(0)).normalize():
            out.add(pe.to_period("M").to_timestamp())
    return out


def apply_filters(df, sources, month_lo, month_hi, exclude_partial):
    d = df[df["Source"].isin(sources)]
    if exclude_partial:
        d = d[~d["Date"].isin(partial_months(d))]
    return d[(d["Date"] >= month_lo) & (d["Date"] <= month_hi)]


def top_expense_lines(leaf, n=10):
    e = leaf[leaf["Category"].str.startswith("Expenses")]
    t = e.groupby(["Particulars", "Category"], as_index=False)["Amount"].sum()
    t["Category"] = t["Category"].map(CAT_LABEL)
    return t.sort_values("Amount", ascending=False).head(n).reset_index(drop=True)


def detect_overlap(df_all):
    """Find files that duplicate, or are the consolidation of, other uploaded files.

    A file is only flagged when its monthly sales AND expenses equal the sum of one or
    more other files from the same company (a single match = duplicate upload).
    Merely being smaller than another branch is NOT overlap.
    Returns [(flagged_source, [component_sources])].
    """
    leaf = df_all[df_all["IsLeaf"]]
    months = sorted(leaf["Date"].unique())
    vec = {}
    for s, g in leaf.groupby("Source"):
        sales = g[g["Category"] == "Income"].groupby("Date")["Amount"].sum().reindex(months, fill_value=0.0)
        exp = g[g["Category"].str.startswith("Expenses")].groupby("Date")["Amount"].sum().reindex(months, fill_value=0.0)
        vec[s] = np.concatenate([sales.to_numpy(), exp.to_numpy()])
    company = df_all.drop_duplicates("Source").set_index("Source")["Company"].to_dict()
    order = [s for s in df_all["Source"].drop_duplicates() if s in vec]
    flagged, found = set(), []
    for x in reversed(order):                     # later uploads are the ones set aside
        if not np.any(vec[x]):
            continue
        pool = [s for s in order if s != x and s not in flagged and company[s] == company[x]]
        sizes = range(1, len(pool) + 1) if len(pool) <= 12 else (1, len(pool))
        hit = None
        for k in sizes:
            for combo in itertools.combinations(pool, k):
                if np.allclose(sum(vec[c] for c in combo), vec[x], rtol=1e-3, atol=1.0):
                    hit = combo
                    break
            if hit:
                break
        if hit:
            flagged.add(x)
            found.append((x, list(hit)))
    return found


# ===========================================================================
# Forecasting
# ===========================================================================
def _linear_fc(hist, horizon):
    n, y = len(hist), hist["y"].to_numpy(float)
    x = np.arange(n)
    slope, icpt = np.polyfit(x, y, 1) if n >= 2 else (0.0, y[0])
    fit = icpt + slope * x
    sd = float(np.std(y - fit, ddof=2)) if n > 2 else 0.0
    fut_dates = pd.date_range(hist["ds"].max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    xf = np.arange(n, n + horizon)
    yhat = np.concatenate([fit, icpt + slope * xf])
    ds = pd.DatetimeIndex(list(hist["ds"]) + list(fut_dates))
    band = 1.2816 * sd
    return pd.DataFrame({"ds": ds, "yhat": yhat, "yhat_lower": yhat - band, "yhat_upper": yhat + band})


def _prophet_fc(hist, horizon):
    from prophet import Prophet
    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.ERROR)
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                    interval_width=0.8, n_changepoints=max(1, min(5, len(hist) - 2)))
    model.fit(hist)
    fc = model.predict(model.make_future_dataframe(periods=horizon, freq="MS"))
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]


@st.cache_data(show_spinner="Fitting forecast model…")
def run_forecast(dates: tuple, values: tuple, horizon: int, model: str):
    hist = pd.DataFrame({"ds": pd.to_datetime(list(dates)), "y": [float(v) for v in values]})
    used, note, fc = "Linear trend", "", None
    if model == "Prophet":
        if HAS_PROPHET and len(hist) >= 6:
            try:
                fc, used = _prophet_fc(hist, horizon), "Prophet"
            except Exception as exc:                       # noqa: BLE001
                note = f"Prophet failed ({exc}); showing the linear trend instead."
        else:
            note = "Prophet needs the package installed and at least 6 months of history; showing the linear trend instead."
    if fc is None:
        fc = _linear_fc(hist, horizon)
    fc = fc.copy()
    fc["kind"] = np.where(fc["ds"] <= hist["ds"].max(), "fit", "forecast")
    return fc, used, note


def forecast_figure(actual, fc, metric, used):
    fig = go.Figure()
    f = fc[fc["kind"] == "forecast"]
    fig.add_trace(go.Scatter(x=pd.concat([fc["ds"], fc["ds"][::-1]]),
                             y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
                             fill="toself", line=dict(width=0), name="80% range", hoverinfo="skip",
                             fillcolor="rgba(120,120,255,0.15)"))
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", name=f"{used} fit", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=f["ds"], y=f["yhat"], mode="lines+markers", name="Forecast"))
    fig.update_layout(title=f"{metric}: actual and forecast ({used})", yaxis_title=CURRENCY, hovermode="x unified")
    fig.update_xaxes(tickformat="%b %Y", dtick="M1")
    return fig


# ===========================================================================
# Variance helpers
# ===========================================================================
def month_vs_month(m, base, comp):
    rows = []
    for ln in LINES:
        b, c = float(m.loc[base, ln]), float(m.loc[comp, ln])
        ch = c - b
        rows.append({"Line": ln, mlabel(base): b, mlabel(comp): c, "Change": ch,
                     "Change %": pct(ch, abs(b)), "Impact": favourability(ln, ch)})
    return pd.DataFrame(rows)


def read_budget(upload):
    name = (getattr(upload, "name", "") or "").lower()
    df = pd.read_csv(upload) if name.endswith(".csv") else pd.read_excel(upload)
    df.columns = [str(c).strip().title() for c in df.columns]
    missing = {"Month", "Line", "Budget"} - set(df.columns)
    if missing:
        raise ValueError("Budget file needs columns Month, Line, Budget (missing: " + ", ".join(sorted(missing)) + ").")
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df["Budget"] = pd.to_numeric(df["Budget"].astype(str).str.replace(",", ""), errors="coerce")
    df["Line"] = df["Line"].astype(str).str.strip()
    unknown = sorted(set(df["Line"]) - set(LINES))
    return df.dropna(subset=["Month", "Budget"])[["Month", "Line", "Budget"]], unknown


def budget_vs_actual(m, budget):
    act = m.reset_index().melt(id_vars="Month", value_vars=LINES, var_name="Line", value_name="Actual")
    j = act.merge(budget, on=["Month", "Line"], how="inner")
    j["Variance"] = j["Actual"] - j["Budget"]
    j["Variance %"] = [pct(v, abs(b)) for v, b in zip(j["Variance"], j["Budget"])]
    j["Impact"] = [favourability(l, v) for l, v in zip(j["Line"], j["Variance"])]
    return j


# ===========================================================================
# Reconciliation
# ===========================================================================
def recon_summary(df_all, controls, tol_abs):
    rows = []
    for src, g in df_all.groupby("Source"):
        c = check(g, controls[src], tol=tol_abs)
        ok = c["OK"]
        rows.append({
            "Source": src,
            "Gross profit (calc)": c["Gross profit"], "INCOME (report)": c["Report INCOME"],
            "Expenses (calc)": c["Expenses"], "EXPENSES (report)": c["Report EXPENSES"],
            "Net profit (calc)": c["Net profit"], "Profit (report)": c["Report profit"],
            "Largest diff": max(abs(c["Gross profit"] - (c["Report INCOME"] or 0)),
                                abs(c["Expenses"] - (c["Report EXPENSES"] or 0)),
                                abs(c["Net profit"] - (c["Report profit"] or 0))),
            "Status": "✅ Ties" if ok else "⚠️ Check"})
    return pd.DataFrame(rows)


# ===========================================================================
# Management report (rendered on screen; the PDF and Excel read the same data)
# ===========================================================================
def render_report(R):
    m = R["m"]
    st.header("📝 Management Report")
    st.caption(f"{R['period']}  ·  {R['n_months']} months  ·  amounts in {CURRENCY}  ·  the PDF on the Export tab contains the same sections")

    st.subheader("Companies and branches covered")
    for co, brs in R["scope"].items():
        st.markdown(f"**{co}**\n" + "\n".join(f"- {b}" for b in brs))

    st.subheader("Executive summary")
    bullets(R["summary"])

    # ---- 1. previous periods ------------------------------------------------------------------------
    st.subheader("1. Performance against previous periods")
    bullets(R["cmp_notes"])
    for key, title in (("cmp_month", "Latest month vs previous month"), ("cmp_quarter", "Latest 3 months vs previous 3 months")):
        c = R[key]
        if c:
            st.markdown(f"**{title}** ({c['current']} vs {c['previous']})")
            t = c["table"].rename(columns={"Current": c["current"], "Previous": c["previous"]})
            tbl(t, ints=[c["current"], c["previous"], "Change"], pcts=["Change %"], signed=["Change %"])

    # ---- 2. branches ---------------------------------------------------------------------------------
    bt = R["branch"]
    st.subheader("2. Branch performance and comparison")
    bullets(R["branch_notes"])
    cols = [c for c in ["Branch", "Revenue", "Revenue share %", "Gross Profit", "Gross margin %", "GM vs group (pp)",
                        "Total Expenses", "Net Profit", "Net margin %", "Net profit rank", "Rev MoM %", "Rev 3M vs prior %", "Trend %/mo"] if c in bt]
    tbl(bt[cols], ints=["Revenue", "Gross Profit", "Total Expenses", "Net Profit", "Net profit rank"],
        pcts=["Revenue share %", "Gross margin %", "GM vs group (pp)", "Net margin %", "Rev MoM %", "Rev 3M vs prior %", "Trend %/mo"],
        signed=["GM vs group (pp)", "Rev MoM %", "Rev 3M vs prior %", "Trend %/mo"])
    top = bt.head(20)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(top.iloc[::-1], x="Revenue", y="Branch", orientation="h", title="Revenue by branch"),
                    key="rp_br_rev", **_width_kw(st.plotly_chart))
    mg = top.melt(id_vars="Branch", value_vars=["Gross margin %", "Net margin %"], var_name="Measure", value_name="%")
    c2.plotly_chart(px.bar(mg, x="Branch", y="%", color="Measure", barmode="group", title="Margins by branch"),
                    key="rp_br_mg", **_width_kw(st.plotly_chart))
    names = list(bt["Branch"].head(8))
    brm = R["branch_rev_monthly"].loc[names].T
    brm.index.name = "Month"
    long = brm.reset_index().melt(id_vars="Month", var_name="Branch", value_name="Revenue")
    fig = px.line(long, x="Month", y="Revenue", color="Branch", markers=True, title=f"Monthly revenue, top {len(names)} branches")
    fig.update_xaxes(tickformat="%b %Y", dtick="M1")
    show_fig(fig, "rp_br_trend")

    # ---- 3. income lines -------------------------------------------------------------------------------
    st.subheader("3. Income lines")
    bullets(R["income_notes"])
    ig = R["income_groups"]
    mcols = ["Total", "Share %", "Last month", "Prev month", "MoM %", "3M vs prior 3M %", "Trend %/mo"]
    st.markdown("**By income group**")
    tbl(ig, ints=["Total", "Last month", "Prev month"], pcts=["Share %", "MoM %", "3M vs prior 3M %", "Trend %/mo"],
        signed=["MoM %", "3M vs prior 3M %", "Trend %/mo"])
    st.markdown("**Top income accounts**")
    tbl(R["income_accounts"].head(15), ints=["Total", "Last month", "Prev month"],
        pcts=["Share %", "MoM %", "3M vs prior 3M %", "Trend %/mo"], signed=["MoM %", "3M vs prior 3M %", "Trend %/mo"])
    if len(R["product_margin"]):
        st.markdown("**Gross margin by product** (sales account vs the matching COGS account)")
        tbl(R["product_margin"], ints=["Sales", "COGS", "Gross profit"], pcts=["Gross margin %", "Share of sales %"])
    bm = R["income_monthly_by_group"].rename_axis("Month").reset_index().melt(id_vars="Month", var_name="Income group", value_name=CURRENCY)
    fig = px.bar(bm, x="Month", y=CURRENCY, color="Income group", title="Monthly income by group")
    fig.update_xaxes(tickformat="%b %Y", dtick="M1")
    show_fig(fig, "rp_inc")

    # ---- 4. expenditure ------------------------------------------------------------------------------------
    st.subheader("4. Expenditure analysis")
    bullets(R["exp_notes"])
    st.markdown("**By expense group**")
    tbl(R["exp_groups"], ints=["Total", "Last month", "Prev month"], pcts=["Share %", "% of revenue", "MoM %", "3M vs prior 3M %", "Trend %/mo"],
        signed=["MoM %", "3M vs prior 3M %", "Trend %/mo"])
    st.markdown("**Largest expense accounts**")
    tbl(R["exp_accounts"].head(15), ints=["Total", "Last month", "Prev month"], pcts=["Share %", "% of revenue", "MoM %", "3M vs prior 3M %", "Trend %/mo"],
        signed=["MoM %", "3M vs prior 3M %", "Trend %/mo"])
    mv = R["exp_movers"]
    if len(mv):
        st.markdown(f"**Biggest movers** ({R['exp_movers_basis']})")
        tbl(pd.concat([mv.head(8), mv.tail(8)]).drop_duplicates(), ints=["Previous", "Current", "Change"], pcts=["Change %"], signed=["Change %"])
    c1, c2 = st.columns(2)
    em = R["exp_monthly_by_group"].rename_axis("Month").reset_index().melt(id_vars="Month", var_name="Group", value_name=CURRENCY)
    fig = px.bar(em, x="Month", y=CURRENCY, color="Group", title="Operating expenses by group")
    fig.update_xaxes(tickformat="%b %Y", dtick="M1")
    c1.plotly_chart(fig, key="rp_exp", **_width_kw(st.plotly_chart))
    ms = R["margin_series"].rename_axis("Month").reset_index()
    fig = px.line(ms, x="Month", y="Expenses % revenue", markers=True, title="Operating expenses as % of revenue")
    fig.update_xaxes(tickformat="%b %Y", dtick="M1")
    c2.plotly_chart(fig, key="rp_exp_ratio", **_width_kw(st.plotly_chart))

    # ---- 5. fixed vs variable -----------------------------------------------------------------------------------
    st.subheader("5. Fixed and variable costs")
    bullets(R["cost_notes"])
    be = R["break_even"]
    if "break_even_monthly" in be:
        kpi_cards([("Contribution margin", f"{be['contribution_margin_pct']:.1f}%", "of revenue after variable costs"),
                   ("Fixed costs per month", compact(be["fixed_monthly"]), f"semi-variable treated {R['semi_fixed_share'] * 100:.0f}% fixed"),
                   ("Break-even revenue / month", compact(be["break_even_monthly"]), f"actual average {compact(be['avg_revenue_monthly'])}"),
                   ("Margin of safety", f"{be['margin_of_safety_pct']:.0f}%", "how far revenue is above break-even")])
    tbl(R["cost_buckets"], ints=["Total", "Monthly average", "Last month", "Prev month"],
        pcts=["% of all costs", "% of revenue", "MoM %"], signed=["MoM %"])
    cm = R["cost_monthly"][BUCKETS].rename_axis("Month").reset_index().melt(id_vars="Month", var_name="Cost type", value_name=CURRENCY)
    fig = px.bar(cm, x="Month", y=CURRENCY, color="Cost type", title="Operating expenses by cost behaviour")
    fig.update_xaxes(tickformat="%b %Y", dtick="M1")
    show_fig(fig, "rp_cost")
    ca = R["cost_accounts"]
    cols = st.columns(3)
    for col, b in zip(cols, BUCKETS):
        col.markdown(f"**Largest {b.lower()} accounts**")
        with col:
            show_df(fmt_table(ca[ca["Cost type"] == b].head(8)[["Account", "Total"]], int_cols=["Total"]), hide_index=True)
    if hasattr(st, "data_editor"):
        with st.expander("Review or change how each expense account is classified"):
            st.caption("Labels are first guessed from the account name. Change any account to Fixed, Semi-variable or Variable "
                       "and the whole report, PDF and Excel update.")
            edited = st.data_editor(R["classification"], hide_index=True, disabled=["Account"], key="cost_editor",
                                    column_config={"Cost type": st.column_config.SelectboxColumn("Cost type", options=BUCKETS, required=True)})
            new = {a: b for a, b in zip(edited["Account"], edited["Cost type"]) if b != classify_cost(a)}
            if new != st.session_state.get("cost_overrides", {}):
                st.session_state["cost_overrides"] = new
                st.rerun()

    # ---- 6. trends ------------------------------------------------------------------------------------------------
    st.subheader("6. Trends")
    bullets(R["trend_notes"])
    if len(m) >= 2:
        long = m.reset_index().melt(id_vars="Month", value_vars=["Revenue", "Gross Profit", "Net Profit"], var_name="Line", value_name=CURRENCY)
        fig = px.line(long, x="Month", y=CURRENCY, color="Line", markers=True, title="Revenue, gross profit and net profit")
        fig.update_xaxes(tickformat="%b %Y", dtick="M1")
        show_fig(fig, "rp_tr1")
        mg = R["margin_series"].rename_axis("Month").reset_index().melt(id_vars="Month", var_name="Measure", value_name="%")
        fig = px.line(mg, x="Month", y="%", color="Measure", markers=True, title="Margins and expense ratio")
        fig.update_xaxes(tickformat="%b %Y", dtick="M1")
        show_fig(fig, "rp_tr2")


# ===========================================================================
# App
# ===========================================================================
def main():
    st.set_page_config(page_title="Profit & Loss Dashboard", layout="wide")
    st.title("📊 Profit & Loss Dashboard")

    # ---- Data input ---------------------------------------------------------------
    st.sidebar.title("📥 Data Input")
    files = st.sidebar.file_uploader("Upload one or more P&L exports", type=["xlsx", "xlsm", "csv"],
                                     accept_multiple_files=True)
    if not files:
        st.info("Upload one or more Profit & Loss exports (.xlsx or .csv) in the sidebar to begin.")
        st.stop()

    frames, controls = [], {}
    for f in files:
        try:
            d, ctl = parse_upload(f.name, f.getvalue())
        except Exception as exc:                                    # noqa: BLE001
            st.sidebar.error(f"{f.name}: {exc}")
            continue
        label = make_label(f.name, d["Company"].iloc[0] if len(d) else "Unknown",
                           d["Branch"].iloc[0] if len(d) else "")
        while label in controls:
            label += "*"
        d = d.copy()
        d["Source"] = label
        frames.append(d)
        controls[label] = ctl
    if not frames:
        st.stop()
    df_all = pd.concat(frames, ignore_index=True)
    df_all["Date"] = pd.to_datetime(df_all["Date"])

    # ---- Filters --------------------------------------------------------------------
    st.sidebar.subheader("📍 Filters")
    sources = list(controls)
    overlaps = detect_overlap(df_all)
    meta = df_all.drop_duplicates("Source").set_index("Source")[["Company", "Branch"]]
    meta = meta.loc[[x for x in sources if x in meta.index]]

    companies = sorted(meta["Company"].unique())
    sel_co = st.sidebar.multiselect("Company", companies, default=companies)
    branch_opts = sorted(meta[meta["Company"].isin(sel_co)]["Branch"].unique())
    sel_br = st.sidebar.multiselect("Branch", branch_opts, default=branch_opts)
    candidates = meta[meta["Company"].isin(sel_co) & meta["Branch"].isin(sel_br)].index.tolist()

    # A file that duplicates, or is the consolidation of, other selected files is left out
    # so nothing is counted twice. Filter to its own branch label to view it on its own.
    left_out = [(x, parts) for x, parts in overlaps if x in candidates and any(p in candidates for p in parts)]
    drop = {x for x, _ in left_out}
    sel = [x for x in candidates if x not in drop]
    for x, parts in left_out:
        st.sidebar.warning(f"**{x}** equals the sum of {', '.join(parts)} (sales and expenses, every month), so it "
                           "looks like a consolidated or duplicate file and is left out to avoid double counting.")
    if not sel:
        st.warning("No files match the Company / Branch selection.")
        st.stop()
    st.sidebar.caption(f"{len(sel)} file(s) in view: " + "; ".join(meta.loc[sel, "Branch"]))

    excl = st.sidebar.checkbox("Exclude incomplete month", value=True,
                               help="The report period ends mid-month, so the last month is partial. "
                                    "Leaving it in distorts trends, growth rates and forecasts.")
    avail = df_all[df_all["Source"].isin(sel)]["Date"].drop_duplicates().sort_values().tolist()
    lab2d = {mlabel(d): d for d in avail}
    if len(avail) > 1:
        lo_l, hi_l = st.sidebar.select_slider("Month range", options=list(lab2d), value=(mlabel(avail[0]), mlabel(avail[-1])))
        lo, hi = lab2d[lo_l], lab2d[hi_l]
    else:
        lo = hi = avail[0]

    d_sel = apply_filters(df_all, sel, lo, hi, excl)
    leaf = leaves(d_sel)
    if leaf.empty:
        st.warning("No data for the current filters.")
        st.stop()
    m = monthly_pnl(leaf)
    partial = partial_months(df_all[df_all["Source"].isin(sel)]) if excl else set()
    partial = {p for p in partial if lo <= p <= hi}

    tol = st.sidebar.number_input(f"Reconciliation tolerance ({CURRENCY})", min_value=0.0, value=1.0, step=1.0)
    recon = recon_summary(df_all, controls, tol)
    recon_ok = bool(recon[recon["Source"].isin(sel)]["Status"].str.contains("Ties").all())
    semi_share = st.sidebar.slider("Semi-variable costs treated as fixed (%)", 0, 100, 50,
                                   help="Only used for the break-even figures in the Management Report.") / 100
    overrides = st.session_state.get("cost_overrides", {})
    R = cached_report(leaf, tuple(mlabel(p) for p in sorted(partial)), recon_ok, semi_share, tuple(sorted(overrides.items())))
    if not recon_ok:
        st.error("⚠️ A selected file does not tie out to its own report totals. Open the Reconciliation tab before using these numbers.")

    tabs = st.tabs(["📊 Overview", "📝 Management Report", "📈 Trends", "🔮 Forecasts", "📉 Variance Analysis", "📤 Export", "🔎 Reconciliation"])

    # ================= Overview =================
    with tabs[0]:
        st.header("📊 Profit & Loss Overview")
        st.caption(f"{R['period']}  ·  {R['n_months']} months  ·  {R['n_entities']} branch(es)  ·  amounts in {CURRENCY}")
        k, mm = R["kpis"], R["m"]
        kpi_cards([("Revenue", compact(k["Revenue"]), f"{k['Revenue']:,.0f}"),
                   ("Cost of goods sold", compact(k["COGS"]), f"{fpct(pct(k['COGS'], k['Revenue']), signed=False)} of revenue"),
                   ("Gross profit", compact(k["Gross Profit"]), f"{fpct(k['Gross margin %'], signed=False)} gross margin"),
                   ("Operating expenses", compact(k["Total Expenses"]), f"{fpct(k['Expenses % revenue'], signed=False)} of revenue"),
                   ("Net profit", compact(k["Net Profit"]), f"{fpct(k['Net margin %'], signed=False)} net margin")])
        tot = mm.sum()
        st.markdown("**Operating expenses by group**")
        kpi_cards([(ln, compact(tot[ln]), f"{fpct(pct(tot[ln], tot['Total Expenses']), signed=False)} of operating expenses") for ln in EXPENSE_LINES])
        if partial:
            st.caption("Totals exclude the incomplete month(s): " + ", ".join(mlabel(p) for p in sorted(partial))
                       + ". Untick 'Exclude incomplete month' in the sidebar to include them.")

        st.subheader("Key takeaways")
        bullets(R["summary"])
        st.caption("The Management Report tab explains each of these in detail: branches, income lines, costs and trends.")

        st.subheader("P&L statement")
        st.caption("Each column is a month; the last column is the total for the period.")
        stmt = R["statement"]
        show_df(fmt_table(stmt, int_cols=list(stmt.columns)), height=min(460, 36 * (len(stmt) + 1) + 3))

        st.subheader("Monthly performance")
        c1, c2 = st.columns(2)
        fig = px.bar(mm.reset_index(), x="Month", y="Revenue", title="Revenue by month")
        fig.update_xaxes(tickformat="%b %Y", dtick="M1")
        c1.plotly_chart(fig, key="ov_rev", **_width_kw(st.plotly_chart))
        pn = mm.reset_index().melt(id_vars="Month", value_vars=["Gross Profit", "Net Profit"], var_name="Line", value_name=CURRENCY)
        fig = px.bar(pn, x="Month", y=CURRENCY, color="Line", barmode="group", title="Gross profit and net profit by month")
        fig.update_xaxes(tickformat="%b %Y", dtick="M1")
        c2.plotly_chart(fig, key="ov_pn", **_width_kw(st.plotly_chart))

        st.subheader("Where the money goes")
        left, right = st.columns(2)
        exp_tot = mm[EXPENSE_LINES].sum()
        exp_tot = exp_tot[exp_tot > 0]
        if not exp_tot.empty:
            left.plotly_chart(px.pie(values=exp_tot.values, names=exp_tot.index, hole=0.5, title="Operating expenses by group"),
                              key="ov_pie", **_width_kw(st.plotly_chart))
        top = top_expense_lines(leaf, 10)
        if not top.empty:
            right.plotly_chart(px.bar(top.iloc[::-1], x="Amount", y="Particulars", color="Category", orientation="h",
                                      title="Top 10 expense accounts"), key="ov_top", **_width_kw(st.plotly_chart))

        if R["n_entities"] > 1:
            st.subheader("Branch comparison")
            bt = R["branch"]
            tbl(bt[["Branch", "Revenue", "Revenue share %", "Gross Profit", "Gross margin %", "Net Profit", "Net margin %"]],
                ints=["Revenue", "Gross Profit", "Net Profit"], pcts=["Revenue share %", "Gross margin %", "Net margin %"])
            show_fig(px.bar(bt, x="Branch", y="Net Profit", title="Net profit by branch"), "ov_branch")
            st.caption("Branch net profit only reflects costs posted to that branch. See the Management Report for the full comparison.")

        with st.expander("Underlying accounts (clean data)"):
            clean_view = leaf[["Source", "Date", "Category", "Section", "Particulars", "Debit", "Credit", "Amount"]]
            show_df(clean_view.sort_values(["Date", "Category", "Particulars"]), hide_index=True)

    # ================= Management Report =================
    with tabs[1]:
        render_report(R)

    # ================= Trends =================
    with tabs[2]:
        st.header("📈 Trend Analysis")
        if len(m) < 2:
            st.info("Select at least two months to see trends.")
        else:
            long = m.reset_index().melt(id_vars="Month", value_vars=["Revenue", "Gross Profit", "Net Profit"],
                                        var_name="Line", value_name=CURRENCY)
            fig = px.line(long, x="Month", y=CURRENCY, color="Line", markers=True, title="Revenue, gross profit and net profit")
            fig.update_xaxes(tickformat="%b %Y", dtick="M1")
            show_fig(fig, "tr_main")

            ex = m.reset_index().melt(id_vars="Month", value_vars=EXPENSE_LINES, var_name="Group", value_name=CURRENCY)
            fig = px.bar(ex, x="Month", y=CURRENCY, color="Group", title="Operating expenses by group")
            fig.update_xaxes(tickformat="%b %Y", dtick="M1")
            show_fig(fig, "tr_exp")

            mg = pd.DataFrame({"Month": m.index,
                               "Gross margin %": [pct(a, b) for a, b in zip(m["Gross Profit"], m["Revenue"])],
                               "Net margin %": [pct(a, b) for a, b in zip(m["Net Profit"], m["Revenue"])]})
            fig = px.line(mg.melt(id_vars="Month", var_name="Measure", value_name="%"), x="Month", y="%",
                          color="Measure", markers=True, title="Margins")
            fig.update_xaxes(tickformat="%b %Y", dtick="M1")
            show_fig(fig, "tr_margin")

            st.subheader("Account drill-down")
            order = leaf.groupby("Particulars")["Amount"].apply(lambda s: s.abs().sum()).sort_values(ascending=False)
            picks = st.multiselect("Accounts", list(order.index), default=list(order.index[:3]))
            if picks:
                dd = (leaf[leaf["Particulars"].isin(picks)].groupby(["Date", "Particulars"], as_index=False)["Amount"].sum())
                fig = px.line(dd, x="Date", y="Amount", color="Particulars", markers=True, title="Selected accounts")
                fig.update_xaxes(tickformat="%b %Y", dtick="M1")
                show_fig(fig, "tr_drill")

    # ================= Forecasts =================
    fc_pack = None
    with tabs[3]:
        st.header("🔮 Forecasts")
        if len(m) < 3:
            st.info("At least 3 months of history are needed to forecast.")
        else:
            a, b, c = st.columns(3)
            metric = a.selectbox("Metric", FORECAST_METRICS)
            horizon = b.slider("Months ahead", 1, 12, 6)
            model = c.selectbox("Model", ["Prophet", "Linear trend"] if HAS_PROPHET else ["Linear trend"])
            if not excl and partial_months(df_all[df_all["Source"].isin(sel)]):
                st.warning("The incomplete month is included in the history, so the forecast will be biased low.")
            actual = m[metric]
            fc, used, note = run_forecast(tuple(str(d.date()) for d in actual.index),
                                          tuple(float(v) for v in actual.values), horizon, model)
            if note:
                st.warning(note)
            show_fig(forecast_figure(actual, fc, metric, used), "fc_main")
            fut = fc[fc["kind"] == "forecast"][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            st.metric(f"Forecast {metric.lower()}, next {horizon} months", compact(fut["yhat"].sum()))
            fut["Month"] = fut["ds"].map(mlabel)
            fut = fut.rename(columns={"yhat": "Forecast", "yhat_lower": "Low (80%)", "yhat_upper": "High (80%)"})
            show_df(fmt_table(fut[["Month", "Forecast", "Low (80%)", "High (80%)"]],
                              int_cols=["Forecast", "Low (80%)", "High (80%)"]), hide_index=True)
            st.caption(f"With only {len(actual)} months of history treat this as a rough trend, not a budget. "
                       "Seasonality is switched off because there is less than two years of data.")
            fc_pack = (metric, actual, fc, used)

    # ================= Variance =================
    with tabs[4]:
        st.header("📉 Variance Analysis")
        mode = st.radio("Compare", ["Month vs month", "Actual vs budget file", "Actual vs model fit"], horizontal=True)
        if mode == "Month vs month":
            if len(m) < 2:
                st.info("Select at least two months.")
            else:
                a, b = st.columns(2)
                opts = list(m.index)
                base = a.selectbox("Base month", opts, index=len(opts) - 2, format_func=mlabel)
                comp = b.selectbox("Compare month", opts, index=len(opts) - 1, format_func=mlabel)
                if base == comp:
                    st.info("Pick two different months.")
                else:
                    vt = month_vs_month(m, base, comp)
                    show_df(fmt_table(vt, int_cols=[mlabel(base), mlabel(comp), "Change"], pct_cols=["Change %"]), hide_index=True)
                    fig = px.bar(vt, x="Line", y="Change", color="Impact", title=f"Change: {mlabel(base)} to {mlabel(comp)}",
                                 color_discrete_map={"Favourable": "#2e9e5b", "Unfavourable": "#d64545", "–": "#999999"})
                    show_fig(fig, "va_mm")
        elif mode == "Actual vs budget file":
            tpl = pd.DataFrame([(d.strftime("%Y-%m"), ln, None) for d in m.index for ln in LINES],
                               columns=["Month", "Line", "Budget"])
            st.download_button("Download budget template (CSV)", tpl.to_csv(index=False).encode(),
                               "budget_template.csv", "text/csv")
            up = st.file_uploader("Upload budget (CSV or Excel with columns Month, Line, Budget)", type=["csv", "xlsx"], key="budget")
            if up is None:
                st.info("Upload a budget to compare it with actuals. Lines: " + ", ".join(LINES) + ".")
            else:
                try:
                    bud, unknown = read_budget(up)
                    if unknown:
                        st.warning("Ignored unknown line names: " + ", ".join(unknown))
                    bv = budget_vs_actual(m, bud)
                    if bv.empty:
                        st.warning("No budget rows match the selected months and line names.")
                    else:
                        summ = bv.groupby("Line", as_index=False)[["Actual", "Budget", "Variance"]].sum()
                        summ["Variance %"] = [pct(v, abs(b_)) for v, b_ in zip(summ["Variance"], summ["Budget"])]
                        summ["Impact"] = [favourability(l, v) for l, v in zip(summ["Line"], summ["Variance"])]
                        summ["Line"] = pd.Categorical(summ["Line"], LINES, ordered=True)
                        summ = summ.sort_values("Line")
                        show_df(fmt_table(summ, int_cols=["Actual", "Budget", "Variance"], pct_cols=["Variance %"]), hide_index=True)
                        pick = st.selectbox("Line detail", [l for l in LINES if l in set(bv["Line"])])
                        det = bv[bv["Line"] == pick].sort_values("Month")
                        fig = go.Figure()
                        fig.add_bar(x=det["Month"], y=det["Budget"], name="Budget")
                        fig.add_bar(x=det["Month"], y=det["Actual"], name="Actual")
                        fig.update_layout(barmode="group", title=f"{pick}: actual vs budget")
                        fig.update_xaxes(tickformat="%b %Y", dtick="M1")
                        show_fig(fig, "va_bud")
                except Exception as exc:                                        # noqa: BLE001
                    st.error(f"Could not read the budget file: {exc}")
        else:
            if fc_pack is None:
                st.info("Set up a forecast on the Forecasts tab first (needs 3+ months).")
            else:
                metric, actual, fc, used = fc_pack
                fit = fc[fc["kind"] == "fit"].set_index("ds")
                vt = pd.DataFrame({"Month": [mlabel(d) for d in actual.index], "Actual": actual.values,
                                   "Model fit": fit["yhat"].reindex(actual.index).values})
                vt["Variance"] = vt["Actual"] - vt["Model fit"]
                vt["Variance %"] = [pct(v, abs(b_)) for v, b_ in zip(vt["Variance"], vt["Model fit"])]
                vt["Outside 80% range"] = ["yes" if (a_ < lo_ or a_ > hi_) else "" for a_, lo_, hi_ in
                                           zip(actual.values, fit["yhat_lower"].reindex(actual.index), fit["yhat_upper"].reindex(actual.index))]
                st.caption(f"{metric}: actual compared with the {used} fit. This is in-sample, so it highlights unusual months; "
                           "it is not a true budget comparison.")
                show_df(fmt_table(vt, int_cols=["Actual", "Model fit", "Variance"], pct_cols=["Variance %"]), hide_index=True)
                show_fig(px.bar(vt, x="Month", y="Variance", color="Outside 80% range", title=f"{metric}: variance from model fit"), "va_fit")

    # ================= Export =================
    with tabs[5]:
        st.header("📤 Export")
        clean = leaf[["Source", "Company", "Branch", "Date", "Category", "Section", "Particulars", "Debit", "Credit", "Amount"]]\
            .sort_values(["Date", "Category", "Particulars"]).reset_index(drop=True)
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ Excel workbook", excel_bytes(R, recon, clean), "pl_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        c2.download_button("⬇️ PDF management report", pdf_bytes(R, fc_pack), "pl_management_report.pdf", "application/pdf")
        c3.download_button("⬇️ Clean data (CSV)", clean.to_csv(index=False).encode(), "pl_clean_data.csv", "text/csv")
        st.caption("Exports follow the current filters. The PDF has the same sections as the Management Report tab and "
                   "adds the forecast if you have set one up on the Forecasts tab.")

    # ================= Reconciliation =================
    with tabs[6]:
        st.header("🔎 Reconciliation")
        st.write("Each file is rebuilt from its lowest-level accounts and compared with the report's own INCOME, "
                 "EXPENSES and Profit rows. This ignores the sidebar filters.")
        num = [c_ for c_ in recon.columns if c_ not in ("Source", "Status")]
        show_df(fmt_table(recon, dec_cols=num), hide_index=True)
        for x, parts in overlaps:
            st.warning(f"**{x}** equals the sum of {', '.join(parts)} (sales and expenses, every month). "
                       "Do not add it to those files.")
        other = df_all[df_all["Category"] == "Other"]
        if not other.empty:
            st.error("Accounts in sections the app does not recognise (excluded from the P&L): "
                     + ", ".join(sorted(other["Section"].unique())))
        src = st.selectbox("Month-by-month tie-out for", sources)
        mm = check_monthly(df_all[df_all["Source"] == src], controls[src])
        if not mm.empty:
            mm2 = mm.copy()
            mm2.index = [mlabel(d) for d in mm2.index]
            show_df(fmt_table(mm2, dec_cols=list(mm2.columns)))
        with st.expander("Credit entries on expense accounts (reversals)"):
            rev = df_all[df_all["IsLeaf"] & df_all["Category"].str.startswith(("Expenses", "COGS")) & (df_all["Credit"] > 0)]
            if rev.empty:
                st.write("None.")
            else:
                show_df(rev[["Source", "Date", "Particulars", "Debit", "Credit"]], hide_index=True)


if __name__ == "__main__":
    main()