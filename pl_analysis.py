"""Analysis behind the management report.  Pure pandas - no Streamlit, no plotting.

build_report(leaf, ...) returns one dict `R` that the app tab, the PDF and the Excel
export all read from, so the three can never disagree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CURRENCY = "KSh"
CAT_LABEL = {
    "Income": "Revenue",
    "COGS": "COGS",
    "Expenses - Administration": "Administration",
    "Expenses - Staff Costs": "Staff Costs",
    "Expenses - Operational Costs": "Operational Costs",
    "Expenses - Finance Costs": "Finance Costs",
}
EXPENSE_LINES = ["Administration", "Staff Costs", "Operational Costs", "Finance Costs"]
LINES = ["Revenue", "COGS", "Gross Profit"] + EXPENSE_LINES + ["Total Expenses", "Net Profit"]
COST_LINES = {"COGS", "Total Expenses", *EXPENSE_LINES}
BUCKETS = ["Fixed", "Semi-variable", "Variable"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def mlabel(d):
    return pd.Timestamp(d).strftime("%b %Y")


def pct(a, b):
    return np.nan if (b is None or pd.isna(b) or b == 0) else a / b * 100


def pct_s(a, b):
    """Element-wise a / |b| * 100, NaN where b == 0."""
    return (a / b.abs()).where(b != 0) * 100


def fpct(x, signed=True):
    """Format a percentage; 'n/m' when the base was too small for it to mean anything."""
    if x is None or pd.isna(x):
        return "n/m"
    if abs(x) > 500:
        return "n/m (small base)"
    return f"{x:+.1f}%" if signed else f"{x:.1f}%"


def compact(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    a = abs(x)
    s = f"{x/1e9:,.2f}B" if a >= 1e9 else f"{x/1e6:,.2f}M" if a >= 1e6 else f"{x:,.0f}"
    return f"{CURRENCY} {s}"


def favourability(line, change):
    if pd.isna(change) or abs(change) < 0.5:
        return "–"
    good = change > 0
    if line in COST_LINES:
        good = not good
    return "Favourable" if good else "Unfavourable"


def slope_pct(values):
    """Average monthly trend as % of the mean (least-squares slope / mean)."""
    y = np.asarray(values, float)
    if len(y) < 3 or np.mean(np.abs(y)) == 0:
        return np.nan
    return float(np.polyfit(np.arange(len(y)), y, 1)[0] / abs(y.mean()) * 100)


def monthly_pnl(leaf):
    """One row per month, one column per P&L line (leaf rows only)."""
    if leaf.empty:
        return pd.DataFrame(columns=LINES)
    p = leaf.pivot_table(index="Date", columns="Category", values="Amount", aggfunc="sum", fill_value=0.0)
    out = pd.DataFrame(index=p.index)
    for cat, lab in CAT_LABEL.items():
        out[lab] = p[cat] if cat in p else 0.0
    out["Gross Profit"] = out["Revenue"] - out["COGS"]
    out["Total Expenses"] = out[EXPENSE_LINES].sum(axis=1)
    out["Net Profit"] = out["Gross Profit"] - out["Total Expenses"]
    out = out.reindex(pd.date_range(out.index.min(), out.index.max(), freq="MS"), fill_value=0.0)
    out.index.name = "Month"
    return out[LINES]


def statement(m):
    s = m.T.copy()
    s.columns = [mlabel(c) for c in s.columns]
    s["Total"] = s.sum(axis=1)
    s.index.name = "Line"
    return s


def add_entity(df):
    """'Entity' = branch name, prefixed with the company when several companies are in view."""
    df = df.copy()
    if df["Company"].nunique() > 1:
        df["Entity"] = df["Company"] + " · " + df["Branch"]
    else:
        df["Entity"] = df["Branch"]
    return df


# ---------------------------------------------------------------------------
# cost behaviour (fixed / semi-variable / variable)
# ---------------------------------------------------------------------------
_COST_RULES = [
    ("Variable", ["FUEL", "HIRE", "TRANSPORT", "LOADING", "GAS SEAL", "GAS LEAK", "LOSS", "PILFER", "LUBE",
                  "MILLEAGE", "MILEAGE", "PARKING", "TRAVEL", "INCENTIVE", "MARKETING", "BANK CHARGES",
                  "MPESA", "BAD DEBT", "WEAR & TEAR", "COMMISSION", "PACKAGING", "OVERTIME"]),
    ("Semi-variable", ["AGENCY FEES", "REPAIR", "MAINTEN", "ELECTRICITY", "WATER", "CLEANING", "PRINTING",
                       "STATIONERY", "UNIFORM", "WELFARE", "GARBAGE", "SEPTIC", "MISC", "INSPECTION",
                       "CAR WASH", "MEETINGS", "TRAINING"]),
    ("Fixed", ["SALAR", "WAGES", "NSSF", "HOUSING LEVY", "NITA", "MEDICAL", "RENT", "INSURANCE", "LICEN",
               "AUDIT", "LEGAL", "SECURITY", "DEPRECIATION", "INTEREST", "GUARANTEE", "INTERNET", "AIRTIME",
               "TELEPHONE", "PHONES", "ADMIN", "CONSTRUCTION", "RELOCATION"]),
]


def classify_cost(name):
    """Default behaviour of an operating-expense account, judged from its name. Editable in the app."""
    n = str(name).upper()
    for bucket, kws in _COST_RULES:
        if any(k in n for k in kws):
            return bucket
    return "Semi-variable"


# ---------------------------------------------------------------------------
# generic "line" table: totals, share, latest vs previous month, 3M vs prior 3M, trend
# ---------------------------------------------------------------------------
def line_table(df, key, months):
    n = len(months)
    piv = (df.pivot_table(index=key, columns="Date", values="Amount", aggfunc="sum", fill_value=0.0)
           .reindex(columns=months, fill_value=0.0))
    out = pd.DataFrame(index=piv.index)
    out["Total"] = piv.sum(axis=1)
    tot = out["Total"].sum()
    out["Share %"] = out["Total"] / tot * 100 if tot else np.nan
    if n >= 2:
        out["Last month"], out["Prev month"] = piv.iloc[:, -1], piv.iloc[:, -2]
        out["MoM %"] = pct_s(out["Last month"] - out["Prev month"], out["Prev month"])
    if n >= 6:
        l3, p3 = piv.iloc[:, -3:].sum(axis=1), piv.iloc[:, -6:-3].sum(axis=1)
        out["3M vs prior 3M %"] = pct_s(l3 - p3, p3)
    out["Trend %/mo"] = [slope_pct(r) for r in piv.to_numpy()]
    return out.sort_values("Total", ascending=False)


def period_compare(m, n):
    """Latest n months vs the n months before, for every P&L line."""
    if len(m) < 2 * n:
        return None
    cur, prev = m.iloc[-n:].sum(), m.iloc[-2 * n:-n].sum()
    lab = (lambda a, b: mlabel(a) if n == 1 else f"{mlabel(a)} - {mlabel(b)}")
    rows = []
    for ln in LINES:
        ch = cur[ln] - prev[ln]
        rows.append({"Line": ln, "Current": cur[ln], "Previous": prev[ln], "Change": ch,
                     "Change %": pct(ch, abs(prev[ln])), "Impact": favourability(ln, ch)})
    return {"table": pd.DataFrame(rows),
            "current": lab(m.index[-n], m.index[-1]),
            "previous": lab(m.index[-2 * n], m.index[-n - 1])}


# ---------------------------------------------------------------------------
# branches
# ---------------------------------------------------------------------------
def branch_table(leaf, months):
    rows = []
    for ent, g in leaf.groupby("Entity"):
        bm = monthly_pnl(g).reindex(months, fill_value=0.0)
        t = bm.sum()
        r = {"Branch": ent, "Revenue": t["Revenue"], "COGS": t["COGS"], "Gross Profit": t["Gross Profit"],
             "Total Expenses": t["Total Expenses"], "Net Profit": t["Net Profit"],
             "Gross margin %": pct(t["Gross Profit"], t["Revenue"]),
             "Net margin %": pct(t["Net Profit"], t["Revenue"]),
             "Expenses % rev": pct(t["Total Expenses"], t["Revenue"]),
             "Trend %/mo": slope_pct(bm["Revenue"].to_numpy())}
        if len(bm) >= 2:
            lr, pr = bm["Revenue"].iloc[-1], bm["Revenue"].iloc[-2]
            r.update({"Last month rev": lr, "Prev month rev": pr, "Rev MoM %": pct(lr - pr, abs(pr)),
                      "Last month NP": bm["Net Profit"].iloc[-1], "Prev month NP": bm["Net Profit"].iloc[-2]})
        if len(bm) >= 6:
            l3, p3 = bm["Revenue"].iloc[-3:].sum(), bm["Revenue"].iloc[-6:-3].sum()
            r["Rev 3M vs prior %"] = pct(l3 - p3, abs(p3))
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("Revenue", ascending=False).reset_index(drop=True)
    tot_rev = df["Revenue"].sum()
    df.insert(2, "Revenue share %", df["Revenue"] / tot_rev * 100 if tot_rev else np.nan)
    gm_all = pct(df["Gross Profit"].sum(), tot_rev)
    df["GM vs group (pp)"] = df["Gross margin %"] - gm_all
    df["Net profit rank"] = df["Net Profit"].rank(ascending=False, method="min").astype(int)
    return df


def branch_monthly(leaf, months, line="Revenue"):
    out = {}
    for ent, g in leaf.groupby("Entity"):
        out[ent] = monthly_pnl(g).reindex(months, fill_value=0.0)[line]
    return pd.DataFrame(out).T.reindex(columns=months)


# ---------------------------------------------------------------------------
# income / expenses / products
# ---------------------------------------------------------------------------
def _product_key(name):
    n = str(name).upper()
    n = n.split("-", 1)[1] if "-" in n else n
    return n.replace("ACCESORIES", "ACCESSORIES").strip()


def product_margins(leaf):
    s = leaf[(leaf["Category"] == "Income") & leaf["Particulars"].str.upper().str.startswith("SALES-")].copy()
    c = leaf[leaf["Category"] == "COGS"].copy()
    s["Key"], c["Key"] = s["Particulars"].map(_product_key), c["Particulars"].map(_product_key)
    sales, cogs = s.groupby("Key")["Amount"].sum(), c.groupby("Key")["Amount"].sum()
    keys = sales.index.intersection(cogs.index)
    if len(keys) == 0:
        return pd.DataFrame(columns=["Product", "Sales", "COGS", "Gross profit", "Gross margin %", "Share of sales %"])
    out = pd.DataFrame({"Product": keys, "Sales": sales[keys].to_numpy(), "COGS": cogs[keys].to_numpy()})
    out["Gross profit"] = out["Sales"] - out["COGS"]
    out["Gross margin %"] = (out["Gross profit"] / out["Sales"].where(out["Sales"] != 0)) * 100
    out["Share of sales %"] = out["Sales"] / leaf.loc[leaf["Category"] == "Income", "Amount"].sum() * 100
    return out.sort_values("Sales", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def build_report(leaf, partial_labels=(), recon_ok=True, semi_fixed_share=0.5, overrides=None):
    overrides = overrides or {}
    leaf = add_entity(leaf)
    m = monthly_pnl(leaf)
    months = m.index
    n = len(m)
    tot = m.sum()
    R = {"m": m, "statement": statement(m), "n_months": n, "recon_ok": recon_ok,
         "partial": list(partial_labels), "semi_fixed_share": semi_fixed_share}

    # ---- scope ----------------------------------------------------------------
    scope = {}
    for co, g in leaf.groupby("Company"):
        scope[co] = sorted(g["Branch"].unique())
    R["scope"] = scope
    R["period"] = f"{mlabel(months.min())} - {mlabel(months.max())}"
    R["n_entities"] = leaf["Entity"].nunique()

    # ---- headline ---------------------------------------------------------------
    R["kpis"] = {"Revenue": tot["Revenue"], "COGS": tot["COGS"], "Gross Profit": tot["Gross Profit"],
                 "Total Expenses": tot["Total Expenses"], "Net Profit": tot["Net Profit"],
                 "Gross margin %": pct(tot["Gross Profit"], tot["Revenue"]),
                 "Net margin %": pct(tot["Net Profit"], tot["Revenue"]),
                 "Expenses % revenue": pct(tot["Total Expenses"], tot["Revenue"])}

    # ---- vs previous periods ---------------------------------------------------------
    R["cmp_month"] = period_compare(m, 1)
    R["cmp_quarter"] = period_compare(m, 3)
    cmp_notes = []
    for key, name in (("cmp_month", "month"), ("cmp_quarter", "quarter")):
        c = R[key]
        if not c:
            continue
        t = c["table"].set_index("Line")
        cmp_notes.append(
            f"{c['current']} vs {c['previous']}: revenue {fpct(t.loc['Revenue', 'Change %'])} "
            f"({compact(t.loc['Revenue', 'Change'])}), gross profit {fpct(t.loc['Gross Profit', 'Change %'])}, "
            f"total expenses {fpct(t.loc['Total Expenses', 'Change %'])}, net profit {fpct(t.loc['Net Profit', 'Change %'])} "
            f"({compact(t.loc['Net Profit', 'Change'])}).")
    if n < 6:
        cmp_notes.append("Fewer than six months are in view, so no quarter-on-quarter comparison is shown.")
    R["cmp_notes"] = cmp_notes

    # ---- branches ----------------------------------------------------------------------
    bt = branch_table(leaf, months)
    R["branch"] = bt
    R["branch_rev_monthly"] = branch_monthly(leaf, months, "Revenue")
    R["branch_np_monthly"] = branch_monthly(leaf, months, "Net Profit")
    bn = []
    if len(bt) >= 1:
        top = bt.iloc[0]
        bn.append(f"{top['Branch']} is the largest branch with {top['Revenue share %']:.0f}% of revenue "
                  f"({compact(top['Revenue'])}).")
        if len(bt) >= 3:
            bn.append(f"The top three branches together contribute {bt['Revenue share %'].head(3).sum():.0f}% of revenue.")
    if len(bt) >= 2:
        valid = bt[bt["Revenue"] > 0]
        if len(valid) >= 2:
            hi, lo = valid.loc[valid["Gross margin %"].idxmax()], valid.loc[valid["Gross margin %"].idxmin()]
            bn.append(f"Gross margin ranges from {lo['Gross margin %']:.1f}% ({lo['Branch']}) to {hi['Gross margin %']:.1f}% "
                      f"({hi['Branch']}); the group average is {R['kpis']['Gross margin %']:.1f}%.")
        best, worst = bt.loc[bt["Net Profit"].idxmax()], bt.loc[bt["Net Profit"].idxmin()]
        bn.append(f"Highest net profit: {best['Branch']} ({compact(best['Net Profit'])}); "
                  f"lowest: {worst['Branch']} ({compact(worst['Net Profit'])}).")
        loss = bt[bt["Net Profit"] < 0]["Branch"].tolist()
        if loss:
            bn.append("Branches with a net loss over the period: " + ", ".join(loss) + ".")
        if "Rev MoM %" in bt:
            dec = bt[(bt["Rev MoM %"] < -10) & (bt["Prev month rev"] > 0)].sort_values("Rev MoM %")
            if len(dec):
                bn.append("Revenue fell by more than 10% last month in: " + "; ".join(
                    f"{r['Branch']} ({r['Rev MoM %']:.0f}%)" for _, r in dec.head(5).iterrows()) + ".")
            inc = bt[(bt["Rev MoM %"] > 10) & (bt["Prev month rev"] > 0)].sort_values("Rev MoM %", ascending=False)
            if len(inc):
                bn.append("Revenue grew by more than 10% last month in: " + "; ".join(
                    f"{r['Branch']} (+{r['Rev MoM %']:.0f}%)" for _, r in inc.head(5).iterrows()) + ".")
        bn.append("Branch net profit reflects only the costs posted to that branch. Shared costs (for example head-office "
                  "salaries or loan interest) may sit in one branch, so compare gross margin first and net profit second.")
    R["branch_notes"] = bn

    # ---- income lines --------------------------------------------------------------------
    inc = leaf[leaf["Category"] == "Income"]
    R["income_groups"] = line_table(inc, "Parent", months).reset_index().rename(columns={"Parent": "Income group"})
    ia = line_table(inc, ["Parent", "Particulars"], months).reset_index().rename(
        columns={"Parent": "Income group", "Particulars": "Account"})
    R["income_accounts"] = ia
    R["income_monthly_by_group"] = (inc.pivot_table(index="Date", columns="Parent", values="Amount", aggfunc="sum",
                                                    fill_value=0.0).reindex(months, fill_value=0.0))
    R["product_margin"] = product_margins(leaf)
    inn = []
    ig = R["income_groups"]
    if len(ig):
        inn.append("Revenue mix: " + "; ".join(f"{r['Income group'].title()} {r['Share %']:.1f}%" for _, r in ig.iterrows()) + ".")
    if len(ia):
        t = ia.iloc[0]
        inn.append(f"The largest income line is {t['Account']} at {t['Share %']:.1f}% of revenue ({compact(t['Total'])}).")
        big = ia[ia["Share %"] >= 2].dropna(subset=["Trend %/mo"])
        if len(big):
            up, dn = big.loc[big["Trend %/mo"].idxmax()], big.loc[big["Trend %/mo"].idxmin()]
            inn.append(f"Among lines with at least 2% of revenue, {up['Account']} has the strongest trend "
                       f"({up['Trend %/mo']:+.1f}% a month) and {dn['Account']} the weakest ({dn['Trend %/mo']:+.1f}% a month).")
        if "MoM %" in ia:
            mv = ia[(ia["Share %"] >= 2) & (ia["Prev month"] > 0)]
            if len(mv):
                a, b = mv.loc[mv["MoM %"].idxmax()], mv.loc[mv["MoM %"].idxmin()]
                inn.append(f"Last month vs the month before, among lines with at least 2% of revenue: strongest {a['Account']} "
                           f"({a['MoM %']:+.1f}%), weakest {b['Account']} ({b['MoM %']:+.1f}%).")
    pm = R["product_margin"]
    if len(pm) >= 1:
        inn.append("Gross margin by product: " + "; ".join(f"{r['Product'].title()} {r['Gross margin %']:.1f}%"
                                                         for _, r in pm.iterrows()) + ".")
    R["income_notes"] = inn

    # ---- expenditure -----------------------------------------------------------------------
    exp = leaf[leaf["Category"].str.startswith("Expenses")].copy()
    exp["Group"] = exp["Category"].map(CAT_LABEL)
    eg = line_table(exp, "Group", months).reset_index()
    eg["% of revenue"] = eg["Total"] / tot["Revenue"] * 100 if tot["Revenue"] else np.nan
    R["exp_groups"] = eg
    ea = line_table(exp, ["Group", "Particulars"], months).reset_index().rename(columns={"Particulars": "Account"})
    ea["% of revenue"] = ea["Total"] / tot["Revenue"] * 100 if tot["Revenue"] else np.nan
    R["exp_accounts"] = ea
    # movers: change in the latest 3 months vs the 3 before (or latest month vs previous if short history)
    piv = (exp.pivot_table(index=["Group", "Particulars"], columns="Date", values="Amount", aggfunc="sum", fill_value=0.0)
           .reindex(columns=months, fill_value=0.0))
    if n >= 6:
        cur, prv, basis = piv.iloc[:, -3:].sum(axis=1), piv.iloc[:, -6:-3].sum(axis=1), "last 3 months vs prior 3 months"
    elif n >= 2:
        cur, prv, basis = piv.iloc[:, -1], piv.iloc[:, -2], "last month vs previous month"
    else:
        cur = prv = None
        basis = ""
    if cur is not None:
        mv = pd.DataFrame({"Current": cur, "Previous": prv})
        mv["Change"] = mv["Current"] - mv["Previous"]
        mv["Change %"] = pct_s(mv["Change"], mv["Previous"])
        R["exp_movers"] = mv.reset_index().rename(columns={"Particulars": "Account"}).sort_values("Change", ascending=False)
    else:
        R["exp_movers"] = pd.DataFrame()
    R["exp_movers_basis"] = basis
    R["exp_monthly_by_group"] = (exp.pivot_table(index="Date", columns="Group", values="Amount", aggfunc="sum", fill_value=0.0)
                                 .reindex(months, fill_value=0.0).reindex(columns=EXPENSE_LINES, fill_value=0.0))
    en = []
    if tot["Total Expenses"] > 0:
        en.append(f"Operating expenses total {compact(tot['Total Expenses'])}, which is {R['kpis']['Expenses % revenue']:.1f}% of revenue "
                  f"({pct(tot['Total Expenses'], tot['Gross Profit']):.0f}% of gross profit).")
        g = eg.iloc[0]
        en.append(f"{g['Group']} is the largest group at {g['Share %']:.0f}% of operating expenses.")
        top3 = ea.head(3)
        en.append("Largest accounts: " + "; ".join(f"{r['Account']} ({compact(r['Total'])}, {r['Share %']:.0f}%)"
                                                  for _, r in top3.iterrows()) + ".")
        if len(R["exp_movers"]):
            mvs = R["exp_movers"]
            up, dn = mvs.iloc[0], mvs.iloc[-1]
            if up["Change"] > 0:
                en.append(f"Biggest increase ({basis}): {up['Account']} {compact(up['Change'])} higher.")
            if dn["Change"] < 0:
                en.append(f"Biggest decrease ({basis}): {dn['Account']} {compact(abs(dn['Change']))} lower.")
        if n >= 4:
            base = m["Total Expenses"].iloc[:-1].mean()
            if base > 0 and m["Total Expenses"].iloc[-1] < 0.6 * base:
                en.append(f"CHECK: expenses in {mlabel(months[-1])} are only {m['Total Expenses'].iloc[-1] / base * 100:.0f}% of the "
                          "earlier monthly average. Payroll or other accruals may not be posted yet, so that month's profit "
                          "may be overstated.")
    R["exp_notes"] = en

    # ---- fixed vs variable ------------------------------------------------------------------------
    accts = sorted(exp["Particulars"].unique())
    bucket_of = {a: overrides.get(a, classify_cost(a)) for a in accts}
    exp["Bucket"] = exp["Particulars"].map(bucket_of)
    cb_month = (exp.pivot_table(index="Date", columns="Bucket", values="Amount", aggfunc="sum", fill_value=0.0)
                .reindex(months, fill_value=0.0).reindex(columns=BUCKETS, fill_value=0.0))
    cb_month["COGS (variable)"] = m["COGS"]
    R["cost_monthly"] = cb_month
    tb = []
    tot_cost = cb_month.sum().sum()
    for b in BUCKETS + ["COGS (variable)"]:
        s = cb_month[b]
        r = {"Cost type": b, "Total": s.sum(), "% of all costs": pct(s.sum(), tot_cost),
             "% of revenue": pct(s.sum(), tot["Revenue"]), "Monthly average": s.mean() if n else np.nan}
        if n >= 2:
            r["Last month"], r["Prev month"] = s.iloc[-1], s.iloc[-2]
            r["MoM %"] = pct(s.iloc[-1] - s.iloc[-2], abs(s.iloc[-2]))
        tb.append(r)
    R["cost_buckets"] = pd.DataFrame(tb)
    ca = ea[["Group", "Account", "Total"]].copy()
    ca["Cost type"] = ca["Account"].map(bucket_of)
    R["cost_accounts"] = ca.sort_values(["Cost type", "Total"], ascending=[True, False]).reset_index(drop=True)
    R["classification"] = pd.DataFrame({"Account": accts, "Cost type": [bucket_of[a] for a in accts]})

    fx, sv, vr = cb_month["Fixed"].sum(), cb_month["Semi-variable"].sum(), cb_month["Variable"].sum()
    share = semi_fixed_share
    fixed_total = fx + share * sv
    var_total = tot["COGS"] + vr + (1 - share) * sv
    cm_ratio = (tot["Revenue"] - var_total) / tot["Revenue"] if tot["Revenue"] else np.nan
    be = {"fixed_monthly": fixed_total / n if n else np.nan, "contribution_margin_pct": cm_ratio * 100}
    if cm_ratio and cm_ratio > 0 and n:
        be["break_even_monthly"] = be["fixed_monthly"] / cm_ratio
        avg_rev = tot["Revenue"] / n
        be["avg_revenue_monthly"] = avg_rev
        be["margin_of_safety_pct"] = (avg_rev - be["break_even_monthly"]) / avg_rev * 100
    R["break_even"] = be
    cn = []
    if tot_cost > 0:
        cn.append(f"Of all costs (COGS plus operating expenses, {compact(tot_cost)}), "
                  f"{pct(cb_month['COGS (variable)'].sum() + vr, tot_cost):.0f}% moves with volume (variable), "
                  f"{pct(fx, tot_cost):.1f}% is fixed and {pct(sv, tot_cost):.1f}% is semi-variable.")
    opex = fx + sv + vr
    if opex > 0:
        cn.append(f"Within operating expenses alone: fixed {pct(fx, opex):.0f}%, semi-variable {pct(sv, opex):.0f}%, "
                  f"variable {pct(vr, opex):.0f}%.")
    if "break_even_monthly" in be:
        cn.append(f"Contribution margin is {be['contribution_margin_pct']:.1f}% of revenue. With fixed costs of about "
                  f"{compact(be['fixed_monthly'])} a month (semi-variable costs treated {share*100:.0f}% fixed), break-even revenue is "
                  f"about {compact(be['break_even_monthly'])} a month against an actual average of {compact(be['avg_revenue_monthly'])}. "
                  f"Margin of safety: {be['margin_of_safety_pct']:.0f}%.")
    cn.append("Fixed / variable labels come from account names and can be changed in the app. Costs that are not posted to the "
              "selected branches (for example head-office payroll) are not in these figures.")
    R["cost_notes"] = cn

    # ---- trends ---------------------------------------------------------------------------------------
    ts = {}
    rev = m["Revenue"]
    if n >= 2:
        mom = rev.pct_change().replace([np.inf, -np.inf], np.nan).dropna() * 100
        ts["avg_mom_growth"] = float(mom.mean()) if len(mom) else np.nan
    ts["rev_trend"] = slope_pct(rev.to_numpy())
    ts["np_trend"] = slope_pct(m["Net Profit"].to_numpy())
    ts["rev_cv"] = float(rev.std() / rev.mean() * 100) if n >= 3 and rev.mean() else np.nan
    R["trend_stats"] = ts
    gm_series = (m["Gross Profit"] / rev.where(rev != 0)) * 100
    nm_series = (m["Net Profit"] / rev.where(rev != 0)) * 100
    ex_series = (m["Total Expenses"] / rev.where(rev != 0)) * 100
    R["margin_series"] = pd.DataFrame({"Gross margin %": gm_series, "Net margin %": nm_series, "Expenses % revenue": ex_series})
    tn = []
    if n >= 3:
        tn.append(f"Revenue trend: {ts['rev_trend']:+.1f}% a month on average (least-squares line); net profit trend "
                  f"{ts['np_trend']:+.1f}% a month.")
        if not np.isnan(ts["rev_cv"]):
            tn.append(f"Month-to-month volatility of revenue is {ts['rev_cv']:.0f}% of the average "
                      f"({'high' if ts['rev_cv'] > 20 else 'moderate' if ts['rev_cv'] > 10 else 'low'}).")
        tn.append(f"Highest revenue month: {mlabel(rev.idxmax())} ({compact(rev.max())}); lowest: {mlabel(rev.idxmin())} ({compact(rev.min())}).")
        tn.append(f"Gross margin has moved between {gm_series.min():.1f}% and {gm_series.max():.1f}%; "
                  f"net margin between {nm_series.min():.1f}% and {nm_series.max():.1f}%.")
        half = n // 2
        if half >= 2:
            e1, e2 = ex_series.iloc[:half].mean(), ex_series.iloc[half:].mean()
            tn.append(f"Operating expenses averaged {e1:.1f}% of revenue in the first {half} months and {e2:.1f}% in the remaining {n - half}.")
    else:
        tn.append("Select at least three months to see trend statistics.")
    R["trend_notes"] = tn

    # ---- executive summary ------------------------------------------------------------------------------
    k = R["kpis"]
    es = [f"Period: {R['period']} ({n} months) across {R['n_entities']} branch(es)."
          + (f" Incomplete month excluded: {', '.join(R['partial'])}." if R["partial"] else ""),
          f"Revenue {compact(k['Revenue'])}, gross profit {compact(k['Gross Profit'])} ({k['Gross margin %']:.1f}% margin), "
          f"operating expenses {compact(k['Total Expenses'])} ({k['Expenses % revenue']:.1f}% of revenue), "
          f"net profit {compact(k['Net Profit'])} ({k['Net margin %']:.1f}% margin)."]
    if R["cmp_month"] is not None:
        es.append(cmp_notes[0])
    if bn:
        es.append(bn[0])
    gm_note = next((x for x in bn if x.startswith("Gross margin ranges")), None)
    if gm_note:
        es.append(gm_note)
    if inn:
        es.append(inn[0])
    if en:
        es.append(en[0])
    if tot_cost > 0:
        es.append(cn[0])
    if n >= 3:
        es.append(tn[0])
    flags = [x for x in en if x.startswith("CHECK")]
    es += flags
    es.append("All selected files tie out to their source reports." if recon_ok
              else "WARNING: at least one selected file does not tie out to its source report - see Reconciliation.")
    R["summary"] = es
    return R