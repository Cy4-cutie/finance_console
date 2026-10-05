"""PDF and Excel builders for the management report (matplotlib / openpyxl, no Streamlit)."""
from __future__ import annotations

import io
import math
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from pl_analysis import BUCKETS, CURRENCY, EXPENSE_LINES, compact, fpct, mlabel

PAGE = (11.69, 8.27)
NAVY, GREY = "#1f3a5f", "#555555"
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def _page(title, subtitle=""):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.05, 0.972, title, fontsize=17, weight="bold", color=NAVY, va="top")
    if subtitle:
        fig.text(0.05, 0.928, subtitle, fontsize=8.5, color=GREY, va="top")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.908, 0.908], color=NAVY, lw=1.2, transform=fig.transFigure))
    return fig


def _bullets(fig, items, y, x=0.05, wrap=148, size=8.8, step=0.0215, gap=0.007, heading=None):
    """Draw bullet points downwards from y; returns the new y."""
    if heading:
        fig.text(x, y, heading, fontsize=11, weight="bold", color=NAVY, va="top")
        y -= 0.03
    for it in items:
        lines = textwrap.wrap(str(it), wrap) or [""]
        for j, ln in enumerate(lines):
            fig.text(x + (0.012 if j == 0 else 0.026), y, ("•  " + ln) if j == 0 else ln, fontsize=size, va="top")
            y -= step
        y -= gap
    return y


def _fmt(df, ints=(), pcts=(), decs=(), signed=()):
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        col = df[c]
        if c in ints:
            out[c] = col.map(lambda v: "–" if pd.isna(v) else f"{v:,.0f}  ")
        elif c in pcts:
            out[c] = col.map(lambda v: "–" if pd.isna(v) else (("n/m" if abs(v) > 500 else (f"{v:+.1f}%" if c in signed else f"{v:.1f}%")) + "  "))
        elif c in decs:
            out[c] = col.map(lambda v: "–" if pd.isna(v) else f"{v:,.2f}")
        else:
            out[c] = col.astype(str)
    return out


def _table(fig, df, top, left=0.05, width=0.90, row_h=0.029, size=7.2, col_w=None, first_left=True, title=None):
    """Draw a string DataFrame as a table; returns the y below it."""
    if title:
        fig.text(left, top, title, fontsize=10.5, weight="bold", color=NAVY, va="top")
        top -= 0.028
    n = len(df) + 1
    h = n * row_h
    ax = fig.add_axes([left, top - h, width, h])
    ax.axis("off")
    tbl = ax.table(cellText=df.values.tolist(), colLabels=list(df.columns), loc="center", cellLoc="right",
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(size)
    ncol = len(df.columns)
    if col_w is None:
        lens = [max([len(str(c))] + [len(str(v)) for v in df.iloc[:, i]]) for i, c in enumerate(df.columns)]
        lens = [min(max(l, 6), 46) for l in lens]
        col_w = [l / sum(lens) for l in lens]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_edgecolor("#bbbbbb")
        cell.set_width(col_w[c])
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_ha("center")
        else:
            cell.set_facecolor("#f4f6fa" if r % 2 == 0 else "white")
            if c == 0 and first_left:
                cell.get_text().set_ha("left")
                cell._loc = "left"
    return top - h - 0.015


def _bar_h(ax, labels, values, title, color=None, fmt=lambda v: f"{v:,.1f}", size=7):
    y = np.arange(len(labels))
    cols = color or [PALETTE[0]] * len(labels)
    ax.barh(y, values, color=cols)
    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.shorten(str(l), 28, placeholder="…") for l in labels], fontsize=size)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9.5, weight="bold", color=NAVY, loc="left")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    span = (max(values) - min(0, min(values))) or 1
    for yi, v in zip(y, values):
        ax.text(v + span * 0.01 if v >= 0 else v - span * 0.01, yi, fmt(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=6.5)
    ax.margins(x=0.18)


def _style(ax, title):
    ax.set_title(title, fontsize=9.5, weight="bold", color=NAVY, loc="left")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)


def _month_axis(ax, months):
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([mlabel(d) for d in months], rotation=45, ha="right", fontsize=7)


def _footer(fig, n):
    fig.text(0.95, 0.02, f"Page {n}", fontsize=7, color=GREY, ha="right")
    fig.text(0.05, 0.02, f"All amounts in {CURRENCY}. Figures built from lowest-level accounts only (no subtotal double counting).",
             fontsize=7, color=GREY)


# ---------------------------------------------------------------------------
# the PDF
# ---------------------------------------------------------------------------
def build_pdf(R, fc_pack=None):
    bio = io.BytesIO()
    m = R["m"]
    months = m.index
    k = R["kpis"]
    pg = [0]

    def save(pdf, fig):
        pg[0] += 1
        _footer(fig, pg[0])
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(bio) as pdf:
        # ---------------- 1. summary ----------------
        fig = _page("Management Report: Profit & Loss", f"{R['period']}  |  {R['n_months']} months  |  {R['n_entities']} branch(es)")
        # scope as vertical bullets, in as many columns as needed
        entries = []
        for co, brs in R["scope"].items():
            entries.append(("co", co))
            entries += [("br", b) for b in brs]
        per_col = 14
        ncols = max(1, math.ceil(len(entries) / per_col))
        fig.text(0.05, 0.885, "Companies and branches covered", fontsize=11, weight="bold", color=NAVY, va="top")
        colw = 0.9 / max(ncols, 3)
        for i, (kind, name) in enumerate(entries):
            c, r = divmod(i, per_col)
            txt = name if kind == "co" else "•  " + name
            fig.text(0.05 + c * colw, 0.855 - r * 0.0185, textwrap.shorten(txt, 44, placeholder="…"),
                     fontsize=8 if kind == "br" else 8.5, weight="bold" if kind == "co" else "normal", va="top")
        y = 0.855 - min(len(entries), per_col) * 0.0185 - 0.02
        # KPI strip
        boxes = [("Revenue", compact(k["Revenue"]), ""), ("Gross profit", compact(k["Gross Profit"]), f"{k['Gross margin %']:.1f}% margin"),
                 ("Operating expenses", compact(k["Total Expenses"]), f"{k['Expenses % revenue']:.1f}% of revenue"),
                 ("Net profit", compact(k["Net Profit"]), f"{k['Net margin %']:.1f}% margin")]
        for i, (lab, val, sub) in enumerate(boxes):
            x0 = 0.05 + i * 0.228
            fig.add_artist(plt.Rectangle((x0, y - 0.085), 0.215, 0.085, transform=fig.transFigure, fc="#f4f6fa", ec="#cfd6e4"))
            fig.text(x0 + 0.01, y - 0.012, lab, fontsize=8, color=GREY, va="top")
            fig.text(x0 + 0.01, y - 0.033, val, fontsize=14, weight="bold", va="top")
            fig.text(x0 + 0.01, y - 0.065, sub, fontsize=7.5, color=GREY, va="top")
        y -= 0.115
        _bullets(fig, R["summary"], y, heading="Executive summary")
        save(pdf, fig)

        # ---------------- 2. vs previous periods ----------------
        fig = _page("Performance against previous periods")
        y = _bullets(fig, R["cmp_notes"], 0.885)
        for key, ttl in (("cmp_month", "Latest month vs previous month"), ("cmp_quarter", "Latest 3 months vs previous 3 months")):
            c = R[key]
            if not c:
                continue
            t = c["table"].rename(columns={"Current": c["current"], "Previous": c["previous"]})
            y = _table(fig, _fmt(t, ints=[c["current"], c["previous"], "Change"], pcts=["Change %"], signed=["Change %"]),
                       y, title=f"{ttl}  ({c['current']} vs {c['previous']})", row_h=0.032, width=0.9, size=7.8)
        save(pdf, fig)

        # ---------------- 3. branch table (paginated) ----------------
        bt = R["branch"]
        cols = ["Branch", "Revenue", "Revenue share %", "Gross Profit", "Gross margin %", "GM vs group (pp)",
                "Total Expenses", "Net Profit", "Net margin %", "Net profit rank"]
        for extra in ("Rev MoM %", "Rev 3M vs prior %"):
            if extra in bt:
                cols.append(extra)
        per = 17
        pages = max(1, math.ceil(len(bt) / per))
        for p in range(pages):
            fig = _page("Branch performance and comparison" + (f" ({p + 1}/{pages})" if pages > 1 else ""))
            y = 0.885
            if p == 0:
                y = _bullets(fig, R["branch_notes"], y)
            part = bt.iloc[p * per:(p + 1) * per][cols]
            if p == pages - 1 and len(bt) > 1:
                tot = pd.DataFrame([{"Branch": "TOTAL / GROUP", "Revenue": bt["Revenue"].sum(), "Revenue share %": 100.0,
                                     "Gross Profit": bt["Gross Profit"].sum(), "Gross margin %": k["Gross margin %"],
                                     "Total Expenses": bt["Total Expenses"].sum(), "Net Profit": bt["Net Profit"].sum(),
                                     "Net margin %": k["Net margin %"]}])
                part = pd.concat([part, tot], ignore_index=True)
            disp = _fmt(part, ints=["Revenue", "Gross Profit", "Total Expenses", "Net Profit", "Net profit rank"],
                        pcts=["Revenue share %", "Gross margin %", "GM vs group (pp)", "Net margin %", "Rev MoM %", "Rev 3M vs prior %"],
                        signed=["GM vs group (pp)", "Rev MoM %", "Rev 3M vs prior %"]).replace("nan", "–")
            disp = disp.rename(columns={"Revenue share %": "Share %", "GM vs group (pp)": "GM vs grp", "Net profit rank": "NP rank",
                                        "Rev 3M vs prior %": "Rev 3M vs prior", "Total Expenses": "Expenses"})
            y = _table(fig, disp, y, row_h=0.03, size=7, title="Branch scorecard (sorted by revenue; NP = net profit, GM = gross margin)")
            save(pdf, fig)

        # ---------------- 4. branch charts ----------------
        fig = _page("Branch comparison: charts")
        top = bt.head(15)
        ax = fig.add_axes([0.14, 0.52, 0.32, 0.36])
        _bar_h(ax, top["Branch"], top["Revenue"] / 1e6, f"Revenue by branch ({CURRENCY} M)", fmt=lambda v: f"{v:,.1f}")
        ax = fig.add_axes([0.62, 0.52, 0.32, 0.36])
        gm = top["Gross margin %"].fillna(0)
        _bar_h(ax, top["Branch"], gm, "Gross margin % by branch", color=[PALETTE[2] if v >= k["Gross margin %"] else PALETTE[3] for v in gm],
               fmt=lambda v: f"{v:.1f}%")
        ax.axvline(k["Gross margin %"], color="black", lw=0.8, ls="--")
        ax = fig.add_axes([0.14, 0.07, 0.32, 0.36])
        nm = top["Net margin %"].fillna(0)
        _bar_h(ax, top["Branch"], nm, "Net margin % by branch", color=[PALETTE[2] if v >= 0 else PALETTE[3] for v in nm], fmt=lambda v: f"{v:.1f}%")
        ax = fig.add_axes([0.58, 0.07, 0.37, 0.36])
        brm = R["branch_rev_monthly"].loc[top["Branch"].head(8)]
        for i, (name, row) in enumerate(brm.iterrows()):
            ax.plot(range(len(months)), row.to_numpy() / 1e6, marker="o", ms=3, label=textwrap.shorten(name, 22, placeholder="…"), color=PALETTE[i % 8])
        _month_axis(ax, months)
        _style(ax, f"Monthly revenue, top {min(8, len(top))} branches ({CURRENCY} M)")
        ax.legend(fontsize=6, ncol=2)
        save(pdf, fig)

        # ---------------- 5. income lines ----------------
        fig = _page("Income lines: performance and comparison")
        y = _bullets(fig, R["income_notes"], 0.885)
        ig = R["income_groups"]
        gcols = [c for c in ["Income group", "Total", "Share %", "Last month", "Prev month", "MoM %", "3M vs prior 3M %", "Trend %/mo"] if c in ig]
        y = _table(fig, _fmt(ig[gcols], ints=["Total", "Last month", "Prev month"], pcts=["Share %", "MoM %", "3M vs prior 3M %", "Trend %/mo"],
                             signed=["MoM %", "3M vs prior 3M %", "Trend %/mo"]), y, title="By income group", row_h=0.027, size=7)
        ia = R["income_accounts"].head(10)
        acols = [c for c in ["Account", "Total", "Share %", "Last month", "Prev month", "MoM %", "3M vs prior 3M %", "Trend %/mo"] if c in ia]
        y = _table(fig, _fmt(ia[acols], ints=["Total", "Last month", "Prev month"], pcts=["Share %", "MoM %", "3M vs prior 3M %", "Trend %/mo"],
                             signed=["MoM %", "3M vs prior 3M %", "Trend %/mo"]), y, title="Top income accounts", row_h=0.026, size=6.8)
        pm = R["product_margin"]
        if len(pm) and y > 0.3:
            y = _table(fig, _fmt(pm, ints=["Sales", "COGS", "Gross profit"], pcts=["Gross margin %", "Share of sales %"]), y,
                       title="Gross margin by product (sales vs matching COGS account)", row_h=0.026, size=7, width=0.7)
        if y > 0.2:
            ax = fig.add_axes([0.07, 0.11, 0.86, max(0.08, y - 0.15)])
            bm = R["income_monthly_by_group"]
            bottom = np.zeros(len(months))
            for i, c_ in enumerate(bm.columns):
                ax.bar(range(len(months)), bm[c_] / 1e6, bottom=bottom, label=c_.title(), color=PALETTE[i % 8])
                bottom += bm[c_].to_numpy() / 1e6
            _month_axis(ax, months)
            _style(ax, f"Monthly income by group ({CURRENCY} M)")
            ax.legend(fontsize=7, ncol=3)
        save(pdf, fig)

        # ---------------- 6. expenditure ----------------
        fig = _page("Expenditure analysis")
        y = _bullets(fig, R["exp_notes"], 0.885)
        eg = R["exp_groups"]
        ecols = [c for c in ["Group", "Total", "Share %", "% of revenue", "Last month", "Prev month", "MoM %", "3M vs prior 3M %", "Trend %/mo"] if c in eg]
        y = _table(fig, _fmt(eg[ecols], ints=["Total", "Last month", "Prev month"],
                             pcts=["Share %", "% of revenue", "MoM %", "3M vs prior 3M %", "Trend %/mo"], signed=["MoM %", "3M vs prior 3M %", "Trend %/mo"]),
                   y, title="By expense group", row_h=0.027, size=7)
        ea = R["exp_accounts"].head(10)
        acols = [c for c in ["Account", "Group", "Total", "Share %", "% of revenue", "MoM %", "Trend %/mo"] if c in ea]
        y = _table(fig, _fmt(ea[acols], ints=["Total"], pcts=["Share %", "% of revenue", "MoM %", "Trend %/mo"], signed=["MoM %", "Trend %/mo"]),
                   y, title="Ten largest expense accounts", row_h=0.026, size=6.8)
        save(pdf, fig)

        fig = _page("Expenditure analysis (continued)")
        y = 0.885
        mv = R["exp_movers"]
        if len(mv):
            mm = pd.concat([mv.head(6), mv.tail(5)]).drop_duplicates()[["Account", "Group", "Previous", "Current", "Change", "Change %"]]
            mm["Account"] = mm["Account"].map(lambda v: textwrap.shorten(v, 48, placeholder="…"))
            y = _table(fig, _fmt(mm, ints=["Previous", "Current", "Change"], pcts=["Change %"], signed=["Change %"]), y,
                       title=f"Biggest movers ({R['exp_movers_basis']})", row_h=0.03, size=7.2)
        ax = fig.add_axes([0.07, 0.11, 0.40, max(0.12, y - 0.16)])
        em = R["exp_monthly_by_group"]
        bottom = np.zeros(len(months))
        for i, c_ in enumerate(em.columns):
            ax.bar(range(len(months)), em[c_] / 1e6, bottom=bottom, label=c_, color=PALETTE[i])
            bottom += em[c_].to_numpy() / 1e6
        _month_axis(ax, months)
        _style(ax, f"Operating expenses by group ({CURRENCY} M)")
        ax.legend(fontsize=6.5, ncol=2)
        ax2 = fig.add_axes([0.57, 0.11, 0.37, max(0.12, y - 0.16)])
        ax2.plot(range(len(months)), R["margin_series"]["Expenses % revenue"], marker="o", color=PALETTE[3])
        _month_axis(ax2, months)
        _style(ax2, "Operating expenses as % of revenue")
        save(pdf, fig)

        # ---------------- 7. fixed vs variable ----------------
        fig = _page("Fixed and variable costs")
        y = _bullets(fig, R["cost_notes"], 0.885)
        cb = R["cost_buckets"]
        ccols = [c for c in ["Cost type", "Total", "% of all costs", "% of revenue", "Monthly average", "Last month", "Prev month", "MoM %"] if c in cb]
        y = _table(fig, _fmt(cb[ccols], ints=["Total", "Monthly average", "Last month", "Prev month"],
                             pcts=["% of all costs", "% of revenue", "MoM %"], signed=["MoM %"]), y, title="Cost behaviour summary", row_h=0.027, size=7)
        ca = R["cost_accounts"]
        yy = y
        for i, b in enumerate(BUCKETS):
            sub = ca[ca["Cost type"] == b].head(6)[["Account", "Total"]].copy()
            sub["Account"] = sub["Account"].map(lambda v: textwrap.shorten(v, 30, placeholder="…"))
            if len(sub):
                _table(fig, _fmt(sub, ints=["Total"]), yy, left=0.05 + i * 0.315, width=0.29, title=f"Largest {b.lower()} accounts",
                       row_h=0.026, size=6.3, col_w=[0.68, 0.32])
        ax = fig.add_axes([0.07, 0.12, 0.86, 0.15])
        cm = R["cost_monthly"][BUCKETS]
        bottom = np.zeros(len(months))
        for i, c_ in enumerate(BUCKETS):
            ax.bar(range(len(months)), cm[c_] / 1e6, bottom=bottom, label=c_, color=PALETTE[i])
            bottom += cm[c_].to_numpy() / 1e6
        _month_axis(ax, months)
        _style(ax, f"Operating expenses by cost behaviour ({CURRENCY} M)")
        ax.legend(fontsize=7, ncol=3)
        save(pdf, fig)

        # ---------------- 8. trends ----------------
        fig = _page("Trends")
        y = _bullets(fig, R["trend_notes"], 0.885)
        top_h = max(0.1, (y - 0.05 - 0.11 - 0.1) / 2)
        y0 = 0.11 + top_h + 0.1
        ax = fig.add_axes([0.07, y0, 0.40, top_h])
        for i, ln in enumerate(["Revenue", "Gross Profit", "Net Profit"]):
            ax.plot(range(len(months)), m[ln] / 1e6, marker="o", label=ln, color=PALETTE[i])
        _month_axis(ax, months)
        _style(ax, f"Revenue, gross profit and net profit ({CURRENCY} M)")
        ax.legend(fontsize=7)
        ax = fig.add_axes([0.55, y0, 0.40, top_h])
        g = (m["Revenue"].pct_change().replace([np.inf, -np.inf], np.nan) * 100).fillna(0)
        ax.bar(range(len(months)), g, color=[PALETTE[2] if v >= 0 else PALETTE[3] for v in g])
        _month_axis(ax, months)
        _style(ax, "Revenue growth, month on month (%)")
        ax = fig.add_axes([0.07, 0.11, 0.40, top_h])
        for i, c_ in enumerate(R["margin_series"].columns):
            ax.plot(range(len(months)), R["margin_series"][c_], marker="o", label=c_, color=PALETTE[i])
        _month_axis(ax, months)
        _style(ax, "Margins and expense ratio (%)")
        ax.legend(fontsize=7)
        ax = fig.add_axes([0.55, 0.11, 0.40, top_h])
        ax.bar(range(len(months)), m["Net Profit"] / 1e6, color=[PALETTE[2] if v >= 0 else PALETTE[3] for v in m["Net Profit"]])
        _month_axis(ax, months)
        _style(ax, f"Net profit by month ({CURRENCY} M)")
        save(pdf, fig)

        # ---------------- 9. statement ----------------
        st_df = R["statement"].reset_index()
        fig = _page("P&L statement by month")
        _table(fig, _fmt(st_df, ints=[c for c in st_df.columns if c != "Line"]), 0.88, row_h=0.04, size=7, title=f"Amounts in {CURRENCY}")
        save(pdf, fig)

        # ---------------- 10. forecast ----------------
        if fc_pack is not None:
            metric, actual, fc, used = fc_pack
            fig = _page(f"Forecast: {metric}", f"Model: {used}. Treat as a rough guide: the history is short.")
            ax = fig.add_axes([0.08, 0.1, 0.86, 0.72])
            f = fc[fc["kind"] == "forecast"]
            ax.fill_between(fc["ds"], fc["yhat_lower"] / 1e6, fc["yhat_upper"] / 1e6, alpha=0.15, label="80% range")
            ax.plot(actual.index, actual.values / 1e6, marker="o", label="Actual")
            ax.plot(f["ds"], f["yhat"] / 1e6, marker="o", label="Forecast")
            _style(ax, f"{metric} ({CURRENCY} M)")
            ax.legend(fontsize=8)
            save(pdf, fig)

        # ---------------- 11. notes ----------------
        fig = _page("Basis of preparation")
        notes = [
            "Source: Profit & Loss exports from the accounting system. Every figure is built from lowest-level accounts; parent rows are subtotals and are ignored.",
            "Revenue is the credit balance on sales accounts (including rebates and rental income). Gross profit = revenue - COGS. Net profit = gross profit - operating expenses.",
            "Reconciliation: " + ("each selected file's rebuilt Income, Expenses and Profit totals agree with the totals printed in its own report."
                                  if R["recon_ok"] else "AT LEAST ONE selected file does not agree with its own report totals. Review before relying on these figures."),
            ("Incomplete month(s) excluded from every figure: " + ", ".join(R["partial"]) + ".") if R["partial"] else "No incomplete month was excluded.",
            "Previous-period comparisons: latest month vs the month before, and latest 3 months vs the 3 months before. Percentages above 500% are shown as n/m because the base is too small.",
            "Fixed / semi-variable / variable labels are assigned from account names (and may have been edited in the app). COGS is treated as variable. "
            f"For break-even, semi-variable costs are treated as {R['semi_fixed_share'] * 100:.0f}% fixed.",
            "Branch net profit reflects only the costs posted to that branch. Shared costs booked centrally are not allocated.",
            "Trend %/mo is the least-squares slope of the monthly values divided by their average.",
        ]
        _bullets(fig, notes, 0.885, size=9)
        save(pdf, fig)
    return bio.getvalue()


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------
def build_excel(R, recon, clean):
    from openpyxl.styles import Font
    bio = io.BytesIO()
    scope_rows = [{"Company": co, "Branch": b} for co, brs in R["scope"].items() for b in brs]
    sheets = [
        ("Summary", pd.DataFrame({"Executive summary": R["summary"]}), False),
        ("Scope", pd.DataFrame(scope_rows), False),
        ("P&L Statement", R["statement"], True),
        ("Vs previous month", R["cmp_month"]["table"] if R["cmp_month"] else pd.DataFrame(), False),
        ("Vs previous quarter", R["cmp_quarter"]["table"] if R["cmp_quarter"] else pd.DataFrame(), False),
        ("Branch comparison", R["branch"], False),
        ("Branch revenue by month", R["branch_rev_monthly"].rename(columns=mlabel), True),
        ("Income groups", R["income_groups"], False),
        ("Income accounts", R["income_accounts"], False),
        ("Product margins", R["product_margin"], False),
        ("Expense groups", R["exp_groups"], False),
        ("Expense accounts", R["exp_accounts"], False),
        ("Expense movers", R["exp_movers"], False),
        ("Cost behaviour", R["cost_buckets"], False),
        ("Cost classification", R["classification"], False),
        ("Reconciliation", recon, False),
        ("Clean data", clean, False),
    ]
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        for name, df, keep_index in sheets:
            if df is None or df.empty:
                continue
            df.to_excel(xw, sheet_name=name[:31], index=keep_index)
        for ws in xw.book.worksheets:
            for c in ws[1]:
                c.font = Font(bold=True)
            ws.freeze_panes = "B2"
            for col in ws.columns:
                w = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(w + 2, 10), 70 if ws.title == "Summary" else 45)
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    if isinstance(c.value, (int, float)) and not isinstance(c.value, bool):
                        c.number_format = "#,##0.0" if abs(c.value) < 1000 and c.value != int(c.value) else "#,##0"
                    elif hasattr(c.value, "year"):
                        c.number_format = "mmm yyyy"
    return bio.getvalue()