"""Loader for the ERP 'Profit and loss' export (wide, hierarchical layout).

Accepts .xlsx/.xlsm or a .csv saved from the same report.  Returns

    df        tidy table, one row per (account, month)
    controls  the report's own totals (INCOME, EXPENSES, Profit for the period)
              so the app can prove its numbers tie out to the source report.

Columns of df:
    Company, Branch, Section, Parent, Particulars, Level, IsLeaf, Date,
    Debit, Credit, Category, Amount, PeriodStart, PeriodEnd

* IsLeaf  - True for lowest-level accounts.  ONLY leaf rows may be summed;
            parent rows (e.g. STAFF COSTS) are subtotals of their children.
* Amount  - net amount with a "normal" sign: Credit-Debit for income,
            Debit-Credit for COGS/expenses.  Never use Debit or Credit alone
            (expense accounts can carry credit reversals).
"""
import csv
import io
import re
import warnings

import openpyxl
import pandas as pd

warnings.filterwarnings("ignore", message="Workbook contains no default style")

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# Level-3 headings in the export -> dashboard category.
SECTION_MAP = {
    "SALES ACCOUNTS": "Income",
    "COGS ACCOUNTS": "COGS",
    "ADMINISTRATION COSTS": "Expenses - Administration",
    "STAFF COSTS": "Expenses - Staff Costs",
    "OPERATING COSTS": "Expenses - Operational Costs",
    "FINANCE COSTS": "Expenses - Finance Costs",
}
# Fallback if a future export adds / renames a section.
_SECTION_KEYWORDS = [
    ("COGS", "COGS"), ("COST OF", "COGS"), ("SALES", "Income"), ("INCOME", "Income"),
    ("ADMIN", "Expenses - Administration"), ("STAFF", "Expenses - Staff Costs"),
    ("OPERAT", "Expenses - Operational Costs"), ("FINANCE", "Expenses - Finance Costs"),
]


def _section_category(section):
    if section in SECTION_MAP:
        return SECTION_MAP[section]
    up = (section or "").upper()
    for kw, cat in _SECTION_KEYWORDS:
        if kw in up:
            return cat
    return "Other"


def _num(v):
    if v is None:
        return 0.0
    s = str(v).strip().replace(",", "")
    if s in ("", "-", "None"):
        return 0.0
    if s.startswith("(") and s.endswith(")"):          # (1,234) accounting negative
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return 0.0


def _grid(file_obj, filename=""):
    """Return the sheet as a list of equal-length lists (0-based)."""
    name = (filename or getattr(file_obj, "name", "") or "").lower()
    if name.endswith(".csv"):
        raw = file_obj.read() if hasattr(file_obj, "read") else open(file_obj, "rb").read()
        text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw
        rows = list(csv.reader(io.StringIO(text)))
    else:
        wb = openpyxl.load_workbook(file_obj, data_only=True)
        rows = [list(r) for r in wb.active.iter_rows(values_only=True)]
    width = max((len(r) for r in rows), default=0)
    return [r + [None] * (width - len(r)) for r in rows]


def _cell(grid, r, c):
    return grid[r][c] if 0 <= r < len(grid) and 0 <= c < len(grid[r]) else None


def load_pl(file_obj, filename=""):
    grid = _grid(file_obj, filename)

    # ---- header block: company, branch, period -----------------------------
    company, branch = "Unknown", "Company-wide"
    p_start = p_end = None
    hdr_row = None
    for r, row in enumerate(grid):
        first = str(row[0]).strip() if row[0] is not None else ""
        if first == "Particulars":
            hdr_row = r
            break
        for v in row:
            if not isinstance(v, str) or not v.strip():
                continue
            t = v.strip()
            if t.startswith("Branch Name"):
                branch = t.split("=", 1)[1].strip()
            m = re.search(r"(\d{2})/(\d{2})/(\d{4})\s*to\s*(\d{2})/(\d{2})/(\d{4})", t)
            if m:
                d1, m1, y1, d2, m2, y2 = map(int, m.groups())
                p_start, p_end = pd.Timestamp(y1, m1, d1), pd.Timestamp(y2, m2, d2)
            elif r == 0:
                company = t
    if hdr_row is None:
        raise ValueError("This doesn't look like the P&L export: no 'Particulars' header row found.")
    if p_start is None:
        raise ValueError("Could not find the period line (e.g. '01/01/2026 to 02/10/2026').")

    # ---- column map ----------------------------------------------------------
    hdr = [str(x).strip() if x is not None else "" for x in grid[hdr_row]]
    lab = [str(x).strip() if x is not None else "" for x in grid[hdr_row - 1]]
    try:
        tot_dr, tot_cr = hdr.index("Debit"), hdr.index("Credit")
    except ValueError:
        raise ValueError("Header row must contain 'Debit' and 'Credit' total columns.")

    month_cols, year, prev = [], p_start.year, None
    for c, text in enumerate(lab):
        if text in MONTHS:
            midx = MONTHS.index(text) + 1
            if prev is not None and midx < prev:        # financial year crosses New Year
                year += 1
            prev = midx
            if hdr[c] == "Cr" and c > 0 and hdr[c - 1] == "Dr":
                dr, cr = c - 1, c
            elif hdr[c] == "Dr" and c + 1 < len(hdr) and hdr[c + 1] == "Cr":
                dr, cr = c, c + 1
            else:
                raise ValueError(f"Could not find Dr/Cr columns for {text}.")
            month_cols.append((pd.Timestamp(year, midx, 1), dr, cr))
    if not month_cols:
        raise ValueError("No month columns found above the Dr/Cr header row.")

    # ---- walk the account rows -------------------------------------------------
    rows = []
    for r in range(hdr_row + 1, len(grid)):
        raw = grid[r][0]
        if raw is None or not str(raw).strip():
            continue
        rows.append((r, str(raw)))

    recs, controls, section, stack = [], {}, None, []
    for i, (r, raw) in enumerate(rows):
        name = raw.strip()
        level = len(raw) - len(raw.lstrip())

        if name in ("Profit for the period", "Grand Total"):
            net = _num(_cell(grid, r, tot_dr)) - _num(_cell(grid, r, tot_cr))
            monthly = {d: _num(_cell(grid, r, dr)) - _num(_cell(grid, r, cr)) for d, dr, cr in month_cols}
            controls[name] = {"Debit": _num(_cell(grid, r, tot_dr)), "Credit": _num(_cell(grid, r, tot_cr)),
                              "Net": net, "Monthly": monthly}
            continue
        if level == 0:                                   # INCOME / EXPENSES banners
            controls[name] = {"Debit": _num(_cell(grid, r, tot_dr)), "Credit": _num(_cell(grid, r, tot_cr)),
                              "Monthly": {d: (_num(_cell(grid, r, cr)) if name == "INCOME" else _num(_cell(grid, r, dr)))
                                          for d, dr, cr in month_cols}}
            section, stack = None, []
            continue
        if level == 3:
            section = name
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1] if stack else ""
        stack.append((level, name))

        nxt = rows[i + 1][1] if i + 1 < len(rows) else ""
        next_level = len(nxt) - len(nxt.lstrip())
        is_leaf = next_level <= level
        cat = _section_category(section)
        for d, dr_c, cr_c in month_cols:
            dr, cr = _num(_cell(grid, r, dr_c)), _num(_cell(grid, r, cr_c))
            if dr or cr:
                recs.append(dict(Company=company, Branch=branch, Section=section or "", Parent=parent,
                                 Particulars=name, Level=level, IsLeaf=is_leaf, Date=d,
                                 Debit=dr, Credit=cr, Category=cat,
                                 PeriodStart=p_start, PeriodEnd=p_end))
    cols = ["Company", "Branch", "Section", "Parent", "Particulars", "Level", "IsLeaf", "Date",
            "Debit", "Credit", "Category", "PeriodStart", "PeriodEnd"]
    df = pd.DataFrame(recs, columns=cols)
    df["Amount"] = df["Credit"] - df["Debit"]
    df.loc[df["Category"] != "Income", "Amount"] *= -1
    return df, controls


def leaves(df):
    """Rows that are safe to sum (parents/subtotals excluded)."""
    return df[df["IsLeaf"]]


def _pnl_from_leaves(lf):
    sales = lf.loc[lf.Category == "Income", "Amount"].sum()
    cogs = lf.loc[lf.Category == "COGS", "Amount"].sum()
    exp = lf.loc[lf.Category.str.startswith("Expenses"), "Amount"].sum()
    return sales, cogs, exp


def check(df, controls, tol=1.0):
    """Rebuild the report's own totals from leaf rows and compare."""
    sales, cogs, exp = _pnl_from_leaves(leaves(df))
    gp, net = sales - cogs, sales - cogs - exp
    rep_inc = controls.get("INCOME", {}).get("Credit")
    rep_exp = controls.get("EXPENSES", {}).get("Debit")
    rep_net = controls.get("Profit for the period", {}).get("Net")
    out = {"Sales": sales, "COGS": cogs, "Gross profit": gp, "Expenses": exp, "Net profit": net,
           "Report INCOME": rep_inc, "Report EXPENSES": rep_exp, "Report profit": rep_net}
    diffs = [abs(a - b) for a, b in ((gp, rep_inc), (exp, rep_exp), (net, rep_net)) if b is not None]
    out["OK"] = bool(diffs) and max(diffs) <= tol and not (df["Category"] == "Other").any()
    return out


def check_monthly(df, controls):
    """Month-by-month tie-out against the report's INCOME / EXPENSES / Profit rows."""
    lf = leaves(df)
    if lf.empty:
        return pd.DataFrame()
    p = lf.pivot_table(index="Date", columns="Category", values="Amount", aggfunc="sum", fill_value=0.0)
    for c in ("Income", "COGS"):
        if c not in p:
            p[c] = 0.0
    exp_cols = [c for c in p.columns if c.startswith("Expenses")]
    out = pd.DataFrame(index=p.index)
    out["Gross profit (calc)"] = p["Income"] - p["COGS"]
    out["Expenses (calc)"] = p[exp_cols].sum(axis=1) if exp_cols else 0.0
    out["Net profit (calc)"] = out["Gross profit (calc)"] - out["Expenses (calc)"]
    ctl = lambda k: pd.Series(controls.get(k, {}).get("Monthly", {}), dtype=float)
    out["INCOME (report)"] = ctl("INCOME")
    out["EXPENSES (report)"] = ctl("EXPENSES")
    out["Profit (report)"] = ctl("Profit for the period")
    out = out.fillna(0.0)
    out["Gross profit diff"] = out["Gross profit (calc)"] - out["INCOME (report)"]
    out["Expenses diff"] = out["Expenses (calc)"] - out["EXPENSES (report)"]
    out["Net profit diff"] = out["Net profit (calc)"] - out["Profit (report)"]
    out.index.name = "Month"
    return out