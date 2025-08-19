#!/usr/bin/env python3
"""
Lead Cleaner Tool (diagnostic build)

Changes vs previous:
- Adds --sheet to pick a specific Excel sheet by name or index.
- Adds --debug to print detected columns and a few sample values.
- Friendlier error messages when required columns aren't found.
- Extra header aliases for "Licenced by/Agent" and "Trading Name".
- Brands column now lists names only (comma-separated). Brand-specific contacts go to Description.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import sys
import pandas as pd
import numpy as np

# -------- Configuration --------

CANON = {
    "trading_name": [
        "trading name", "brand", "brand name", "brand/trading name", "product brand",
        "trading_name", "tradingname", "brand (trading name)"
    ],
    "licenser": [
        "licenced by/agent", "licensed by/agent", "licensor", "licenced by", "licensed by",
        "licenced by / agent", "licenced by agent", "licenced/agent", "licensor/agent",
        "licence holder", "license holder", "licensing company", "licencee", "licensee",
        "licenced by/ agent", "licenced by /agent", "licenced by/ agent"
    ],
    "contact_name": ["contact name", "first name", "name", "contact first name"],
    "contact_surname": ["contact surname", "last name", "surname", "contact last name"],
    "title": ["title", "job title", "role", "position"],
    "contact_number": ["contact number", "phone", "telephone", "mobile", "cell", "phone number"],
    "email": ["email address", "email", "e-mail", "mail"],
    "website": ["website", "url", "site", "web site"],
    "description": ["description"],  # optional input
    "sku": ["sku", "skus", "sku count", "number of skus", "num skus", "no. of skus", "sku range"]
}

CONTACT_FIELDS = ["contact_name", "contact_surname", "title", "contact_number", "email", "website"]

def parse_sku_max(s) -> Optional[int]:
    """
    Extract the maximum integer from an SKU string.
    Examples:
      "100-500" -> 500
      "1,200 – 1,500" -> 1500
      "up to 250" -> 250
      "500+" -> 500
      "approx. 40" -> 40
    Returns None if no integer found.
    """
    if s is None:
        return None
    s = str(s)
    # replace common dash variants with spaces to not split numbers incorrectly
    s = s.replace("–", " ").replace("—", " ").replace("-", " ")
    # remove commas in numbers
    s = s.replace(",", " ")
    import re as _re
    nums = [_re.sub(r"\D", "", tok) for tok in _re.findall(r"\d[\d,]*", s)]
    vals = []
    for n in nums:
        if not n:
            continue
        try:
            vals.append(int(n))
        except ValueError:
            pass
    if not vals:
        return None
    return max(vals)

def _norm(s):
    if s is None:
        return None
    if isinstance(s, float) and np.isnan(s):
        return None
    return str(s).strip()

def _lc(s: str) -> str:
    return s.lower().strip()

def build_column_map(cols: List[str]) -> Dict[str, str]:
    col_map = {}
    if not cols:
        return col_map
    lc_cols = { _lc(c): c for c in cols }

    def closest_match(aliases: List[str]) -> str:
        for alias in aliases:
            alias_lc = _lc(alias)
            if alias_lc in lc_cols:
                return lc_cols[alias_lc]
            alias2 = alias_lc.replace("/", " ").replace("?", "").replace("-", " ").replace(",", " ")
            for c_lc, orig in lc_cols.items():
                c2 = c_lc.replace("/", " ").replace("?", "").replace("-", " ").replace(",", " ")
                if c2 == alias2:
                    return orig
            for c_lc, orig in lc_cols.items():
                if alias_lc in c_lc:
                    return orig
        return ""

    for canon_key, aliases in CANON.items():
        match = closest_match(aliases)
        if match:
            col_map[canon_key] = match

    return col_map

def choose_primary_contact(group_df: pd.DataFrame, colmap: Dict[str, str]):
    def filled_count(row):
        cnt = 0
        for k in CONTACT_FIELDS:
            c = colmap.get(k, "")
            if c and _norm(row.get(c)):
                cnt += 1
        return cnt

    idx = group_df.apply(filled_count, axis=1).astype(int)
    best_pos = idx.values.argmax()
    primary_row = group_df.iloc[best_pos]

    primary = {}
    for k in CONTACT_FIELDS:
        c = colmap.get(k, "")
        primary[k] = _norm(primary_row.get(c)) if c else None

    return primary_row, primary

def unique_contact_tuples(df: pd.DataFrame, colmap: Dict[str, str]) -> List[Tuple]:
    tuples = []
    seen = set()
    for _, r in df.iterrows():
        t = tuple(_norm(r.get(colmap.get(k, ""), None)) for k in CONTACT_FIELDS)
        if t not in seen:
            seen.add(t)
            tuples.append(t)
    return tuples

def build_description(alt_contacts: List[Tuple], primary_tuple: Tuple) -> str:
    lines = []
    for t in alt_contacts:
        if t == primary_tuple:
            continue
        c_name, c_surname, title, phone, email, website = t
        if not any([c_name, c_surname, title, phone, email, website]):
            continue
        parts = []
        if c_name or c_surname: parts.append(" ".join(filter(None, [c_name, c_surname])))
        if title: parts.append(f"Title: {title}")
        if phone: parts.append(f"Phone: {phone}")
        if email: parts.append(f"Email: {email}")
        if website: parts.append(f"Website: {website}")
        lines.append("• " + "; ".join(parts))
    return "\n".join(lines)

def clean_leads(df: pd.DataFrame, debug: bool=False) -> pd.DataFrame:
    colmap = build_column_map(df.columns.tolist())

    missing = []
    if "licenser" not in colmap:
        missing.append('Licenced by/Agent (licenser)')
    if "trading_name" not in colmap:
        missing.append('Trading Name (brand)')
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    # Normalize columns
    for canon, actual in colmap.items():
        df[actual] = df[actual].apply(_norm)

    # Fill licenser empty
    lic_col = colmap["licenser"]
    df[lic_col] = df[lic_col].fillna("UNKNOWN")
    df[lic_col] = df[lic_col].replace("", "UNKNOWN")

    if debug:
        print("== Detected column map ==")
        for k, v in colmap.items():
            print(f"  {k:15s} -> {v}")
        print("\nSample rows:")
        print(df.head(3).to_string(index=False))
        print()

    out_rows = []
    for licenser, g in df.groupby(lic_col, dropna=False):
        g = g.copy()
        primary_row, primary = choose_primary_contact(g, colmap)
        primary_tuple = tuple(primary.get(k) for k in CONTACT_FIELDS)

        # alt contacts for licenser
        alt_tuples = unique_contact_tuples(g, colmap)
        description = build_description(alt_tuples, primary_tuple)

        # Build Brands (names only) and collect brand-specific diffs to append to Description
        t_col = colmap["trading_name"]
        brand_names = set()
        brand_diff_lines = []

        for brand, gb in g.groupby(t_col, dropna=False):
            brand_norm = _norm(brand)
            if not brand_norm or brand_norm.lower() == "nan":
                continue
            brand_names.add(brand_norm)

            # pick best row for brand
            _, brand_best = choose_primary_contact(gb, colmap)

            # Only record details that differ from licenser primary
            parts = []
            cname = brand_best.get("contact_name"); csurname = brand_best.get("contact_surname")
            fulln = " ".join([x for x in [cname, csurname] if x])
            if fulln and fulln != " ".join([x for x in [primary.get("contact_name"), primary.get("contact_surname")] if x]):
                parts.append(fulln)
            if brand_best.get("title") and brand_best.get("title") != primary.get("title"):
                parts.append(f"Title: {brand_best.get('title')}")
            if brand_best.get("contact_number") and brand_best.get("contact_number") != primary.get("contact_number"):
                parts.append(f"Phone: {brand_best.get('contact_number')}")
            if brand_best.get("email") and brand_best.get("email") != primary.get("email"):
                parts.append(f"Email: {brand_best.get('email')}")
            if brand_best.get("website") and brand_best.get("website") != primary.get("website"):
                parts.append(f"Website: {brand_best.get('website')}")

            if parts:
                brand_diff_lines.append("• " + brand_norm + " — " + "; ".join(parts))

        if brand_diff_lines:
            description = (description + ("\n" if description else "")) + "\n".join(brand_diff_lines)

        brands_block = ", ".join(sorted(brand_names, key=lambda s: s.lower()))

        # Compute SKU (Max) across group
        sku_col = colmap.get("sku")
        sku_max = None
        if sku_col and sku_col in g.columns:
            for val in g[sku_col].dropna().tolist():
                m = parse_sku_max(val)
                if m is not None:
                    sku_max = m if sku_max is None else max(sku_max, m)

        out_rows.append({
            "Licenser": licenser,
            "Contact Name": primary.get("contact_name", ""),
            "Contact Surname": primary.get("contact_surname", ""),
            "Title": primary.get("title", ""),
            "Contact Number": primary.get("contact_number", ""),
            "Email Address": primary.get("email", ""),
            "Website": primary.get("website", ""),
            "SKU (Max)": sku_max if sku_max is not None else "",
            "Description": description,
            "Brands": brands_block
        })

    out_df = pd.DataFrame(out_rows, columns=[
        "Licenser", "Contact Name", "Contact Surname", "Title",
        "Contact Number", "Email Address", "Website", "SKU (Max)", "Description", "Brands"
    ])

    out_df = out_df.sort_values(by=["Licenser", "Contact Name"], na_position="last", kind="mergesort").reset_index(drop=True)
    return out_df

def write_excel_summary(out_df: pd.DataFrame, path: Path):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Summary")
        wb = writer.book
        ws = writer.sheets["Summary"]
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        widths = {"A": 28, "B": 18, "C": 18, "D": 18, "E": 18, "F": 28, "G": 28, "H": 14, "I": 60, "J": 60}
        for col_letter, width in widths.items():
            ws.set_column(f"{col_letter}:{col_letter}", width, wrap)
        ws.freeze_panes(1, 0)

def write_excel_per_licenser(out_df: pd.DataFrame, input_df: pd.DataFrame, lic_col: str, path: Path):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Summary")
        wb = writer.book
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        ws = writer.sheets["Summary"]
        ws.set_column("A:J", 30, wrap)
        ws.freeze_panes(1, 0)

        for licenser, g in input_df.groupby(lic_col, dropna=False):
            sheet = str(licenser)[:31] or "UNKNOWN"
            try:
                g.to_excel(writer, index=False, sheet_name=sheet)
                writer.sheets[sheet].set_column(0, g.shape[1]-1, 20, wrap)
                writer.sheets[sheet].freeze_panes(1, 0)
            except Exception:
                safe = "Sheet_" + str(abs(hash(sheet)) % (10**8))
                g.to_excel(writer, index=False, sheet_name=safe)
                writer.sheets[safe].set_column(0, g.shape[1]-1, 20, wrap)
                writer.sheets[safe].freeze_panes(1, 0)

def read_input(path: Path, sheet: Optional[Union[str,int]]=None) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".xlsx", ".xls"]:
        if sheet is None:
            return pd.read_excel(path, dtype=str)
        else:
            return pd.read_excel(path, dtype=str, sheet_name=sheet)
    elif ext in [".csv"]:
        return pd.read_csv(path, dtype=str)
    else:
        raise ValueError("Unsupported file type. Please provide .xlsx, .xls or .csv")

def main():
    parser = argparse.ArgumentParser(description="Lead Cleaner Tool (diagnostic build)")
    parser.add_argument("input_file", type=str, help="Input .xlsx or .csv")
    parser.add_argument("-o", "--output", type=str, default="", help="Output .xlsx (default: <input>_cleaned.xlsx)")
    parser.add_argument("--per-licenser-sheets", dest="per_licenser_sheets", action="store_true",
                        help="Also produce a workbook with raw rows per licenser")
    parser.add_argument("--sheet", type=str, default="",
                        help="Excel sheet name or index (0-based) to read")
    parser.add_argument("--debug", action="store_true", help="Print detected columns and sample rows")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Decide output names
    out_main = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_cleaned.xlsx")
    out_detail = input_path.with_name(input_path.stem + "_by_licenser.xlsx")

    # Parse sheet target
    sheet: Optional[Union[str,int]] = None
    if args.sheet:
        if args.sheet.isdigit():
            sheet = int(args.sheet)
        else:
            sheet = args.sheet

    try:
        df = read_input(input_path, sheet=sheet)
    except Exception as e:
        print("[ERROR] Failed to read the input file. Common causes:", file=sys.stderr)
        print(" - File is open/locked by Excel", file=sys.stderr)
        print(" - Requires engine: try 'pip install openpyxl' for .xlsx", file=sys.stderr)
        print(" - Wrong sheet name/index (use --sheet)", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    df_copy = df.copy()
    try:
        cleaned = clean_leads(df, debug=args.debug)
    except ValueError as ve:
        print("[ERROR] " + str(ve), file=sys.stderr)
        print("Tips:", file=sys.stderr)
        print(" - Ensure your columns include 'Licenced by/Agent' and 'Trading Name' (any close variants work).", file=sys.stderr)
        print(" - Exact strings not required, but they must be present somewhere in the header row.", file=sys.stderr)
        print(" - If your data is on another sheet, use: --sheet Sheet1 (or --sheet 0).", file=sys.stderr)
        if args.debug:
            print("Headers found:", list(df.columns), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("[ERROR] Unexpected failure while cleaning:", file=sys.stderr)
        print(str(e), file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Write outputs
    try:
        write_excel_summary(cleaned, out_main)
    except Exception as e:
        print("[ERROR] Failed to write the summary Excel. Try:", file=sys.stderr)
        print(" pip install xlsxwriter", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    if args.per_licenser_sheets:
        try:
            colmap = build_column_map(df_copy.columns.tolist())
            lic_col = colmap["licenser"]/Users/leighchejaikarran/Downloads/app_streamlit.py
            write_excel_per_licenser(cleaned, df_copy, lic_col, out_detail)
        except Exception as e:
            print("[WARN] Summary written, but failed to write per-licenser workbook:", file=sys.stderr)
            print(str(e), file=sys.stderr)

    print(f"Wrote summary: {out_main}")
    if args.per_licenser_sheets:
        print(f"Wrote detailed per-licenser workbook: {out_detail}")

if __name__ == "__main__":
    main()
