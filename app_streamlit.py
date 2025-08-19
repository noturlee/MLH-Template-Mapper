import io
import zipfile
from datetime import datetime

import pandas as pd
import streamlit as st
import numpy as np

# ====== App Setup ======
st.set_page_config(page_title="Lead Central", page_icon="⛮", layout="wide")

st.markdown('''
<style>
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
div.stButton>button, .stDownloadButton>button {
  border-radius: 14px;
  padding: 0.6rem 1rem;
  border: 1px solid rgba(0,0,0,0.08);
  box-shadow: 0 1px 8px rgba(0,0,0,0.06);
}
.card {
  padding: 1.1rem 1.2rem;
  border-radius: 16px;
  border: 1px solid rgba(0,0,0,.08);
  box-shadow: 0 4px 16px rgba(0,0,0,.06);
}
</style>
''', unsafe_allow_html=True)

# ====== Lead Accelerator owner groups ======
OWNER_GROUPS = {
    "DSR": [
        {"id": "005Kf000002JqsgIAC", "name": "anathibe"},
        {"id": "005at000000SlDhAAK", "name": "lddyakal"},
        {"id": "005at000000AtT3AAK", "name": "tkngcong"},
        {"id": "005at000000ecxpAAA", "name": "precimah"},
        {"id": "005at000000PzJdAAK", "name": "rkagiso"},
        {"id": "005Kf000000jsm5IAA", "name": "breezib"},
        {"id": "005at000002qDEXAA2", "name": "Agapah"},
        {"id": "005at000002G4BNAA0", "name": "tebogos"},
    ],
    "SMB": [
        {"id": "005at000000eUSPAA2", "name": "kngobeni"},
        {"id": "005at000000dsJtAAI", "name": "dimphth"},
        {"id": "005Kf000001jWpHIAU", "name": "kbopape"},
    ],
    "CC": [
        {"id": "005at0000037FVIAA2", "name": "taufelma"},
        {"id": "005at000002LBowAAG", "name": "trjzwane"},
    ],
}

# ====== Cleaner core logic ======
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
    "description": ["description"],
    "sku": ["sku", "skus", "sku count", "number of skus", "num skus",
            "no. of skus", "no of skus", "sku range", "skus range", "sku's", "SKU"]
}
CONTACT_FIELDS = ["contact_name", "contact_surname", "title", "contact_number", "email", "website"]

def parse_sku_max(s):
    if s is None:
        return None
    s = str(s)
    # keep commas; regex will strip non-digits inside each token
    s = s.replace("–", " ").replace("—", " ").replace("-", " ")
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
    return max(vals) if vals else None

def _norm(s):
    if s is None: return None
    if isinstance(s, float) and np.isnan(s): return None
    return str(s).strip()

def _lc(s: str) -> str:
    return s.lower().strip()

def build_column_map(cols):
    col_map = {}
    if not cols:
        return col_map
    lc_cols = { _lc(c): c for c in cols }

    def closest_match(aliases):
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

def choose_primary_contact(group_df, colmap):
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

def unique_contact_tuples(df, colmap):
    tuples = []
    seen = set()
    for _, r in df.iterrows():
        t = tuple(_norm(r.get(colmap.get(k, ""), None)) for k in CONTACT_FIELDS)
        if t not in seen:
            seen.add(t)
            tuples.append(t)
    return tuples

def build_description(alt_contacts, primary_tuple):
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

def clean_leads(df):
    colmap = build_column_map(df.columns.tolist())

    missing = []
    if "licenser" not in colmap:
        missing.append('Licenced by/Agent (licenser)')
    if "trading_name" not in colmap:
        missing.append('Trading Name (brand)')
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    # normalize values in matched columns
    for _, actual in colmap.items():
        df[actual] = df[actual].apply(_norm)

    lic_col = colmap["licenser"]
    df[lic_col] = df[lic_col].fillna("UNKNOWN").replace("", "UNKNOWN")

    out_rows = []

    for licenser, g in df.groupby(lic_col, dropna=False):
        g = g.copy()
        primary_row, primary = choose_primary_contact(g, colmap)
        primary_tuple = tuple(primary.get(k) for k in CONTACT_FIELDS)

        alt_tuples = unique_contact_tuples(g, colmap)
        description = build_description(alt_tuples, primary_tuple)

        # Brands (names only) + brand diffs
        t_col = colmap["trading_name"]
        brand_names = set()
        brand_diff_lines = []
        for brand, gb in g.groupby(t_col, dropna=False):
            brand_norm = _norm(brand)
            if not brand_norm or brand_norm.lower() == "nan":
                continue
            brand_names.add(brand_norm)
            _, brand_best = choose_primary_contact(gb, colmap)
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

        # SKU max per licenser
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

def to_excel_bytes(df, sheet_name="Summary"):
    import xlsxwriter  # ensure installed
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        wb = writer.book
        ws = writer.sheets[sheet_name]
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        widths = {"A": 28, "B": 18, "C": 18, "D": 18, "E": 18, "F": 28, "G": 28, "H": 14, "I": 60, "J": 60}
        for col_letter, width in widths.items():
            try:
                ws.set_column(f"{col_letter}:{col_letter}", width, wrap)
            except Exception:
                pass
        ws.freeze_panes(1, 0)
    output.seek(0)
    return output.getvalue()

def to_per_licenser_workbook_bytes(input_df, lic_col_name):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        for lic, g in input_df.groupby(lic_col_name, dropna=False):
            name = (str(lic)[:31] or "UNKNOWN")
            try:
                g.to_excel(writer, index=False, sheet_name=name)
                writer.sheets[name].set_column(0, g.shape[1]-1, 20, wrap)
                writer.sheets[name].freeze_panes(1, 0)
            except Exception:
                safe = "Sheet_" + str(abs(hash(name)) % (10**8))
                g.to_excel(writer, index=False, sheet_name=safe)
                writer.sheets[safe].set_column(0, g.shape[1]-1, 20, wrap)
                writer.sheets[safe].freeze_panes(1, 0)
    output.seek(0)
    return output.getvalue()

# ====== Helpers for Accelerator ======
def _pick_col(df, aliases):
    aliases = [a.lower() for a in aliases]
    for c in df.columns:
        if str(c).lower().strip() in aliases:
            return c
    return None

# ====== Pages ======
def show_home():
    st.title("⛮ Lead Central")
    st.caption("Pick a tool to get started.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
          <h3>🧼 Lead Cleaner</h3>
          <p>Clean, dedupe, and summarize licensers, contacts, brands, and SKU max.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Lead Cleaner", use_container_width=True, type="primary"):
            st.session_state.page = "cleaner"
            st.rerun()
    with col2:
        st.markdown("""
        <div class="card">
          <h3>🚀 Lead Accelerator</h3>
          <p>Assign Leads & Lead Products to owners (evenly or with weights).</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Lead Accelerator", use_container_width=True):
            st.session_state.page = "accelerator"
            st.rerun()

def show_lead_cleaner():
    if st.button("← Back to Lead Central"):
        st.session_state.page = "home"
        st.rerun()

    st.title("🧼 Lead Cleaner")
    st.caption("Upload your raw leads file (Excel or CSV). We'll group by **Licenced by/Agent**, keep one primary contact, merge alternates into **Description**, list **Brands** (names only), and compute **SKU (Max)**.")

    left, right = st.columns([1.2, 1])

    with left:
        uploaded = st.file_uploader("Upload Excel (.xlsx/.xls) or CSV", type=["xlsx", "xls", "csv"])
        per_licenser = st.toggle("Also create raw per-licenser workbook", value=False,
                                 help="Adds a workbook with the original rows split into sheets per licenser.")
        if uploaded and uploaded.name.lower().endswith((".xlsx", ".xls")):
            try:
                xls = pd.ExcelFile(uploaded)
                sheet_name = st.selectbox("Select sheet", options=xls.sheet_names, index=0)
                df_raw = xls.parse(sheet_name=sheet_name, dtype=str)
            except Exception as e:
                st.error(f"Failed to read Excel: {e}")
                st.stop()
        elif uploaded and uploaded.name.lower().endswith(".csv"):
            try:
                df_raw = pd.read_csv(uploaded, dtype=str)
                sheet_name = None
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.stop()
        else:
            df_raw = None
            sheet_name = None

        run = st.button("🧼 Clean Leads", type="primary", disabled=(uploaded is None))

    with right:
        st.info("Make sure your header row includes **Licenced by/Agent** and **Trading Name** (close variants are OK).")
        st.caption("Output includes: Licenser primary contact, merged Description (other contacts + brand-specific diffs), Brands (names only), and SKU (Max).")

    if run and df_raw is not None:
        try:
            with st.spinner("Cleaning in progress…"):
                cleaned = clean_leads(df_raw.copy())
            st.success("Done! Preview below.")
        except Exception as e:
            st.error(f"Cleaning failed: {e}")
            st.stop()

        st.subheader("Preview (first 50 rows)")
        n = min(50, len(cleaned))
        preview = cleaned.head(n).copy()
        preview.index = range(1, n + 1)
        preview.index.name = "Row #"
        st.dataframe(preview, use_container_width=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = uploaded.name.rsplit(".", 1)[0]

        summary_bytes = to_excel_bytes(cleaned, sheet_name="Summary")
        st.download_button(
            "⬇️ Download Summary (.xlsx)",
            data=summary_bytes,
            file_name=f"{base}_cleaned_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if per_licenser:
            colmap = build_column_map(df_raw.columns.tolist())
            lic_col = colmap.get("licenser")
            if lic_col:
                per_lic_bytes = to_per_licenser_workbook_bytes(df_raw, lic_col)
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
                    z.writestr(f"{base}_cleaned_{ts}.xlsx", summary_bytes)
                    z.writestr(f"{base}_by_licenser_{ts}.xlsx", per_lic_bytes)
                zbuf.seek(0)
                st.download_button(
                    "⬇️ Download Summary + Per-Licenser (ZIP)",
                    data=zbuf.getvalue(),
                    file_name=f"{base}_outputs_{ts}.zip",
                    mime="application/zip"
                )
            else:
                st.warning("Couldn't detect the Licenser column; skipping per-licenser workbook.")

def show_lead_accelerator():
    if st.button("← Back to Lead Central"):
        st.session_state.page = "home"
        st.rerun()

    st.title("🚀 Lead Accelerator")
    st.caption("Assign Leads and Lead Products to owners evenly, or with weighted percentages.")

    left, right = st.columns([1.2, 1])

    with left:
        uploaded = st.file_uploader("Upload Excel (.xlsx/.xls)", type=["xlsx", "xls"])

        team = st.selectbox("Choose team", options=list(OWNER_GROUPS.keys()))
        assign_mode = st.radio("Assignment mode", ["Even distribution", "Specific members with weights"])

        team_members = OWNER_GROUPS.get(team, [])

        owner_pool = []
        chosen = []
        weights = {}

        if assign_mode == "Specific members with weights" and team_members:
            names = [m["name"] for m in team_members]
            chosen = st.multiselect("Pick members", names, default=names[:1] if names else [])
            for nm in chosen:
                weights[nm] = st.slider(f"{nm} %", min_value=0, max_value=100, value=0, step=1)
            total = sum(weights.values())
            if chosen:
                st.write(f"**Total:** {total}%")
            if chosen and total == 100:
                # Build a 100-slot pool, repeated by percentage
                for m in team_members:
                    if m["name"] in weights and weights[m["name"]] > 0:
                        owner_pool.extend([m["id"]] * int(weights[m["name"]]))
        else:
            owner_pool = [m["id"] for m in team_members]

        # Column aliases
        lead_col_aliases = ["lead id", "lead_id", "leadid", "lead sfid"]
        prod_col_aliases = ["lead product id", "lead_product_id", "lead productid", "lead product sfid"]

        # Enable button only when everything is ready
        need_weights_ok = (assign_mode != "Specific members with weights") or (chosen and sum(weights.values()) == 100)
        run = st.button("⚡ Assign & Generate", type="primary",
                        disabled=(uploaded is None or not team_members or not need_weights_ok))

    with right:
        st.info("Input must contain **Lead ID** and/or **Lead Product ID** columns (case-insensitive).")
        st.caption("Outputs: **leads_YYYYMMDD_HHMMSS.xlsx**, **lead_products_YYYYMMDD_HHMMSS.xlsx**, and a combined ZIP.")

    if run and uploaded:
        try:
            df = pd.read_excel(uploaded, dtype=str)
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")
            return

        lead_col = _pick_col(df, lead_col_aliases)
        prod_col = _pick_col(df, prod_col_aliases)

        if not lead_col and not prod_col:
            st.error("Couldn't find 'Lead ID' or 'Lead Product ID' columns.")
            return
        if not owner_pool:
            st.error("No owners available/selected.")
            return

        def assign_ids(series, pool):
            s = series.dropna().astype(str)
            out = pd.DataFrame({"Id": s.values})
            out["Owner Id"] = [pool[i % len(pool)] for i in range(len(out))]
            return out

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputs = {}

        if lead_col:
            lead_df = assign_ids(df[lead_col], owner_pool)
            outputs[f"leads_{ts}.xlsx"] = to_excel_bytes(lead_df, sheet_name="Leads")
        if prod_col:
            prod_df = assign_ids(df[prod_col], owner_pool)
            outputs[f"lead_products_{ts}.xlsx"] = to_excel_bytes(prod_df, sheet_name="LeadProducts")

        # Preview
        st.subheader("Preview")
        if lead_col:
            st.write("**Leads (first 10 rows)**")
            st.dataframe(lead_df.head(10), use_container_width=True)
        if prod_col:
            st.write("**Lead Products (first 10 rows)**")
            st.dataframe(prod_df.head(10), use_container_width=True)

        # Individual downloads
        st.subheader("Download")
        for fname, data in outputs.items():
            st.download_button(f"⬇️ Download {fname}", data=data, file_name=fname,
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ZIP download
        if outputs:
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
                for fname, data in outputs.items():
                    z.writestr(fname, data)
            zbuf.seek(0)
            st.download_button("⬇️ Download All (ZIP)", data=zbuf.getvalue(),
                               file_name=f"assigned_data_{ts}.zip", mime="application/zip")

# ====== Router ======
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "cleaner":
    show_lead_cleaner()
elif st.session_state.page == "accelerator":
    show_lead_accelerator()
