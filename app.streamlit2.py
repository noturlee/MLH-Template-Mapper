import io
import os
import re
import difflib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import html as _html

# ============================
# Utilities
# ============================

RE_REQUIRED_PAT = re.compile(r"\b(required|mandatory|compulsory)\b", re.I)

def clean_header_name(s: str) -> str:
    s = str(s or "")
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"[\s]+", " ", s).strip()
    s = re.sub(r"[\*\[\(]\s*(required|mandatory|compulsory)\s*[\]\)]?", "", s, flags=re.I)
    s = re.sub(r"[:\-–]\s*(required|mandatory|compulsory)\b", "", s, flags=re.I)
    s = re.sub(r"\[[^\]]*\]", "", s).strip()
    s = re.sub(r"[•:\-–]+$", "", s).strip()
    return s

def normalize_for_match(s: str) -> str:
    s = str(s or "").lower()
    s = s.replace("\n", " ")
    s = re.sub(r"[\s\-_]+", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

# Synonyms
SYNONYMS: Dict[str, List[str]] = {
    "sku": ["seller sku","item sku","product sku","stock keeping unit","sku id","sku"],
    "asin": ["amazon asin","asin code"],
    "brand": ["brand name","manufacturer brand","brand"],
    "product name": ["title","item name","product title","listing title","name"],
    "product description": ["description","long description","full description","details"],
    "short description": ["bullet summary","short desc","brief description"],
    "gtin": ["ean","upc","jan","barcode","bar code","ean/upc","product code","ean code","upc code"],
    "mpn": ["manufacturer part number","model number","model#","mfg part","part number","mnf code"],
    "gcid": ["global catalog id","catalog id"],
    "color": ["colour","col","clr","shade","tone"],
    "size": ["size name","variant size","dimension size","size"],
    "material": ["fabric","material type","composition"],
    "pattern": ["print","style pattern"],
    "gender": ["target gender","audience gender","sex"],
    "age range": ["age group","age","age grade"],
    "department": ["target department","dept"],
    "style": ["design style"],
    "price": ["retail price","list price","selling price","unit price","msrp","rrp","standard price","regular price"],
    "sale price": ["discount price","promotional price","offer price"],
    "unit count": ["units per pack","count","qty","quantity"],
    "unit count type": ["unit type","unit of measure","uom"],
    "pack size": ["pack quantity","pack count"],
    "item weight": ["weight","net weight","product weight"],
    "package weight": ["shipping weight","boxed weight"],
    "item dimensions": ["dimensions","product dimensions","size (l x w x h)","lwh","length width height"],
    "package dimensions": ["shipping dimensions","box dimensions","carton dimensions"],
    "main image url": ["main image","primary image","image url","default image","image","images","feature image","featured image"],
    "image url1": ["image1","additional image 1","alt image 1","other image url1","gallery 1","picture 1","img 1","thumbnail 1"],
    "image url2": ["image2","additional image 2","alt image 2","other image url2","gallery 2","picture 2","img 2","thumbnail 2"],
    "image url3": ["image3","additional image 3","alt image 3","other image url3","gallery 3","picture 3","img 3","thumbnail 3"],
    "image url4": ["image4","additional image 4","alt image 4","other image url4","gallery 4","picture 4","img 4","thumbnail 4"],
    "image url5": ["image5","additional image 5","alt image 5","other image url5","gallery 5","picture 5","img 5","thumbnail 5"],
    "video url": ["product video","video link"],
    "bullet point1": ["bullet1","feature1","key feature 1","highlight 1","bullet point 1"],
    "bullet point2": ["bullet2","feature2","key feature 2","highlight 2","bullet point 2"],
    "bullet point3": ["bullet3","feature3","key feature 3","highlight 3","bullet point 3"],
    "category": ["product category","browse node","node","taxonomy"],
    "subcategory": ["sub category","sub-category","child category"],
    "batteries required": ["battery required","requires batteries","batteries needed"],
    "batteries included": ["battery included","includes batteries"],
    "battery type": ["battery cell type","battery form factor"],
    "wattage": ["power","power rating"],
    "voltage": ["input voltage","rated voltage"],
    "warranty": ["warranty description","warranty info"],
    "fit type": ["fit","fitment"],
    "care instructions": ["wash care","washing instructions","care","garment care"],
    "fabric type": ["material","fabric"],
    "ingredients": ["ingredient list","composition","ingr"],
    "allergen information": ["allergens","allergy info"],
    "flavour": ["flavor","taste"],
    "servings per container": ["serves","servings"],
    "serving size": ["portion size","recommended serving"],
    "expiration date": ["expiry date","best before","bb date"],
    "storage instructions": ["storage","keep refrigerated","store info"],
    "net content": ["net quantity","net content amount"],
    "item name": ["name","product title","title"],
    "external product id": ["gtin","ean","upc","barcode","jan","isbn","gtin, upc, ean, or isbn","meta:_ean","meta:_barcode","meta:_sku_barcode"],
    "manufacturer": ["brand","brand name","manufacturer"],
    "quantity": ["stock","stock quantity","qty","quantity"],
    "colour": ["color","colour","pa_color","attribute 1 value(s)","attribute color","attribute colour"],
    "material type": ["material","fabric","attribute 3 value(s)","material type"],
    "generic keywords": ["tags","keywords","search terms"],
    "parent_child": ["parent_child","parent child","relationship type"],
    "variation_theme": ["variation_theme","variation theme","theme"],
    "scent": ["fragrance","perfume","aroma","scent"],
    "package dimensions in cm": ["package dimensions in cm","length (cm) x width (cm) x height (cm)"],
}

def canonicalize(name: str) -> str:
    cleaned = clean_header_name(name)
    norm = normalize_for_match(cleaned)
    if norm in {"gtin,upc,ean,orisbn", "gtinupceanorisbn"}:
        norm = "externalproductid"
    for canon, alts in SYNONYMS.items():
        canon_norm = normalize_for_match(canon)
        alt_norms = [normalize_for_match(a) for a in alts]
        if norm == canon_norm or norm in alt_norms:
            return canon_norm
    m = re.match(r"^(ean|upc|gtin|jan|barcode)(\d+)$", norm)
    if m: return f"gtin{m.group(2)}"
    if norm in {"ean","upc","jan","barcode"}: return "gtin"
    m = re.match(r"^(imageurl|image|additionalimage|altimage)(\d+)$", norm)
    if m: return f"imageurl{m.group(2)}"
    if norm in {"image","imageurl","additionalimage","altimage"}: return "main image url"
    m = re.match(r"^(bulletpoint|bullet|feature|highlight)(\d+)$", norm)
    if m: return f"bullet point{m.group(2)}"
    return norm

def read_first_sheet(path_or_buffer) -> pd.DataFrame:
    xl = pd.ExcelFile(path_or_buffer)
    df = xl.parse(xl.sheet_names[0], header=None, dtype=str)
    return df.fillna("")

# ---------- Text cleaning / extraction helpers ----------
TAG_RE = re.compile(r"<[^>]+>")
WS_RE  = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"[•\u2022;\n\r]+|(?<=[.!?])\s+")
DIM_TOKEN_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*(mm|cm|in|inch|inches|\"))"
    r"(?:\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|in|inch|inches|\"))?"
    r"(?:\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm|in|inch|inches|\"))?",
    re.I
)

STOPWORDS = {
    "the","and","with","for","from","your","you","our","are","this","that","these","those","a","an","of","to","on","in",
    "by","at","it","its","is","as","or","be","can","will","out","up","over","into","per","made","how","while","when",
    "than","then","also","very","more","most","has","have","had","their","there","which"
}

def strip_html(text: str) -> str:
    s = _html.unescape(str(text or ""))
    s = re.sub(r"<\s*br\s*/?>", " ", s, flags=re.I)
    s = TAG_RE.sub(" ", s)
    s = re.sub(r"[_]{2,}", " ", s)
    s = re.sub(r"(#?[A-Za-z]{1,6}_[0-9A-Za-z]{1,6})", " ", s)
    s = re.sub(r"&[a-z]{2,6};", " ", s, flags=re.I)
    s = s.replace("\\n", " ").replace("\n", " ").replace("\r", " ").replace("“","").replace("”","").replace("’","'")
    s = WS_RE.sub(" ", s).strip()
    return s

def to_cm(value: float, unit: str) -> float:
    unit = unit.lower().strip('" ')
    if unit in ("cm",): return value
    if unit in ("mm",): return value / 10.0
    if unit in ("in","inch","inches",'"'): return value * 2.54
    return value

def extract_dimensions_cm(text: str) -> str:
    s = strip_html(text)
    m = DIM_TOKEN_RE.search(s)
    if not m: return ""
    parts = []
    for i in (1,3,5):
        v = m.group(i)
        u = m.group(i+1) if i+1 <= 6 else None
        if v and u:
            try:
                cm = to_cm(float(v), u)
                parts.append(f"{cm:.2f}")
            except Exception:
                pass
    if not parts: return ""
    return " x ".join(parts) + " cm"

def sentences_from_text(text: str) -> List[str]:
    s = strip_html(text)
    raw = [t.strip(" -–•\t") for t in SENT_SPLIT_RE.split(s) if t and len(t.strip()) > 1]
    return [t[:200] for t in raw][:10]

def brand_like(s: str) -> bool:
    words = [w for w in re.split(r"\s+", s.strip()) if w]
    if not (1 <= len(words) <= 3): return False
    if all(w.lower() in STOPWORDS for w in words): return False
    if not any(any(c.isupper() for c in w) for w in words): return False
    if any(len(w) < 2 or len(w) > 20 for w in words): return False
    if any(re.search(r"[^\w\-&']", w) for w in words): return False
    return True

def guess_brand_from_text(*texts: str) -> str:
    blob = " ".join([str(t or "") for t in texts])
    m = re.search(r"\bbrand\s*[:\-]\s*([A-Za-z0-9&'’\-\s]{2,60})", blob, flags=re.I)
    if m:
        cand = m.group(1).strip(" -–—")
        cand = re.sub(r"\s{2,}", " ", cand)
        return cand if brand_like(cand) else ""
    m = re.search(r"\bby\s+([A-Za-z0-9&'’\-\s]{2,60})", blob, flags=re.I)
    if m:
        cand = m.group(1).strip(" -–—")
        cand = re.sub(r"\s{2,}", " ", cand)
        return cand if brand_like(cand) else ""
    return ""

def tokens_for_keywords(text: str) -> List[str]:
    s = strip_html(text).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in STOPWORDS and (t.isalpha() or t.isalnum()) and len(t) >= 3]
    return toks[:200]

def bigrams(tokens: List[str]) -> List[str]:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]

def keywords_to_bullets(desc: str, item_name: str = "", n: int = 3) -> List[str]:
    toks = tokens_for_keywords(desc + " " + (item_name or ""))
    bi = bigrams(toks)
    from collections import Counter
    bi_cnt = Counter(bi)
    uni_cnt = Counter(toks)
    bullets: List[str] = []
    for phrase, _ in bi_cnt.most_common():
        if len(bullets) >= n: break
        if not phrase.strip(): continue
        bullets.append(phrase.title())
    if len(bullets) < n:
        for w, _ in uni_cnt.most_common():
            if len(bullets) >= n: break
            if w.title() in bullets: continue
            bullets.append(w.title())
    return bullets[:n] if bullets else ["Quality", "Durable", "Comfort"][:n]

def make_generic_keywords(desc: str, name: str, brand: str, colour: str, size: str, material: str, limit_chars: int = 200) -> str:
    base = " ".join([desc or "", name or ""]).strip()
    toks = tokens_for_keywords(base)
    bi = bigrams(toks)
    from collections import Counter
    bi_cnt = Counter(bi)
    uni_cnt = Counter(toks)
    excludes = set(filter(None, [
        (brand or "").lower(), (colour or "").lower(), (size or "").lower(), (material or "").lower()
    ]))
    terms: List[str] = []
    for phrase, _ in bi_cnt.most_common():
        low = phrase.lower()
        if any(x and x in low for x in excludes): continue
        if phrase not in terms: terms.append(phrase)
        if len(terms) >= 12: break
    if len(terms) < 12:
        for w, _ in uni_cnt.most_common():
            if w in excludes: continue
            if w not in terms: terms.append(w)
            if len(terms) >= 18: break
    out, total = [], 0
    for t in terms:
        t = t.strip()
        if not t: continue
        add = (", " if out else "") + t
        if total + len(add) > limit_chars: break
        out.append(t); total += len(add)
    return ", ".join(out)

# --------------------------------------------------------
# Heuristics for attributes & VARIATION TOKENS

COMMON_COLOURS = {
    "black","white","red","blue","green","yellow","orange","purple","pink","brown","grey","gray",
    "navy","beige","ivory","cream","gold","silver","teal","maroon","burgundy","turquoise",
    "lavender","khaki","charcoal","multicolour","multicolor","multicolored"
}
SIZE_TOKENS = ["xxs","xs","s","m","l","xl","xxl","xxxl","2xl","3xl","4xl","one size","onesize"]
APP_SIZE_WORDS = {
    "extra small": "XS", "x small": "XS",
    "small": "S", "medium": "M", "large": "L",
    "extra large": "XL", "x large": "XL"
}
GENDER_MAP = {
    r"\bmen'?s?\b|\bmale\b|\bgents?\b": "Men",
    r"\bwomen'?s?\b|\bfemale\b|\blad(?:y|ies)\b": "Women",
    r"\bunisex\b": "Unisex",
    r"\bboys?\b": "Boys",
    r"\bgirls?\b": "Girls",
    r"\bkids?\b|children": "Kids",
}
MATERIAL_KEYWORDS = [
    "cotton","polyester","leather","wool","silk","linen","denim","spandex","elastane","nylon",
    "viscose","rayon","acrylic","cashmere","fleece","suede","velvet","satin","mesh","lace","rubber","eva","neoprene"
]

# Bed/home textile sizes
BED_SIZE_MAP = {
    "single": "Single",
    "3 quarter": "3 Quarter",
    "three quarter": "3 Quarter",
    "double": "Double",
    "queen": "Queen",
    "king": "King",
    "super king": "Super King",
    "cal king": "Cal King",
    "twin": "Twin",
    "full": "Full",
}
BED_SIZE_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, BED_SIZE_MAP.keys()), key=len, reverse=True)) + r")\b",
    flags=re.I
)

# Packs
PACK_PATTERNS = [
    re.compile(r"\b(pack\s*of\s*|pk\s*of\s*)(\d{1,3})\b", re.I),
    re.compile(r"\b(\d{1,3})\s*(pack|pk|pcs|pieces|ct)\b", re.I),
    re.compile(r"\b(\d{1,3})\s*[x×]\b", re.I),
]
def find_pack_label(text: str) -> str:
    s = strip_html(text).lower()
    for pat in PACK_PATTERNS:
        m = pat.search(s)
        if m:
            nums = [g for g in m.groups() if g and g.isdigit()]
            if nums:
                return f"Pack of {int(nums[-1])}"
    return ""

# Volume/Weight
VOLWT_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(ml|l|litre|liter|g|kg|oz|fl\s*oz|ounce|lb|pound)s?\b",
    re.I
)
def find_volwt_label(text: str) -> str:
    s = strip_html(text)
    m = VOLWT_PATTERN.search(s)
    if not m: return ""
    val, unit = m.group(1), m.group(2).lower().replace(" ", "")
    unit_map = {"litre": "L", "liter": "L", "floz": "fl oz", "ounce": "oz", "pound": "lb"}
    unit = unit_map.get(unit, unit.upper() if unit in {"ml","g","kg","oz","lb"} else unit)
    if unit == "L": return f"{val} L"
    if unit == "fl oz": return f"{val} fl oz"
    return f"{val} {unit}"

# Scents
SCENT_KEYWORDS = {
    "lavender","vanilla","citrus","lemon","orange","rose","jasmine","sandalwood","aloe",
    "tea tree","coconut","peppermint","mango","strawberry","unscented","original","ocean",
    "fresh","apple","cherry","grape","peach"
}
SCENT_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, SCENT_KEYWORDS), key=len, reverse=True)) + r")\b",
    re.I
)

def guess_scent_from_text(*texts: str) -> str:
    blob = " ".join([str(t or "") for t in texts])
    m = re.search(r"\b(scent|fragrance)\s*[:\-]\s*([A-Za-z \-]{2,40})", blob, flags=re.I)
    if m:
        cand = m.group(2).strip()
        return cand.title()
    m = SCENT_PATTERN.search(blob)
    if m:
        return m.group(1).title()
    return ""

# ===== Stronger COLOUR inference =====
def guess_colour_from_text(*texts: str) -> str:
    blob = " ".join([str(t or "") for t in texts]).lower()
    s = re.sub(r"[^a-z0-9/&\-\s]", " ", blob)
    s = re.sub(r"\s+", " ", s).strip()

    combos = [
        (r"\bblue\s*(?:&|and|\/)\s*grey\b", "Blue & Grey"),
        (r"\bblue\s*(?:&|and|\/)\s*gray\b", "Blue & Grey"),
        (r"\bwhite\s*(?:&|and|\/)\s*grey\b", "White & Grey"),
        (r"\bwhite\s*(?:&|and|\/)\s*gray\b", "White & Grey"),
        (r"\bblack\s*(?:&|and|\/)\s*white\b", "Black & White"),
        (r"\bnavy\s*blue\b", "Navy"),
        (r"\boff[\s\-]?white\b", "Offwhite"),
    ]
    for pat, canon in combos:
        if re.search(pat, s, flags=re.I):
            return canon

    tokens = {
        "black":"Black","white":"White","red":"Red","blue":"Blue","green":"Green","yellow":"Yellow",
        "orange":"Orange","purple":"Purple","pink":"Pink","brown":"Brown","grey":"Grey","gray":"Grey",
        "navy":"Navy","beige":"Beige","ivory":"Ivory","cream":"Cream","gold":"Gold","silver":"Silver",
        "teal":"Teal","maroon":"Maroon","burgundy":"Burgundy","turquoise":"Turquoise","lavender":"Lavender",
        "khaki":"Khaki","charcoal":"Charcoal","multicolour":"Multicolour","multicolor":"Multicolour",
        "multicolored":"Multicolour","offwhite":"Offwhite"
    }
    for k, canon in tokens.items():
        if re.search(rf"\b{re.escape(k)}\b", s, flags=re.I):
            return canon
    return ""

def guess_size_from_text(*texts: str) -> str:
    blob = " ".join([str(t or "") for t in texts]).lower().strip()
    for tok in SIZE_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\b", blob):
            return tok.upper().replace("ONESIZE", "One Size")
    for k, v in APP_SIZE_WORDS.items():
        if re.search(rf"\b{re.escape(k)}\b", blob):
            return v
    m = BED_SIZE_PATTERN.search(blob)
    if m:
        key = m.group(1).lower()
        return BED_SIZE_MAP.get(key, key.title())
    volwt = find_volwt_label(blob)
    pack = find_pack_label(blob)
    if volwt and pack: return f"{volwt} ({pack})"
    if volwt: return volwt
    if pack: return pack
    m = re.search(r"\b(size|waist|chest)\s*([0-9]{2,3})\b", blob)
    return m.group(2) if m else ""

def guess_gender_from_text(*texts: str) -> str:
    blob = " ".join([str(t or "") for t in texts]).lower()
    for pat, val in GENDER_MAP.items():
        if re.search(pat, blob): return val
    return ""

def guess_material_from_text(*texts: str) -> str:
    blob = " ".join([str(t or "") for t in texts]).lower()
    hits = [m for m in MATERIAL_KEYWORDS if re.search(rf"\b{re.escape(m)}\b", blob)]
    return hits[0].capitalize() if hits else ""

# --------------------------------------------------------
# Woo helpers (unchanged)
def derive_from_woocommerce(df: pd.DataFrame,
                            use_bullets_from_shortdesc: bool = True,
                            split_images: bool = True) -> pd.DataFrame:
    df = df.copy()
    if split_images and "Images" in df.columns:
        def _split_imgs(s):
            s = str(s or "").strip()
            return [p.strip() for p in s.split(",") if p.strip()] if s else []
        imgs = df["Images"].map(_split_imgs)
        df["Main Image URL"] = imgs.map(lambda L: L[0] if len(L) > 0 else "")
        for i in range(3):
            df[f"Other Image URL{i+1}"] = imgs.map(lambda L, j=i: L[j+1] if len(L) > j+1 else "")
        for i in range(3):
            df[f"Image URL{i+1}"] = df.get(f"Image URL{i+1}", "")
            df[f"Image URL{i+1}"] = df[f"Image URL{i+1}"].where(df[f"Image URL{i+1}"].astype(str).str.strip()!="",
                                                                df[f"Other Image URL{i+1}"])
    if use_bullets_from_shortdesc and "Short description" in df.columns:
        def _bullets(text):
            text = strip_html(text)
            if not text: return [""]*3
            parts = sentences_from_text(text)
            parts = parts[:3] + [""]*3
            return parts[:3]
        b = df["Short description"].map(_bullets)
        for i in range(3):
            df[f"Bullet Point {i+1}"] = b.map(lambda L, j=i: L[j])
    L, W, H = df.get("Length (cm)"), df.get("Width (cm)"), df.get("Height (cm)")
    if L is not None or W is not None or H is not None:
        def _dim(row):
            vals = [str(row.get(k, "")).strip() for k in ["Length (cm)","Width (cm)","Height (cm)"]]
            vals = [v for v in vals if v]; return " x ".join(vals)
        df["Package Dimensions in CM"] = df.apply(_dim, axis=1)
    if "Type" in df.columns:
        t = df["Type"].astype(str).str.lower()
        df["Parent_Child"] = t.map(lambda x: "child" if x == "variation" else ("parent" if x == "variable" else ""))
    attr_name_cols  = [c for c in df.columns if re.fullmatch(r"Attribute \d+ name", c)]
    attr_value_cols = [c for c in df.columns if re.fullmatch(r"Attribute \d+ value\(s\)", c)]
    name_idx = {int(re.findall(r"\d+", c)[0]): c for c in attr_name_cols}
    value_idx = {int(re.findall(r"\d+", c)[0]): c for c in attr_value_cols}
    def _extract_attr(row, target_names):
        for i in sorted(set(name_idx.keys()) & set(value_idx.keys())):
            label = str(row.get(name_idx[i], "")).strip().lower()
            if label in target_names: return str(row.get(value_idx[i], "")).strip()
        return ""
    if "Colour" not in df.columns:
        df["Colour"] = df.apply(lambda r: _extract_attr(r, {"color","colour","pa_color"}), axis=1)
    if "Size" not in df.columns:
        df["Size"] = df.apply(lambda r: _extract_attr(r, {"size","pa_size"}), axis=1)
    if "Material Type" not in df.columns:
        df["Material Type"] = df.apply(lambda r: _extract_attr(r, {"material","fabric","material type"}), axis=1)
    if "Gender" not in df.columns:
        df["Gender"] = df.apply(lambda r: _extract_attr(r, {"gender","target gender"}), axis=1)
    if "Variation_theme" not in df.columns:
        has_color = df["Colour"].astype(str).str.strip().ne("")
        has_size  = df["Size"].astype(str).str.strip().ne("")
        df["Variation_theme"] = (has_color & has_size).map({True:"SizeName-ColorName", False:""})
        df.loc[(~has_color) & has_size,  "Variation_theme"] = "SizeName"
        df.loc[has_color & (~has_size), "Variation_theme"] = "ColorName"
    if "Standard Price" not in df.columns and "Regular price" in df.columns:
        df["Standard Price"] = df["Regular price"]
    if "Quantity" not in df.columns:
        for cand in ["Stock quantity","Stock","qty","quantity"]:
            if cand in df.columns: df["Quantity"] = df[cand]; break
        else:
            df["Quantity"] = ""
    if "Item Name" not in df.columns and "Name" in df.columns:
        df["Item Name"] = df["Name"]
    if "Product Description" not in df.columns and "Description" in df.columns:
        df["Product Description"] = df["Description"]
    if "External Product ID" not in df.columns:
        if "GTIN, UPC, EAN, or ISBN" in df.columns:
            df["External Product ID"] = df["GTIN, UPC, EAN, or ISBN"]
        else:
            for cand in ["EAN","UPC","Barcode","Meta: _ean","Meta: _barcode"]:
                if cand in df.columns: df["External Product ID"] = df[cand]; break
            else:
                df["External Product ID"] = ""
    if "Colour" in df.columns:
        df["Colour"] = df.apply(
            lambda r: (r["Colour"] if str(r.get("Colour","")).strip()
                       else guess_colour_from_text(r.get("Product Description",""),
                                                   r.get("Short description",""),
                                                   r.get("Item Name",""))),
            axis=1
        )
    if "Brand" not in df.columns:
        df["Brand"] = df.apply(
            lambda r: guess_brand_from_text(r.get("Product Description",""), r.get("Short description",""), r.get("Item Name","")),
            axis=1
        )
    return df

def first_nonempty(*vals: str) -> str:
    for v in vals:
        v = str(v or "").strip()
        if v: return v
    return ""

def force_fill_external_id(df: pd.DataFrame, barcodes_available: str) -> pd.DataFrame:
    df = df.copy()
    ext_cols = [c for c in df.columns if canonicalize(c) == "externalproductid"]
    if not ext_cols:
        return df
    for c in ext_cols:
        if barcodes_available == "No":
            df[c] = "GTIN EXEMPT"
        else:
            df[c] = (
                df[c]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace({"": "GTIN EXEMPT"})
            )
    return df

def infer_template_schema(df_raw: pd.DataFrame) -> Tuple[List[str], Dict[str, bool]]:
    header_row_idx = None
    for i in range(min(15, len(df_raw))):
        non_empty = (df_raw.iloc[i].astype(str).str.strip() != "").sum()
        if non_empty >= 3:
            header_row_idx = i; break
    if header_row_idx is None: header_row_idx = 0
    raw_headers = list(df_raw.iloc[header_row_idx])
    columns: List[str] = []
    required: Dict[str, bool] = {}
    for x in raw_headers:
        cell = str(x or "").strip()
        inline_req = bool(RE_REQUIRED_PAT.search(cell))
        col_clean = clean_header_name(cell)
        if col_clean and str(col_clean).lower() != "nan":
            columns.append(col_clean); required[col_clean] = inline_req
        else:
            columns.append(""); required[""] = False
    while columns and columns[-1] == "":
        last = columns.pop(); required.pop(last, None)
    req_idx = header_row_idx + 1
    if req_idx < len(df_raw):
        req_row = [str(x).strip().lower() for x in list(df_raw.iloc[req_idx][:len(columns)])]
        for col, cell in zip(columns, req_row):
            if not col: continue
            if any(k in cell for k in ["required","mandatory","compulsory"]): required[col] = True
            elif cell in ["yes","y","true","must","needed"]: required[col] = True
    return columns, required

def best_match(target_col: str, src_cols: List[str]) -> Tuple[str, float]:
    t_norm = canonicalize(target_col)
    src_norms = [canonicalize(c) for c in src_cols]
    if t_norm in src_norms:
        idx = src_norms.index(t_norm); return src_cols[idx], 1.0
    m = re.match(r"^gtin(\d+)$", t_norm)
    if m and "gtin" in src_norms:
        idx = src_norms.index("gtin"); return src_cols[idx], 0.9
    fam = re.match(r"^(imageurl|bulletpoint|bullet point)(\d+)$", t_norm)
    if fam:
        for i, sn in enumerate(src_norms):
            if sn == t_norm:
                return src_cols[i], 0.95
    close = difflib.get_close_matches(t_norm, src_norms, n=1, cutoff=0.75)
    if close:
        idx = src_norms.index(close[0])
        conf = difflib.SequenceMatcher(None, t_norm, src_norms[idx]).ratio()
        return src_cols[idx], float(conf)
    return "", 0.0

def map_dataframe_to_template(
    df_source: pd.DataFrame,
    template_cols: List[str],
    required_flags: Dict[str, bool],
    mapping_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    src_cols = list(df_source.columns)
    preferred_first = [
        "Product Description","Description","Short description",
        "Bullet Point 1","Bullet Point 2","Bullet Point 3","Bullet Point 4","Bullet Point 5",
        "Main Image URL","Other Image URL1","Other Image URL2","Other Image URL3",
        "Image URL1","Image URL2","Image URL3","Image URL4","Image URL5",
    ]
    preferred_first = [c for c in preferred_first if c in src_cols]
    src_cols = preferred_first + [c for c in src_cols if c not in preferred_first]

    mapping_rows = []
    selected_sources: Dict[str, Tuple[str, float]] = {}
    for tcol in template_cols:
        if mapping_overrides and tcol in mapping_overrides:
            src = mapping_overrides[tcol] or ""
            conf = 1.0 if src else 0.0
        else:
            src, conf = best_match(tcol, src_cols)
        if src and src not in df_source.columns:
            src, conf = "", 0.0
        selected_sources[tcol] = (src, conf)
        mapping_rows.append({
            "template_column": tcol,
            "mapped_from": src,
            "confidence": round(conf, 3),
            "required": required_flags.get(tcol, False),
        })
    mapping_df = pd.DataFrame(mapping_rows)

    out = pd.DataFrame(index=df_source.index)
    for tcol in template_cols:
        src, _ = selected_sources[tcol]
        if src:
            out[tcol] = (
                df_source[src]
                .astype(str)
                .replace({np.nan: ""})
                .map(lambda x: x.strip())
            )
        else:
            out[tcol] = ""

    req_cols = [c for c, is_req in required_flags.items() if is_req and c in out.columns]
    if req_cols:
        req_mask = (out[req_cols].astype(str).apply(lambda s: s.str.strip() == "")).any(axis=1)
        ready_df = out.loc[~req_mask].copy()
        fix_df = out.loc[req_mask].copy()
        fix_df["__missing_required_fields"] = fix_df.apply(
            lambda row: ",".join([c for c in req_cols if str(row[c]).strip() == ""]), axis=1
        )
    else:
        ready_df = out.copy()
        fix_df = pd.DataFrame(columns=list(out.columns) + ["__missing_required_fields"])
    return ready_df, fix_df, mapping_df

# ============= NEW: Parent→Children sequential propagation =============
def propagate_parent_details_in_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    In display order: when we hit a 'parent', cache its text + images.
    For each following 'child' until the next parent, overwrite child with
    the parent's: Description, Generic Keywords, Bullet Points 1–5, and Images.
    Also backfill Parent SKU + Relationship Type if blank.
    """
    df = df.copy()

    core_cols = [
        "Generic Keywords",
        "Product Description", "Description",
        "Bullet Point 1", "Bullet Point 2", "Bullet Point 3",
        "Bullet Point 4", "Bullet Point 5",
    ]
    img_cols = ["Main Image URL"] + [f"Image URL{i}" for i in range(1, 6)] + [f"Other Image URL{i}" for i in range(1, 4)]

    for c in core_cols + img_cols:
        if c not in df.columns:
            df[c] = ""

    if "Parent_Child" not in df.columns:
        return df
    if "Parent SKU" not in df.columns: df["Parent SKU"] = ""
    if "Relationship Type" not in df.columns: df["Relationship Type"] = ""

    last_parent_payload = None
    last_parent_sku = ""

    for idx in df.index:
        role = str(df.at[idx, "Parent_Child"]).strip().lower()

        if role == "parent":
            # choose the best description field and mirror to both columns
            desc = first_nonempty(
                df.at[idx, "Product Description"] if "Product Description" in df.columns else "",
                df.at[idx, "Description"] if "Description" in df.columns else "",
                df.at[idx, "Short description"] if "Short description" in df.columns else ""
            )
            payload = {c: str(df.at[idx, c]) for c in core_cols + img_cols}
            if desc:
                payload["Product Description"] = desc
                payload["Description"] = desc
            last_parent_payload = payload
            last_parent_sku = str(df.at[idx, "SKU"]) if "SKU" in df.columns else ""
            continue

        if role == "child" and last_parent_payload:
            for c in core_cols + img_cols:
                pval = last_parent_payload.get(c, "")
                if pval != "":
                    df.at[idx, c] = pval  # always overwrite with parent's value

            if not str(df.at[idx, "Parent SKU"]).strip() and last_parent_sku:
                df.at[idx, "Parent SKU"] = last_parent_sku
            if not str(df.at[idx, "Relationship Type"]).strip():
                df.at[idx, "Relationship Type"] = "variation"

    return df
# ======================================================================

# ============================
# Streamlit App
# ============================

st.set_page_config(page_title="AM Feed → MLH Template Mapper", page_icon="⛮", layout="wide")

st.title("AM Feed → MLH Template Mapper")
st.caption("Upload a messy seller/AM file, pick a template, review/override column mapping, then export **READY** & **FIX_ME** Excel files.")

with st.sidebar:
    st.header("Template Selection")
    template_choice = st.selectbox(
        "Choose Amazon template",
        options=["Hardlines (OHL)", "Softlines", "Consumables", "Consumer Electronics (CE)"],
        index=1,
    )
    st.markdown("**Upload the matching template Excel** (contains header row + Required row).")
    template_file = st.file_uploader("Template Excel (.xlsx)", type=["xlsx"], key="tmpl")
    default_paths = {
        "Hardlines (OHL)": "MLH_Listings_Template - OHL.xlsx",
        "Softlines": "MLH_Listings_Template - Softlines.xlsx",
        "Consumables": "MLH_Listings_Template - Consumables.xlsx",
        "Consumer Electronics (CE)": "MLH_Listings_Template+-+CE.xlsx",
    }
    use_default = False
    if not template_file and os.path.exists(default_paths.get(template_choice, "")):
        use_default = st.checkbox("Use default template found on disk", value=True)

# ============================
# 1) Upload + SAFE READ BLOCK
# ============================

st.subheader("1) Upload source file (messy vendor feed)")
src = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])

# 🚧 Hard guard: don't touch `src` if it's None
if src is None:
    st.info("Upload a CSV/XLSX to begin.")
    st.stop()

st.markdown("**Header Row (source file)**")
header_row = st.number_input(
    "If your source header isn’t the first row, set its 0-based index here:",
    min_value=0, max_value=100, value=0, step=1
)

# --- Read safely (rewindable buffer + encoding/delimiter fallbacks) ---
raw_bytes = src.read()
buf = io.BytesIO(raw_bytes)

def _read_csv_with_fallbacks(b: io.BytesIO, header_idx: int) -> pd.DataFrame:
    """
    Robust CSV reader:
      - tries encodings: utf-8, utf-8-sig, cp1252, latin1, utf-16, utf-16-le, utf-16-be
      - tries delimiter sniffing (None) and common separators
      - final fallback: decode bytes with replacement (cp1252 -> StringIO)
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16", "utf-16-le", "utf-16-be"]
    delimiters = [None, ",", ";", "\t", "|"]  # None = sniff (engine="python")

    for enc in encodings:
        for sep in delimiters:
            b.seek(0)
            try:
                return pd.read_csv(
                    b,
                    dtype=str,
                    keep_default_na=False,
                    header=header_idx,
                    sep=sep,
                    engine="python",
                    encoding=enc,
                )
            except UnicodeDecodeError:
                # wrong encoding, try next
                continue
            except Exception:
                # wrong delimiter/other parse issue, try next combo
                continue

    # Final fallback: decode to text with replacement characters and parse
    b.seek(0)
    txt = b.read().decode("cp1252", errors="replace")
    return pd.read_csv(
        io.StringIO(txt),
        dtype=str,
        keep_default_na=False,
        header=header_idx,
        sep=None,          # sniff again on the decoded text
        engine="python",
    )

def _looks_like_xlsx(b: bytes) -> bool:
    # XLSX (ZIP) magic header: PK\x03\x04
    return len(b) >= 4 and b[:4] == b"PK\x03\x04"

def _looks_like_xls(b: bytes) -> bool:
    # Legacy XLS (CFBF) header: D0 CF 11 E0 A1 B1 1A E1
    return len(b) >= 8 and b[:8] == bytes.fromhex("D0CF11E0A1B11AE1")

# Use a safe filename (some uploaders can be nameless/None)
safe_name = (getattr(src, "name", "") or "").lower()

try:
    if safe_name.endswith(".csv"):
        # CSV by extension
        df_source = _read_csv_with_fallbacks(buf, header_row)
    elif safe_name.endswith((".xlsx", ".xls")) or _looks_like_xlsx(raw_bytes) or _looks_like_xls(raw_bytes):
        # Excel by extension or by signature
        buf.seek(0)
        df_source = pd.read_excel(buf, dtype=str, header=header_row)
    else:
        # Unknown/nameless: try Excel by signature, otherwise CSV fallbacks
        if _looks_like_xlsx(raw_bytes) or _looks_like_xls(raw_bytes):
            buf.seek(0)
            df_source = pd.read_excel(buf, dtype=str, header=header_row)
        else:
            df_source = _read_csv_with_fallbacks(buf, header_row)
except Exception as e:
    st.error("Couldn’t read the file. Error below:")
    st.exception(e)
    st.stop()

# Clean up dataframe
df_source = df_source.dropna(how="all", axis=1).fillna("")

# ==============================================
# 1b) Split any "combined bullets" into columns
# ==============================================
BULLET_COL_HINT_RE = re.compile(r"\b(bullets?|bullet\s*points?|key\s*features?|highlights?|features)\b", re.I)

def split_bullets_cell(text: str, max_pts: int = 5) -> List[str]:
    """
    Split a messy string cell into up to `max_pts` bullet points.
    Handles •, ;, |, newlines, and dash-led items. Always returns a list.
    Coerces to str to avoid 'argument of type method is not iterable' etc.
    """
    try:
        s = "" if text is None else str(text)
    except Exception:
        s = ""
    if not s.strip():
        return []

    # Normalize escaped newlines
    s = s.replace("\\n", "\n")

    # Split on common bullet delimiters: •, newlines, semicolons, pipes
    parts = re.split(r"[•\u2022]+|\n|\r|;|\|", s)

    # Also split on dash-led items (" - " and leading "- ")
    more = []
    for p in parts:
        try:
            p_str = str(p)
        except Exception:
            p_str = ""
        more.extend(re.split(r"(?:\s+-\s+)|(?:^\s*-\s+)", p_str))

    # Clean + keep nontrivial items
    out = []
    for t in more:
        try:
            t_str = str(t)
        except Exception:
            t_str = ""
        t_str = strip_html(t_str).strip(" -–—•\t")
        if len(t_str) >= 2:
            out.append(t_str)

    # Dedupe while preserving order
    seen = set()
    deduped = []
    for t in out:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return deduped[:max_pts]

# 1) Prefer headers that look like bullet containers
candidate_cols = [c for c in df_source.columns if BULLET_COL_HINT_RE.search(c)]

# 2) If none found, scan columns whose content often splits into >=2 items
if not candidate_cols and len(df_source):
    best_col, best_ratio = None, 0.0
    sample_n = min(len(df_source), 200)
    for c in df_source.columns:
        # skip obvious numeric-only columns
        try:
            if df_source[c].astype(str).str.fullmatch(r"\s*\d+(\.\d+)?\s*").mean() > 0.7:
                continue
        except Exception:
            pass
        col_series = df_source[c].astype(str).head(sample_n)
        # proportion of rows yielding 2+ bullet items
        hits = col_series.map(lambda v: len(split_bullets_cell(v)) >= 2).mean()
        if hits > best_ratio:
            best_ratio, best_col = hits, c
    if best_col and best_ratio >= 0.30:
        candidate_cols = [best_col]

# Now, if we found a likely "combined bullets" column, populate Bullet Point 1–5 (only if empty)
if candidate_cols:
    bullet_src_col = candidate_cols[0]
    # ensure target columns exist
    for i in range(1, 6):
        colname = f"Bullet Point {i}"
        if colname not in df_source.columns:
            df_source[colname] = ""

    def _apply_bullet_split(row: pd.Series) -> pd.Series:
        # If we already have at least 1–3 bullets, leave row as-is
        try:
            already = all(str(row.get(f"Bullet Point {i}", "")).strip() for i in range(1, 4))
        except Exception:
            already = False
        if already:
            return row

        pieces = split_bullets_cell(row.get(bullet_src_col, ""))
        if pieces:
            for i in range(5):
                tgt = f"Bullet Point {i+1}"
                if i < len(pieces) and not str(row.get(tgt, "")).strip():
                    row[tgt] = pieces[i][:200]  # cap length lightly
        return row

    df_source = df_source.apply(_apply_bullet_split, axis=1)


    # ------------------------------
    # NEW: Split "combined bullets" into Bullet Point 1–5
    # ------------------------------
    BULLET_COL_HINT_RE = re.compile(r"\b(bullets?|bullet\s*points?|key\s*features?|highlights?|features)\b", re.I)

    def split_bullets_cell(text: str, max_pts: int = 5) -> List[str]:
        """
        Split a messy string cell into up to `max_pts` bullet points.
        Handles •, ;, |, newlines, and dash-led items. Always returns a list.
        Hardened to avoid 'argument of type method is not iterable' by coercing to str.
        """
        try:
            s = "" if text is None else str(text)
        except Exception:
            s = ""
        if not s.strip():
            return []

        s = s.replace("\\n", "\n")
        parts = re.split(r"[•\u2022]+|\n|\r|;|\|", s)

        more = []
        for p in parts:
            try:
                p_str = str(p)
            except Exception:
                p_str = ""
            more.extend(re.split(r"(?:\s+-\s+)|(?:^\s*-\s+)", p_str))

        out = []
        for t in more:
            try:
                t_str = str(t)
            except Exception:
                t_str = ""
            t_str = strip_html(t_str).strip(" -–—•\t")
            if len(t_str) >= 2:
                out.append(t_str)

        seen = set()
        deduped = []
        for t in out:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        return deduped[:max_pts]

    # 1) Prefer headers that look like bullet containers
    candidate_cols = [c for c in df_source.columns if BULLET_COL_HINT_RE.search(c)]

    # 2) If none found, scan columns whose content often splits into >=2 items
    if not candidate_cols:
        best_col, best_ratio = None, 0.0
        sample_n = min(len(df_source), 200)
        for c in df_source.columns:
            # skip obvious numeric-only columns
            if df_source[c].astype(str).str.fullmatch(r"\s*\d+(\.\d+)?\s*").mean() > 0.7:
                continue
            col_series = df_source[c].astype(str).head(sample_n)
            # proportion of rows yielding 2+ bullet items
            hits = col_series.map(lambda v: len(split_bullets_cell(v)) >= 2).mean()
            if hits > best_ratio:
                best_ratio, best_col = hits, c
        if best_col and best_ratio >= 0.30:
            candidate_cols = [best_col]

    # Now, if we found a likely "combined bullets" column, populate Bullet Point 1–5 (only if empty)
    if candidate_cols:
        bullet_src_col = candidate_cols[0]
        # ensure target columns exist
        for i in range(1, 6):
            colname = f"Bullet Point {i}"
            if colname not in df_source.columns:
                df_source[colname] = ""

        def _apply_bullet_split(row):
            if all(str(row.get(f"Bullet Point {i}", "")).strip() for i in range(1, 4)):
                return row
            pieces = split_bullets_cell(row.get(bullet_src_col, ""))
            if pieces:
                for i in range(5):
                    tgt = f"Bullet Point {i+1}"
                    if i < len(pieces) and not str(row.get(tgt, "")).strip():
                        row[tgt] = pieces[i][:200]
            return row

        df_source = df_source.apply(_apply_bullet_split, axis=1)

    # === Seller / Manufacturer inputs ===
    st.markdown("### Seller / Manufacturer")
    seller_is_manufacturer = st.checkbox("Seller is the manufacturer", value=False)
    seller_name = ""
    if seller_is_manufacturer:
        seller_name = st.text_input("Seller/Manufacturer name to use:", value="", placeholder="e.g., Acme Apparel")

    # --- Barcodes available? ---
    barcodes_available = st.selectbox(
        "Barcodes available in this feed?",
        options=["No", "Yes"],
        index=0,
        help="If 'No', External Product ID will be auto-filled as GTIN EXEMPT."
    )

    # Woo derives
    use_derives = st.checkbox("Derive bullets from Short description & split Images (WooCommerce)", value=True)
    if use_derives and (("Images" in df_source.columns) or ("Short description" in df_source.columns) or ("GTIN, UPC, EAN, or ISBN" in df_source.columns)):
        df_source = derive_from_woocommerce(df_source, True, True)

    # --- Clean descriptions & mine dims ---
    for col in ["Product Description", "Short description", "Description"]:
        if col in df_source.columns:
            df_source[col] = df_source[col].map(strip_html)

    if "Package Dimensions in CM" not in df_source.columns:
        df_source["Package Dimensions in CM"] = ""
    desc_for_dims = df_source.get("Product Description", df_source.get("Description", pd.Series([""]*len(df_source))))
    dim_from_desc = desc_for_dims.map(extract_dimensions_cm)
    mask_empty_dims = df_source["Package Dimensions in CM"].astype(str).str.strip().eq("")
    df_source.loc[mask_empty_dims, "Package Dimensions in CM"] = dim_from_desc

    # --- Brand & Manufacturer (safe) ---
    if "Brand" not in df_source.columns:
        df_source["Brand"] = ""
    def _brand_row(row):
        if seller_is_manufacturer and seller_name.strip():
            return seller_name.strip()
        b = str(row.get("Brand","")).strip()
        if b:
            return b
        cand = guess_brand_from_text(row.get("Product Description",""), row.get("Short description",""), row.get("Item Name",""))
        return cand if cand else "Generic"
    df_source["Brand"] = df_source.apply(_brand_row, axis=1)
    df_source["Manufacturer"] = df_source["Brand"]

    # Ensure key attribute columns exist up-front
    for col in ["Size", "Gender", "Material Type", "Colour", "Scent"]:
        if col not in df_source.columns:
            df_source[col] = ""

    # --- External Product ID ---
    if "External Product ID" not in df_source.columns:
        df_source["External Product ID"] = ""
    if barcodes_available == "No":
        df_source["External Product ID"] = "GTIN EXEMPT"
    else:
        barcode_like_norms = {
            "ean","ean13","ean_13","upc","barcode","bar code","gtin","jan","isbn",
            "meta:_ean","meta:_barcode","meta:_sku_barcode","gtin, upc, ean, or isbn"
        }
        def _fill_barcode(row):
            val = str(row.get("External Product ID","") or "").strip()
            if val: return val
            for col in row.index:
                norm = normalize_for_match(col)
                if norm in {normalize_for_match(x) for x in barcode_like_norms}:
                    v = str(row[col] or "").strip()
                    if v: return v
            return ""
        df_source["External Product ID"] = df_source.apply(_fill_barcode, axis=1)
        df_source["External Product ID"] = df_source["External Product ID"].fillna("").astype(str).str.strip()
        df_source.loc[df_source["External Product ID"] == "", "External Product ID"] = "GTIN EXEMPT"

    # --- SKU fallback ---
    if "SKU" not in df_source.columns:
        df_source["SKU"] = ""
    df_source["SKU"] = df_source["SKU"].fillna("").astype(str).str.strip()
    df_source.loc[df_source["SKU"] == "", "SKU"] = "(create new SKU)"

    # --- Ensure image columns exist up-front ---
    for c in ["Main Image URL"] + [f"Image URL{i}" for i in range(1,6)] + [f"Other Image URL{i}" for i in range(1,4)]:
        if c not in df_source.columns: df_source[c] = ""

    # --- Robust image URL harvesting ---
    IMG_HINT_RE = re.compile(r"(image|img|picture|photo|gallery|thumbnail|url)", re.I)
    URL_RE = re.compile(r"(https?://[^\s,;|'\"<>]+)")
    def _parse_urls(val: str) -> List[str]:
        s = str(val or "")
        urls = URL_RE.findall(s)
        parts = re.split(r"[,\|\s]+", s)
        urls += [p for p in parts if p.startswith("http://") or p.startswith("https://")]
        seen, out = set(), []
        for u in urls:
            u = u.strip()
            if not u or u in seen: continue
            seen.add(u); out.append(u)
        return out

    img_cols = [c for c in df_source.columns if IMG_HINT_RE.search(c) and c not in {"Main Image URL"} | {f"Image URL{i}" for i in range(1,6)} | {f"Other Image URL{i}" for i in range(1,4)}]

    def _collect_row_imgs(row) -> List[str]:
        urls: List[str] = []
        if str(row.get("Main Image URL","")).strip():
            urls.append(str(row.get("Main Image URL","")).strip())
        for i in range(1,6):
            v = str(row.get(f"Image URL{i}","")).strip()
            if v: urls.append(v)
        for i in range(1,4):
            v = str(row.get(f"Other Image URL{i}","")).strip()
            if v: urls.append(v)
        for c in img_cols:
            urls += _parse_urls(row.get(c, ""))
        seen, out = set(), []
        for u in urls:
            if u and u not in seen:
                seen.add(u); out.append(u)
        return out

    df_source["_img_list"] = df_source.apply(_collect_row_imgs, axis=1)

    # Fill main + Image URL1..5 if empty
    df_source["Main Image URL"] = df_source.apply(
        lambda r: (r["Main Image URL"] if str(r["Main Image URL"]).strip() else (r["_img_list"][0] if r["_img_list"] else "")),
        axis=1
    )
    for i in range(1,6):
        df_source[f"Image URL{i}"] = df_source.apply(
            lambda r, j=i: (r[f"Image URL{j}"] if str(r[f"Image URL{j}"]).strip()
                            else (r["_img_list"][j] if len(r["_img_list"]) > j else "")),
            axis=1
        )
    # If all empty, set placeholder
    img_all_empty = (
        (df_source["Main Image URL"].astype(str).str.strip() == "") &
        (df_source["Image URL1"].astype(str).str.strip() == "") &
        (df_source["Image URL2"].astype(str).str.strip() == "") &
        (df_source["Image URL3"].astype(str).str.strip() == "") &
        (df_source["Image URL4"].astype(str).str.strip() == "") &
        (df_source["Image URL5"].astype(str).str.strip() == "")
    )
    df_source.loc[img_all_empty, "Main Image URL"] = "(seller will update when listed on lens)"

    # --- Colour, Size, Gender, Material, Scent (fill-ins) ---
    df_source["Colour"] = df_source.apply(
        lambda r: (r["Colour"] if str(r.get("Colour","")).strip()
                   else guess_colour_from_text(r.get("Item Name",""), r.get("Product Description",""), r.get("Short description",""))),
        axis=1
    )
    df_source["Size"] = df_source.apply(
        lambda r: (str(r.get("Size","")).strip() or
                   guess_size_from_text(r.get("Item Name",""), r.get("Short description",""), r.get("Product Description",""))),
        axis=1
    )
    df_source["Gender"] = df_source.apply(
        lambda r: (str(r.get("Gender","")).strip() or
                   guess_gender_from_text(r.get("Item Name",""), r.get("Short description",""), r.get("Product Description","")) or
                   "Unisex"),
        axis=1
    )
    df_source["Material Type"] = df_source.apply(
        lambda r: (str(r.get("Material Type","")).strip() or
                   guess_material_from_text(r.get("Item Name",""), r.get("Short description",""), r.get("Product Description",""))),
        axis=1
    )
    if "Scent" not in df_source.columns:
        df_source["Scent"] = ""
    df_source["Scent"] = df_source.apply(
        lambda r: (str(r.get("Scent","")).strip() or
                   guess_scent_from_text(r.get("Item Name",""), r.get("Short description",""), r.get("Product Description",""))),
        axis=1
    )

    # --- Bullet Points 1–5 (ensure cols exist; auto-generate 1–3) ---
    for c in ["Bullet Point 1","Bullet Point 2","Bullet Point 3","Bullet Point 4","Bullet Point 5"]:
        if c not in df_source.columns: df_source[c] = ""
    def _ensure_bullets(row):
        desc = first_nonempty(row.get("Product Description",""), row.get("Short description",""), row.get("Description",""))
        name = str(row.get("Item Name",""))
        if not (str(row.get("Bullet Point 1","")).strip() and str(row.get("Bullet Point 2","")).strip() and str(row.get("Bullet Point 3","")).strip()):
            b1,b2,b3 = keywords_to_bullets(desc, name, n=3)
            if not str(row["Bullet Point 1"]).strip(): row["Bullet Point 1"] = b1
            if not str(row["Bullet Point 2"]).strip(): row["Bullet Point 2"] = b2
            if not str(row["Bullet Point 3"]).strip(): row["Bullet Point 3"] = b3
        for key in ["Bullet Point 1","Bullet Point 2","Bullet Point 3"]:
            words = re.split(r"\s+", str(row[key]).strip())
            row[key] = " ".join(words[:3])
        return row
    df_source = df_source.apply(_ensure_bullets, axis=1)

    # --- Generic Keywords (auto) ---
    if "Generic Keywords" not in df_source.columns:
        df_source["Generic Keywords"] = ""
    df_source["Generic Keywords"] = df_source.apply(
        lambda r: (str(r.get("Generic Keywords","")).strip() or
                   make_generic_keywords(
                       desc=str(r.get("Product Description","")),
                       name=str(r.get("Item Name","")),
                       brand=str(r.get("Brand","")),
                       colour=str(r.get("Colour","")),
                       size=str(r.get("Size","")),
                       material=str(r.get("Material Type","")),
                       limit_chars=200
                   )),
        axis=1
    )

    # --- VARIATIONS (word-boundary safe) ---
    st.subheader("Variations")

    TITLE_CANDS = ["Item Name","Item name","Name","Product Name","Product name","Product Title","Title"]
    title_col = next((c for c in TITLE_CANDS if c in df_source.columns), None)

    variations_present = st.checkbox(
        "This feed contains variations (same product with size/color/etc.)",
        value=True,
        help="Groups by a base title (removing size, color, pack, volume/weight, scent terms) and copies description/bullets/images from the parent."
    )

    def _alts_with_word_boundaries(terms):
        terms = [t for t in set(terms) if t.strip()]
        if not terms:
            return r"$a^"
        parts = [re.escape(t) for t in sorted(terms, key=len, reverse=True)]
        return r"\b(?:%s)\b" % "|".join(parts)

    if variations_present:
        if title_col:
            colour_terms = list(COMMON_COLOURS) + ["blue & grey","blue and grey"]
            colour_pat = _alts_with_word_boundaries(colour_terms)
            size_terms = SIZE_TOKENS + list(BED_SIZE_MAP.keys()) + list(APP_SIZE_WORDS.keys())
            size_pat  = _alts_with_word_boundaries(size_terms)
            scent_pat = _alts_with_word_boundaries(SCENT_KEYWORDS)
            pack_pat  = r"(?:\bpack\s*of\s*\d{1,3}\b|\b\d{1,3}\s*(?:pack|pk|pcs|pieces|ct)\b|\b\d{1,3}\s*[x×]\b)"
            volwt_pat = r"(?:\b\d+(?:\.\d+)?\s*(?:ml|l|litre|liter|g|kg|oz|fl\s*oz|ounce|lb|pound)s?\b)"
            REMOVE_PAT = re.compile("|".join([colour_pat, size_pat, scent_pat, pack_pat, volwt_pat]), flags=re.I)

            def base_title_key(s: str) -> str:
                s0 = strip_html(str(s or "")).lower()
                s1 = REMOVE_PAT.sub(" ", s0)
                s1 = re.sub(r"\b(size|waist|chest)\s*\d{2,3}\b", " ", s1, flags=re.I)
                s1 = re.sub(r"\([^\)]*\)$", " ", s1)
                s1 = re.sub(r"\[[^\]]*\]$", " ", s1)
                s1 = re.sub(r"\s+", " ", s1).strip(" -–—")
                return " ".join(s1.split()[:14])

            def variation_hits(name: str) -> int:
                s = str(name or "").lower()
                hits = 0
                hits += len(re.findall(colour_pat, s, flags=re.I))
                hits += len(re.findall(size_pat,   s, flags=re.I))
                hits += len(re.findall(scent_pat,  s, flags=re.I))
                hits += len(re.findall(pack_pat,   s, flags=re.I))
                hits += len(re.findall(volwt_pat,  s, flags=re.I))
                hits += len(re.findall(r"\b(size|waist|chest)\s*\d{2,3}\b", s, flags=re.I))
                return hits

            def parent_sort_key(row):
                nm = str(row.get(title_col, ""))
                return (variation_hits(nm), len(nm), 0 if str(row.get("Product Description","")).strip() else 1)

            for col in ["Parent_Child", "Parent SKU", "Relationship Type", "Variation_theme"]:
                if col not in df_source.columns:
                    df_source[col] = ""

            df_source["_var_key"] = df_source[title_col].map(base_title_key)

            if st.checkbox("Show variation debug keys", value=False):
                st.dataframe(df_source[[title_col, "_var_key"]].head(50), use_container_width=True)

            for key, g in df_source.groupby("_var_key"):
                if key == "" or len(g) <= 1:
                    continue

                parent_idx = sorted(g.index, key=lambda ix: parent_sort_key(df_source.loc[ix]))[0]
                child_idx  = [ix for ix in g.index if ix != parent_idx]

                df_source.loc[parent_idx, "Parent_Child"] = "parent"
                df_source.loc[child_idx,  "Parent_Child"] = "child"

                p_desc = str(df_source.at[parent_idx, "Product Description"]) if "Product Description" in df_source.columns else ""
                for c in ["Bullet Point 1","Bullet Point 2","Bullet Point 3"]:
                    if c not in df_source.columns: df_source[c] = ""
                p_b1 = str(df_source.at[parent_idx, "Bullet Point 1"])
                p_b2 = str(df_source.at[parent_idx, "Bullet Point 2"])
                p_b3 = str(df_source.at[parent_idx, "Bullet Point 3"])

                if "Product Description" in df_source.columns and p_desc:
                    df_source.loc[child_idx, "Product Description"] = df_source.loc[child_idx, "Product Description"] \
                        .where(df_source.loc[child_idx, "Product Description"].astype(str).str.strip() != "", p_desc)
                if p_b1:
                    df_source.loc[child_idx, "Bullet Point 1"] = df_source.loc[child_idx, "Bullet Point 1"] \
                        .where(df_source.loc[child_idx, "Bullet Point 1"].astype(str).str.strip() != "", p_b1)
                if p_b2:
                    df_source.loc[child_idx, "Bullet Point 2"] = df_source.loc[child_idx, "Bullet Point 2"] \
                        .where(df_source.loc[child_idx, "Bullet Point 2"].astype(str).str.strip() != "", p_b2)
                if p_b3:
                    df_source.loc[child_idx, "Bullet Point 3"] = df_source.loc[child_idx, "Bullet Point 3"] \
                        .where(df_source.loc[child_idx, "Bullet Point 3"].astype(str).str.strip() != "", p_b3)

                for col in ["Main Image URL"] + [f"Image URL{i}" for i in range(1,6)] + [f"Other Image URL{i}" for i in range(1,4)]:
                    if col not in df_source.columns: df_source[col] = ""
                    p_val = str(df_source.at[parent_idx, col]) if col in df_source.columns else ""
                    if p_val:
                        df_source.loc[child_idx, col] = df_source.loc[child_idx, col] \
                            .where(df_source.loc[child_idx, col].astype(str).str.strip() != "", p_val)

                # Try per-child inference with more context
                for ix in child_idx:
                    nm = str(df_source.at[ix, title_col])
                    desc_txt = first_nonempty(
                        df_source.at[ix, "Product Description"] if "Product Description" in df_source.columns else "",
                        df_source.at[ix, "Description"] if "Description" in df_source.columns else ""
                    )
                    if "Size" in df_source.columns and not str(df_source.at[ix, "Size"]).strip():
                        df_source.at[ix, "Size"] = guess_size_from_text(nm, desc_txt)
                    if "Scent" in df_source.columns and not str(df_source.at[ix, "Scent"]).strip():
                        df_source.at[ix, "Scent"] = guess_scent_from_text(nm, desc_txt)
                    if "Colour" in df_source.columns and not str(df_source.at[ix, "Colour"]).strip():
                        df_source.at[ix, "Colour"] = guess_colour_from_text(nm, desc_txt)

                parent_sku = str(df_source.at[parent_idx, "SKU"]) if "SKU" in df_source.columns else ""
                df_source.loc[child_idx, "Parent SKU"] = parent_sku
                df_source.loc[child_idx, "Relationship Type"] = "variation"

                # group attributes -> variation theme
                group_sizes, group_colours, group_scents = set(), set(), set()
                if "Size" in df_source.columns:
                    group_sizes = set(str(x).strip() for x in df_source.loc[g.index, "Size"].values)
                if "Colour" in df_source.columns:
                    group_colours = set(str(x).strip() for x in df_source.loc[g.index, "Colour"].values)
                if "Scent" in df_source.columns:
                    group_scents = set(str(x).strip() for x in df_source.loc[g.index, "Scent"].values)

                # If the group effectively has one non-empty colour, fill blanks with that colour
                non_empty_colours = [x for x in group_colours if str(x).strip()]
                if len(non_empty_colours) == 1:
                    only = non_empty_colours[0]
                    df_source.loc[g.index, "Colour"] = df_source.loc[g.index, "Colour"].where(
                        df_source.loc[g.index, "Colour"].astype(str).str.strip() != "", only
                    )

                has_size_var   = len([x for x in group_sizes   if x]) > 1
                has_colour_var = len([x for x in group_colours if x]) > 1
                has_scent_var  = len([x for x in group_scents  if x]) > 1

                theme = ""
                if has_size_var and has_colour_var:
                    theme = "SizeName-ColorName"
                elif has_size_var:
                    theme = "SizeName"
                elif has_colour_var:
                    theme = "ColorName"
                elif has_scent_var:
                    theme = "ScentName"

                if theme:
                    df_source.loc[g.index, "Variation_theme"] = theme

            df_source.drop(columns=["_var_key"], errors="ignore", inplace=True)
        else:
            st.warning("Couldn’t find a title column for variation grouping (looked for Item Name/Name/Product Name/Title).")

    # ===== NEW: enforce sequential parent→child copying across runs =====
    df_source = propagate_parent_details_in_order(df_source)

    # --- Variation theme (fallback if not set above) ---
    if "Variation_theme" not in df_source.columns: df_source["Variation_theme"] = ""
    has_color = df_source.get("Colour", pd.Series([""]*len(df_source))).astype(str).str.strip().ne("")
    has_size  = df_source.get("Size",   pd.Series([""]*len(df_source))).astype(str).str.strip().ne("")
    has_scent = df_source.get("Scent",  pd.Series([""]*len(df_source))).astype(str).str.strip().ne("")
    df_source.loc[has_color & has_size & (df_source["Variation_theme"]==""), "Variation_theme"] = "SizeName-ColorName"
    df_source.loc[(~has_color) & has_size & (df_source["Variation_theme"]==""), "Variation_theme"] = "SizeName"
    df_source.loc[has_color & (~has_size) & (df_source["Variation_theme"]==""), "Variation_theme"] = "ColorName"
    df_source.loc[(~has_color) & (~has_size) & has_scent & (df_source["Variation_theme"]==""), "Variation_theme"] = "ScentName"

    st.markdown("**Detected source columns:**")
    st.code(", ".join(df_source.columns.tolist()) or "<none>")

    # ============= Template load + mapping =============
    st.subheader("2) Load template & infer schema")
    tmpl_buf = None
    if template_file is not None:
        tmpl_buf = io.BytesIO(template_file.read())
    elif use_default:
        with open(default_paths[template_choice], "rb") as f:
            tmpl_buf = io.BytesIO(f.read())

    if tmpl_buf is None:
        st.info("Upload the Amazon template Excel (left sidebar) to continue.")
        st.stop()

    df_tmpl_raw = read_first_sheet(tmpl_buf)
    template_cols, required_flags = infer_template_schema(df_tmpl_raw)

    st.markdown("**Template columns (in order):**")
    st.code(", ".join(template_cols) or "<none>")
    req_count = sum(1 for v in required_flags.values() if v)
    st.write(f"Required fields detected: **{req_count}**")

    unmatched = []
    for t in template_cols:
        s, _ = best_match(t, list(df_source.columns))
        if not s: unmatched.append(t)
    if unmatched:
        st.warning("No match (after cleaning + synonyms): " + ", ".join(unmatched))

    st.subheader("3) Review/override column mapping")
    mapping_defaults = []
    src_cols = list(df_source.columns)
    for tcol in template_cols:
        src_match, conf = best_match(tcol, src_cols)
        mapping_defaults.append({
            "template_column": tcol,
            "mapped_from": src_match,
            "confidence": round(conf, 3),
            "required": required_flags.get(tcol, False),
        })
    map_df = pd.DataFrame(mapping_defaults)

    st.caption("Adjust 'Mapped From' using the dropdowns.")
    edited = st.data_editor(
        map_df,
        use_container_width=True,
        height=420,
        hide_index=True,
        column_config={
            "template_column": st.column_config.TextColumn("Template Column", disabled=True),
            "mapped_from": st.column_config.SelectboxColumn("Mapped From (source column)", options=[""] + src_cols),
            "confidence": st.column_config.NumberColumn("Confidence", disabled=True, help="Similarity estimate"),
            "required": st.column_config.CheckboxColumn("Required?", disabled=True),
        },
        key="mapping_editor",
    )

    st.subheader("4) Validate & export")
    if st.button("Validate & Export", type="primary", use_container_width=True):
        overrides = {row["template_column"]: row["mapped_from"] for _, row in edited.iterrows()}
        ready_df, fix_df, final_mapping = map_dataframe_to_template(
            df_source, template_cols, required_flags, mapping_overrides=overrides
        )
        ready_df = force_fill_external_id(ready_df, barcodes_available)
        fix_df   = force_fill_external_id(fix_df,   barcodes_available)

        total_rows = len(df_source)
        st.success(f"Validation complete. Ready: **{len(ready_df)}** | Needs fixes: **{len(fix_df)}** | Total: **{total_rows}**")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        ready_buf = io.BytesIO()
        with pd.ExcelWriter(ready_buf, engine="xlsxwriter") as writer:
            ready_df.to_excel(writer, index=False)
        ready_buf.seek(0)
        st.download_button(
            "Download READY.xlsx", ready_buf, file_name=f"ready_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True
        )

        fix_buf = io.BytesIO()
        with pd.ExcelWriter(fix_buf, engine="xlsxwriter") as writer:
            fix_df.to_excel(writer, index=False)
        fix_buf.seek(0)
        st.download_button(
            "Download FIX_ME.xlsx", fix_buf, file_name=f"fixme_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True
        )

        map_buf = io.StringIO()
        final_mapping.to_csv(map_buf, index=False)
        st.download_button(
            "Download COLUMN_MAPPING.csv", map_buf.getvalue(), file_name=f"mapping_{ts}.csv",
            mime="text/csv", use_container_width=True
        )

        if "__missing_required_fields" in fix_df.columns and len(fix_df):
            with st.expander("See rows needing fixes (sample)"):
                st.dataframe(fix_df.head(50), use_container_width=True)

    st.divider()
    st.caption("Next: units parsing, GTIN checksum, per-template business rules, and per-seller profiles.")
else:
    st.info("Upload a CSV/XLSX to begin.")
