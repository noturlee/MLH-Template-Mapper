import os, re, json, time, zipfile, mimetypes
from pathlib import Path
from urllib.parse import urlparse, urljoin
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ---------- Headers ----------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-ZA,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# ---------- Excel ----------
def read_urls_from_excel(path: str, column="Url") -> list[str]:
    df = pd.read_excel(path, engine="openpyxl")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Columns: {list(df.columns)}")
    urls = [str(u).strip() for u in df[column].dropna().tolist()]
    seen, result = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            result.append(u)
    return result

# ---------- Utils ----------
def guess_ext_from_resp(url: str, resp: requests.Response) -> str:
    p = Path(urlparse(url).path).suffix
    if p:
        return p
    ctype = resp.headers.get("Content-Type", "").split(";")[0].strip()
    ext = mimetypes.guess_extension(ctype) or ""
    return ".jpg" if ext in (".jpe", ".jpeg", ".jpg") else (ext or ".bin")

MEDIA_RE = re.compile(
    r'https://(?:media|assets)\.takealot\.com/[^\s"\'<>]+?\.(?:jpg|jpeg|png|webp)(?:\?[^\s"\'<>]*)?',
    re.I
)

def _split_srcset(srcset: str):
    out = []
    for part in srcset.split(","):
        u = part.strip().split(" ")[0]
        if u:
            out.append(u)
    return out

def _clean_and_unique(urls: list[str], base_url: str | None = None) -> list[str]:
    clean, seen = [], set()
    bad = ("sprite", "placeholder", "logo", "/icons/", "favicon")
    for u in urls:
        if not isinstance(u, str): 
            continue
        s = u.strip()
        if s.startswith("//"): s = "https:" + s
        if base_url and s.startswith("/"): s = urljoin(base_url, s)
        if not s.lower().startswith("http"): 
            continue
        if any(b in s.lower() for b in bad): 
            continue
        if s not in seen:
            seen.add(s)
            clean.append(s)
    return clean

# ---------- HTTP extractor (fast path) ----------
def extract_images_http(product_url: str, sess: requests.Session) -> list[str]:
    # Warm cookie
    try: sess.get("https://www.takealot.com/", headers=HEADERS, timeout=15)
    except Exception: pass

    r = sess.get(product_url, headers=HEADERS, timeout=25, allow_redirects=True)
    r.raise_for_status()
    html, final_url = r.text, r.url

    low = html.lower()
    if "cf-challenge" in low or "captcha" in low:
        return []

    soup = BeautifulSoup(html, "html.parser")
    found = []

    # meta og:image
    for prop in ("og:image", "og:image:secure_url"):
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            found.append(tag["content"].strip())

    # <img> tags
    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-lazysrc"):
            v = img.get(attr)
            if v: found.append(v)
        if img.get("srcset"):
            found.extend(_split_srcset(img["srcset"]))

    # JSON blobs + regex
    for script in soup.find_all("script"):
        txt = script.string or script.text or ""
        st = txt.strip()
        if (st.startswith("{") and st.endswith("}")) or (st.startswith("[") and st.endswith("]")):
            try:
                data = json.loads(st)
                stack = [data]
                while stack:
                    node = stack.pop()
                    if isinstance(node, dict):
                        stack.extend(node.values())
                    elif isinstance(node, (list, tuple)):
                        stack.extend(node)
                    elif isinstance(node, str):
                        if MEDIA_RE.search(node): found.append(node)
            except Exception:
                pass
        found.extend(MEDIA_RE.findall(txt))

    # whole HTML regex
    found.extend(MEDIA_RE.findall(html))
    return _clean_and_unique(found, base_url=final_url)

# ---------- Playwright fallback (rendered) ----------
def extract_images_playwright(product_url: str) -> list[str]:
    from playwright.sync_api import sync_playwright
    urls = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=HEADERS["User-Agent"])
        page = context.new_page()
        # visit homepage to get cookies
        try: page.goto("https://www.takealot.com/", wait_until="domcontentloaded", timeout=30000)
        except Exception: pass

        # go to product; wait for network to settle
        page.goto(product_url, wait_until="networkidle", timeout=45000)
        final_url = page.url

        # expand gallery if lazy (try clicking thumbnails)
        try:
            thumbs = page.locator("img").all()
            # hover/scroll to force loading
            for t in thumbs[:12]:
                try:
                    t.scroll_into_view_if_needed(timeout=2000)
                except Exception:
                    pass
        except Exception:
            pass

        # collect img src + srcset from the rendered DOM
        imgs = page.locator("img")
        n = imgs.count()
        for i in range(n):
            try:
                el = imgs.nth(i)
                s = el.get_attribute("src") or ""
                ds = el.get_attribute("data-src") or ""
                ss = el.get_attribute("srcset") or ""
                if s: urls.append(s)
                if ds: urls.append(ds)
                if ss: urls.extend(_split_srcset(ss))
            except Exception:
                continue

        # also inspect <source srcset> (picture elements)
        sources = page.locator("source")
        m = sources.count()
        for i in range(m):
            try:
                ss = sources.nth(i).get_attribute("srcset") or ""
                if ss:
                    urls.extend(_split_srcset(ss))
            except Exception:
                continue

        browser.close()
    # Filter to media.takealot.com images and clean
    urls = [u for u in urls if MEDIA_RE.search(u)]
    return _clean_and_unique(urls, base_url=product_url)

# ---------- Download / Zip ----------
def download_images(img_urls: list[str], out_dir: Path, sess: requests.Session, start_idx=1) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, url in enumerate(img_urls, start_idx):
        try:
            resp = sess.get(url, headers=HEADERS, timeout=25, stream=True)
            if resp.status_code != 200: 
                continue
            ext = guess_ext_from_resp(url, resp)
            fname = out_dir / f"img_{i:05d}{ext}"
            with open(fname, "wb") as f:
                for chunk in resp.iter_content(64 * 1024):
                    if chunk: f.write(chunk)
            count += 1
        except Exception:
            continue
        time.sleep(0.15)
    return count

def zip_dir(src_dir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.glob("*")):
            if p.is_file():
                zf.write(p, arcname=p.name)

# ---------- Main ----------
def run(excel_path: str, column="Url", out_zip="takealot_images.zip"):
    sess = requests.Session()
    product_urls = read_urls_from_excel(excel_path, column=column)
    print(f"Found {len(product_urls)} product URLs")

    work = Path("downloaded_images")
    if work.exists():
        for f in work.glob("*"): f.unlink()
    work.mkdir(exist_ok=True)

    idx = 1
    total_downloaded = 0
    for n, purl in enumerate(product_urls, 1):
        # Try fast HTTP first
        gallery = []
        try:
            gallery = extract_images_http(purl, sess)
        except Exception as e:
            print(f"[{n}/{len(product_urls)}] HTTP fetch failed: {purl} | {e}")

        # Fallback to Playwright if needed
        if not gallery:
            try:
                gallery = extract_images_playwright(purl)
            except Exception as e:
                print(f"[{n}/{len(product_urls)}] Playwright fetch failed: {purl} | {e}")

        if not gallery:
            print(f"[{n}/{len(product_urls)}] No images found: {purl}")
            time.sleep(0.6)
            continue

        got = download_images(gallery, work, sess, start_idx=idx)
        idx += len(gallery)
        total_downloaded += got
        print(f"[{n}/{len(product_urls)}] {purl} -> {got} images")
        time.sleep(0.6)

    zip_dir(work, Path(out_zip))
    print(f"\nDone. Downloaded {total_downloaded} images.")
    print(f"Zipped at: {Path(out_zip).resolve()}")

if __name__ == "__main__":
    run("/Users/leighchejaikarran/Library/CloudStorage/OneDrive-Personal/DB360_Cleaner/takealot.xlsx",
        column="Url",
        out_zip="takealot_images.zip")
