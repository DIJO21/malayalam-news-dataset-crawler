#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from queue import Queue, Empty
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from dateutil.tz import tzoffset
from tqdm import tqdm

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("mal_crawler_fast")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)

# -------------------------
# Constants & simple utils
# -------------------------
USER_AGENT = "Mozilla/5.0 (compatible; MalayalamDatasetFast/1.0; +https://example.com)"
EMOJI_RE = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+", flags=re.UNICODE)
URL_RE = re.compile(r'https?://\S+|www\.\S+')
CTRL_RE = re.compile(r'[\x00-\x1f\x7f-\x9f]')

# tzinfos mapping for dateutil to handle IST / common variants
_TZINFOS = {"IST": tzoffset("IST", int(5 * 3600 + 30 * 60)), "IST0": tzoffset("IST", int(5 * 3600 + 30 * 60))}

def ensure_dir(p):
    if not p:
        return
    os.makedirs(p, exist_ok=True)

def normalize_unicode(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace('\u200d','').replace('\u200c','')
    text = CTRL_RE.sub(' ', text)
    return text.strip()

def clean_text(text, min_chars=0):
    if not text:
        return ""
    t = normalize_unicode(text)
    t = URL_RE.sub(' ', t)
    t = EMOJI_RE.sub(' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    if min_chars and len(t) < min_chars:
        return ""
    return t

def text_hash(text):
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()

def normalize_url(base, href):
    if not href:
        return None
    href = href.strip()
    if href.startswith("javascript:") or href.startswith("mailto:") or href.startswith("tel:"):
        return None
    try:
        joined = urljoin(base, href)
        clean, _ = urldefrag(joined)
        parsed = urlparse(clean)
        if parsed.scheme not in ("http","https"):
            return None
        return clean
    except Exception:
        return None

def parse_date(dstr):
    if not dstr:
        return None
    try:
        # use tzinfos to avoid UnknownTimezoneWarning for 'IST' etc.
        dt = dateparser.parse(dstr, fuzzy=True, tzinfos=_TZINFOS)
        if dt:
            # if dt has no tzinfo, set to UTC to be explicit
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
    except Exception:
        pass
    # last resort
    try:
        return dstr.strip()
    except Exception:
        return None

# -------------------------
# Site selectors (kept minimal)
# -------------------------
SITE_SELECTORS = {
    "manoramaonline.com": {"headline":["h1.headline","h1"], "body":["div.article div.article-body p","article p"], "date":["meta[property='article:published_time']","time"]},
    "mathrubhumi.com": {"headline":["h1.article-title","h1"], "body":["div.article-body p","article p"], "date":["time"]},
    "malayalam.news18.com": {"headline":["h1"], "body":["div.article-body p","article p"], "date":["time","meta[property='article:published_time']"]},
    "twentyfournews.com": {"headline":["h1"], "body":["div.entry-content p","article p"], "date":["time"]},
    "asianetnews.com": {"headline":["h1"], "body":["div.article__content p","article p"], "date":["time"]},
    # fallback generic selectors are handled by extract_generic
}

def extract_with_selectors(html, selectors_map):
    soup = BeautifulSoup(html, "lxml")
    headline = ""
    for sel in selectors_map.get("headline", []):
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            headline = node.get_text(" ", strip=True)
            break
    body = ""
    for sel in selectors_map.get("body", []):
        nodes = soup.select(sel)
        if nodes:
            texts = [n.get_text(" ", strip=True) for n in nodes if n.get_text(strip=True)]
            cand = "\n".join(texts).strip()
            if len(cand) > len(body):
                body = cand
    date_val = None
    for sel in selectors_map.get("date", []):
        node = soup.select_one(sel)
        if node:
            if node.name == "meta":
                date_val = node.get("content")
            else:
                date_val = node.get("datetime") or node.get_text(strip=True)
            if date_val:
                break
    return headline, body, date_val

def extract_generic(html, url):
    soup = BeautifulSoup(html, "lxml")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    body = ""
    # choose largest text block among article/section/div
    blocks = soup.find_all(["article","section","div"], recursive=True)
    best = ""
    for b in blocks:
        t = b.get_text(" ", strip=True)
        if len(t) > len(best):
            best = t
    body = best or soup.get_text(" ", strip=True)
    meta = soup.find("meta", attrs={"name":"description"}) or soup.find("meta", property="og:description")
    summary = meta.get("content").strip() if meta and meta.get("content") else ""
    return title, body, summary

def extract_article_from_html(html, url, selectors_key=None):
    try:
        if selectors_key:
            # find matching known selector key (domain in key)
            sel_map = None
            for known in SITE_SELECTORS:
                if known in selectors_key:
                    sel_map = SITE_SELECTORS.get(known)
                    break
            if sel_map:
                h,b,d = extract_with_selectors(html, sel_map)
                if not b or len(b) < 80:
                    t, body, summary = extract_generic(html, url)
                    h = h or t
                    b = b or body
                    return {"headline": clean_text(h), "body": clean_text(b, min_chars=1), "summary": clean_text(summary), "published_date": parse_date(d)}
                else:
                    return {"headline": clean_text(h), "body": clean_text(b), "summary": "", "published_date": parse_date(d)}
        t, body, summary = extract_generic(html, url)
        return {"headline": clean_text(t), "body": clean_text(body), "summary": clean_text(summary), "published_date": None}
    except Exception as e:
        logger.debug("extract error: %s", e)
        return None

# -------------------------
# robots.txt (simple caching)
# -------------------------
_rp_cache = {}
def can_fetch_robots(session, url, user_agent=USER_AGENT):
    parsed = urlparse(url)
    domain = parsed.netloc
    if domain in _rp_cache:
        rp = _rp_cache[domain]
    else:
        rp = None
        try:
            rp_txt = session.get(f"https://{domain}/robots.txt", timeout=6, headers={"User-Agent": user_agent})
            if rp_txt.status_code == 200:
                from urllib.robotparser import RobotFileParser
                rp = RobotFileParser()
                rp.parse(rp_txt.text.splitlines())
        except Exception:
            rp = None
        _rp_cache[domain] = rp
    if rp:
        try:
            return rp.can_fetch(user_agent, url)
        except Exception:
            return True
    return True

# -------------------------
# polite GET with short default sleep
# -------------------------
def polite_get(session, url, respect_robots=True, timeout=12, retries=2, rate=(0.05,0.15), save_raw_dir=None):
    if respect_robots and not can_fetch_robots(session, url):
        raise PermissionError("Blocked by robots.txt")
    headers = {"User-Agent": USER_AGENT}
    backoff = 1.0
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or r.encoding or 'utf-8'
            html = r.text
            time.sleep(random.uniform(*rate))
            if save_raw_dir:
                try:
                    ensure_dir(save_raw_dir)
                    fname = os.path.join(save_raw_dir, hashlib.sha1(url.encode('utf-8')).hexdigest()+".html")
                    with open(fname, "w", encoding="utf-8") as f:
                        f.write(html)
                except Exception:
                    pass
            return html
        except PermissionError:
            raise
        except Exception as e:
            logger.debug("GET error %s attempt=%d: %s", url, attempt, e)
            time.sleep(backoff)
            backoff *= 1.8
    raise RuntimeError(f"Failed GET {url}")

# -------------------------
# sitemap/rss parsing (fast)
# -------------------------
def parse_rss_links(xml_text, base):
    out = []
    try:
        soup = BeautifulSoup(xml_text, "xml")
        for item in soup.find_all("item"):
            loc = item.find("link")
            if loc and loc.text:
                out.append(normalize_url(base, loc.text.strip()))
        for u in soup.find_all("url"):
            loc = u.find("loc")
            if loc and loc.text:
                out.append(normalize_url(base, loc.text.strip()))
    except Exception:
        pass
    return [u for u in out if u]

def try_fetch_sitemap_or_rss(session, domain, respect_robots=True):
    candidates = [
        f"https://{domain}/sitemap.xml",
        f"http://{domain}/sitemap.xml",
        f"https://{domain}/feed",
        f"https://{domain}/rss",
        f"https://{domain}/feeds",
        f"https://{domain}/rss.xml",
    ]
    out = []
    for url in candidates:
        try:
            html = polite_get(session, url, respect_robots=respect_robots, timeout=6, retries=1, rate=(0.01,0.03))
            out.extend(parse_rss_links(html, url))
        except Exception:
            continue
    return list(dict.fromkeys([u for u in out if u]))

# -------------------------
# BFS crawl (kept but depth-limited)
# -------------------------
ARTICLE_PATH_KEYWORDS = ["news", "article", "story", "/202", "/20", "articles", "read", "fact-check", "factcheck", "gallery", "video"]

def is_probable_article(url):
    parsed = urlparse(url)
    path = parsed.path.lower()
    if any(k in path for k in ARTICLE_PATH_KEYWORDS):
        return True
    if re.search(r'/\d{4,}($|/)|\d+\.html$', path):
        return True
    if len(path.split('/')) > 2 and len(path) < 140:
        return True
    return False

def crawl_seed(session, seed_url, allow_domains, max_links=2000, max_depth=2, respect_robots=True):
    seen = set()
    found = []
    q = [(seed_url, 0)]
    while q and len(found) < max_links:
        url, depth = q.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            html = polite_get(session, url, respect_robots=respect_robots, timeout=8, retries=1, rate=(0.01,0.03))
        except Exception:
            continue
        # sitemap/rss quick check at shallow depth
        if depth <= 1:
            try:
                domain = urlparse(url).netloc
                found += try_fetch_sitemap_or_rss(session, domain, respect_robots=respect_robots)
            except Exception:
                pass
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href]"):
            href = a.get("href")
            full = normalize_url(url, href)
            if not full:
                continue
            dom = urlparse(full).netloc
            if not any(d in dom for d in allow_domains):
                continue
            if full in seen:
                continue
            if is_probable_article(full) and full not in found:
                found.append(full)
            if depth < max_depth:
                q.append((full, depth+1))
        if len(found) >= max_links:
            break
    return list(dict.fromkeys(found))[:max_links]

# -------------------------
# Seeds (kept from your script)
# -------------------------
REAL_SEEDS = {
    "manoramaonline.com": ("https://www.manoramaonline.com/news", ["manoramaonline.com"]),
    "mathrubhumi.com": ("https://www.mathrubhumi.com/", ["mathrubhumi.com"]),
    "malayalam.news18.com": ("https://malayalam.news18.com/", ["malayalam.news18.com", "news18.com"]),
    "twentyfournews.com": ("https://www.twentyfournews.com/", ["twentyfournews.com"]),
    "indianexpress.com": ("https://malayalam.indianexpress.com/", ["indianexpress.com"]),
    "asianetnews.com": ("https://www.asianetnews.com/latest-news", ["asianetnews.com"]),
    "mediaoneonline.com": ("https://www.mediaoneonline.com/", ["mediaoneonline.com"])
}
FAKE_SEEDS = {
    "malayalam.news18.com_fake": ("https://malayalam.news18.com/tag/fake-news/", ["malayalam.news18.com","news18.com"]),
    "mathrubhumi.com_fact": ("https://www.mathrubhumi.com/fact-check", ["mathrubhumi.com"]),
    "boomlive.in": ("https://www.boomlive.in/fact-check", ["boomlive.in"]),
    "altnews.in": ("https://www.altnews.in/tag/malayalam/", ["altnews.in"]),
    "factly.in": ("https://factly.in/tag/malayalam/", ["factly.in"])
}

# -------------------------
# Fact-check parsing (generic)
# -------------------------
def parse_factcheck_generic(html):
    soup = BeautifulSoup(html, "lxml")
    claim = ""
    if soup.find("h1"):
        claim = soup.find("h1").get_text(" ", strip=True)
    else:
        mt = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name":"twitter:title"})
        claim = mt.get("content").strip() if mt and mt.get("content") else ""
    selectors = ["div.article__content p","div.entry-content p","div.article-body p","article p","div.story p"]
    nodes = []
    for sel in selectors:
        nodes = soup.select(sel)
        if nodes:
            break
    explanation = " ".join([n.get_text(" ", strip=True) for n in nodes]) if nodes else soup.get_text(" ", strip=True)[:2000]
    corrected = ""
    paras = soup.find_all("p")
    if paras and len(paras) >= 2:
        corrected = " ".join([p.get_text(" ", strip=True) for p in paras[-2:]])
    published = None
    if soup.find("time"):
        t = soup.find("time")
        published = t.get("datetime") or t.get_text(strip=True)
    else:
        m = soup.find("meta", property="article:published_time")
        published = m.get("content") if m and m.get("content") else None
    return {"claim": clean_text(claim), "explanation": clean_text(explanation), "corrected": clean_text(corrected), "published_date": parse_date(published)}

# -------------------------
# CSV writer thread
# -------------------------
def csv_writer_thread(out_csv, queue_obj, fieldnames, checkpoint_every=200):
    ensure_dir(os.path.dirname(out_csv) or ".")
    partial_path = out_csv + ".partial.csv"
    written = 0
    # open file and stream writes
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        while True:
            # block until item available
            item = queue_obj.get()
            if item is None:
                # sentinel: done producing
                break
            try:
                w.writerow({k: item.get(k,"") if item.get(k) is not None else "" for k in fieldnames})
                written += 1
                if written % checkpoint_every == 0:
                    try:
                        f.flush()
                        # safe partial copy using shutil
                        shutil.copyfile(out_csv, partial_path)
                        logger.info("Checkpoint saved: %s (written=%d)", partial_path, written)
                    except Exception:
                        logger.debug("Failed partial checkpoint copy")
            except Exception as e:
                logger.debug("Failed to write row: %s", e)
    logger.info("CSV writer finished. Total written: %d", written)

# -------------------------
# Orchestration: producer functions to fetch+extract and push to queue
# -------------------------
def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s

def fetch_and_extract(session, url, selectors_key=None, respect_robots=True, save_raw_dir=None):
    try:
        html = polite_get(session, url, respect_robots=respect_robots, save_raw_dir=save_raw_dir)
    except PermissionError:
        return None
    except Exception:
        return None
    art = None
    try:
        art = extract_article_from_html(html, url, selectors_key)
    except Exception:
        art = None
    return art

def collect_from_candidates(candidates, source_key, allow_domains, seen_urls, seen_hashes, queue_out, label, session_factory, min_body_len=120, save_raw_dir=None, selectors_key=None, target_count=None, respect_robots=True, workers=20):
    """Parallel extraction from candidate URLs. Streams results into queue_out.
       Returns count added.
    """
    added = 0
    if not candidates:
        return added
    futures_map = {}
    with ThreadPoolExecutor(max_workers=workers) as exe:
        for url in candidates:
            if target_count and added >= target_count:
                break
            if url in seen_urls:
                continue
            # submit fetch_and_extract with new session
            fut = exe.submit(fetch_and_extract, make_session(), url, selectors_key, respect_robots, save_raw_dir)
            futures_map[fut] = url
            # throttle in-flight futures
            if len(futures_map) > workers * 3:
                # process a few completions
                for f in list(as_completed(list(futures_map.keys())[:5])):
                    u = futures_map.pop(f, None)
                    try:
                        art = f.result()
                    except Exception:
                        art = None
                    if not art or not art.get("body"):
                        if u:
                            seen_urls.add(u)
                        continue
                    body = art.get("body","")
                    if len(body) < min_body_len:
                        seen_urls.add(u); continue
                    h = text_hash(art.get("headline","") + "\n" + body)
                    if h in seen_hashes:
                        seen_urls.add(u); continue
                    seen_hashes.add(h); seen_urls.add(u)
                    item = {
                        "id": f"{source_key}_{('R' if label==1 else 'F')}_{int(time.time()*1000) % 1000000}",
                        "headline": art.get("headline",""),
                        "body": body,
                        "summary": art.get("summary",""),
                        "source": source_key,
                        "url": u,
                        "published_date": art.get("published_date"),
                        "label": label,
                        "synthetic": False
                    }
                    queue_out.put(item)
                    added += 1
                    if target_count and added >= target_count:
                        break
        # finish remaining futures
        for f in as_completed(list(futures_map.keys())):
            u = futures_map.pop(f, None)
            try:
                art = f.result()
            except Exception:
                art = None
            if not art or not art.get("body"):
                if u:
                    seen_urls.add(u)
                continue
            body = art.get("body","")
            if len(body) < min_body_len:
                seen_urls.add(u); continue
            h = text_hash(art.get("headline","") + "\n" + body)
            if h in seen_hashes:
                seen_urls.add(u); continue
            seen_hashes.add(h); seen_urls.add(u)
            item = {
                "id": f"{source_key}_{('R' if label==1 else 'F')}_{int(time.time()*1000) % 1000000}",
                "headline": art.get("headline",""),
                "body": body,
                "summary": art.get("summary",""),
                "source": source_key,
                "url": u,
                "published_date": art.get("published_date"),
                "label": label,
                "synthetic": False
            }
            queue_out.put(item)
            added += 1
            if target_count and added >= target_count:
                break
    return added

def synthetic_items(n):
    examples = [
        "ഒരു വീഡിയോ കാണുക: പാലം തകർന്നതായി റിപ്പോർട്ട് ചെയ്യുന്നു.",
        "ഒരു ചിത്രം പ്രചരിക്കുന്നു: ആശുപത്രിയിൽ ദുരിതം ഉണ്ടെന്ന് കാണിക്കുന്നു.",
        "പഞ്ചായത്ത് ഓഫീസിൽ പുതിയ കർമ്മനയം പ്രകാരം സാമ്പത്തിക സഹായം കണ്ടെത്തിയെന്ന് ആരോപണം."
    ]
    out = []
    for i in range(n):
        claim = random.choice(examples) + f" ({i})"
        corrected = "ഈ വിവരം തെറ്റാണ്; ഇത് പരിശോധിക്കപ്പെട്ടിട്ടില്ല."
        explanation = "പര്യാപ്തമായ ഉറവിടങ്ങൾ ഇല്ല; പ്രാഥമിക പരിശോധനയിൽ ഇത് സ്ഥിരീകരിച്ചിട്ടില്ല."
        out.append({"claim": clean_text(claim), "explanation": clean_text(explanation), "corrected": clean_text(corrected), "publisher":"SYNTHETIC"})
    return out

# -------------------------
# Main build
# -------------------------
def build(out_csv="outputs/malayalam_dataset_fast.csv",
          max_per_site=2000,
          target_real=1000,
          target_fake=1000,
          respect_robots=True,
          crawl_depth=2,
          workers=40,
          checkpoint_every=200,
          save_raw=False):
    ensure_dir(os.path.dirname(out_csv) or ".")
    queue_out = Queue()
    fieldnames = ["id","headline","body","summary","source","url","published_date","label","synthetic"]

    # Start CSV writer (runs in background thread)
    import threading
    writer_thread = threading.Thread(target=csv_writer_thread, args=(out_csv, queue_out, fieldnames, checkpoint_every), daemon=True)
    writer_thread.start()

    seen_urls = set()
    seen_hashes = set()
    total_added_real = 0
    total_added_fake = 0

    session_main = make_session()

    # 1) Real seeds: sitemap/rss-first
    for domain_key, (seed, domains) in REAL_SEEDS.items():
        if target_real and total_added_real >= target_real:
            break
        logger.info("Seed (real): %s", domain_key)
        candidates = []
        try:
            # feed try_fetch_sitemap_or_rss domain (seed netloc)
            candidates = try_fetch_sitemap_or_rss(session_main, urlparse(seed).netloc, respect_robots=respect_robots)
        except Exception:
            candidates = []
        # fallback to shallow parse of seed page
        if not candidates:
            try:
                html = polite_get(session_main, seed, respect_robots=respect_robots, timeout=8, retries=1, rate=(0.01,0.03), save_raw_dir=(domain_key if save_raw else None))
                soup = BeautifulSoup(html, "lxml")
                for a in soup.select("a[href]"):
                    u = normalize_url(seed, a.get("href"))
                    if u and is_probable_article(u):
                        candidates.append(u)
            except Exception:
                pass
        candidates = list(dict.fromkeys(candidates))[:max_per_site]
        random.shuffle(candidates)
        need = (target_real - total_added_real) if target_real else None
        added = collect_from_candidates(candidates, domain_key, domains, seen_urls, seen_hashes, queue_out, 1,
                                        make_session, min_body_len=120, save_raw_dir=(domain_key if save_raw else None),
                                        selectors_key=domain_key, target_count=need, respect_robots=respect_robots, workers=min(workers, 24))
        total_added_real += added
        logger.info("Collected %d real (cumulative %d) from %s", added, total_added_real, domain_key)

    # fallback BFS if still short
    if target_real and total_added_real < target_real:
        logger.info("Falling back to BFS for real articles, need %d more", target_real - total_added_real)
        for domain_key, (seed, domains) in REAL_SEEDS.items():
            if target_real and total_added_real >= target_real:
                break
            try:
                candidates = crawl_seed(session_main, seed, domains, max_links=max_per_site, max_depth=crawl_depth, respect_robots=respect_robots)
            except Exception:
                candidates = []
            random.shuffle(candidates)
            need = (target_real - total_added_real) if target_real else None
            added = collect_from_candidates(candidates, domain_key, domains, seen_urls, seen_hashes, queue_out, 1, make_session,
                                            min_body_len=120, save_raw_dir=(domain_key if save_raw else None),
                                            selectors_key=domain_key, target_count=need, respect_robots=respect_robots, workers=min(workers,24))
            total_added_real += added
            logger.info("BFS added %d real (cumulative %d) from %s", added, total_added_real, domain_key)

    # Fake/fact-check seeds (sitemap-first)
    for domain_key, (seed, domains) in FAKE_SEEDS.items():
        if target_fake and total_added_fake >= target_fake:
            break
        logger.info("Seed (fake/fact): %s", domain_key)
        candidates = []
        try:
            candidates = try_fetch_sitemap_or_rss(session_main, urlparse(seed).netloc, respect_robots=respect_robots)
        except Exception:
            candidates = []
        if not candidates:
            try:
                html = polite_get(session_main, seed, respect_robots=respect_robots, timeout=8, retries=1, rate=(0.01,0.03), save_raw_dir=(domain_key if save_raw else None))
                soup = BeautifulSoup(html, "lxml")
                for a in soup.select("a[href]"):
                    u = normalize_url(seed, a.get("href"))
                    if u:
                        candidates.append(u)
            except Exception:
                pass
        candidates = list(dict.fromkeys(candidates))[:max_per_site]
        random.shuffle(candidates)
        added = 0
        with ThreadPoolExecutor(max_workers=min(workers,20)) as exe:
            futures = {exe.submit(lambda u,dk=domain_key: polite_get(make_session(), u, respect_robots=respect_robots, save_raw_dir=(dk if save_raw else None)), url): url for url in candidates if url not in seen_urls}
            for f in as_completed(list(futures.keys())):
                u = futures.pop(f, None)
                if not u:
                    continue
                try:
                    html = f.result()
                except Exception:
                    seen_urls.add(u); continue
                fc = parse_factcheck_generic(html)
                text_to_use = fc.get("claim") or fc.get("explanation","")
                if not text_to_use or len(text_to_use) < 60:
                    seen_urls.add(u); continue
                h = text_hash(text_to_use)
                if h in seen_hashes:
                    seen_urls.add(u); continue
                seen_hashes.add(h); seen_urls.add(u)
                item = {
                    "id": f"{domain_key}_F_{int(time.time()*1000) % 1000000}",
                    "headline": fc.get("claim",""),
                    "body": fc.get("explanation",""),
                    "summary": fc.get("corrected",""),
                    "source": domain_key,
                    "url": u,
                    "published_date": fc.get("published_date"),
                    "label": 0,
                    "synthetic": False
                }
                queue_out.put(item)
                added += 1
                total_added_fake += 1
                if target_fake and total_added_fake >= target_fake:
                    break
        logger.info("Collected %d fake (cumulative %d) from %s", added, total_added_fake, domain_key)

    # BFS fallback for fake
    if target_fake and total_added_fake < target_fake:
        logger.info("Falling back to BFS for fake/fact pages, need %d more", target_fake - total_added_fake)
        for domain_key, (seed, domains) in FAKE_SEEDS.items():
            if target_fake and total_added_fake >= target_fake:
                break
            try:
                candidates = crawl_seed(session_main, seed, domains, max_links=max_per_site, max_depth=crawl_depth, respect_robots=respect_robots)
            except Exception:
                candidates = []
            random.shuffle(candidates)
            with ThreadPoolExecutor(max_workers=min(workers,20)) as exe:
                futures = {exe.submit(lambda u,dk=domain_key: polite_get(make_session(), u, respect_robots=respect_robots, save_raw_dir=(dk if save_raw else None)), url): url for url in candidates if url not in seen_urls}
                for f in as_completed(list(futures.keys())):
                    u = futures.pop(f, None)
                    if not u:
                        continue
                    try:
                        html = f.result()
                    except Exception:
                        seen_urls.add(u); continue
                    fc = parse_factcheck_generic(html)
                    text_to_use = fc.get("claim") or fc.get("explanation","")
                    if not text_to_use or len(text_to_use) < 60:
                        seen_urls.add(u); continue
                    h = text_hash(text_to_use)
                    if h in seen_hashes:
                        seen_urls.add(u); continue
                    seen_hashes.add(h); seen_urls.add(u)
                    item = {
                        "id": f"{domain_key}_F_{int(time.time()*1000) % 1000000}",
                        "headline": fc.get("claim",""),
                        "body": fc.get("explanation",""),
                        "summary": fc.get("corrected",""),
                        "source": domain_key,
                        "url": u,
                        "published_date": fc.get("published_date"),
                        "label": 0,
                        "synthetic": False
                    }
                    queue_out.put(item)
                    total_added_fake += 1
                    if target_fake and total_added_fake >= target_fake:
                        break
            logger.info("BFS added fake items; cumulative fake=%d", total_added_fake)

    # Fill synthetic if fake < real (we keep all real)
    if target_real:
        need_to_match = max(0, total_added_real - total_added_fake)
    else:
        need_to_match = 0
    if need_to_match > 0:
        logger.info("Generating %d synthetic fake items to match real count", need_to_match)
        for s in synthetic_items(need_to_match):
            h = text_hash(s.get("claim","") + "\n" + s.get("explanation",""))
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            item = {
                "id": f"SYN_F_{int(time.time()*1000) % 1000000}",
                "headline": s.get("claim",""),
                "body": s.get("explanation",""),
                "summary": s.get("corrected",""),
                "source": s.get("publisher","SYNTHETIC"),
                "url": "",
                "published_date": None,
                "label": 0,
                "synthetic": True
            }
            queue_out.put(item)
            total_added_fake += 1

    # Signal writer thread to finish
    queue_out.put(None)
    writer_thread.join(timeout=600)

    # create manifest using tracked counters
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {"real": int(total_added_real), "fake": int(total_added_fake), "total": int(total_added_real + total_added_fake)},
        "seeds_real": list(REAL_SEEDS.keys()),
        "seeds_fake": list(FAKE_SEEDS.keys())
    }
    with open(out_csv.replace(".csv",".manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)
    logger.info("Done. Manifest saved. CSV: %s", out_csv)
    return out_csv, out_csv.replace(".csv",".manifest.json")

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/malayalam_dataset_fast.csv")
    p.add_argument("--max-per-site", type=int, default=2000)
    p.add_argument("--target-real", type=int, default=1000)
    p.add_argument("--target-fake", type=int, default=1000)
    p.add_argument("--no-respect-robots", action="store_true")
    p.add_argument("--crawl-depth", type=int, default=2)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--save-raw", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        out_csv, manifest = build(out_csv=args.out,
                                  max_per_site=args.max_per_site,
                                  target_real=args.target_real,
                                  target_fake=args.target_fake,
                                  respect_robots=not args.no_respect_robots,
                                  crawl_depth=args.crawl_depth,
                                  workers=args.workers,
                                  checkpoint_every=args.checkpoint_every,
                                  save_raw=args.save_raw)
        print("Saved:", out_csv, manifest)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Error: %s", e)
        sys.exit(1)
 