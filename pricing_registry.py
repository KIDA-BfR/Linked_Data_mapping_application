# pricing_registry.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import os
import re
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
from lxml import html
try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    from langchain.callbacks.base import BaseCallbackHandler

PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"  # official pricing doc

@dataclass(frozen=True)
class TokenPricing:
    input_per_million: float
    output_per_million: float
    currency: str = "USD"
    source: str = PRICING_URL
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass(frozen=True)
class ModelPricing:
    text_input_per_1m: float
    text_output_per_1m: float
    currency: str = "USD"
    source: str = PRICING_URL
    fetched_at_iso: str = ""

def _normalize_model_name(model: str) -> str:
    # handles "models/gemini-2.5-flash" etc.
    m = model.strip()
    m = m.removeprefix("models/")
    return m.lower()

def _alias_candidates(model: str) -> list[str]:
    """
    Try common alias patterns without hardcoding specific versions.
    Example: gemini-2.5-flash -> gemini-2.5-flash-preview-09-2025
    We generate candidates by stripping suffixes or adding them based on what we fetched.
    """
    m = _normalize_model_name(model)
    cands = [m]

    # also try stripping known suffix patterns (preview / exp / dated)
    cands.append(re.sub(r"-(preview|exp)-\d{2}-\d{4}$", "", m))
    cands.append(re.sub(r"-\d{2}-\d{4}$", "", m))
    
    # Family bases
    if "flash" in m: cands.append("gemini-1.5-flash")
    if "pro" in m: cands.append("gemini-1.5-pro")

    return list(dict.fromkeys([x for x in cands if x]))

class PricingRegistry:
    def __init__(self, cache_path: str = ".cache/gemini_pricing.json", ttl_hours: int = 24):
        self.cache_path = cache_path
        self.ttl = timedelta(hours=ttl_hours)
        self._pricing: Dict[str, ModelPricing] = {}
        self._fetched_at: Optional[datetime] = None
        self._load_cache()

    @property
    def fetched_at(self) -> Optional[datetime]:
        return self._fetched_at

    def _load_cache(self) -> None:
        if not os.path.exists(self.cache_path):
            print(f"[Pricing] Cache not found at {self.cache_path}")
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            fetched_at = datetime.fromisoformat(blob["fetched_at_iso"])
            self._fetched_at = fetched_at
            for model_id, d in blob["pricing"].items():
                self._pricing[model_id] = ModelPricing(**d)
            print(f"[Pricing] Loaded {len(self._pricing)} models from cache (fetched at {self._fetched_at})")
        except Exception as e:
            print(f"[Pricing] Failed to load cache: {e}")
            self._pricing = {}
            self._fetched_at = None

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            blob = {
                "fetched_at_iso": (self._fetched_at or datetime.now(timezone.utc)).isoformat(),
                "pricing": {m: {k: v for k, v in vars(p).items()} for m, p in self._pricing.items()},
            }
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(blob, f, indent=2)
            print(f"[Pricing] Cache saved with {len(self._pricing)} models")
        except Exception as e:
            print(f"[Pricing] Failed to save cache: {e}")

    def _cache_valid(self) -> bool:
        if not self._fetched_at:
            return False
        valid = datetime.now(timezone.utc) - self._fetched_at < self.ttl
        print(f"[Pricing] Cache valid: {valid} (Age: {datetime.now(timezone.utc) - self._fetched_at})")
        return valid

    def refresh_if_needed(self) -> None:
        if self._cache_valid() and self._pricing:
            return
        print("[Pricing] Refreshing pricing from official site...")
        self._fetch_from_ai_google_dev()

    def _fetch_from_ai_google_dev(self) -> None:
        try:
            print(f"[Pricing] GET {PRICING_URL}")
            r = requests.get(PRICING_URL, timeout=20)
            r.raise_for_status()

            tree = html.fromstring(r.text)
            new_pricing: Dict[str, ModelPricing] = {}

            headings = tree.xpath("//h2|//h3")
            print(f"[Pricing] Found {len(headings)} sections in HTML")
            
            for h in headings:
                title = " ".join(h.text_content().split())
                if not title.lower().startswith("gemini "):
                    continue
                
                print(f"[Pricing] Parsing section: {title}")

                tables = h.xpath("following::table[1]")
                if not tables:
                    print(f"[Pricing] No table found after heading {title}")
                    continue

                table_html = html.tostring(tables[0], encoding="unicode")
                try:
                    df = pd.read_html(table_html)[0]
                except Exception as e:
                    print(f"[Pricing] Pandas failed to read table for {title}: {e}")
                    continue

                if df.shape[1] < 2:
                    print(f"[Pricing] Table for {title} has too few columns")
                    continue

                rows = [(str(a), str(b)) for a, b in zip(df.iloc[:, 0], df.iloc[:, 1])]
                text_in = text_out = None

                def _usd(x: str) -> Optional[float]:
                    m = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)", x.replace(",", ""))
                    return float(m.group(1)) if m else None

                for k, v in rows:
                    kl = k.lower()
                    if "text" in kl and "input" in kl:
                        text_in = _usd(v)
                    if "text" in kl and ("output" in kl or "response" in kl):
                        text_out = _usd(v)

                if text_in is None or text_out is None:
                    print(f"[Pricing] Could not find input/output prices in table for {title}")
                    continue

                # Model IDs
                model_ids = []
                for code in h.xpath("following::code/text()"):
                    c = code.strip().lower()
                    if c.startswith("gemini-") and len(c) < 80:
                        model_ids.append(c)
                model_ids = list(dict.fromkeys(model_ids))

                if not model_ids:
                    synth = re.sub(r"\s+", "-", title.lower())
                    model_ids = [synth]
                    print(f"[Pricing] No model codes found, using synthetic ID: {synth}")

                fetched_iso = datetime.now(timezone.utc).isoformat()
                for mid in model_ids:
                    print(f"[Pricing] Registering {mid}: IN={text_in}, OUT={text_out}")
                    new_pricing[mid] = ModelPricing(
                        text_input_per_1m=float(text_in),
                        text_output_per_1m=float(text_out),
                        fetched_at_iso=fetched_iso,
                    )
                    alias = re.sub(r"-(preview|exp)-\d{2}-\d{4}$", "", mid)
                    if alias != mid and alias not in new_pricing:
                        new_pricing[alias] = new_pricing[mid]

            if not new_pricing:
                raise ValueError("Scraper found no pricing entries")

            self._pricing = new_pricing
            self._fetched_at = datetime.now(timezone.utc)
            self._save_cache()
            
        except Exception as e:
            print(f"[Pricing] Scraping failed: {e}. Using fallback values.")
            # Fallback to some hardcoded values if scraping fails
            fallback_iso = datetime.now(timezone.utc).isoformat()
            self._pricing = {
                "gemini-1.5-flash": ModelPricing(0.075, 0.30, fetched_at_iso=fallback_iso),
                "gemini-1.5-pro": ModelPricing(1.25, 5.00, fetched_at_iso=fallback_iso),
                "gemini-2.0-flash": ModelPricing(0.10, 0.40, fetched_at_iso=fallback_iso),
                "gemini-2.5-flash": ModelPricing(0.30, 2.50, fetched_at_iso=fallback_iso),
            }
            self._fetched_at = datetime.now(timezone.utc)

    def get_pricing(self, model: str) -> Optional[ModelPricing]:
        self.refresh_if_needed()
        norm = _normalize_model_name(model)
        print(f"[Pricing] Looking up pricing for model: {model} (norm: {norm})")

        for cand in _alias_candidates(norm):
            if cand in self._pricing:
                print(f"[Pricing] Match found via alias: {cand}")
                return self._pricing[cand]

        for k, v in self._pricing.items():
            if k.startswith(norm + "-") or norm.startswith(k + "-"):
                print(f"[Pricing] Match found via prefix/family: {k}")
                return v
        
        print(f"[Pricing] No pricing found for {norm}. Available keys: {list(self._pricing.keys())}")
        return None

# Global instance
registry = PricingRegistry()

def get_pricing(provider: str, model: str) -> Optional[TokenPricing]:
    if provider.lower() == "gemini":
        p = registry.get_pricing(model)
        if p:
            return TokenPricing(
                input_per_million=p.text_input_per_1m,
                output_per_million=p.text_output_per_1m,
                last_updated=p.fetched_at_iso
            )
    return None

def estimate_cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = get_pricing(provider, model)
    if not pricing:
        return 0.0
    in_cost = (input_tokens / 1_000_000.0) * pricing.input_per_million
    out_cost = (output_tokens / 1_000_000.0) * pricing.output_per_million
    return float(in_cost + out_cost)

@dataclass
class CostSummary:
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    model: Optional[str] = None
    provider: Optional[str] = None

class GeminiCostCallback(BaseCallbackHandler):
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.summary = CostSummary(provider=provider, model=model)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        # response is typically an LLMResult
        for gen_list in getattr(response, "generations", []) or []:
            for gen in gen_list or []:
                msg = getattr(gen, "message", None)
                # usage_metadata is common in newer langchain-google-genai versions
                usage = getattr(msg, "usage_metadata", None) if msg else None
                if usage:
                    self.summary.input_tokens += int(usage.get("input_tokens", 0))
                    self.summary.output_tokens += int(usage.get("output_tokens", 0))
                else:
                    # Fallback for other formats
                    meta = getattr(msg, "response_metadata", {}) if msg else {}
                    token_usage = meta.get("token_usage")
                    if token_usage:
                        self.summary.input_tokens += int(token_usage.get("prompt_tokens", 0))
                        self.summary.output_tokens += int(token_usage.get("completion_tokens", 0))

        self.summary.total_cost_usd = estimate_cost_usd(
            self.provider, self.model, self.summary.input_tokens, self.summary.output_tokens
        )
