# -*- coding: utf-8 -*-
"""
===================================
大盤復盤分析模塊
===================================

職責：
1. 取得大盤指數資料（上證、深證、創業板）
2. 搜尋市場新聞形成復盤情報
3. 使用大型語言模型產生每日大盤復盤報告
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from inspect import getattr_static
from typing import Optional, Dict, Any, List

import pandas as pd

from src.agent.provider_trace import resolved_model_provider_identity
from src.config import get_config
from src.report_language import normalize_report_language
from src.search_service import SearchService
from src.core.market_profile import get_profile, MarketProfile
from src.core.market_strategy import get_market_strategy_blueprint
from src.llm.backend_registry import (
    LOCAL_CLI_GENERATION_BACKEND_IDS,
    LITELLM_BACKEND_ID,
    resolve_generation_backend_id,
    resolve_generation_fallback_backend_id,
)
from src.llm.generation_backend import GenerationError, GenerationResult
from src.schemas.market_light import MARKET_LIGHT_REGIONS, MarketLightSnapshot
from src.services.run_diagnostics import record_llm_run, record_llm_run_started
from src.services.intelligence_service import IntelligenceService
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)

_LEGACY_ANALYZER_BACKEND_ID = "legacy_analyzer"

_ENGLISH_SECTION_PATTERNS = {
    "market_summary": r"###\s*(?:1\.\s*)?Market Summary",
    "index_commentary": r"###\s*(?:2\.\s*)?(?:Index Commentary|Major Indices)",
    "sector_highlights": r"###\s*(?:4\.\s*)?(?:Sector Highlights|Sector/Theme Highlights)",
}

_CHINESE_SECTION_PATTERNS = {
    "market_summary": r"###\s*一、(?:盤面總覽|市場總結)",
    "index_commentary": r"###\s*二、(?:指數結構|指數點評|主要指數)",
    "sector_highlights": r"###\s*三、(?:類股主流|類股主線|板塊主線|熱點解讀|類股表現)",
    "funds_sentiment": r"###\s*四、(?:資金與情緒|資金動向)",
    "news_catalysts": r"###\s*五、(?:消息催化|後市展望)",
}


def _compact_markdown(text: Optional[str]) -> str:
    """移除多餘空行與行尾空白（程式碼區塊內保留），讓手機 Discord 一屏看完。"""
    if not text:
        return ""
    out: List[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
        elif in_fence:
            out.append(line)
        elif line.strip():
            out.append(line.rstrip())
    return "\n".join(out).strip()


@dataclass
class MarketIndex:
    """大盤指數資料"""
    code: str                    # 指數代碼
    name: str                    # 指數名稱
    current: float = 0.0         # 當前點位
    change: float = 0.0          # 漲跌點數
    change_pct: float = 0.0      # 漲跌幅(%)
    open: float = 0.0            # 開盤點位
    high: float = 0.0            # 最高點位
    low: float = 0.0             # 最低點位
    prev_close: float = 0.0      # 昨收點位
    volume: float = 0.0          # 成交量（手）
    amount: float = 0.0          # 成交金額（元）
    amplitude: float = 0.0       # 振幅(%)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """市場概覽資料"""
    date: str                           # 日期
    indices: List[MarketIndex] = field(default_factory=list)  # 主要指數
    up_count: int = 0                   # 上漲家數
    down_count: int = 0                 # 下跌家數
    flat_count: int = 0                 # 平盤家數
    limit_up_count: int = 0             # 漲停家數
    limit_down_count: int = 0           # 跌停家數
    total_amount: float = 0.0           # 兩市成交金額（億元）
    # north_flow: float = 0.0           # 北向資金淨流入（億元）- 已廢棄，介面不可用

    # 類股漲幅榜
    top_sectors: List[Dict] = field(default_factory=list)     # 漲幅前5類股
    bottom_sectors: List[Dict] = field(default_factory=list)  # 跌幅前5類股
    top_concepts: List[Dict] = field(default_factory=list)    # 漲幅前5概念
    bottom_concepts: List[Dict] = field(default_factory=list) # 跌幅前5概念


@dataclass
class MarketLightReviewResult:
    """Internal market-review parts built from one overview fetch."""

    overview: MarketOverview
    report: str
    market_light_snapshot: Optional[Dict[str, Any]]
    structured_payload: Dict[str, Any] = field(default_factory=dict)


class MarketAnalyzer:
    """
    大盤復盤分析器

    功能：
    1. 取得大盤指數即時行情
    2. 取得市場漲跌統計
    3. 取得類股漲跌榜
    4. 搜尋市場新聞
    5. 產生大盤復盤報告
    """

    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        analyzer=None,
        region: str = "cn",
        config: Optional[Any] = None,
    ):
        """
        初始化大盤分析器

        Args:
            search_service: 搜尋服務實例
            analyzer: AI分析器實例（用於呼叫LLM）
            region: 市場區域 cn=陸股 hk=港股 us=美股 jp=日股 kr=韓股
            config: 本次復盤使用的設定；未傳入時讀取全域設定
        """
        self.config = config or get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.data_manager = DataFetcherManager()
        self.region = region if region in ("cn", "us", "hk", "jp", "kr") else "cn"
        self.profile: MarketProfile = get_profile(self.region)
        self.strategy = get_market_strategy_blueprint(self.region)

    def _log_context(self) -> str:
        return f"component=market_review region={self.region}"

    @staticmethod
    def _resolve_configured_response_provider(
        configured_model: str,
        response_model: str,
        model_list: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Match the actual response model against all deployments of one alias."""
        normalized_configured_model = str(configured_model or "").strip()
        normalized_response_model = str(response_model or "").strip().lower()
        if not normalized_configured_model or not normalized_response_model or not model_list:
            return ""

        for entry in model_list:
            params = entry.get("litellm_params", {}) or {}
            model_name = str(entry.get("model_name") or "").strip()
            if not model_name:
                model_name = str(params.get("model") or "").strip()
            if model_name != normalized_configured_model:
                continue

            deployment_model = str(params.get("model") or "").strip()
            if deployment_model.lower() != normalized_response_model:
                continue

            normalized_deployment_model = deployment_model.lower()
            if normalized_deployment_model.startswith("openai/~") or "openrouter" in normalized_deployment_model:
                return "openrouter"

            _resolved_model, resolved_provider = resolved_model_provider_identity(
                deployment_model,
            )
            if resolved_provider:
                return resolved_provider
        return ""

    def _resolve_recorded_provider(
        self,
        *,
        provider: str,
        model: str,
        backend: str,
        usage_provider: str = "",
        response_model: str = "",
    ) -> str:
        """Resolve LiteLLM router aliases before persisting diagnostics."""
        normalized_backend = str(backend or "").strip().lower()
        normalized_model = str(model or "").strip()
        normalized_usage_provider = str(usage_provider or "").strip()
        normalized_response_model = str(response_model or "").strip()
        normalized_provider = str(provider or "").strip()
        resolved_route_provider = ""
        if normalized_backend == "litellm" and normalized_model:
            resolved_route_model, resolved_route_provider = resolved_model_provider_identity(
                normalized_model,
                getattr(self.config, "llm_model_list", None) or [],
            )
            normalized_route = str(resolved_route_model or normalized_model).strip().lower()
            if normalized_route.startswith("openai/~") or "openrouter" in normalized_route:
                resolved_route_provider = "openrouter"
        resolved_response_provider = ""
        if normalized_response_model:
            resolved_response_provider = self._resolve_configured_response_provider(
                normalized_model,
                normalized_response_model,
                getattr(self.config, "llm_model_list", None) or [],
            )
            if resolved_response_provider == "openrouter":
                return resolved_response_provider
        if normalized_usage_provider:
            return normalized_usage_provider
        if normalized_response_model:
            if resolved_response_provider:
                return resolved_response_provider
            if resolved_route_provider == "openrouter":
                return resolved_route_provider
            _wire_model, resolved_provider = resolved_model_provider_identity(
                normalized_response_model,
            )
            if resolved_provider:
                return resolved_provider
        if normalized_backend != "litellm" or not normalized_model:
            return normalized_provider
        return resolved_route_provider or normalized_provider or "openai"

    def _resolve_recorded_error_model(
        self,
        *,
        error: Any,
        fallback_model: str = "",
    ) -> str:
        """Preserve route/model diagnostics for LiteLLM configuration failures."""
        details = getattr(error, "details", None)
        if isinstance(details, dict):
            visited: set[int] = set()

            def _find_error_model(payload: Dict[str, Any]) -> str:
                payload_id = id(payload)
                if payload_id in visited:
                    return ""
                visited.add(payload_id)
                for key in ("last_model", "route_name"):
                    candidate = str(payload.get(key) or "").strip()
                    if candidate:
                        return candidate
                fallback_error = payload.get("fallback_error")
                if isinstance(fallback_error, dict):
                    nested_details = fallback_error.get("details")
                    if isinstance(nested_details, dict):
                        candidate = _find_error_model(nested_details)
                        if candidate:
                            return candidate
                    return _find_error_model(fallback_error)
                return ""

            candidate = _find_error_model(details)
            if candidate:
                return candidate
        backend = str(getattr(error, "backend", "") or "").strip()
        configured_model = str(getattr(self.config, "litellm_model", "") or "").strip()
        normalized_fallback_model = str(fallback_model or "").strip()
        if backend == "litellm":
            return normalized_fallback_model or configured_model or backend
        return normalized_fallback_model or backend

    def _get_output_language(self) -> str:
        """Return the truthful report language (zh/en/ko) for payload and directives."""
        return normalize_report_language(
            getattr(getattr(self, "config", None), "report_language", "zh")
        )

    def _get_review_language(self) -> str:
        # Structural/template language. Korean reuses the English scaffolding;
        # the Korean output directive is applied in the prompt builder.
        language = self._get_output_language()
        return "en" if language == "ko" else language

    def _get_template_review_language(self) -> str:
        return self._get_review_language()

    def _get_market_scope_name(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "us":
            return "US market" if review_language == "en" else "美股市場"
        if self.region == "hk":
            return "Hong Kong market" if review_language == "en" else "港股市場"
        if self.region == "jp":
            return "Japan market" if review_language == "en" else "日本市場"
        if self.region == "kr":
            return "Korea market" if review_language == "en" else "韓國市場"
        if review_language == "en":
            return "A-share market"
        return "陸股市場"

    def _get_turnover_unit_label(self) -> str:
        """Return the turnover unit label for the current market/language."""
        if self.region == "us":
            return "USD bn" if self._get_review_language() == "en" else "十億美元"
        if self.region == "hk":
            return "HKD bn" if self._get_review_language() == "en" else "十億港元"
        if self.region == "jp":
            return "JPY bn" if self._get_review_language() == "en" else "十億日元"
        if self.region == "kr":
            return "KRW bn" if self._get_review_language() == "en" else "十億韓元"
        return "CNY 100m" if self._get_review_language() == "en" else "億"

    def _format_turnover_value(self, amount_raw: float) -> str:
        """Format raw turnover according to market-specific units."""
        if amount_raw == 0.0:
            return "N/A"
        if self.region in ("us", "hk", "jp", "kr"):
            return f"{amount_raw / 1e9:.2f}"
        if amount_raw > 1e6:
            return f"{amount_raw / 1e8:.0f}"
        return f"{amount_raw:.0f}"

    def _get_index_change_arrow(self, change_pct: float) -> str:
        if change_pct == 0:
            return "⚪"
        color_scheme = getattr(getattr(self, "config", None), "market_review_color_scheme", "green_up")
        if color_scheme == "red_up":
            return "🔴" if change_pct > 0 else "🟢"
        return "🟢" if change_pct > 0 else "🔴"

    def _get_review_title(self, date: str) -> str:
        if self._get_review_language() == "en":
            market_names = {
                "us": "US Market Recap",
                "hk": "HK Market Recap",
                "jp": "Japan Market Recap",
                "kr": "Korea Market Recap",
            }
            market_name = market_names.get(self.region, "A-share Market Recap")
            return f"## {date} {market_name}"
        return f"## {date} 大盤復盤"

    def _get_index_hint(self) -> str:
        if self._get_review_language() == "en":
            if self.region == "us":
                return "Analyze the key moves in the S&P 500, Nasdaq, Dow, and other major indices."
            if self.region == "hk":
                return "Analyze the key moves in the HSI, Hang Seng Tech, HSCEI, and other major indices."
            if self.region == "jp":
                return "Analyze the key moves in the Nikkei 225, TOPIX, and other major Japanese indices."
            if self.region == "kr":
                return "Analyze the key moves in the KOSPI, KOSDAQ, and other major Korean indices."
            return "Analyze the price action in the SSE, SZSE, ChiNext, and other major indices."
        return self.profile.prompt_index_hint

    def _get_strategy_prompt_block(self) -> str:
        if self.region == "hk" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Hong Kong Market Regime Strategy
Focus on HSI trend, southbound flow dynamics, and sector rotation to define next-session risk posture.

### Strategy Principles
- Read market regime from HSI, HSTECH, and HSCEI alignment first.
- Track southbound capital flow as a key sentiment driver.
- Translate recap into actionable risk-on/risk-off stance with clear invalidation points.

### Analysis Dimensions
- Trend Regime: Classify the market as momentum, range, or risk-off.
  - Are HSI/HSTECH/HSCEI directionally aligned
  - Did volume confirm the move
  - Are key index levels reclaimed or lost
- Capital Flows: Map southbound flow and macro narrative into equity risk appetite.
  - Southbound net flow direction and magnitude
  - USD/HKD and China policy implications
  - Breadth and leadership concentration
- Sector Themes: Identify persistent leaders and vulnerable laggards.
  - Tech/internet platform trend persistence
  - Financials/property sensitivity to policy shifts
  - Defensive vs growth factor rotation

### Action Framework
- Risk-on: broad index breakout with expanding southbound participation.
- Neutral: mixed index signals; focus on selective relative strength.
- Risk-off: failed breakouts and rising volatility; prioritize capital preservation."""
        if self.region == "jp" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Japan Market Regime Strategy
Focus on Nikkei 225, TOPIX, currency dynamics, and global risk appetite to define the next-session trading plan.

### Strategy Principles
- Read Nikkei 225 and TOPIX alignment first, then assess yen moves, semiconductor/export chains, and financials.
- Translate index conclusions into position sizing, trading pace, and risk-control actions.
- Base judgments only on available index data, news, and price action without inventing breadth or sector statistics.

### Analysis Dimensions
- Trend Regime: Classify Japan equities as advancing, range-bound, or defensive.
  - Are Nikkei 225 and TOPIX directionally aligned
  - Have key index ranges been reclaimed or lost
  - Are large-cap weights and growth chains moving together
- Macro & FX: Map yen, rates, and global risk appetite into equity impact.
  - Yen direction and implications for exporters
  - Bank of Japan and US Treasury yield narratives
  - Overseas technology and semiconductor read-through
- Theme Signals: Identify durable leadership and crowded areas to avoid.
  - Semiconductor, automation, and auto-chain persistence
  - Rotation between financials and domestic-demand stocks
  - Whether news catalysts confirm price action

### Action Framework
- Risk-on: major indices rise together with improving external risk appetite and stronger leadership.
- Neutral: index divergence or FX disruption; avoid chasing and wait for confirmation.
- Risk-off: major indices weaken or external risk rises; prioritize position control."""
        if self.region == "kr" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Korea Market Regime Strategy
Focus on KOSPI, KOSDAQ, semiconductor heavyweights, and global technology risk appetite to define the next-session trading plan.

### Strategy Principles
- Read KOSPI and KOSDAQ alignment first, then assess heavyweight signals from Samsung Electronics, SK Hynix, and related technology leaders.
- Separate broad index beta, semiconductor cycle exposure, and growth-stock risk appetite.
- Base judgments only on available index data, news, and price action without inventing breadth or sector statistics.

### Analysis Dimensions
- Trend Regime: Classify Korea equities as advancing, range-bound, or defensive.
  - Are KOSPI and KOSDAQ directionally aligned
  - Are heavyweight technology names supporting the indices
  - Have key support or resistance levels been reclaimed or lost
- Technology Cycle: Map semiconductor, AI hardware, and global technology moves into Korea equity risk.
  - Memory and semiconductor-chain catalysts
  - US technology-market read-through
  - Foreign investor risk appetite signals
- Theme Signals: Identify durable leadership and crowded areas to avoid.
  - Rotation across batteries, autos, and internet platforms
  - KOSDAQ growth-stock risk appetite
  - Whether news catalysts confirm price action

### Action Framework
- Risk-on: KOSPI and KOSDAQ rise together with confirmed technology leadership and improving external risk appetite.
- Neutral: index or heavyweight divergence; keep sizing controlled and wait for confirmation.
- Risk-off: technology heavyweights weaken or external risk rises; prioritize drawdown control."""
        if self.region == "us" and self._get_review_language() == "zh":
            return """## 美股市場三段式復盤策略
聚焦指數趨勢、宏觀敘事與類股輪動，給出次日風控與部位框架。

### 策略原則
- 先看標普500、那斯達克、道瓊斯是否同向，確認主線是否一致。
- 結合宏觀與流動性指標，識別風險偏好是修復還是轉弱。
- 將復盤輸出映射為「進攻/均衡/防守」動作建議，並給出明確觸發失效條件。

### 分析維度
- 趨勢結構：明確市場處於上衝、震盪還是防守轉向，判斷是否存在關鍵支撐位背離。
- 資金與情緒：區分宏觀政策、貨幣面與波動率對權益風險的影響。
- 主題線索：識別持續性最強的主題與類股輪動是否形成可交易主線。

### 行動框架
- 進攻：主類股聯動上行且量能/風險位同步改善。
- 均衡：指數分化或量能未明顯放大，部位保守執行。
- 防守：突破失守且波動率抬升時，優先減碼並保留反彈可交易性。"""
        if not (self.region == "cn" and self._get_review_language() == "en"):
            return self.strategy.to_prompt_block()
        return """## Strategy Blueprint: A-share Three-Phase Recap Strategy
Focus on index trend, liquidity, and sector rotation to shape the next-session trading plan.

### Strategy Principles
- Read index direction first, then confirm liquidity structure, and finally test sector persistence.
- Every conclusion must map to position sizing, trading pace, and risk-control actions.
- Base judgments on today's data and the latest 3-day news flow without inventing unverified information.

### Analysis Dimensions
- Trend Structure: Determine whether the market is in an uptrend, range, or defensive phase.
  - Are the SSE, SZSE, and ChiNext moving in the same direction
  - Is the market advancing on expanding volume or slipping on contracting volume
  - Have key support or resistance levels been reclaimed or broken
- Liquidity & Sentiment: Identify near-term risk appetite and market temperature.
  - Advance/decline breadth and limit-up/limit-down structure
  - Whether turnover is expanding or fading
  - Whether high-beta leaders are showing divergence
- Leading Themes: Distill tradable leadership and areas to avoid.
  - Whether leading sectors have clear event catalysts
  - Whether sector leaders are pulling the group higher
  - Whether weakness is broadening across lagging sectors

### Action Framework
- Offensive: indices rise in sync, turnover expands, and core themes strengthen.
- Balanced: index divergence or low-volume consolidation; keep sizing controlled and wait for confirmation.
- Defensive: indices weaken and laggards broaden; prioritize risk control and de-risking."""

    def _get_strategy_markdown_block(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "hk" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify the market as momentum, range, or risk-off based on HSI/HSTECH/HSCEI alignment.
- **Capital Flows**: Track southbound flow direction and macro narrative for risk appetite signals.
- **Sector Themes**: Focus on tech/internet platform persistence and financials/property policy sensitivity.
"""
        if self.region == "jp" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify Japan equities as advancing, range-bound, or defensive based on Nikkei 225/TOPIX alignment.
- **Macro & FX**: Track yen, rates, and global risk appetite for exporter and financial-sector implications.
- **Theme Signals**: Focus on semiconductor, automation, auto-chain, financial, and domestic-demand rotation.
"""
        if self.region == "kr" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify Korea equities as advancing, range-bound, or defensive based on KOSPI/KOSDAQ alignment.
- **Technology Cycle**: Track semiconductor, AI hardware, and global technology read-through for market risk appetite.
- **Theme Signals**: Focus on battery, auto, internet-platform, and KOSDAQ growth-stock rotation.
"""
        if self.region == "us" and review_language == "zh":
            return """### 六、策略框架
- **趨勢結構**：判斷市場在進攻、震盪與防守中的狀態是否一致。
- **資金與情緒**：結合波動率、廣度和主題輪動評估風險偏好。
- **主題主線**：識別可延續和可放大的產業主線與防守線索。
"""
        if not (self.region == "cn" and review_language == "en"):
            return self.strategy.to_markdown_block()
        return """### 6. Strategy Framework
- **Trend Structure**: Determine whether the market is in an uptrend, range, or defensive phase.
- **Liquidity & Sentiment**: Track breadth, turnover expansion, and whether leaders are diverging.
- **Leading Themes**: Focus on sectors with catalysts and sustained leadership while avoiding broadening weakness.
"""

    def _get_market_mood_text(self, mood_key: str, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if review_language == "en":
            mapping = {
                "strong_up": "strong gains",
                "mild_up": "moderate gains",
                "mild_down": "mild losses",
                "strong_down": "clear weakness",
                "range": "range-bound trading",
            }
        else:
            mapping = {
                "strong_up": "強勢上漲",
                "mild_up": "小幅上漲",
                "mild_down": "小幅下跌",
                "strong_down": "明顯下跌",
                "range": "震盪整理",
            }
        return mapping[mood_key]

    def get_market_overview(self) -> MarketOverview:
        """
        取得市場概覽資料

        Returns:
            MarketOverview: 市場概覽資料物件
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)

        # 1. 取得主要指數行情（依 region 切換 陸股/美股）
        overview.indices = self._get_main_indices()

        # 2. 取得漲跌統計（陸股有，美股無等效資料）
        if self.profile.has_market_stats:
            self._get_market_statistics(overview)

        # 3. 取得類股漲跌榜（陸股有，美股暫無）
        if self.profile.has_sector_rankings:
            self._get_sector_rankings(overview)
            self._get_concept_rankings(overview)

        # 4. 取得北向資金（可選）
        # self._get_north_flow(overview)

        return overview


    def _get_main_indices(self) -> List[MarketIndex]:
        """取得主要指數即時行情"""
        indices = []

        try:
            logger.info("[大盤] %s action=get_main_indices status=start", self._log_context())

            # 使用 DataFetcherManager 取得指數行情（依 region 切換）
            data_list = self.data_manager.get_main_indices(region=self.region)

            if data_list:
                for item in data_list:
                    index = MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=item['current'],
                        change=item['change'],
                        change_pct=item['change_pct'],
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        prev_close=item['prev_close'],
                        volume=item['volume'],
                        amount=item['amount'],
                        amplitude=item['amplitude']
                    )
                    indices.append(index)

            if not indices:
                logger.warning("[大盤] %s action=get_main_indices status=empty", self._log_context())
            else:
                logger.info(
                    "[大盤] %s action=get_main_indices status=success count=%d",
                    self._log_context(),
                    len(indices),
                )

        except Exception as e:
            logger.error("[大盤] %s action=get_main_indices status=failed error=%s", self._log_context(), e)

        return indices

    def _get_market_statistics(self, overview: MarketOverview):
        """取得市場漲跌統計"""
        try:
            logger.info("[大盤] %s action=get_market_stats status=start", self._log_context())

            stats = self.data_manager.get_market_stats(purpose=f"market_review:{self.region}")

            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)

                logger.info(
                    "[大盤] %s action=get_market_stats status=success up=%s down=%s flat=%s "
                    "limit_up=%s limit_down=%s amount=%.0f億",
                    self._log_context(),
                    overview.up_count,
                    overview.down_count,
                    overview.flat_count,
                    overview.limit_up_count,
                    overview.limit_down_count,
                    overview.total_amount,
                )
            else:
                logger.warning("[大盤] %s action=get_market_stats status=empty", self._log_context())

        except Exception as e:
            logger.error("[大盤] %s action=get_market_stats status=failed error=%s", self._log_context(), e)

    def _get_sector_rankings(self, overview: MarketOverview):
        """取得類股漲跌榜"""
        try:
            logger.info("[大盤] %s action=get_sector_rankings status=start", self._log_context())

            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)

            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors

                logger.info(
                    "[大盤] %s action=get_sector_rankings status=success top=%s bottom=%s",
                    self._log_context(),
                    [s['name'] for s in overview.top_sectors],
                    [s['name'] for s in overview.bottom_sectors],
                )
            else:
                logger.warning("[大盤] %s action=get_sector_rankings status=empty", self._log_context())

        except Exception as e:
            logger.error("[大盤] %s action=get_sector_rankings status=failed error=%s", self._log_context(), e)

    def _get_concept_rankings(self, overview: MarketOverview):
        """取得概念/題材漲跌榜（fail-open）。"""
        try:
            logger.info("[大盤] %s action=get_concept_rankings status=start", self._log_context())

            top_concepts, bottom_concepts = self.data_manager.get_concept_rankings(5)

            if top_concepts or bottom_concepts:
                overview.top_concepts = top_concepts
                overview.bottom_concepts = bottom_concepts

                logger.info(
                    "[大盤] %s action=get_concept_rankings status=success top=%s bottom=%s",
                    self._log_context(),
                    [s.get('name') for s in overview.top_concepts],
                    [s.get('name') for s in overview.bottom_concepts],
                )
            else:
                logger.warning("[大盤] %s action=get_concept_rankings status=empty", self._log_context())

        except Exception as e:
            logger.warning("[大盤] %s action=get_concept_rankings status=failed error=%s", self._log_context(), e)

    # def _get_north_flow(self, overview: MarketOverview):
    #     """取得北向資金流向入"""
    #     try:
    #         logger.info("[大盤] 取得北向資金...")
    #
    #         # 取得北向資金資料
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
    #
    #         if df is not None and not df.empty:
    #             # 取最新一條資料
    #             latest = df.iloc[-1]
    #             if '当日净流入' in df.columns:
    #                 overview.north_flow = float(latest['当日净流入']) / 1e8  # 转为亿元
    #             elif '净流入' in df.columns:
    #                 overview.north_flow = float(latest['净流入']) / 1e8
    #
    #             logger.info(f"[大盤] 北向資金淨流入: {overview.north_flow:.2f}億")
    #
    #     except Exception as e:
    #         logger.warning(f"[大盤] 取得北向資金失敗: {e}")

    def search_market_news(self) -> List[Dict]:
        """
        搜尋市場新聞

        Returns:
            新聞列表
        """
        if not self.search_service:
            logger.warning(
                "[大盤] %s action=search_market_news status=skipped reason=no_search_service",
                self._log_context(),
            )
            return []

        all_news = []

        # 依 region 使用不同的新聞搜尋詞
        search_queries = self.profile.news_queries
        review_language = self._get_review_language()
        market_names = {
            "cn": "大盘" if review_language == "zh" else "A-share market",
            "us": "美股市场" if review_language == "zh" else "US market",
            "hk": "港股市场" if review_language == "zh" else "HK market",
            "jp": "日本股市" if review_language == "zh" else "Japan stock market",
            "kr": "韩国股市" if review_language == "zh" else "Korea stock market",
        }

        try:
            logger.info("[大盤] %s action=search_market_news status=start", self._log_context())

            # 根據 region 設定搜尋上下文名稱，避免美股搜尋被解讀為 陸股語境
            market_name = market_names.get(self.region, "大盘")
            for query in search_queries:
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name=market_name,
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(
                        "[大盤] %s action=search_market_news status=query_success count=%d",
                        self._log_context(),
                        len(response.results),
                    )

            logger.info(
                "[大盤] %s action=search_market_news status=success count=%d",
                self._log_context(),
                len(all_news),
            )

        except Exception as e:
            logger.error("[大盤] %s action=search_market_news status=failed error=%s", self._log_context(), e)

        return all_news

    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大型語言模型產生大盤復盤報告

        Args:
            overview: 市場概覽資料
            news: 市場新聞列表 (SearchResult 物件列表)

        Returns:
            大盤復盤報告文字
        """
        backend_error = self._get_analyzer_generation_backend_config_error()
        if backend_error is not None:
            logger.error(
                "[大盤] %s action=generate_review status=failed error_type=%s error=%s",
                self._log_context(),
                type(backend_error).__name__,
                backend_error,
            )
            record_llm_run(
                success=False,
                provider=backend_error.provider or backend_error.backend,
                model=self._resolve_recorded_error_model(error=backend_error),
                call_type="market_review",
                error_type=type(backend_error).__name__,
                error_message=backend_error,
            )
            raise backend_error

        if not self.analyzer or not self.analyzer.is_available():
            logger.warning(
                "[大盤] %s action=generate_review status=fallback_template reason=no_analyzer",
                self._log_context(),
            )
            return self._generate_template_review(overview, news)

        # 建立 Prompt
        prompt = self._build_review_prompt(overview, news)

        logger.info("[大盤] %s action=generate_review status=start", self._log_context())
        # Use public analyzer APIs so diagnostics reflect the actual execution backend.
        llm_started_at = time.perf_counter()
        provider, model = self._get_analyzer_generation_backend_identity()
        try:
            record_llm_run_started(
                provider=provider,
                model=model,
                call_type="market_review",
            )
            generation_result = self._generate_market_review_with_metadata(
                prompt,
                provider=provider,
                model=model,
            )
            review = generation_result.text if generation_result else None
            generation_diagnostics = (
                getattr(generation_result, "diagnostics", None) if generation_result is not None else None
            )
            if generation_result is not None:
                model = generation_result.model or model
                usage_payload = getattr(generation_result, "usage", None)
                provider = self._resolve_recorded_provider(
                    provider=generation_result.provider or generation_result.backend or provider,
                    model=model,
                    backend=generation_result.backend or "",
                    usage_provider=(
                        usage_payload.get("provider") if isinstance(usage_payload, dict) else ""
                    ),
                    response_model=(
                        usage_payload.get("response_model") if isinstance(usage_payload, dict) else ""
                    ),
                )
        except Exception as exc:
            error_provider = getattr(exc, "provider", None) or getattr(exc, "backend", None) or provider
            error_model = self._resolve_recorded_error_model(
                error=exc,
                fallback_model=model,
            )
            record_llm_run(
                success=False,
                provider=error_provider,
                model=error_model,
                call_type="market_review",
                duration_ms=int((time.perf_counter() - llm_started_at) * 1000),
                error_type=type(exc).__name__,
                error_message=exc,
            )
            raise

        failed_all_models = (
            isinstance(generation_diagnostics, dict)
            and generation_diagnostics.get("reason") == "all_models_failed"
        )
        record_llm_run(
            success=bool(review),
            provider=provider,
            model=model,
            call_type="market_review",
            duration_ms=int((time.perf_counter() - llm_started_at) * 1000),
            error_type=None if review else ("AllModelsFailed" if failed_all_models else "EmptyResponse"),
            error_message=(
                None
                if review
                else (
                    generation_diagnostics
                    if failed_all_models
                    else "empty market review response"
                )
            ),
        )

        if review:
            logger.info(
                "[大盤] %s action=generate_review status=success length=%d",
                self._log_context(),
                len(review),
            )
            # Inject structured data tables into LLM prose sections
            return self._inject_data_into_review(review, overview, news)

        logger.warning(
            "[大盤] %s action=generate_review status=fallback_template reason=empty_llm_response",
            self._log_context(),
        )
        return self._generate_template_review(overview, news)

    def _get_analyzer_generation_backend_config_error(self) -> Optional[GenerationError]:
        """Return analyzer backend config errors without relying on dynamic mock attributes."""
        if self.analyzer is None:
            try:
                resolve_generation_backend_id(self.config)
                resolve_generation_fallback_backend_id(self.config)
            except GenerationError as exc:
                return exc
            return None
        missing = object()
        if getattr_static(self.analyzer, "get_generation_backend_config_error", missing) is missing:
            return None
        method = getattr(self.analyzer, "get_generation_backend_config_error", None)
        if not callable(method):
            return None
        error = method()
        return error if isinstance(error, GenerationError) else None

    def _get_configured_generation_backend_identity(self) -> tuple[str, str]:
        """Best-effort backend identity for legacy analyzers without metadata APIs."""
        backend_id = str(getattr(self.config, "generation_backend", "") or "").strip().lower()
        if not backend_id:
            backend_id = LITELLM_BACKEND_ID
        if backend_id in LOCAL_CLI_GENERATION_BACKEND_IDS:
            return backend_id, backend_id
        return backend_id, str(getattr(self.config, "litellm_model", "") or "")

    def _get_legacy_analyzer_generation_backend_identity(self) -> tuple[str, str]:
        """Return a neutral identity for injected analyzers that expose no metadata APIs."""
        return _LEGACY_ANALYZER_BACKEND_ID, _LEGACY_ANALYZER_BACKEND_ID

    def _get_analyzer_generation_backend_identity(self) -> tuple[str, str]:
        """Use analyzer metadata API when available, otherwise fall back to configured identity."""
        if self.analyzer is None:
            return self._get_configured_generation_backend_identity()
        missing = object()
        if getattr_static(self.analyzer, "get_generation_backend_identity", missing) is not missing:
            method = getattr(self.analyzer, "get_generation_backend_identity", None)
            if callable(method):
                return method()
        return self._get_legacy_analyzer_generation_backend_identity()

    def _generate_market_review_with_metadata(
        self,
        prompt: str,
        *,
        provider: str,
        model: str,
    ) -> Optional[GenerationResult]:
        """Support legacy analyzers that only implement the public generate_text() contract."""
        if self.analyzer is None:
            return None
        missing = object()
        if getattr_static(self.analyzer, "generate_text_with_metadata", missing) is not missing:
            method = getattr(self.analyzer, "generate_text_with_metadata", None)
            if callable(method):
                return method(
                    prompt,
                    max_tokens=8192,
                    temperature=0.7,
                )

        if getattr_static(self.analyzer, "generate_text", missing) is missing:
            raise AttributeError(
                "analyzer must implement generate_text_with_metadata() or generate_text()"
            )
        legacy_method = getattr(self.analyzer, "generate_text", None)
        if not callable(legacy_method):
            raise AttributeError(
                "analyzer must implement generate_text_with_metadata() or generate_text()"
            )
        review = legacy_method(
            prompt,
            max_tokens=8192,
            temperature=0.7,
        )
        if review is None:
            return None
        return GenerationResult(
            text=review,
            provider=provider,
            model=model,
            backend=provider,
            usage={},
        )

    def build_market_review_payload(
        self,
        overview: MarketOverview,
        news: List,
        report: str,
        market_light_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the structured market-review contract consumed by API, Web, and notifications."""
        language = self._get_output_language()
        sections = self._split_report_sections(report)
        title = self._extract_report_title(report) or self._get_review_title(overview.date).lstrip("# ").strip()
        light = (
            market_light_snapshot or self.build_market_light_snapshot(overview)
            if self._supports_market_light()
            else None
        )
        breadth_dimensions = None
        if isinstance(light, dict):
            dimensions = light.get("dimensions")
            if isinstance(dimensions, dict):
                breadth_dimensions = dimensions.get("breadth")

        breadth_supported = bool(self.profile.has_market_stats)
        if breadth_supported and isinstance(breadth_dimensions, dict) and "available" in breadth_dimensions:
            breadth_supported = bool(breadth_dimensions.get("available"))

        has_breadth_data = False
        if breadth_supported:
            if isinstance(breadth_dimensions, dict) and "available" in breadth_dimensions:
                has_breadth_data = bool(breadth_dimensions.get("available"))
            else:
                breadth_available = overview.up_count + overview.down_count + overview.flat_count > 0
                limit_available = overview.limit_up_count + overview.limit_down_count > 0
                has_breadth_data = bool(breadth_available or limit_available)

        payload = {
            "version": 1,
            "kind": "market_review",
            "region": self.region,
            "language": language,
            "color_scheme": getattr(
                getattr(self, "config", None),
                "market_review_color_scheme",
                "green_up",
            ),
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "date": overview.date,
            "market_scope": self._get_market_scope_name(language),
            "indices": [idx.to_dict() for idx in overview.indices],
            "sectors": {
                "top": list(overview.top_sectors or []),
                "bottom": list(overview.bottom_sectors or []),
            },
            "concepts": {
                "top": list(overview.top_concepts or []),
                "bottom": list(overview.bottom_concepts or []),
            },
            "news": [self._normalize_news_item(item) for item in (news or [])[:8]],
            "sections": sections,
            "markdown_report": report,
        }

        if light is not None:
            payload["market_light"] = light

        if has_breadth_data:
            payload["breadth"] = {
                "up_count": overview.up_count,
                "down_count": overview.down_count,
                "flat_count": overview.flat_count,
                "limit_up_count": overview.limit_up_count,
                "limit_down_count": overview.limit_down_count,
                "total_amount": overview.total_amount,
                "turnover_unit": self._get_turnover_unit_label(),
            }

        return payload

    def _supports_market_light(self) -> bool:
        return self.region in MARKET_LIGHT_REGIONS

    @staticmethod
    def _extract_report_title(report: str) -> str:
        for line in (report or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return ""

    @classmethod
    def _split_report_sections(cls, report: str) -> List[Dict[str, str]]:
        text = (report or "").strip()
        if not text:
            return []
        matches = list(re.finditer(r"^(#{2,3})\s+(.+?)\s*$", text, flags=re.MULTILINE))
        if not matches:
            return [{"key": "full_review", "title": "Review", "markdown": text}]

        sections: List[Dict[str, str]] = []
        first_match = matches[0]
        starts_with_report_title = first_match.start() == 0 and first_match.group(1) == "##"
        content_start_index = 1 if starts_with_report_title else 0
        intro_start = first_match.end() if starts_with_report_title else 0
        intro_end = (
            matches[1].start()
            if starts_with_report_title and len(matches) > 1
            else (len(text) if starts_with_report_title else matches[0].start())
        )
        intro = text[intro_start:intro_end].strip()
        if intro:
            sections.append({"key": "overview", "title": "Overview", "markdown": intro})

        for index, match in enumerate(matches[content_start_index:], start=content_start_index):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            title = match.group(2).strip()
            markdown = text[start:end].strip()
            if not markdown:
                continue
            key = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", title).strip("_").lower()
            sections.append({
                "key": key or f"section_{index + 1}",
                "title": title,
                "markdown": markdown,
            })
        return sections

    @classmethod
    def _normalize_news_item(cls, item: Any) -> Dict[str, str]:
        return {
            "title": cls._compact_news_text(cls._get_news_field(item, "title"), limit=120),
            "snippet": cls._compact_news_text(cls._get_news_field(item, "snippet"), limit=260),
            "source": cls._compact_news_text(cls._get_news_field(item, "source"), limit=80),
            "published_date": cls._compact_news_text(cls._get_news_field(item, "published_date"), limit=40),
            "url": cls._compact_news_text(cls._get_news_field(item, "url"), limit=240),
        }

    def _inject_data_into_review(
        self,
        review: str,
        overview: MarketOverview,
        news: Optional[List] = None,
    ) -> str:
        """Inject structured data tables into the corresponding LLM prose sections."""
        # Build data blocks
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)
        patterns = (
            _ENGLISH_SECTION_PATTERNS
            if self._get_review_language() == "en"
            else _CHINESE_SECTION_PATTERNS
        )

        if stats_block:
            review = self._insert_after_section(
                review,
                patterns["market_summary"],
                stats_block,
            )

        if indices_block:
            review = self._insert_after_section(
                review,
                patterns["index_commentary"],
                indices_block,
            )

        if sector_block:
            original_review = review
            review = self._insert_after_section(
                review,
                patterns["sector_highlights"],
                sector_block,
            )
            if review == original_review and sector_block not in review:
                fallback_heading = (
                    "### 4. Sector Highlights"
                    if self._get_review_language() == "en"
                    else "### 三、類股主流"
                )
                review = f"{review.rstrip()}\n{fallback_heading}\n{sector_block}\n"

        return _compact_markdown(review)

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        """Insert a data block at the end of a markdown section (before the next ### heading)."""
        import re
        # Find the heading
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        # Find the next ### heading after this one
        next_heading = re.search(r'\n###\s', text[start:])
        if next_heading:
            insert_pos = start + next_heading.start()
        else:
            # No next heading — append at end
            insert_pos = len(text)
        # Insert the block before the next heading, with spacing
        return text[:insert_pos].rstrip() + '\n' + block + '\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        """Build market statistics block."""
        has_stats = overview.up_count or overview.down_count or overview.total_amount
        has_market_signal = bool(self._supports_market_light() and (overview.indices or has_stats))
        if not has_stats and not has_market_signal:
            return ""
        light = self.build_market_light_snapshot(overview) if has_market_signal else None
        if self._get_review_language() == "en":
            lines = []
            if isinstance(light, dict):
                lines.extend(
                    [
                        f"- **Market Signal**: {light['score']}/100 "
                        f"({light['temperature_label']}, {light['label']})",
                        f"- **Drivers**: {'; '.join(light['reasons'])}",
                        f"- **Guidance**: {light['guidance']}",
                    ]
                )
            if has_stats:
                lines.append(
                    f"- **Breadth**: Advancers {overview.up_count} / Decliners {overview.down_count} / "
                    f"Flat {overview.flat_count}; "
                    f"Limit-up {overview.limit_up_count} / Limit-down {overview.limit_down_count}; "
                    f"Turnover {overview.total_amount:.0f} ({self._get_turnover_unit_label()})"
                )
            return "\n".join(lines)
        lines = []
        score = light["score"] if isinstance(light, dict) else None
        label = light["temperature_label"] if isinstance(light, dict) else ""
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else 0.0
        limit_spread = overview.limit_up_count - overview.limit_down_count
        if isinstance(light, dict) and score is not None:
            lines.extend(
                [
                    f"- **盤面訊號**：{score}/100（{label}，{light['label']}）",
                    f"- **訊號依據**：{'；'.join(light['reasons'])}",
                    f"- **操作建議**：{light['guidance']}",
                ]
            )
        if has_stats:
            lines.extend(
                [
                    "| 指標 | 數值 | 觀察 |",
                    "|------|------|------|",
                    f"| 上漲/下跌/平盤 | {overview.up_count} / {overview.down_count} / {overview.flat_count} | 上漲佔比(不含平盤) {up_ratio:.1%} |",
                    f"| 漲停/跌停 | {overview.limit_up_count} / {overview.limit_down_count} | 漲跌停差 {limit_spread:+d} |",
                    f"| 兩市成交金額 | {overview.total_amount:.0f} 億 | {self._describe_turnover(overview.total_amount)} |",
                ]
            )
        return "\n".join(lines)

    def build_market_light_snapshot(self, overview: MarketOverview) -> Dict[str, Any]:
        """Build a deterministic market-light snapshot from structured breadth data."""
        scores = self._build_market_light_scores(overview)
        score = int(scores["score"])
        temperature_label = str(scores["temperature_label"])
        if score >= 60:
            status = "green"
        elif score >= 40:
            status = "yellow"
        else:
            status = "red"

        if self._get_review_language() == "en":
            label_map = {
                "green": "risk-on",
                "yellow": "balanced",
                "red": "risk-off",
            }
            guidance_map = {
                "green": "Risk appetite is acceptable; focus on leading themes and position discipline.",
                "yellow": "Signals are mixed; keep position sizing moderate and wait for confirmation.",
                "red": "Risk is elevated; prioritize drawdown control and avoid chasing weak rebounds.",
            }
            reasons = self._build_market_light_reasons_en(overview, score)
        else:
            label_map = {
                "green": "可進攻",
                "yellow": "需觀察",
                "red": "偏防守",
            }
            guidance_map = {
                "green": "風險偏好尚可，關注主線延續與部位紀律。",
                "yellow": "訊號分化，控制部位並等待量價確認。",
                "red": "風險偏高，優先控制回檔，避免追高弱反彈。",
            }
            reasons = self._build_market_light_reasons_zh(overview, score)

        snapshot = MarketLightSnapshot(
            region=self.region,
            trade_date=overview.date,
            status=status,
            label=label_map[status],
            score=score,
            temperature_label=temperature_label,
            reasons=reasons,
            guidance=guidance_map[status],
            dimensions=scores["dimensions"],
            data_quality=str(scores["data_quality"]),
        )
        return snapshot.model_dump()

    def _build_market_light_reasons_zh(self, overview: MarketOverview, score: int) -> List[str]:
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = []
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"上漲家數佔比 {up_ratio:.0%}，賺錢效應擴散")
            elif up_ratio <= 0.4:
                reasons.append(f"上漲家數佔比 {up_ratio:.0%}，賠錢效應較強")
            else:
                reasons.append(f"上漲家數佔比 {up_ratio:.0%}，市場分化")
        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            reasons.append(f"主要指數平均漲跌幅 {avg_change:+.2f}%")
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"漲跌停差 {overview.limit_up_count - overview.limit_down_count:+d}")
        if not reasons and overview.total_amount:
            reasons.append(f"成交金額 {overview.total_amount:.0f} 億，{self._describe_turnover(overview.total_amount)}")
        if not reasons:
            reasons.append("結構化漲跌資料有限，按可用行情綜合判斷")
        return reasons[:4]

    def _build_market_light_reasons_en(self, overview: MarketOverview, score: int) -> List[str]:
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = []
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is expanding")
            elif up_ratio <= 0.4:
                reasons.append(f"advancers ratio {up_ratio:.0%}, downside pressure dominates")
            else:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is mixed")
        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            reasons.append(f"average major-index change {avg_change:+.2f}%")
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"limit-up/down spread {overview.limit_up_count - overview.limit_down_count:+d}")
        if not reasons and overview.total_amount:
            reasons.append(f"turnover {overview.total_amount:.0f} ({self._get_turnover_unit_label()})")
        if not reasons:
            reasons.append("limited structured breadth data; using available market inputs")
        return reasons[:4]

    def _build_indices_block(self, overview: MarketOverview) -> str:
        """建立指數行情表格"""
        if not overview.indices:
            return ""
        if self._get_review_language() == "en":
            lines = [
                f"| Index | Last | Change % | Open | High | Low | Amplitude | Turnover ({self._get_turnover_unit_label()}) |",
                "|-------|------|----------|------|------|-----|-----------|-----------------|",
            ]
        else:
            lines = [
                "| 指數 | 最新 | 漲跌幅 | 開盤 | 最高 | 最低 | 振幅 | 成交金額(億) |",
                "|------|------|--------|------|------|------|------|-----------|",
            ]
        for idx in overview.indices:
            arrow = self._get_index_change_arrow(idx.change_pct)
            amount_raw = idx.amount or 0.0
            amount_str = self._format_turnover_value(amount_raw)
            lines.append(
                f"| {idx.name} | {idx.current:.2f} | {arrow} {idx.change_pct:+.2f}% | "
                f"{self._format_optional_number(idx.open)} | {self._format_optional_number(idx.high)} | "
                f"{self._format_optional_number(idx.low)} | {self._format_optional_pct(idx.amplitude)} | {amount_str} |"
            )
        return "\n".join(lines)

    def _build_sector_block(self, overview: MarketOverview) -> str:
        """Build industry and concept ranking blocks."""
        if (
            not overview.top_sectors
            and not overview.bottom_sectors
            and not overview.top_concepts
            and not overview.bottom_concepts
        ):
            return ""
        lines = []
        language = self._get_review_language()

        def append_ranking(title: str, name_label: str, rows: List[Dict]) -> None:
            if not rows:
                return
            lines.extend([
                title,
                f"| {'Rank' if language == 'en' else '排名'} | {name_label} | {'Change' if language == 'en' else '漲跌幅'} |",
                "|------|------|--------|",
            ])
            for rank, item in enumerate(rows[:5], 1):
                lines.append(
                    f"| {rank} | {item.get('name', '-')} | {self._format_signed_pct(item.get('change_pct'))} |"
                )

        if language == "en":
            append_ranking("#### Leading Industry Sectors", "Sector", overview.top_sectors)
            append_ranking("#### Lagging Industry Sectors", "Sector", overview.bottom_sectors)
            append_ranking("#### Leading Concept Themes", "Concept", overview.top_concepts)
            append_ranking("#### Lagging Concept Themes", "Concept", overview.bottom_concepts)
        else:
            append_ranking("#### 產業類股領漲 Top 5", "產業類股", overview.top_sectors)
            append_ranking("#### 產業類股領跌 Top 5", "產業類股", overview.bottom_sectors)
            append_ranking("#### 概念類股領漲 Top 5", "概念類股", overview.top_concepts)
            append_ranking("#### 概念類股領跌 Top 5", "概念類股", overview.bottom_concepts)
        return "\n".join(lines)

    def _build_news_block(self, news: List) -> str:
        """Build a compact source-aware news catalyst list for the rendered report."""
        if not news:
            return ""
        language = self._get_review_language()
        if language == "en":
            lines = [
                "#### News Catalysts",
            ]
        else:
            lines = [
                "#### 近三日市場線索",
            ]

        for idx, item in enumerate(news[:5], 1):
            lines.append(self._format_news_catalyst_line(idx, item, language=language))
        return "\n".join(lines)

    @staticmethod
    def _get_news_field(item: Any, field: str) -> str:
        if hasattr(item, field):
            value = getattr(item, field, "") or ""
        elif isinstance(item, dict):
            value = item.get(field, "") or ""
        else:
            value = ""
        return str(value).strip()

    @classmethod
    def _format_news_catalyst_line(cls, idx: int, item: Any, *, language: str = "zh") -> str:
        fallback_title = "Untitled catalyst" if language == "en" else "未命名線索"
        title = cls._compact_news_text(cls._get_news_field(item, "title"), limit=90) or fallback_title
        source = cls._compact_news_text(cls._get_news_field(item, "source"), limit=40)
        date_text = cls._compact_news_text(cls._get_news_field(item, "published_date"), limit=24)
        url = cls._compact_news_text(cls._get_news_field(item, "url"), limit=0)
        title_text = cls._escape_markdown_link_label(title)
        if url:
            title_text = f"[{title_text}]({url})"
        meta_parts = [part for part in (source, date_text) if part]
        if language == "en":
            meta = f" ({' / '.join(meta_parts)})" if meta_parts else ""
        else:
            meta = f"（{' / '.join(meta_parts)}）" if meta_parts else ""
        return f"- {idx}. {title_text}{meta}"

    @staticmethod
    def _compact_news_text(value: str, *, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if limit <= 0 or len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _format_optional_number(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}"

    @staticmethod
    def _format_optional_pct(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}%"

    @staticmethod
    def _format_signed_pct(value: Any) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{numeric_value:+.2f}%"

    @classmethod
    def _format_ranking_summary(cls, rows: List[Dict], limit: int = 3) -> str:
        parts = []
        for item in (rows or [])[:limit]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            parts.append(f"{name}({cls._format_signed_pct(item.get('change_pct'))})")
        return ", ".join(parts)

    @staticmethod
    def _escape_markdown_link_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")

    @staticmethod
    def _describe_turnover(total_amount: float) -> str:
        if total_amount >= 15000:
            return "高活躍度"
        if total_amount >= 9000:
            return "中等活躍"
        if total_amount > 0:
            return "縮量觀望"
        return "暫無資料"

    def _build_market_light_scores(self, overview: MarketOverview) -> Dict[str, Any]:
        """Build the canonical Market Light scores used by reports and alerts."""

        participants = overview.up_count + overview.down_count
        breadth_available = bool(self.profile.has_market_stats and participants > 0)
        breadth_score = 50
        if breadth_available:
            breadth_score = int(overview.up_count / participants * 100)

        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        index_available = bool(overview.indices and index_changes)
        index_score = 50
        if index_available:
            avg_change = sum(index_changes) / len(index_changes)
            index_score = int(max(0, min(100, 50 + avg_change * 12)))

        limit_total = overview.limit_up_count + overview.limit_down_count
        limit_available = bool(self.profile.has_market_stats and limit_total > 0)
        limit_score = 50
        if limit_available:
            limit_score = int(overview.limit_up_count / limit_total * 100)

        dimensions = {
            "breadth": {"score": breadth_score, "available": breadth_available},
            "index": {"score": index_score, "available": index_available},
            "limit": {"score": limit_score, "available": limit_available},
        }

        if not index_available:
            data_quality = "unavailable"
        elif all(dimension["available"] for dimension in dimensions.values()):
            data_quality = "ok"
        else:
            data_quality = "partial"

        score = int(round(breadth_score * 0.45 + index_score * 0.35 + limit_score * 0.20))
        if self._get_review_language() == "en":
            if score >= 70:
                label = "risk-on"
            elif score >= 55:
                label = "constructive"
            elif score >= 40:
                label = "mixed"
            else:
                label = "defensive"
        else:
            if score >= 70:
                label = "強勢"
            elif score >= 55:
                label = "偏暖"
            elif score >= 40:
                label = "震盪"
            else:
                label = "偏弱"
        return {
            "score": score,
            "temperature_label": label,
            "dimensions": dimensions,
            "data_quality": data_quality,
        }

    def _build_market_temperature(self, overview: MarketOverview) -> tuple[int, str]:
        scores = self._build_market_light_scores(overview)
        score = int(scores["score"])
        label = str(scores["temperature_label"])
        return score, label

    def _build_output_template_sections(self, review_language: str) -> str:
        """Build LLM output sections according to market data capabilities."""
        if review_language == "en":
            if self.profile.has_market_stats and self.profile.has_sector_rankings:
                return """### 3. Fund Flows
(Interpret what turnover, participation, and flow signals imply.)

### 4. Sector Highlights
(Distinguish industry-sector moves from concept/theme moves, then analyze drivers and persistence.)

### 5. Outlook
(Provide the near-term outlook based on price action and news.)

### 6. Risk Alerts
(List the main risks to monitor.)

### 7. Strategy Plan
(Provide an offensive/balanced/defensive stance, a position-sizing guideline, one invalidation trigger, and end with "For reference only, not investment advice.")"""

            section_number = 3
            sections: List[str] = []
            if self.profile.has_market_stats:
                sections.append(f"""### {section_number}. Fund Flows
(Interpret only the provided turnover, participation, breadth, and flow signals.)""")
                section_number += 1
            if self.profile.has_sector_rankings:
                sections.append(f"""### {section_number}. Sector Highlights
(Analyze only the provided industry-sector and concept/theme rankings.)""")
                section_number += 1
            sections.extend([
                f"""### {section_number}. News Catalysts
(Connect recent news to index price action and macro/external-market clues. Do not infer unsupported breadth, fund-flow, or sector-ranking data.)""",
                f"""### {section_number + 1}. Outlook
(Provide the near-term outlook based on index price action and the available news.)""",
                f"""### {section_number + 2}. Risk Alerts
(List the main risks to monitor.)""",
                f"""### {section_number + 3}. Strategy Plan
(Provide an offensive/balanced/defensive stance, a position-sizing guideline, one invalidation trigger, and end with "For reference only, not investment advice.")""",
            ])
            return "\n".join(sections)

        if self.profile.has_market_stats and self.profile.has_sector_rankings:
            return """### 三、類股主流
（區分產業類股與概念題材，分析領漲/領跌背後的邏輯、持續性和是否形成主線）
### 四、資金與情緒
（解讀成交金額、漲跌停結構、市場廣度和風險偏好）
### 五、消息催化
（結合近三日新聞，提煉真正影響明日交易的催化或擾動）
### 六、明日交易計畫
（給出進攻/均衡/防守結論、部位區間、關注方向、迴避方向和一個觸發失效條件）
### 七、風險提示
（列出需要關注的風險點；最後補充「建議僅供參考，不構成投資建議」。）"""

        numerals = ["一", "二", "三", "四", "五", "六", "七", "八"]
        section_number = 3
        sections: List[str] = []

        def add_section(title: str, hint: str) -> None:
            nonlocal section_number
            sections.append(f"### {numerals[section_number - 1]}、{title}\n{hint}")
            section_number += 1

        if self.profile.has_sector_rankings:
            add_section("類股主流", "（僅分析已提供的產業類股與概念題材榜單，不擴展未提供的資料）")
        if self.profile.has_market_stats:
            add_section("資金與情緒", "（僅解讀已提供的成交金額、漲跌停結構、市場廣度和風險偏好資料）")
        add_section(
            "消息催化",
            "（結合近三日新聞和指數表現，提煉真正影響明日交易的催化或擾動；不要推斷未提供的資金流向、市場廣度或類股榜）",
        )
        add_section("明日交易計畫", "（給出進攻/均衡/防守結論、部位區間、關注方向、迴避方向和一個觸發失效條件）")
        add_section("風險提示", "（列出需要關注的風險點；最後補充「建議僅供參考，不構成投資建議」。）")
        return "\n".join(sections)

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """建立復盤報告 Prompt"""
        review_language = self._get_review_language()
        # Korean reuses the English structural template but the model is told to
        # write the entire shell, headings, guidance and conclusion in Korean.
        shell_language_label = "Korean (한국어)" if self._get_output_language() == "ko" else "English"

        # 指數行情資訊（簡潔格式，不用emoji）
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"

        # 類股資訊
        top_sectors_text = self._format_ranking_summary(overview.top_sectors)
        bottom_sectors_text = self._format_ranking_summary(overview.bottom_sectors)
        top_concepts_text = self._format_ranking_summary(overview.top_concepts)
        bottom_concepts_text = self._format_ranking_summary(overview.bottom_concepts)

        # 新聞資訊 - 支援 SearchResult 物件或字典
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            # 相容 SearchResult 物件和字典
            title = self._compact_news_text(self._get_news_field(n, "title"), limit=90)
            snippet = self._compact_news_text(self._get_news_field(n, "snippet"), limit=220)
            source = self._compact_news_text(self._get_news_field(n, "source"), limit=60)
            published_date = self._compact_news_text(self._get_news_field(n, "published_date"), limit=30)
            url = self._compact_news_text(self._get_news_field(n, "url"), limit=180)
            meta_parts = [part for part in (source, published_date) if part]
            meta = f" ({' / '.join(meta_parts)})" if meta_parts else ""
            url_line = f"\n   URL: {url}" if url else ""
            news_text += f"{i}. {title}{meta}\n   {snippet or '-'}{url_line}\n"

        # 依 region 組裝市場概況與類股區塊（美股/港股/日韓無漲跌家數、類股資料）
        stats_block = ""
        sector_block = ""
        data_limits_block = ""
        if review_language == "en":
            if self.profile.has_market_stats:
                stats_block = f"""## Market Breadth
- Advancers: {overview.up_count} | Decliners: {overview.down_count} | Flat: {overview.flat_count}
- Limit-up: {overview.limit_up_count} | Limit-down: {overview.limit_down_count}
- Turnover: {overview.total_amount:.0f} ({self._get_turnover_unit_label()})"""

            if self.profile.has_sector_rankings:
                sector_block = f"""## Sector / Theme Performance
Industry leading: {top_sectors_text if top_sectors_text else "N/A"}
Industry lagging: {bottom_sectors_text if bottom_sectors_text else "N/A"}
Concept leading: {top_concepts_text if top_concepts_text else "N/A"}
Concept lagging: {bottom_concepts_text if bottom_concepts_text else "N/A"}"""

            data_limit_lines = []
            if not self.profile.has_market_stats:
                data_limit_lines.append(
                    "- Market breadth, aggregate turnover, participation, and fund-flow signals are not available for this market."
                )
            if not self.profile.has_sector_rankings:
                data_limit_lines.append("- Sector/theme ranking data is not available for this market.")
            if data_limit_lines:
                data_limits_block = "## Data Limits\n" + "\n".join(data_limit_lines)
        else:
            if self.profile.has_market_stats:
                stats_block = f"""## 市場概況
- 上漲: {overview.up_count} 家 | 下跌: {overview.down_count} 家 | 平盤: {overview.flat_count} 家
- 漲停: {overview.limit_up_count} 家 | 跌停: {overview.limit_down_count} 家
- 兩市成交金額: {overview.total_amount:.0f} 億元"""

            if self.profile.has_sector_rankings:
                sector_block = f"""## 類股表現
產業領漲: {top_sectors_text if top_sectors_text else "暫無資料"}
產業領跌: {bottom_sectors_text if bottom_sectors_text else "暫無資料"}
概念領漲: {top_concepts_text if top_concepts_text else "暫無資料"}
概念領跌: {bottom_concepts_text if bottom_concepts_text else "暫無資料"}"""

            data_limit_lines = []
            if not self.profile.has_market_stats:
                data_limit_lines.append("- 該市場暫無漲跌家數、漲跌停、成交金額彙總、參與度或資金流向訊號。")
            if not self.profile.has_sector_rankings:
                data_limit_lines.append("- 該市場暫無產業類股/概念題材漲跌榜。")
            if data_limit_lines:
                data_limits_block = "## 資料邊界\n" + "\n".join(data_limit_lines)

        data_no_indices_hint = (
            "注意：由於行情資料取得失敗，請主要根據【市場新聞】進行定性分析和總結，不要編造具體的指數點位。"
            if not indices_text
            else ""
        )
        if review_language == "en":
            data_no_indices_hint = (
                "Note: Market data fetch failed. Rely mainly on [Market News] for qualitative analysis. Do not invent index levels."
                if not indices_text
                else ""
            )
            indices_placeholder = indices_text if indices_text else "No index data (API error)"
            news_placeholder = news_text if news_text else "No relevant news"
            data_boundary_requirement = (
                "- Respect Data Limits: do not invent or over-interpret unsupported breadth, fund-flow, turnover, participation, or sector-ranking data.\n"
                if data_limits_block
                else ""
            )
            market_summary_hint = (
                "2-3 sentences summarizing overall market tone, index moves, and liquidity."
                if self.profile.has_market_stats
                else "2-3 sentences summarizing overall market tone, index moves, and available news context."
            )
        else:
            indices_placeholder = indices_text if indices_text else "暫無指數資料（介面異常）"
            news_placeholder = news_text if news_text else "暫無相關新聞"
            data_boundary_requirement = (
                "- 嚴格遵守資料邊界：未提供漲跌家數、資金流向、成交金額彙總或類股榜時，不要編造或過度解讀。\n"
                if data_limits_block
                else ""
            )
            market_summary_hint = (
                "2-3句話概括指數、漲跌家數、成交金額和情緒溫度，明確「強勢/偏暖/震盪/偏弱」判斷"
                if self.profile.has_market_stats
                else "2-3句話概括指數表現、新聞線索和整體風險狀態，不要補寫未提供的市場廣度或資金流向資料"
            )

        output_template_sections = self._build_output_template_sections(review_language)
        zh_market_scope_name = self._get_market_scope_name("zh")
        zh_report_title = f"{overview.date} 大盤復盤"
        if self.region in ("jp", "kr"):
            zh_report_title = f"{overview.date} {zh_market_scope_name}大盤復盤"
        workflow_hint = (
            "報告要像交易員盤後工作台：先給結論，再按資料表、主線、催化、計畫展開"
            if self.profile.has_market_stats or self.profile.has_sector_rankings
            else "報告要像交易員盤後工作台：先給結論，再按指數、新聞催化和計畫展開"
        )

        if review_language == "en":
            report_title = self._get_review_title(overview.date).removeprefix("## ").strip()
            return f"""You are a professional {self._get_market_scope_name('en')} analyst. Please produce a concise market recap report based on the data below.

[Requirements]
- Output pure Markdown only
- No JSON
- No code blocks
- Use emoji sparingly in headings (at most one per heading)
- The entire fixed shell, headings, guidance, and conclusion must be in {shell_language_label}
{data_boundary_requirement}

---

# Today's Market Data

## Date
{overview.date}

## Major Indices
{indices_placeholder}

{stats_block}

{sector_block}

{data_limits_block}

## Market News
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# Output Template (follow this structure)

## {report_title}

### 1. Market Summary
({market_summary_hint})

### 2. Index Commentary
({self._get_index_hint()})

{output_template_sections}

---

Output the report content directly, no extra commentary.
"""

        # 陸股場景使用中文提示語
        return f"""你是一位專業的{self._get_market_scope_name('zh')}分析師，請根據以下資料產生一份結構化的{self._get_market_scope_name('zh')}大盤復盤報告。

【重要】輸出要求：
- 必須輸出純 Markdown 文字格式
- 禁止輸出 JSON 格式
- 禁止輸出程式碼區塊
- emoji 僅在標題處少量使用（每個標題最多1個）
- {workflow_hint}
- 全文不要空行：標題、段落與表格之間只用單一換行，不要留空白行
- 不要重複列出已由系統注入的表格資料；正文負責解釋表格背後的含義
{data_boundary_requirement}

---

# 今日市場資料

## 日期
{overview.date}

## 主要指數
{indices_placeholder}

{stats_block}

{sector_block}

{data_limits_block}

## 市場新聞
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# 輸出格式模板（請嚴格按此格式輸出）

## {zh_report_title}
> 一句話給出今日市場狀態、核心矛盾和明日優先觀察方向。
### 一、盤面總覽
（{market_summary_hint}）
### 二、指數結構
（{self._get_index_hint()}，說明誰在護盤、誰在拖累，以及關鍵支撐/壓力）
{output_template_sections}

---

請直接輸出復盤報告內容，不要輸出其他說明文字。
"""

    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """使用模板產生復盤報告（無大型語言模型時的備選方案）"""
        template_language = self._get_template_review_language()
        mood_code = self.profile.mood_index_code
        # 根據 mood_index_code 查找對應指數
        # cn: mood_code="000001"，idx.code 可能為 "sh000001"（以 mood_code 結尾）
        # us: mood_code="SPX"，idx.code 直接為 "SPX"
        mood_index = next(
            (
                idx
                for idx in overview.indices
                if idx.code == mood_code or idx.code.endswith(mood_code)
            ),
            None,
        )
        if mood_index:
            if mood_index.change_pct > 1:
                market_mood = self._get_market_mood_text("strong_up", template_language)
            elif mood_index.change_pct > 0:
                market_mood = self._get_market_mood_text("mild_up", template_language)
            elif mood_index.change_pct > -1:
                market_mood = self._get_market_mood_text("mild_down", template_language)
            else:
                market_mood = self._get_market_mood_text("strong_down", template_language)
        else:
            market_mood = self._get_market_mood_text("range", template_language)

        # 指數行情（簡潔格式）
        indices_text = ""
        for idx in overview.indices[:4]:
            marker = self._get_index_change_arrow(idx.change_pct)
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({marker} {idx.change_pct:+.2f}%)\n"

        # 類股資訊
        separator = ", " if template_language == "en" else "、"
        top_text = separator.join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = separator.join([s['name'] for s in overview.bottom_sectors[:3]])
        top_concept_text = separator.join([s['name'] for s in overview.top_concepts[:3]])
        bottom_concept_text = separator.join([s['name'] for s in overview.bottom_concepts[:3]])

        if template_language == "en":
            stats_section = ""
            if self._supports_market_light() or self.profile.has_market_stats:
                stats_block = self._build_stats_block(overview)
                if stats_block:
                    stats_section = f"""
### 3. Breadth & Liquidity
{stats_block}
"""
            sector_section = ""
            if self.profile.has_sector_rankings and (top_text or bottom_text or top_concept_text or bottom_concept_text):
                sector_section = f"""
### 4. Sector / Theme Highlights
- **Industry Leaders**: {top_text or "N/A"}
- **Industry Laggards**: {bottom_text or "N/A"}
- **Concept Leaders**: {top_concept_text or "N/A"}
- **Concept Laggards**: {bottom_concept_text or "N/A"}
"""
            market_names = {
                "us": "US Market Recap",
                "hk": "HK Market Recap",
                "jp": "Japan Market Recap",
                "kr": "Korea Market Recap",
            }
            market_name = market_names.get(self.region, "A-share Market Recap")
            report = f"""## {overview.date} {market_name}

### 1. Market Summary
Today's {self._get_market_scope_name(template_language)} showed **{market_mood}**.

### 2. Major Indices
{indices_text or "- No index data available"}
{stats_section}
{sector_section}
### 5. Risk Alerts
Market conditions can change quickly. The data above is for reference only and does not constitute investment advice.

{self._get_strategy_markdown_block(template_language)}

---
*Review Time: {datetime.now().strftime('%H:%M')}*
"""
            return _compact_markdown(report)

        market_labels = {"cn": "陸股", "us": "美股", "hk": "港股", "jp": "日股", "kr": "韓股"}
        market_label = market_labels.get(self.region, "陸股")
        dashboard_block = (
            self._build_stats_block(overview)
            if self._supports_market_light() or self.profile.has_market_stats
            else ""
        )
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview) if self.profile.has_sector_rankings else ""
        summary_focus = (
            "指數承接、成交金額變化和類股持續性"
            if self.profile.has_market_stats and self.profile.has_sector_rankings
            else "指數承接、消息催化和整體風險狀態"
        )
        market_summary_block = (
            dashboard_block
            if dashboard_block
            else (
                "暫無市場廣度資料。"
                if self.profile.has_market_stats
                else "- 當前以主要指數與可用新聞線索評估整體風險狀態。"
            )
        )
        sector_section = (
            f"""
### 三、類股主流
{sector_block or "- 暫無類股漲跌榜資料。"}"""
            if self.profile.has_sector_rankings
            else ""
        )
        funds_section = (
            """
### 四、資金與情緒
- 結合成交金額和漲跌家數看，當前更適合等待確認，避免僅憑單一熱點追高。"""
            if self.profile.has_market_stats
            else ""
        )
        report = f"""## {overview.date} 大盤復盤
> 今日{market_label}市場整體呈現**{market_mood}**態勢，優先觀察{summary_focus}。
### 一、盤面總覽
{market_summary_block}
### 二、指數結構
{indices_block or indices_text or "暫無指數資料。"}{sector_section}{funds_section}
### 五、消息催化
- 暫無可用新聞時，應降低對題材持續性的確定性判斷。
{self._get_strategy_markdown_block(template_language)}
### 七、風險提示
- 市場有風險，投資需謹慎。以上資料僅供參考，不構成投資建議。
---
*復盤時間: {datetime.now().strftime('%H:%M')}*
"""
        return _compact_markdown(report)

    def _run_daily_review_parts(self) -> MarketLightReviewResult:
        """Run market review once and keep report/snapshot on the same overview."""
        logger.info("========== 開始大盤復盤分析 ==========")

        # 1. 取得市場概覽
        overview = self.get_market_overview()

        # 2. 搜尋市場新聞
        news = self.search_market_news()
        news = self._merge_persisted_market_intelligence(news)

        # 3. 產生復盤報告
        report = self.generate_market_review(overview, news)
        snapshot = self.build_market_light_snapshot(overview) if self._supports_market_light() else None
        structured_payload = self.build_market_review_payload(
            overview,
            news,
            report,
            snapshot,
        )

        logger.info("========== 大盤復盤分析完成 ==========")

        return MarketLightReviewResult(
            overview=overview,
            report=report,
            market_light_snapshot=snapshot,
            structured_payload=structured_payload,
        )

    def _merge_persisted_market_intelligence(self, news: List) -> List:
        """Merge local persisted market intelligence and search news with bounded prompt/payload slot preservation."""
        search_news = list(news or [])
        merged_local = []
        seen_urls = {
            self._get_news_field(item, "url")
            for item in search_news
            if self._get_news_field(item, "url")
        }
        try:
            service = IntelligenceService(config=self.config)
            service.refresh_auto_sources()
            payload = service.list_items(
                scope_type="market",
                market=self.region,
                published_days=max(1, int(self.config.get_effective_news_window_days() or 1)),
                page=1,
                page_size=6,
            )
            for item in payload.get("items", []):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "")
                if url and url in seen_urls:
                    continue
                seen_urls.add(url)
                merged_local.append({
                    "title": item.get("title") or "未命名資訊",
                    "snippet": item.get("summary") or "",
                    "source": item.get("source") or item.get("source_name") or "local-intel",
                    "published_date": item.get("published_at") or "",
                    "url": "" if url.startswith("no-url:intel:") else url,
                })
        except Exception as exc:
            logger.debug("[大盤] %s action=load_local_intelligence status=failed error=%s", self._log_context(), exc)
        merged_news = []
        merged_local_index = 0
        merged_search_index = 0
        while merged_local_index < len(merged_local) or merged_search_index < len(search_news):
            if merged_local_index < len(merged_local):
                merged_news.append(merged_local[merged_local_index])
                merged_local_index += 1
            if merged_search_index < len(search_news):
                merged_news.append(search_news[merged_search_index])
                merged_search_index += 1
        return merged_news

    def run_daily_review(self) -> str:
        """
        執行每日大盤復盤流程

        Returns:
            復盤報告文字
        """
        return self.run_daily_review_with_snapshot().report

    def run_daily_review_with_snapshot(self) -> MarketLightReviewResult:
        """Run daily review and return the report plus its structured Market Light snapshot."""
        return self._run_daily_review_parts()


# 測試入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )

    analyzer = MarketAnalyzer()

    # 測試取得市場概覽
    overview = analyzer.get_market_overview()
    print(f"\n=== 市場概覽 ===")
    print(f"日期: {overview.date}")
    print(f"指數數量: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"上漲: {overview.up_count} | 下跌: {overview.down_count}")
    print(f"成交金額: {overview.total_amount:.0f}億")

    # 測試產生模板報告
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== 復盤報告 ===")
    print(report)
