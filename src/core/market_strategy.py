# -*- coding: utf-8 -*-
"""Market strategy blueprints for CN/HK/US daily market recap."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class StrategyDimension:
    """Single strategy dimension used by market recap prompts."""

    name: str
    objective: str
    checkpoints: List[str]


@dataclass(frozen=True)
class MarketStrategyBlueprint:
    """Region specific market strategy blueprint."""

    region: str
    title: str
    positioning: str
    principles: List[str]
    dimensions: List[StrategyDimension]
    action_framework: List[str]

    def to_prompt_block(self) -> str:
        """Render blueprint as prompt instructions."""
        principles_text = "\n".join([f"- {item}" for item in self.principles])
        action_text = "\n".join([f"- {item}" for item in self.action_framework])

        dims = []
        for dim in self.dimensions:
            checkpoints = "\n".join([f"  - {cp}" for cp in dim.checkpoints])
            dims.append(f"- {dim.name}: {dim.objective}\n{checkpoints}")
        dimensions_text = "\n".join(dims)

        return (
            f"## Strategy Blueprint: {self.title}\n"
            f"{self.positioning}\n"
            f"### Strategy Principles\n{principles_text}\n"
            f"### Analysis Dimensions\n{dimensions_text}\n"
            f"### Action Framework\n{action_text}"
        )

    def to_markdown_block(self) -> str:
        """Render blueprint as markdown section for template fallback report."""
        dims = "\n".join([f"- **{dim.name}**: {dim.objective}" for dim in self.dimensions])
        section_title = "### VI. Strategy Framework" if self.region == "us" else "### 六、策略框架"
        return f"{section_title}\n{dims}"


CN_BLUEPRINT = MarketStrategyBlueprint(
    region="cn",
    title="陸股市場三段式復盤策略",
    positioning="聚焦指數趨勢、資金博弈與類股輪動，形成次日交易計畫。",
    principles=[
        "先看指數方向，再看量能結構，最後看類股持續性。",
        "結論必須映射到部位、節奏與風險控制動作。",
        "判斷使用當日資料與近3日新聞，不臆測未驗證資訊。",
    ],
    dimensions=[
        StrategyDimension(
            name="趨勢結構",
            objective="判斷市場處於上升、震盪還是防守階段。",
            checkpoints=["上證/深證/創業板是否同向", "放量上漲或縮量下跌是否成立", "關鍵支撐阻力是否被突破"],
        ),
        StrategyDimension(
            name="資金情緒",
            objective="識別短線風險偏好與情緒溫度。",
            checkpoints=["漲跌家數與漲跌停結構", "成交金額是否擴張", "高位股是否出現分歧"],
        ),
        StrategyDimension(
            name="主線類股",
            objective="提煉可交易主線與規避方向。",
            checkpoints=["領漲類股是否具備事件催化", "類股內部是否有龍頭帶動", "領跌類股是否擴散"],
        ),
    ],
    action_framework=[
        "進攻：指數共振上行 + 成交金額放大 + 主線強化。",
        "均衡：指數分化或縮量震盪，控制部位並等待確認。",
        "防守：指數轉弱 + 領跌擴散，優先風控與減碼。",
    ],
)

US_BLUEPRINT = MarketStrategyBlueprint(
    region="us",
    title="US Market Regime Strategy",
    positioning="Focus on index trend, macro narrative, and sector rotation to define next-session risk posture.",
    principles=[
        "Read market regime from S&P 500, Nasdaq, and Dow alignment first.",
        "Separate beta move from theme-driven alpha rotation.",
        "Translate recap into actionable risk-on/risk-off stance with clear invalidation points.",
    ],
    dimensions=[
        StrategyDimension(
            name="Trend Regime",
            objective="Classify the market as momentum, range, or risk-off.",
            checkpoints=[
                "Are SPX/NDX/DJI directionally aligned",
                "Did volume confirm the move",
                "Are key index levels reclaimed or lost",
            ],
        ),
        StrategyDimension(
            name="Macro & Flows",
            objective="Map policy/rates narrative into equity risk appetite.",
            checkpoints=[
                "Treasury yield and USD implications",
                "Breadth and leadership concentration",
                "Defensive vs growth factor rotation",
            ],
        ),
        StrategyDimension(
            name="Sector Themes",
            objective="Identify persistent leaders and vulnerable laggards.",
            checkpoints=[
                "AI/semiconductor/software trend persistence",
                "Energy/financials sensitivity to macro data",
                "Volatility signals from VIX and large-cap earnings",
            ],
        ),
    ],
    action_framework=[
        "Risk-on: broad index breakout with expanding participation.",
        "Neutral: mixed index signals; focus on selective relative strength.",
        "Risk-off: failed breakouts and rising volatility; prioritize capital preservation.",
    ],
)

HK_BLUEPRINT = MarketStrategyBlueprint(
    region="hk",
    title="港股市場三段式復盤策略",
    positioning="聚焦恒生指數趨勢、南向資金博弈與類股輪動，形成次日交易計畫。",
    principles=[
        "先看恒指/恒科/國企指數方向，再看南向資金情緒，最後看類股持續性。",
        "結論必須映射到部位、節奏與風險控制動作。",
        "判斷使用當日資料與近3日新聞，不臆測未驗證資訊。",
    ],
    dimensions=[
        StrategyDimension(
            name="趨勢結構",
            objective="判斷市場處於上升、震盪還是防守階段。",
            checkpoints=["恒指/恒科/國企指數是否同向", "放量上漲或縮量下跌是否成立", "關鍵支撐阻力是否被突破"],
        ),
        StrategyDimension(
            name="資金情緒",
            objective="識別南向資金風險偏好與情緒溫度。",
            checkpoints=["南向資金淨流入方向與規模", "港元匯率與中國大陸政策含義", "市場廣度與龍頭集中度"],
        ),
        StrategyDimension(
            name="主線類股",
            objective="提煉可交易主線與規避方向。",
            checkpoints=["科技/網路平台趨勢持續性", "金融/地產對政策轉向的敏感度", "防禦與成長因子輪動"],
        ),
    ],
    action_framework=[
        "進攻：恒指共振上行 + 南向資金持續流入 + 主線強化。",
        "均衡：指數分化或縮量震盪，控制部位並等待確認。",
        "防守：指數轉弱 + 波動率上升，優先風控與減碼。",
    ],
)


JP_BLUEPRINT = MarketStrategyBlueprint(
    region="jp",
    title="日本市場三段式復盤策略",
    positioning="聚焦日經225、東證指數、匯率與全球風險偏好，形成次日交易計畫。",
    principles=[
        "先看日經225與TOPIX是否同向，再看日元、半導體/出口鏈與金融股表現。",
        "把指數結論映射到部位、節奏與風險控制動作。",
        "只基於可得指數、新聞和價格行為判斷，不臆造市場廣度或類股統計。",
    ],
    dimensions=[
        StrategyDimension(
            name="趨勢結構",
            objective="判斷日本市場處於上攻、震盪還是防守階段。",
            checkpoints=["日經225/TOPIX是否同向", "指數是否突破或跌破關鍵區間", "大盤權重與成長鏈是否共振"],
        ),
        StrategyDimension(
            name="宏觀與匯率",
            objective="識別日元、利率和全球風險偏好對權益市場的影響。",
            checkpoints=["日元方向對出口鏈的影響", "日本央行和美債利率敘事", "海外科技股與半導體鏈映射"],
        ),
        StrategyDimension(
            name="主題線索",
            objective="提煉可延續主線與需要規避的擁擠方向。",
            checkpoints=["半導體/自動化/汽車鏈持續性", "金融與內需股是否輪動", "新聞催化是否支撐價格行為"],
        ),
    ],
    action_framework=[
        "進攻：主要指數共振上行 + 外部風險偏好改善 + 主線強化。",
        "均衡：指數分化或匯率擾動，降低追漲並等待確認。",
        "防守：主要指數轉弱或外部風險升溫，優先控制部位。",
    ],
)

KR_BLUEPRINT = MarketStrategyBlueprint(
    region="kr",
    title="韓國市場三段式復盤策略",
    positioning="聚焦 KOSPI、KOSDAQ、半導體權重與全球科技風險偏好，形成次日交易計畫。",
    principles=[
        "先看 KOSPI/KOSDAQ 是否同向，再看三星電子、SK 海力士等權重線索。",
        "區分指數 beta、半導體週期和成長股風險偏好的貢獻。",
        "只基於可得指數、新聞和價格行為判斷，不臆造市場廣度或類股統計。",
    ],
    dimensions=[
        StrategyDimension(
            name="趨勢結構",
            objective="判斷韓國市場處於上攻、震盪還是防守階段。",
            checkpoints=["KOSPI/KOSDAQ 是否同向", "權重股是否支撐指數", "關鍵支撐阻力是否被突破"],
        ),
        StrategyDimension(
            name="科技週期",
            objective="識別半導體、AI 硬體和全球科技股對韓國市場的映射。",
            checkpoints=["記憶體/半導體鏈新聞催化", "美股科技方向聯動", "外資風險偏好變化"],
        ),
        StrategyDimension(
            name="主題線索",
            objective="提煉可延續主線與需要規避的擁擠方向。",
            checkpoints=["電池/汽車/網路是否輪動", "KOSDAQ 成長股風險偏好", "新聞催化是否支撐價格行為"],
        ),
    ],
    action_framework=[
        "進攻：KOSPI/KOSDAQ 共振上行 + 科技權重確認 + 外部風險偏好改善。",
        "均衡：指數或權重股分化，控制部位並等待確認。",
        "防守：科技權重轉弱或外部風險升溫，優先控制回檔。",
    ],
)

def get_market_strategy_blueprint(region: str) -> MarketStrategyBlueprint:
    """Return strategy blueprint by market region."""
    if region == "us":
        return US_BLUEPRINT
    if region == "hk":
        return HK_BLUEPRINT
    if region == "jp":
        return JP_BLUEPRINT
    if region == "kr":
        return KR_BLUEPRINT
    return CN_BLUEPRINT
