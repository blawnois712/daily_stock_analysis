# -*- coding: utf-8 -*-
"""
Discord 發送提醒服務

職責：
1. 通過 webhook 或 Discord bot API 發送 Discord 訊息
"""
import logging
import time
from typing import Optional

import requests

from src.config import Config
from src.formatters import (
    MIN_MAX_WORDS,
    chunk_content_by_max_words,
    strip_hidden_markdown_metadata,
)


logger = logging.getLogger(__name__)


DISCORD_MAX_CONTENT_LENGTH = 2000
DISCORD_MAX_RETRIES = 3
DISCORD_CHUNK_SLEEP_SECONDS = 1
# 緊湊排版：發送前移除多餘空行，手機上一屏看完（程式碼區塊內保留）
DISCORD_COMPACT_LAYOUT = True

# 發送前統一轉繁體（台灣用語）。優先使用 OpenCC（pip install opencc-python-reimplemented），
# 未安裝時退回下方內建對照表；兩者之後都會再套用台灣交易用語對照。
DISCORD_FORCE_TRADITIONAL = True
_S2T_SIMPLIFIED = (
    "与专业两严个为义亏亿仅仓优传价体关写冲况净减凭划则创别务动势区单压参发变叙号后响"
    "国场块处备复宽对将币师应废并开异弃弹强当态总执扩扰护报换据撑数断无时显暂权条构标"
    "汇洁测涨温点炼热状现琼盘码确离称简纪纯纳线组结给统绪续维综编缩职联荡获补装观览触"
    "计认议讯论设证评识词试话该语说请读谁调谨负责败货资赚趋跃转轮较辑输边达过还进适选"
    "逻释钱键间闻险韩须领题额风于内实员错误视尝龙头规东经验带储导电拥挤车链债际网广周"
    "记阶节产御长张献贡递归类丢书从会检环种积级细缀缓这隐页项预飞万临乐买仪侧决准剧劲"
    "卖历启坏坚够录彻恶惯愿战抛损机杀来极栏术沪浓润潜独监税稳筹红约继绩缘舆营蕴虽轻运"
    "远迹随难饮飙驱阴库庞属倾质财赁虑苹习帮伤灾烦扬轨钟针铁银铜钢镇阅阔队阳顺顾顿频颗"
    "饱馆骗鱼鸟鸡麦黄齐贝贪贯贴贷贸赏赔赢软载辅辆辩迁脑药荐见觉订讨训访许诊诚诱谈谓谢"
    "谱贵帐户烧满灭济汉湿兴举乱"
)
_S2T_TRADITIONAL = (
    "與專業兩嚴個為義虧億僅倉優傳價體關寫衝況淨減憑劃則創別務動勢區單壓參發變敘號後響"
    "國場塊處備復寬對將幣師應廢並開異棄彈強當態總執擴擾護報換據撐數斷無時顯暫權條構標"
    "匯潔測漲溫點煉熱狀現瓊盤碼確離稱簡紀純納線組結給統緒續維綜編縮職聯盪獲補裝觀覽觸"
    "計認議訊論設證評識詞試話該語說請讀誰調謹負責敗貨資賺趨躍轉輪較輯輸邊達過還進適選"
    "邏釋錢鍵間聞險韓須領題額風於內實員錯誤視嘗龍頭規東經驗帶儲導電擁擠車鏈債際網廣週"
    "記階節產禦長張獻貢遞歸類丟書從會檢環種積級細綴緩這隱頁項預飛萬臨樂買儀側決準劇勁"
    "賣歷啟壞堅夠錄徹惡慣願戰拋損機殺來極欄術滬濃潤潛獨監稅穩籌紅約繼績緣輿營蘊雖輕運"
    "遠跡隨難飲飆驅陰庫龐屬傾質財賃慮蘋習幫傷災煩揚軌鐘針鐵銀銅鋼鎮閱闊隊陽順顧頓頻顆"
    "飽館騙魚鳥雞麥黃齊貝貪貫貼貸貿賞賠贏軟載輔輛辯遷腦藥薦見覺訂討訓訪許診誠誘談謂謝"
    "譜貴帳戶燒滿滅濟漢濕興舉亂"
)
_S2T_TABLE = str.maketrans(_S2T_SIMPLIFIED, _S2T_TRADITIONAL)
_TW_TERMS = (
    ("板塊", "類股"), ("信號", "訊號"), ("A股", "陸股"), ("倉位", "部位"), ("回撤", "回檔"),
    ("成交額", "成交金額"), ("數據", "資料"), ("信息", "資訊"), ("接口", "介面"),
    ("現金分紅", "現金股利"), ("分紅", "配息"), ("股息率", "殖利率"), ("同比", "年增率"),
    ("換手率", "週轉率"), ("買入", "買進"), ("減倉", "減碼"), ("兜底", "保底"),
    ("關系", "關係"), ("儀表盤", "儀表板"), ("鏈接", "連結"),
    ("制造", "製造"), ("制作", "製作"), ("精制", "精製"), ("制品", "製品"),
    ("實時", "即時"), ("計劃", "計畫"), ("分布", "分佈"),
    ("軟件", "軟體"), ("硬件", "硬體"), ("默認", "預設"), ("設置", "設定"), ("質量", "品質"),
)
_opencc_converter = None
_opencc_checked = False


def _get_opencc_converter():
    global _opencc_converter, _opencc_checked
    if _opencc_checked:
        return _opencc_converter
    _opencc_checked = True
    try:
        from opencc import OpenCC
    except ImportError:
        return None
    for config in ("s2twp", "s2twp.json"):
        try:
            _opencc_converter = OpenCC(config)
            break
        except Exception:
            continue
    return _opencc_converter


def to_traditional(text: str) -> str:
    """簡體轉台灣繁體；內建對照表只涵蓋常用字，建議安裝 OpenCC 以獲得完整詞彙轉換。"""
    converter = _get_opencc_converter()
    text = converter.convert(text) if converter else text.translate(_S2T_TABLE)
    for old, new in _TW_TERMS:
        text = text.replace(old, new)
    return text



class DiscordSender:

    def __init__(self, config: Config):
        """
        初始化 Discord 設定

        Args:
            config: 設定物件
        """
        self._discord_config = {
            'bot_token': getattr(config, 'discord_bot_token', None),
            'channel_id': getattr(config, 'discord_main_channel_id', None),
            'webhook_url': getattr(config, 'discord_webhook_url', None),
        }
        self._discord_max_words = self._normalize_max_words(
            getattr(config, 'discord_max_words', DISCORD_MAX_CONTENT_LENGTH)
        )
        self._webhook_verify_ssl = getattr(config, 'webhook_verify_ssl', True)

    @staticmethod
    def _compact_content(content: str) -> str:
        """移除多餘空行與行尾空白；程式碼區塊內的內容原樣保留。"""
        out = []
        in_fence = False
        for line in content.splitlines():
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                out.append(line)
            elif in_fence:
                out.append(line)
            elif line.strip():
                out.append(line.rstrip())
        return "\n".join(out).strip()

    @staticmethod
    def _normalize_max_words(value) -> int:
        try:
            configured = int(value)
        except (TypeError, ValueError):
            configured = DISCORD_MAX_CONTENT_LENGTH
        return max(MIN_MAX_WORDS, min(configured, DISCORD_MAX_CONTENT_LENGTH))

    def _is_discord_configured(self) -> bool:
        """检查 Discord 設定是否完整（支持 Bot 或 Webhook）"""
        # 只要設定了 Webhook 或完整的 Bot Token+Channel，即視為可用
        bot_ok = bool(self._discord_config['bot_token'] and self._discord_config['channel_id'])
        webhook_ok = bool(self._discord_config['webhook_url'])
        return bot_ok or webhook_ok

    def send_to_discord(self, content: str, *, timeout_seconds: Optional[float] = None) -> bool:
        """
        推送訊息到 Discord（支持 Webhook 和 Bot API）

        Args:
            content: Markdown 格式的訊息內容

        Returns:
            是否發送成功
        """
        sanitized_content = strip_hidden_markdown_metadata(content).strip()
        if DISCORD_COMPACT_LAYOUT:
            sanitized_content = self._compact_content(sanitized_content)
        if DISCORD_FORCE_TRADITIONAL:
            sanitized_content = to_traditional(sanitized_content)
        if not sanitized_content:
            logger.warning("Discord 訊息內容為空，跳過推送")
            return False

        # 分割內容，避免單條訊息超過 Discord 限制
        chunks = self._split_discord_content(sanitized_content)

        # 優先使用 Webhook（設定簡單，權限低）
        if self._discord_config['webhook_url']:
            return self._send_discord_chunks(
                chunks,
                self._send_discord_webhook,
                "Webhook",
                timeout_seconds=timeout_seconds,
            )

        # 其次使用 Bot API（權限高，需要 channel_id）
        if self._discord_config['bot_token'] and self._discord_config['channel_id']:
            return self._send_discord_chunks(
                chunks,
                self._send_discord_bot,
                "Bot",
                timeout_seconds=timeout_seconds,
            )

        logger.warning("Discord 設定不完整，跳過推送")
        return False

    def _split_discord_content(self, content: str) -> list[str]:
        """按 Discord content 上限拆分訊息。"""
        try:
            chunks = chunk_content_by_max_words(content, self._discord_max_words)
            if len(chunks) > 1:
                chunks = chunk_content_by_max_words(
                    content,
                    self._discord_max_words,
                    add_page_marker=True,
                )
            return chunks
        except ValueError as e:
            logger.error("分割 Discord 訊息失敗: %s", e)
            return chunk_content_by_max_words(
                content,
                DISCORD_MAX_CONTENT_LENGTH,
                add_page_marker=True,
            )

    def _send_discord_chunks(
        self,
        chunks: list[str],
        send_once,
        channel_name: str,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """逐片發送 Discord 訊息；失敗片不應阻斷後續片嘗試。"""
        total_chunks = len(chunks)
        success_count = 0

        if total_chunks > 1:
            logger.info("Discord %s 分批發送：共 %d 批", channel_name, total_chunks)

        for i, chunk in enumerate(chunks):
            if send_once(chunk, timeout_seconds=timeout_seconds):
                success_count += 1
                if total_chunks > 1:
                    logger.info("Discord %s 第 %d/%d 批發送成功", channel_name, i + 1, total_chunks)
            else:
                logger.error("Discord %s 第 %d/%d 批發送失敗", channel_name, i + 1, total_chunks)

            if i < total_chunks - 1:
                time.sleep(DISCORD_CHUNK_SLEEP_SECONDS)

        return success_count == total_chunks


    def _send_discord_webhook(self, content: str, *, timeout_seconds: Optional[float] = None) -> bool:
        """
        使用 Webhook 發送訊息到 Discord

        Discord Webhook 支持 Markdown 格式

        Args:
            content: Markdown 格式的訊息內容

        Returns:
            是否發送成功
        """
        payload = {
            'content': content,
            'username': '仿生助手前瞻',
            'avatar_url': 'https://ibb.co/s9vJNKGb'
        }

        return self._post_discord_message(
            self._discord_config['webhook_url'],
            payload,
            success_statuses=(200, 204),
            verify=self._webhook_verify_ssl,
            timeout_seconds=timeout_seconds,
            channel_name="Webhook",
        )

    def _send_discord_bot(self, content: str, *, timeout_seconds: Optional[float] = None) -> bool:
        """
        使用 Bot API 發送訊息到 Discord

        Args:
            content: Markdown 格式的訊息內容

        Returns:
            是否發送成功
        """
        headers = {
            'Authorization': f'Bot {self._discord_config["bot_token"]}',
            'Content-Type': 'application/json'
        }
        payload = {'content': content}
        url = f'https://discord.com/api/v10/channels/{self._discord_config["channel_id"]}/messages'

        return self._post_discord_message(
            url,
            payload,
            headers=headers,
            success_statuses=(200,),
            timeout_seconds=timeout_seconds,
            channel_name="Bot",
        )

    def _post_discord_message(
        self,
        url: str,
        payload: dict,
        *,
        success_statuses: tuple[int, ...],
        headers: Optional[dict] = None,
        verify: Optional[bool] = None,
        timeout_seconds: Optional[float] = None,
        channel_name: str,
    ) -> bool:
        """發送單條 Discord 訊息，並重用 Telegram 的有限重試思路處理 429/5xx。"""
        request_kwargs = {
            'json': payload,
            'timeout': timeout_seconds or 10,
        }
        if headers:
            request_kwargs['headers'] = headers
        if verify is not None:
            request_kwargs['verify'] = verify

        for attempt in range(1, DISCORD_MAX_RETRIES + 1):
            try:
                response = requests.post(url, **request_kwargs)
            except requests.exceptions.RequestException as e:
                if attempt < DISCORD_MAX_RETRIES:
                    delay = 2 ** attempt
                    logger.warning(
                        "Discord %s 請求異常（%d/%d）：%s，%s 秒後重試",
                        channel_name,
                        attempt,
                        DISCORD_MAX_RETRIES,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                logger.error("Discord %s 請求重試後仍失敗: %s", channel_name, e)
                return False

            if response.status_code in success_statuses:
                logger.info("Discord %s 訊息發送成功", channel_name)
                return True

            if response.status_code == 429 and attempt < DISCORD_MAX_RETRIES:
                retry_after = self._get_retry_after_seconds(response, attempt)
                logger.warning(
                    "Discord %s 觸發限流，%s 秒後重試（%d/%d）",
                    channel_name,
                    retry_after,
                    attempt,
                    DISCORD_MAX_RETRIES,
                )
                time.sleep(retry_after)
                continue

            if response.status_code >= 500 and attempt < DISCORD_MAX_RETRIES:
                delay = 2 ** attempt
                logger.warning(
                    "Discord %s 伺服器端錯誤 HTTP %s（%d/%d），%s 秒後重試",
                    channel_name,
                    response.status_code,
                    attempt,
                    DISCORD_MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
                continue

            logger.error(
                "Discord %s 發送失敗: %s %s",
                channel_name,
                response.status_code,
                response.text,
            )
            return False

        return False

    @staticmethod
    def _get_retry_after_seconds(response, attempt: int) -> float:
        try:
            retry_after = response.json().get('retry_after')
            if retry_after is not None:
                return max(0.0, float(retry_after))
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            retry_after = response.headers.get('Retry-After')
            if retry_after is not None:
                return max(0.0, float(retry_after))
        except AttributeError:
            pass

        return float(2 ** attempt)
