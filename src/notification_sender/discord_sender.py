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
