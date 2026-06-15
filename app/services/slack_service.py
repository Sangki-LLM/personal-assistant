import logging

from slack_sdk.web.async_client import AsyncWebClient

from app.core.config import settings

logger = logging.getLogger(__name__)


def _client() -> AsyncWebClient:
    return AsyncWebClient(token=settings.slack_bot_token)


async def send_message(channel_id: str, text: str, blocks: list | None = None) -> None:
    try:
        kwargs: dict = {"channel": channel_id, "text": text}
        if blocks:
            kwargs["blocks"] = blocks
        await _client().chat_postMessage(**kwargs)
        logger.info("[slack] sent to channel=%s", channel_id)
    except Exception as e:
        logger.warning("[slack] send_message failed: %s", e)


async def send_dm(user_id: str, text: str) -> None:
    """DM 채널을 열고 메시지를 전송한다."""
    try:
        client = _client()
        resp = await client.conversations_open(users=user_id)
        channel_id = resp["channel"]["id"]
        await client.chat_postMessage(channel=channel_id, text=text)
        logger.info("[slack] DM sent to user=%s", user_id)
    except Exception as e:
        logger.warning("[slack] send_dm failed: %s", e)
