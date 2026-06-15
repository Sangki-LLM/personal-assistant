import logging

import httpx

logger = logging.getLogger(__name__)


async def get_weather(location: str) -> str:
    """wttr.in API로 날씨 정보를 가져온다. API 키 불필요."""
    try:
        url = f"https://wttr.in/{location}?format=j1&lang=ko"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers={"User-Agent": "curl/7.68.0"})
            resp.raise_for_status()
            data = resp.json()

        current = data["current_condition"][0]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        desc = current["lang_ko"][0]["value"] if current.get("lang_ko") else current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind_kmph = current["windspeedKmph"]

        # 오늘 최고/최저 기온
        today = data["weather"][0]
        max_c = today["maxtempC"]
        min_c = today["mintempC"]

        # 강수 확률 (hourly 평균)
        hourly = today.get("hourly", [])
        rain_chances = [int(h.get("chanceofrain", 0)) for h in hourly]
        max_rain = max(rain_chances) if rain_chances else 0

        rain_warn = f" ☔ 강수 확률 최대 {max_rain}%" if max_rain >= 30 else ""

        return (
            f"📍 *{location} 현재 날씨*\n"
            f"• 날씨: {desc}\n"
            f"• 현재 기온: {temp_c}°C (체감 {feels_like}°C)\n"
            f"• 오늘 최고/최저: {max_c}°C / {min_c}°C\n"
            f"• 습도: {humidity}% | 바람: {wind_kmph}km/h"
            f"{rain_warn}"
        )
    except Exception as e:
        logger.warning("[weather] get_weather failed location=%s: %s", location, e)
        return f"{location} 날씨 정보를 가져오지 못했습니다."
