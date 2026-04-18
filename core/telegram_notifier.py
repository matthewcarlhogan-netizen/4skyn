"""
OpenClaw V7.5 — Telegram Notifier

V7.5 changes over V7:
  - send_meme_alert(candidate, stable_inflow): formatted MemeCoinCandidate alert
  - Version header updated to V7.5
  - TYPE_CHECKING guard prevents circular import from meme_scanner

Non-blocking. Fails silent — never brings down the main loop.
"""
from __future__ import annotations
import threading
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from core.meme_scanner import MemeCoinCandidate


class TelegramNotifier:

    def __init__(self, bot_token: str, chat_id: str):
        self.url     = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.chat_id = chat_id

    def send(self, message: str) -> None:
        """Send a plain text message. Non-blocking, fails silent."""
        if not self.chat_id or not self.url.endswith("sendMessage"):
            return

        def _send():
            try:
                with httpx.Client(timeout=5.0) as c:
                    c.post(self.url, json={
                        "chat_id": self.chat_id,
                        "text": f"🦂 OpenClaw V7.5\n{message}",
                    })
            except Exception:
                pass

        threading.Thread(target=_send, daemon=True).start()

    def send_meme_alert(
        self,
        candidate: "MemeCoinCandidate",
        stable_inflow: bool = False,
    ) -> None:
        """
        Format and send a MemeCoinCandidate alert via Telegram.
        Non-blocking. Lazy import guard prevents circular import.
        """
        if candidate.contrarian_exit:
            signal_icon = "🚨"
            header = f"CONTRARIAN EXIT [{candidate.score}/8]"
        elif candidate.score >= 7:
            signal_icon = "🚀"
            header = f"MEME GEM [{candidate.score}/8]"
        else:
            signal_icon = "⚡"
            header = f"MEME WATCH [{candidate.score}/8]"

        stable_str = "▲ inflow active" if stable_inflow else "▼ inflow flat"
        macro_str  = f"macro={candidate.macro_composite:+.3f}" if candidate.macro_composite != 0.0 else ""

        criteria_lines = "\n".join(
            f"  {'✅' if v else '❌'} {k}"
            for k, v in candidate.criteria.items()
        )

        lines = [
            f"{signal_icon} {header}",
            f"{candidate.symbol} ({candidate.name}) | {candidate.chain.upper()}/{candidate.dex}",
            f"Price: ${candidate.price_usd:.8f}  MCap: ${candidate.market_cap_usd:,.0f}",
            f"Liq: ${candidate.liquidity_usd:,.0f}  Vol24h: ${candidate.volume_24h_usd:,.0f}",
            f"Age: {candidate.age_hours:.1f}h  1h: {candidate.price_change_1h_pct:+.1f}%",
            f"Cycle: {candidate.cycle_sentiment}  {macro_str}  {stable_str}",
            "",
            criteria_lines,
            "",
            candidate.dexscreener_url,
        ]

        if candidate.contrarian_exit:
            lines.append(f"\n⚠️ EXIT: {candidate.contrarian_reason}")
        else:
            lines.append("\n⚠️ GAMBLE ONLY — lottery budget only, 97% → zero")

        message = "\n".join(lines)

        def _send():
            try:
                with httpx.Client(timeout=5.0) as c:
                    c.post(self.url, json={
                        "chat_id": self.chat_id,
                        "text": message,
                    })
            except Exception:
                pass

        threading.Thread(target=_send, daemon=True).start()
