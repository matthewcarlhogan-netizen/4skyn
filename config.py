"""
OpenClaw V7.5 — Config loader

Loads config.yaml from the project root into a plain dict.
All core modules receive this dict directly; no pydantic, no global state.
"""
from __future__ import annotations
from pathlib import Path
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load and return the config.yaml as a plain dict."""
    p = Path(path)
    if not p.is_absolute():
        # Resolve relative to project root (same dir as this file)
        p = Path(__file__).parent / p
    with open(p, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Backwards-compat shim — if anything does `from config import settings`
# it gets the raw dict rather than an ImportError.
# ---------------------------------------------------------------------------
try:
    settings = load_config()
except FileNotFoundError:
    settings = {}


# Intentionally not a class — this is dead code kept to avoid
    birdeye_api_key: str = Field(default="", description="Birdeye API key")
    helius_api_key: str = Field(default="", description="Helius RPC API key")
    quicknode_rpc_url: str = Field(
        default="https://api.mainnet-beta.solana.com",
        description="QuickNode RPC endpoint"
    )
    jito_rpc_url: str = Field(
        default="https://mainnet.block-engine.jito.wtf/api/v1/transactions",
        description="Jito RPC endpoint for MEV-aware transactions"
    )
    tiktok_api_key: str = Field(default="", description="TikTok Data API key")

    # Solana Wallet Configuration
    wallet_path: str = Field(
        default="~/.config/solana/id.json",
        description="Path to Solana keypair JSON file"
    )
    rpc_endpoints: list = Field(
        default=[
            "https://api.mainnet-beta.solana.com",
            "https://solana-mainnet.g.alchemy.com/v2/",
        ],
        description="List of RPC endpoints for failover"
    )

    # Trading Mode and Risk Parameters
    trading_mode: str = Field(
        default="PAPER",
        description="Trading mode: PAPER, BENCHMARK, or LIVE"
    )
    max_trade_sol: float = Field(
        default=1.0,
        description="Maximum SOL per trade"
    )
    max_meme_exposure_pct: float = Field(
        default=20.0,
        description="Maximum % of portfolio in meme coins"
    )
    kelly_fraction: float = Field(
        default=0.25,
        description="Kelly criterion fraction (0.0-1.0)"
    )
    max_slippage_pct: float = Field(
        default=3.0,
        description="Maximum acceptable slippage %"
    )
    position_timeout_minutes: int = Field(
        default=120,
        description="Close position if no profit after N minutes"
    )

    # Scoring and Signal Thresholds
    meme_score_threshold: float = Field(
        default=65.0,
        description="Minimum MemeScore to trade (0-100)"
    )
    minimum_liquidity_usd: float = Field(
        default=100_000,
        description="Minimum liquidity required in USD"
    )
    score_weights: dict = Field(
        default={
            "social_signal": 0.25,
            "volume_spike": 0.20,
            "holder_distribution": 0.15,
            "macro_regime": 0.15,
            "technical": 0.10,
            "onchain_activity": 0.10,
            "news_sentiment": 0.05,
        },
        description="Signal component weights for MemeScore"
    )

    # Database Configuration
    database_url: str = Field(
        default="postgresql://openclaw:openclaw@localhost/openclaw_db",
        description="PostgreSQL connection string"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )

    # Logging and Monitoring
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    log_file: str = Field(
        default="/var/log/openclaw/openclaw.log",
        description="Log file path"
    )

    # Telegram Alerts
    telegram_bot_token: str = Field(
        default="",
        description="Telegram bot token for alerts"
    )
    telegram_chat_id: str = Field(
        default="",
        description="Telegram chat ID for alerts"
    )
    alert_on_trade: bool = Field(
        default=True,
        description="Send Telegram alert on trade execution"
    )
    alert_on_exit: bool = Field(
        default=True,
        description="Send Telegram alert on position exit"
    )

    # Monitor Check Intervals (seconds)
    pumpfun_check_interval: int = Field(
        default=5,
        description="PumpFun monitor poll interval"
    )
    social_check_interval: int = Field(
        default=30,
        description="Social media monitor poll interval"
    )
    macro_check_interval: int = Field(
        default=300,
        description="Macro regime monitor poll interval"
    )
    exit_check_interval: int = Field(
        default=60,
        description="Exit monitor poll interval"
    )
    dashboard_refresh_interval: int = Field(
        default=30,
        description="Dashboard auto-refresh interval"
    )

    # Dashboard Configuration
    dashboard_port: int = Field(
        default=8080,
        description="Dashboard server port"
    )
    dashboard_host: str = Field(
        default="0.0.0.0",
        description="Dashboard bind address"
    )
    dashboard_username: str = Field(
        default="admin",
        description="Dashboard basic auth username"
    )
    dashboard_password: str = Field(
        default="changeme",
        description="Dashboard basic auth password"
    )

    # System Configuration
    startup_timeout_seconds: int = Field(
        default=30,
        description="Timeout for component startup health checks"
    )
    error_restart_delay: int = Field(
        default=10,
        description="Seconds to wait before restarting failed loop"
    )
    max_concurrent_trades: int = Field(
        default=5,
        description="Maximum open positions simultaneously"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("trading_mode")
    def validate_trading_mode(cls, v):
        allowed = {"PAPER", "BENCHMARK", "LIVE"}
        if v not in allowed:
            raise ValueError(f"trading_mode must be one of {allowed}")
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v

    @validator("kelly_fraction")
    def validate_kelly_fraction(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("kelly_fraction must be between 0.0 and 1.0")
        return v

    @validator("meme_score_threshold")
    def validate_meme_score_threshold(cls, v):
        if not 0.0 <= v <= 100.0:
            raise ValueError("meme_score_threshold must be between 0 and 100")
        return v

    @validator("score_weights")
    def validate_score_weights(cls, v):
        if not v:
            raise ValueError("score_weights cannot be empty")
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"score_weights must sum to 1.0, got {total}")
        return v

    def get_rpc_endpoint(self) -> str:
        """Get primary RPC endpoint."""
        return self.quicknode_rpc_url or self.rpc_endpoints[0]

    def is_live_mode(self) -> bool:
        """Check if in live trading mode."""
        return self.trading_mode == "LIVE"

    def is_paper_mode(self) -> bool:
        """Check if in paper trading mode."""
        return self.trading_mode == "PAPER"


# Global settings instance
settings = Settings()
