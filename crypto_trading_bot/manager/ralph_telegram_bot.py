"""
Telegram bot to control Ralph Manager.
Commands require a shared secret.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from manager.ralph_manager import RalphManager


ROOT_DIR = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT_DIR / "logs" / "ralph_bot_state.json"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            output = subprocess.check_output(["tasklist", "/FI", f"PID eq {pid}"]) 
            return str(pid) in output.decode("utf-8", errors="ignore")
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _start_ralph_auto(interval: int, days: int, trials: int) -> int:
    cmd = [
        sys.executable,
        "main.py",
        "--mode",
        "ralph-auto",
        "--interval",
        str(interval),
        "--ralph-days",
        str(days),
        "--ralph-max-trials",
        str(trials)
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR))
    return proc.pid


def _stop_pid(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            subprocess.check_call(["taskkill", "/PID", str(pid), "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, 15)
        return True
    except Exception:
        return False


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass

    if raw.startswith("{") or raw.startswith("["):
        try:
            return json.loads(raw)
        except Exception:
            return raw

    return raw


def _set_config_value(config: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = config
    for key in parts[:-1]:
        node = node.setdefault(key, {})
    node[parts[-1]] = value


def _get_config_value(config: Dict[str, Any], path: str) -> Any:
    parts = path.split(".")
    node = config
    for key in parts:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _latest_summary() -> Optional[Dict[str, Any]]:
    summaries = sorted((ROOT_DIR / "logs").glob("daily_summary_*.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        return None
    try:
        with open(summaries[-1], "r") as f:
            return json.load(f)
    except Exception:
        return None


def _latest_summary_path() -> Optional[Path]:
    summaries = sorted((ROOT_DIR / "logs").glob("daily_summary_*.json"), key=lambda p: p.stat().st_mtime)
    return summaries[-1] if summaries else None


def _check_secret(args: list, expected: str) -> Optional[str]:
    if not args:
        return "Missing shared secret."
    if args[0] != expected:
        return "Invalid shared secret."
    return None


def _check_user_allowed(user_id: int) -> bool:
    """Check if user ID is in the allowlist. If env var not set, allow all."""
    allowed_users = os.getenv("TELEGRAM_ALLOWED_USERS", "")
    if not allowed_users:
        return True  # No restriction if not configured
    
    allowed_ids = [int(uid.strip()) for uid in allowed_users.split(",") if uid.strip().isdigit()]
    return user_id in allowed_ids


def _format_summary(summary: Dict[str, Any]) -> str:
    return (
        f"Date: {summary.get('date', '')}\n"
        f"Status: {summary.get('status', '')}\n"
        f"Trades: {summary.get('trades_count', 0)}\n"
        f"Skipped: {summary.get('skipped_count', 0)}\n"
        f"Errors: {summary.get('errors_count', 0)}\n"
        f"Return: {summary.get('return_pct', 0):+.2f}%\n"
        f"Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%\n"
        f"Win Rate: {summary.get('win_rate', 0):.1%}\n"
        f"Improvement: {summary.get('improvement_score', 0):+.2f}\n"
        f"Plan: {summary.get('plan', '')}"
    )


async def _watch_loop(app, chat_id: int, interval_seconds: int) -> None:
    bot_data = app.bot_data
    last_sent = bot_data.get("last_summary_mtime", 0)
    while bot_data.get("watch_running", False):
        latest_path = _latest_summary_path()
        if latest_path:
            mtime = latest_path.stat().st_mtime
            if mtime > last_sent:
                try:
                    with open(latest_path, "r") as f:
                        summary = json.load(f)
                    await app.bot.send_message(chat_id=chat_id, text=_format_summary(summary))
                    last_sent = mtime
                    bot_data["last_summary_mtime"] = last_sent
                except Exception as e:
                    logger.warning(f"Watch loop error: {e}")
        await asyncio.sleep(interval_seconds)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Unauthorized user.")
        return
    
    await update.message.reply_text(
        "Commands:\n"
        "/status <secret>\n"
        "/summary <secret>\n"
        "/cycle <secret> [days] [trials]\n"
        "/multitest <secret> [days] - Run multi-crypto backtest\n"
        "/start <secret> [interval] [days] [trials]\n"
        "/stop <secret>\n"
        "/watch <secret> [seconds]\n"
        "/unwatch <secret>\n"
        "/set <secret> key=value\n"
        "/get <secret> key\n"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Unauthorized user.")
        return
    
    secret = os.getenv("RALPH_SHARED_SECRET", "")
    err = _check_secret(context.args, secret)
    if err:
        await update.message.reply_text(err)
        return

    manager = RalphManager()
    strategy = manager.config.get("strategy", {})
    data_cfg = manager.config.get("data", {})

    state = _load_state()
    running = _is_process_running(state.get("pid", 0))

    msg = (
        f"Ralph running: {running}\n"
        f"min_confidence: {strategy.get('min_confidence', 0.55)}\n"
        f"rsi_low/high: {strategy.get('rsi_low', 30)}/{strategy.get('rsi_high', 70)}\n"
        f"momentum_window: {strategy.get('momentum_window', 20)}\n"
        f"buy_threshold: {strategy.get('signal_buy_threshold', 0.3)}\n"
        f"strong_threshold: {strategy.get('signal_strong_threshold', 0.5)}\n"
        f"lookback_days: {data_cfg.get('lookback_days', 90)}\n"
    )
    await update.message.reply_text(msg)


async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Unauthorized user.")
        return
    
    secret = os.getenv("RALPH_SHARED_SECRET", "")
    err = _check_secret(context.args, secret)
    if err:
        await update.message.reply_text(err)
        return

    summary = _latest_summary()
    if not summary:
        await update.message.reply_text("No summary found yet.")
        return

    await update.message.reply_text(_format_summary(summary))


async def cmd_cycle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Unauthorized user.")
        return
    
    secret = os.getenv("RALPH_SHARED_SECRET", "")
    err = _check_secret(context.args, secret)
    if err:
        await update.message.reply_text(err)
        return

    days = int(context.args[1]) if len(context.args) > 1 else None
    trials = int(context.args[2]) if len(context.args) > 2 else None

    manager = RalphManager()

    async def run_cycle():
        manager.run_optimization_cycle(days=days, max_trials=trials)

    await update.message.reply_text("Starting optimization cycle...")
    await asyncio.get_event_loop().run_in_executor(None, run_cycle)

    summary = _latest_summary()
    if summary:
        await update.message.reply_text(_format_summary(summary))
    else:
        await update.message.reply_text("Optimization cycle completed, but no summary found.")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Unauthorized user.")
        return
    
    secret = os.getenv("RALPH_SHARED_SECRET", "")
    err = _check_secret(context.args, secret)
    if err:
        await update.message.reply_text(err)
        return

    interval = int(context.args[1]) if len(context.args) > 1 else 60
    days = int(context.args[2]) if len(context.args) > 2 else 90
    trials = int(context.args[3]) if len(context.args) > 3 else 50

    state = _load_state()
    pid = state.get("pid", 0)
    if _is_process_running(pid):
        await update.message.reply_text(f"Ralph already running (PID {pid}).")
        return

    pid = _start_ralph_auto(interval, days, trials)
    _save_state({
        "pid": pid,
        "started_at": datetime.now().isoformat(),
        "interval": interval,
        "days": days,
        "trials": trials
    })
    await update.message.reply_text(f"Started Ralph auto (PID {pid}).")

    # Auto-watch for new summaries
    app = context.application
    if not app.bot_data.get("watch_running", False):
        app.bot_data["watch_running"] = True
        app.bot_data["watch_chat_id"] = update.effective_chat.id
        interval_seconds = 30
        task = asyncio.create_task(_watch_loop(app, update.effective_chat.id, interval_seconds))
        app.bot_data["watch_task"] = task
        await update.message.reply_text("Auto-watch enabled (30s polling).")


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Unauthorized user.")
        return
    
    secret = os.getenv("RALPH_SHARED_SECRET", "")
    err = _check_secret(context.args, secret)
    if err:
        await update.message.reply_text(err)
        return

    state = _load_state()
    pid = state.get("pid", 0)
    if not _is_process_running(pid):
        await update.message.reply_text("Ralph is not running.")
        return

    stopped = _stop_pid(pid)
    if stopped:
        _save_state({})
        await update.message.reply_text("Stopped Ralph auto.")
    else:
        await update.message.reply_text("Could not stop Ralph auto.")


async def cmd_watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Unauthorized user.")
        return
    
    secret = os.getenv("RALPH_SHARED_SECRET", "")
    err = _check_secret(context.args, secret)
    if err:
        await update.message.reply_text(err)
        return

    interval_seconds = int(context.args[1]) if len(context.args) > 1 else 30

    app = context.application
    if app.bot_data.get("watch_running", False):
        await update.message.reply_text("Watch already running.")
        return

    app.bot_data["watch_running"] = True
    app.bot_data["watch_chat_id"] = update.effective_chat.id
    task = asyncio.create_task(_watch_loop(app, update.effective_chat.id, interval_seconds))
    app.bot_data["watch_task"] = task
    await update.message.reply_text(f"Watch started. Polling every {interval_seconds}s.")


async def cmd_unwatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Unauthorized user.")
        return
    
    secret = os.getenv("RALPH_SHARED_SECRET", "")
    err = _check_secret(context.args, secret)
    if err:
        await update.message.reply_text(err)
        return

    app = context.application
    app.bot_data["watch_running"] = False
    task = app.bot_data.get("watch_task")
    if task:
        task.cancel()
    await update.message.reply_text("Watch stopped.")


async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Unauthorized user.")
        return
    
    secret = os.getenv("RALPH_SHARED_SECRET", "")
    err = _check_secret(context.args, secret)
    if err:
        await update.message.reply_text(err)
        return

    if len(context.args) < 2 or "=" not in context.args[1]:
        await update.message.reply_text("Usage: /set <secret> key=value")
        return

    key, raw = context.args[1].split("=", 1)
    value = _parse_value(raw)

    manager = RalphManager()
    config = manager.config
    _set_config_value(config, key, value)
    manager._save_config(config)

    await update.message.reply_text(f"Updated {key} = {value}")


async def cmd_get(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Unauthorized user.")
        return
    
    secret = os.getenv("RALPH_SHARED_SECRET", "")
    err = _check_secret(context.args, secret)
    if err:
        await update.message.reply_text(err)
        return

    if len(context.args) < 2:
        await update.message.reply_text("Usage: /get <secret> key")
        return

    key = context.args[1]
    manager = RalphManager()
    value = _get_config_value(manager.config, key)
    await update.message.reply_text(f"{key} = {value}")


async def cmd_multitest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run multi-crypto backtest on top 10 cryptos."""
    if not _check_user_allowed(update.effective_user.id):
        await update.message.reply_text("Unauthorized user.")
        return
    
    secret = os.getenv("RALPH_SHARED_SECRET", "")
    err = _check_secret(context.args, secret)
    if err:
        await update.message.reply_text(err)
        return

    days = int(context.args[1]) if len(context.args) > 1 else 90
    
    await update.message.reply_text(f"Starting multi-crypto backtest ({days} days)...\nThis may take a few minutes.")
    
    manager = RalphManager()
    
    def run_multitest():
        return manager.run_multi_crypto_backtest(days=days)
    
    # Run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, run_multitest)
    
    if not results:
        await update.message.reply_text("Multi-crypto backtest completed but no results returned.")
        return
    
    # Format results summary
    msg_lines = ["🎯 Multi-Crypto Backtest Complete!\n"]
    msg_lines.append(f"{'Symbol':<10} {'Return':<10} {'WinRate':<10} {'Trades':<8}")
    msg_lines.append("-" * 40)
    
    for symbol, res in list(results.items())[:10]:  # First 10
        msg_lines.append(
            f"{symbol:<10} "
            f"{res['total_return_pct']:>+8.2f}%  "
            f"{res['win_rate']:>8.1%}  "
            f"{res['num_trades']:>6}"
        )
    
    # Calculate averages
    avg_return = sum(r['total_return_pct'] for r in results.values()) / len(results)
    avg_win_rate = sum(r['win_rate'] for r in results.values()) / len(results)
    total_trades = sum(r['num_trades'] for r in results.values())
    
    msg_lines.append("-" * 40)
    msg_lines.append(f"Avg Return: {avg_return:+.2f}%")
    msg_lines.append(f"Avg Win Rate: {avg_win_rate:.1%}")
    msg_lines.append(f"Total Trades: {total_trades}")
    msg_lines.append("\nView full results in Streamlit (Ralph Multi-Crypto tab)")
    
    await update.message.reply_text("\n".join(msg_lines))


def main() -> None:
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("Missing TELEGRAM_BOT_TOKEN in environment.")
        sys.exit(1)

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("cycle", cmd_cycle))
    app.add_handler(CommandHandler("multitest", cmd_multitest))
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("watch", cmd_watch))
    app.add_handler(CommandHandler("unwatch", cmd_unwatch))
    app.add_handler(CommandHandler("set", cmd_set))
    app.add_handler(CommandHandler("get", cmd_get))

    logger.info("Ralph Telegram bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
