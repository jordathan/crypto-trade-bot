# Telegram Bot Setup Guide

## Step 1: Create Your Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` command
3. Choose a display name (e.g., "Ralph Trading Manager")
4. Choose a username (must end in "bot", e.g., "jordos_ralph_bot")
5. Copy the **bot token** you receive (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

## Step 2: Get Your User ID

**Method 1: Using @userinfobot**
1. Search for **@userinfobot** in Telegram
2. Send `/start` command
3. It will reply with your user ID (a number like `123456789`)

**Method 2: Using @RawDataBot**
1. Search for **@RawDataBot** in Telegram
2. Send any message
3. Look for `"id": 123456789` in the JSON response

## Step 3: Configure Environment Variables

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set these values:
   ```env
   TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   RALPH_SHARED_SECRET=MySecretPassword123
   TELEGRAM_ALLOWED_USERS=123456789
   ```

   - **TELEGRAM_BOT_TOKEN**: The token from BotFather
   - **RALPH_SHARED_SECRET**: A password you choose (used with every command)
   - **TELEGRAM_ALLOWED_USERS**: Your user ID from step 2 (can be comma-separated list: `123,456,789`)

## Step 4: Start the Bot

```bash
python main.py --mode ralph-telegram
```

## Step 5: Test It

1. Find your bot on Telegram (search for the username you created)
2. Send `/help` command
3. Try a command with your secret:
   ```
   /status MySecretPassword123
   ```

## Security Features

✅ **User ID Allowlist**: Only users in `TELEGRAM_ALLOWED_USERS` can use the bot  
✅ **Shared Secret**: All commands require your secret password  
✅ **Two-Factor Protection**: Both user ID AND secret must be correct

## Available Commands

- `/help` - Show command list
- `/status <secret>` - Show Ralph status and current parameters
- `/summary <secret>` - Show latest backtest summary
- `/cycle <secret> [days] [trials]` - Run one optimization cycle
- `/start <secret> [interval] [days] [trials]` - Start continuous auto mode
- `/stop <secret>` - Stop continuous auto mode
- `/watch <secret> [seconds]` - Auto-push new summaries every N seconds
- `/unwatch <secret>` - Stop auto-push
- `/set <secret> key=value` - Update config (e.g., `strategy.min_confidence=0.6`)
- `/get <secret> key` - Get config value

## Example Usage

```
# Get current status
/status MySecretPassword123

# Start auto optimization every 60 minutes
/start MySecretPassword123 60 90 50

# Enable auto-push of summaries every 30 seconds
/watch MySecretPassword123 30

# Update minimum confidence threshold
/set MySecretPassword123 strategy.min_confidence=0.65

# Stop auto mode
/stop MySecretPassword123
```

## Troubleshooting

**"Unauthorized user" message**
- Check that `TELEGRAM_ALLOWED_USERS` contains your correct user ID
- Restart the bot after changing `.env`

**"Invalid shared secret" message**
- Check that you're using the exact secret from `.env`
- Secrets are case-sensitive

**Bot doesn't respond**
- Check that `python main.py --mode ralph-telegram` is running
- Check the console for errors
- Verify `TELEGRAM_BOT_TOKEN` is correct

**How to allow multiple users**
- Add multiple IDs separated by commas:
  ```env
  TELEGRAM_ALLOWED_USERS=123456789,987654321,555111222
  ```
