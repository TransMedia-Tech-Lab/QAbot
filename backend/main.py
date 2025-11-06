from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from qabot.bot import LabSlackBot
from qabot.config import ConfigError, Settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    load_dotenv()

    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        logging.error(str(exc))
        sys.exit(1)

    bot = LabSlackBot(settings)

    try:
        logging.info("Starting Slack bot in Socket Mode...")
        bot.start()
    except KeyboardInterrupt:
        logging.info("Shutting down gracefully")


if __name__ == "__main__":
    main()
