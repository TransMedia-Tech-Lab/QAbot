"""Entry point for the ResearchLabBot."""

from __future__ import annotations

from ragbot.bot import ResearchLabBot


def main() -> None:
    bot = ResearchLabBot()
    bot.run()


if __name__ == "__main__":
    main()
