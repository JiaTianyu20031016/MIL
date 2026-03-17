#!/usr/bin/env python3
"""Schedule a command to run at the next occurrence of HH:MM."""

import os
import sys
import time
from datetime import datetime, timedelta


def _compute_delay(time_spec: str) -> tuple[float, datetime]:
	try:
		hour_str, minute_str = time_spec.split(":", 1)
		hour = int(hour_str)
		minute = int(minute_str)
	except (ValueError, AttributeError):
		raise ValueError("Time must be in HH:MM format") from None

	if not (0 <= hour < 24 and 0 <= minute < 60):
		raise ValueError("Hour must be [0,23] and minute [0,59]")

	now = datetime.now()
	target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
	if target <= now:
		target += timedelta(days=1)

	delay = (target - now).total_seconds()
	return delay, target


def main() -> None:
	if len(sys.argv) < 3:
		print("Usage: alarm HH:MM command [args...]", file=sys.stderr)
		sys.exit(1)

	time_spec = sys.argv[1]
	command = sys.argv[2:]

	try:
		delay, target_time = _compute_delay(time_spec)
	except ValueError as exc:
		print(f"Invalid time specification '{time_spec}': {exc}", file=sys.stderr)
		sys.exit(1)

	pid = os.fork()
	if pid == 0:
		try:
			if delay > 0:
				time.sleep(delay)
			os.execvp(command[0], command)
		except FileNotFoundError:
			print(f"alarm: command not found: {command[0]}", file=sys.stderr)
		except Exception as exc:  # pragma: no cover - best effort logging
			print(f"alarm: failed to run command: {exc}", file=sys.stderr)
		finally:
			os._exit(1)

	command_str = " ".join(command)
	readable_time = target_time.strftime("%Y-%m-%d %H:%M")
	print(f"Scheduled '{command_str}' to run at {readable_time} (PID {pid})")


if __name__ == "__main__":
	main()