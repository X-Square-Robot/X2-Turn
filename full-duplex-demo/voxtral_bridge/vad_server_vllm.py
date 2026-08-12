#!/usr/bin/env python3
"""Deprecated compatibility entry point for the standalone turn bridge."""

import sys

from voxtral_realtime.cli import main


if __name__ == "__main__":
    main(["serve", *sys.argv[1:]])
