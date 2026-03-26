#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 training entrypoint with fine-grained region labels:
L_FP, R_FP, L_F, R_F, C, L_T, R_T, P, O
"""

from __future__ import annotations

import sys

try:
    import train_soz_locator_with_brain_networks as _base
except ImportError:
    from . import train_soz_locator_with_brain_networks as _base


def main() -> int:
    argv = sys.argv[1:]
    if '--region-label-mode' not in argv:
        argv = ['--region-label-mode', 'fine_lateralized'] + argv
    sys.argv = [sys.argv[0]] + argv
    return _base.main()


if __name__ == '__main__':
    raise SystemExit(main())
