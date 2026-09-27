"""Moved to neo.security.policy (pure stdlib: the web tools use it without loading macOS
frameworks). This name is the same module object, private names and monkeypatches included."""

import sys

from neo.security import policy as _p

sys.modules[__name__] = _p
