"""NEO's app-agnostic decision layer: which installed app can do a task on THIS Mac, judged from
what apps declare about themselves, how this user actually uses them and what worked before.

* ``inventory`` - the installed apps, their declared abilities and this user's use of them.
* ``sdef``      - an app's AppleScript dictionary read from its bundle, and generic capability tags.
* ``engine``    - Laya picks the task kind and orders the candidate apps; outcomes are learned.
* ``media``     - the driver for the generic "play something" task.

No app names, bundle ids or websites are hardcoded in any decision logic.
"""
