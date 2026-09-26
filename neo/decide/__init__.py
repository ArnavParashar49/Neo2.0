"""NEO's app-agnostic decision layer: which installed app can do a task on THIS Mac, judged from
what apps declare about themselves, how this user actually uses them and what worked before.

* ``inventory`` - the installed apps, their declared abilities and this user's use of them.
* ``evidence``  - what a bundle declares beyond its dictionary (intents, App Intents, extensions).
* ``sdef``      - an app's AppleScript dictionary read from its bundle, and generic capability tags.
* ``engine``    - Laya picks the task kind and orders the candidate apps; outcomes are learned.
* ``run``       - the loop every kind's driver shares: budgets, commit points, what is learned.
* ``media``     - the driver for the generic "play something" task.
* ``aescript`` / ``axkit`` - scripting through an app's own dictionary / its UI (Accessibility).
* ``eventkit`` / ``routes`` / ``people`` / ``workspace`` - the OS's calendar and reminder stores,
  Contacts and default handlers.
* ``when`` / ``sites`` / ``learned`` - time phrases, websites by evidence, small learned stores.

No app names, bundle ids or websites are hardcoded in any decision logic.
"""
