"""Naming the song that's playing: capture (the microphone and the Mac's own sound, in memory only),
recognize (Shazam's fingerprint of it, one request to Shazam), the chat's song card (live) and the
song_identify tool (neo2.core.tools.music.tool, imported by neo2.core.tools.load_all on macOS).

Importing this package registers no tool: the websocket server imports the card alone.
"""
