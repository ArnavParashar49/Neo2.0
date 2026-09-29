// NEO's song card: the wire messages (the core's song recogniser). NEO listens only when the user
// asks ("what song is this?"), for a few seconds, and the card is the visible sign that it is.
// Audio never reaches the UI: only how loud it is (`level`) and, at the end, what it was.

export type SongStatus = "listening" | "matching" | "found" | "not_found" | "error";
/** What NEO listens to: the microphone (a song in the room), the Mac's own output, or both at once. */
export type SongSource = "mic" | "mac" | "both";
export type SongCloseReason = "done" | "stopped" | "error";
export type SongLinkKind = "apple_music" | "shazam";

/** A recognised song. `cover` is always a data URL the core fetched: the UI never loads a remote one. */
export interface SongResult {
  title: string;
  artist: string;
  album: string | null;
  year: string | null;
  genre: string | null;
  cover: string | null;
  links: Record<SongLinkKind, string | null>;
}

/** `song_open`: NEO starts listening for this request. */
export interface SongOpen { sid: string; source: SongSource; listen_s: number }

/** `song_state`: where it is (level at most 10 times a second while listening). */
export interface SongStateMsg {
  sid: string;
  status: SongStatus;
  level: number;
  elapsed_s: number;
  message: string | null;
  result: SongResult | null;
}

/** `song_close`: listening is over (the card stays in the transcript). */
export interface SongClose { sid: string; reason: SongCloseReason }
