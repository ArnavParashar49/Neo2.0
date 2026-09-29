// NEO's camera card: the wire messages (neo2/core/tools/camera/live.py). The camera turns on only
// on the user's own request and off again after the answer; the card is its visible sign.

export type CamStatus = "starting" | "live" | "analysing" | "answered" | "off";

/** One detection, in 0..1 coordinates of the (unmirrored) camera frame. `id` stays the same while
 *  that thing is in view: the card keys its box by it, so a box moves with its thing. */
export interface CamBox { x: number; y: number; w: number; h: number; label: string; score: number; id?: number }

/** `camera_open`: the camera is starting for this request. */
export interface CamOpen { sid: string; reason?: string; job?: string }

/** `camera_state`: the latest status, labels and boxes (and the answer once there is one).
 *  Labels may come as strings, [label, conf] pairs or {label, score|conf} objects. */
export interface CamState {
  sid: string;
  status: CamStatus;
  labels?: unknown[];
  boxes?: unknown[];
  answer?: string | null;
}

/** `camera_close`: the camera is off. */
export interface CamClose { sid: string; reason?: string }

/** What the card shows for the live session. */
export interface CamLive {
  sid: string;
  reason: string;
  status: CamStatus;
  labels: string[];
  boxes: CamBox[];
  answer: string;
  startedAt: number; // Date.now() when it opened here
}

/** A session that ended: only what was said, never an image. */
export interface CamEnded { sid: string; answer: string; ms: number; reason: string }

/** `hello.data.camera` (when the core sends one): the live session, if any, for a (re)connecting client. */
export interface CamHelloSnapshot { live?: (CamState & { reason?: string }) | null }
