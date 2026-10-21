// The selected area: clear inside, the rest dimmed by its shadow, a hairline, and four white corner
// brackets that snap into place when it is picked. Tab's candidate and smart pick's hover use the
// same box, lighter.

import { bracketPath } from "./Trail";
import type { Box } from "./geometry";

/** The parent keys it by the area's number, so a new area snaps in and an adjusted one doesn't. */
export function Selection({ box, candidate = false, label }: { box: Box; candidate?: boolean; label?: string }) {
  const pad = 2;
  const w = box.w + pad * 2, h = box.h + pad * 2;
  return (
    <div
      className={"lx-sel" + (candidate ? " candidate" : " enter")}
      style={{ left: box.x, top: box.y, width: box.w, height: box.h }}
      role="img"
      aria-label={label ?? `Selected area, ${Math.round(box.w)} by ${Math.round(box.h)} points`}
    >
      <svg className="lx-brackets" viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
        <path d={bracketPath({ x: pad, y: pad, w: box.w, h: box.h })} />
      </svg>
    </div>
  );
}

export function Hover({ box }: { box: Box }) {
  return <div className="lx-hover" style={{ left: box.x, top: box.y, width: box.w, height: box.h }} aria-hidden="true" />;
}
