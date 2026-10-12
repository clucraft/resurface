/**
 * Run `fn` on the next animation frame, or after `timeoutMs` if frames are not being produced
 * (hidden window / background tab), whichever comes first.
 */
export function nextFrame(fn: () => void, timeoutMs = 50): void {
  let done = false;
  const run = () => {
    if (done) return;
    done = true;
    fn();
  };
  requestAnimationFrame(run);
  setTimeout(run, timeoutMs);
}
