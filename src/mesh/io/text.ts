/**
 * Byte-level text scanning helpers shared by the text mesh parsers (ASCII STL, OBJ, ASCII PLY,
 * PTX).
 *
 * The parsers never decode the whole file into a JS string and never split it into lines:
 * they walk the raw bytes with a {@link ByteScanner} and parse numbers directly from bytes
 * (Clinger fast path: up to ~16 significant digits accumulated exactly as an integer, then a
 * single multiply/divide by an exact power of ten when |exp| <= 22). Only odd tokens such as
 * `nan`, `inf` or MSVC's `1.#INF` are decoded to a (tiny) string.
 *
 * Pure TypeScript: no DOM, no globals — runs in Node and Web Workers.
 */

/** Exact powers of ten representable as doubles (10^0 .. 10^22). */
const POW10 = new Float64Array([
  1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
  1e20, 1e21, 1e22,
]);

/**
 * Correctly rounded powers of ten 10^0 .. 10^308 (decimal literals are correctly rounded by
 * the JS engine; entries 0..22 are exact). Used beyond the exact range instead of Math.pow,
 * which may be off by an ulp.
 */
const POW10_ALL = (() => {
  const t = new Float64Array(309);
  for (let i = 0; i < 309; i++) t[i] = Number("1e" + i);
  return t;
})();

/** Mantissa accumulation limit: m * 10 + 9 stays below 2^53 (so m stays exact). */
const MANT_LIMIT = 9e14;

const LF = 10;

/** Space, tab, CR, vertical tab, form feed (everything "blank" except LF). */
export function isBlankByte(c: number): boolean {
  return c === 32 || c === 9 || c === 13 || c === 11 || c === 12;
}

/** Any ASCII whitespace including LF. */
export function isSpaceByte(c: number): boolean {
  return c === 32 || c === 10 || c === 9 || c === 13 || c === 11 || c === 12;
}

/**
 * Normalise supported inputs to a byte view without copying binary data.
 * Strings are UTF-8 encoded with TextEncoder. Views keep their byteOffset (they may point
 * into a larger, possibly shared, buffer).
 */
export function toBytes(data: string | ArrayBuffer | ArrayBufferView | SharedArrayBuffer): Uint8Array {
  if (typeof data === "string") return new TextEncoder().encode(data);
  if (data instanceof Uint8Array) return data;
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return new Uint8Array(data as ArrayBuffer);
}

/** DataView over exactly the bytes of `u8` (respects byteOffset; works for shared buffers). */
export function dataViewOf(u8: Uint8Array): DataView {
  return new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
}

/** Decode bytes [start, end) as Latin-1 (only for short tokens / header lines). */
export function latin1(buf: Uint8Array, start: number, end: number): string {
  let s = "";
  for (let i = start; i < end; i++) s += String.fromCharCode(buf[i]);
  return s;
}

/** Interpret an odd numeric token (nan, inf, infinity, 1.#INF, ...) that the fast path rejected. */
function parseOddNumber(token: string): number {
  const t = token.toLowerCase();
  let neg = false;
  let u = t;
  if (u.charCodeAt(0) === 45 /* - */) {
    neg = true;
    u = u.slice(1);
  } else if (u.charCodeAt(0) === 43 /* + */) {
    u = u.slice(1);
  }
  if (u.startsWith("nan")) return NaN;
  if (u.startsWith("inf")) return neg ? -Infinity : Infinity;
  if (u.includes("#inf")) return neg ? -Infinity : Infinity;
  if (u.includes("#")) return NaN; // 1.#IND, 1.#QNAN, 1.#SNAN
  return parseFloat(t);
}

/**
 * Forward-only cursor over a byte buffer with allocation-free number parsing.
 *
 * Conventions:
 *  - "blank" = whitespace other than LF. `readFloat`/`readInt` skip blanks but never cross
 *    a line feed, so line-oriented formats can detect the end of a record.
 *  - After a read, `ok` tells whether a token was present (false at end of line / data).
 */
export class ByteScanner {
  readonly buf: Uint8Array;
  pos: number;
  readonly end: number;
  /** Whether the last read found a token. */
  ok = true;

  constructor(buf: Uint8Array, start = 0, end = buf.length) {
    this.buf = buf;
    this.pos = start;
    this.end = end;
  }

  eof(): boolean {
    return this.pos >= this.end;
  }

  /** Current byte, or -1 at end of data. */
  peek(): number {
    return this.pos < this.end ? this.buf[this.pos] : -1;
  }

  /** Skip spaces/tabs/CR (not LF). */
  skipBlanks(): void {
    const b = this.buf, end = this.end;
    let p = this.pos;
    while (p < end) {
      const c = b[p];
      if (c === 32 || c === 9 || c === 13 || c === 11 || c === 12) p++;
      else break;
    }
    this.pos = p;
  }

  /** Skip all whitespace including line feeds. */
  skipWhitespace(): void {
    const b = this.buf, end = this.end;
    let p = this.pos;
    while (p < end) {
      const c = b[p];
      if (c === 32 || c === 10 || c === 9 || c === 13 || c === 11 || c === 12) p++;
      else break;
    }
    this.pos = p;
  }

  /** Skip blanks; true when the cursor is then at LF or end of data. */
  atLineEnd(): boolean {
    this.skipBlanks();
    return this.pos >= this.end || this.buf[this.pos] === LF;
  }

  /** Advance just past the next LF (or to the end of data). */
  nextLine(): void {
    if (this.pos >= this.end) return;
    const i = this.buf.indexOf(LF, this.pos);
    this.pos = i < 0 || i >= this.end ? this.end : i + 1;
  }

  /** Skip the current run of non-whitespace bytes. */
  skipToken(): void {
    const b = this.buf, end = this.end;
    let p = this.pos;
    while (p < end) {
      const c = b[p];
      if (c === 32 || c === 10 || c === 9 || c === 13 || c === 11 || c === 12) break;
      p++;
    }
    this.pos = p;
  }

  /** Read the next whitespace-delimited token on the current line as a string ("" if none). */
  readTokenString(): string {
    this.skipBlanks();
    const start = this.pos;
    this.skipToken();
    return latin1(this.buf, start, this.pos);
  }

  /** Rest of the current line as a string (without CR/LF); moves to the next line. */
  readLineString(): string {
    const start = this.pos;
    let i = this.buf.indexOf(LF, start);
    if (i < 0 || i > this.end) i = this.end;
    let e = i;
    while (e > start && this.buf[e - 1] === 13) e--;
    this.pos = i < this.end ? i + 1 : this.end;
    return latin1(this.buf, start, e);
  }

  /**
   * If the bytes at the cursor equal `word` (ASCII, compared case-insensitively; `word` must be
   * lower case) followed by whitespace or end of data, consume the word and return true.
   */
  matchWord(word: string): boolean {
    const b = this.buf, p = this.pos, n = word.length;
    if (p + n > this.end) return false;
    for (let i = 0; i < n; i++) {
      let c = b[p + i];
      if (c >= 65 && c <= 90) c += 32;
      if (c !== word.charCodeAt(i)) return false;
    }
    if (p + n < this.end && !isSpaceByte(b[p + n])) return false;
    this.pos = p + n;
    return true;
  }

  /**
   * Parse a signed decimal integer at the cursor WITHOUT skipping leading blanks. Stops at the
   * first non-digit (e.g. '/' in OBJ face tokens). Sets `ok` = false when there is no digit.
   */
  readIntRaw(): number {
    const b = this.buf, end = this.end;
    let p = this.pos;
    let neg = false;
    if (p < end) {
      const c = b[p];
      if (c === 45) {
        neg = true;
        p++;
      } else if (c === 43) p++;
    }
    const digitsStart = p;
    let v = 0;
    while (p < end) {
      const d = b[p] - 48;
      if (d < 0 || d > 9) break;
      v = v * 10 + d;
      p++;
    }
    if (p === digitsStart) {
      this.ok = false;
      return 0;
    }
    this.ok = true;
    this.pos = p;
    return neg ? -v : v;
  }

  /**
   * Parse the next number on the current line (skips blanks, never crosses LF). Returns NaN and
   * sets `ok` = false when the line has no more tokens. Trailing junk glued to a number
   * (e.g. "1.5f") is skipped up to the next whitespace.
   */
  readFloat(): number {
    const b = this.buf, end = this.end;
    let p = this.pos;
    let c = 0;
    while (p < end) {
      c = b[p];
      if (c === 32 || c === 9 || c === 13 || c === 11 || c === 12) p++;
      else break;
    }
    if (p >= end || c === LF) {
      this.pos = p;
      this.ok = false;
      return NaN;
    }
    this.ok = true;
    const start = p;
    let neg = false;
    if (c === 45) {
      neg = true;
      p++;
    } else if (c === 43) {
      p++;
    }
    let mant = 0;
    let exp = 0;
    let nDigits = 0;
    let dropped = -1; // first digit that did not fit in the exact mantissa
    // integer part
    while (p < end) {
      const d = b[p] - 48;
      if (d < 0 || d > 9) break;
      if (mant < MANT_LIMIT) mant = mant * 10 + d;
      else {
        exp++;
        if (dropped < 0) dropped = d;
      }
      nDigits++;
      p++;
    }
    // fraction
    if (p < end && b[p] === 46 /* . */) {
      p++;
      while (p < end) {
        const d = b[p] - 48;
        if (d < 0 || d > 9) break;
        if (mant < MANT_LIMIT) {
          mant = mant * 10 + d;
          exp--;
        } else if (dropped < 0) dropped = d;
        nDigits++;
        p++;
      }
    }
    if (dropped >= 5) mant += 1; // round the truncated mantissa
    if (nDigits === 0) {
      // nan / inf / garbage: decode the token
      let q = start;
      while (q < end && !isSpaceByte(b[q])) q++;
      this.pos = q;
      return parseOddNumber(latin1(b, start, q));
    }
    // exponent
    if (p < end && (b[p] | 0x20) === 101 /* e */) {
      let q = p + 1;
      let eneg = false;
      if (q < end && (b[q] === 45 || b[q] === 43)) {
        eneg = b[q] === 45;
        q++;
      }
      if (q < end && b[q] >= 48 && b[q] <= 57) {
        let ev = 0;
        while (q < end) {
          const d = b[q] - 48;
          if (d < 0 || d > 9) break;
          if (ev < 100000) ev = ev * 10 + d;
          q++;
        }
        exp += eneg ? -ev : ev;
        p = q;
      }
    }
    // glued junk (e.g. MSVC "1.#INF", "1.5f")
    if (p < end && !isSpaceByte(b[p])) {
      if (b[p] === 35 /* # */) {
        let q = p;
        while (q < end && !isSpaceByte(b[q])) q++;
        this.pos = q;
        return parseOddNumber(latin1(b, start, q));
      }
      while (p < end && !isSpaceByte(b[p])) p++;
    }
    this.pos = p;
    let v: number;
    if (exp === 0 || mant === 0) v = mant;
    else if (exp > 0) {
      // Clinger fast path when exact; otherwise one extra rounding from the power table
      if (exp <= 22) v = mant * POW10[exp];
      else v = exp <= 308 ? mant * POW10_ALL[exp] : mant * 1e308 * Math.pow(10, exp - 308);
    } else if (exp >= -22) v = mant / POW10[-exp];
    else if (exp >= -308) v = mant / POW10_ALL[-exp];
    else v = mant / 1e308 / POW10_ALL[Math.min(308, -exp - 308)];
    return neg ? -v : v;
  }

  /** `readFloat`, but returns `def` when the line has no more tokens. */
  readFloatOr(def: number): number {
    const v = this.readFloat();
    return this.ok ? v : def;
  }

  /** Parse the next number, skipping any whitespace including line feeds. */
  readFloatAny(): number {
    this.skipWhitespace();
    return this.readFloat();
  }
}

/** Parse one number from a string with the byte parser (mainly for tests). */
export function parseNumberString(s: string): number {
  const sc = new ByteScanner(new TextEncoder().encode(s));
  return sc.readFloat();
}

/** Throttled progress reporting helper (avoids per-element callbacks). */
export class ProgressTicker {
  private next: number;
  private readonly fn: ((fraction: number, message?: string) => void) | undefined;
  private readonly total: number;
  private readonly step: number;
  private readonly message: string;

  constructor(
    fn: ((fraction: number, message?: string) => void) | undefined,
    total: number,
    step: number,
    message: string,
  ) {
    this.fn = fn;
    this.total = total;
    this.step = step;
    this.message = message;
    this.next = step;
  }

  /** Report when `done` crossed the next threshold. */
  tick(done: number): void {
    if (done < this.next || !this.fn) return;
    this.next = done + this.step;
    const f = this.total > 0 ? Math.min(1, done / this.total) : 0;
    this.fn(f, this.message);
  }

  /** Unconditional report. */
  report(fraction: number, message?: string): void {
    this.fn?.(fraction, message ?? this.message);
  }
}
