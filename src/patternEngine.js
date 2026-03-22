import React, { useState, useEffect, useCallback, useRef, memo } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// PatternEngine — single PTN tab
// Pure pattern recognition on full 12k+ round dataset.
// No statistics, no ML, no probability theory.
// Analyses: clustering, trend, autocorrelation, momentum per target.
// Everything (locked windows + history) stored in DB.
// DB is single source of truth — survives refresh / multi-tab.
// ─────────────────────────────────────────────────────────────────────────────

const API_URL = process.env.REACT_APP_API_URL || '';
const apiFetch = (url) => fetch(url, { cache: 'no-store' }).then(r => r.json()).catch(() => ({ ok: false }));

export const TARGETS = [
  { label: '5x',    min: 5,    color: '#00e87a', emoji: '🪐', maxWidth: 3  },
  { label: '10x',   min: 10,   color: '#00c8f0', emoji: '💫', maxWidth: 5  },
  { label: '20x',   min: 20,   color: '#9060f0', emoji: '🌠', maxWidth: 7  },
  { label: '50x',   min: 50,   color: '#e040a0', emoji: '🌙', maxWidth: 12 },
  { label: '100x',  min: 100,  color: '#f0c040', emoji: '☀️',  maxWidth: 18 },
  { label: '250x',  min: 250,  color: '#f07030', emoji: '🔥', maxWidth: 25 },
  { label: '500x',  min: 500,  color: '#c060f0', emoji: '💎', maxWidth: 35 },
  { label: '1000x', min: 1000, color: '#f03050', emoji: '🚀', maxWidth: 50 },
];

const MIN_ROUNDS = 150;
const HIST_PAGE  = 10;
const SOURCE     = 'pattern';

const C = {
  bg: '#060810', card: '#0d1120', border: '#1e2d50', borderHi: '#2a4080',
  white: '#ffffff', text: '#dce8ff', label: '#8ab4e8', meta: '#5a82c8',
  green: '#00ff88', cyan: '#00d4ff', yellow: '#ffd84d',
  orange: '#ff8c42', red: '#ff4560',
};

// ─────────────────────────────────────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────────────────────────────────────
function injectStyles() {
  if (document.getElementById('ptn-css')) return;
  const el = document.createElement('style');
  el.id = 'ptn-css';
  el.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700;800&family=Inter:wght@400;500;600;700;800&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; }
    .ptn { font-family:'Inter',sans-serif; background:${C.bg}; color:${C.text}; font-size:13px; }
    .ptn-card         { background:${C.card}; border:1px solid ${C.border}; border-radius:10px; padding:11px 14px; }
    .ptn-card-active  { background:#001a0d; border:1.5px solid ${C.green}; border-radius:10px; padding:11px 14px; box-shadow:0 0 18px #00ff8818; }
    .ptn-card-waiting { background:${C.card}; border:1px solid ${C.borderHi}; border-radius:10px; padding:11px 14px; }
    .ptn-card-stale   { background:${C.card}; border:1px solid #ff8c4240; border-radius:10px; padding:11px 14px; opacity:.8; }
    .ptn-card-empty   { background:${C.card}; border:1px solid ${C.border}; border-radius:10px; padding:11px 14px; opacity:.4; }
    .ptn-badge { display:inline-flex; align-items:center; padding:2px 7px; border-radius:4px; font-size:10px; font-weight:800; letter-spacing:.06em; font-family:'JetBrains Mono',monospace; white-space:nowrap; }
    .ptn-hrow { display:grid; grid-template-columns:52px 68px 1fr 62px; gap:6px; align-items:center; padding:6px 10px; border-radius:5px; }
    .ptn-hrow:nth-child(odd)  { background:#0a1020; }
    .ptn-hrow:nth-child(even) { background:${C.card}; }
    .ptn-more { display:block; width:100%; margin-top:8px; padding:7px; background:none; border:1px solid ${C.border}; border-radius:6px; color:${C.label}; font-size:11px; font-weight:700; font-family:'JetBrains Mono',monospace; cursor:pointer; text-align:center; letter-spacing:.06em; }
    .ptn-more:hover { border-color:${C.cyan}; color:${C.cyan}; }
  `;
  document.head.appendChild(el);
}

// ─────────────────────────────────────────────────────────────────────────────
// ROUND SEARCH HELPERS
// Binary search on sorted rounds array — O(log n) instead of O(n) for 12k+ rounds.
// ─────────────────────────────────────────────────────────────────────────────

// Find first index where roundId >= targetId
function bisectLeft(rounds, targetId) {
  let lo = 0, hi = rounds.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (rounds[mid].roundId < targetId) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// Find first round in [fromId, toId] with multiplier >= minMult
function findHitInRange(rounds, fromId, toId, minMult) {
  const start = bisectLeft(rounds, fromId);
  for (let i = start; i < rounds.length; i++) {
    if (rounds[i].roundId > toId) break;
    if (rounds[i].multiplier >= minMult) return rounds[i];
  }
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// DB HELPERS
// Only ONE write operation from the frontend: save resolved outcomes.
// Locked windows are written ONLY by the server (patternEngine.js).
// Frontend reads locked windows, checks resolution, saves outcome.
// This mirrors exactly how AdvancedEngines works.
// ─────────────────────────────────────────────────────────────────────────────
async function dbSaveResolved(target, outcome, lo, hi, hitRound, generation) {
  try {
    await fetch(`${API_URL}/predictions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target, outcome, lo, hi,
        hitRound: hitRound ?? null,
        minMult: TARGETS.find(t => t.label === target)?.min ?? 0,
        generation: generation ?? 1,
        source: SOURCE,
      }),
    });
  } catch(e) { console.error('[PTN] save resolved:', e.message); }
}

// ─────────────────────────────────────────────────────────────────────────────
// usePTN HOOK
// Architecture (mirrors AdvancedEngines exactly):
//   1. Server (patternEngine.js) writes locked windows → locked-pattern table
//   2. Frontend polls /locked-pattern every 8s → displays windows
//   3. Frontend detects resolution (win/loss/early) → POSTs to /predictions
//   4. Frontend polls /predictions?source=pattern → displays history
// DB is single source of truth. No frontend writes to locked-pattern.
// ─────────────────────────────────────────────────────────────────────────────
function usePTN(rounds) {
  const [locked,  setLocked]  = useState({});
  const [history, setHistory] = useState([]);

  // FIX 1: lockedRef mirrors locked state — used inside async run() to avoid
  // stale closure. When locked state updates mid-async-execution, lockedRef
  // always holds the latest value, preventing stale window data.
  const lockedRef      = useRef({});
  const resolvedRef    = useRef(new Set());
  const prevIdRef      = useRef(0);
  // FIX 2: processingRef prevents concurrent run() calls — same pattern as
  // AdvancedEngines.jsx useEngine hook. Without this, if a round arrives while
  // run() is awaiting a DB write, a second run() starts on the same round.
  const processingRef  = useRef(false);
  const pendingRef     = useRef(null);

  // Pre-build a lookup map for maxWidth per target — avoids O(n) TARGETS.find()
  // on every history row filter
  const targetMaxWidth = useRef(
    Object.fromEntries(TARGETS.map(t => [t.label, t.maxWidth]))
  );

  // Poll locked windows from DB every 8s (server writes these)
  const loadLocked = useCallback(async () => {
    const res = await apiFetch(`${API_URL}/locked-pattern`);
    if (!res.ok || !res.preds) return;
    const map = {};
    for (const [label, p] of Object.entries(res.preds)) {
      if (!p) continue;
      const eta = p.eta || {};
      map[label] = {
        lo:            Number(p.lo),
        hi:            Number(p.hi),
        roundWhenMade: Number(p.roundWhenMade ?? p.lo),
        generation:    p.generation ?? 1,
        conf:          eta.conf        ?? 50,
        direction:     eta.direction   ?? 'neutral',
        expectedGap:   eta.expectedGap ?? null,
        composite:     eta.composite   ?? null,
      };
    }
    lockedRef.current = map; // FIX: keep ref in sync
    setLocked(map);
  }, []);

  // Poll history from DB every 10s
  // FIX 3: raised limit from 500 to 5000 — 500 only showed ~5% of real history
  const loadHistory = useCallback(async () => {
    const res = await apiFetch(`${API_URL}/predictions?source=${SOURCE}&limit=5000`);
    if (!res.ok) return;
    const mw = targetMaxWidth.current;
    const rows = (res.predictions || []).filter(r => {
      if (r.hitRound == null) return true;
      const w = mw[r.target] ?? 5;
      return r.hitRound >= r.lo - w && r.hitRound <= r.hi + w;
    });
    setHistory(rows);
  }, []);

  useEffect(() => {
    loadLocked();
    loadHistory();
    const lockedId  = setInterval(loadLocked,  8000);
    const historyId = setInterval(loadHistory, 10000);
    return () => { clearInterval(lockedId); clearInterval(historyId); };
  }, [loadLocked, loadHistory]);

  // Resolution check — runs on every new round
  // Uses lockedRef (not locked state) to avoid stale closure.
  // processingRef prevents concurrent executions.
  const processRound = useCallback(async (snap) => {
    if (!snap.length) return;
    const newId = snap[snap.length - 1]?.roundId ?? 0;
    if (newId <= prevIdRef.current) return;
    prevIdRef.current = newId;

    let dirty = false;

    for (const target of TARGETS) {
      const w = lockedRef.current[target.label]; // FIX: use ref, not state
      if (!w) continue;

      const { lo, hi, roundWhenMade, generation } = w;
      const key = `${SOURCE}:${target.label}:${lo}:${hi}`;
      if (resolvedRef.current.has(key)) continue;

      const isExpired = newId > hi;

      // Check early hit (hit before window opened)
      const fromId   = roundWhenMade ?? lo;
      const earlyHit = lo > fromId ? findHitInRange(snap, fromId, lo - 1, target.min) : null;
      if (earlyHit) {
        resolvedRef.current.add(key);
        await dbSaveResolved(target.label, 'early', lo, hi, earlyHit.roundId, generation);
        dirty = true;
        continue;
      }

      // FIX: check for hit inside active window immediately — don't wait for expiry.
      // Previously used "if (!isExpired) continue" which delayed win saves until
      // after window closed. Server covers this within 8s but frontend should too.
      const hit = findHitInRange(snap, lo, hi, target.min);
      if (hit) {
        resolvedRef.current.add(key);
        await dbSaveResolved(target.label, 'win', lo, hi, hit.roundId, generation);
        dirty = true;
        continue;
      }

      if (!isExpired) continue; // window still active, no hit yet — wait

      resolvedRef.current.add(key);
      await dbSaveResolved(target.label, 'loss', lo, hi, null, generation);
      dirty = true;
    }

    if (dirty) setTimeout(() => { loadHistory(); loadLocked(); }, 800);
  }, [loadHistory, loadLocked]);

  // FIX: processingRef guard — queues pending round instead of running concurrently
  useEffect(() => {
    if (!rounds.length) return;
    const snap = [...rounds];
    if (processingRef.current) { pendingRef.current = snap; return; }
    processingRef.current = true;
    pendingRef.current = null;
    processRound(snap).finally(() => {
      processingRef.current = false;
      const p = pendingRef.current;
      if (p) {
        pendingRef.current = null;
        processingRef.current = true;
        processRound(p).finally(() => { processingRef.current = false; });
      }
    });
  }, [rounds, processRound]);

  return { locked, history };
}

// ─────────────────────────────────────────────────────────────────────────────
// PRED CARD
// ─────────────────────────────────────────────────────────────────────────────
const PredCard = memo(({ target, locked, currentRoundId }) => {
  if (!locked) return (
    <div className="ptn-card-empty">
      <div style={{ display:'flex', alignItems:'center', gap:9 }}>
        <span style={{ fontSize:15 }}>{target.emoji}</span>
        <span style={{ fontSize:14, fontWeight:900, color:target.color, fontFamily:'JetBrains Mono,monospace' }}>{target.label}</span>
        <span style={{ marginLeft:'auto', fontSize:11, color:C.meta, fontFamily:'JetBrains Mono,monospace' }}>computing…</span>
      </div>
    </div>
  );

  const { lo, hi, conf, direction, expectedGap, composite } = locked;
  const isActive    = currentRoundId >= lo && currentRoundId <= hi;
  const isCountdown = currentRoundId < lo;
  const isPast      = currentRoundId > hi;
  const roundsToOpen = isCountdown ? lo - currentRoundId : 0;
  const roundsLeft   = isActive    ? hi - currentRoundId : 0;

  let cardClass = 'ptn-card';
  if (isActive)    cardClass = 'ptn-card-active';
  else if (isPast) cardClass = 'ptn-card-stale';
  else             cardClass = 'ptn-card-waiting';

  const confColor = conf >= 70 ? C.green : conf >= 50 ? C.yellow : C.orange;
  const dirColor  = direction === 'bullish' ? C.green : direction === 'bearish' ? C.red : C.meta;
  const dirLabel  = direction === 'bullish' ? '▲ BULLISH' : direction === 'bearish' ? '▼ BEARISH' : '→ NEUTRAL';

  return (
    <div className={cardClass}>
      <div style={{ display:'flex', alignItems:'center', gap:8, flexWrap:'wrap' }}>
        <span style={{ fontSize:16 }}>{target.emoji}</span>
        <span style={{ fontSize:15, fontWeight:900, color:isActive?C.green:target.color, fontFamily:'JetBrains Mono,monospace' }}>{target.label}</span>

        {isActive    && <span className="ptn-badge" style={{ background:'#00ff8820', color:C.green,  border:`1px solid ${C.green}` }}>● ACTIVE NOW</span>}
        {isCountdown && roundsToOpen<=500 && <span className="ptn-badge" style={{ background:'#ffd84d14', color:C.yellow, border:`1px solid ${C.yellow}50` }}>OPENS IN {roundsToOpen}r</span>}
        {isCountdown && roundsToOpen >500 && <span className="ptn-badge" style={{ background:'#ff8c4214', color:C.orange, border:`1px solid ${C.orange}50` }}>OPENS IN {roundsToOpen}r</span>}
        {isPast      && <span className="ptn-badge" style={{ background:'#ff456014', color:C.red, border:`1px solid ${C.red}50` }}>EXPIRED</span>}
        <span className="ptn-badge" style={{ background:`${dirColor}14`, color:dirColor, border:`1px solid ${dirColor}50`, fontSize:9 }}>{dirLabel}</span>

        <span style={{ marginLeft:'auto', display:'flex', alignItems:'center', gap:10 }}>
          {isActive && <span style={{ fontSize:11, color:C.green, fontFamily:'JetBrains Mono,monospace', fontWeight:700 }}>{roundsLeft}r left</span>}
          <span style={{ fontSize:12, fontWeight:900, fontFamily:'JetBrains Mono,monospace', color:confColor }}>{conf}% conf</span>
          <span style={{ fontSize:12, fontWeight:800, color:isActive?C.green:C.label, fontFamily:'JetBrains Mono,monospace' }}>#{lo}–#{hi}</span>
        </span>
      </div>
      <div style={{ marginTop:7, display:'flex', gap:12, flexWrap:'wrap', fontSize:10, color:C.meta, fontFamily:'JetBrains Mono,monospace' }}>
        {expectedGap   != null && <span>gap~{expectedGap}r</span>}
        {locked.gapSinceLast != null && <span>since last:{locked.gapSinceLast}r</span>}
        {composite     != null && <span>signal={composite>0?'+':''}{composite}</span>}
        {locked.momentum != null && <span>mom={locked.momentum>0?'+':''}{locked.momentum}</span>}
        {isCountdown   && <span style={{ color:C.cyan }}>opens #{lo}</span>}
      </div>
    </div>
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// HISTORY PANEL
// ─────────────────────────────────────────────────────────────────────────────
const HistoryPanel = memo(({ history }) => {
  const [filter,  setFilter]  = useState('ALL');
  const [visible, setVisible] = useState(HIST_PAGE);

  useEffect(() => { setVisible(HIST_PAGE); }, [filter]);

  const filtered = filter === 'ALL' ? history : history.filter(r => r.target === filter);
  const wins   = filtered.filter(r => r.outcome === 'win').length;
  const losses = filtered.filter(r => r.outcome === 'loss').length;
  const early  = filtered.filter(r => r.outcome === 'early').length;
  const total  = wins + losses;
  const wr     = total > 0 ? Math.round(wins / total * 100) : null;

  return (
    <div style={{ marginTop:14, background:C.card, border:`1px solid ${C.border}`, borderRadius:10, padding:'12px 14px' }}>
      <div style={{ display:'flex', alignItems:'center', flexWrap:'wrap', gap:8, marginBottom:10 }}>
        <span style={{ fontSize:12, fontWeight:800, color:C.cyan, letterSpacing:'.08em', fontFamily:'JetBrains Mono,monospace' }}>PTN HISTORY</span>
        <div style={{ marginLeft:'auto', display:'flex', alignItems:'center', gap:10, flexWrap:'wrap' }}>
          {wr != null && <span style={{ fontSize:12, fontWeight:800, fontFamily:'JetBrains Mono,monospace', color:wr>=60?C.green:wr>=45?C.yellow:C.red }}>{wr}% HIT RATE</span>}
          <span style={{ fontSize:11, color:C.green,  fontWeight:700, fontFamily:'JetBrains Mono,monospace' }}>{wins}W</span>
          <span style={{ fontSize:11, color:C.red,    fontWeight:700, fontFamily:'JetBrains Mono,monospace' }}>{losses}L</span>
          {early > 0 && <span style={{ fontSize:11, color:C.yellow, fontWeight:700, fontFamily:'JetBrains Mono,monospace' }}>{early}E</span>}
          <span style={{ fontSize:11, color:C.label, fontFamily:'JetBrains Mono,monospace' }}>{total} total</span>
          <select value={filter} onChange={e => setFilter(e.target.value)}
            style={{ background:'#0a1428', border:`1px solid ${C.border}`, borderRadius:5, padding:'3px 7px', fontSize:10, color:C.text, outline:'none', cursor:'pointer', fontFamily:'JetBrains Mono,monospace' }}>
            <option value="ALL">All targets</option>
            {TARGETS.map(t => <option key={t.label} value={t.label}>{t.label}</option>)}
          </select>
        </div>
      </div>

      {filtered.length === 0 ? (
        <div style={{ textAlign:'center', padding:'20px 0', fontSize:12, color:C.label }}>No resolved predictions yet</div>
      ) : (
        <>
          <div className="ptn-hrow" style={{ fontSize:9, fontWeight:800, color:C.meta, letterSpacing:'.08em', fontFamily:'JetBrains Mono,monospace' }}>
            <span>TARGET</span><span>RESULT</span><span>WINDOW</span><span>HIT AT</span>
          </div>
          {filtered.slice(0, visible).map((r, i) => {
            const tc      = TARGETS.find(t => t.label === r.target)?.color ?? C.label;
            const isWin   = r.outcome === 'win';
            const isEarly = r.outcome === 'early';
            return (
              <div key={`${r.target}-${r.lo}-${r.hi}-${i}`} className="ptn-hrow">
                <span style={{ fontWeight:800, color:tc, fontFamily:'JetBrains Mono,monospace' }}>{r.target}</span>
                <span style={{ fontWeight:800, fontFamily:'JetBrains Mono,monospace', color:isWin?C.green:isEarly?C.yellow:C.red }}>
                  {isWin ? 'HIT ✓' : isEarly ? 'EARLY' : 'MISS ✗'}
                </span>
                <span style={{ color:C.label, fontFamily:'JetBrains Mono,monospace', fontSize:10, fontWeight:600 }}>#{r.lo}–#{r.hi}</span>
                <span style={{ color:r.hitRound?C.white:C.meta, fontFamily:'JetBrains Mono,monospace', fontWeight:r.hitRound?700:400 }}>
                  {r.hitRound ? `#${r.hitRound}` : '—'}
                </span>
              </div>
            );
          })}
          {visible < filtered.length && (
            <button className="ptn-more" onClick={() => setVisible(v => v + HIST_PAGE)}>
              + SHOW MORE ({filtered.length - visible} remaining)
            </button>
          )}
        </>
      )}
    </div>
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// MAIN EXPORT
// ─────────────────────────────────────────────────────────────────────────────
export default function PatternEngine({ rounds, currentRoundId, totalTracked }) {
  useEffect(() => { injectStyles(); }, []);

  const sortedRounds = React.useMemo(() => {
    if (!rounds.length) return [];
    let ok = true;
    for (let i = 1; i < rounds.length; i++) {
      if (rounds[i].roundId < rounds[i - 1].roundId) { ok = false; break; }
    }
    return ok ? rounds : [...rounds].sort((a, b) => a.roundId - b.roundId);
  }, [rounds]);

  const lastId = sortedRounds.length
    ? sortedRounds[sortedRounds.length - 1].roundId
    : (currentRoundId ?? 0);

  const { locked, history } = usePTN(sortedRounds);

  if (sortedRounds.length < MIN_ROUNDS) {
    return (
      <div className="ptn" style={{ padding:20, textAlign:'center', paddingTop:50 }}>
        <div style={{ fontSize:28, marginBottom:14 }}>📊</div>
        <div style={{ fontSize:13, color:C.label, marginBottom:12, fontFamily:'JetBrains Mono,monospace' }}>
          {sortedRounds.length} / {MIN_ROUNDS} rounds needed
        </div>
        <div style={{ background:'#0a1428', height:4, borderRadius:2, maxWidth:200, margin:'0 auto' }}>
          <div style={{ width:`${(sortedRounds.length/MIN_ROUNDS)*100}%`, height:'100%', background:C.green, borderRadius:2 }} />
        </div>
      </div>
    );
  }

  return (
    <div className="ptn" style={{ padding:13 }}>

      {/* Header */}
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:12 }}>
        <div style={{ display:'flex', alignItems:'center', gap:10 }}>
          <span style={{ fontSize:13, fontWeight:800, color:C.cyan, letterSpacing:'.1em', fontFamily:'JetBrains Mono,monospace' }}>PTN</span>
          <span className="ptn-badge" style={{ background:'#00ff8814', color:C.green, border:`1px solid ${C.green}40` }}>PATTERN ENGINE</span>
          <span className="ptn-badge" style={{ background:'#00c8f014', color:C.cyan,  border:`1px solid ${C.cyan}40`  }}>DB-BACKED</span>
        </div>
        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          <span style={{ fontSize:11, color:C.meta, fontFamily:'JetBrains Mono,monospace' }}>TRAINING DATA</span>
          <span style={{ background:'#00c8f018', border:`1px solid ${C.cyan}40`, borderRadius:5, padding:'2px 8px', fontSize:12, fontWeight:800, color:C.cyan, fontFamily:'JetBrains Mono,monospace' }}>
            {(totalTracked ?? sortedRounds.length).toLocaleString()}r
          </span>
          {lastId > 0 && (
            <span style={{ background:'#00ff8814', border:`1px solid ${C.green}`, borderRadius:5, padding:'2px 8px', fontSize:11, fontWeight:800, color:C.green, fontFamily:'JetBrains Mono,monospace' }}>
              #{lastId}
            </span>
          )}
        </div>
      </div>

      {/* Info bar */}
      <div style={{ marginBottom:12, padding:'7px 11px', background:'#0a1428', borderRadius:6, border:`1px solid ${C.border}`, fontSize:11, color:C.label }}>
        Pure pattern recognition — clustering · trend · autocorrelation · momentum · no stats, no ML
      </div>

      {/* Prediction cards */}
      <div style={{ display:'flex', flexDirection:'column', gap:7 }}>
        {TARGETS.map(target => (
          <PredCard
            key={target.label}
            target={target}
            locked={locked[target.label] ?? null}
            currentRoundId={lastId}
          />
        ))}
      </div>

      {/* History */}
      <HistoryPanel history={history} />

      {/* Footer */}
      <div style={{ marginTop:16, fontSize:10, color:C.meta, textAlign:'center', fontFamily:'JetBrains Mono,monospace', letterSpacing:'.06em' }}>
        PURE PATTERN RECOGNITION · FULL {(totalTracked ?? sortedRounds.length).toLocaleString()} ROUNDS · DB-BACKED
      </div>
    </div>
  );
}