import React, { useCallback, useEffect, useRef, useState } from 'react';
import { API_BASE } from '../config/apiBase';

const API_URL = API_BASE;
const API_TIMEOUT_MS = 20000;
const AUTO_REFRESH_MS = 20000;

const FALLBACK_WINDOWS = [
  { key: '5m',  label: '5 Minutes' },
  { key: '10m', label: '10 Minutes' },
  { key: '30m', label: '30 Minutes' },
  { key: '1h',  label: '1 Hour' },
  { key: '2h',  label: '2 Hours' },
  { key: '5h',  label: '5 Hours' },
];

const FALLBACK_TARGETS = [
  { value: 5,    label: '5x' },
  { value: 10,   label: '10x' },
  { value: 20,   label: '20x' },
  { value: 50,   label: '50x' },
  { value: 100,  label: '100x' },
  { value: 500,  label: '500x' },
  { value: 1000, label: '1000x' },
];

function fetchJsonWithTimeout(url, timeoutMs = API_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { cache: 'no-store', signal: controller.signal })
    .then(async (res) => {
      const raw = await res.text();
      let data = null;
      try { data = JSON.parse(raw); } catch {}
      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
      return data;
    })
    .catch((err) => {
      if (err?.name === 'AbortError') throw new Error(`Timeout after ${Math.ceil(timeoutMs / 1000)}s`);
      throw err;
    })
    .finally(() => clearTimeout(timer));
}

function pct(value) {
  const n = Number(value || 0);
  return `${n.toFixed(1)}%`;
}

function liftLabel(lift) {
  const n = Number(lift || 1);
  if (!Number.isFinite(n)) return '-';
  return `${n.toFixed(2)}x`;
}

function HeatmapGrid({ heatmap }) {
  if (!heatmap || !heatmap.cells || !heatmap.days || !heatmap.hours) return null;
  const { days, hours, cells } = heatmap;
  const lookup = {};
  for (const cell of cells) {
    if (!lookup[cell.weekday]) lookup[cell.weekday] = {};
    lookup[cell.weekday][cell.hour] = cell;
  }
  const maxRate = Math.max(...cells.map((c) => c.hitRate || 0), 0.001);

  return (
    <div className="timing-heatmap-wrap" style={{ overflowX: 'auto' }}>
      <div
        className="timing-heatmap-grid"
        style={{ display: 'grid', gridTemplateColumns: `48px repeat(${hours.length}, 1fr)`, gap: 2, minWidth: 600 }}
      >
        <div />
        {hours.map((h) => (
          <div key={h.value} style={{ fontSize: 10, color: 'var(--muted)', textAlign: 'center', padding: '2px 0' }}>{h.label}</div>
        ))}
        {days.map((day) => (
          <React.Fragment key={day}>
            <div style={{ fontSize: 11, color: 'var(--muted)', display: 'flex', alignItems: 'center', fontWeight: 600 }}>{day}</div>
            {hours.map((h) => {
              const cell = lookup[day]?.[h.value];
              const rate = cell?.hitRate || 0;
              const intensity = maxRate > 0 ? rate / maxRate : 0;
              const alpha = 0.07 + intensity * 0.83;
              const bg = cell?.rounds > 0
                ? `rgba(34, 211, 238, ${alpha.toFixed(2)})`
                : 'rgba(255,255,255,0.04)';
              return (
                <div
                  key={h.value}
                  title={cell ? `${day} ${h.label}: ${pct(cell.hitRate * 100)} hit rate (${cell.rounds} rounds)` : 'No data'}
                  style={{
                    background: bg,
                    borderRadius: 3,
                    padding: '4px 2px',
                    fontSize: 9,
                    color: 'rgba(255,255,255,0.85)',
                    textAlign: 'center',
                    minHeight: 28,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {cell?.rounds > 0 ? pct(rate * 100) : ''}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function BestWindowsList({ items }) {
  if (!items || !items.length) return <div className="pattern-grid-empty">No window data available yet.</div>;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {items.slice(0, 8).map((item, i) => (
        <div
          key={i}
          style={{
            display: 'flex', gap: 10, alignItems: 'center', padding: '8px 12px',
            background: item.recommended ? 'rgba(34,211,238,0.08)' : 'rgba(255,255,255,0.03)',
            borderRadius: 6, border: item.recommended ? '1px solid rgba(34,211,238,0.25)' : '1px solid rgba(255,255,255,0.06)',
          }}
        >
          <span style={{ color: 'var(--muted)', fontSize: 12, width: 24 }}>#{i + 1}</span>
          <span style={{ flex: 1, fontSize: 13, fontWeight: 600 }}>{item.slotLabel || item.windowLabel || '-'}</span>
          <span style={{ fontSize: 13, color: '#22d3ee' }}>{pct(Number(item.hitRate || 0) * 100)}</span>
          <span style={{ fontSize: 12, color: 'var(--muted)' }}>{liftLabel(item.lift)} lift</span>
          <span style={{ fontSize: 11, color: 'var(--muted)' }}>{item.rounds || 0} rounds</span>
          {item.recommended && (
            <span style={{ fontSize: 10, background: '#22d3ee', color: '#000', borderRadius: 3, padding: '1px 5px', fontWeight: 700 }}>REC</span>
          )}
        </div>
      ))}
    </div>
  );
}

function HourlyTable({ hourlyHistory }) {
  const rows = hourlyHistory?.rows || [];
  const bestHours = new Set((hourlyHistory?.bestHours || []).map((h) => h.hour));
  if (!rows.length) return <div className="pattern-grid-empty">No hourly data yet.</div>;
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
            {['Hour', 'Hit Rate', 'Lift', 'Rounds', 'Hits'].map((h) => (
              <th key={h} style={{ padding: '6px 10px', textAlign: 'left', color: 'var(--muted)', fontWeight: 600, fontSize: 12 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.hour}
              style={{
                borderBottom: '1px solid rgba(255,255,255,0.05)',
                background: bestHours.has(row.hour) ? 'rgba(34,211,238,0.06)' : 'transparent',
              }}
            >
              <td style={{ padding: '6px 10px', fontWeight: bestHours.has(row.hour) ? 700 : 400 }}>{row.hourLabel || `${row.hour}:00`}</td>
              <td style={{ padding: '6px 10px', color: '#22d3ee' }}>{pct(Number(row.hitRate || 0) * 100)}</td>
              <td style={{ padding: '6px 10px' }}>{liftLabel(row.lift)}</td>
              <td style={{ padding: '6px 10px', color: 'var(--muted)' }}>{row.rounds || 0}</td>
              <td style={{ padding: '6px 10px', color: 'var(--muted)' }}>{row.hits || 0}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function TimingAnalyticsTab() {
  const [windowKey, setWindowKey] = useState('5m');
  const [focusTarget, setFocusTarget] = useState('5');
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const reportRef = useRef(null);
  const requestSeqRef = useRef(0);

  useEffect(() => { reportRef.current = report; }, [report]);

  const loadReport = useCallback(async () => {
    if (!API_URL) { setError('API URL not configured'); setLoading(false); return; }
    const hasReport = Boolean(reportRef.current);
    if (!hasReport) setLoading(true); else setRefreshing(true);
    const seq = ++requestSeqRef.current;
    try {
      const params = new URLSearchParams({
        window: windowKey,
        focusTarget: String(focusTarget || '5'),
        timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
      });
      const data = await fetchJsonWithTimeout(`${API_URL}/analytics/timing?${params.toString()}`);
      if (!data?.ok) throw new Error(data?.error || 'Timing analytics request failed');
      if (seq !== requestSeqRef.current) return;
      setReport(data);
      setError('');
    } catch (err) {
      if (seq !== requestSeqRef.current) return;
      setError(err.message || 'Timing analytics request failed');
    } finally {
      if (seq === requestSeqRef.current) { setLoading(false); setRefreshing(false); }
    }
  }, [focusTarget, windowKey]);

  useEffect(() => {
    requestSeqRef.current += 1;
    setReport(null); setError(''); setLoading(true); setRefreshing(false);
  }, [focusTarget, windowKey]);

  useEffect(() => { loadReport(); }, [loadReport]);
  useEffect(() => {
    const id = setInterval(() => {
      if (document.visibilityState !== 'visible') return;
      loadReport();
    }, AUTO_REFRESH_MS);
    return () => clearInterval(id);
  }, [loadReport]);

  const availableWindows = report?.availableWindows || FALLBACK_WINDOWS;
  const availableTargets = report?.availableTargets || FALLBACK_TARGETS;
  const prediction = report?.patternPrediction || null;
  const decision = report?.decision || null;
  const dataset = report?.dataset || {};
  const comparison = report?.comparison || null;
  const bestWindowsToday = report?.bestWindowsToday || null;
  const hourlyHistory = report?.hourlyHistory || null;
  const heatmap = report?.dayHourHeatmap || null;
  const currentWindow = report?.currentWindow || null;
  const baseline = report?.baseline || null;
  const targetCards = report?.targetCards || [];
  const cooldowns = report?.cooldowns || [];

  const decisionAction = decision?.action || prediction?.action || 'SKIP';
  const isGood = decisionAction === 'PLAY';

  return (
    <main className="pattern-page">
      <div className="pattern-shell">

        {/* HEADER */}
        <section className="panel pattern-hero-panel">
          <div className="panel-header">
            <span className="panel-icon">TM</span>
            <h2>TIMING ENGINE</h2>
          </div>
          <p className="panel-subtitle">
            Time-window analysis. Compares the current time slot against all historical patterns to determine if this window is statistically favourable for your chosen target multiplier.
          </p>

          <div className="pattern-controls">
            <div className="pattern-control">
              <label>Time Window</label>
              <select className="timing-select" value={windowKey} onChange={(e) => setWindowKey(e.target.value)}>
                {availableWindows.map((w) => (
                  <option key={w.key} value={w.key}>{w.label}</option>
                ))}
              </select>
            </div>
            <div className="pattern-control">
              <label>Target</label>
              <select className="timing-select" value={focusTarget} onChange={(e) => setFocusTarget(e.target.value)}>
                {availableTargets.map((t) => (
                  <option key={t.value} value={t.value}>{t.label}</option>
                ))}
              </select>
            </div>
            <div className="pattern-actions">
              <button className="btn btn-refresh" onClick={() => loadReport()} disabled={loading || refreshing}>
                {refreshing ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>
          </div>

          <div className="pattern-note-strip">
            <span>Dataset</span>
            <b>{dataset?.totalRounds?.toLocaleString?.() || 0} rounds</b>
            <span className="pattern-note-sep">|</span>
            <span>Span</span>
            <b>{Number(dataset?.spanDays || 0).toFixed(1)} days</b>
            <span className="pattern-note-sep">|</span>
            <span>Window</span>
            <b>{report?.window?.label || '-'}</b>
            <span className="pattern-note-sep">|</span>
            <span>Target</span>
            <b>{report?.focusTargetLabel || `${focusTarget}x`}</b>
          </div>

          {error ? <div className="timing-error">{error}</div> : null}
        </section>

        {loading && !report ? (
          <section className="panel timing-loading-panel">
            <div className="timing-loading-copy">Loading timing analytics...</div>
          </section>
        ) : null}

        {report ? (
          <>
            {/* DECISION */}
            <section className={`panel pattern-decision-panel ${isGood ? 'pattern-decision-good' : 'pattern-decision-bad'}`}>
              <div className="pattern-decision-top">
                <div>
                  <span className="pattern-decision-label">Current Window Call</span>
                  <b>{decisionAction}</b>
                </div>
                <div className="pattern-pill">{comparison?.label || decisionAction}</div>
              </div>
              <p className="pattern-decision-copy">
                {decision?.summary || prediction?.summary || comparison?.message || 'Analysing current timing window...'}
              </p>

              <div className="pattern-facts-grid">
                <div className="pattern-fact-card">
                  <span>Window Hit Rate</span>
                  <b>{pct(Number(currentWindow?.hitRate || 0) * 100)}</b>
                  <small>Current time slot</small>
                </div>
                <div className="pattern-fact-card">
                  <span>Baseline Hit Rate</span>
                  <b>{pct(Number(baseline?.perRoundHitRates?.[focusTarget] || baseline?.hitRate || 0) * 100)}</b>
                  <small>All-time average</small>
                </div>
                <div className="pattern-fact-card">
                  <span>Timing Edge</span>
                  <b>{prediction?.timingEdgeLabel || '-'}</b>
                  <small>Signal strength</small>
                </div>
                <div className="pattern-fact-card">
                  <span>Window Lift</span>
                  <b>{liftLabel(prediction?.currentWindowLift || prediction?.currentLift)}</b>
                  <small>vs baseline</small>
                </div>
                <div className="pattern-fact-card">
                  <span>Pattern Prediction</span>
                  <b>{prediction?.action || '-'}</b>
                  <small>{pct(Number(prediction?.matchedHitRatePercent || 0))} matched hit rate</small>
                </div>
                <div className="pattern-fact-card">
                  <span>Matched Windows</span>
                  <b>{prediction?.matchedWindows || 0}</b>
                  <small>Historical matches used</small>
                </div>
                <div className="pattern-fact-card">
                  <span>Expected Round</span>
                  <b>{prediction?.expectedRoundIdLabel || '-'}</b>
                  <small>{prediction?.expectedRoundIdBasis || 'Round estimate'}</small>
                </div>
                <div className="pattern-fact-card">
                  <span>Hits So Far</span>
                  <b>{prediction?.hitsSoFar ?? 0}</b>
                  <small>In current window</small>
                </div>
              </div>
            </section>

            {/* BEST WINDOWS */}
            {bestWindowsToday?.items?.length > 0 && (
              <section className="panel">
                <div className="panel-header">
                  <span className="panel-icon">BW</span>
                  <h2>BEST WINDOWS TODAY</h2>
                </div>
                <p className="panel-subtitle">
                  Historical hit-rate rankings for today's time slots — highest performing windows first.
                </p>
                <BestWindowsList items={bestWindowsToday.items} />
              </section>
            )}

            {/* TARGET CARDS */}
            {targetCards.length > 0 && (
              <section className="panel">
                <div className="panel-header">
                  <span className="panel-icon">TC</span>
                  <h2>TARGET READINESS</h2>
                </div>
                <p className="panel-subtitle">Current window performance for each target multiplier.</p>
                <div className="pattern-facts-grid">
                  {targetCards.map((card) => (
                    <div key={card.target} className="pattern-fact-card">
                      <span>{card.targetLabel || `${card.target}x`}</span>
                      <b>{pct(Number(card.currentHitRate || 0) * 100)}</b>
                      <small>{liftLabel(card.lift)} lift · {card.hits || 0} hits / {card.rounds || 0} rounds</small>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* COOLDOWNS */}
            {cooldowns.length > 0 && (
              <section className="panel">
                <div className="panel-header">
                  <span className="panel-icon">CD</span>
                  <h2>COOLDOWNS</h2>
                </div>
                <p className="panel-subtitle">
                  Time since last hit for each target — useful for spotting targets that are overdue or recently hit.
                </p>
                <div className="pattern-facts-grid">
                  {cooldowns.map((cd) => (
                    <div key={cd.target} className="pattern-fact-card">
                      <span>{cd.targetLabel || `${cd.target}x`}</span>
                      <b>{cd.cooldownLabel || '-'}</b>
                      <small>{cd.status || ''}</small>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* HEATMAP */}
            {heatmap && (
              <section className="panel">
                <div className="panel-header">
                  <span className="panel-icon">HM</span>
                  <h2>DAY × HOUR HEATMAP</h2>
                </div>
                <p className="panel-subtitle">
                  Hit-rate by weekday and hour. Brighter = higher hit rate for {report?.focusTargetLabel || `${focusTarget}x`}.
                </p>
                <HeatmapGrid heatmap={heatmap} />
              </section>
            )}

            {/* HOURLY HISTORY */}
            {hourlyHistory && (
              <section className="panel">
                <div className="panel-header">
                  <span className="panel-icon">HH</span>
                  <h2>HOURLY HISTORY</h2>
                </div>
                <p className="panel-subtitle">
                  Hit rate and lift by hour of day across the full dataset.
                </p>
                <HourlyTable hourlyHistory={hourlyHistory} />
              </section>
            )}
          </>
        ) : null}
      </div>
    </main>
  );
}