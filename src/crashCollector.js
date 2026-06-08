'use strict';

const fetch = require('node-fetch');
const {
  getCrashSites,
  getCrashSiteById,
  upsertCrashSite,
  saveCrashRounds,
  saveTowerPredictionHistory,
  touchCrashSite,
} = require('./db');

const DEFAULT_CRASH_SOURCES = [
  {
    sourceKey: 'degencoinflip-crash',
    label: 'Degen Coin Flip',
    gameUrl: 'https://app.degencoinflip.com/crash',
    adapter: 'degencoinflip',
    apiUrl: 'https://api.dealer.degencoinflip.com/v1/game/2/room/1/rounds?limit=100',
    roundsPath: '?limit=100',
  },
  {
    sourceKey: 'solanafatboys-bustonaut',
    label: 'Solana Fat Boys Bustonaut',
    gameUrl: 'https://www.solanafatboys.com/bustonaut/',
    adapter: 'sfb',
    apiUrl: 'https://sfb-api-service-mainnet.up.railway.app/api/games/bustonaut/latest?page=0&limit=100',
    roundsPath: '?page=0&limit=100',
  },
  {
    sourceKey: 'degencoinflip-towers',
    label: 'Degen Coin Flip Towers',
    gameUrl: 'https://app.degencoinflip.com/towers',
    adapter: 'degencoinflip',
    apiUrl: 'https://api.dealer.degencoinflip.com/v1/game/3/room/1/rounds?limit=100',
    roundsPath: '?limit=100',
  },
];

const DEPRECATED_CRASH_SOURCE_KEYS = new Set([
  'bcgame-crash',
  'stake-crash',
  'solcrash-crash',
  'solpump-crash',
]);

const DEPRECATED_CRASH_URL_PATTERNS = [
  'bcgame52.com/game/crash',
  'stake.bet/casino/games/crash',
  'solcrash.io/play',
  'solpump.io/fairness/crash',
];

const POLL_LOOP_MS = Number.parseInt(process.env.CRASH_WATCH_POLL_MS || '30000', 10);
const POLL_TIMEOUT_MS = Number.parseInt(process.env.CRASH_WATCH_TIMEOUT_MS || '12000', 10);
const DISCOVERY_TIMEOUT_MS = Number.parseInt(process.env.CRASH_WATCH_DISCOVERY_TIMEOUT_MS || '12000', 10);
const DISCOVERY_BUNDLE_LIMIT = Number.parseInt(process.env.CRASH_WATCH_DISCOVERY_BUNDLE_LIMIT || '4', 10);
const DISCOVERY_TEXT_LIMIT = Number.parseInt(process.env.CRASH_WATCH_DISCOVERY_TEXT_LIMIT || '450000', 10);

let collectorStarted = false;
let tickTimer = null;
let inFlight = false;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function withTimeout(promise, ms, label) {
  let timer = null;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timer) clearTimeout(timer);
  });
}

function normalizeUrl(raw) {
  const value = String(raw || '').trim();
  if (!value) return '';
  return value.replace(/\/+$/, '');
}

function normalizeGameKey(raw) {
  const value = normalizeUrl(raw);
  if (!value) return '';
  try {
    return new URL(value).origin.toLowerCase() + new URL(value).pathname.toLowerCase();
  } catch {
    return value.toLowerCase();
  }
}

function resolveUrl(baseUrl, maybeRelative) {
  try {
    return new URL(maybeRelative, baseUrl).href;
  } catch {
    return '';
  }
}

function looksLikeChallengePage(text) {
  const body = String(text || '');
  return /just a moment|enable javascript and cookies|cf-challenge|challenge-platform/i.test(body);
}

function parseFloatish(raw) {
  if (raw == null) return NaN;
  const text = String(raw).replace(/x$/i, '').replace(/,/g, '').trim();
  const match = text.match(/-?\d+(?:\.\d+)?/);
  return match ? Number.parseFloat(match[0]) : Number.parseFloat(text);
}

const TOWER_RESULT_MAP = {
  3: 'A',
  5: 'B',
  6: 'C',
};

function parseTowerGameResult(raw) {
  if (raw == null) return [];
  let value = raw;
  if (typeof value === 'string') {
    try {
      value = JSON.parse(value);
    } catch {
      return [];
    }
  }
  if (!Array.isArray(value)) return [];
  return value
    .map((entry) => {
      const level = Number(entry?.level);
      const result = String(entry?.result ?? entry?.choice ?? entry?.value ?? '').trim();
      const letter = TOWER_RESULT_MAP[Number(result)] || TOWER_RESULT_MAP[result] || null;
      return {
        level: Number.isFinite(level) ? level : null,
        result,
        letter,
      };
    })
    .filter((entry) => entry.letter);
}

function makeSyntheticRoundId(row, multiplier, timestamp) {
  const ts = Number.isFinite(timestamp) ? Math.floor(timestamp) : Date.now();
  const mult = Number.isFinite(multiplier) ? Math.round(multiplier * 1000) : 0;
  const seedSource = String(
    row?.roundId ??
    row?.round_id ??
    row?.id ??
    row?.gameId ??
    row?.gameRoundId ??
    row?.uuid ??
    row?.hash ??
    row?.createdAt ??
    row?.created_at ??
    row?.time ??
    row?.date ??
    ''
  );
  let hash = 0;
  for (let i = 0; i < seedSource.length; i += 1) {
    hash = ((hash << 5) - hash + seedSource.charCodeAt(i)) | 0;
  }
  const suffix = Math.abs(hash % 1000);
  return Number(`${ts}${String(mult).padStart(4, '0')}${String(suffix).padStart(3, '0')}`.slice(0, 15));
}

function extractRoundArray(payload) {
  if (!payload) return [];
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload.payload)) return payload.payload;
  if (Array.isArray(payload.rounds)) return payload.rounds;
  if (Array.isArray(payload.history)) return payload.history;
  if (Array.isArray(payload.items)) return payload.items;
  if (Array.isArray(payload.data)) return payload.data;
  if (Array.isArray(payload.entries)) return payload.entries;
  if (Array.isArray(payload.list)) return payload.list;
  if (Array.isArray(payload.result)) return payload.result;
  if (Array.isArray(payload.response)) return payload.response;
  if (payload.data && Array.isArray(payload.data.rounds)) return payload.data.rounds;
  if (payload.data && Array.isArray(payload.data.history)) return payload.data.history;
  if (payload.data && Array.isArray(payload.data.entries)) return payload.data.entries;
  if (payload.data && Array.isArray(payload.data.list)) return payload.data.list;
  if (payload.data && Array.isArray(payload.data.items)) return payload.data.items;
  if (payload.data && Array.isArray(payload.data.result)) return payload.data.result;
  if (payload.data && Array.isArray(payload.data.response)) return payload.data.response;
  if (payload.result && Array.isArray(payload.result.history)) return payload.result.history;
  if (payload.result && Array.isArray(payload.result.entries)) return payload.result.entries;
  if (payload.result && Array.isArray(payload.result.list)) return payload.result.list;
  return [];
}

function normalizeRoundRow(row) {
  if (Array.isArray(row)) {
    const [roundIdRaw, multiplierRaw, timestampRaw] = row;
    const multiplier = parseFloatish(multiplierRaw);
    const timestamp = timestampRaw != null ? Number(timestampRaw) : Date.now();
    const roundId = Number.isFinite(Number(roundIdRaw))
      ? Number(roundIdRaw)
      : makeSyntheticRoundId(row, multiplier, timestamp);
    if (!Number.isFinite(multiplier)) return null;
    return {
      roundId,
      multiplier,
      timestamp: Number.isFinite(timestamp) ? timestamp : Date.now(),
      raw: row,
    };
  }

  if (!row || typeof row !== 'object') return null;
  const towerSequence = parseTowerGameResult(row.gameResult);
  const roundId = Number(
    row.roundId ??
    row.round_id ??
    row.id ??
    row.gameId ??
    row.gameRoundId ??
    row.round ??
    row.sequence ??
    row.block ??
    row.index ??
    row.number ??
    row.game_number
  );
  const multiplier = parseFloatish(
    row.multiplier ??
    row.bustMultiplier ??
    row.crashMultiplier ??
    row.bust ??
    row.gameResult ??
    row.crashPoint ??
    row.crash_point ??
    row.result ??
    row.payout ??
    row.cashout ??
    row.value ??
    row.score ??
    row.finalMultiplier
  );
  const resolvedMultiplier = Number.isFinite(multiplier)
    ? multiplier
    : (towerSequence.length ? towerSequence.length : NaN);
  const timestamp = row.timestamp != null
    ? Number(row.timestamp)
    : row.createdAt
      ? Number(new Date(row.createdAt))
      : row.created_at
        ? Number(new Date(row.created_at))
        : row.time
          ? Number(new Date(row.time))
          : row.date
            ? Number(new Date(row.date))
            : Date.now();
  const resolvedRoundId = Number.isFinite(roundId)
    ? roundId
    : makeSyntheticRoundId(row, resolvedMultiplier, timestamp);
  if (!Number.isFinite(resolvedRoundId) || !Number.isFinite(resolvedMultiplier)) return null;
  return {
    roundId: resolvedRoundId,
    multiplier: resolvedMultiplier,
    timestamp: Number.isFinite(timestamp) ? timestamp : Date.now(),
    rawPayload: towerSequence.length
      ? {
          ...row,
          towerSequence,
          towerSequenceText: towerSequence.map((entry) => entry.letter).join(''),
        }
      : row,
    raw: row,
  };
}

function extractTowerSequenceFromRound(round) {
  const raw = round?.rawPayload || round?.raw || round || null;
  if (!raw) return '';
  const parsed = parseTowerGameResult(raw.gameResult || raw.game_result || null);
  if (parsed.length) {
    return parsed.map((entry) => entry.letter).join('');
  }
  if (Array.isArray(raw.towerSequence) && raw.towerSequence.length) {
    return raw.towerSequence.map((entry) => entry?.letter).filter(Boolean).join('');
  }
  if (typeof raw.towerSequenceText === 'string' && raw.towerSequenceText.trim()) {
    return raw.towerSequenceText.trim();
  }
  const fallback = parseTowerGameResult(raw.sequence || null);
  return fallback.map((entry) => entry.letter).join('');
}

function normalizeTowerRounds(rounds = []) {
  return rounds
    .map((round) => ({
      ...round,
      sequence: extractTowerSequenceFromRound(round),
    }))
    .filter((round) => round.sequence && round.sequence.length >= 4);
}

function createTowerScoreBucket() {
  return { A: 0, B: 0, C: 0 };
}

function createTowerTransitionBucket() {
  return { A: createTowerScoreBucket(), B: createTowerScoreBucket(), C: createTowerScoreBucket() };
}

function getTowerMapBucket(map, key) {
  if (!map.has(key)) map.set(key, createTowerScoreBucket());
  return map.get(key);
}

function trainTowerModel(rounds) {
  const depth = 10;
  const positionStats = Array.from({ length: depth }, () => createTowerScoreBucket());
  const transition1 = Array.from({ length: depth }, () => createTowerTransitionBucket());
  const transition2 = Array.from({ length: depth }, () => new Map());
  const transition3 = Array.from({ length: depth }, () => new Map());
  const totals = createTowerScoreBucket();
  const recentBias = createTowerScoreBucket();

  rounds.slice(-60).forEach((round, reverseIndex) => {
    const weight = Math.pow(0.93, reverseIndex);
    const letters = String(round.sequence || '')
      .split('')
      .filter((letter) => letter === 'A' || letter === 'B' || letter === 'C');

    for (let i = 0; i < Math.min(letters.length, depth); i += 1) {
      const letter = letters[i];
      positionStats[i][letter] += weight;
      totals[letter] += weight;
      recentBias[letter] += weight;

      if (i >= 1) {
        const prev1 = letters[i - 1];
        transition1[i][prev1][letter] += weight;
      }
      if (i >= 2) {
        const key2 = `${letters[i - 2]}${letters[i - 1]}`;
        getTowerMapBucket(transition2[i], key2)[letter] += weight;
      }
      if (i >= 3) {
        const key3 = `${letters[i - 3]}${letters[i - 2]}${letters[i - 1]}`;
        getTowerMapBucket(transition3[i], key3)[letter] += weight;
      }
    }
  });

  return { positionStats, transition1, transition2, transition3, totals, recentBias };
}

function applyTowerPatternPenalty(scores, prefix) {
  const tail = prefix.slice(-4);
  if (tail.length < 4) return;
  const [a, b, c, d] = tail;
  if (a === b && b === c) scores[a] *= 0.74;
  if (b === c && c === d) scores[b] *= 0.74;
  if (a === c && b === d && a !== b) {
    scores[a] *= 0.9;
    scores[b] *= 0.9;
  }
  if (a === b && c === d && a !== c) {
    scores[a] *= 0.92;
    scores[c] *= 0.92;
  }
}

function scoreTowerPosition(model, prefix, position) {
  const scores = createTowerScoreBucket();
  const positionWeight = position < 3 ? 1.08 : position < 7 ? 1 : 0.92;

  for (const letter of ['A', 'B', 'C']) {
    scores[letter] += (model.positionStats[position]?.[letter] || 0) * positionWeight;
    scores[letter] += (model.totals[letter] || 0) * (position === 0 ? 0.06 : 0.02);
    scores[letter] += (model.recentBias[letter] || 0) * (position === 0 ? 0.18 : 0.04);
  }

  if (prefix.length >= 1) {
    const prev1 = prefix[prefix.length - 1];
    for (const letter of ['A', 'B', 'C']) {
      scores[letter] += (model.transition1[position]?.[prev1]?.[letter] || 0) * 0.58;
    }
  }

  if (prefix.length >= 2) {
    const key2 = prefix.slice(-2);
    const bucket2 = model.transition2[position]?.get(key2);
    if (bucket2) {
      for (const letter of ['A', 'B', 'C']) {
        scores[letter] += (bucket2[letter] || 0) * 0.47;
      }
    }
  }

  if (prefix.length >= 3) {
    const key3 = prefix.slice(-3);
    const bucket3 = model.transition3[position]?.get(key3);
    if (bucket3) {
      for (const letter of ['A', 'B', 'C']) {
        scores[letter] += (bucket3[letter] || 0) * 0.38;
      }
    }
  }

  applyTowerPatternPenalty(scores, prefix);

  const ordered = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [bestLetter, bestScore] = ordered[0];
  const secondScore = ordered[1]?.[1] || 0;
  const confidence = bestScore > 0 ? clamp(((bestScore - secondScore) / bestScore) * 100, 0, 100) : 0;

  return { letter: bestLetter || 'A', confidence };
}

function predictTowerSequence(rounds, horizon = 10) {
  if (!rounds.length) {
    return { forecast: '', confidence: 0 };
  }

  const model = trainTowerModel(rounds);
  const forecast = [];
  const stepConfidence = [];

  for (let i = 0; i < horizon; i += 1) {
    const { letter, confidence } = scoreTowerPosition(model, forecast, i);
    forecast.push(letter);
    stepConfidence.push(confidence);
  }

  const avgConfidence = stepConfidence.length
    ? stepConfidence.reduce((sum, value) => sum + value, 0) / stepConfidence.length
    : 0;

  return {
    forecast: forecast.join(''),
    confidence: clamp(avgConfidence, 0, 100),
  };
}

function evaluateTowerWindow(rounds, windowSize) {
  const usable = normalizeTowerRounds(rounds);
  if (usable.length < 8) {
    return { score: 0, exactRate: 0, partialRate: 0, samples: 0 };
  }

  const start = Math.max(5, usable.length - 18);
  let exactWins = 0;
  let partialMatches = 0;
  let samples = 0;

  for (let i = start; i < usable.length; i += 1) {
    const training = usable.slice(Math.max(0, i - windowSize), i);
    if (training.length < 5) continue;

    const predicted = predictTowerSequence(training, 4).forecast.slice(0, 4);
    const actual = usable[i].sequence.slice(0, 4);
    const matches = predicted.split('').reduce((count, letter, idx) => count + (letter === actual[idx] ? 1 : 0), 0);

    exactWins += matches === 4 ? 1 : 0;
    partialMatches += matches / 4;
    samples += 1;
  }

  if (!samples) {
    return { score: 0, exactRate: 0, partialRate: 0, samples: 0 };
  }

  const exactRate = exactWins / samples;
  const partialRate = partialMatches / samples;
  return {
    score: (partialRate * 0.7) + (exactRate * 0.3),
    exactRate,
    partialRate,
    samples,
  };
}

function buildTowerForecast(rounds) {
  const usableRounds = normalizeTowerRounds(rounds);
  if (!usableRounds.length) return '';

  const candidates = [18, 24, 36, 48, 60, 80].map((windowSize) => {
    const training = usableRounds.slice(-windowSize);
    const forecastInfo = predictTowerSequence(training, 10);
    const scoreInfo = evaluateTowerWindow(usableRounds, windowSize);
    return {
      windowSize,
      ...forecastInfo,
      ...scoreInfo,
      combinedScore: (scoreInfo.score * 100) + (forecastInfo.confidence * 0.35),
    };
  });

  const best = candidates.sort((a, b) => {
    if (b.combinedScore !== a.combinedScore) return b.combinedScore - a.combinedScore;
    if (b.samples !== a.samples) return b.samples - a.samples;
    return b.windowSize - a.windowSize;
  })[0];

  return best?.forecast || '';
}

function buildTowerPredictionHistory(rounds) {
  const usableRounds = normalizeTowerRounds(rounds)
    .sort((a, b) => Number(a.roundId) - Number(b.roundId));

  const rows = [];
  for (let i = 5; i < usableRounds.length; i += 1) {
    const trainingRounds = usableRounds.slice(0, i);
    const forecast = buildTowerForecast(trainingRounds).slice(0, 4);
    const actual = usableRounds[i].sequence.slice(0, 4);
    const matches = forecast.split('').reduce((count, letter, idx) => count + (letter === actual[idx] ? 1 : 0), 0);
    rows.push({
      roundId: usableRounds[i].roundId,
      forecast,
      actual,
      matches,
      accuracy: Math.round((matches / 4) * 100),
      outcome: matches === 4 ? 'WIN' : 'LOSS',
    });
  }
  return rows;
}

function scoreCandidate(url, baseOrigin) {
  if (!url) return -Infinity;
  const lower = url.toLowerCase();
  let score = 0;
  if (lower.includes(baseOrigin)) score += 6;
  if (/rounds?/.test(lower)) score += 6;
  if (/history/.test(lower)) score += 5;
  if (/crash/.test(lower)) score += 4;
  if (/game/.test(lower)) score += 3;
  if (/ws|socket/.test(lower)) score += 3;
  if (/api/.test(lower)) score += 4;
  if (/dealer/.test(lower)) score += 4;
  if (/live/.test(lower)) score += 2;
  if (/recent/.test(lower)) score += 2;
  if (/\?limit=\d+/.test(lower)) score += 1;
  if (/cdn|google|sentry|fontawesome|analytics|facebook|twitter|telegram|discord/.test(lower)) score -= 8;
  if (/\.(js|css|png|jpg|jpeg|gif|webp|svg)(\?|$)/.test(lower)) score -= 10;
  if (!/(api|round|history|ws|socket|game|dealer|crash)/.test(lower)) score -= 3;
  return score;
}

async function fetchText(url, timeoutMs = POLL_TIMEOUT_MS, headers = {}) {
  const res = await withTimeout(
    fetch(url, {
      headers: {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36',
        accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        ...headers,
      },
    }),
    timeoutMs,
    `fetch:${url}`
  );
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} while fetching ${url}`);
  }
  return res.text();
}

async function fetchJson(url, timeoutMs = POLL_TIMEOUT_MS, headers = {}) {
  const res = await withTimeout(
    fetch(url, {
      headers: {
        accept: 'application/json,text/plain,*/*',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36',
        ...headers,
      },
    }),
    timeoutMs,
    `fetch:${url}`
  );
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} while fetching ${url}`);
  }
  return res.json();
}

function getFetchHeadersForUrl(url) {
  const lower = String(url || '').toLowerCase();
  if (lower.includes('degencoinflip.com')) {
    return {
      origin: 'https://app.degencoinflip.com',
      referer: 'https://app.degencoinflip.com/crash',
    };
  }
  if (lower.includes('solanafatboys.com') || lower.includes('api.solanafatboys.com')) {
    return {
      origin: 'https://www.solanafatboys.com',
      referer: 'https://www.solanafatboys.com/bustonaut/',
    };
  }
  return {};
}

async function discoverApiUrl(gameUrl) {
  const pageUrl = normalizeUrl(gameUrl);
  if (!pageUrl) return null;

  const html = await fetchText(pageUrl, DISCOVERY_TIMEOUT_MS);
  if (looksLikeChallengePage(html)) return null;

  const origin = new URL(pageUrl).origin;
  const scriptMatches = [...html.matchAll(/<script[^>]+src=["']([^"']+)["']/gi)].map((match) => match[1]);
  const inlineCandidates = [...html.matchAll(/https?:\/\/[^"'\s<>]+|\/[A-Za-z0-9_./?=&%:-]+/g)].map((match) => match[0]);
  const bundleUrls = [];
  for (const src of scriptMatches) {
    const abs = resolveUrl(pageUrl, src);
    if (abs && !bundleUrls.includes(abs)) bundleUrls.push(abs);
  }

  const snippets = [html];
  for (const bundleUrl of bundleUrls.slice(0, DISCOVERY_BUNDLE_LIMIT)) {
    try {
      const bundleText = await fetchText(bundleUrl, DISCOVERY_TIMEOUT_MS);
      snippets.push(bundleText.slice(0, DISCOVERY_TEXT_LIMIT));
    } catch {
      // Discovery is best-effort; keep probing other bundles.
    }
  }

  const candidates = new Map();
  const seen = new Set();
  const addCandidate = (candidate, source) => {
    const clean = String(candidate || '').trim();
    if (!clean || seen.has(clean)) return;
    seen.add(clean);
    candidates.set(clean, {
      url: clean,
      source,
      score: scoreCandidate(clean, origin),
    });
  };

  for (const snippet of snippets) {
    for (const match of snippet.matchAll(/https?:\/\/[^"'`\s<>]+/g)) addCandidate(match[0], 'absolute');
    for (const match of snippet.matchAll(/['"]((?:\/|\.\/)[^"'`<>]+)['"]/g)) {
      const abs = resolveUrl(pageUrl, match[1]);
      if (abs) addCandidate(abs, 'relative');
    }
  }

  const ordered = [...candidates.values()]
    .filter((candidate) => candidate.score > 0)
    .sort((a, b) => b.score - a.score);

  for (const candidate of ordered) {
    try {
      const payload = await fetchJson(candidate.url, DISCOVERY_TIMEOUT_MS, getFetchHeadersForUrl(candidate.url));
      const rounds = extractRoundArray(payload).map(normalizeRoundRow).filter(Boolean);
      if (rounds.length) {
        return {
          apiUrl: candidate.url,
          roundsPath: null,
          rounds,
        };
      }
    } catch {
      // Keep trying the next candidate.
    }
  }

  return null;
}

async function parseRoundsFromApi(apiUrl) {
  const payload = await fetchJson(apiUrl, POLL_TIMEOUT_MS, getFetchHeadersForUrl(apiUrl));
  const rounds = extractRoundArray(payload).map(normalizeRoundRow).filter(Boolean);
  if (!rounds.length) {
    throw new Error(`No rounds found at ${apiUrl}`);
  }
  return rounds;
}

async function ensureSeedData() {
  const existingSites = await getCrashSites();
  const existingKeys = new Set(existingSites.map((site) => site.sourceKey));
  const existingGameKeys = new Map(
    existingSites.map((site) => [normalizeGameKey(site.gameUrl), site])
  );
  for (const site of existingSites) {
    const gameUrl = String(site.gameUrl || '').toLowerCase();
    const shouldDelete = DEPRECATED_CRASH_SOURCE_KEYS.has(site.sourceKey)
      || DEPRECATED_CRASH_URL_PATTERNS.some((pattern) => gameUrl.includes(pattern));
    if (shouldDelete) {
      await deleteCrashSite(site.id);
    }
  }
  for (const site of DEFAULT_CRASH_SOURCES) {
    const gameKey = normalizeGameKey(site.gameUrl);
    const existingByGame = existingGameKeys.get(gameKey);
    if (existingByGame) {
      const needsFix = (
        (site.apiUrl && normalizeUrl(existingByGame.apiUrl) !== normalizeUrl(site.apiUrl)) ||
        (site.roundsPath && String(existingByGame.roundsPath || '') !== String(site.roundsPath || '')) ||
        (site.adapter && existingByGame.adapter !== site.adapter)
      );
      if (needsFix) {
        await upsertCrashSite({
          sourceKey: existingByGame.sourceKey,
          label: existingByGame.label || site.label,
          gameUrl: existingByGame.gameUrl || site.gameUrl,
          adapter: site.adapter || existingByGame.adapter,
          apiUrl: existingByGame.apiUrl || site.apiUrl || null,
          roundsPath: existingByGame.roundsPath || site.roundsPath || null,
          enabled: existingByGame.enabled,
          pollIntervalMs: existingByGame.pollIntervalMs || site.pollIntervalMs || 30000,
        });
      }
      continue;
    }
    if (existingKeys.has(site.sourceKey)) continue;
    await upsertCrashSite({
      sourceKey: site.sourceKey,
      label: site.label,
      gameUrl: site.gameUrl,
      adapter: site.adapter,
      apiUrl: site.apiUrl || null,
      roundsPath: site.roundsPath || null,
      enabled: true,
      pollIntervalMs: site.pollIntervalMs || 30000,
    });
  }
}

async function refreshSiteConfig(site) {
  if (!site) return site;
  if (site.adapter === 'degencoinflip' && !site.apiUrl) {
    return upsertCrashSite({
      sourceKey: site.sourceKey,
      label: site.label,
      gameUrl: site.gameUrl,
      adapter: site.adapter,
      apiUrl: 'https://api.dealer.degencoinflip.com/v1/game/2/room/1/rounds?limit=100',
      roundsPath: '?limit=100',
      enabled: site.enabled,
      pollIntervalMs: site.pollIntervalMs,
    });
  }

  if (site.adapter === 'sfb' && !site.apiUrl) {
    return upsertCrashSite({
      sourceKey: site.sourceKey,
      label: site.label,
      gameUrl: site.gameUrl,
      adapter: site.adapter,
      apiUrl: 'https://sfb-api-service-mainnet.up.railway.app/api/games/bustonaut/latest?page=0&limit=100',
      roundsPath: '?page=0&limit=100',
      enabled: site.enabled,
      pollIntervalMs: site.pollIntervalMs,
    });
  }

  if (site.adapter === 'degencoinflip' && String(site.gameUrl || '').toLowerCase().includes('/towers') && !site.apiUrl) {
    return upsertCrashSite({
      sourceKey: site.sourceKey,
      label: site.label,
      gameUrl: site.gameUrl,
      adapter: site.adapter,
      apiUrl: 'https://api.dealer.degencoinflip.com/v1/game/3/room/1/rounds?limit=100',
      roundsPath: '?limit=100',
      enabled: site.enabled,
      pollIntervalMs: site.pollIntervalMs,
    });
  }

  if ((site.adapter === 'auto' || site.adapter === 'detected') && !site.apiUrl) {
    const discovered = await discoverApiUrl(site.gameUrl);
    if (discovered?.apiUrl) {
      return upsertCrashSite({
        sourceKey: site.sourceKey,
        label: site.label,
        gameUrl: site.gameUrl,
        adapter: 'detected',
        apiUrl: discovered.apiUrl,
        roundsPath: discovered.roundsPath || null,
        enabled: site.enabled,
        pollIntervalMs: site.pollIntervalMs,
      });
    }
  }

  return site;
}

async function pollSite(site) {
  if (!site?.enabled) return { saved: 0, skipped: true };

  const activeSite = await refreshSiteConfig(site);
  const currentSite = activeSite?.id ? activeSite : site;

  if (!currentSite.apiUrl) {
    await touchCrashSite(currentSite.id, {
      lastPolledAt: new Date(),
      lastError: 'No public history feed discovered yet',
    });
    return { saved: 0, skipped: true };
  }

  try {
    const rounds = await parseRoundsFromApi(currentSite.apiUrl);
    const saved = await saveCrashRounds(currentSite, rounds);
    if (String(currentSite.gameUrl || '').toLowerCase().includes('/towers')) {
      const historyRows = buildTowerPredictionHistory(rounds);
      if (historyRows.length) {
        await saveTowerPredictionHistory(currentSite, historyRows);
      }
    }
    await touchCrashSite(currentSite.id, {
      lastPolledAt: new Date(),
      lastSuccessAt: new Date(),
      lastError: null,
      lastRoundId: rounds.reduce((max, round) => Math.max(max, round.roundId), currentSite.lastRoundId || 0),
    });
    return { saved, rounds: rounds.length };
  } catch (error) {
    await touchCrashSite(currentSite.id, {
      lastPolledAt: new Date(),
      lastError: error.message,
    });
    return { saved: 0, error: error.message };
  }
}

async function pollAllSites() {
  if (inFlight) return;
  inFlight = true;
  try {
    const sites = await getCrashSites();
    const dueSites = sites.filter((site) => site.enabled);
    if (!dueSites.length) return;
    await Promise.allSettled(dueSites.map((site) => pollSite(site)));
  } finally {
    inFlight = false;
  }
}

async function startCrashWatchCollector() {
  if (collectorStarted) return;
  collectorStarted = true;

  await ensureSeedData();
  await pollAllSites();

  const loop = async () => {
    try {
      await pollAllSites();
    } catch (error) {
      console.error('[crash-watch] poll loop error:', error.message);
    } finally {
      tickTimer = setTimeout(loop, POLL_LOOP_MS);
    }
  };

  tickTimer = setTimeout(loop, POLL_LOOP_MS);
  console.log(`[crash-watch] collector running every ${POLL_LOOP_MS}ms`);
}

module.exports = {
  DEFAULT_CRASH_SOURCES,
  startCrashWatchCollector,
  pollAllSites,
  discoverApiUrl,
  parseRoundsFromApi,
  buildTowerPredictionHistory,
};
