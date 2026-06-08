'use strict';

const fetch = require('node-fetch');
const {
  getCrashSites,
  getCrashSiteById,
  upsertCrashSite,
  saveCrashRounds,
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
    apiUrl: 'https://api.solanafatboys.com/api/games/bustonaut/latest?page=0&limit=100',
    roundsPath: '?page=0&limit=100',
  },
  {
    sourceKey: 'solpump-crash',
    label: 'SolPump Crash',
    gameUrl: 'https://solpump.io/fairness/crash',
    adapter: 'solpump',
    apiUrl: 'https://solpump.io/api/v1/crash/live/history',
    roundsPath: '/api/v1/crash/live/history',
  },
];

const DEPRECATED_CRASH_SOURCE_KEYS = new Set([
  'bcgame-crash',
  'stake-crash',
  'solcrash-crash',
]);

const DEPRECATED_CRASH_URL_PATTERNS = [
  'bcgame52.com/game/crash',
  'stake.bet/casino/games/crash',
  'solcrash.io/play',
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
    const roundId = Number(roundIdRaw);
    const multiplier = parseFloatish(multiplierRaw);
    if (!Number.isFinite(roundId) || !Number.isFinite(multiplier)) return null;
    const timestamp = timestampRaw != null ? Number(timestampRaw) : Date.now();
    return {
      roundId,
      multiplier,
      timestamp: Number.isFinite(timestamp) ? timestamp : Date.now(),
      raw: row,
    };
  }

  if (!row || typeof row !== 'object') return null;
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
  if (!Number.isFinite(roundId) || !Number.isFinite(multiplier)) return null;
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
  return {
    roundId,
    multiplier,
    timestamp: Number.isFinite(timestamp) ? timestamp : Date.now(),
    raw: row,
  };
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
  if (lower.includes('solpump.io')) {
    return {
      origin: 'https://solpump.io',
      referer: 'https://solpump.io/fairness/crash',
      'x-requested-with': 'XMLHttpRequest',
    };
  }
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
      apiUrl: 'https://api.solanafatboys.com/api/games/bustonaut/latest?page=0&limit=100',
      roundsPath: '?page=0&limit=100',
      enabled: site.enabled,
      pollIntervalMs: site.pollIntervalMs,
    });
  }

  if (site.adapter === 'solpump' && !site.apiUrl) {
    return upsertCrashSite({
      sourceKey: site.sourceKey,
      label: site.label,
      gameUrl: site.gameUrl,
      adapter: site.adapter,
      apiUrl: 'https://solpump.io/api/v1/crash/live/history',
      roundsPath: '/api/v1/crash/live/history',
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
  discoverApiUrl,
  parseRoundsFromApi,
};
