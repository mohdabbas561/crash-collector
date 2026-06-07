const { initDB, pingDB } = require('./db');
const { startCollector } = require('./collector');
const { startCrashWatchCollector } = require('./crashCollector');
const { startAPI, setDatabaseAvailability } = require('./api');

const DB_RETRY_MS = Number.parseInt(process.env.DB_RETRY_MS || '60000', 10);

let collectorStarted = false;
let crashWatchStarted = false;
let dbReady = false;

async function tryInitDatabase() {
  try {
    await initDB();
    dbReady = true;
    setDatabaseAvailability(true);
    console.log('Database ready');
    return true;
  } catch (err) {
    dbReady = false;
    setDatabaseAvailability(false, err.message);
    console.error(`Database init failed: ${err.message}`);
    return false;
  }
}

function startCollectorOnce() {
  if (collectorStarted) return;
  startCollector();
  collectorStarted = true;
  console.log('Collector started');
}

function startCrashWatchOnce() {
  if (crashWatchStarted) return;
  startCrashWatchCollector().catch((err) => {
    console.error('[crash-watch] startup error:', err.message);
  });
  crashWatchStarted = true;
  console.log('Crash watch collector started');
}

function startDbRecoveryLoop() {
  setInterval(async () => {
    if (dbReady) {
      try {
        await pingDB();
      } catch (err) {
        dbReady = false;
        setDatabaseAvailability(false, err.message);
        console.error(`[db-health] Database went offline: ${err.message}`);
      }
      return;
    }

    console.log('[db-recovery] Retrying database init...');
    const recovered = await tryInitDatabase();
    if (recovered) {
      console.log('[db-recovery] Database back online');
      startCollectorOnce();
      startCrashWatchOnce();
    }
  }, Math.max(10000, DB_RETRY_MS));
}

async function main() {
  console.log('Crash Collector starting...');
  const ready = await tryInitDatabase();
  if (ready) {
    startCollectorOnce();
    startCrashWatchOnce();
  } else {
    console.warn('Starting API in degraded mode (database unavailable)');
  }
  startAPI();
  console.log('API started');
  startDbRecoveryLoop();
}

main().catch((err) => {
  console.error('Unhandled startup error (service kept alive):', err);
});
