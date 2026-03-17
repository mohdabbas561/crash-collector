const { initDB, initAccessCodes } = require('./db');
const { startCollector } = require('./collector');
const { startAPI } = require('./api');
const { runPredictionEngine } = require('./predictionEngine');

const ENGINE_INTERVAL_MS  = 5000;  // poll every 5s
const ENGINE_ERROR_BACKOFF = 2000; // wait 2s after a crash before retrying

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function engineLoop() {
  let errors  = 0;
  let cycles  = 0;
  console.log('[engine] loop started (5s interval)');

  while (true) {
    const start = Date.now();
    try {
      await runPredictionEngine();
      errors = 0;
      cycles++;
      if (cycles % 60 === 0) {
        console.log(`[engine] heartbeat — ${cycles} cycles, uptime ${Math.round(process.uptime())}s`);
      }
    } catch (err) {
      errors++;
      console.error(`[engine] crash #${errors}:`, err.message);
      if (errors >= 5) {
        console.error('[engine] 5 consecutive errors — pausing 30s');
        await sleep(30000);
        errors = 0;
      } else {
        await sleep(ENGINE_ERROR_BACKOFF);
      }
      continue;
    }

    const elapsed = Date.now() - start;
    await sleep(Math.max(0, ENGINE_INTERVAL_MS - elapsed));
  }
}

async function main() {
  console.log('🚀 Crash Collector starting...');
  await initDB();
  await initAccessCodes();
  console.log('✅ Database ready');
  startCollector();
  console.log('✅ Collector started');
  startAPI();
  console.log('✅ API started');

  // Start prediction engine loop — runs forever, self-healing on crash
  engineLoop().catch(err => {
    console.error('[engine] FATAL loop exit:', err.message);
    // Restart after 5s if the loop itself somehow exits
    setTimeout(() => engineLoop().catch(console.error), 5000);
  });
  console.log('✅ Engine started');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});