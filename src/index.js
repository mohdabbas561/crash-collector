const { initDB, initAccessCodes } = require('./db');
const { startCollector } = require('./collector');
const { startAPI } = require('./api');

async function main() {
  console.log('🚀 Crash Collector starting...');
  await initDB();
  await initAccessCodes();
  console.log('✅ Database ready');
  startCollector(); // collector calls runPredictionEngine after every poll
  console.log('✅ Collector started');
  startAPI();
  console.log('✅ API started');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});