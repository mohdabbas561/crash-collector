// Load a local .env without requiring an extra package. Existing environment
// variables take precedence (as they do on Railway and other hosts).
const fs = require('fs');
const path = require('path');
const envFile = path.resolve(__dirname, '..', '.env');
if (fs.existsSync(envFile)) {
  for (const line of fs.readFileSync(envFile, 'utf8').split(/\r?\n/)) {
    const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
    if (!match || Object.prototype.hasOwnProperty.call(process.env, match[1])) continue;
    let value = match[2];
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    process.env[match[1]] = value;
  }
}

const { initDB, initAccessCodes } = require('./db');
const { startCollector } = require('./collector');
const { startAPI } = require('./api');

async function main() {
  console.log('🚀 Crash Collector starting...');
  await initDB();
  await initAccessCodes();
  console.log('✅ Database ready');
  startCollector();
  console.log('✅ Collector started');
  startAPI();
  console.log('✅ API started');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
