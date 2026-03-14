const express = require('express');
const cors    = require('cors');
const { getRounds, getStats, getStorageStats } = require('./db');

const app  = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

app.get('/rounds', async (req, res) => {
  try {
    const limit  = Math.min(parseInt(req.query.limit  || '1000'), 10000);
    const offset = parseInt(req.query.offset || '0');
    const from   = req.query.from || null;
    const to     = req.query.to   || null;
    const rounds = await getRounds({ limit, offset, from, to });
    const since  = req.query.since ? Number(req.query.since) : 0;
    const result = since ? rounds.filter(r => r.roundId > since) : rounds;
    res.json({ ok: true, count: result.length, rounds: result });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/stats', async (req, res) => {
  try {
    const stats = await getStats();
    res.json({ ok: true, ...stats });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/storage-stats', async (req, res) => {
  try {
    const stats = await getStorageStats();
    res.json({ ok: true, ...stats });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.get('/health', (req, res) => {
  res.json({ ok: true, ts: new Date().toISOString() });
});

function startAPI() {
  app.listen(PORT, () => console.log(`API listening on port ${PORT}`));
}

module.exports = { startAPI };
