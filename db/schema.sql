CREATE TABLE IF NOT EXISTS trade (
  id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  open_date TEXT NOT NULL,
  close_date TEXT,
  status TEXT CHECK(status IN ('open','closed','unmatched')) NOT NULL,
  quantity REAL NOT NULL,
  avg_entry_price REAL,
  avg_exit_price REAL,
  realized_pnl REAL
);

CREATE TABLE IF NOT EXISTS leg (
  id TEXT PRIMARY KEY,
  trade_id TEXT REFERENCES trade(id),
  type TEXT CHECK(type IN ('buy','sell','dividend')) NOT NULL,
  date TEXT NOT NULL,
  price REAL NOT NULL,
  quantity REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS timing_score (
  leg_id TEXT REFERENCES leg(id),
  window_days INTEGER NOT NULL,
  local_low REAL,
  local_high REAL,
  percentile REAL,
  verdict TEXT,
  PRIMARY KEY (leg_id, window_days)
);

CREATE TABLE IF NOT EXISTS pattern (
  id TEXT PRIMARY KEY,
  description TEXT NOT NULL,
  evidence_trade_ids TEXT NOT NULL,   -- JSON array
  confidence REAL NOT NULL,
  first_seen TEXT NOT NULL,
  last_updated TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS news_context (
  id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  date TEXT NOT NULL,
  source TEXT,
  summary TEXT,
  pinecone_vector_id TEXT
);
