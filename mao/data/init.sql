-- MAO Postgres schema
-- Run automatically by docker-compose on first startup

CREATE TABLE IF NOT EXISTS wikipedia_articles (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    content     TEXT,
    category    TEXT,
    word_count  INTEGER,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS squad_questions (
    id          SERIAL PRIMARY KEY,
    question    TEXT NOT NULL,
    answer      TEXT,
    context_id  INTEGER REFERENCES wikipedia_articles(id),
    score       FLOAT
);

CREATE INDEX IF NOT EXISTS idx_wiki_category ON wikipedia_articles(category);
CREATE INDEX IF NOT EXISTS idx_wiki_word_count ON wikipedia_articles(word_count);
CREATE INDEX IF NOT EXISTS idx_squad_score ON squad_questions(score);

-- RAGAS response quality metrics (written by mao/eval/ragas_evaluator.py)
CREATE TABLE IF NOT EXISTS response_metrics (
    id                 SERIAL PRIMARY KEY,
    request_id         TEXT NOT NULL,
    user_id            TEXT,
    agent_used         TEXT,
    faithfulness       FLOAT,
    answer_relevancy   FLOAT,
    context_precision  FLOAT,
    context_recall     FLOAT,
    latency_ms         FLOAT,
    created_at         TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_response_metrics_agent_time
    ON response_metrics (agent_used, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_response_metrics_request
    ON response_metrics (request_id);

-- Sample data for SQL agent testing
INSERT INTO wikipedia_articles (title, category, word_count) VALUES
  ('Machine learning',                 'AI',        15420),
  ('Natural language processing',      'AI',        12300),
  ('Knowledge graph',                  'Databases',  8900),
  ('Transformer (deep learning)',      'AI',        11200),
  ('BERT (language model)',            'AI',         9800),
  ('Retrieval-augmented generation',   'AI',         4500),
  ('Graph database',                   'Databases',  7200),
  ('Attention mechanism',              'AI',         6100),
  ('Large language model',             'AI',        18000),
  ('Semantic search',                  'IR',         5600)
ON CONFLICT DO NOTHING;
