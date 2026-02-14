"""OLAP star-schema DDL for local SQLite experiment tracking.

Tables:
  dim_experiment        – one row per optimization run (node start→stop for a domain)
  dim_domain            – domain dimension
  fact_round            – one row per optimization round (the main fact table)
  fact_experiment_summary – one row per completed experiment
  _schema_version       – migration tracking
"""

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Migration tracking
CREATE TABLE IF NOT EXISTS _schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL
);

-- Dimension: Domains
CREATE TABLE IF NOT EXISTS dim_domain (
    domain_id TEXT PRIMARY KEY,
    domain_type TEXT,
    description TEXT
);

-- Dimension: Experiments
CREATE TABLE IF NOT EXISTS dim_experiment (
    experiment_id TEXT PRIMARY KEY,
    domain_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    hostname TEXT,
    optimizer_plugin TEXT,
    optimization_config JSON,
    param_bounds JSON,
    target_performance REAL,
    started_at TEXT,
    finished_at TEXT,
    converged BOOLEAN DEFAULT FALSE,
    doin_version TEXT
);

-- Fact: Per-round metrics
CREATE TABLE IF NOT EXISTS fact_round (
    round_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES dim_experiment(experiment_id),
    domain_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    timestamp_utc TEXT NOT NULL,

    -- Performance
    performance REAL NOT NULL,
    best_performance REAL,
    performance_delta REAL,
    is_improvement BOOLEAN,

    -- Parameters
    parameters JSON,
    best_parameters JSON,

    -- Timing
    wall_clock_seconds REAL,
    elapsed_seconds REAL,
    time_to_current_best_seconds REAL,
    time_to_target_seconds REAL,

    -- Network context
    chain_height INTEGER DEFAULT 0,
    peers_count INTEGER DEFAULT 0,
    block_reward_earned REAL DEFAULT 0.0,

    -- Convergence
    converged BOOLEAN DEFAULT FALSE
);

-- Fact: Experiment summary
CREATE TABLE IF NOT EXISTS fact_experiment_summary (
    experiment_id TEXT PRIMARY KEY REFERENCES dim_experiment(experiment_id),
    domain_id TEXT NOT NULL,
    total_rounds INTEGER,
    final_performance REAL,
    best_performance REAL,
    time_to_target_seconds REAL,
    total_elapsed_seconds REAL,
    converged BOOLEAN,
    rounds_to_convergence INTEGER,
    node_id TEXT,
    hostname TEXT
);

-- Fact: On-chain accepted optimae with experiment metrics
CREATE TABLE IF NOT EXISTS fact_chain_optimae (
    optimae_id TEXT PRIMARY KEY,
    domain_id TEXT NOT NULL,
    optimizer_id TEXT NOT NULL,
    experiment_id TEXT,
    round_number INTEGER,
    parameters JSON,
    reported_performance REAL,
    verified_performance REAL,
    effective_increment REAL,
    time_to_this_result_seconds REAL,
    optimization_config_hash TEXT,
    data_hash TEXT,
    previous_best_performance REAL,
    reward_fraction REAL,
    quorum_agree_fraction REAL,
    block_height INTEGER NOT NULL,
    block_timestamp TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chain_optimae_domain ON fact_chain_optimae(domain_id);
CREATE INDEX IF NOT EXISTS idx_chain_optimae_experiment ON fact_chain_optimae(experiment_id);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_fact_round_experiment ON fact_round(experiment_id);
CREATE INDEX IF NOT EXISTS idx_fact_round_domain ON fact_round(domain_id);
CREATE INDEX IF NOT EXISTS idx_fact_round_timestamp ON fact_round(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_fact_round_performance ON fact_round(performance);
CREATE INDEX IF NOT EXISTS idx_dim_experiment_domain ON dim_experiment(domain_id);
CREATE INDEX IF NOT EXISTS idx_fact_summary_domain ON fact_experiment_summary(domain_id);
"""
