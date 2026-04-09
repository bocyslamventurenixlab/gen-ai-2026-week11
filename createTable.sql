-- Reading App: Full Database Schema
-- Compatible with PostgreSQL + pgvector (Supabase)

-- 1. Enable the pgvector extension to work with embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Users Table: Manages user profiles and tiers
CREATE TABLE IF NOT EXISTS users (
id SERIAL PRIMARY KEY,
email TEXT UNIQUE NOT NULL,
tier TEXT DEFAULT 'free' CHECK (tier IN ('free', 'pro')),
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Documents Table: Stores metadata for uploaded files
CREATE TABLE IF NOT EXISTS documents (
id SERIAL PRIMARY KEY,
user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
title TEXT NOT NULL,
upload_date TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Embeddings Table: Stores the actual text chunks and their vectors
-- Note: Dimensions are set to 1536 to match 'text-embedding-3-small'
CREATE TABLE IF NOT EXISTS embeddings (
id SERIAL PRIMARY KEY,
doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
content TEXT NOT NULL,
embedding vector(1536)
);

-- 5. Security Logs: Tracks potential prompt injection attempts
CREATE TABLE IF NOT EXISTS security_logs (
id SERIAL PRIMARY KEY,
query TEXT NOT NULL,
type TEXT NOT NULL, -- e.g., 'injection', 'malicious_intent'
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. Vector Similarity Search Function
-- This function is called by the Librarian Agent to find the most relevant context
CREATE OR REPLACE FUNCTION match_embeddings (
query_embedding vector(1536),
match_threshold float,
match_count int,
filter_doc_id int
)
RETURNS TABLE (
id bigint,
content text,
similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
RETURN QUERY
SELECT
embeddings.id,
embeddings.content,
1 - (embeddings.embedding <=> query_embedding) AS similarity
FROM embeddings
WHERE embeddings.doc_id = filter_doc_id
AND 1 - (embeddings.embedding <=> query_embedding) > match_threshold
ORDER BY similarity DESC
LIMIT match_count;
END;
$$;

-- 7. Performance Indexing
-- Optional: Adds an IVFFlat index for faster vector searching at scale
-- CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);