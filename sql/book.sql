-- Check for available extensions related to vectors
SELECT * 
FROM pg_available_extensions 
WHERE "name" ILIKE '%vector%';

-- List installed extensions
SELECT * 
FROM pg_extension;

-- Install the pgvector extension if not already installed
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop the 'book' table if it exists
DROP TABLE IF EXISTS book;

-- Create the 'book' table
CREATE TABLE IF NOT EXISTS book (
    id SERIAL PRIMARY KEY,          -- Auto-incrementing primary key
    metadata JSONB NOT NULL,        -- Metadata stored as JSONB
    embedding VECTOR(384),          -- Vector column with a dimension of 384
    content TEXT                    -- Content stored as text
);

-- Create an index on the 'embedding' column using ivfflat
CREATE INDEX IF NOT EXISTS embedding_idx 
ON book USING ivfflat (embedding) 
WITH (lists = 100);