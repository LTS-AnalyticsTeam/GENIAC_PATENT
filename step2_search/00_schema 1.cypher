// 既存のベクトルindexを消す（名前は作った時のもの）
DROP INDEX topicEmbeddingIdx IF EXISTS;
DROP INDEX sectionEmbeddingIdx IF EXISTS;

// 1536 次元（text-embedding-3-small）
CREATE VECTOR INDEX topicEmbeddingIdx IF NOT EXISTS
FOR (t:Topic) ON (t.embedding)
OPTIONS {indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: "cosine" }};

CREATE VECTOR INDEX sectionEmbeddingIdx IF NOT EXISTS
FOR (s:Section) ON (s.embedding)
OPTIONS {indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: "cosine" }};
