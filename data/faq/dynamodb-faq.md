# DynamoDB FAQ

## What is DynamoDB?
Amazon DynamoDB is a fully managed, serverless NoSQL database service that provides consistent single-digit millisecond performance at any scale. It is a key-value and document database.

## What is a partition key?
A partition key is the primary key component that determines how data is distributed across DynamoDB partitions. The partition key value is hashed internally to determine which physical storage partition the item will be stored on.

## What is a sort key?
A sort key (also called a range key) is an optional second component of a composite primary key. When present, items with the same partition key are stored together in sorted order by the sort key value, enabling efficient range queries.

## What is the difference between Query and Scan?
A Query operation finds items in a table using only the primary key — you must provide a partition key value. It is efficient and uses indexes. A Scan operation reads every item in the table and filters after the fact. Scans are expensive and should be avoided for frequent operations.

## What are Global Secondary Indexes (GSIs)?
A GSI is an index with a partition key and optional sort key that can be different from the table's primary key. GSIs enable additional query patterns. They are eventually consistent (unless you use strong read-after-write on the table), have their own provisioned capacity (or on-demand), and can be added or removed at any time.

## What is single-table design?
Single-table design is a DynamoDB pattern where multiple entity types are stored in a single table, using key overloading and composite primary keys. Instead of one table per entity (like a relational database), all related entities share one table, improving performance by eliminating joins and enabling efficient single-request access patterns.

## How does DynamoDB handle scaling?
DynamoDB scales horizontally by distributing data across partitions. Tables have no size limit — they can grow to any size. Throughput can be provisioned (you set read/write capacity units) or on-demand (DynamoDB automatically accommodates workload changes).

## What are DynamoDB Streams?
DynamoDB Streams captures a time-ordered sequence of item-level modifications (creates, updates, deletes) in a table. Each stream record contains the before and after state of the item. Streams enable event-driven architectures — Lambda functions can be triggered by stream records to process changes in real-time.
