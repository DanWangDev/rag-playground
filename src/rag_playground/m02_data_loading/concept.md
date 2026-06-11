# Module 02: Data Loading

## What You'll Learn

- What a **Document** abstraction is and why RAG systems need one
- How to load **plain text** and **markdown** files into the document format
- How to traverse **directories** and load all documents within
- What **metadata** is and why it matters for retrieval quality
- How **encoding detection** works (and why UTF-8 isn't always enough)

## Why Data Loading Matters

Before RAG can answer questions about your documents, it needs to _read_ them. This sounds simple — `open("file.txt").read()` — but real-world documents come in many formats, encodings, and structures. A robust RAG system needs a consistent way to:

1. **Read** files of different types (`.txt`, `.md`, `.pdf`, etc.)
2. **Extract metadata** — what is this document? Where did it come from? When was it created?
3. **Normalize** content into a standard `Document` shape that downstream components (chunkers, embedders) can consume

The `Document` abstraction is the universal currency of a RAG pipeline. Every component after this module — chunking, embedding, storing, retrieving — works with `Document` objects.

## The Document Abstraction

A Document has exactly two things:

```
Document
├── page_content: str          ← The actual text
└── metadata: dict             ← Everything else (source, title, date, etc.)
```

This mirrors [LangChain's Document](https://docs.langchain.com/) interface — intentionally. Understanding this abstraction means you can plug into any LangChain-compatible pipeline later. But we're building it ourselves so you understand every line.

### Why a dict for metadata?

Metadata is flexible. Different file types carry different information:

| File Type | Typical Metadata |
|-----------|-----------------|
| `.txt` | filename, path, file size, modified date |
| `.md` | filename, title (from heading), frontmatter fields (author, date, tags) |
| Directory | relative path, parent directory, file count |

A dict lets each loader add its own metadata without changing the `Document` class.

### Why metadata matters for retrieval

When you search a vector store, you often want to filter by metadata:

- "Only show results from documents in the `faq/` directory"
- "Show me documents by this author"
- "Limit results to documents from the last 30 days"

Metadata filtering is covered in Module 05 (Vector Store) and Module 06 (Retrieval). Module 02 focuses on _capturing_ metadata correctly.

## How Data Loading Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  File on     │────▶│  Loader      │────▶│  Document    │
│  Disk        │     │  reads file  │     │  object      │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Step 1: Open the file
Python's built-in `open()` handles most cases. We detect encoding by trying UTF-8 first (covers 95%+ of modern documents), then falling back to latin-1.

### Step 2: Read content
For plain text, the entire file is `page_content`. For markdown, we parse frontmatter (YAML between `---` delimiters) separately from the body.

### Step 3: Extract metadata
Each loader creates a metadata dict:
- `source`: the file path (relative to the data directory)
- `filename`: just the filename
- `file_type`: `.txt`, `.md`, etc.
- Loader-specific fields (frontmatter for markdown, directory info for batch loading)

### Step 4: Return a Document
The loader returns a `Document` object with populated `page_content` and `metadata`.

## Key Concepts

### Encoding

Text files aren't all UTF-8. Legacy documents might use:
- **latin-1** (ISO-8859-1): Common in older Windows documents
- **cp1252**: Windows default encoding in English-language systems
- **UTF-16**: Sometimes used by Windows tools

Our loader tries UTF-8 first, falls back to latin-1 with a warning.

### Frontmatter

Markdown files often have YAML frontmatter:

```markdown
---
title: "My Document"
author: "Jane Doe"
date: 2024-01-15
tags: [tutorial, python]
---

# My Document

Content starts here...
```

The loader splits on `---`, parses the YAML block as metadata, and keeps the rest as `page_content`.

### Directory Loading

Loading entire directories is recursive by default:

```
data/
├── wikipedia/
│   ├── ml.md          ← loaded
│   └── climate.md     ← loaded
└── faq/
    ├── dynamodb.md    ← loaded
    └── typescript.md  ← loaded
```

You can filter by glob pattern (`**/*.md`) or file extension.

## Gotchas

1. **Large files**: Loading a 50MB file into memory works but downstream chunking and embedding will be slow. Consider splitting large files first.
2. **Binary files**: Our loaders only handle text. PDFs, images, and other binary formats need specialized loaders (future extension).
3. **Encoding errors**: If a file has mixed encodings, our fallback to latin-1 will produce garbled characters for non-latin-1 code points.
4. **Metadata conflicts**: When loading from a directory, metadata from the file (frontmatter) takes precedence over metadata from the directory (path).

## What You'll Practice

In the exercise, you will:
1. Load a plain text file and inspect its Document
2. Load a markdown file with frontmatter
3. Load an entire directory of documents
4. Filter documents by metadata
5. Prepare documents for the next module: chunking
