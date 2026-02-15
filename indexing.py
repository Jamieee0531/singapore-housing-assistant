"""
Document Indexing Script for Singapore Housing Assistant RAG System.

This script processes Markdown documents and creates a searchable knowledge base:
1. Reads MD files from docs/ directory
2. Splits into parent chunks (2000-10000 chars) and child chunks (500 chars)
3. Stores parent chunks as JSON files in parent_store/
4. Indexes child chunks into Qdrant vector database

Usage:
    python indexing.py              # Check if index exists, fail if so (safe default)
    python indexing.py --rebuild    # Delete existing index and rebuild from scratch
    python indexing.py --append     # Add new documents without deleting existing data
"""

import os
import argparse
import glob
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import (
    DOCS_DIR,
    MARKDOWN_DIR,
    PARENT_STORE_PATH,
    QDRANT_DB_PATH,
    CHILD_COLLECTION,
    DENSE_MODEL,
    SPARSE_MODEL,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    MIN_PARENT_SIZE,
    MAX_PARENT_SIZE,
    HEADERS_TO_SPLIT_ON
)
from src.db import ParentStoreManager


def ensure_collection(client, collection_name, embedding_dimension):
    """
    Create Qdrant collection if it doesn't exist.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        embedding_dimension: Dimension of dense embeddings
    """
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=embedding_dimension,
                distance=qmodels.Distance.COSINE
            ),
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams()
            },
        )
        print(f"✓ Created collection: {collection_name}")
    else:
        print(f"✓ Collection already exists: {collection_name}")


def merge_small_parents(chunks, min_size):
    """
    Merge consecutive parent chunks that are smaller than min_size.
    
    Args:
        chunks: List of document chunks
        min_size: Minimum size threshold in characters
        
    Returns:
        List of merged chunks
    """
    if not chunks:
        return []
    
    merged, current = [], None
    
    for chunk in chunks:
        if current is None:
            current = chunk
        else:
            current.page_content += "\n\n" + chunk.page_content
            # Keep first chunk's metadata (contains correct header hierarchy)
            # Only add keys that don't already exist
            for k, v in chunk.metadata.items():
                if k not in current.metadata:
                    current.metadata[k] = v

        if len(current.page_content) >= min_size:
            merged.append(current)
            current = None

    if current:
        if merged:
            merged[-1].page_content += "\n\n" + current.page_content
            for k, v in current.metadata.items():
                if k not in merged[-1].metadata:
                    merged[-1].metadata[k] = v
        else:
            merged.append(current)
    
    return merged


def split_large_parents(chunks, max_size, child_splitter):
    """
    Split parent chunks that are larger than max_size.
    
    Args:
        chunks: List of document chunks
        max_size: Maximum size threshold in characters
        child_splitter: Splitter instance for breaking large chunks
        
    Returns:
        List of split chunks
    """
    split_chunks = []
    
    for chunk in chunks:
        if len(chunk.page_content) <= max_size:
            split_chunks.append(chunk)
        else:
            large_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_size,
                chunk_overlap=child_splitter._chunk_overlap
            )
            sub_chunks = large_splitter.split_documents([chunk])
            split_chunks.extend(sub_chunks)
    
    return split_chunks


def clean_small_chunks(chunks, min_size):
    """
    Clean up remaining small chunks by merging with neighbors.
    
    Args:
        chunks: List of document chunks
        min_size: Minimum size threshold in characters
        
    Returns:
        List of cleaned chunks
    """
    cleaned = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk.page_content) < min_size:
            if cleaned:
                # Merge with previous chunk
                cleaned[-1].page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in cleaned[-1].metadata:
                        cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                    else:
                        cleaned[-1].metadata[k] = v
            elif i < len(chunks) - 1:
                # Merge with next chunk
                chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                for k, v in chunk.metadata.items():
                    if k in chunks[i + 1].metadata:
                        chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                    else:
                        chunks[i + 1].metadata[k] = v
            else:
                # Last chunk, keep it
                cleaned.append(chunk)
        else:
            cleaned.append(chunk)
    
    return cleaned


def index_documents(mode: str = "default"):
    """
    Main indexing function that processes all documents.

    Args:
        mode: Indexing mode - "default" (fail if exists), "rebuild" (delete and recreate),
              or "append" (add new documents only)

    Steps:
    1. Initialize embeddings and vector store
    2. Read and process Markdown files
    3. Create parent and child chunks
    4. Save parent chunks to JSON
    5. Index child chunks in Qdrant
    """
    print("\n" + "="*60)
    print("📚 Starting Document Indexing")
    print("="*60 + "\n")
    
    # =========================================================================
    # Step 1: Initialize embeddings
    # =========================================================================
    
    print("🔧 Initializing embeddings...")
    
    dense_embeddings = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)
    
    print(f"✓ Dense embeddings: {DENSE_MODEL}")
    print(f"✓ Sparse embeddings: {SPARSE_MODEL}\n")
    
    # =========================================================================
    # Step 2: Initialize Qdrant
    # =========================================================================
    
    print("🔧 Initializing Qdrant vector database...")
    
    client = QdrantClient(path=QDRANT_DB_PATH)
    embedding_dimension = len(dense_embeddings.embed_query("test"))
    
    # Handle existing collection based on mode
    collection_exists = client.collection_exists(CHILD_COLLECTION)

    if collection_exists:
        if mode == "default":
            print(f"\n❌ Error: Index already exists!")
            print(f"   Collection '{CHILD_COLLECTION}' found in {QDRANT_DB_PATH}/")
            print(f"\nOptions:")
            print(f"   python indexing.py --rebuild    # Delete and rebuild from scratch")
            print(f"   python indexing.py --append     # Add new documents only")
            return
        elif mode == "rebuild":
            print(f"⚠️  Rebuild mode: Removing existing collection '{CHILD_COLLECTION}'...")
            client.delete_collection(CHILD_COLLECTION)
        elif mode == "append":
            print(f"📎 Append mode: Will add to existing collection '{CHILD_COLLECTION}'...")
    
    ensure_collection(client, CHILD_COLLECTION, embedding_dimension)
    
    child_vector_store = QdrantVectorStore(
        client=client,
        collection_name=CHILD_COLLECTION,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="sparse"
    )
    
    print(f"✓ Qdrant initialized\n")
    
    # =========================================================================
    # Step 3: Initialize splitters
    # =========================================================================
    
    print("🔧 Initializing text splitters...")
    
    parent_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    
    print(f"✓ Parent chunks: {MIN_PARENT_SIZE}-{MAX_PARENT_SIZE} chars")
    print(f"✓ Child chunks: {CHILD_CHUNK_SIZE} chars (overlap: {CHILD_CHUNK_OVERLAP})\n")
    
    # =========================================================================
    # Step 4: Process documents
    # =========================================================================
    
    # Check if we should use DOCS_DIR or MARKDOWN_DIR
    # If MARKDOWN_DIR exists and has files, use it; otherwise use DOCS_DIR
    if os.path.exists(MARKDOWN_DIR) and glob.glob(os.path.join(MARKDOWN_DIR, "*.md")):
        source_dir = MARKDOWN_DIR
        print(f"📂 Using markdown directory: {MARKDOWN_DIR}")
    else:
        source_dir = DOCS_DIR
        print(f"📂 Using docs directory: {DOCS_DIR}")
    
    md_files = sorted(glob.glob(os.path.join(source_dir, "*.md")))
    
    if not md_files:
        print(f"⚠️  No .md files found in {source_dir}/")
        print("Please add Markdown documents to index.")
        return
    
    print(f"📄 Found {len(md_files)} document(s)\n")
    
    all_parent_pairs, all_child_chunks = [], []
    
    for doc_path_str in md_files:
        doc_path = Path(doc_path_str)
        print(f"📄 Processing: {doc_path.name}")
        
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                md_text = f.read()
        except Exception as e:
            print(f"❌ Error reading {doc_path.name}: {e}")
            continue
        
        # Split by headers
        parent_chunks = parent_splitter.split_text(md_text)
        print(f"   → Initial chunks: {len(parent_chunks)}")
        
        # Merge small chunks
        merged_parents = merge_small_parents(parent_chunks, MIN_PARENT_SIZE)
        print(f"   → After merging small: {len(merged_parents)}")
        
        # Split large chunks
        split_parents = split_large_parents(merged_parents, MAX_PARENT_SIZE, child_splitter)
        print(f"   → After splitting large: {len(split_parents)}")
        
        # Clean up remaining small chunks
        cleaned_parents = clean_small_chunks(split_parents, MIN_PARENT_SIZE)
        print(f"   → Final parent chunks: {len(cleaned_parents)}")
        
        # Create child chunks and link to parents
        for i, p_chunk in enumerate(cleaned_parents):
            parent_id = f"{doc_path.stem}_parent_{i}"
            p_chunk.metadata.update({
                "source": doc_path.stem + ".md",
                "parent_id": parent_id
            })
            all_parent_pairs.append((parent_id, p_chunk))
            
            # Split parent into child chunks
            children = child_splitter.split_documents([p_chunk])
            all_child_chunks.extend(children)
            print(f"   → Parent {i}: {len(children)} child chunks")
        
        print()
    
    # =========================================================================
    # Step 5: Save parent chunks
    # =========================================================================
    
    if not all_parent_pairs:
        print("⚠️  No parent chunks to save")
        return
    
    print(f"💾 Saving {len(all_parent_pairs)} parent chunks to JSON...")

    parent_manager = ParentStoreManager()
    if mode == "rebuild":
        parent_manager.clear_store()  # Clear old data only in rebuild mode
    parent_manager.save_many(all_parent_pairs)
    
    print(f"✓ Parent chunks saved to: {PARENT_STORE_PATH}\n")
    
    # =========================================================================
    # Step 6: Index child chunks
    # =========================================================================
    
    if not all_child_chunks:
        print("⚠️  No child chunks to index")
        return
    
    print(f"🔍 Indexing {len(all_child_chunks)} child chunks into Qdrant...")
    
    try:
        child_vector_store.add_documents(all_child_chunks)
        print(f"✓ Child chunks indexed successfully\n")
    except Exception as e:
        print(f"❌ Error indexing child chunks: {e}")
        return
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("="*60)
    print("✅ Indexing Complete!")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   - Documents processed: {len(md_files)}")
    print(f"   - Parent chunks: {len(all_parent_pairs)}")
    print(f"   - Child chunks: {len(all_child_chunks)}")
    print(f"   - Vector database: {QDRANT_DB_PATH}/")
    print(f"   - Parent store: {PARENT_STORE_PATH}/")
    print("\n🎉 Your knowledge base is ready!")
    print("   Run 'python test_chat.py' to test the system.\n")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Index documents for Singapore Housing Assistant RAG System"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing index and rebuild from scratch"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add new documents without deleting existing data"
    )
    args = parser.parse_args()

    # Determine mode
    if args.rebuild and args.append:
        print("❌ Error: Cannot use --rebuild and --append together")
        exit(1)
    elif args.rebuild:
        mode = "rebuild"
    elif args.append:
        mode = "append"
    else:
        mode = "default"

    try:
        index_documents(mode=mode)
    except KeyboardInterrupt:
        print("\n\n⚠️  Indexing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()