#!/usr/bin/env python3
"""
RAG Query Interface
===================

Interactive query interface for the RAG system.

Features:
- Hybrid search (semantic + BM25 + keyword)
- 2-stage retrieval (document → chunk)
- Vendor filtering
- Cross-encoder reranking
- Anti-hallucination prompts

Usage:
    python query.py
    
Commands:
    !help      - Show help
    !show      - Show retrieved documents
    !filter    - Set filters (vendor, release)
    !config    - Show configuration
    !weights   - Adjust hybrid weights
    exit       - Exit
"""

import sys
import os
from pathlib import Path
import requests
from typing import List, Dict, Optional
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from retrieval.hybrid_retriever import HybridRetriever


# =============================================================================
# SYSTEM PROMPTS (Anti-hallucination)
# =============================================================================

SYSTEM_PROMPT_BASE = """You are a technical assistant specializing in telecommunications and 3GPP standards.

CRITICAL RULES:
1. Answer ONLY based on the provided context documents.
2. If the specific formula, value, or detail is NOT in the context, say: "This information is not available in the retrieved documents."
3. DO NOT make up formulas, parameters, or technical specifications.
4. DO NOT guess or infer values not explicitly stated.
5. Always cite your source (filename:page) for each fact you state.
6. If multiple documents give conflicting information, point out the discrepancy.

Your response should be:
- Concise and technical
- Properly cited with (source:page) after each fact
- Based ONLY on retrieved context
"""

# Additional instruction when vendor filter is active
VENDOR_PREFERENCE_PROMPT = """
VENDOR RESTRICTION:
The user has specifically requested information from {vendor} documents.
- ONLY use information from {vendor} documents
- Ignore other vendor content even if present
- If {vendor} documents don't contain the answer, say "Not found in {vendor} documentation"
"""

# Additional instruction when multiple vendors present
MULTI_VENDOR_PROMPT = """
MULTIPLE VENDORS DETECTED:
Documents from multiple vendors are present in context.
- Clearly indicate which vendor each piece of information comes from
- Do NOT mix specifications from different vendors in the same statement
- If values differ between vendors, present each separately
"""

USER_TEMPLATE = """Context documents:
{context}

---
Question: {question}

Provide a precise answer based ONLY on the above context. Cite sources (filename:page) for each fact. If the answer is not in the context, explicitly say so."""

USER_TEMPLATE_WITH_VENDOR = """Context documents (filtered for {vendor}):
{context}

---
Question: {question}

Answer using ONLY the {vendor} documentation above. Cite sources (filename:page) for each fact. If {vendor} documents don't contain the answer, say "Not found in {vendor} documentation"."""


# =============================================================================
# GLOBALS
# =============================================================================

CONVERSATION_HISTORY = []
LAST_RETRIEVED = []
ACTIVE_FILTERS = {}


# =============================================================================
# FUNCTIONS
# =============================================================================

def compress_context(retrieved: List[Dict], question: str) -> str:
    """
    Optional: Compress context using LLM to reduce tokens.
    Useful when you have many chunks and need to fit in context window.
    
    This adds latency but improves answer quality for complex queries.
    """
    # Group by section
    sections = defaultdict(list)
    for doc in retrieved:
        section = doc['metadata'].get('section', 'Unknown')
        sections[section].append(doc)
    
    compressed_parts = []
    
    for section, docs in sections.items():
        # Combine texts from same section
        combined = f"Section: {section}\n"
        for doc in docs[:3]:  # Max 3 docs per section
            source = doc['metadata'].get('source', 'unknown')
            combined += f"[{source}]\n{doc['text'][:1000]}\n"
        
        # Ask LLM to summarize
        summary_prompt = f"""Summarize the following technical content, keeping all specific values, formulas, and technical details relevant to: "{question}"

{combined}

Provide a concise summary (max 200 words) with all technical specifics preserved:"""
        
        try:
            response = requests.post(
                settings.LLM_API_URL,
                json={
                    "model": settings.LLM_MODEL_NAME,
                    "messages": [{"role": "user", "content": summary_prompt}],
                    "max_tokens": 300,
                    "temperature": 0.1
                },
                timeout=60
            )
            if response.ok:
                data = response.json()
                if "choices" in data:
                    summary = data["choices"][0].get("message", {}).get("content", combined[:500])
                    compressed_parts.append(f"[{section}]\n{summary}")
        except:
            # Fallback: just truncate
            compressed_parts.append(f"[{section}]\n{combined[:500]}")
    
    return "\n\n---\n\n".join(compressed_parts)


def build_context(retrieved: List[Dict], max_tokens: int = 6000) -> str:
    """Build context string from retrieved documents"""
    pieces = []
    total_chars = 0
    max_chars = max_tokens * 4  # ~4 chars per token
    
    for i, r in enumerate(retrieved, 1):
        meta = r['metadata']
        source = meta.get('source', 'unknown')
        pages = meta.get('pages', [])
        page_str = f"{pages[0]}-{pages[-1]}" if pages and len(pages) > 1 else str(pages[0]) if pages else "N/A"
        section = meta.get('section', '')
        vendor = meta.get('vendor', '')
        
        header = f"[Document {i}] {source} | Page: {page_str}"
        if vendor:
            header += f" | Vendor: {vendor}"
        if section:
            header += f" | Section: {section}"
        header += f" | Score: {r['score']:.3f}\n"
        
        text = r['text'][:2000]  # Truncate per document
        piece = header + text
        
        if total_chars + len(piece) > max_chars:
            break
        
        pieces.append(piece)
        total_chars += len(piece)
    
    return "\n\n---\n\n".join(pieces)


def query_llm(
    question: str, 
    context: str, 
    vendor_filter: Optional[str] = None,
    vendors_in_context: List[str] = None
) -> str:
    """
    Send query to LLM API with appropriate prompting.
    
    Args:
        question: User's question
        context: Retrieved context
        vendor_filter: If set, restrict to this vendor
        vendors_in_context: List of vendors present in context
    """
    # Build system prompt
    system_prompt = SYSTEM_PROMPT_BASE
    
    if vendor_filter:
        system_prompt += VENDOR_PREFERENCE_PROMPT.format(vendor=vendor_filter)
    elif vendors_in_context and len(vendors_in_context) > 1:
        system_prompt += MULTI_VENDOR_PROMPT
    
    # Build user prompt
    if vendor_filter:
        prompt = USER_TEMPLATE_WITH_VENDOR.format(
            vendor=vendor_filter,
            context=context,
            question=question
        )
    else:
        prompt = USER_TEMPLATE.format(context=context, question=question)
    
    payload = {
        "model": settings.LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        **settings.LLM_PARAMS
    }
    
    try:
        response = requests.post(settings.LLM_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            elif "text" in choice:
                return choice["text"]
        
        return "Error: Unexpected response format"
        
    except requests.exceptions.ConnectionError:
        return "❌ Error: Cannot connect to LLM API. Make sure text-generation-webui is running."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def show_help():
    """Show help message"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                       RAG SYSTEM v2.0 - HELP                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  🔍 QUERY TIPS:                                                      ║
║     • Use "quotes" for exact matches: "SN0012" or "f1 function"     ║
║     • Case-insensitive: sn0012 = SN0012                             ║
║     • Specify vendor: "Ericsson MIMO" or use !filter                ║
║                                                                      ║
║  📊 COMMANDS:                                                        ║
║     !help      - Show this help                                      ║
║     !show      - Show retrieved document details                     ║
║     !filter    - Set filters (vendor: ericsson, release: 17)        ║
║     !clear     - Clear filters and history                           ║
║     !config    - Show current configuration                          ║
║     !weights   - Adjust hybrid search weights                        ║
║     !stats     - Show system statistics                              ║
║     exit       - Exit the system                                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def show_retrieved():
    """Show detailed retrieved documents"""
    global LAST_RETRIEVED
    
    if not LAST_RETRIEVED:
        print("❌ No documents retrieved yet!")
        return
    
    print("\n" + "=" * 70)
    print("📚 RETRIEVED DOCUMENTS")
    print("=" * 70)
    
    for i, doc in enumerate(LAST_RETRIEVED, 1):
        meta = doc['metadata']
        scores = doc.get('scores', {})
        
        print(f"\n📌 Document {i}/{len(LAST_RETRIEVED)}")
        print("-" * 70)
        print(f"📁 Source: {meta.get('source', 'unknown')}")
        print(f"📄 Pages: {meta.get('pages', 'N/A')}")
        print(f"📂 Section: {meta.get('section', 'N/A')}")
        print(f"🏢 Vendor: {meta.get('vendor', 'N/A')}")
        print(f"🔍 Type: {doc['retrieval_type']}")
        
        print(f"\n⚖️  Scores:")
        print(f"   Semantic: {scores.get('semantic', 0):.3f}")
        print(f"   BM25: {scores.get('bm25', 0):.3f}")
        print(f"   Keyword: {scores.get('keyword', 0):.3f}")
        print(f"   Rerank: {scores.get('rerank', 0):.3f}")
        print(f"   Final: {doc['score']:.3f}")
        
        print(f"\n📝 Preview (500 chars):")
        print("-" * 70)
        print(doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'])


def set_filters():
    """Interactive filter setting"""
    global ACTIVE_FILTERS
    
    print("\n⚙️  SET FILTERS")
    print("-" * 40)
    print(f"Current filters: {ACTIVE_FILTERS if ACTIVE_FILTERS else 'None'}")
    print("\nAvailable options:")
    print("  1. Set vendor filter (ericsson, nokia, huawei, 3gpp)")
    print("  2. Set release filter (15, 16, 17, 18)")
    print("  3. Clear all filters")
    print("  0. Cancel")
    
    try:
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            vendor = input("Enter vendor: ").strip().lower()
            if vendor:
                ACTIVE_FILTERS['vendor'] = vendor
                print(f"✅ Vendor filter set: {vendor}")
        elif choice == '2':
            release = input("Enter release number: ").strip()
            if release.isdigit():
                ACTIVE_FILTERS['release'] = int(release)
                print(f"✅ Release filter set: {release}")
        elif choice == '3':
            ACTIVE_FILTERS = {}
            print("✅ Filters cleared")
    except Exception as e:
        print(f"Error: {e}")


def show_config():
    """Show current configuration"""
    print("\n" + "=" * 70)
    print("⚙️  CONFIGURATION")
    print("=" * 70)
    
    print(f"\n🔗 LLM API: {settings.LLM_API_URL}")
    print(f"🤖 Model: {settings.LLM_MODEL_NAME}")
    
    print(f"\n📊 Retrieval Settings:")
    print(f"   TOP_K: {settings.TOP_K}")
    print(f"   RERANK_TOP_K: {settings.RERANK_TOP_K}")
    print(f"   CONTEXT_CHUNKS: {settings.CONTEXT_CHUNKS}")
    
    print(f"\n⚖️  Hybrid Weights:")
    print(f"   Semantic: {settings.HYBRID_WEIGHTS['semantic']:.0%}")
    print(f"   BM25: {settings.HYBRID_WEIGHTS['bm25']:.0%}")
    print(f"   Keyword: {settings.HYBRID_WEIGHTS['keyword']:.0%}")
    
    print(f"\n🔍 Active Filters: {ACTIVE_FILTERS if ACTIVE_FILTERS else 'None'}")


def adjust_weights():
    """Adjust hybrid search weights"""
    print("\n" + "=" * 70)
    print("⚖️  HYBRID SEARCH WEIGHTS")
    print("=" * 70)
    
    print(f"\nCurrent: S={settings.HYBRID_WEIGHTS['semantic']:.0%}, "
          f"B={settings.HYBRID_WEIGHTS['bm25']:.0%}, "
          f"K={settings.HYBRID_WEIGHTS['keyword']:.0%}")
    
    print("\nPresets:")
    print("  1. Balanced:     40% semantic, 30% BM25, 30% keyword (default)")
    print("  2. Semantic:     60% semantic, 25% BM25, 15% keyword")
    print("  3. Exact-match:  25% semantic, 45% BM25, 30% keyword")
    print("  4. Technical:    30% semantic, 30% BM25, 40% keyword")
    print("  0. Cancel")
    
    try:
        choice = input("\nSelect preset: ").strip()
        
        presets = {
            '1': {'semantic': 0.40, 'bm25': 0.30, 'keyword': 0.30},
            '2': {'semantic': 0.60, 'bm25': 0.25, 'keyword': 0.15},
            '3': {'semantic': 0.25, 'bm25': 0.45, 'keyword': 0.30},
            '4': {'semantic': 0.30, 'bm25': 0.30, 'keyword': 0.40},
        }
        
        if choice in presets:
            settings.HYBRID_WEIGHTS.update(presets[choice])
            print(f"✅ Updated: S={settings.HYBRID_WEIGHTS['semantic']:.0%}, "
                  f"B={settings.HYBRID_WEIGHTS['bm25']:.0%}, "
                  f"K={settings.HYBRID_WEIGHTS['keyword']:.0%}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main query loop"""
    global LAST_RETRIEVED, CONVERSATION_HISTORY, ACTIVE_FILTERS
    
    # Initialize retriever
    print("\n" + "=" * 70)
    print("🚀 RAG QUERY SYSTEM v2.0")
    print("   Optimized for Technical Documents")
    print("=" * 70)
    
    try:
        retriever = HybridRetriever()
    except Exception as e:
        print(f"\n❌ Error loading retriever: {e}")
        print("   Run 'python build_index.py' first to create the index.")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("🎉 System Ready!")
    print("   Type !help for commands, 'exit' to quit")
    print("=" * 70)
    
    while True:
        try:
            query = input("\n🔎 Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'exit':
                print("\n👋 Goodbye!")
                break
            
            # Handle commands
            if query.startswith('!'):
                cmd = query[1:].lower().split()[0]
                
                if cmd == 'help':
                    show_help()
                elif cmd == 'show':
                    show_retrieved()
                elif cmd == 'filter':
                    set_filters()
                elif cmd == 'config':
                    show_config()
                elif cmd == 'weights':
                    adjust_weights()
                elif cmd == 'clear':
                    ACTIVE_FILTERS = {}
                    CONVERSATION_HISTORY = []
                    LAST_RETRIEVED = []
                    print("✅ Cleared filters and history")
                elif cmd == 'stats':
                    print(f"\n📊 Stats:")
                    print(f"   Chunks indexed: {len(retriever.texts):,}")
                    print(f"   Documents: {len(retriever.documents):,}")
                    print(f"   Conversation turns: {len(CONVERSATION_HISTORY)}")
                else:
                    print(f"❌ Unknown command: {cmd}. Type !help")
                continue
            
            # Retrieve
            print(f"\n🔍 Searching...")
            retrieved = retriever.retrieve(
                query,
                filters=ACTIVE_FILTERS if ACTIVE_FILTERS else None,
                top_k=settings.CONTEXT_CHUNKS
            )
            LAST_RETRIEVED = retrieved
            
            # Show results summary
            print(f"\n📚 Retrieved {len(retrieved)} chunks:")
            for i, doc in enumerate(retrieved[:5], 1):
                meta = doc['metadata']
                source = meta.get('source', 'unknown')[:40]
                vendor = meta.get('vendor', '')
                vendor_str = f"[{vendor}]" if vendor else ""
                pages = meta.get('pages', [])
                page_str = f"p{pages[0]}" if pages else ""
                
                print(f"  {i}. {vendor_str} {source} {page_str} | {doc['score']:.3f} ({doc['retrieval_type']})")
            
            if len(retrieved) > 5:
                print(f"  ... and {len(retrieved) - 5} more (!show for details)")
            
            # Analyze vendors in results
            vendors_in_context = list(set(
                doc['metadata'].get('vendor') 
                for doc in retrieved 
                if doc['metadata'].get('vendor')
            ))
            
            if vendors_in_context:
                print(f"   📊 Vendors in results: {', '.join(vendors_in_context)}")
            
            # Generate response
            print(f"\n🤖 Generating response...")
            context = build_context(retrieved, max_tokens=6000)
            vendor_filter = ACTIVE_FILTERS.get('vendor')
            answer = query_llm(query, context, vendor_filter, vendors_in_context)
            
            print(f"\n💬 ANSWER:\n{answer}")
            
            # Save history
            CONVERSATION_HISTORY.append({
                'question': query,
                'answer': answer,
                'retrieved': [r['idx'] for r in retrieved]
            })
            
            print("\n" + "-" * 70)
            print("💡 Commands: !show | !filter | !weights | !help")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

