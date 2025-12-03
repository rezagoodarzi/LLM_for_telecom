# rag_query.py
import os, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from urllib.parse import urljoin
import re
from difflib import SequenceMatcher

# -------- CONFIG --------
STORE_DIR = "./rag_store"
INDEX_PATH = os.path.join(STORE_DIR, "faiss.index")
META_PATH = os.path.join(STORE_DIR, "metadata.json")
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Must match word_embbeding.py
TOP_K = 5  # Increased for better coverage of exact ID matches
VLLM_URL = "http://localhost:5000/v1/chat/completions"  # text-generation-webui API endpoint
VLLM_MODEL_NAME = "qwen3-4b-bnb4"  # model name as loaded in webui

# Model generation parameters (can be changed at runtime)
MODEL_CONFIG = {
    "temperature": 0.0,      # 0.0 = deterministic, 1.0 = creative
    "top_p": 0.95,          # nucleus sampling threshold
    "max_tokens": 1000,      # max response length
    "top_k": 40,            # top-k sampling
    "repetition_penalty": 1.1,  # penalty for repeating tokens
}

# Default config for reset
DEFAULT_CONFIG = MODEL_CONFIG.copy()
# ------------------------
print("Loading store...")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    store = json.load(f)
texts = store["texts"]
metadatas = store["metadatas"]
keywords_index = store.get("keywords_index", {})
phrases_index = store.get("phrases_index", {})

embedder = SentenceTransformer(EMBED_MODEL)

# Load cross-encoder for re-ranking (optional but recommended)
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    USE_RERANKER = True
    print("Cross-encoder loaded for re-ranking")
except Exception as e:
    print(f"Cross-encoder not available (install: pip install sentence-transformers): {e}")
    USE_RERANKER = False

def query_embedding(q):
    emb = embedder.encode([q], convert_to_numpy=True)
    emb = emb.astype("float32")
    faiss.normalize_L2(emb)
    return emb

def extract_quoted_phrases(query):
    """Extract phrases enclosed in quotes (highest priority)"""
    import re
    # Match both single and double quotes
    patterns = [
        r'"([^"]+)"',  # Double quotes
        r"'([^']+)'",  # Single quotes
    ]
    
    quoted = []
    for pattern in patterns:
        matches = re.findall(pattern, query)
        quoted.extend([m.strip() for m in matches if m.strip()])
    
    return quoted

def extract_phrases(query):
    """Extract important multi-word phrases from query"""
    # Remove common stopwords
    stop_words = {'what', 'is', 'the', 'for', 'in', 'to', 'a', 'an', 'of', 'how', 'when', 'where', 'why', 'do', 'does', 'are', 'am', 'be', 'been'}
    words = query.lower().split()
    
    # Filter out stop words for phrase extraction
    filtered_words = [w for w in words if w not in stop_words]
    
    phrases = []
    # Extract 2-word, 3-word, and 4-word phrases
    for n in [4, 3, 2]:  # Start with longer phrases
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i+n])
            # Check if phrase has at least one meaningful word
            if any(w not in stop_words for w in words[i:i+n]):
                phrases.append(phrase)
    
    # Also add the full query if it's reasonable length
    if 3 <= len(words) <= 10:
        phrases.append(query.lower())
    
    return phrases

def expand_query(query):
    """Expand query with synonyms and related terms"""
    query_lower = query.lower()
    expanded_terms = [query_lower]
    
    # Domain-specific expansions for RF/telecom terms
    expansions = {
        "gap length": ["gap length", "gap distance", "gap size", "inter-site gap"],
        "gap": ["gap", "distance", "spacing", "separation"],
        "site to site": ["site to site", "site-to-site", "inter-site", "ISD", "inter site distance"],
        "coverage": ["coverage", "service area", "cell coverage"],
        "maximum": ["maximum", "max", "upper limit", "threshold", "highest", "peak"],
        "minimum": ["minimum", "min", "lower limit", "lowest"],
        "distance": ["distance", "range", "spacing", "separation"],
        "cell": ["cell", "site", "sector"],
        "configuration": ["configuration", "config", "setup", "setting"],
        "parameter": ["parameter", "setting", "value"],
    }
    
    for key, synonyms in expansions.items():
        if key in query_lower:
            for syn in synonyms:
                if syn not in query_lower:
                    expanded_terms.append(query_lower.replace(key, syn))
    
    return expanded_terms[:5]  # Limit to avoid too many variations

def extract_query_patterns(query):
    """Extract potential IDs, codes, and keywords from query"""
    patterns = []
    query_upper = query.upper()
    
    # Same patterns as in ingestion
    id_patterns = [
        (r'\b[A-Z]{2,5}\d{4,6}\b', 'id'),
        (r'\b\d{4,6}\b', 'number'),
        (r'\b[A-Z]+-\d+\b', 'code'),
        (r'\b\d+\.\d+\.\d+\b', 'version'),
        (r'\b[A-Z]{2,}\.\d+\b', 'ref'),
    ]
    
    for pattern, ptype in id_patterns:
        matches = re.findall(pattern, query_upper)
        for match in matches:
            patterns.append((match, ptype, 1.0))  # exact priority
    
    # Also extract capitalized words
    cap_words = re.findall(r'\b[A-Z][A-Za-z]{3,}\b', query)
    for word in cap_words:
        patterns.append((word.upper(), 'term', 0.8))
    
    return patterns

def fuzzy_match(keyword, target, threshold=0.8):
    """Check if keyword fuzzy matches target"""
    ratio = SequenceMatcher(None, keyword, target).ratio()
    return ratio >= threshold

def retrieve(q, top_k=TOP_K):
    """Advanced retrieval with phrase matching, query expansion, and multi-level scoring"""
    all_results = {}  # Use dict to avoid duplicates, keyed by idx
    
    # === Level 0: QUOTED PHRASES (ULTRA HIGHEST PRIORITY) ===
    quoted_phrases = extract_quoted_phrases(q)
    if quoted_phrases:
        print(f"🔍 Prioritizing quoted terms: {quoted_phrases}")
    
    for quoted in quoted_phrases:
        quoted_lower = quoted.lower()
        # Search in all texts for exact match
        for idx, text in enumerate(texts):
            if quoted_lower in text.lower():
                all_results[idx] = {
                    "idx": idx,
                    "score": 5.0,  # ULTRA HIGH PRIORITY - Above all other matches
                    "text": texts[idx],
                    "meta": metadatas[idx],
                    "match_type": "quoted_exact",
                    "matched": quoted
                }
        
        # Also check in phrases index
        if quoted_lower in phrases_index:
            for idx in phrases_index[quoted_lower]:
                if idx not in all_results:
                    all_results[idx] = {
                        "idx": idx,
                        "score": 5.0,
                        "text": texts[idx],
                        "meta": metadatas[idx],
                        "match_type": "quoted_indexed",
                        "matched": quoted
                    }
    
    # === Level 1: Phrase Matching (High Priority) ===
    query_phrases = extract_phrases(q)
    for phrase in query_phrases:
        if phrase in phrases_index:
            for idx in phrases_index[phrase]:
                if idx not in all_results or all_results[idx]["score"] < 3.0:
                    all_results[idx] = {
                        "idx": idx,
                        "score": 3.0,  # Highest priority for exact phrase match
                        "text": texts[idx],
                        "meta": metadatas[idx],
                        "match_type": "phrase_match",
                        "matched": phrase
                    }
        
        # Also check if phrase appears in text (even if not indexed)
        phrase_lower = phrase.lower()
        for idx, text in enumerate(texts):
            if phrase_lower in text.lower():
                current_score = all_results.get(idx, {}).get("score", 0)
                if current_score < 2.5:
                    all_results[idx] = {
                        "idx": idx,
                        "score": 2.5,
                        "text": texts[idx],
                        "meta": metadatas[idx],
                        "match_type": "phrase_in_text",
                        "matched": phrase
                    }
    
    # === Level 2: Pattern/ID Matching ===
    query_patterns = extract_query_patterns(q)
    for pattern, ptype, priority in query_patterns:
        # Exact matches from keywords index
        if pattern in keywords_index:
            for idx in keywords_index[pattern]:
                current_score = all_results.get(idx, {}).get("score", 0)
                if current_score < 2.0:
                    all_results[idx] = {
                        "idx": idx,
                        "score": 2.0,
                        "text": texts[idx],
                        "meta": metadatas[idx],
                        "match_type": f"exact_{ptype}",
                        "matched": pattern
                    }
        
        # Fuzzy matches for IDs and codes
        if ptype in ['id', 'code', 'ref', 'version']:
            for kw in keywords_index:
                if fuzzy_match(pattern, kw, 0.85):
                    for idx in keywords_index[kw]:
                        current_score = all_results.get(idx, {}).get("score", 0)
                        if current_score < 1.8:
                            all_results[idx] = {
                                "idx": idx,
                                "score": 1.8,
                                "text": texts[idx],
                                "meta": metadatas[idx],
                                "match_type": f"fuzzy_{ptype}",
                                "matched": f"{pattern}~{kw}"
                            }
    
    # === Level 3: Query Expansion with Synonyms ===
    expanded_queries = expand_query(q)
    for exp_query in expanded_queries[1:]:  # Skip original query
        for idx, text in enumerate(texts):
            if exp_query in text.lower():
                current_score = all_results.get(idx, {}).get("score", 0)
                if current_score < 1.5:
                    all_results[idx] = {
                        "idx": idx,
                        "score": 1.5,
                        "text": texts[idx],
                        "meta": metadatas[idx],
                        "match_type": "synonym_match",
                        "matched": exp_query
                    }
    
    # === Level 4: Individual Word Matching ===
    q_upper = q.upper()
    query_words = [w for w in q_upper.split() if len(w) > 3]
    for idx, text in enumerate(texts):
        text_upper = text.upper()
        word_matches = sum(1 for word in query_words if word in text_upper)
        if word_matches > 0:
            word_score = 0.5 + (word_matches * 0.2)
            current_score = all_results.get(idx, {}).get("score", 0)
            if current_score < word_score:
                all_results[idx] = {
                    "idx": idx,
                    "score": word_score,
                    "text": texts[idx],
                    "meta": metadatas[idx],
                    "match_type": "word_match",
                    "matched": f"{word_matches} words"
                }
    
    # === Level 5: Semantic Search (Fallback) ===
    emb = query_embedding(q)
    D, I = index.search(emb, top_k * 4)  # Get more candidates for re-ranking
    
    seen_indices = set(all_results.keys())
    for score, idx in zip(D[0], I[0]):
        if idx not in seen_indices:
            semantic_score = float(score) * 0.8  # Balanced with other methods
            all_results[idx] = {
                "idx": idx,
                "score": semantic_score,
                "text": texts[idx],
                "meta": metadatas[idx],
                "match_type": "semantic",
                "matched": ""
            }
    
    # Convert to list and sort by score
    results_list = list(all_results.values())
    results_list.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top results (more for re-ranking)
    return results_list[:min(top_k * 3, len(results_list))]

def rerank_results(query, results, top_k=TOP_K):
    """Re-rank results using cross-encoder for better accuracy"""
    if not USE_RERANKER or len(results) == 0:
        return results[:top_k]
    
    # Prepare pairs for cross-encoder
    pairs = [[query, r['text'][:1000]] for r in results]  # Limit text length
    
    try:
        # Get cross-encoder scores
        ce_scores = reranker.predict(pairs)
        
        # Combine original scores with cross-encoder scores
        for i, score in enumerate(ce_scores):
            # Weighted combination: 40% original, 60% cross-encoder
            results[i]['original_score'] = results[i]['score']
            results[i]['ce_score'] = float(score)
            results[i]['score'] = (results[i]['original_score'] * 0.4) + (float(score) * 0.6)
            results[i]['match_type'] = f"{results[i]['match_type']}_reranked"
        
        # Re-sort by combined score
        results.sort(key=lambda x: x['score'], reverse=True)
    except Exception as e:
        print(f"Re-ranking error: {e}")
    
    return results[:top_k]

def build_context(retrieved):
    # assemble the retrieved texts; limit total length if necessary
    pieces = []
    for r in retrieved:
        header = f"[{r['meta']['source']} | page {r['meta']['page']} | score={r['score']:.3f}]\n"
        pieces.append(header + r["text"])
    return "\n\n---\n\n".join(pieces)

def show_config():
    """Display current model configuration"""
    print("\n=== Current Model Configuration ===")
    for key, value in MODEL_CONFIG.items():
        print(f"{key}: {value}")
    print("==================================")

def change_config():
    """Interactive menu to change model parameters"""
    print("\n=== Change Model Configuration ===")
    print("Available parameters:")
    for i, (key, value) in enumerate(MODEL_CONFIG.items(), 1):
        print(f"{i}. {key}: {value}")
    print("0. Back to main menu")
    
    try:
        choice = int(input("\nSelect parameter to change (0-{}): ".format(len(MODEL_CONFIG))))
        if choice == 0:
            return
        
        keys = list(MODEL_CONFIG.keys())
        if 1 <= choice <= len(keys):
            param = keys[choice - 1]
            current = MODEL_CONFIG[param]
            
            # Get appropriate input based on parameter type
            if param in ["temperature", "top_p"]:
                print(f"\nCurrent {param}: {current}")
                print(f"Range: 0.0 - 1.0")
                new_value = float(input(f"New value: "))
                MODEL_CONFIG[param] = max(0.0, min(1.0, new_value))
            elif param in ["max_tokens", "top_k"]:
                print(f"\nCurrent {param}: {current}")
                print(f"Range: 1 - 2048 (for max_tokens), 1 - 100 (for top_k)")
                new_value = int(input(f"New value: "))
                if param == "max_tokens":
                    MODEL_CONFIG[param] = max(1, min(2048, new_value))
                else:
                    MODEL_CONFIG[param] = max(1, min(100, new_value))
            elif param == "repetition_penalty":
                print(f"\nCurrent {param}: {current}")
                print(f"Range: 1.0 - 2.0 (1.0 = no penalty)")
                new_value = float(input(f"New value: "))
                MODEL_CONFIG[param] = max(1.0, min(2.0, new_value))
            
            print(f"\n✓ {param} changed to {MODEL_CONFIG[param]}")
    except (ValueError, IndexError):
        print("Invalid input. No changes made.")

def print_help():
    """Show available commands"""
    print("\n=== Available Commands ===")
    print("Type a question to search and get AI response")
    print("\n💡 Search Tips:")
    print('  Use "quotes" around important terms for highest priority')
    print('  Example: what is "gap length" for coverage?')
    print('  Quoted terms get 5x priority boost!')
    print("\nSpecial commands:")
    print("  !config     - Show current model configuration")
    print("  !change     - Change model parameters interactively")
    print("  !presets    - Show common configuration presets")
    print("  !preset <n> - Apply preset configuration (e.g., !preset 3)")
    print("  !reset      - Reset to default configuration")
    print("  !help       - Show this help message")
    print("  exit        - Exit the program")
    print("==========================")

def show_presets():
    """Show common configuration presets"""
    print("\n=== Common Configuration Presets ===")
    print("\n1. Deterministic (current default):")
    print("   temperature=0.0, top_p=0.95")
    print("   Best for: Factual Q&A, consistent answers")
    
    print("\n2. Balanced:")
    print("   temperature=0.3, top_p=0.9")
    print("   Best for: General use with slight variation")
    
    print("\n3. Creative:")
    print("   temperature=0.7, top_p=0.85")
    print("   Best for: More diverse responses")
    
    print("\n4. Very Creative:")
    print("   temperature=1.0, top_p=0.95")
    print("   Best for: Maximum variation and creativity")
    
    print("\n5. Long Form:")
    print("   max_tokens=800, temperature=0.3")
    print("   Best for: Detailed explanations")
    
    print("\nUse !preset <number> to quickly apply a preset")
    print("Use !change to modify individual parameters")
    print("====================================")

def apply_preset(preset_num):
    """Apply a preset configuration"""
    global MODEL_CONFIG
    
    presets = {
        1: {"temperature": 0.0, "top_p": 0.95},
        2: {"temperature": 0.3, "top_p": 0.9},
        3: {"temperature": 0.7, "top_p": 0.85},
        4: {"temperature": 1.0, "top_p": 0.95},
        5: {"max_tokens": 800, "temperature": 0.3}
    }
    
    if preset_num in presets:
        # Apply only the specified values, keep others as is
        for key, value in presets[preset_num].items():
            MODEL_CONFIG[key] = value
        print(f"\n✓ Applied preset {preset_num}")
        show_config()
    else:
        print(f"Invalid preset number. Choose 1-5.")

def reset_config():
    """Reset configuration to defaults"""
    global MODEL_CONFIG
    MODEL_CONFIG = DEFAULT_CONFIG.copy()
    print("\n✓ Configuration reset to defaults")
    show_config()

def ask_vllm_with_context(question, retrieved_texts):
    # Build prompt: system instruction + context + user question
    system = "You are an assistant that uses the provided context (delimited) to answer the user's question. Use only the context, cite sources (filename:page), and be concise. If the answer is not in context, say you don't know."
    context = build_context(retrieved_texts)
    user_message = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely and cite sources."

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message}
        ],
        **MODEL_CONFIG  # Use dynamic config
    }

    r = requests.post(VLLM_URL, json=payload, timeout=120)
    r.raise_for_status()
    j = r.json()
    # extract model reply robustly:
    text = ""
    if "choices" in j and len(j["choices"])>0:
        c = j["choices"][0]
        if "message" in c and "content" in c["message"]:
            text = c["message"]["content"]
        elif "text" in c:
            text = c["text"]
    return text, j

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 RAG Query System with Advanced Search")
    print("=" * 60)
    print('💡 TIP: Use "quotes" for important terms (e.g., "gap length")')
    print("📖 Type !help for all commands")
    print("=" * 60)
    
    while True:
        q = input("\nEnter question (or 'exit'): ").strip()
        
        if not q or q.lower() == "exit":
            break
        
        # Handle special commands
        if q.startswith("!"):
            if q == "!config":
                show_config()
                continue
            elif q == "!change":
                change_config()
                continue
            elif q == "!presets":
                show_presets()
                continue
            elif q.startswith("!preset "):
                try:
                    preset_num = int(q.split()[1])
                    apply_preset(preset_num)
                except (IndexError, ValueError):
                    print("Usage: !preset <number> (e.g., !preset 2)")
                continue
            elif q == "!reset":
                reset_config()
                continue
            elif q == "!help":
                print_help()
                continue
            else:
                print(f"Unknown command: {q}. Type !help for available commands.")
                continue
        
        # Normal question processing
        try:
            # Retrieve candidates
            retrieved = retrieve(q, top_k=TOP_K)
            
            # Re-rank for better accuracy
            retrieved = rerank_results(q, retrieved, top_k=TOP_K)
            
            print("\nRetrieved chunks:")
            for r in retrieved:
                match_type = f"[{r.get('match_type', 'semantic')}]"
                matched = f" (matched: {r.get('matched', '')})" if r.get('matched') else ""
                print(f"- {match_type} {r['meta']['source']} page {r['meta']['page']} score={r['score']:.4f}{matched}")

            ans, raw = ask_vllm_with_context(q, retrieved)
            print("\n=== ANSWER ===")
            print(ans)
            
            # Show current temperature if not default
            if MODEL_CONFIG["temperature"] != 0.0:
                print(f"\n(Generated with temperature: {MODEL_CONFIG['temperature']})")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Check that your text-generation-webui server is running.")
        
        # optionally print raw JSON
        # print(json.dumps(raw, indent=2))
