"""
Metadata Extractor for Document Filtering
==========================================

Extracts vendor, version, document type, and other metadata
from filenames and content for filtering purposes.

This prevents cross-document contamination in retrieval.
"""

import re
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class MetadataExtractor:
    """Extract metadata from documents for filtering"""
    
    # Vendor patterns
    VENDOR_PATTERNS = {
        'ericsson': [r'ericsson', r'eric\.', r'eri_'],
        'nokia': [r'nokia', r'nok\.', r'nsn'],
        'huawei': [r'huawei', r'hw_', r'hwa'],
        'samsung': [r'samsung', r'sam_'],
        'zte': [r'\bzte\b', r'zte_'],
        '3gpp': [r'3gpp', r'ts\s?\d+\.\d+', r'tr\s?\d+\.\d+'],
        'etsi': [r'\betsi\b'],
        'itu': [r'\bitu\b', r'itu-[rt]'],
        'gsma': [r'\bgsma\b'],
    }
    
    # Version patterns
    VERSION_PATTERNS = [
        r'[vV](\d+(?:\.\d+)*)',           # v1.2.3
        r'[Rr]el[ease]*[-_]?(\d+)',       # Rel-16, Release17
        r'[Ll](\d{2}[A-Z])',              # L16A, L17B
        r'(\d{4}[-_]\d{2})',              # 2023-06
        r'[Rr]ev[ision]*[-_]?(\d+)',      # Rev1, Revision2
    ]
    
    # Document type patterns
    DOC_TYPE_PATTERNS = {
        'specification': [r'spec', r'\bts\b', r'technical.specification'],
        'report': [r'report', r'\btr\b', r'technical.report'],
        'guide': [r'guide', r'manual', r'handbook'],
        'procedure': [r'procedure', r'process', r'workflow'],
        'overview': [r'overview', r'introduction', r'summary'],
        'reference': [r'reference', r'datasheet'],
    }
    
    # Topic/category patterns (telecom-specific)
    TOPIC_PATTERNS = {
        'security': [r'security', r'authentication', r'encryption', r'cipher', r'integrity'],
        'mobility': [r'mobility', r'handover', r'handoff', r'reselection'],
        'radio': [r'radio', r'rf', r'antenna', r'mimo', r'beamforming', r'phy'],
        'core': [r'core', r'amf', r'smf', r'upf', r'nrf', r'5gc'],
        'ran': [r'\bran\b', r'gnb', r'enb', r'base.station'],
        'protocol': [r'rrc', r'nas', r'ngap', r's1ap', r'x2ap', r'xnap'],
        'qos': [r'\bqos\b', r'quality.of.service', r'5qi', r'qci'],
    }

    @classmethod
    def extract_from_filename(cls, filepath: str) -> Dict:
        """Extract metadata from filename"""
        path = Path(filepath)
        filename = path.stem.lower()
        full_path = str(path).lower()
        
        metadata = {
            'filename': path.name,
            'extension': path.suffix.lower(),
            'vendor': cls._detect_vendor(filename, full_path),
            'version': cls._detect_version(filename),
            'doc_type': cls._detect_doc_type(filename),
            'topics': cls._detect_topics(filename),
            'release': cls._detect_3gpp_release(filename, full_path),
            'series': cls._detect_3gpp_series(full_path),
        }
        
        return metadata
    
    @classmethod
    def extract_from_content(cls, content: str, existing_meta: Dict = None) -> Dict:
        """Extract/enhance metadata from document content"""
        if existing_meta is None:
            existing_meta = {}
        
        content_lower = content[:5000].lower()  # Check first 5000 chars
        
        # Try to fill missing metadata from content
        if not existing_meta.get('vendor'):
            existing_meta['vendor'] = cls._detect_vendor(content_lower, '')
        
        if not existing_meta.get('topics'):
            existing_meta['topics'] = cls._detect_topics(content_lower)
        
        # Extract document title if present
        title_match = re.search(r'^#\s*(.+)$|^(.+)\n[=\-]{3,}', content[:1000], re.MULTILINE)
        if title_match:
            existing_meta['title'] = (title_match.group(1) or title_match.group(2)).strip()
        
        # Extract 3GPP spec number
        spec_match = re.search(r'(TS|TR)\s*(\d{2}\.\d{3})', content_lower)
        if spec_match:
            existing_meta['spec_number'] = f"{spec_match.group(1).upper()} {spec_match.group(2)}"
        
        return existing_meta
    
    @classmethod
    def _detect_vendor(cls, text: str, path: str) -> Optional[str]:
        """Detect vendor from text or path"""
        combined = f"{text} {path}"
        
        for vendor, patterns in cls.VENDOR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return vendor
        
        return None
    
    @classmethod
    def _detect_version(cls, text: str) -> Optional[str]:
        """Detect version from text"""
        for pattern in cls.VERSION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    @classmethod
    def _detect_doc_type(cls, text: str) -> Optional[str]:
        """Detect document type"""
        for doc_type, patterns in cls.DOC_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return doc_type
        return None
    
    @classmethod
    def _detect_topics(cls, text: str) -> List[str]:
        """Detect topics/categories"""
        topics = []
        for topic, patterns in cls.TOPIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    topics.append(topic)
                    break
        return topics
    
    @classmethod
    def _detect_3gpp_release(cls, text: str, path: str) -> Optional[int]:
        """Detect 3GPP release number"""
        combined = f"{text} {path}"
        
        # Pattern: Rel-16, Release-17, rel17
        match = re.search(r'rel[ease]*[-_]?(\d{1,2})', combined, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        return None
    
    @classmethod
    def _detect_3gpp_series(cls, path: str) -> Optional[str]:
        """Detect 3GPP series from path"""
        # Pattern: 23_series, 38-series, 33series
        match = re.search(r'(\d{2})[-_]?series', path, re.IGNORECASE)
        if match:
            return match.group(1)
        
        return None


def create_document_id(filepath: str, metadata: Dict) -> str:
    """Create a unique document identifier for retrieval"""
    parts = []
    
    if metadata.get('vendor'):
        parts.append(metadata['vendor'])
    
    if metadata.get('spec_number'):
        parts.append(metadata['spec_number'].replace(' ', ''))
    
    if metadata.get('version'):
        parts.append(f"v{metadata['version']}")
    
    if not parts:
        # Fallback to filename
        parts.append(Path(filepath).stem)
    
    return '_'.join(parts).lower()


def filter_by_metadata(documents: List[Dict], filters: Dict) -> List[Dict]:
    """
    Filter documents by metadata criteria
    
    Args:
        documents: List of documents with metadata
        filters: Dict with filter criteria, e.g.:
            {'vendor': 'ericsson', 'release': 17, 'topics': ['security']}
    
    Returns:
        Filtered list of documents
    """
    if not filters:
        return documents
    
    filtered = []
    for doc in documents:
        meta = doc.get('metadata', {})
        matches = True
        
        # Check each filter
        if 'vendor' in filters and meta.get('vendor') != filters['vendor']:
            matches = False
        
        if 'release' in filters and meta.get('release') != filters['release']:
            matches = False
        
        if 'topics' in filters:
            doc_topics = set(meta.get('topics', []))
            required_topics = set(filters['topics'])
            if not required_topics.intersection(doc_topics):
                matches = False
        
        if 'doc_type' in filters and meta.get('doc_type') != filters['doc_type']:
            matches = False
        
        if matches:
            filtered.append(doc)
    
    return filtered

