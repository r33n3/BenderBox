"""
Threat Knowledge Base for BenderBox

Provides structured threat knowledge with:
- YAML-based knowledge files
- Semantic search integration
- Categories: jailbreaks, vulnerabilities, code patterns
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from benderbox.config import get_config
from benderbox.storage.vector_store import Document, SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """Base class for knowledge entries."""

    id: str
    name: str
    description: str
    category: str
    severity: str  # critical, high, medium, low, info
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JailbreakTechnique(KnowledgeEntry):
    """Jailbreak technique knowledge entry."""

    example_prompts: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)


@dataclass
class VulnerabilityPattern(KnowledgeEntry):
    """Vulnerability pattern knowledge entry."""

    code_patterns: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    remediation: List[str] = field(default_factory=list)
    cwe_ids: List[str] = field(default_factory=list)


@dataclass
class CodePattern(KnowledgeEntry):
    """Suspicious code pattern knowledge entry."""

    language: str = "any"
    patterns: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    false_positive_hints: List[str] = field(default_factory=list)


class KnowledgeBase:
    """
    Threat knowledge base with semantic search.

    Loads knowledge from YAML files and indexes into vector store
    for semantic search capabilities.
    """

    KNOWLEDGE_CATEGORIES = ["jailbreaks", "vulnerabilities", "code_patterns"]

    def __init__(
        self,
        kb_path: Optional[str] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        """
        Initialize KnowledgeBase.

        Args:
            kb_path: Path to knowledge YAML files directory.
            vector_store: Vector store for semantic search.
        """
        config = get_config().storage

        self.kb_path = Path(kb_path or config.knowledge_path)
        self._vector_store = vector_store
        self._loaded = False

        # In-memory knowledge cache
        self._jailbreaks: Dict[str, JailbreakTechnique] = {}
        self._vulnerabilities: Dict[str, VulnerabilityPattern] = {}
        self._code_patterns: Dict[str, CodePattern] = {}

    def _get_vector_store(self) -> VectorStore:
        """Get or create vector store."""
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    async def load_knowledge(self) -> None:
        """Load all knowledge from YAML files."""
        if self._loaded:
            return

        logger.info(f"Loading knowledge from: {self.kb_path}")

        # Load each category
        await self._load_jailbreaks()
        await self._load_vulnerabilities()
        await self._load_code_patterns()

        self._loaded = True

        total = len(self._jailbreaks) + len(self._vulnerabilities) + len(self._code_patterns)
        logger.info(f"Loaded {total} knowledge entries")

    async def _load_jailbreaks(self) -> None:
        """Load jailbreak techniques from YAML."""
        jailbreaks_file = self.kb_path / "jailbreaks.yaml"
        if not jailbreaks_file.exists():
            logger.warning(f"Jailbreaks file not found: {jailbreaks_file}")
            return

        with open(jailbreaks_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        techniques = data.get("techniques", [])
        documents = []

        for tech in techniques:
            jailbreak = JailbreakTechnique(
                id=tech.get("id", ""),
                name=tech.get("name", ""),
                description=tech.get("description", ""),
                category=tech.get("category", "jailbreak"),
                severity=tech.get("severity", "medium"),
                example_prompts=tech.get("example_prompts", []),
                indicators=tech.get("indicators", []),
                mitigations=tech.get("mitigations", []),
                metadata=tech.get("metadata", {}),
            )
            self._jailbreaks[jailbreak.id] = jailbreak

            # Create document for vector store
            content = f"{jailbreak.name}\n{jailbreak.description}\n"
            content += " ".join(jailbreak.indicators)
            documents.append(
                Document.create(
                    content=content,
                    metadata={
                        "type": "jailbreak",
                        "id": jailbreak.id,
                        "name": jailbreak.name,
                        "severity": jailbreak.severity,
                        "category": jailbreak.category,
                    },
                    doc_id=f"jailbreak_{jailbreak.id}",
                )
            )

        # Index in vector store
        if documents:
            vector_store = self._get_vector_store()
            await vector_store.add_documents(documents, collection="jailbreaks")

        logger.debug(f"Loaded {len(self._jailbreaks)} jailbreak techniques")

    async def _load_vulnerabilities(self) -> None:
        """Load vulnerability patterns from YAML."""
        vuln_file = self.kb_path / "vulnerabilities.yaml"
        if not vuln_file.exists():
            logger.warning(f"Vulnerabilities file not found: {vuln_file}")
            return

        with open(vuln_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        patterns = data.get("patterns", [])
        documents = []

        for pattern in patterns:
            vuln = VulnerabilityPattern(
                id=pattern.get("id", ""),
                name=pattern.get("name", ""),
                description=pattern.get("description", ""),
                category=pattern.get("category", "vulnerability"),
                severity=pattern.get("severity", "medium"),
                code_patterns=pattern.get("code_patterns", []),
                indicators=pattern.get("indicators", []),
                remediation=pattern.get("remediation", []),
                cwe_ids=pattern.get("cwe_ids", []),
                metadata=pattern.get("metadata", {}),
            )
            self._vulnerabilities[vuln.id] = vuln

            # Create document for vector store
            content = f"{vuln.name}\n{vuln.description}\n"
            content += " ".join(vuln.indicators)
            content += " ".join(vuln.code_patterns)
            documents.append(
                Document.create(
                    content=content,
                    metadata={
                        "type": "vulnerability",
                        "id": vuln.id,
                        "name": vuln.name,
                        "severity": vuln.severity,
                        "category": vuln.category,
                        "cwe_ids": ",".join(vuln.cwe_ids),
                    },
                    doc_id=f"vuln_{vuln.id}",
                )
            )

        if documents:
            vector_store = self._get_vector_store()
            await vector_store.add_documents(documents, collection="vulnerabilities")

        logger.debug(f"Loaded {len(self._vulnerabilities)} vulnerability patterns")

    async def _load_code_patterns(self) -> None:
        """Load code patterns from YAML."""
        patterns_file = self.kb_path / "code_patterns.yaml"
        if not patterns_file.exists():
            logger.warning(f"Code patterns file not found: {patterns_file}")
            return

        with open(patterns_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        patterns = data.get("patterns", [])
        documents = []

        for pattern in patterns:
            code_pattern = CodePattern(
                id=pattern.get("id", ""),
                name=pattern.get("name", ""),
                description=pattern.get("description", ""),
                category=pattern.get("category", "code_pattern"),
                severity=pattern.get("severity", "medium"),
                language=pattern.get("language", "any"),
                patterns=pattern.get("patterns", []),
                examples=pattern.get("examples", []),
                false_positive_hints=pattern.get("false_positive_hints", []),
                metadata=pattern.get("metadata", {}),
            )
            self._code_patterns[code_pattern.id] = code_pattern

            # Create document for vector store
            content = f"{code_pattern.name}\n{code_pattern.description}\n"
            content += " ".join(code_pattern.patterns)
            documents.append(
                Document.create(
                    content=content,
                    metadata={
                        "type": "code_pattern",
                        "id": code_pattern.id,
                        "name": code_pattern.name,
                        "severity": code_pattern.severity,
                        "language": code_pattern.language,
                    },
                    doc_id=f"code_{code_pattern.id}",
                )
            )

        if documents:
            vector_store = self._get_vector_store()
            await vector_store.add_documents(documents, collection="code_patterns")

        logger.debug(f"Loaded {len(self._code_patterns)} code patterns")

    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> List[KnowledgeEntry]:
        """
        Search knowledge base semantically.

        Args:
            query: Search query.
            category: Optional category filter (jailbreaks, vulnerabilities, code_patterns).
            top_k: Number of results.

        Returns:
            List of matching knowledge entries.
        """
        await self.load_knowledge()

        vector_store = self._get_vector_store()
        results: List[KnowledgeEntry] = []

        # Determine collections to search
        if category:
            collections = [category]
        else:
            collections = self.KNOWLEDGE_CATEGORIES

        for collection in collections:
            search_results = await vector_store.search(
                query=query,
                collection=collection,
                top_k=top_k,
            )

            for result in search_results:
                entry = self._get_entry_by_id(
                    result.document.metadata.get("type", ""),
                    result.document.metadata.get("id", ""),
                )
                if entry and entry not in results:
                    results.append(entry)

        return results[:top_k]

    def _get_entry_by_id(
        self, entry_type: str, entry_id: str
    ) -> Optional[KnowledgeEntry]:
        """Get knowledge entry by type and ID."""
        if entry_type == "jailbreak":
            return self._jailbreaks.get(entry_id)
        elif entry_type == "vulnerability":
            return self._vulnerabilities.get(entry_id)
        elif entry_type == "code_pattern":
            return self._code_patterns.get(entry_id)
        return None

    async def get_jailbreak_techniques(
        self, category: Optional[str] = None
    ) -> List[JailbreakTechnique]:
        """
        Get all jailbreak techniques.

        Args:
            category: Optional category filter.

        Returns:
            List of jailbreak techniques.
        """
        await self.load_knowledge()

        techniques = list(self._jailbreaks.values())
        if category:
            techniques = [t for t in techniques if t.category == category]
        return techniques

    async def get_vulnerability_patterns(
        self, category: Optional[str] = None
    ) -> List[VulnerabilityPattern]:
        """
        Get all vulnerability patterns.

        Args:
            category: Optional category filter.

        Returns:
            List of vulnerability patterns.
        """
        await self.load_knowledge()

        patterns = list(self._vulnerabilities.values())
        if category:
            patterns = [p for p in patterns if p.category == category]
        return patterns

    async def get_code_patterns(
        self, language: Optional[str] = None
    ) -> List[CodePattern]:
        """
        Get all code patterns.

        Args:
            language: Optional language filter.

        Returns:
            List of code patterns.
        """
        await self.load_knowledge()

        patterns = list(self._code_patterns.values())
        if language:
            patterns = [
                p for p in patterns if p.language in (language, "any")
            ]
        return patterns

    async def add_custom_knowledge(
        self, entry: KnowledgeEntry, category: str
    ) -> str:
        """
        Add custom knowledge entry.

        Args:
            entry: Knowledge entry to add.
            category: Category (jailbreaks, vulnerabilities, code_patterns).

        Returns:
            Entry ID.
        """
        await self.load_knowledge()

        # Add to appropriate cache
        if category == "jailbreaks" and isinstance(entry, JailbreakTechnique):
            self._jailbreaks[entry.id] = entry
        elif category == "vulnerabilities" and isinstance(entry, VulnerabilityPattern):
            self._vulnerabilities[entry.id] = entry
        elif category == "code_patterns" and isinstance(entry, CodePattern):
            self._code_patterns[entry.id] = entry
        else:
            raise ValueError(f"Invalid category or entry type: {category}")

        # Index in vector store
        vector_store = self._get_vector_store()
        doc = Document.create(
            content=f"{entry.name}\n{entry.description}",
            metadata={
                "type": category.rstrip("s"),  # Remove plural
                "id": entry.id,
                "name": entry.name,
                "severity": entry.severity,
            },
            doc_id=f"{category}_{entry.id}",
        )
        await vector_store.add_document(doc, collection=category)

        return entry.id

    async def refresh(self) -> None:
        """Reload knowledge from files."""
        self._loaded = False
        self._jailbreaks.clear()
        self._vulnerabilities.clear()
        self._code_patterns.clear()

        # Clear vector store collections
        vector_store = self._get_vector_store()
        for category in self.KNOWLEDGE_CATEGORIES:
            await vector_store.clear_collection(category)

        await self.load_knowledge()

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "kb_path": str(self.kb_path),
            "loaded": self._loaded,
            "jailbreaks_count": len(self._jailbreaks),
            "vulnerabilities_count": len(self._vulnerabilities),
            "code_patterns_count": len(self._code_patterns),
            "total_entries": (
                len(self._jailbreaks)
                + len(self._vulnerabilities)
                + len(self._code_patterns)
            ),
        }


class KnowledgeBaseError(Exception):
    """Exception for knowledge base errors."""

    pass
