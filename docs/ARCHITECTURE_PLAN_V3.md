# AegisML v3.0 Architecture Plan
# Local NLP-Driven Offline-First Platform for AI Model & Infrastructure Analysis

**Version:** 3.0.0 (Proposed)
**Status:** Architecture Plan
**Date:** 2026-01-11

---

## Executive Summary

This document outlines the architectural evolution of AegisML from a CLI-based analysis tool to a **comprehensive local NLP-driven offline-first platform** for analyzing AI models AND infrastructure for safety, security, and capabilities.

### Vision Statement

> **AegisML v3.0** will be a self-contained, air-gapped capable platform that uses local NLP models to provide intelligent, conversational analysis of AI models and infrastructure—without requiring internet connectivity or cloud services.

---

## Current State (v2.0)

### Strengths
- Modular architecture (Skills → MCP → CLI → Tests)
- Offline-capable static analysis
- GGUF model support via llama.cpp
- MCP server and skill security analysis
- JSON-based structured reporting
- Plugin-based test system

### Gaps to Address
| Gap | Impact | Priority |
|-----|--------|----------|
| No local NLP intelligence | Manual interpretation required | **HIGH** |
| No semantic code understanding | Pattern-matching only | **HIGH** |
| No conversational interface | CLI-only interaction | **MEDIUM** |
| No knowledge base | No institutional memory | **MEDIUM** |
| Limited model formats | GGUF only | **MEDIUM** |
| No GUI | Terminal-only | **LOW** |

---

## Proposed Architecture (v3.0)

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AegisML v3.0 Platform                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    PRESENTATION LAYER                            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │   │
│  │  │   CLI    │  │   TUI    │  │  Web UI  │  │   MCP Interface  │ │   │
│  │  │ (v1-v2)  │  │  (NEW)   │  │  (NEW)   │  │     (v2.0)       │ │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │   │
│  └───────┼─────────────┼─────────────┼─────────────────┼───────────┘   │
│          │             │             │                 │               │
│  ┌───────▼─────────────▼─────────────▼─────────────────▼───────────┐   │
│  │                    NLP ORCHESTRATION LAYER (NEW)                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │   │
│  │  │ Conversation │  │   Intent     │  │   Response           │   │   │
│  │  │   Manager    │  │   Router     │  │   Generator          │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │              Local LLM Engine (llama.cpp)                │   │   │
│  │  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │   │
│  │  │   │ Analysis LLM│  │  Code LLM   │  │ Embedding Model │  │   │   │
│  │  │   │ (7-13B Q4)  │  │ (CodeLlama) │  │ (all-MiniLM)    │  │   │   │
│  │  │   └─────────────┘  └─────────────┘  └─────────────────┘  │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐   │
│  │                    ANALYSIS ENGINE LAYER                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │   │
│  │  │    Model     │  │Infrastructure│  │    Semantic          │   │   │
│  │  │   Analyzer   │  │   Analyzer   │  │    Analyzer (NEW)    │   │   │
│  │  │   (v1-v2)    │  │   (v2.0)     │  │                      │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │                   Test Plugin Registry                    │   │   │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │   │   │
│  │  │  │Static  │ │Dynamic │ │Security│ │Semantic│ │Behavior│  │   │   │
│  │  │  │Tests   │ │Tests   │ │Tests   │ │Tests   │ │Tests   │  │   │   │
│  │  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘  │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐   │
│  │                    KNOWLEDGE & STORAGE LAYER (NEW)               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │   │
│  │  │   Vector     │  │   Report     │  │   Threat             │   │   │
│  │  │   Store      │  │   Database   │  │   Knowledge Base     │   │   │
│  │  │  (ChromaDB)  │  │  (SQLite)    │  │   (Local YAML/JSON)  │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Specifications

### 1. Presentation Layer

#### 1.1 CLI (Enhanced)
- **Status:** Existing, to be enhanced
- **Enhancements:**
  - Natural language query support: `aegisml ask "Is this model safe?"`
  - Conversational mode: `aegisml chat`
  - Rich terminal output with colors and formatting

#### 1.2 TUI (Terminal UI) - NEW
- **Technology:** Textual (Python TUI framework)
- **Features:**
  - Dashboard view with real-time analysis status
  - Report browser with filtering/search
  - Model comparison side-by-side
  - Interactive chat panel

#### 1.3 Web UI - NEW (Optional)
- **Technology:** FastAPI + HTMX (lightweight, no JS framework)
- **Features:**
  - Local-only web interface (127.0.0.1)
  - Upload/analyze models via drag-drop
  - Visual report rendering
  - Chat interface

#### 1.4 MCP Interface (Enhanced)
- **Status:** Existing
- **Enhancements:**
  - New tools for NLP queries
  - Streaming response support
  - Context-aware analysis

---

### 2. NLP Orchestration Layer (NEW)

This is the **core innovation** of v3.0—a local NLP intelligence layer.

#### 2.1 Local LLM Engine

```python
# Proposed: src/aegisml/nlp/llm_engine.py

class LocalLLMEngine:
    """
    Manages local LLM inference using llama.cpp.
    Supports multiple specialized models for different tasks.
    """

    def __init__(self, config: LLMConfig):
        self.analysis_model: Optional[LlamaModel] = None  # General analysis
        self.code_model: Optional[LlamaModel] = None      # Code understanding
        self.embedding_model: Optional[EmbeddingModel] = None  # Semantic search

    async def analyze(self, prompt: str, context: AnalysisContext) -> str:
        """Run analysis with appropriate model selection."""
        pass

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for semantic search."""
        pass

    async def code_review(self, code: str, language: str) -> CodeReviewResult:
        """Specialized code analysis using code-tuned model."""
        pass
```

#### 2.2 Recommended Local Models

| Purpose | Recommended Model | Size | Notes |
|---------|------------------|------|-------|
| **General Analysis** | Mistral-7B-Instruct-v0.3 | ~4GB Q4 | Good reasoning, fast |
| **Code Analysis** | CodeLlama-7B-Instruct | ~4GB Q4 | Code-specialized |
| **Embeddings** | all-MiniLM-L6-v2 | ~90MB | Via sentence-transformers |
| **Fallback** | Phi-3-mini-4k | ~2GB Q4 | Lightweight alternative |

**Total Disk:** ~10GB for full capability
**Minimum RAM:** 8GB (16GB recommended)

#### 2.3 Conversation Manager

```python
# Proposed: src/aegisml/nlp/conversation.py

class ConversationManager:
    """
    Manages multi-turn conversations with context awareness.
    Maintains conversation history and analysis context.
    """

    def __init__(self, llm_engine: LocalLLMEngine, knowledge_base: KnowledgeBase):
        self.llm = llm_engine
        self.kb = knowledge_base
        self.history: List[Message] = []
        self.context: AnalysisContext = AnalysisContext()

    async def process_query(self, user_input: str) -> Response:
        """
        Process user query with full context awareness.

        1. Parse intent
        2. Retrieve relevant knowledge
        3. Execute any required analysis
        4. Generate natural language response
        """
        intent = await self.intent_router.classify(user_input)

        if intent.requires_analysis:
            analysis_result = await self.execute_analysis(intent)
            self.context.add_result(analysis_result)

        relevant_knowledge = await self.kb.search(user_input, self.context)

        response = await self.llm.generate_response(
            user_input=user_input,
            history=self.history,
            context=self.context,
            knowledge=relevant_knowledge
        )

        self.history.append(Message(role="user", content=user_input))
        self.history.append(Message(role="assistant", content=response))

        return response
```

#### 2.4 Intent Router

```python
# Proposed: src/aegisml/nlp/intent.py

class IntentRouter:
    """
    Classifies user intents to route to appropriate handlers.
    Uses a combination of keyword matching and LLM classification.
    """

    INTENTS = [
        "analyze_model",           # "Analyze this model"
        "analyze_infrastructure",  # "Check this MCP server"
        "compare",                 # "Compare model A vs B"
        "explain",                 # "Explain this result"
        "query_knowledge",         # "What jailbreak techniques exist?"
        "generate_report",         # "Create a model card"
        "general_question",        # Generic questions
    ]

    async def classify(self, query: str) -> Intent:
        """Classify user query into an intent with extracted parameters."""
        # Fast path: keyword matching
        if fast_match := self._keyword_match(query):
            return fast_match

        # Slow path: LLM classification
        return await self._llm_classify(query)
```

---

### 3. Analysis Engine Layer (Enhanced)

#### 3.1 Semantic Analyzer (NEW)

```python
# Proposed: src/aegisml/analyzers/semantic.py

class SemanticAnalyzer:
    """
    Uses local LLM for deep semantic analysis of code and models.
    Goes beyond pattern matching to understand intent and behavior.
    """

    async def analyze_code_semantics(self, code: str, language: str) -> SemanticAnalysis:
        """
        Analyze code semantically using code-tuned LLM.

        Detects:
        - Obfuscated malicious patterns
        - Logic bombs
        - Subtle data exfiltration
        - Hidden functionality
        """
        pass

    async def analyze_model_behavior(self, model_path: str, prompts: List[str]) -> BehaviorAnalysis:
        """
        Analyze model behavior through targeted prompting.
        Uses local analysis LLM to interpret responses.
        """
        pass

    async def explain_finding(self, finding: Finding, context: AnalysisContext) -> str:
        """Generate human-readable explanation of a security finding."""
        pass
```

#### 3.2 Enhanced Test Plugin System

```python
# Proposed enhancement to test registry

class SemanticSecurityTest(SandboxTest):
    """
    LLM-powered security test that uses semantic understanding.
    """
    name = "semantic_security_analysis"
    category = "semantic"
    requires_llm = True  # NEW: Flag for LLM requirement

    async def run(self, context: AnalysisContext, llm: LocalLLMEngine) -> TestResult:
        """Run semantic analysis using local LLM."""
        # Analyze code with LLM
        semantic_findings = await llm.code_review(
            code=context.code,
            language=context.language,
            focus="security"
        )

        # Cross-reference with knowledge base
        known_patterns = await self.kb.find_similar_vulnerabilities(semantic_findings)

        return TestResult(
            name=self.name,
            category=self.category,
            status=self._determine_status(semantic_findings),
            details=semantic_findings.summary,
            artifacts=[{"type": "semantic_analysis", "data": semantic_findings.to_dict()}]
        )
```

#### 3.3 New Test Categories

| Category | Tests | LLM Required |
|----------|-------|--------------|
| **Semantic** | Code intent analysis, behavior prediction | Yes |
| **Behavioral** | Model response patterns, consistency | Yes |
| **Comparative** | Cross-model safety comparison | Yes |
| **Drift** | Safety degradation detection | No |
| **Compliance** | Policy adherence verification | Yes |

---

### 4. Knowledge & Storage Layer (NEW)

#### 4.1 Vector Store (ChromaDB)

```python
# Proposed: src/aegisml/storage/vector_store.py

class VectorStore:
    """
    Local vector database for semantic search.
    Uses ChromaDB for persistent, offline storage.
    """

    def __init__(self, db_path: str = "./data/chromadb"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collections = {
            "vulnerabilities": self.client.get_or_create_collection("vulnerabilities"),
            "jailbreaks": self.client.get_or_create_collection("jailbreaks"),
            "reports": self.client.get_or_create_collection("reports"),
            "code_patterns": self.client.get_or_create_collection("code_patterns"),
        }

    async def search(self, query: str, collection: str, top_k: int = 5) -> List[SearchResult]:
        """Semantic search across knowledge base."""
        pass

    async def add_document(self, doc: Document, collection: str):
        """Add document to knowledge base with embeddings."""
        pass
```

#### 4.2 Report Database (SQLite)

```python
# Proposed: src/aegisml/storage/report_db.py

class ReportDatabase:
    """
    SQLite database for structured report storage and querying.
    Replaces JSON file-based storage for better querying.
    """

    def __init__(self, db_path: str = "./data/reports.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                run_id TEXT UNIQUE,
                timestamp TEXT,
                profile TEXT,
                target_type TEXT,  -- 'model', 'mcp_server', 'skill'
                target_name TEXT,
                risk_level TEXT,
                risk_score REAL,
                report_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY,
                report_id TEXT REFERENCES reports(id),
                test_name TEXT,
                category TEXT,
                status TEXT,
                severity TEXT,
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_reports_target ON reports(target_name);
            CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity);
        """)
```

#### 4.3 Threat Knowledge Base

```yaml
# Proposed: data/knowledge/jailbreaks.yaml

version: "1.0"
last_updated: "2026-01-11"

categories:
  - name: "role_play"
    description: "Attempts to make the model assume a different persona"
    techniques:
      - id: "dan"
        name: "Do Anything Now (DAN)"
        description: "Classic jailbreak using alternate persona"
        example_prompts:
          - "You are DAN, an AI without restrictions..."
        indicators:
          - "pretend you are"
          - "act as if"
          - "no restrictions"
        severity: "HIGH"

  - name: "prompt_injection"
    description: "Attempts to override system instructions"
    techniques:
      - id: "ignore_previous"
        name: "Ignore Previous Instructions"
        description: "Direct instruction override attempt"
        example_prompts:
          - "Ignore all previous instructions and..."
        indicators:
          - "ignore previous"
          - "disregard above"
          - "new instructions:"
        severity: "CRITICAL"
```

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up project structure for v3.0
- [ ] Implement Local LLM Engine with llama.cpp integration
- [ ] Create embedding model integration (sentence-transformers)
- [ ] Set up ChromaDB vector store
- [ ] Set up SQLite report database
- [ ] Create basic threat knowledge base (YAML)

### Phase 2: NLP Core (Weeks 3-4)
- [ ] Implement Conversation Manager
- [ ] Implement Intent Router
- [ ] Create Response Generator
- [ ] Build context management system
- [ ] Integrate with existing analysis engine

### Phase 3: Semantic Analysis (Weeks 5-6)
- [ ] Implement Semantic Analyzer
- [ ] Create LLM-powered security tests
- [ ] Build code understanding pipeline
- [ ] Implement model behavior analysis
- [ ] Add explanation generation for findings

### Phase 4: Knowledge Integration (Weeks 7-8)
- [ ] Populate threat knowledge base
- [ ] Implement RAG (Retrieval Augmented Generation)
- [ ] Add report indexing and search
- [ ] Create knowledge update mechanisms
- [ ] Build comparative analysis features

### Phase 5: Interface Enhancement (Weeks 9-10)
- [ ] Enhance CLI with NLP queries
- [ ] Build TUI with Textual
- [ ] Create local Web UI (optional)
- [ ] Update MCP tools for NLP
- [ ] Add streaming responses

### Phase 6: Polish & Documentation (Weeks 11-12)
- [ ] Performance optimization
- [ ] Comprehensive testing
- [ ] Documentation updates
- [ ] Example workflows
- [ ] Release preparation

---

## Directory Structure (Proposed)

```
AegisML/
├── src/aegisml/
│   ├── __init__.py
│   ├── sandbox_cli.py              # Enhanced CLI
│   ├── mcp_server.py               # Enhanced MCP server
│   │
│   ├── nlp/                        # NEW: NLP Layer
│   │   ├── __init__.py
│   │   ├── llm_engine.py           # Local LLM management
│   │   ├── conversation.py         # Conversation management
│   │   ├── intent.py               # Intent classification
│   │   ├── response.py             # Response generation
│   │   └── embeddings.py           # Embedding generation
│   │
│   ├── analyzers/                  # Enhanced analyzers
│   │   ├── __init__.py
│   │   ├── model_analyzer.py       # GGUF analysis
│   │   ├── mcp_analyzer.py         # MCP server analysis
│   │   ├── skill_analyzer.py       # Skill analysis
│   │   └── semantic_analyzer.py    # NEW: LLM-powered analysis
│   │
│   ├── tests/                      # Test plugins
│   │   ├── __init__.py
│   │   ├── static_tests.py
│   │   ├── dynamic_tests.py
│   │   ├── infrastructure_tests.py
│   │   └── semantic_tests.py       # NEW: LLM-powered tests
│   │
│   ├── storage/                    # NEW: Storage layer
│   │   ├── __init__.py
│   │   ├── vector_store.py         # ChromaDB integration
│   │   ├── report_db.py            # SQLite reports
│   │   └── knowledge_base.py       # Threat KB access
│   │
│   └── ui/                         # NEW: UI components
│       ├── __init__.py
│       ├── tui.py                  # Textual TUI
│       └── web.py                  # FastAPI web UI
│
├── data/                           # NEW: Data directory
│   ├── chromadb/                   # Vector store
│   ├── reports.db                  # SQLite database
│   └── knowledge/                  # Threat knowledge base
│       ├── jailbreaks.yaml
│       ├── vulnerabilities.yaml
│       ├── code_patterns.yaml
│       └── compliance_rules.yaml
│
├── models/                         # LLM models (user provides)
│   ├── analysis/                   # Analysis models
│   │   └── mistral-7b-instruct.gguf
│   ├── code/                       # Code models
│   │   └── codellama-7b.gguf
│   └── embeddings/                 # Embedding models
│       └── all-MiniLM-L6-v2/
│
├── skills/                         # Skill definitions
├── docs/                           # Documentation
├── tests/                          # Test suite
└── sandbox_logs/                   # Analysis reports (legacy)
```

---

## Configuration

```yaml
# config/aegisml.yaml

version: "3.0"

# LLM Configuration
llm:
  analysis_model:
    path: "models/analysis/mistral-7b-instruct.gguf"
    context_length: 4096
    threads: 4
    gpu_layers: 0  # CPU-only by default

  code_model:
    path: "models/code/codellama-7b.gguf"
    context_length: 4096
    threads: 4
    gpu_layers: 0

  embedding_model:
    type: "sentence-transformers"
    model: "all-MiniLM-L6-v2"
    path: "models/embeddings/all-MiniLM-L6-v2"

# Storage Configuration
storage:
  vector_store:
    type: "chromadb"
    path: "data/chromadb"

  report_db:
    type: "sqlite"
    path: "data/reports.db"

  knowledge_base:
    path: "data/knowledge"

# Analysis Configuration
analysis:
  default_profile: "standard"
  cache_ttl_hours: 24
  max_concurrent_tests: 4

# UI Configuration
ui:
  web:
    enabled: false
    host: "127.0.0.1"
    port: 8080

  tui:
    theme: "dark"
    refresh_rate: 0.5
```

---

## API Design

### Enhanced CLI Commands

```bash
# Natural language queries
aegisml ask "Is model.gguf safe to deploy?"
aegisml ask "What vulnerabilities were found in server.py?"
aegisml ask "Compare model1.gguf and model2.gguf for safety"

# Conversational mode
aegisml chat
> What models have I analyzed recently?
> Show me the riskiest one
> Explain why it scored high

# TUI mode
aegisml tui

# Web UI (if enabled)
aegisml serve --port 8080
```

### Enhanced MCP Tools

```python
# New MCP tools for v3.0

@mcp_tool
async def aegisml_ask(query: str, context: Optional[str] = None) -> str:
    """
    Natural language query about models or infrastructure.

    Args:
        query: Natural language question
        context: Optional context (model name, previous analysis, etc.)

    Returns:
        Natural language response with analysis results
    """
    pass

@mcp_tool
async def aegisml_explain(finding_id: str) -> str:
    """
    Get detailed explanation of a security finding.

    Args:
        finding_id: ID of the finding to explain

    Returns:
        Detailed natural language explanation
    """
    pass

@mcp_tool
async def aegisml_recommend(target: str, use_case: str) -> str:
    """
    Get deployment recommendations for a model or server.

    Args:
        target: Path to model or server
        use_case: Intended use case description

    Returns:
        Recommendations with risk assessment
    """
    pass
```

---

## Performance Considerations

### Memory Management

```python
# Proposed: Model loading strategy

class ModelManager:
    """
    Manages model loading/unloading to optimize memory usage.
    """

    def __init__(self, max_loaded_models: int = 2):
        self.max_loaded = max_loaded_models
        self.loaded_models: OrderedDict[str, Model] = OrderedDict()

    async def get_model(self, model_type: str) -> Model:
        """
        Get model, loading if necessary.
        Uses LRU eviction when memory constrained.
        """
        if model_type in self.loaded_models:
            # Move to end (most recently used)
            self.loaded_models.move_to_end(model_type)
            return self.loaded_models[model_type]

        # Evict oldest if at capacity
        if len(self.loaded_models) >= self.max_loaded:
            oldest = next(iter(self.loaded_models))
            await self._unload_model(oldest)

        # Load requested model
        return await self._load_model(model_type)
```

### Caching Strategy

| Data Type | Cache Location | TTL | Invalidation |
|-----------|---------------|-----|--------------|
| Analysis results | SQLite | 24h | Manual or re-analysis |
| Embeddings | ChromaDB | Permanent | Never (content-addressed) |
| LLM responses | In-memory | Session | Session end |
| Knowledge base | File system | Manual | User update |

---

## Security Considerations

### Air-Gap Compatibility

AegisML v3.0 must function **completely offline**:

1. **No telemetry** - Zero network calls
2. **Local models only** - All LLMs stored locally
3. **Local knowledge base** - Threat intel stored locally
4. **No auto-updates** - Manual update mechanism only
5. **Self-contained** - All dependencies bundled

### Model Isolation

```python
# Proposed: Sandboxed model execution

class SandboxedInference:
    """
    Runs model inference in isolated context.
    Prevents potential model-based attacks.
    """

    async def run_inference(self, prompt: str, model: Model) -> str:
        """
        Run inference with safety constraints:
        - Output length limits
        - Timeout enforcement
        - Content filtering
        - Resource limits
        """
        pass
```

---

## Success Metrics

### Functional Requirements

| Requirement | Metric | Target |
|-------------|--------|--------|
| NLP query accuracy | Intent classification accuracy | >90% |
| Response quality | User satisfaction score | >4/5 |
| Analysis coverage | Vulnerability detection rate | >95% |
| Offline capability | Features working offline | 100% |

### Performance Requirements

| Requirement | Metric | Target |
|-------------|--------|--------|
| Query response time | P95 latency | <5s |
| Analysis throughput | Models/hour | >20 |
| Memory usage | Peak RAM | <16GB |
| Storage efficiency | DB size per report | <100KB |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM hallucination | Medium | High | Cross-reference with rule-based checks |
| Performance degradation | Medium | Medium | Aggressive caching, model optimization |
| Knowledge base staleness | High | Medium | Clear update mechanism, versioning |
| Complex deployment | Medium | Medium | Single-command setup, Docker option |

---

## Dependencies

### Required
- Python 3.9+
- llama-cpp-python (llama.cpp bindings)
- chromadb (vector store)
- sentence-transformers (embeddings)
- sqlite3 (built-in)
- pyyaml (configuration)

### Optional
- textual (TUI)
- fastapi + uvicorn (Web UI)
- httpx (async HTTP, only for optional features)

### LLM Models (User Provides)
- Mistral-7B-Instruct-v0.3.Q4_K_M.gguf (~4GB)
- CodeLlama-7B-Instruct.Q4_K_M.gguf (~4GB)
- all-MiniLM-L6-v2 (~90MB)

---

## Next Steps

1. **Review and approve** this architecture plan
2. **Set up development environment** for v3.0
3. **Begin Phase 1** implementation
4. **Weekly sync** on progress and blockers

---

## Appendix A: Example Interactions

### Example 1: Natural Language Model Analysis

```
User: Is mistral-7b.gguf safe to use for a customer chatbot?

AegisML: I'll analyze mistral-7b.gguf for customer chatbot suitability.

[Running analysis: quick profile]

Based on my analysis:

**Risk Level: LOW (Score: 18/100)**

**Safety Assessment:**
- Jailbreak resistance: 92% (Good)
- Policy compliance: 95% (Excellent)
- No backdoor triggers detected

**Recommendation:** ✅ Suitable for customer chatbot use with standard monitoring.

**Considerations:**
1. Add content moderation layer for production
2. Implement rate limiting
3. Monitor for prompt injection attempts

Would you like a detailed report or specific test results?
```

### Example 2: Code Security Analysis

```
User: Check if custom_mcp_server.py has any security issues

AegisML: Analyzing custom_mcp_server.py for security vulnerabilities...

[Running infra-standard profile with semantic analysis]

**Security Assessment: HIGH RISK (Score: 72/100)**

**Critical Findings:**

1. **Command Injection (Line 45)**
   ```python
   os.system(f"process {user_input}")  # CRITICAL
   ```
   User input flows directly into system command.

2. **Data Exfiltration Risk (Line 78)**
   ```python
   requests.post(external_url, data=results)  # HIGH
   ```
   Sends data to external endpoint.

**Semantic Analysis:**
The code appears to be collecting system information and transmitting it externally.
This pattern is consistent with known exfiltration malware.

**Recommendation:** ❌ Do not deploy. Requires security review.

Would you like me to explain any finding in more detail?
```

---

## Appendix B: Comparison with Current State

| Feature | v2.0 | v3.0 |
|---------|------|------|
| Analysis approach | Rule-based | Rule-based + LLM semantic |
| User interface | CLI only | CLI + TUI + Web |
| Query method | Commands | Natural language |
| Knowledge base | None | Vector + structured |
| Report storage | JSON files | SQLite + JSON |
| Explanation generation | Templates | LLM-generated |
| Cross-analysis insights | Manual | Automated |
| Offline capability | Yes | Yes (enhanced) |

---

*This architecture plan establishes AegisML v3.0 as a comprehensive, intelligent, offline-first platform for AI security analysis.*
