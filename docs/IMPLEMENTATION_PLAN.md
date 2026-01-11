# AegisML v3.0 Implementation Plan
# Executable Steps for Claude Code

**Version:** 3.0.0
**Status:** Implementation Ready
**Created:** 2026-01-11

---

## Overview

This document provides **step-by-step executable instructions** for implementing AegisML v3.0. Each phase contains discrete tasks that can be executed sequentially by Claude Code.

---

## Pre-Implementation Checklist

Before starting, verify:
- [ ] Python 3.9+ installed
- [ ] Git repository initialized
- [ ] Current v2.0 code is stable on main branch

---

# PHASE 1: Foundation
**Goal:** Set up project structure, LLM engine, and storage layer

---

## Step 1.1: Create New Directory Structure

```
TASK: Create the v3.0 directory structure

ACTIONS:
1. Create the following directories:
   - src/aegisml/nlp/
   - src/aegisml/analyzers/
   - src/aegisml/storage/
   - src/aegisml/ui/
   - src/aegisml/tests/
   - data/knowledge/
   - data/chromadb/
   - models/analysis/
   - models/code/
   - models/embeddings/
   - config/

2. Create __init__.py files in each Python package directory

VERIFICATION:
- All directories exist
- __init__.py files present in Python packages
```

---

## Step 1.2: Update pyproject.toml with New Dependencies

```
TASK: Update project dependencies for v3.0

FILE: pyproject.toml

ACTIONS:
1. Update version to "3.0.0-alpha"
2. Add new dependencies:
   - llama-cpp-python>=0.2.0
   - chromadb>=0.4.0
   - sentence-transformers>=2.2.0
   - pyyaml>=6.0
   - aiosqlite>=0.19.0

3. Add optional dependencies:
   - textual>=0.40.0 (for TUI)
   - fastapi>=0.100.0 (for Web UI)
   - uvicorn>=0.23.0 (for Web UI)

4. Add new entry points:
   - aegisml-chat = "aegisml.nlp.conversation:main"
   - aegisml-tui = "aegisml.ui.tui:main"

VERIFICATION:
- pip install -e . succeeds
- All imports available
```

---

## Step 1.3: Create Configuration System

```
TASK: Create configuration management system

FILE: src/aegisml/config.py

ACTIONS:
1. Create Config dataclass with sections:
   - LLMConfig (model paths, context length, threads, gpu_layers)
   - StorageConfig (vector store path, db path, knowledge path)
   - AnalysisConfig (default profile, cache TTL, max concurrent)
   - UIConfig (web enabled, host, port, TUI theme)

2. Create load_config() function:
   - Load from config/aegisml.yaml if exists
   - Fall back to defaults
   - Support environment variable overrides

3. Create default config template

FILE: config/aegisml.yaml

ACTIONS:
1. Create default configuration file with all options documented

VERIFICATION:
- from aegisml.config import load_config works
- Config loads defaults when no file exists
- Config loads from YAML when file exists
```

---

## Step 1.4: Implement Local LLM Engine

```
TASK: Create the local LLM engine using llama-cpp-python

FILE: src/aegisml/nlp/__init__.py

ACTIONS:
1. Export main classes: LocalLLMEngine, ConversationManager, IntentRouter

FILE: src/aegisml/nlp/llm_engine.py

ACTIONS:
1. Create LlamaModel wrapper class:
   - __init__(model_path, context_length, threads, gpu_layers)
   - generate(prompt, max_tokens, temperature, stop) -> str
   - tokenize(text) -> List[int]
   - detokenize(tokens) -> str
   - unload() -> None

2. Create LocalLLMEngine class:
   - __init__(config: LLMConfig)
   - _analysis_model: Optional[LlamaModel]
   - _code_model: Optional[LlamaModel]
   - _loaded_models: Dict[str, LlamaModel]

3. Implement methods:
   - async def load_model(model_type: str) -> None
   - async def unload_model(model_type: str) -> None
   - async def generate(prompt: str, model_type: str = "analysis") -> str
   - async def is_model_loaded(model_type: str) -> bool
   - def get_model_info() -> Dict[str, Any]

4. Implement lazy loading:
   - Models load on first use
   - LRU eviction when memory constrained (max 2 models)

5. Handle missing models gracefully:
   - Return informative error if model file not found
   - Suggest download instructions

VERIFICATION:
- Engine initializes without models present
- Engine loads model when generate() called
- Engine handles missing model files gracefully
```

---

## Step 1.5: Implement Embedding Model Integration

```
TASK: Create embedding generation using sentence-transformers

FILE: src/aegisml/nlp/embeddings.py

ACTIONS:
1. Create EmbeddingModel class:
   - __init__(model_name_or_path: str)
   - embed(text: str) -> List[float]
   - embed_batch(texts: List[str]) -> List[List[float]]
   - dimension -> int

2. Handle offline mode:
   - Load from local path if provided
   - Cache model after first download
   - Support air-gapped deployment

3. Create EmbeddingCache class:
   - Cache embeddings to avoid recomputation
   - Use content hash as key
   - Persist to disk (optional)

VERIFICATION:
- EmbeddingModel loads successfully
- embed() returns correct dimension vector
- embed_batch() handles multiple texts
```

---

## Step 1.6: Implement ChromaDB Vector Store

```
TASK: Create vector store using ChromaDB

FILE: src/aegisml/storage/__init__.py

ACTIONS:
1. Export main classes: VectorStore, ReportDatabase, KnowledgeBase

FILE: src/aegisml/storage/vector_store.py

ACTIONS:
1. Create VectorStore class:
   - __init__(db_path: str, embedding_model: EmbeddingModel)
   - Collections: vulnerabilities, jailbreaks, reports, code_patterns

2. Implement methods:
   - async def add_document(doc: Document, collection: str) -> str
   - async def search(query: str, collection: str, top_k: int = 5) -> List[SearchResult]
   - async def delete_document(doc_id: str, collection: str) -> bool
   - async def get_document(doc_id: str, collection: str) -> Optional[Document]
   - async def list_collections() -> List[str]
   - async def get_collection_stats(collection: str) -> Dict[str, Any]

3. Create Document dataclass:
   - id: str
   - content: str
   - metadata: Dict[str, Any]
   - embedding: Optional[List[float]]

4. Create SearchResult dataclass:
   - document: Document
   - score: float
   - distance: float

VERIFICATION:
- VectorStore initializes and creates collections
- Documents can be added and retrieved
- Semantic search returns relevant results
```

---

## Step 1.7: Implement SQLite Report Database

```
TASK: Create SQLite database for report storage

FILE: src/aegisml/storage/report_db.py

ACTIONS:
1. Create ReportDatabase class:
   - __init__(db_path: str)
   - Uses aiosqlite for async operations

2. Implement schema:
   - reports table (id, run_id, timestamp, profile, target_type, target_name, risk_level, risk_score, report_json)
   - findings table (id, report_id, test_name, category, status, severity, details)
   - Proper indexes for common queries

3. Implement methods:
   - async def save_report(report: SandboxRunResult) -> str
   - async def get_report(report_id: str) -> Optional[SandboxRunResult]
   - async def get_latest_report(target_name: str) -> Optional[SandboxRunResult]
   - async def list_reports(filters: ReportFilters) -> List[ReportSummary]
   - async def search_findings(severity: str = None, category: str = None) -> List[Finding]
   - async def get_report_history(target_name: str, limit: int = 10) -> List[ReportSummary]
   - async def delete_report(report_id: str) -> bool

4. Create migration system:
   - Track schema version
   - Auto-migrate on startup

VERIFICATION:
- Database creates on first run
- Reports save and load correctly
- Queries return expected results
- Migration system works
```

---

## Step 1.8: Create Threat Knowledge Base

```
TASK: Create structured threat knowledge base

FILE: src/aegisml/storage/knowledge_base.py

ACTIONS:
1. Create KnowledgeBase class:
   - __init__(kb_path: str, vector_store: VectorStore)
   - Load YAML files from knowledge directory
   - Index into vector store for semantic search

2. Implement methods:
   - async def load_knowledge() -> None
   - async def search(query: str, category: str = None) -> List[KnowledgeEntry]
   - async def get_jailbreak_techniques() -> List[JailbreakTechnique]
   - async def get_vulnerability_patterns() -> List[VulnerabilityPattern]
   - async def get_code_patterns(language: str = None) -> List[CodePattern]
   - async def add_custom_knowledge(entry: KnowledgeEntry) -> str
   - async def refresh() -> None

FILE: data/knowledge/jailbreaks.yaml

ACTIONS:
1. Create jailbreak techniques database:
   - Categories: role_play, prompt_injection, context_switch, encoding, emotional
   - Each technique: id, name, description, example_prompts, indicators, severity

FILE: data/knowledge/vulnerabilities.yaml

ACTIONS:
1. Create vulnerability patterns database:
   - Categories: command_injection, data_exfiltration, credential_harvesting, backdoor
   - Each pattern: id, name, description, code_patterns, indicators, severity, remediation

FILE: data/knowledge/code_patterns.yaml

ACTIONS:
1. Create suspicious code patterns database:
   - Languages: python, javascript, shell
   - Patterns: dangerous_functions, obfuscation, exfiltration, privilege_escalation

VERIFICATION:
- Knowledge base loads all YAML files
- Semantic search returns relevant techniques
- All categories populated with initial data
```

---

# PHASE 2: NLP Core
**Goal:** Implement conversation management and intent routing

---

## Step 2.1: Implement Intent Router

```
TASK: Create intent classification system

FILE: src/aegisml/nlp/intent.py

ACTIONS:
1. Create Intent dataclass:
   - intent_type: str
   - confidence: float
   - parameters: Dict[str, Any]
   - requires_analysis: bool
   - requires_llm: bool

2. Create IntentRouter class:
   - __init__(llm_engine: LocalLLMEngine)
   - Define INTENTS enum/list:
     - ANALYZE_MODEL
     - ANALYZE_INFRASTRUCTURE
     - COMPARE
     - EXPLAIN
     - QUERY_KNOWLEDGE
     - GENERATE_REPORT
     - LIST_REPORTS
     - GENERAL_QUESTION

3. Implement methods:
   - async def classify(query: str) -> Intent
   - def _keyword_match(query: str) -> Optional[Intent]  # Fast path
   - async def _llm_classify(query: str) -> Intent  # Slow path
   - def _extract_parameters(query: str, intent_type: str) -> Dict[str, Any]

4. Create keyword patterns for fast matching:
   - "analyze", "scan", "check" -> ANALYZE_*
   - "compare", "versus", "vs" -> COMPARE
   - "explain", "why", "how" -> EXPLAIN
   - "list", "show", "reports" -> LIST_REPORTS
   - "what is", "tell me about" -> QUERY_KNOWLEDGE

5. Create LLM classification prompt template for ambiguous queries

VERIFICATION:
- Fast path handles common queries <10ms
- LLM path handles ambiguous queries
- Parameters extracted correctly
```

---

## Step 2.2: Implement Response Generator

```
TASK: Create response generation system

FILE: src/aegisml/nlp/response.py

ACTIONS:
1. Create ResponseGenerator class:
   - __init__(llm_engine: LocalLLMEngine, knowledge_base: KnowledgeBase)

2. Implement methods:
   - async def generate(context: ResponseContext) -> str
   - async def format_analysis_result(result: SandboxRunResult) -> str
   - async def format_comparison(results: List[SandboxRunResult]) -> str
   - async def explain_finding(finding: Finding, context: AnalysisContext) -> str
   - async def answer_knowledge_query(query: str, knowledge: List[KnowledgeEntry]) -> str

3. Create ResponseContext dataclass:
   - intent: Intent
   - user_query: str
   - analysis_result: Optional[SandboxRunResult]
   - knowledge: List[KnowledgeEntry]
   - history: List[Message]

4. Create response templates:
   - Analysis summary template
   - Comparison table template
   - Finding explanation template
   - Error response template

5. Implement streaming support:
   - async def generate_stream(context: ResponseContext) -> AsyncIterator[str]

VERIFICATION:
- Responses are coherent and informative
- Templates produce consistent formatting
- Streaming works correctly
```

---

## Step 2.3: Implement Conversation Manager

```
TASK: Create conversation management system

FILE: src/aegisml/nlp/conversation.py

ACTIONS:
1. Create Message dataclass:
   - role: str ("user", "assistant", "system")
   - content: str
   - timestamp: datetime
   - metadata: Dict[str, Any]

2. Create AnalysisContext dataclass:
   - current_target: Optional[str]
   - current_target_type: Optional[str]
   - last_analysis: Optional[SandboxRunResult]
   - referenced_findings: List[Finding]

3. Create ConversationManager class:
   - __init__(llm_engine, intent_router, response_generator, knowledge_base, report_db)
   - history: List[Message]
   - context: AnalysisContext
   - max_history: int = 20

4. Implement methods:
   - async def process_query(user_input: str) -> Response
   - async def _execute_analysis(intent: Intent) -> SandboxRunResult
   - async def _retrieve_knowledge(query: str) -> List[KnowledgeEntry]
   - def _update_context(intent: Intent, result: Any) -> None
   - def clear_history() -> None
   - def get_history() -> List[Message]
   - async def summarize_conversation() -> str

5. Implement the main flow:
   a. Classify intent
   b. Execute analysis if required
   c. Retrieve relevant knowledge
   d. Generate response
   e. Update history and context

6. Create main() entry point for CLI chat mode

VERIFICATION:
- Multi-turn conversations maintain context
- Analysis results integrated into responses
- History properly managed
```

---

## Step 2.4: Implement Context Management

```
TASK: Create context tracking and management

FILE: src/aegisml/nlp/context.py

ACTIONS:
1. Create ContextManager class:
   - Tracks conversation state
   - Resolves references ("it", "that model", "the previous one")
   - Maintains analysis history

2. Implement methods:
   - def resolve_reference(text: str) -> Optional[str]
   - def get_current_target() -> Optional[str]
   - def set_current_target(target: str, target_type: str) -> None
   - def add_analysis_result(result: SandboxRunResult) -> None
   - def get_recent_analyses(limit: int = 5) -> List[SandboxRunResult]
   - def serialize() -> Dict[str, Any]
   - def deserialize(data: Dict[str, Any]) -> None

3. Implement reference resolution patterns:
   - "it", "this", "that" -> current target
   - "the model", "the server" -> most recent of type
   - "model X" -> specific named target
   - "the previous analysis" -> last result

VERIFICATION:
- References resolve correctly
- Context persists across turns
- Serialization works for session save/restore
```

---

## Step 2.5: Integrate NLP with Existing Analysis Engine

```
TASK: Connect NLP layer to sandbox_cli analysis

FILE: src/aegisml/nlp/analysis_bridge.py

ACTIONS:
1. Create AnalysisBridge class:
   - Bridges between NLP layer and existing analysis engine
   - Handles async/sync conversion
   - Manages analysis execution

2. Implement methods:
   - async def analyze_model(model_path: str, profile: str = "standard") -> SandboxRunResult
   - async def analyze_mcp_server(server_path: str, profile: str = "infra-standard") -> SandboxRunResult
   - async def analyze_skill(skill_path: str, profile: str = "infra-quick") -> SandboxRunResult
   - async def compare_targets(targets: List[str], comparison_type: str) -> ComparisonResult
   - async def get_available_tests() -> List[TestInfo]
   - async def get_available_profiles() -> List[ProfileInfo]

3. Update sandbox_cli.py:
   - Add async wrappers for main functions
   - Ensure clean JSON output for programmatic use
   - Add progress callbacks for streaming

VERIFICATION:
- NLP layer can trigger all analysis types
- Results properly formatted
- Async execution works
```

---

# PHASE 3: Semantic Analysis
**Goal:** Add LLM-powered security analysis capabilities

---

## Step 3.1: Implement Semantic Code Analyzer

```
TASK: Create LLM-powered code analysis

FILE: src/aegisml/analyzers/semantic_analyzer.py

ACTIONS:
1. Create SemanticAnalyzer class:
   - __init__(llm_engine: LocalLLMEngine, knowledge_base: KnowledgeBase)

2. Implement methods:
   - async def analyze_code(code: str, language: str, focus: str = "security") -> SemanticAnalysis
   - async def detect_obfuscation(code: str) -> ObfuscationAnalysis
   - async def analyze_data_flow(code: str) -> DataFlowAnalysis
   - async def identify_intent(code: str) -> IntentAnalysis
   - async def explain_code(code: str, question: str) -> str

3. Create SemanticAnalysis dataclass:
   - summary: str
   - security_concerns: List[SecurityConcern]
   - code_intent: str
   - confidence: float
   - recommendations: List[str]

4. Create analysis prompts:
   - Security analysis prompt
   - Obfuscation detection prompt
   - Data flow analysis prompt
   - Intent identification prompt

5. Implement chunking for large files:
   - Split code into analyzable chunks
   - Aggregate findings
   - Handle cross-chunk patterns

VERIFICATION:
- Analyzer identifies security issues
- Obfuscation detection works
- Large files handled correctly
```

---

## Step 3.2: Implement Semantic Security Tests

```
TASK: Create LLM-powered test plugins

FILE: src/aegisml/tests/semantic_tests.py

ACTIONS:
1. Create SemanticSecurityTest base class:
   - Extends SandboxTest
   - requires_llm = True
   - async def run(context, llm_engine) -> TestResult

2. Create SemanticCodeReviewTest:
   - Uses LLM to review code for security issues
   - Cross-references with knowledge base
   - Generates detailed findings

3. Create SemanticObfuscationTest:
   - Detects code obfuscation patterns
   - Identifies hidden functionality
   - Assesses malicious intent

4. Create SemanticBehaviorTest:
   - Analyzes model responses for safety
   - Uses LLM to evaluate outputs
   - Detects subtle policy violations

5. Register tests in TEST_REGISTRY:
   - "semantic_code_review"
   - "semantic_obfuscation"
   - "semantic_behavior"

6. Add to profiles:
   - "deep" profile includes semantic tests
   - "semantic" profile for semantic-only tests

VERIFICATION:
- Tests integrate with existing framework
- Results match expected schema
- LLM fallback when models unavailable
```

---

## Step 3.3: Implement Model Behavior Analyzer

```
TASK: Create model behavior analysis using local LLM

FILE: src/aegisml/analyzers/behavior_analyzer.py

ACTIONS:
1. Create BehaviorAnalyzer class:
   - __init__(llm_engine: LocalLLMEngine)
   - Uses analysis LLM to evaluate target model outputs

2. Implement methods:
   - async def analyze_responses(model_path: str, prompts: List[str]) -> BehaviorAnalysis
   - async def evaluate_safety(response: str, prompt: str) -> SafetyEvaluation
   - async def detect_inconsistency(responses: List[str]) -> ConsistencyAnalysis
   - async def assess_capability(model_path: str, capability: str) -> CapabilityScore

3. Create BehaviorAnalysis dataclass:
   - safety_score: float
   - consistency_score: float
   - capability_scores: Dict[str, float]
   - concerning_patterns: List[str]
   - recommendations: List[str]

4. Create evaluation prompts:
   - Safety evaluation prompt
   - Consistency check prompt
   - Capability assessment prompts

VERIFICATION:
- Behavior analysis produces meaningful results
- Safety scores correlate with actual safety
- Works with various model sizes
```

---

## Step 3.4: Implement Finding Explanation Generator

```
TASK: Create natural language explanations for findings

FILE: src/aegisml/nlp/explainer.py

ACTIONS:
1. Create FindingExplainer class:
   - __init__(llm_engine: LocalLLMEngine, knowledge_base: KnowledgeBase)

2. Implement methods:
   - async def explain_finding(finding: Finding, context: AnalysisContext) -> str
   - async def explain_risk_score(report: SandboxRunResult) -> str
   - async def suggest_remediation(finding: Finding) -> List[str]
   - async def compare_findings(findings: List[Finding]) -> str
   - async def generate_executive_summary(report: SandboxRunResult) -> str

3. Create explanation templates:
   - Technical explanation (for developers)
   - Executive summary (for managers)
   - Remediation guide (for fixes)

4. Implement context-aware explanations:
   - Reference related knowledge base entries
   - Include code examples when relevant
   - Suggest specific fixes

VERIFICATION:
- Explanations are clear and actionable
- Remediation suggestions are practical
- Different audiences served appropriately
```

---

# PHASE 4: Knowledge Integration
**Goal:** Implement RAG and knowledge management

---

## Step 4.1: Implement RAG Pipeline

```
TASK: Create Retrieval-Augmented Generation pipeline

FILE: src/aegisml/nlp/rag.py

ACTIONS:
1. Create RAGPipeline class:
   - __init__(vector_store, knowledge_base, llm_engine)

2. Implement methods:
   - async def query(question: str, context: Optional[str] = None) -> RAGResponse
   - async def retrieve(query: str, top_k: int = 5) -> List[Document]
   - async def generate_with_context(query: str, documents: List[Document]) -> str
   - async def rerank(query: str, documents: List[Document]) -> List[Document]

3. Create RAGResponse dataclass:
   - answer: str
   - sources: List[Document]
   - confidence: float

4. Implement retrieval strategies:
   - Semantic search (default)
   - Hybrid search (semantic + keyword)
   - Filtered search (by category)

5. Create augmentation prompts:
   - Context injection template
   - Source citation template

VERIFICATION:
- RAG returns relevant answers
- Sources properly cited
- Handles no-match gracefully
```

---

## Step 4.2: Implement Report Indexing

```
TASK: Index reports for semantic search

FILE: src/aegisml/storage/report_indexer.py

ACTIONS:
1. Create ReportIndexer class:
   - __init__(vector_store: VectorStore, report_db: ReportDatabase)

2. Implement methods:
   - async def index_report(report: SandboxRunResult) -> None
   - async def search_reports(query: str, filters: ReportFilters) -> List[ReportSummary]
   - async def find_similar_findings(finding: Finding) -> List[Finding]
   - async def get_historical_trends(target_name: str) -> TrendAnalysis
   - async def reindex_all() -> int

3. Create indexing strategy:
   - Index report summary
   - Index each finding separately
   - Index test results
   - Store metadata for filtering

4. Implement incremental indexing:
   - Index new reports on save
   - Background re-indexing option

VERIFICATION:
- Reports searchable after indexing
- Similar findings returned correctly
- Performance acceptable for large datasets
```

---

## Step 4.3: Implement Knowledge Update System

```
TASK: Create knowledge base update mechanism

FILE: src/aegisml/storage/knowledge_updater.py

ACTIONS:
1. Create KnowledgeUpdater class:
   - __init__(knowledge_base: KnowledgeBase)

2. Implement methods:
   - async def add_jailbreak_technique(technique: JailbreakTechnique) -> str
   - async def add_vulnerability_pattern(pattern: VulnerabilityPattern) -> str
   - async def import_from_yaml(file_path: str) -> int
   - async def export_to_yaml(category: str, file_path: str) -> None
   - async def validate_knowledge(entry: KnowledgeEntry) -> ValidationResult
   - async def get_version() -> str
   - async def diff_versions(v1: str, v2: str) -> KnowledgeDiff

3. Create versioning system:
   - Track knowledge base version
   - Support incremental updates
   - Maintain update history

4. Implement validation:
   - Schema validation
   - Duplicate detection
   - Consistency checks

VERIFICATION:
- Knowledge can be added and updated
- Versioning tracks changes
- Validation prevents bad data
```

---

## Step 4.4: Implement Comparative Analysis

```
TASK: Create cross-analysis comparison features

FILE: src/aegisml/analyzers/comparative.py

ACTIONS:
1. Create ComparativeAnalyzer class:
   - __init__(report_db: ReportDatabase, llm_engine: LocalLLMEngine)

2. Implement methods:
   - async def compare_models(model_paths: List[str]) -> ModelComparison
   - async def compare_reports(report_ids: List[str]) -> ReportComparison
   - async def detect_drift(target_name: str, baseline_id: str) -> DriftAnalysis
   - async def rank_by_safety(targets: List[str]) -> List[RankedTarget]
   - async def generate_comparison_report(comparison: Any) -> str

3. Create comparison data structures:
   - ModelComparison (side-by-side metrics)
   - ReportComparison (findings diff)
   - DriftAnalysis (changes over time)

4. Implement comparison logic:
   - Metric normalization
   - Statistical significance
   - Trend detection

VERIFICATION:
- Comparisons are accurate
- Drift detection works
- Rankings are meaningful
```

---

# PHASE 5: Interface Enhancement
**Goal:** Add TUI, enhanced CLI, and optional Web UI

---

## Step 5.1: Enhance CLI with NLP Queries

```
TASK: Add natural language support to CLI

FILE: src/aegisml/sandbox_cli.py (update)

ACTIONS:
1. Add new CLI arguments:
   - --ask "question" : Natural language query
   - --chat : Enter interactive chat mode
   - --explain <finding-id> : Explain a finding
   - --compare <target1> <target2> : Compare targets

2. Implement ask command:
   - Parse natural language
   - Route to appropriate handler
   - Display formatted response

3. Implement chat command:
   - Enter REPL mode
   - Maintain conversation context
   - Support commands within chat (/help, /clear, /exit)

4. Update output formatting:
   - Rich terminal output (colors, tables)
   - Progress indicators
   - Streaming responses

5. Add model management commands:
   - --list-models : Show available LLM models
   - --download-model <name> : Download recommended model
   - --model-info : Show loaded model status

VERIFICATION:
- NLP queries work from CLI
- Chat mode maintains context
- Output is well-formatted
```

---

## Step 5.2: Build Terminal UI (TUI)

```
TASK: Create TUI using Textual framework

FILE: src/aegisml/ui/tui.py

ACTIONS:
1. Create AegisMLApp class (extends textual.App):
   - Dashboard view
   - Analysis view
   - Reports view
   - Chat view
   - Settings view

2. Implement Dashboard:
   - Recent analyses summary
   - Quick stats (total scans, risk distribution)
   - Quick actions (analyze, compare, chat)

3. Implement Analysis View:
   - File browser for model/server selection
   - Profile selector
   - Real-time analysis progress
   - Results display

4. Implement Reports View:
   - Searchable report list
   - Report detail view
   - Finding browser
   - Export options

5. Implement Chat View:
   - Chat input
   - Message history
   - Context indicator
   - Suggested queries

6. Create key bindings:
   - Tab: Switch views
   - Ctrl+A: New analysis
   - Ctrl+C: Chat mode
   - Ctrl+Q: Quit
   - /: Command palette

7. Create main() entry point

VERIFICATION:
- TUI launches and displays correctly
- All views functional
- Keyboard navigation works
```

---

## Step 5.3: Build Web UI (Optional)

```
TASK: Create local web interface

FILE: src/aegisml/ui/web.py

ACTIONS:
1. Create FastAPI application:
   - Bind to 127.0.0.1 only (security)
   - CORS disabled (local only)

2. Implement API endpoints:
   - POST /api/analyze : Start analysis
   - GET /api/reports : List reports
   - GET /api/reports/{id} : Get report
   - POST /api/chat : Send chat message
   - GET /api/status : System status
   - WebSocket /ws/analysis : Streaming analysis

3. Create HTMX templates:
   - index.html : Dashboard
   - analyze.html : Analysis form
   - reports.html : Report list
   - report.html : Report detail
   - chat.html : Chat interface

4. Implement file upload:
   - Drag-drop model upload
   - Server file browser
   - Temporary file handling

5. Create static assets:
   - Minimal CSS (Pico CSS or similar)
   - HTMX library
   - No JavaScript framework needed

6. Create main() entry point

VERIFICATION:
- Web UI accessible at localhost
- All features work
- No external network calls
```

---

## Step 5.4: Update MCP Tools for NLP

```
TASK: Add NLP-enabled MCP tools

FILE: src/aegisml/mcp_server.py (update)

ACTIONS:
1. Add new MCP tools:
   - aegisml_ask: Natural language query
   - aegisml_explain: Explain finding
   - aegisml_chat: Multi-turn conversation
   - aegisml_compare: Compare targets
   - aegisml_recommend: Get recommendations

2. Implement aegisml_ask:
   - Input: query (str), context (optional str)
   - Output: Natural language response

3. Implement aegisml_explain:
   - Input: finding_id (str)
   - Output: Detailed explanation

4. Implement aegisml_chat:
   - Input: message (str), session_id (optional str)
   - Output: Response with session_id for continuity

5. Implement aegisml_compare:
   - Input: targets (List[str]), comparison_type (str)
   - Output: Comparison results

6. Implement aegisml_recommend:
   - Input: target (str), use_case (str)
   - Output: Recommendations

7. Add streaming support:
   - Implement Server-Sent Events for long responses

VERIFICATION:
- All new tools work via MCP
- Responses are well-formatted
- Streaming works
```

---

# PHASE 6: Polish & Documentation
**Goal:** Testing, optimization, and documentation

---

## Step 6.1: Create Test Suite

```
TASK: Implement comprehensive tests

FILE: tests/test_nlp_engine.py

ACTIONS:
1. Test LLM engine:
   - Model loading/unloading
   - Generation quality
   - Error handling

FILE: tests/test_conversation.py

ACTIONS:
1. Test conversation manager:
   - Intent classification
   - Multi-turn context
   - Response generation

FILE: tests/test_storage.py

ACTIONS:
1. Test storage layer:
   - Vector store operations
   - Report database CRUD
   - Knowledge base queries

FILE: tests/test_semantic.py

ACTIONS:
1. Test semantic analysis:
   - Code analysis accuracy
   - Finding detection
   - Explanation quality

FILE: tests/test_integration.py

ACTIONS:
1. End-to-end tests:
   - Full analysis workflow
   - Chat workflow
   - Comparison workflow

VERIFICATION:
- All tests pass
- Coverage > 80%
- No regressions
```

---

## Step 6.2: Performance Optimization

```
TASK: Optimize performance

ACTIONS:
1. Profile critical paths:
   - LLM inference time
   - Vector search time
   - Report generation time

2. Implement caching:
   - LRU cache for embeddings
   - Query result cache
   - Model response cache

3. Optimize LLM usage:
   - Batch similar queries
   - Use smaller models for simple tasks
   - Implement early stopping

4. Optimize storage:
   - Index frequently queried fields
   - Implement connection pooling
   - Optimize vector dimensions

5. Memory management:
   - Unload unused models
   - Limit history size
   - Clear caches periodically

VERIFICATION:
- Query response < 5s (P95)
- Memory usage < 16GB peak
- No memory leaks
```

---

## Step 6.3: Update Documentation

```
TASK: Comprehensive documentation update

FILE: README.md (update)

ACTIONS:
1. Update for v3.0:
   - New features overview
   - NLP capabilities
   - Model requirements
   - Quick start guide

FILE: docs/NLP_GUIDE.md (new)

ACTIONS:
1. Create NLP usage guide:
   - Natural language queries
   - Chat mode usage
   - Context management
   - Best practices

FILE: docs/DEPLOYMENT.md (new)

ACTIONS:
1. Create deployment guide:
   - Model download instructions
   - Air-gap deployment
   - Resource requirements
   - Configuration options

FILE: docs/API_REFERENCE.md (update)

ACTIONS:
1. Update API reference:
   - New MCP tools
   - Python API
   - Response schemas

VERIFICATION:
- Documentation complete
- Examples work
- No outdated information
```

---

## Step 6.4: Create Release Package

```
TASK: Prepare release

ACTIONS:
1. Update version:
   - pyproject.toml: "3.0.0"
   - sandbox_cli.py: RUNNER_VERSION = "3.0.0"
   - Schema version if changed

2. Create changelog:
   - New features
   - Breaking changes
   - Migration guide

3. Build distribution:
   - python -m build
   - Test installation

4. Create release artifacts:
   - Source distribution
   - Wheel
   - Documentation

5. Tag release:
   - git tag v3.0.0
   - Push to repository

VERIFICATION:
- Package installs cleanly
- All features work
- Documentation accurate
```

---

# Quick Reference: File Creation Order

```
Phase 1:
  1. Directory structure
  2. pyproject.toml
  3. src/aegisml/config.py
  4. config/aegisml.yaml
  5. src/aegisml/nlp/__init__.py
  6. src/aegisml/nlp/llm_engine.py
  7. src/aegisml/nlp/embeddings.py
  8. src/aegisml/storage/__init__.py
  9. src/aegisml/storage/vector_store.py
  10. src/aegisml/storage/report_db.py
  11. src/aegisml/storage/knowledge_base.py
  12. data/knowledge/jailbreaks.yaml
  13. data/knowledge/vulnerabilities.yaml
  14. data/knowledge/code_patterns.yaml

Phase 2:
  15. src/aegisml/nlp/intent.py
  16. src/aegisml/nlp/response.py
  17. src/aegisml/nlp/conversation.py
  18. src/aegisml/nlp/context.py
  19. src/aegisml/nlp/analysis_bridge.py

Phase 3:
  20. src/aegisml/analyzers/semantic_analyzer.py
  21. src/aegisml/tests/semantic_tests.py
  22. src/aegisml/analyzers/behavior_analyzer.py
  23. src/aegisml/nlp/explainer.py

Phase 4:
  24. src/aegisml/nlp/rag.py
  25. src/aegisml/storage/report_indexer.py
  26. src/aegisml/storage/knowledge_updater.py
  27. src/aegisml/analyzers/comparative.py

Phase 5:
  28. src/aegisml/sandbox_cli.py (update)
  29. src/aegisml/ui/tui.py
  30. src/aegisml/ui/web.py (optional)
  31. src/aegisml/mcp_server.py (update)

Phase 6:
  32. tests/*
  33. docs/NLP_GUIDE.md
  34. docs/DEPLOYMENT.md
  35. README.md (update)
```

---

# Execution Commands

To start implementation, use these commands:

```bash
# Start Phase 1
# "Begin implementing AegisML v3.0 Phase 1: Foundation. Follow the steps in docs/IMPLEMENTATION_PLAN.md starting with Step 1.1"

# Continue to Phase 2
# "Continue with AegisML v3.0 Phase 2: NLP Core. Follow docs/IMPLEMENTATION_PLAN.md"

# And so on for each phase...
```

---

*This implementation plan provides complete, executable steps for building AegisML v3.0.*
