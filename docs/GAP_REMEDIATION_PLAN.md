# BenderBox Gap Remediation Plan
# Phased Approach for Completing v3.0 Features

**Version:** 3.0.1
**Status:** Ready for Implementation
**Created:** 2026-01-12

---

## Overview

This document provides a **phased implementation approach** to close the gaps identified in the BenderBox v3.0 architecture review. Each priority level contains discrete phases that can be executed independently.

---

## Gap Summary

| Priority | Gap | Current % | Target % | Effort |
|----------|-----|-----------|----------|--------|
| **HIGH** | Semantic Analysis LLM Integration | 60% | 100% | Medium |
| **HIGH** | Vector Store Report Querying | 70% | 100% | Medium |
| **MEDIUM** | Behavior Analysis | 50% | 100% | Medium |
| **MEDIUM** | Comparative Analysis | 50% | 100% | Medium |
| **MEDIUM** | HTML Report Generation | 40% | 100% | Medium |
| **LOW** | TUI (Textual Interface) | 20% | 100% | High |

> **Note:** Web UI (FastAPI) removed from scope. Focus is on terminal-based TUI with rich HTML reports viewable in any browser.

---

# PRIORITY HIGH: Core NLP Features

## H1: Semantic Analysis LLM Integration

**Goal:** Complete the wiring between semantic analyzers and the NLP conversation layer

### Phase H1.1: Analysis Bridge Completion
**Files:** `src/benderbox/nlp/analysis_bridge.py`, `src/benderbox/analyzers/semantic.py`

**Tasks:**
1. Wire `SemanticAnalyzer` to `AnalysisBridge.analyze_code()`
2. Add LLM engine injection to semantic analyzer
3. Implement fallback behavior when LLM unavailable
4. Add progress callbacks for streaming analysis updates

**Acceptance Criteria:**
- [ ] `analyze_code()` uses LLM when available
- [ ] Graceful template fallback when LLM unavailable
- [ ] Progress updates stream to UI

### Phase H1.2: Conversation Integration
**Files:** `src/benderbox/nlp/conversation.py`, `src/benderbox/nlp/response.py`

**Tasks:**
1. Add semantic analysis intent routing in `ConversationManager`
2. Implement `_execute_semantic_analysis()` handler
3. Wire semantic results to `ResponseGenerator.format_semantic_result()`
4. Add semantic-specific response templates

**Acceptance Criteria:**
- [ ] "Analyze this code for security issues" triggers semantic analysis
- [ ] Results formatted as natural language
- [ ] Follow-up questions reference previous analysis

### Phase H1.3: Chat UI Integration
**Files:** `src/benderbox/ui/chat_ui.py`

**Tasks:**
1. Add `semantic` command alias
2. Implement inline code analysis (paste code → analyze)
3. Add `/explain <finding>` command for semantic explanations
4. Display semantic confidence scores

**Acceptance Criteria:**
- [ ] `semantic <file>` command works in chat
- [ ] Pasted code blocks can be analyzed inline
- [ ] Explanations are contextual and helpful

---

## H2: Vector Store Report Querying

**Goal:** Enable semantic search across stored reports and knowledge

### Phase H2.1: Report Indexing Pipeline
**Files:** `src/benderbox/storage/report_indexer.py` (new), `src/benderbox/storage/vector_store.py`

**Tasks:**
1. Create `ReportIndexer` class
2. Implement `index_report(report: SandboxRunResult)` method
3. Index report summary, findings, and test results separately
4. Add metadata filtering (date, target, severity)
5. Hook indexing into report save workflow

**Acceptance Criteria:**
- [ ] Reports auto-indexed on save
- [ ] Findings searchable by natural language
- [ ] Metadata filters work correctly

### Phase H2.2: Semantic Search API
**Files:** `src/benderbox/storage/vector_store.py`, `src/benderbox/nlp/rag.py`

**Tasks:**
1. Implement `search_reports(query: str, filters: dict)` method
2. Add `find_similar_findings(finding: Finding)` method
3. Create `RAGPipeline` for augmented responses
4. Implement reranking for better relevance

**Acceptance Criteria:**
- [ ] "Show me all command injection findings" returns relevant results
- [ ] Similar findings grouped across reports
- [ ] RAG responses cite sources

### Phase H2.3: UI Integration
**Files:** `src/benderbox/ui/chat_ui.py`, `src/benderbox/ui/app.py`

**Tasks:**
1. Add `search` command to chat UI
2. Add `--search` flag to CLI
3. Display search results with snippets
4. Enable drill-down into full reports

**Acceptance Criteria:**
- [ ] `search "injection vulnerabilities"` works
- [ ] Results show relevant context
- [ ] Can navigate to full report from search

---

# PRIORITY MEDIUM: Analysis Enhancements

## M1: Behavior Analysis

**Goal:** Implement model behavior fingerprinting and capability detection

### Phase M1.1: Core Analyzer Implementation
**Files:** `src/benderbox/analyzers/behavior.py`

**Tasks:**
1. Implement `BehaviorAnalyzer` class fully
2. Add `analyze_responses(model, prompts)` method
3. Implement `evaluate_safety(response, prompt)` method
4. Add `detect_inconsistency(responses)` method
5. Create behavior analysis prompts

**Acceptance Criteria:**
- [ ] Can fingerprint model behavior patterns
- [ ] Safety evaluation produces meaningful scores
- [ ] Inconsistencies detected across responses

### Phase M1.2: Capability Detection
**Files:** `src/benderbox/analyzers/behavior.py`

**Tasks:**
1. Implement `assess_capability(model, capability)` method
2. Create capability test prompts (reasoning, coding, math, etc.)
3. Add capability scoring system
4. Implement capability comparison

**Acceptance Criteria:**
- [ ] Can assess model capabilities
- [ ] Scores are reproducible
- [ ] Capabilities comparable across models

### Phase M1.3: Integration
**Files:** `src/benderbox/interrogation/engine.py`, `src/benderbox/nlp/analysis_bridge.py`

**Tasks:**
1. Add behavior analysis to interrogation engine
2. Include behavior results in reports
3. Wire to conversation manager
4. Add behavior-specific CLI commands

**Acceptance Criteria:**
- [ ] `interrogate --behavior` includes behavior analysis
- [ ] Behavior results in JSON/Markdown reports
- [ ] Can ask "What are this model's capabilities?"

---

## M2: Comparative Analysis

**Goal:** Enable side-by-side model and report comparison

### Phase M2.1: Core Comparison Logic
**Files:** `src/benderbox/analyzers/comparative.py`

**Tasks:**
1. Implement `ComparativeAnalyzer` class fully
2. Add `compare_models(model_paths)` method
3. Add `compare_reports(report_ids)` method
4. Implement metric normalization
5. Create comparison data structures

**Acceptance Criteria:**
- [ ] Can compare two or more models
- [ ] Metrics normalized for fair comparison
- [ ] Structured comparison output

### Phase M2.2: Drift Detection
**Files:** `src/benderbox/analyzers/comparative.py`

**Tasks:**
1. Implement `detect_drift(target, baseline_id)` method
2. Add statistical significance testing
3. Create drift visualization data
4. Implement trend detection

**Acceptance Criteria:**
- [ ] Can detect changes between runs
- [ ] Significant changes highlighted
- [ ] Trend direction indicated

### Phase M2.3: UI and Reporting
**Files:** `src/benderbox/ui/chat_ui.py`, `src/benderbox/reporting/report_generator.py`

**Tasks:**
1. Add `compare` command enhancements
2. Create comparison report templates
3. Add side-by-side Markdown tables
4. Implement comparison export

**Acceptance Criteria:**
- [ ] `compare model1 model2` produces useful output
- [ ] Comparison reports well-formatted
- [ ] Can export comparison as standalone document

---

# PRIORITY LOW: Interface Enhancements

## L1: TUI (Textual Interface)

**Goal:** Build a full-featured terminal UI using Textual

### Phase L1.1: Core Application Shell
**Files:** `src/benderbox/ui/tui.py`

**Tasks:**
1. Create `BenderBoxApp(App)` class
2. Implement screen navigation (Dashboard, Analysis, Reports, Chat)
3. Add header with status bar
4. Add footer with key bindings
5. Create base CSS theme

**Acceptance Criteria:**
- [ ] TUI launches with `benderbox tui`
- [ ] Can navigate between screens
- [ ] Consistent look and feel

### Phase L1.2: Dashboard Screen
**Files:** `src/benderbox/ui/tui.py`, `src/benderbox/ui/tui_widgets.py` (new)

**Tasks:**
1. Create `DashboardScreen` class
2. Add recent analyses widget
3. Add quick stats panel (total scans, risk distribution)
4. Add quick action buttons
5. Implement auto-refresh

**Acceptance Criteria:**
- [ ] Dashboard shows recent activity
- [ ] Stats update in real-time
- [ ] Quick actions trigger operations

### Phase L1.3: Analysis Screen
**Files:** `src/benderbox/ui/tui.py`

**Tasks:**
1. Create `AnalysisScreen` class
2. Add file browser for target selection
3. Add profile selector dropdown
4. Implement real-time progress display
5. Show results in structured view

**Acceptance Criteria:**
- [ ] Can browse and select targets
- [ ] Profile selection works
- [ ] Progress visible during analysis

### Phase L1.4: Reports Screen
**Files:** `src/benderbox/ui/tui.py`

**Tasks:**
1. Create `ReportsScreen` class
2. Add searchable report list (DataTable)
3. Implement report detail view
4. Add finding browser with filtering
5. Add export button

**Acceptance Criteria:**
- [ ] Reports searchable and sortable
- [ ] Detail view shows full report
- [ ] Can export from TUI

### Phase L1.5: Chat Screen
**Files:** `src/benderbox/ui/tui.py`

**Tasks:**
1. Create `ChatScreen` class
2. Add message history panel
3. Add input field with command completion
4. Show context indicator
5. Implement suggested queries

**Acceptance Criteria:**
- [ ] Chat works within TUI
- [ ] History scrollable
- [ ] Context maintained

---

## M3: HTML Report Generation

**Goal:** Generate rich, self-contained HTML reports viewable in any HTML5 browser

### Phase M3.1: HTML Template System
**Files:** `src/benderbox/reporting/html_templates.py` (new), `src/benderbox/reporting/templates/report.html` (new)

**Tasks:**
1. Create base HTML5 template with embedded CSS
2. Design responsive layout (works on mobile/desktop)
3. Add syntax highlighting for code blocks
4. Create color-coded severity indicators
5. Ensure no external dependencies (self-contained)

**Acceptance Criteria:**
- [ ] HTML opens in any modern browser
- [ ] No external CSS/JS required
- [ ] Responsive design works
- [ ] Print-friendly layout

### Phase M3.2: Report Sections
**Files:** `src/benderbox/reporting/html_generator.py` (new)

**Tasks:**
1. Create `HTMLReportGenerator` class
2. Implement executive summary section
3. Implement findings table with sorting
4. Add risk score visualization (charts via inline SVG)
5. Create code snippet sections with highlighting
6. Add recommendation sections

**Acceptance Criteria:**
- [ ] All report sections render correctly
- [ ] Findings sortable by severity
- [ ] Risk visualizations display
- [ ] Code syntax highlighted

### Phase M3.3: Interactive Features (No JS Framework)
**Files:** `src/benderbox/reporting/html_generator.py`

**Tasks:**
1. Add collapsible sections (CSS-only or minimal vanilla JS)
2. Create table of contents with anchor links
3. Add search/filter for findings (vanilla JS)
4. Implement dark/light mode toggle
5. Add export to PDF button (print dialog)

**Acceptance Criteria:**
- [ ] Sections expand/collapse
- [ ] TOC navigation works
- [ ] Filtering works without page reload
- [ ] Theme toggle works

### Phase M3.4: Comparison Reports
**Files:** `src/benderbox/reporting/html_generator.py`, `src/benderbox/reporting/templates/comparison.html` (new)

**Tasks:**
1. Create side-by-side comparison template
2. Add diff highlighting for changes
3. Create comparison charts (SVG)
4. Implement capability comparison tables
5. Add trend indicators

**Acceptance Criteria:**
- [ ] Comparison reports show both models
- [ ] Differences highlighted visually
- [ ] Charts render without JS libraries

### Phase M3.5: CLI Integration
**Files:** `src/benderbox/ui/app.py`, `src/benderbox/reporting/export.py`

**Tasks:**
1. Add `--format html` flag to analysis commands
2. Add `export --html` command
3. Auto-open report in browser option
4. Add batch export capability

**Acceptance Criteria:**
- [ ] `benderbox interrogate model.gguf --format html` works
- [ ] `benderbox export --html report_id` works
- [ ] Browser auto-opens when requested

---

# PRIORITY FUTURE: RLM Integration

## R1: RLM Integration (Recursive Language Models)

**Goal:** Enable analysis of arbitrarily large inputs using Recursive Language Models

**Reference:** [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) - Recursive Language Models

RLM allows LLMs to process inputs 2 orders of magnitude beyond context windows by treating
long contexts as Python variables in a REPL environment, enabling programmatic examination,
decomposition, and recursive self-calls over snippets.

### Phase R1.1: RLM Infrastructure
**Files:** `src/benderbox/nlp/rlm_controller.py` (new), `src/benderbox/nlp/rlm_environment.py` (new)

**Tasks:**
1. Integrate [alexzhang13/rlm](https://github.com/alexzhang13/rlm) library
2. Create `RLMController` wrapper for LocalLLMEngine
3. Implement REPL environment with context variables
4. Add recursive call depth management
5. Implement cost tracking for recursive calls

**Acceptance Criteria:**
- [ ] RLM library integrated
- [ ] Can process inputs beyond context window
- [ ] Recursive calls work correctly
- [ ] Cost per query tracked

### Phase R1.2: Codebase-Scale Analysis
**Files:** `src/benderbox/analyzers/codebase_analyzer.py` (new), `src/benderbox/nlp/analysis_bridge.py`

**Tasks:**
1. Add `analyze_codebase(path)` method to AnalysisBridge
2. Implement recursive file decomposition strategy
3. Create cross-file vulnerability correlation
4. Add intelligent file sampling (prioritize entry points, configs)
5. Implement finding aggregation across files

**Acceptance Criteria:**
- [ ] `semantic ./src/` analyzes entire directory
- [ ] Cross-file issues detected (e.g., tainted data flow)
- [ ] Results aggregated into single report
- [ ] Performance acceptable for 1000+ file codebases

### Phase R1.3: Batch Response Analysis
**Files:** `src/benderbox/analyzers/behavior.py`, `src/benderbox/interrogation/batch_analyzer.py` (new)

**Tasks:**
1. Enable analysis of 1000+ interrogation responses
2. Implement pattern detection across responses
3. Add behavioral fingerprinting at scale
4. Create response clustering for anomaly detection
5. Generate behavioral summary reports

**Acceptance Criteria:**
- [ ] Can analyze full interrogation history
- [ ] Patterns detected across responses
- [ ] Behavioral fingerprints generated
- [ ] Anomalies highlighted

---

# Implementation Order

## Recommended Sequence

```
Sprint 1: HIGH Priority - Semantic Analysis
├── H1.1: Analysis Bridge Completion
├── H1.2: Conversation Integration
└── H1.3: Chat UI Integration

Sprint 2: HIGH Priority - Vector Store
├── H2.1: Report Indexing Pipeline
├── H2.2: Semantic Search API
└── H2.3: UI Integration

Sprint 3: MEDIUM Priority - Behavior & Comparison
├── M1.1: Core Behavior Analyzer
├── M1.2: Capability Detection
├── M2.1: Core Comparison Logic
└── M2.2: Drift Detection

Sprint 4: MEDIUM Priority - Integration & HTML
├── M1.3: Behavior Integration
├── M2.3: Comparison UI/Reporting
├── M3.1: HTML Template System
└── M3.2: Report Sections

Sprint 5: MEDIUM Priority - HTML Polish
├── M3.3: Interactive Features
├── M3.4: Comparison Reports
└── M3.5: CLI Integration

Sprint 6: LOW Priority - TUI Foundation
├── L1.1: TUI Application Shell
├── L1.2: Dashboard Screen
└── L1.3: Analysis Screen

Sprint 7: LOW Priority - TUI Completion
├── L1.4: Reports Screen
└── L1.5: Chat Screen

Sprint 8: FUTURE - RLM Integration
├── R1: RLM Infrastructure
├── R2: Codebase-Scale Analysis
└── R3: Batch Response Analysis
```

---

# Quick Reference: Files to Create/Modify

## New Files
```
src/benderbox/storage/report_indexer.py              # H2.1
src/benderbox/reporting/html_templates.py            # M3.1
src/benderbox/reporting/html_generator.py            # M3.2
src/benderbox/reporting/templates/report.html        # M3.1
src/benderbox/reporting/templates/comparison.html    # M3.4
src/benderbox/ui/tui_widgets.py                      # L1.2
```

## Files to Modify
```
src/benderbox/nlp/analysis_bridge.py        # H1.1
src/benderbox/analyzers/semantic.py         # H1.1
src/benderbox/nlp/conversation.py           # H1.2
src/benderbox/nlp/response.py               # H1.2
src/benderbox/ui/chat_ui.py                 # H1.3, H2.3
src/benderbox/storage/vector_store.py       # H2.1, H2.2
src/benderbox/nlp/rag.py                    # H2.2
src/benderbox/ui/app.py                     # H2.3, M3.5
src/benderbox/analyzers/behavior.py         # M1.1, M1.2
src/benderbox/interrogation/engine.py       # M1.3
src/benderbox/analyzers/comparative.py      # M2.1, M2.2
src/benderbox/reporting/report_generator.py # M2.3
src/benderbox/reporting/export.py           # M3.5
src/benderbox/ui/tui.py                     # L1.*
```

---

# Verification Checklist

## HIGH Priority Complete When:
- [ ] Semantic analysis works in chat with "analyze this code"
- [ ] LLM fallback produces useful template responses
- [ ] Reports searchable by natural language query
- [ ] Similar findings discoverable across reports
- [ ] RAG responses include source citations

## MEDIUM Priority Complete When:
- [ ] Model behavior fingerprinting works
- [ ] Capability scores produced and comparable
- [ ] Side-by-side model comparison works
- [ ] Drift detection identifies significant changes
- [ ] Comparison reports export correctly
- [ ] HTML reports open in any browser
- [ ] HTML reports are self-contained (no external deps)
- [ ] Risk visualizations render via inline SVG
- [ ] HTML comparison reports show side-by-side

## LOW Priority Complete When:
- [ ] TUI launches with `benderbox tui`
- [ ] TUI Dashboard shows recent activity
- [ ] TUI Analysis screen works end-to-end
- [ ] TUI Reports screen with filtering works
- [ ] TUI Chat maintains conversation context

---

# Sprint Tracking Template

Use this to track progress in each sprint:

```markdown
## Sprint [N]: [Name]
**Status:** Not Started | In Progress | Complete
**Started:** YYYY-MM-DD
**Completed:** YYYY-MM-DD

### Tasks
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

### Blockers
- None

### Notes
-
```

---

*This plan provides executable phases for completing BenderBox v3.0 gap remediation.*
