# BenderBox v3.0 Completion Roadmap

**Created:** 2026-01-16
**Current Progress:** ~80%
**Target:** Production-ready v3.0 release

---

## Protected Capabilities (No Regression)

These features are complete and must remain functional throughout remaining work:

### Core NLP System
- [x] Keyword mapping & synonym expansion (`command_mapper.py`)
- [x] Intent classification (`intent.py`)
- [x] Workflow engine with interactive prompts (`workflows.py`, `interactive.py`)
- [x] Conversation management with context (`conversation.py`)
- [x] Bender personality responses (`persona.py`)
- [x] Streaming responses

### Analysis Engine
- [x] Model analysis (GGUF metadata, static tests)
- [x] MCP server security analysis
- [x] Skill/context analysis
- [x] Semantic code analysis
- [x] Behavior analysis
- [x] Comparative analysis

### Storage & Knowledge
- [x] SQLite report database
- [x] ChromaDB vector store
- [x] Report indexing & search
- [x] RAG pipeline
- [x] Threat knowledge base (base)

### User Interface
- [x] Chat UI with NLP (`chat_ui.py`)
- [x] Terminal UI with Rich formatting
- [x] TUI dashboard (Textual)
- [x] Web UI (FastAPI)
- [x] Model list with loaded indicators
- [x] Greetings/farewells
- [x] System resource monitoring
- [x] Download progress display

### Integration
- [x] HuggingFace model download via NLP
- [x] Analysis bridge to sandbox_cli
- [x] JSON report auto-save (suppressed from terminal)
- [x] HTML report viewer

---

## Remaining Work Phases

### **PHASE A: Stability & Testing** (Recommended First)
**Goal:** Ensure existing features are solid before adding more
**Duration:** 1 sprint (1 week)

#### Sprint A.1: Regression Test Suite
| Task | Priority | Effort |
|------|----------|--------|
| Create test fixtures for NLP commands | HIGH | 4h |
| Add integration tests for workflows | HIGH | 4h |
| Test model load/unload cycle | HIGH | 2h |
| Test analysis pipeline end-to-end | HIGH | 4h |
| Test report generation & viewing | MEDIUM | 2h |
| Add CI smoke tests | MEDIUM | 2h |

**Deliverables:**
- `tests/test_nlp_integration.py`
- `tests/test_workflows.py`
- `tests/test_analysis_pipeline.py`
- GitHub Actions CI workflow

---

### **PHASE B: Knowledge Base Expansion**
**Goal:** Enrich threat detection with more patterns
**Duration:** 1 sprint (1 week)

#### Sprint B.1: Threat Pattern Library
| Task | Priority | Effort |
|------|----------|--------|
| Expand jailbreak patterns (DAN, roleplay, etc.) | HIGH | 4h |
| Add prompt injection patterns | HIGH | 4h |
| Add data exfiltration code patterns | HIGH | 3h |
| Add backdoor trigger patterns | MEDIUM | 3h |
| Add OWASP LLM Top 10 patterns | MEDIUM | 4h |
| Create pattern validation tests | HIGH | 2h |

**Deliverables:**
- `data/knowledge/jailbreaks.yaml` (expanded)
- `data/knowledge/prompt_injection.yaml`
- `data/knowledge/code_patterns.yaml` (expanded)
- `data/knowledge/owasp_llm.yaml`

---

### **PHASE C: LLM-Powered Tests**
**Goal:** Add semantic security tests using local LLM
**Duration:** 2 sprints (2 weeks)

#### Sprint C.1: Semantic Test Framework
| Task | Priority | Effort |
|------|----------|--------|
| Create `SemanticSecurityTest` base class | HIGH | 4h |
| Add LLM requirement flag to test registry | HIGH | 2h |
| Implement graceful fallback when LLM unavailable | HIGH | 3h |
| Add test result caching for LLM tests | MEDIUM | 3h |

#### Sprint C.2: Semantic Test Implementation
| Task | Priority | Effort |
|------|----------|--------|
| Implement `semantic_code_intent` test | HIGH | 4h |
| Implement `semantic_behavior_prediction` test | HIGH | 4h |
| Implement `semantic_jailbreak_detection` test | HIGH | 4h |
| Implement `semantic_data_flow` test | MEDIUM | 4h |
| Add to standard/full profiles | HIGH | 2h |

**Deliverables:**
- `src/benderbox/tests/semantic_tests.py`
- Updated `PROFILE_TESTS` in `sandbox_cli.py`
- Documentation for new tests

---

### **PHASE D: MCP NLP Tools**
**Goal:** Expose NLP capabilities via MCP protocol
**Duration:** 1 sprint (1 week)

#### Sprint D.1: MCP Tool Enhancement
| Task | Priority | Effort |
|------|----------|--------|
| Add `benderbox_ask` tool (NL queries) | HIGH | 4h |
| Add `benderbox_explain` tool (finding details) | HIGH | 3h |
| Add `benderbox_recommend` tool (deployment advice) | MEDIUM | 3h |
| Add `benderbox_compare` tool (model comparison) | MEDIUM | 3h |
| Update MCP server streaming support | MEDIUM | 2h |
| Add MCP tool tests | HIGH | 3h |

**Deliverables:**
- Updated `mcp_server.py` with new tools
- `tests/test_mcp_tools.py`
- MCP tool documentation

---

### **PHASE E: Documentation & Examples**
**Goal:** Make BenderBox accessible to new users
**Duration:** 1 sprint (1 week)

#### Sprint E.1: User Documentation
| Task | Priority | Effort |
|------|----------|--------|
| Update README with v3.0 features | HIGH | 3h |
| Create quick-start guide | HIGH | 3h |
| Document all NLP commands | HIGH | 3h |
| Document analysis profiles | MEDIUM | 2h |
| Create example workflows | MEDIUM | 3h |
| Add troubleshooting guide | MEDIUM | 2h |

**Deliverables:**
- Updated `README.md`
- `docs/QUICKSTART.md`
- `docs/NLP_COMMANDS.md`
- `docs/PROFILES.md`
- `docs/TROUBLESHOOTING.md`

---

### **PHASE F: Release Preparation**
**Goal:** Package for distribution
**Duration:** 1 sprint (1 week)

#### Sprint F.1: Release Engineering
| Task | Priority | Effort |
|------|----------|--------|
| Version bump to 3.0.0 | HIGH | 1h |
| Update `pyproject.toml` dependencies | HIGH | 2h |
| Create release notes | HIGH | 2h |
| Test fresh install on Windows | HIGH | 3h |
| Test fresh install on Linux | HIGH | 3h |
| Test fresh install on macOS | MEDIUM | 3h |
| Create binary distribution (optional) | LOW | 4h |
| Tag release and publish | HIGH | 1h |

**Deliverables:**
- `CHANGELOG.md`
- `RELEASE_NOTES_v3.0.md`
- GitHub Release with assets
- PyPI package (optional)

---

## Recommended Execution Order

```
┌─────────────────────────────────────────────────────────────┐
│                    RECOMMENDED PATH                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   PHASE A ──► PHASE B ──► PHASE C ──► PHASE D ──► PHASE E  │
│   Stability   Knowledge   LLM Tests   MCP Tools   Docs     │
│   (1 week)    (1 week)    (2 weeks)   (1 week)    (1 week) │
│                                                             │
│                           │                                 │
│                           ▼                                 │
│                       PHASE F                               │
│                       Release                               │
│                       (1 week)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Total Estimated Time: 7-8 weeks
```

---

## Next Phase Recommendation: **PHASE A (Stability & Testing)**

### Rationale:
1. **Protects existing work** - Regression tests ensure we don't break what works
2. **Enables confident iteration** - Tests catch issues early in subsequent phases
3. **Low risk** - Doesn't add new features that could destabilize
4. **Foundation for CI/CD** - Enables automated testing in future

### Sprint A.1 Kickoff Tasks:
```bash
# Create test structure
mkdir -p tests/integration
touch tests/integration/__init__.py
touch tests/integration/test_nlp_commands.py
touch tests/integration/test_workflows.py
touch tests/integration/test_analysis.py

# Run existing tests to establish baseline
pytest tests/ -v --tb=short
```

### Success Criteria for Phase A:
- [ ] All NLP commands have integration tests
- [ ] Workflow execution has end-to-end tests
- [ ] Analysis pipeline has regression tests
- [ ] CI runs tests on every push
- [ ] 0 regressions in existing functionality

---

## Phase Dependencies

```
Phase A (Stability) ─── Required before all other phases
     │
     ├──► Phase B (Knowledge) ─── Can run independently
     │
     ├──► Phase C (LLM Tests) ─── Depends on B for patterns
     │         │
     │         └──► Phase D (MCP) ─── Can leverage C's tests
     │
     └──► Phase E (Docs) ─── Can start after A, finalize after D
              │
              └──► Phase F (Release) ─── Requires all phases complete
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Regression in NLP | Phase A creates safety net |
| LLM tests too slow | Add caching, make optional in quick profile |
| Knowledge base staleness | Version files, document update process |
| Cross-platform issues | Test on Win/Linux/Mac in Phase F |

---

## Quick Reference: What's Done vs Remaining

| Area | Done | Remaining |
|------|------|-----------|
| NLP Core | 100% | - |
| UI/UX | 95% | Minor polish |
| Analysis Engine | 85% | Semantic tests |
| Knowledge Base | 50% | Pattern expansion |
| MCP Integration | 60% | NLP tools |
| Testing | 30% | Integration tests |
| Documentation | 40% | User guides |
| Release | 0% | Full process |

---

*Start Phase A when ready. Run `/phase A` to begin.*
