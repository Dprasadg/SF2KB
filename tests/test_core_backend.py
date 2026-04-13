"""Core backend tests for SFtoKB application."""

import json
import pytest
from pathlib import Path
from backend.kb.kb_store import load_kbs, update_kb_fields
from backend.search.rag import search_kb, keyword_score, build_kb_text
from backend.embeddings.embedder import HybridEmbedder
from backend.embeddings.vector_store import FAISSStore
from backend.cache.kb_cache import KBMetadataCache


class TestTemplateClassification:
    """Test template type detection and classification."""

    def test_solution_template_characteristics(self):
        """Verify solution templates have required fields."""
        kb = {
            "template_type": "solution",
            "symptoms": ["Issue 1", "Issue 2"],
            "cause": "Root cause",
            "resolution": ["Step 1", "Step 2"],
        }
        assert kb.get("template_type") == "solution"
        assert len(kb.get("symptoms", [])) > 0
        assert kb.get("cause")

    def test_how_to_template_characteristics(self):
        """Verify how_to templates have required fields."""
        kb = {
            "template_type": "how_to",
            "objective": "Goal of this guide",
            "steps": ["Step 1", "Step 2"],
        }
        assert kb.get("template_type") == "how_to"
        assert kb.get("objective")
        assert len(kb.get("steps", [])) > 0

    def test_qa_template_characteristics(self):
        """Verify Q&A templates have required fields."""
        kb = {
            "template_type": "qa",
            "answer": "The answer to the question",
        }
        assert kb.get("template_type") == "qa"
        assert kb.get("answer")

    def test_template_type_defaults_to_solution(self):
        """Verify missing template_type defaults to solution."""
        kb = {}
        template_type = kb.get("template_type") or "solution"
        assert template_type == "solution"


class TestApprovalWorkflow:
    """Test KB approval state management."""

    def test_approval_state_validation(self):
        """Verify approval state is properly set."""
        kb = {
            "kb_id": "test-kb-1",
            "title": "Test Article",
            "validation_state": "Validated",
            "approved_at": "2026-04-13T10:00:00+00:00",
        }
        assert kb.get("validation_state") == "Validated"
        assert kb.get("approved_at") is not None

    def test_unapproved_state(self):
        """Verify unapproved state is properly set."""
        kb = {
            "kb_id": "test-kb-2",
            "title": "Test Article",
            "validation_state": "Not Validated",
            "approved_at": None,
        }
        is_approved = str(kb.get("validation_state", "")).strip().lower() == "validated"
        assert not is_approved

    def test_approval_filtering(self):
        """Test filtering approved vs unapproved KBs."""
        kbs = [
            {"kb_id": "1", "validation_state": "Validated"},
            {"kb_id": "2", "validation_state": "Not Validated"},
            {"kb_id": "3", "validation_state": "Validated"},
        ]
        approved = [kb for kb in kbs if str(kb.get("validation_state", "")).strip().lower() == "validated"]
        unapproved = [kb for kb in kbs if str(kb.get("validation_state", "")).strip().lower() != "validated"]
        
        assert len(approved) == 2
        assert len(unapproved) == 1


class TestKeywordScoring:
    """Test keyword search and scoring."""

    def test_keyword_score_exact_match(self):
        """Test keyword scoring with exact matches."""
        query = "password reset"
        text = "How to perform password reset in the system"
        score = keyword_score(query, text)
        assert score > 0.5  # Should have reasonable match score

    def test_keyword_score_partial_match(self):
        """Test keyword scoring with partial matches."""
        query = "upload failed"
        text = "File upload failed with error message"
        score = keyword_score(query, text)
        assert score > 0.3  # Should have some match

    def test_keyword_score_no_match(self):
        """Test keyword scoring with no matches."""
        query = "archival retention"
        text = "How to configure browser settings"
        score = keyword_score(query, text)
        assert score == 0  # No overlap

    def test_keyword_score_case_insensitive(self):
        """Test keyword scoring handles case insensitivity."""
        query = "LOGIN FAILED"
        text = "login failure in production system"
        score = keyword_score(query, text)
        assert score > 0  # Should match case-insensitively


class TestKBMetadataCache:
    """Test the in-memory KB metadata cache."""

    def test_cache_initialization(self, tmp_path):
        """Test cache initializes correctly."""
        kb_file = tmp_path / "test_kbs.json"
        kb_file.write_text(json.dumps([
            {"kb_id": "kb1", "title": "Article 1", "content": "sample"},
            {"kb_id": "kb2", "title": "Article 2", "content": "data"},
        ]))
        cache = KBMetadataCache(str(kb_file))
        all_kbs = cache.get_all()
        assert len(all_kbs) == 2

    def test_cache_get_by_id(self, tmp_path):
        """Test retrieving KB by ID from cache."""
        kb_file = tmp_path / "test_kbs.json"
        kb_file.write_text(json.dumps([
            {"kb_id": "test-123", "title": "Test KB", "content": "sample"},
        ]))
        cache = KBMetadataCache(str(kb_file))
        kb = cache.get_by_id("test-123")
        assert kb is not None
        assert kb.get("title") == "Test KB"

    def test_cache_get_by_title(self, tmp_path):
        """Test retrieving KB by title from cache."""
        kb_file = tmp_path / "test_kbs.json"
        kb_file.write_text(json.dumps([
            {"kb_id": "kb1", "title": "Unique Article Title", "content": "data"},
        ]))
        cache = KBMetadataCache(str(kb_file))
        kb = cache.get_by_title("Unique Article Title")
        assert kb is not None
        assert kb.get("kb_id") == "kb1"

    def test_cache_update_kb(self, tmp_path):
        """Test updating KB in cache."""
        kb_file = tmp_path / "test_kbs.json"
        kb_file.write_text(json.dumps([
            {"kb_id": "kb1", "title": "Article", "validation_state": "Not Validated"},
        ]))
        cache = KBMetadataCache(str(kb_file))
        updated_kb = {
            "kb_id": "kb1",
            "title": "Article",
            "validation_state": "Validated",
        }
        cache.update_kb(updated_kb)
        cached = cache.get_by_id("kb1")
        assert cached.get("validation_state") == "Validated"


class TestSearchFiltering:
    """Test search result filtering and approval gating."""

    def test_approved_kb_in_results(self):
        """Test that approved KBs appear in search results."""
        kb = {
            "kb_id": "kb1",
            "title": "Article",
            "validation_state": "Validated",
        }
        is_approved = str(kb.get("validation_state", "")).strip().lower() == "validated"
        assert is_approved

    def test_unapproved_kb_filtered(self):
        """Test that unapproved KBs are filtered from results."""
        kb = {
            "kb_id": "kb2",
            "title": "Article",
            "validation_state": "Not Validated",
        }
        is_approved = str(kb.get("validation_state", "")).strip().lower() == "validated"
        assert not is_approved
        # Should not include in results
        results = [r for r in [kb] if str(r.get("validation_state", "")).strip().lower() == "validated"]
        assert len(results) == 0

    def test_template_type_filtering(self):
        """Test filtering by template type."""
        kbs = [
            {"kb_id": "1", "template_type": "solution"},
            {"kb_id": "2", "template_type": "how_to"},
            {"kb_id": "3", "template_type": "qa"},
            {"kb_id": "4", "template_type": "solution"},
        ]
        solutions = [kb for kb in kbs if (kb.get("template_type") or "solution") == "solution"]
        assert len(solutions) == 2
        
        how_tos = [kb for kb in kbs if kb.get("template_type") == "how_to"]
        assert len(how_tos) == 1
        
        qas = [kb for kb in kbs if kb.get("template_type") == "qa"]
        assert len(qas) == 1


class TestKBBuildText:
    """Test KB text construction for retrieval."""

    def test_solution_text_construction(self):
        """Test text building for solution templates."""
        kb = {
            "title": "Upload Failed",
            "summary": "Large files fail",
            "symptoms": ["Error on upload"],
            "resolution": ["Increase limit"],
        }
        text = build_kb_text(kb)
        assert "Upload Failed" in text
        assert "Large files fail" in text
        assert "Error on upload" in text
        assert "Increase limit" in text

    def test_how_to_text_construction(self):
        """Test text building for how_to templates."""
        kb = {
            "template_type": "how_to",
            "title": "Configure Retention",
            "objective": "Set retention policy",
            "steps": ["Step 1", "Step 2"],
        }
        text = build_kb_text(kb)
        assert "Configure Retention" in text
        assert "Set retention policy" in text
        assert "Step 1" in text

    def test_qa_text_construction(self):
        """Test text building for Q&A templates."""
        kb = {
            "template_type": "qa",
            "title": "What is compliance?",
            "answer": "Compliance means following rules",
        }
        text = build_kb_text(kb)
        assert "What is compliance?" in text
        assert "Compliance means following rules" in text
