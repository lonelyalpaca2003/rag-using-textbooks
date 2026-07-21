"""
Tests for app.py helper functions.

app.py calls create_query_engine() at module level. conftest.py patches it before
this module is imported, so `app.query_engine` and `app.index` are MagicMocks
for the entire test session.
"""

import pytest
from unittest.mock import MagicMock

import app


class TestAskQuestion:

    def test_passes_prompt_to_query_engine(self, mock_engine):
        mock_engine.query.return_value.response = "Answer"
        app.ask_question("What is gradient descent?")
        mock_engine.query.assert_called_once_with("What is gradient descent?")

    def test_returns_response_text(self, mock_engine):
        mock_engine.query.return_value.response = "Expected answer"
        result = app.ask_question("Any question?")
        assert result == "Expected answer"


class TestGenerateQuiz:

    def _make_mock_qe(self, mock_index, response_text="Q1: ..."):
        mock_response = MagicMock()
        mock_response.response = response_text
        mock_qe = MagicMock()
        mock_qe.query.return_value = mock_response
        mock_index.as_query_engine.return_value = mock_qe
        return mock_qe

    def test_filters_by_correct_lecture_filename(self, mock_index):
        self._make_mock_qe(mock_index)
        app.generate_quiz(lecture_num=5, num_questions=3)
        filters = mock_index.as_query_engine.call_args[1]["filters"]
        filter_values = [f.value for f in filters.filters]
        assert "ST443_Lecture_5.pdf" in filter_values

    def test_lecture_number_appears_in_query(self, mock_index):
        mock_qe = self._make_mock_qe(mock_index)
        app.generate_quiz(lecture_num=5, num_questions=3)
        prompt = mock_qe.query.call_args[0][0]
        assert "ST443_Lecture_5.pdf" in prompt

    def test_num_questions_appears_in_query(self, mock_index):
        mock_qe = self._make_mock_qe(mock_index)
        app.generate_quiz(lecture_num=2, num_questions=7)
        prompt = mock_qe.query.call_args[0][0]
        assert "7" in prompt

    def test_returns_response_text(self, mock_index):
        self._make_mock_qe(mock_index, response_text="Quiz content")
        result = app.generate_quiz(3, 5)
        assert result == "Quiz content"


class TestFindTextbookPages:

    def test_unknown_textbook_returns_error_message(self, mock_index):
        result = app.find_textbook_pages("Unknown Book", "regression")
        assert "not found" in result.lower()
        assert "Unknown Book" in result

    def _make_mock_qe(self, mock_index, response_text="Pages found"):
        mock_response = MagicMock()
        mock_response.response = response_text
        mock_response.source_nodes = []
        mock_qe = MagicMock()
        mock_qe.query.return_value = mock_response
        mock_index.as_query_engine.return_value = mock_qe
        return mock_qe

    def test_islr_uses_correct_filename(self, mock_index):
        self._make_mock_qe(mock_index)
        app.find_textbook_pages("ISLR", "ridge regression")
        filters = mock_index.as_query_engine.call_args[1]["filters"]
        filter_values = [f.value for f in filters.filters]
        assert "ISLRv2_corrected_June_2023.pdf" in filter_values

    def test_esl_uses_correct_filename(self, mock_index):
        self._make_mock_qe(mock_index)
        app.find_textbook_pages("Elements of Statistical Learning", "boosting")
        filters = mock_index.as_query_engine.call_args[1]["filters"]
        filter_values = [f.value for f in filters.filters]
        assert "elements_of_statistical_learning.pdf" in filter_values

    def test_filters_by_textbook_doc_type(self, mock_index):
        self._make_mock_qe(mock_index)
        app.find_textbook_pages("ISLR", "lasso")
        filters = mock_index.as_query_engine.call_args[1]["filters"]
        filter_values = [f.value for f in filters.filters]
        assert "textbook" in filter_values

    def test_returns_response_text(self, mock_index):
        self._make_mock_qe(mock_index, response_text="Page 42: ridge regression")
        result = app.find_textbook_pages("ISLR", "ridge regression")
        assert "Page 42: ridge regression" in result


class TestSummarizeLecture:

    def _make_mock_qe(self, mock_index, response_text="Summary"):
        mock_response = MagicMock()
        mock_response.response = response_text
        mock_qe = MagicMock()
        mock_qe.query.return_value = mock_response
        mock_index.as_query_engine.return_value = mock_qe
        return mock_qe

    def test_filter_uses_correct_lecture_filename(self, mock_index):
        self._make_mock_qe(mock_index)
        app.summarize_lecture(7)
        filters = mock_index.as_query_engine.call_args[1]["filters"]
        filter_values = [f.value for f in filters.filters]
        assert "ST443_Lecture_7.pdf" in filter_values

    def test_filter_value_reflects_lecture_number(self, mock_index):
        self._make_mock_qe(mock_index)
        app.summarize_lecture(3)
        filters = mock_index.as_query_engine.call_args[1]["filters"]
        filter_values = [f.value for f in filters.filters]
        assert "ST443_Lecture_3.pdf" in filter_values

    def test_uses_high_top_k_for_full_coverage(self, mock_index):
        self._make_mock_qe(mock_index)
        app.summarize_lecture(1)
        kwargs = mock_index.as_query_engine.call_args[1]
        assert kwargs.get("similarity_top_k", 0) >= 20

    def test_returns_response_text(self, mock_index):
        self._make_mock_qe(mock_index, response_text="Lecture summary text")
        result = app.summarize_lecture(3)
        assert result == "Lecture summary text"


class TestExamPrep:

    def test_topic_appears_in_query(self, mock_engine):
        mock_engine.query.return_value.response = "Study guide"
        app.exam_prep("SVMs")
        prompt = mock_engine.query.call_args[0][0]
        assert "SVMs" in prompt

    def test_returns_response_text(self, mock_engine):
        mock_engine.query.return_value.response = "Exam guide"
        result = app.exam_prep("neural networks")
        assert result == "Exam guide"
