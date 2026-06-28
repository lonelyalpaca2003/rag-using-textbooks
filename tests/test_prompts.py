import pytest
from src.prompts import QUIZ_GENERATION, SUMMARIZE_LECTURE, FIND_TEXTBOOK_PAGES, EXAM_PREP


class TestQuizGenerationTemplate:

    def test_contains_source_placeholder(self):
        assert "{source}" in QUIZ_GENERATION

    def test_contains_num_questions_placeholder(self):
        assert "{num_questions}" in QUIZ_GENERATION

    def test_formats_with_valid_args(self):
        result = QUIZ_GENERATION.format(source="ST443_Lecture 3", num_questions=5)
        assert "ST443_Lecture 3" in result
        assert "5" in result

    def test_no_unresolved_placeholders_after_format(self):
        result = QUIZ_GENERATION.format(source="Lecture 1", num_questions=3)
        assert "{" not in result


class TestSummarizeLectureTemplate:

    def test_is_non_empty_string(self):
        assert isinstance(SUMMARIZE_LECTURE, str) and len(SUMMARIZE_LECTURE) > 0

    def test_covers_key_concepts_section(self):
        assert "Key Concepts" in SUMMARIZE_LECTURE

    def test_covers_mathematical_formulas_section(self):
        assert "Mathematical Formulas" in SUMMARIZE_LECTURE

    def test_format_call_does_not_raise(self):
        # Template accepts but doesn't use the `lecture` kwarg — format() must not raise
        result = SUMMARIZE_LECTURE.format(lecture="ST443_Lecture_5")
        assert isinstance(result, str)

    def test_no_unresolved_placeholders_in_template(self):
        # Template has no format placeholders that need substitution
        assert "{" not in SUMMARIZE_LECTURE


class TestFindTextbookPagesTemplate:

    def test_contains_textbook_placeholder(self):
        assert "{textbook}" in FIND_TEXTBOOK_PAGES

    def test_contains_topic_placeholder(self):
        assert "{topic}" in FIND_TEXTBOOK_PAGES

    def test_formats_with_valid_args(self):
        result = FIND_TEXTBOOK_PAGES.format(textbook="ISLR", topic="ridge regression")
        assert "ISLR" in result
        assert "ridge regression" in result

    def test_no_unresolved_placeholders_after_format(self):
        result = FIND_TEXTBOOK_PAGES.format(textbook="ISLR", topic="regression")
        assert "{" not in result


class TestExamPrepTemplate:

    def test_contains_topic_placeholder(self):
        assert "{topic}" in EXAM_PREP

    def test_formats_with_valid_args(self):
        result = EXAM_PREP.format(topic="regularization")
        assert "regularization" in result

    def test_no_unresolved_placeholders_after_format(self):
        result = EXAM_PREP.format(topic="SVMs")
        assert "{" not in result
