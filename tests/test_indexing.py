import pytest
from unittest.mock import MagicMock
from llama_index.core import Document
from chromadb.errors import NotFoundError
from src.indexing import load_documents_with_metadata_included, create_or_load_vector_store


def make_mock_reader(pages_text):
    """Build a pypdf.PdfReader mock with the given per-page text strings."""
    mock_reader = MagicMock()
    mock_reader.pages = []
    for text in pages_text:
        page = MagicMock()
        page.extract_text.return_value = text
        mock_reader.pages.append(page)
    return mock_reader


class TestLoadDocumentsWithMetadata:

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert load_documents_with_metadata_included(str(tmp_path)) == []

    def test_non_pdf_files_are_skipped(self, tmp_path):
        (tmp_path / "notes.txt").write_text("notes")
        (tmp_path / "image.png").write_bytes(b"img")
        assert load_documents_with_metadata_included(str(tmp_path)) == []

    def test_lecture_filename_produces_lecture_doc_type(self, tmp_path, mocker):
        (tmp_path / "ST443_Lecture_1.pdf").write_bytes(b"pdf")
        mocker.patch("pypdf.PdfReader", return_value=make_mock_reader(["Lecture text"]))
        docs = load_documents_with_metadata_included(str(tmp_path))
        assert docs[0].metadata["doc_type"] == "lecture"

    def test_non_lecture_pdf_produces_textbook_doc_type(self, tmp_path, mocker):
        (tmp_path / "ISLRv2.pdf").write_bytes(b"pdf")
        mocker.patch("pypdf.PdfReader", return_value=make_mock_reader(["Textbook text"]))
        docs = load_documents_with_metadata_included(str(tmp_path))
        assert docs[0].metadata["doc_type"] == "textbook"

    def test_blank_pages_are_skipped(self, tmp_path, mocker):
        (tmp_path / "ISLRv2.pdf").write_bytes(b"pdf")
        mocker.patch(
            "pypdf.PdfReader",
            return_value=make_mock_reader(["Content", "   \n", "", "More content"]),
        )
        docs = load_documents_with_metadata_included(str(tmp_path))
        assert len(docs) == 2

    def test_metadata_contains_required_keys(self, tmp_path, mocker):
        (tmp_path / "ISLRv2.pdf").write_bytes(b"pdf")
        mocker.patch("pypdf.PdfReader", return_value=make_mock_reader(["Content"]))
        docs = load_documents_with_metadata_included(str(tmp_path))
        assert {"file_name", "page_num", "doc_type", "course"}.issubset(docs[0].metadata)

    def test_file_name_in_metadata_matches_filename(self, tmp_path, mocker):
        (tmp_path / "ST443_Lecture_3.pdf").write_bytes(b"pdf")
        mocker.patch("pypdf.PdfReader", return_value=make_mock_reader(["Content"]))
        docs = load_documents_with_metadata_included(str(tmp_path))
        assert docs[0].metadata["file_name"] == "ST443_Lecture_3.pdf"

    def test_page_numbers_are_zero_indexed_and_sequential(self, tmp_path, mocker):
        (tmp_path / "ISLRv2.pdf").write_bytes(b"pdf")
        mocker.patch(
            "pypdf.PdfReader", return_value=make_mock_reader(["P1", "P2", "P3"])
        )
        docs = load_documents_with_metadata_included(str(tmp_path))
        assert [d.metadata["page_num"] for d in docs] == [0, 1, 2]

    def test_course_metadata_value(self, tmp_path, mocker):
        (tmp_path / "ISLRv2.pdf").write_bytes(b"pdf")
        mocker.patch("pypdf.PdfReader", return_value=make_mock_reader(["Content"]))
        docs = load_documents_with_metadata_included(str(tmp_path))
        assert docs[0].metadata["course"] == "Machine Learning"

    def test_returns_document_instances(self, tmp_path, mocker):
        (tmp_path / "ISLRv2.pdf").write_bytes(b"pdf")
        mocker.patch("pypdf.PdfReader", return_value=make_mock_reader(["Content"]))
        docs = load_documents_with_metadata_included(str(tmp_path))
        assert all(isinstance(d, Document) for d in docs)

    def test_document_text_matches_page_text(self, tmp_path, mocker):
        (tmp_path / "ISLRv2.pdf").write_bytes(b"pdf")
        mocker.patch("pypdf.PdfReader", return_value=make_mock_reader(["Hello world"]))
        docs = load_documents_with_metadata_included(str(tmp_path))
        assert docs[0].text == "Hello world"

    def test_multiple_pdfs_each_produce_documents(self, tmp_path, mocker):
        (tmp_path / "ST443_Lecture_1.pdf").write_bytes(b"pdf")
        (tmp_path / "ISLRv2.pdf").write_bytes(b"pdf")
        mocker.patch("pypdf.PdfReader", return_value=make_mock_reader(["Content"]))
        docs = load_documents_with_metadata_included(str(tmp_path))
        assert len(docs) == 2


class TestCreateOrLoadVectorStore:

    @pytest.fixture(autouse=True)
    def mock_deps(self, mocker):
        self.mock_client = MagicMock()
        mocker.patch("chromadb.PersistentClient", return_value=self.mock_client)
        mocker.patch("src.indexing.HuggingFaceEmbedding")
        # Patch Settings to skip llama_index's isinstance(embed_model, BaseEmbedding) check
        mocker.patch("src.indexing.Settings")

    def test_creates_new_collection_when_not_found(self):
        self.mock_client.get_collection.side_effect = NotFoundError("new_db")
        mock_collection = MagicMock()
        self.mock_client.create_collection.return_value = mock_collection

        result = create_or_load_vector_store("new_db")

        self.mock_client.create_collection.assert_called_once_with("new_db")
        assert result == mock_collection

    def test_loads_existing_non_empty_collection(self):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        self.mock_client.get_collection.return_value = mock_collection

        result = create_or_load_vector_store("existing_db")

        assert result == mock_collection
        self.mock_client.create_collection.assert_not_called()
        self.mock_client.delete_collection.assert_not_called()

    def test_empty_collection_is_deleted_and_recreated(self):
        mock_empty = MagicMock()
        mock_empty.count.return_value = 0
        mock_new = MagicMock()
        self.mock_client.get_collection.return_value = mock_empty
        self.mock_client.create_collection.return_value = mock_new

        result = create_or_load_vector_store("stale_db")

        self.mock_client.delete_collection.assert_called_once_with("stale_db")
        self.mock_client.create_collection.assert_called_once_with("stale_db")
        assert result == mock_new

    def test_default_db_name_is_ml_notes(self):
        self.mock_client.get_collection.side_effect = NotFoundError("ml_notes")
        self.mock_client.create_collection.return_value = MagicMock()

        create_or_load_vector_store()

        self.mock_client.create_collection.assert_called_once_with("ml_notes")
