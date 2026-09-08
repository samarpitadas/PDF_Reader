from pathlib import Path


def test_upload_ready_lifecycle_database_side(app_module, temp_app_paths):
    app_module.init_database()
    chat_id = app_module.create_chat()
    document_path = str(Path(temp_app_paths) / 'sample.pdf')
    Path(document_path).write_bytes(b'%PDF-test')

    app_module.save_document(chat_id, 'sample.pdf', document_path, 'hash123')

    docs = app_module.get_documents(chat_id)
    assert len(docs) == 1
    assert docs[0]['filename'] == 'sample.pdf'
    assert docs[0]['file_hash'] == 'hash123'
    assert app_module.document_exists(chat_id, 'hash123') is True


def test_duplicate_document_is_detected(app_module, temp_app_paths):
    app_module.init_database()
    chat_id = app_module.create_chat()
    path = str(Path(temp_app_paths) / 'sample.pdf')
    Path(path).write_bytes(b'x')
    app_module.save_document(chat_id, 'sample.pdf', path, 'same-hash')

    assert app_module.document_exists(chat_id, 'same-hash') is True
    assert app_module.document_exists(chat_id, 'other-hash') is False


def test_delete_removes_milvus_records_for_chat(app_module, temp_app_paths, monkeypatch):
    app_module.init_database()
    chat_id = app_module.create_chat()

    class FakeMilvus:
        def __init__(self):
            self.calls = []
        def delete(self, **kwargs):
            self.calls.append(kwargs)

    fake = FakeMilvus()
    monkeypatch.setattr(app_module, 'ensure_milvus_collection', lambda: fake)

    app_module.delete_chat(chat_id)

    assert fake.calls
    assert fake.calls[0]['collection_name'] == app_module.MILVUS_COLLECTION
    assert fake.calls[0]['filter'] == f'chat_id == "{chat_id}"'
    assert app_module.get_chat(chat_id) is None


def test_process_pdf_upload_ready_lifecycle(app_module, temp_app_paths, monkeypatch):
    app_module.init_database()
    chat_id = app_module.create_chat()
    chunks = [type('Chunk', (), {'page_content': 'hello', 'metadata': {'page': 0}})()]

    monkeypatch.setattr(app_module, 'extract_pdf_documents', lambda path: [type('Page', (), {'page_content': 'hello', 'metadata': {'page': 0}})()])
    monkeypatch.setattr(app_module, 'split_documents', lambda docs: chunks)
    monkeypatch.setattr(app_module, 'add_documents_to_vector_store', lambda chat_id, chunks, ids: 'FAKE_MILVUS')
    monkeypatch.setattr(app_module, 'write_structured_log', lambda *a, **k: None)

    vector_store, added, processing_time = app_module.process_pdf(chat_id, b'fake-pdf-bytes', 'sample.pdf')

    assert vector_store == 'FAKE_MILVUS'
    assert added is True
    assert processing_time >= 0
    docs = app_module.get_documents(chat_id)
    assert len(docs) == 1
    assert docs[0]['filename'] == 'sample.pdf'
    assert app_module.document_exists(chat_id, app_module.calculate_file_hash(b'fake-pdf-bytes')) is True
