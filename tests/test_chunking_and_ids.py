from langchain_core.documents import Document


def test_page_aware_chunking(app_module):
    text_a = ' '.join(['page-one'] * 250)
    text_b = ' '.join(['page-two'] * 250)
    docs = [
        Document(page_content=text_a, metadata={'page': 0, 'source': 'a.pdf'}),
        Document(page_content=text_b, metadata={'page': 1, 'source': 'a.pdf'}),
    ]

    chunks = app_module.split_documents(docs)

    assert chunks
    assert all('page' in chunk.metadata for chunk in chunks)
    assert {chunk.metadata['page'] for chunk in chunks} == {0, 1}
    assert all(len(chunk.page_content) <= app_module.CHUNK_SIZE for chunk in chunks)


def test_chunk_ids_are_deterministic(app_module):
    args = ('chat1', 'abc123', 2, 4, 'same text')
    assert app_module.make_stable_chunk_id(*args) == app_module.make_stable_chunk_id(*args)


def test_chunk_id_changes_when_identity_changes(app_module):
    base = ('chat1', 'abc123', 2, 4, 'same text')
    variants = [
        ('chat2', 'abc123', 2, 4, 'same text'),
        ('chat1', 'different', 2, 4, 'same text'),
        ('chat1', 'abc123', 3, 4, 'same text'),
        ('chat1', 'abc123', 2, 5, 'same text'),
        ('chat1', 'abc123', 2, 4, 'different text'),
    ]
    original = app_module.make_stable_chunk_id(*base)
    for variant in variants:
        assert app_module.make_stable_chunk_id(*variant) != original
