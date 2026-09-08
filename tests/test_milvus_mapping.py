
def test_milvus_result_mapping(app_module):
    result = {
        'id': 'fallback-id',
        'distance': 0.88,
        'entity': {
            'chunk_id': 'chunk-123',
            'chat_id': 'chat-1',
            'source_file': 'paper.pdf',
            'file_hash': 'abc',
            'page': 4,
            'chunk_index': 2,
            'text': 'Paxos text',
        },
    }

    mapped = app_module._milvus_doc_from_result(result, 'dense', 1, 0.88)

    assert mapped['page_content'] == 'Paxos text'
    assert mapped['metadata']['chunk_id'] == 'chunk-123'
    assert mapped['metadata']['source_file'] == 'paper.pdf'
    assert mapped['metadata']['page'] == 4
    assert mapped['metadata']['retrieval_source'] == 'dense'
    assert mapped['metadata']['retrieval_rank'] == 1
    assert mapped['metadata']['retrieval_score'] == 0.88
