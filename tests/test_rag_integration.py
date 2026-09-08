import re


def _milvus_result(chunk_id, chat_id, source_file, page, text, distance):
    return {
        'id': chunk_id,
        'distance': distance,
        'entity': {
            'chunk_id': chunk_id,
            'chat_id': chat_id,
            'source_file': source_file,
            'file_hash': 'hash',
            'page': page,
            'chunk_index': 0,
            'text': text,
        },
    }


class FakeClient:
    def __init__(self, dense_rows, sparse_rows):
        self.dense_rows = dense_rows
        self.sparse_rows = sparse_rows
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs['anns_field'] == 'dense_vector':
            return [self.dense_rows]
        if kwargs['anns_field'] == 'sparse_vector':
            return [self.sparse_rows]
        raise AssertionError('Unexpected search field')


def test_selected_document_filter_prevents_leakage(app_module, monkeypatch):
    chat_id = 'chat-A'
    client = FakeClient(
        [_milvus_result('A1', chat_id, 'selected.pdf', 0, 'Selected evidence', 0.9)],
        [_milvus_result('A1', chat_id, 'selected.pdf', 0, 'Selected evidence', 8.0)],
    )

    monkeypatch.setattr(app_module, 'embeddings', type('E', (), {'embed_query': lambda self, q: [0.1, 0.2]})())
    monkeypatch.setattr(app_module, 'jina_rerank', lambda q, docs, top_n: (docs[:top_n], {'model': 'test'}, 0.001))
    monkeypatch.setattr(app_module, 'invoke_llama_with_metrics', lambda prompt: ('Answer [S1]', {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}))
    monkeypatch.setattr(app_module, 'write_structured_log', lambda *a, **k: None)

    result = app_module.generate_response(client, 'What is the fact?', [], chat_id)
    matching_docs = result[4]

    assert len(client.calls) == 2
    assert all(call['filter'] == f'chat_id == "{chat_id}"' for call in client.calls)
    assert client.calls[0]['limit'] == app_module.DENSE_K
    assert client.calls[1]['limit'] == app_module.SPARSE_K
    assert client.calls[1]['data'] == ['What is the fact?']
    assert all(d.metadata['chat_id'] == chat_id for d in matching_docs)
    assert all(d.metadata['source_file'] == 'selected.pdf' for d in matching_docs)


def test_dense_bm25_rrf_jina_llama_pipeline(app_module, monkeypatch):
    chat_id = 'chat-1'
    client = FakeClient(
        [
            _milvus_result('A', chat_id, 'paper.pdf', 0, 'Dense evidence', 0.95),
            _milvus_result('B', chat_id, 'paper.pdf', 1, 'Another evidence', 0.90),
        ],
        [
            _milvus_result('B', chat_id, 'paper.pdf', 1, 'Another evidence', 20.0),
            _milvus_result('A', chat_id, 'paper.pdf', 0, 'Dense evidence', 15.0),
        ],
    )

    monkeypatch.setattr(app_module, 'embeddings', type('E', (), {'embed_query': lambda self, q: [0.1, 0.2]})())
    def fake_jina(query, docs, top_n):
        docs = list(reversed(docs))[:top_n]
        for d in docs:
            d.metadata['jina_score'] = 0.99
        return docs, {'model': 'jina-reranker-v3.5', 'candidate_count': 2, 'returned_count': len(docs)}, 0.002
    monkeypatch.setattr(app_module, 'jina_rerank', fake_jina)
    monkeypatch.setattr(app_module, 'invoke_llama_with_metrics', lambda prompt: ('Final answer [S1]', {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}))
    captured = {}
    monkeypatch.setattr(app_module, 'write_structured_log', lambda *a, **k: captured.update(k))

    result = app_module.generate_response(client, 'What is the fact?', [], chat_id)
    answer = result[0]
    matching_docs = result[4]
    stage_timings = result[7]

    assert answer == 'Final answer [S1]'
    assert len(matching_docs) <= app_module.RERANK_K
    assert 'dense_retrieval_s' in stage_timings
    assert 'bm25_retrieval_s' in stage_timings
    assert 'rrf_fusion_s' in stage_timings
    assert 'jina_rerank_s' in stage_timings
    assert 'llama_generation_s' in stage_timings
    assert 'total_pipeline_s' in stage_timings
    assert captured['retrieval']['dense_k'] == app_module.DENSE_K
    assert captured['retrieval']['sparse_k'] == app_module.SPARSE_K
    assert captured['fusion']['method'] == 'RRF'
    assert captured['reranking']['model'] == 'jina-reranker-v3.5'


def test_every_returned_citation_maps_to_retrieved_source(app_module, monkeypatch):
    chat_id = 'chat-1'
    client = FakeClient(
        [_milvus_result('A', chat_id, 'paper.pdf', 2, 'Evidence', 0.9)],
        [_milvus_result('A', chat_id, 'paper.pdf', 2, 'Evidence', 8.0)],
    )
    monkeypatch.setattr(app_module, 'embeddings', type('E', (), {'embed_query': lambda self, q: [0.1]})())
    monkeypatch.setattr(app_module, 'jina_rerank', lambda q, docs, top_n: (docs, {'model': 'test'}, 0.001))
    monkeypatch.setattr(app_module, 'invoke_llama_with_metrics', lambda prompt: ('The answer is X [S1].', {'total_tokens': 1}))
    monkeypatch.setattr(app_module, 'write_structured_log', lambda *a, **k: None)

    result = app_module.generate_response(client, 'question', [], chat_id)
    answer, matching_docs = result[0], result[4]

    cited_ids = set(re.findall(r'\[S(\d+)\]', answer))
    assert cited_ids
    assert all(1 <= int(i) <= len(matching_docs) for i in cited_ids)


def test_unsupported_question_results_in_abstention(app_module, monkeypatch):
    chat_id = 'chat-1'
    client = FakeClient(
        [_milvus_result('A', chat_id, 'paper.pdf', 0, 'Paxos is a consensus algorithm.', 0.9)],
        [_milvus_result('A', chat_id, 'paper.pdf', 0, 'Paxos is a consensus algorithm.', 8.0)],
    )
    monkeypatch.setattr(app_module, 'embeddings', type('E', (), {'embed_query': lambda self, q: [0.1]})())
    monkeypatch.setattr(app_module, 'jina_rerank', lambda q, docs, top_n: (docs, {'model': 'test'}, 0.001))
    monkeypatch.setattr(app_module, 'invoke_llama_with_metrics', lambda prompt: ("I couldn't find that information in the uploaded PDFs.", {'total_tokens': 1}))
    monkeypatch.setattr(app_module, 'write_structured_log', lambda *a, **k: None)

    result = app_module.generate_response(client, 'What is the weather on Mars?', [], chat_id)

    assert result[0].startswith("I couldn't find")
    assert result[5] == 'NOT FOUND IN UPLOADED PDFs'
