
def doc(chunk_id, source, rank, score):
    return {
        'page_content': chunk_id,
        'metadata': {
            'chunk_id': chunk_id,
            'retrieval_source': source,
            'retrieval_rank': rank,
            'retrieval_score': score,
        },
    }


def test_rrf_fusion_deduplicates_and_rewards_overlap(app_module):
    dense = [doc('A', 'dense', 1, 0.9), doc('B', 'dense', 2, 0.8)]
    sparse = [doc('B', 'sparse_bm25', 1, 10.0), doc('C', 'sparse_bm25', 2, 9.0)]

    fused, records = app_module._rrf_fuse(dense, sparse)

    ids = [d['metadata']['chunk_id'] for d in fused]
    assert len(ids) == 3
    assert len(ids) == len(set(ids))
    assert ids[0] == 'B'
    assert records[0]['chunk_id'] == 'B'
    assert records[0]['dense_rank'] == 2
    assert records[0]['sparse_rank'] == 1


def test_rrf_score_uses_rank_not_raw_score_scale(app_module):
    dense = [doc('A', 'dense', 1, 0.99)]
    sparse = [doc('B', 'sparse_bm25', 1, 9999.0)]

    fused, _ = app_module._rrf_fuse(dense, sparse)

    assert fused[0]['metadata']['rrf_score'] == fused[1]['metadata']['rrf_score']


def test_rrf_respects_fusion_k(app_module, monkeypatch):
    monkeypatch.setattr(app_module, 'FUSION_K', 2)
    dense = [doc(f'D{i}', 'dense', i + 1, 1.0) for i in range(5)]
    sparse = [doc(f'S{i}', 'sparse_bm25', i + 1, 1.0) for i in range(5)]

    fused, records = app_module._rrf_fuse(dense, sparse)

    assert len(fused) == 2
    assert len(records) == 2
