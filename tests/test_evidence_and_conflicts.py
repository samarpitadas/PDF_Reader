from langchain_core.documents import Document


def D(text, source):
    return Document(page_content=text, metadata={'source_file': source, 'page': 0, 'chunk_id': source + '-1'})


def test_answerability_policy_marks_missing_evidence(app_module):
    docs = [D('The paper discusses Paxos.', 'a.pdf')]
    assert app_module.assess_evidence("I couldn't find that information in the uploaded PDFs.", docs) == 'NOT FOUND IN UPLOADED PDFs'


def test_answerability_policy_marks_supported_answer(app_module):
    docs = [D('Paxos is a consensus algorithm.', 'a.pdf')]
    assert app_module.assess_evidence('Paxos is a consensus algorithm. [S1]', docs) == 'SUPPORTED BY RETRIEVED PDF EVIDENCE'


def test_empty_retrieval_is_not_answerable(app_module):
    assert app_module.assess_evidence('Anything', []) == 'NO RETRIEVED EVIDENCE'


def test_numeric_cross_document_conflict_is_flagged(app_module):
    docs = [
        D('Accuracy was 91%.', 'a.pdf'),
        D('Accuracy was 87%.', 'b.pdf'),
    ]
    warnings = app_module.detect_cross_document_conflicts(docs)
    assert warnings
    assert 'a.pdf' in warnings[0]
    assert 'b.pdf' in warnings[0]


def test_same_numeric_evidence_does_not_trigger_conflict(app_module):
    docs = [
        D('Accuracy was 91%.', 'a.pdf'),
        D('Accuracy was 91%.', 'b.pdf'),
    ]
    assert app_module.detect_cross_document_conflicts(docs) == []


def test_evidence_coverage_counts_unique_valid_source_citations(app_module):
    docs = [D('Evidence A', 'a.pdf'), D('Evidence B', 'b.pdf')]
    if not hasattr(app_module, 'calculate_evidence_coverage'):
        raise AssertionError(
            'Add calculate_evidence_coverage(answer, docs) to app.py before running this test.'
        )

    coverage = app_module.calculate_evidence_coverage('Claim one [S1]. Claim two [S1]. Claim three [S2].', docs)
    assert coverage == 1.0


def test_evidence_coverage_ignores_invalid_source_ids(app_module):
    docs = [D('Evidence A', 'a.pdf'), D('Evidence B', 'b.pdf')]
    if not hasattr(app_module, 'calculate_evidence_coverage'):
        raise AssertionError(
            'Add calculate_evidence_coverage(answer, docs) to app.py before running this test.'
        )

    coverage = app_module.calculate_evidence_coverage('Claim [S1]. Invalid [S9].', docs)
    assert coverage == 0.5
