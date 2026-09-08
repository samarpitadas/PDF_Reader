import ast
import importlib.util
from pathlib import Path

import pytest
import streamlit as st


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="session")
def app_module():
    if not APP_PATH.exists():
        raise FileNotFoundError(
            f"Could not find app.py at: {APP_PATH}"
        )

    # Give Streamlit the session-state values that the real UI
    # normally creates before the application logic uses them.
    defaults = {
        "current_chat_id": None,
        "current_chat": None,
        "messages": [],
        "uploaded_files": [],
        "selected_documents": [],
        "chat_history": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(APP_PATH))

    selected = []

    for node in tree.body:

        # Keep imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            selected.append(node)
            continue

        # Keep configuration/constants
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            selected.append(node)
            continue

        # Keep functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            selected.append(node)
            continue

        # Keep try/except blocks used for optional imports/configuration
        if isinstance(node, ast.Try):
            selected.append(node)
            continue

        # Keep module docstrings
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
        ):
            selected.append(node)
            continue

        # IMPORTANT:
        # Skip top-level if blocks because those normally contain
        # Streamlit UI/session-state initialization.
        if isinstance(node, ast.If):
            continue

    module_ast = ast.Module(
        body=selected,
        type_ignores=[]
    )

    ast.fix_missing_locations(module_ast)

    spec = importlib.util.spec_from_loader(
        "app_under_test",
        loader=None
    )

    module = importlib.util.module_from_spec(spec)

    exec(
        compile(module_ast, str(APP_PATH), "exec"),
        module.__dict__
    )

    return module


@pytest.fixture
def temp_app_paths(tmp_path, app_module, monkeypatch):
    app_dir = tmp_path / "rag_data"
    chats_dir = app_dir / "chats"

    app_dir.mkdir()
    chats_dir.mkdir()

    monkeypatch.setattr(
        app_module,
        "APP_DIR",
        str(app_dir)
    )

    monkeypatch.setattr(
        app_module,
        "DB_PATH",
        str(app_dir / "chat_history.db")
    )

    monkeypatch.setattr(
        app_module,
        "CHATS_DIR",
        str(chats_dir)
    )

    return tmp_path