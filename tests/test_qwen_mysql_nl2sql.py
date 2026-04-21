import json
import os
from pathlib import Path

import pytest


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.qianwen import QianWenAI_Chat


class QwenMySQLVanna(ChromaDB_VectorStore, QianWenAI_Chat):
    """Legacy-compatible Vanna setup for Qwen + local Chroma + MySQL."""

    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        QianWenAI_Chat.__init__(self, config=config)


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_qwen_api_key() -> str:
    return os.getenv("QWEN_API_KEY") or _get_required_env("OPENAI_API_KEY")


def _load_json_list(env_name: str) -> list:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return []

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{env_name} must be valid JSON") from exc

    if not isinstance(value, list):
        raise RuntimeError(f"{env_name} must be a JSON array")

    return value


def _build_vanna() -> QwenMySQLVanna:
    config = {
        "api_key": _get_qwen_api_key(),
        "model": os.getenv("QWEN_MODEL", "qwen3.6-plus"),
        "base_url": os.getenv(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "path": os.getenv(
            "VANNA_CHROMA_PATH",
            str(Path(".chroma") / "qwen_mysql_nl2sql"),
        ),
        "dialect": "MySQL",
        "language": os.getenv("VANNA_RESPONSE_LANGUAGE", "Chinese"),
    }
    return QwenMySQLVanna(config=config)


def _connect_mysql(vn: QwenMySQLVanna) -> None:
    vn.connect_to_mysql(
        host=_get_required_env("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        dbname=_get_required_env("MYSQL_DATABASE"),
        user=_get_required_env("MYSQL_USER"),
        password=_get_required_env("MYSQL_PASSWORD"),
    )


def _iter_target_tables(vn: QwenMySQLVanna) -> list[str]:
    table_filter = {
        name.strip()
        for name in os.getenv("MYSQL_TRAIN_TABLES", "").split(",")
        if name.strip()
    }
    database = _get_required_env("MYSQL_DATABASE")
    df_tables = vn.run_sql(f"SHOW TABLES FROM `{database}`")

    if df_tables.empty:
        raise RuntimeError(f"No tables found in database `{database}`")

    table_column = df_tables.columns[0]
    table_names = df_tables[table_column].tolist()

    if table_filter:
        table_names = [name for name in table_names if name in table_filter]

    if not table_names:
        raise RuntimeError("No tables selected for training")

    return table_names


def train_mysql_metadata(vn: QwenMySQLVanna) -> None:
    """Train schema, business docs, and example question-SQL pairs."""
    for table_name in _iter_target_tables(vn):
        df_ddl = vn.run_sql(f"SHOW CREATE TABLE `{table_name}`")
        create_sql = df_ddl.iloc[0, 1]
        vn.train(ddl=create_sql)

    for doc in _load_json_list("VANNA_TRAIN_DOCUMENTATION_JSON"):
        vn.train(documentation=str(doc))

    for item in _load_json_list("VANNA_TRAIN_EXAMPLES_JSON"):
        if not isinstance(item, dict):
            raise RuntimeError("VANNA_TRAIN_EXAMPLES_JSON items must be JSON objects")
        question = item.get("question")
        sql = item.get("sql")
        if not question or not sql:
            raise RuntimeError(
                "Each VANNA_TRAIN_EXAMPLES_JSON item must contain question and sql"
            )
        vn.train(question=question, sql=sql)


def generate_sql_with_training(question: str) -> str:
    """Helper for manual runs and reuse in other tests."""
    vn = _build_vanna()
    _connect_mysql(vn)
    train_mysql_metadata(vn)
    return vn.generate_sql(question)


@pytest.mark.integration
@pytest.mark.mysql
def test_generate_sql_with_qwen_mysql_chroma():
    required_env = [
        "MYSQL_HOST",
        "MYSQL_DATABASE",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
        "VANNA_TEST_QUESTION",
    ]
    missing = [name for name in required_env if not os.getenv(name)]
    if not (os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")):
        missing.append("QWEN_API_KEY/OPENAI_API_KEY")
    if missing:
        pytest.skip(f"Missing environment variables: {', '.join(missing)}")

    try:
        import chromadb  # noqa: F401
        import openai  # noqa: F401
        import pymysql  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"Missing optional dependency: {exc}")

    question = os.getenv("VANNA_TEST_QUESTION", "").strip()
    sql = generate_sql_with_training(question)

    assert sql is not None
    assert sql.strip()
    assert "SELECT" in sql.upper() or "WITH" in sql.upper()


if __name__ == "__main__":
    question = os.getenv("VANNA_TEST_QUESTION", "").strip()
    if not question:
        raise SystemExit("Please set VANNA_TEST_QUESTION before running this file.")

    generated_sql = generate_sql_with_training(question)
    print("\nGenerated SQL:\n")
    print(generated_sql)
