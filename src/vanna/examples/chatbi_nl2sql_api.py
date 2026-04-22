"""FastAPI app for ChatBI NL2SQL based on legacy Vanna + Qwen + Chroma + MySQL.

Features:
- Generate SQL from natural language
- Persist training docs/examples in chatbi_sys MySQL
- Incrementally sync changed docs/examples into Vanna/Chroma
- Track business-table DDL fingerprints and retrain only changed tables

Run example:
    uvicorn vanna.examples.chatbi_nl2sql_api:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.qianwen import QianWenAI_Chat


if load_dotenv is not None:
    load_dotenv()


class ChatBIVanna(ChromaDB_VectorStore, QianWenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        QianWenAI_Chat.__init__(self, config=config)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    return datetime.utcnow()


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class GenerateSqlRequest(BaseModel):
    question: str = Field(..., description="需要转换为 SQL 的自然语言问题。")
    auto_sync_training: bool = Field(
        default=False,
        description="是否在生成 SQL 前，先同步最新的训练数据和 DDL 指纹。",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "查询最近30天订单总金额排名前10的客户",
                "auto_sync_training": False,
            }
        }
    }


class GenerateSqlResponse(BaseModel):
    question: str = Field(description="输入的自然语言问题。")
    sql: str = Field(description="模型生成的 SQL 语句。")


class DocumentationCreateRequest(BaseModel):
    content: str = Field(description="用于训练的业务文档内容。")

    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "订单表 order_info 中 status=1 表示已支付，amount 字段单位为元。"
            }
        }
    }


class DocumentationUpdateRequest(BaseModel):
    content: str = Field(description="更新后的业务文档内容。")

    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "订单表 order_info 中 status=1 表示已支付，refund_status=2 表示已退款。"
            }
        }
    }


class ExampleCreateRequest(BaseModel):
    question: str = Field(description="示例中的自然语言问题。")
    sql: str = Field(description="与问题对应的标准 SQL。")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "查询今天新增用户数",
                "sql": "SELECT COUNT(*) AS user_count FROM user_info WHERE DATE(create_time) = CURDATE();",
            }
        }
    }


class ExampleUpdateRequest(BaseModel):
    question: str = Field(description="更新后的自然语言问题。")
    sql: str = Field(description="更新后的标准 SQL。")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "查询昨天新增用户数",
                "sql": "SELECT COUNT(*) AS user_count FROM user_info WHERE DATE(create_time) = CURDATE() - INTERVAL 1 DAY;",
            }
        }
    }


class SyncTrainRequest(BaseModel):
    sync_ddl: bool = Field(default=True, description="是否同步业务表 DDL 训练数据。")
    sync_docs: bool = Field(default=True, description="是否同步业务文档训练数据。")
    sync_examples: bool = Field(default=True, description="是否同步问答 SQL 示例训练数据。")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sync_ddl": True,
                "sync_docs": True,
                "sync_examples": True,
            }
        }
    }


class TrainingDataResponse(BaseModel):
    id: int = Field(description="记录主键 ID。")
    content: str | None = Field(default=None, description="业务文档内容。")
    question: str | None = Field(default=None, description="训练示例中的问题。")
    sql: str | None = Field(default=None, description="训练示例中的 SQL。")
    content_hash: str = Field(description="当前内容的哈希值。")
    trained_content_hash: str | None = Field(default=None, description="最近一次完成训练时的内容哈希值。")
    vanna_training_id: str | None = Field(default=None, description="Vanna 中对应的训练记录 ID。")
    created_at: datetime | None = Field(default=None, description="创建时间。")
    updated_at: datetime | None = Field(default=None, description="更新时间。")
    last_trained_at: datetime | None = Field(default=None, description="最近一次训练时间。")


class SyncResult(BaseModel):
    docs_added: int = Field(default=0, description="新增同步的业务文档数量。")
    docs_updated: int = Field(default=0, description="更新同步的业务文档数量。")
    docs_deleted: int = Field(default=0, description="删除的业务文档数量。")
    examples_added: int = Field(default=0, description="新增同步的示例数量。")
    examples_updated: int = Field(default=0, description="更新同步的示例数量。")
    examples_deleted: int = Field(default=0, description="删除的示例数量。")
    ddl_added: int = Field(default=0, description="新增同步的 DDL 数量。")
    ddl_updated: int = Field(default=0, description="更新同步的 DDL 数量。")
    ddl_deleted: int = Field(default=0, description="删除的 DDL 数量。")
    ddl_unchanged: int = Field(default=0, description="未发生变化的 DDL 数量。")


@dataclass
class ServiceConfig:
    qwen_api_key: str
    qwen_model: str
    qwen_base_url: str
    qwen_request_timeout: float
    chroma_path: str
    response_language: str
    business_host: str
    business_port: int
    business_database: str
    business_user: str
    business_password: str
    metadata_host: str
    metadata_port: int
    metadata_database: str
    metadata_user: str
    metadata_password: str
    ddl_table_filter: list[str]

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        qwen_api_key = os.getenv("QWEN_API_KEY") or _required_env("OPENAI_API_KEY")
        ddl_filter = [
            item.strip()
            for item in os.getenv("MYSQL_TRAIN_TABLES", "").split(",")
            if item.strip()
        ]
        return cls(
            qwen_api_key=qwen_api_key,
            qwen_model=os.getenv("QWEN_MODEL", "qwen3.6-plus"),
            qwen_base_url=os.getenv(
                "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            qwen_request_timeout=float(os.getenv("QWEN_HTTP_TIMEOUT_SECONDS", "60")),
            chroma_path=os.getenv(
                "VANNA_CHROMA_PATH", str(Path(".chroma") / "chatbi_nl2sql")
            ),
            response_language=os.getenv("VANNA_RESPONSE_LANGUAGE", "Chinese"),
            business_host=_required_env("MYSQL_HOST"),
            business_port=int(os.getenv("MYSQL_PORT", "3306")),
            business_database=_required_env("MYSQL_DATABASE"),
            business_user=_required_env("MYSQL_USER"),
            business_password=_required_env("MYSQL_PASSWORD"),
            metadata_host=os.getenv("CHATBI_SYS_HOST", os.getenv("MYSQL_HOST", "")),
            metadata_port=int(
                os.getenv("CHATBI_SYS_PORT", os.getenv("MYSQL_PORT", "3306"))
            ),
            metadata_database=os.getenv("CHATBI_SYS_DATABASE", "chatbi_sys"),
            metadata_user=os.getenv("CHATBI_SYS_USER", os.getenv("MYSQL_USER", "")),
            metadata_password=os.getenv(
                "CHATBI_SYS_PASSWORD", os.getenv("MYSQL_PASSWORD", "")
            ),
            ddl_table_filter=ddl_filter,
        )


class ChatBINL2SQLService:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.metadata = MetaData()
        # Store manually maintained business documentation used for NL2SQL context.
        self.documentation_table = Table(
            "chatbi_vanna_documentation",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True, comment="主键ID"),
            Column("content", Text, nullable=False, comment="业务文档内容"),
            Column("content_hash", String(64), nullable=False, comment="当前内容哈希"),
            Column(
                "trained_content_hash",
                String(64),
                comment="最近一次完成训练时的内容哈希",
            ),
            Column("vanna_training_id", String(128), comment="Vanna/Chroma中的训练记录ID"),
            Column("created_at", DateTime, nullable=False, comment="创建时间"),
            Column("updated_at", DateTime, nullable=False, comment="更新时间"),
            Column("last_trained_at", DateTime, comment="最近一次训练时间"),
            comment="ChatBI NL2SQL 业务文档训练数据表",
        )
        # Store curated question-SQL examples for few-shot training.
        self.example_table = Table(
            "chatbi_vanna_example_sql",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True, comment="主键ID"),
            Column("question", Text, nullable=False, comment="示例问题"),
            Column("sql_text", Text, nullable=False, comment="示例SQL"),
            Column("content_hash", String(64), nullable=False, comment="问题和SQL组合后的哈希"),
            Column(
                "trained_content_hash",
                String(64),
                comment="最近一次完成训练时的内容哈希",
            ),
            Column("vanna_training_id", String(128), comment="Vanna/Chroma中的训练记录ID"),
            Column("created_at", DateTime, nullable=False, comment="创建时间"),
            Column("updated_at", DateTime, nullable=False, comment="更新时间"),
            Column("last_trained_at", DateTime, comment="最近一次训练时间"),
            comment="ChatBI NL2SQL 示例问答SQL训练数据表",
        )
        # Persist DDL fingerprints so only changed tables are retrained.
        self.ddl_table = Table(
            "chatbi_vanna_ddl_fingerprint",
            self.metadata,
            Column("table_name", String(255), primary_key=True, comment="业务表名"),
            Column("ddl_sql", Text, nullable=False, comment="当前DDL文本"),
            Column("ddl_hash", String(64), nullable=False, comment="DDL哈希指纹"),
            Column("vanna_training_id", String(128), comment="Vanna/Chroma中的训练记录ID"),
            Column("last_trained_at", DateTime, comment="最近一次训练时间"),
            Column("updated_at", DateTime, nullable=False, comment="最近一次检测到DDL的时间"),
            comment="ChatBI NL2SQL 业务库DDL指纹表",
        )
        # Keep a simple audit trail for manual training sync executions.
        self.sync_log_table = Table(
            "chatbi_vanna_sync_log",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True, comment="主键ID"),
            Column("sync_type", String(64), nullable=False, comment="同步类型"),
            Column("result_json", JSON, nullable=False, comment="同步结果JSON"),
            Column("created_at", DateTime, nullable=False, comment="创建时间"),
            comment="ChatBI NL2SQL 训练同步日志表",
        )
        self._ensure_metadata_database_exists()
        self.metadata_engine = self._create_engine(
            host=config.metadata_host,
            port=config.metadata_port,
            database=config.metadata_database,
            user=config.metadata_user,
            password=config.metadata_password,
        )
        self.metadata.create_all(self.metadata_engine)

    def _ensure_metadata_database_exists(self) -> None:
        import pymysql

        # The metadata database stores training configs and sync state.
        conn = pymysql.connect(
            host=self.config.metadata_host,
            port=self.config.metadata_port,
            user=self.config.metadata_user,
            password=self.config.metadata_password,
            charset="utf8mb4",
            autocommit=True,
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.config.metadata_database}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
                )
        finally:
            conn.close()

    def _create_engine(
        self, *, host: str, port: int, database: str, user: str, password: str
    ) -> Engine:
        return create_engine(
            f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4",
            future=True,
        )

    @contextmanager
    def _metadata_connection(self) -> Generator[Any, None, None]:
        with self.metadata_engine.begin() as conn:
            yield conn

    def _build_vanna(self) -> ChatBIVanna:
        # Create a fresh Vanna instance per request so API calls stay stateless.
        vn = ChatBIVanna(
            config={
                "api_key": self.config.qwen_api_key,
                "model": self.config.qwen_model,
                "base_url": self.config.qwen_base_url,
                "request_timeout": self.config.qwen_request_timeout,
                "path": self.config.chroma_path,
                "dialect": "MySQL",
                "language": self.config.response_language,
            }
        )
        vn.connect_to_mysql(
            host=self.config.business_host,
            port=self.config.business_port,
            dbname=self.config.business_database,
            user=self.config.business_user,
            password=self.config.business_password,
        )
        return vn

    def _save_sync_log(self, sync_type: str, result: SyncResult) -> None:
        with self._metadata_connection() as conn:
            conn.execute(
                insert(self.sync_log_table).values(
                    sync_type=sync_type,
                    result_json=result.model_dump(),
                    created_at=_utcnow(),
                )
            )

    def list_documentation(self) -> list[TrainingDataResponse]:
        with self._metadata_connection() as conn:
            rows = conn.execute(
                select(self.documentation_table).order_by(self.documentation_table.c.id)
            ).mappings()
            return [
                TrainingDataResponse(
                    id=row["id"],
                    content=row["content"],
                    content_hash=row["content_hash"],
                    trained_content_hash=row["trained_content_hash"],
                    vanna_training_id=row["vanna_training_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    last_trained_at=row["last_trained_at"],
                )
                for row in rows
            ]

    def create_documentation(self, payload: DocumentationCreateRequest) -> TrainingDataResponse:
        now = _utcnow()
        content_hash = _sha256(payload.content)
        with self._metadata_connection() as conn:
            result = conn.execute(
                insert(self.documentation_table).values(
                    content=payload.content,
                    content_hash=content_hash,
                    trained_content_hash=None,
                    vanna_training_id=None,
                    created_at=now,
                    updated_at=now,
                    last_trained_at=None,
                )
            )
            new_id = result.inserted_primary_key[0]
        return self.get_documentation(new_id)

    def get_documentation(self, doc_id: int) -> TrainingDataResponse:
        with self._metadata_connection() as conn:
            row = conn.execute(
                select(self.documentation_table).where(self.documentation_table.c.id == doc_id)
            ).mappings().first()
        if row is None:
            raise HTTPException(status_code=404, detail="Documentation not found")
        return TrainingDataResponse(
            id=row["id"],
            content=row["content"],
            content_hash=row["content_hash"],
            trained_content_hash=row["trained_content_hash"],
            vanna_training_id=row["vanna_training_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_trained_at=row["last_trained_at"],
        )

    def update_documentation(
        self, doc_id: int, payload: DocumentationUpdateRequest
    ) -> TrainingDataResponse:
        with self._metadata_connection() as conn:
            row = conn.execute(
                select(self.documentation_table).where(self.documentation_table.c.id == doc_id)
            ).mappings().first()
            if row is None:
                raise HTTPException(status_code=404, detail="Documentation not found")

            conn.execute(
                update(self.documentation_table)
                .where(self.documentation_table.c.id == doc_id)
                .values(
                    content=payload.content,
                    content_hash=_sha256(payload.content),
                    updated_at=_utcnow(),
                )
            )
        return self.get_documentation(doc_id)

    def delete_documentation(self, doc_id: int) -> dict[str, Any]:
        vn = self._build_vanna()
        with self._metadata_connection() as conn:
            row = conn.execute(
                select(self.documentation_table).where(self.documentation_table.c.id == doc_id)
            ).mappings().first()
            if row is None:
                raise HTTPException(status_code=404, detail="Documentation not found")

            if row["vanna_training_id"]:
                vn.remove_training_data(row["vanna_training_id"])

            conn.execute(
                delete(self.documentation_table).where(self.documentation_table.c.id == doc_id)
            )
        return {"deleted": True, "id": doc_id}

    def list_examples(self) -> list[TrainingDataResponse]:
        with self._metadata_connection() as conn:
            rows = conn.execute(
                select(self.example_table).order_by(self.example_table.c.id)
            ).mappings()
            return [
                TrainingDataResponse(
                    id=row["id"],
                    question=row["question"],
                    sql=row["sql_text"],
                    content_hash=row["content_hash"],
                    trained_content_hash=row["trained_content_hash"],
                    vanna_training_id=row["vanna_training_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    last_trained_at=row["last_trained_at"],
                )
                for row in rows
            ]

    def create_example(self, payload: ExampleCreateRequest) -> TrainingDataResponse:
        now = _utcnow()
        content_hash = _sha256(f"{payload.question}\n{payload.sql}")
        with self._metadata_connection() as conn:
            result = conn.execute(
                insert(self.example_table).values(
                    question=payload.question,
                    sql_text=payload.sql,
                    content_hash=content_hash,
                    trained_content_hash=None,
                    vanna_training_id=None,
                    created_at=now,
                    updated_at=now,
                    last_trained_at=None,
                )
            )
            new_id = result.inserted_primary_key[0]
        return self.get_example(new_id)

    def get_example(self, example_id: int) -> TrainingDataResponse:
        with self._metadata_connection() as conn:
            row = conn.execute(
                select(self.example_table).where(self.example_table.c.id == example_id)
            ).mappings().first()
        if row is None:
            raise HTTPException(status_code=404, detail="Example not found")
        return TrainingDataResponse(
            id=row["id"],
            question=row["question"],
            sql=row["sql_text"],
            content_hash=row["content_hash"],
            trained_content_hash=row["trained_content_hash"],
            vanna_training_id=row["vanna_training_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_trained_at=row["last_trained_at"],
        )

    def update_example(self, example_id: int, payload: ExampleUpdateRequest) -> TrainingDataResponse:
        with self._metadata_connection() as conn:
            row = conn.execute(
                select(self.example_table).where(self.example_table.c.id == example_id)
            ).mappings().first()
            if row is None:
                raise HTTPException(status_code=404, detail="Example not found")

            conn.execute(
                update(self.example_table)
                .where(self.example_table.c.id == example_id)
                .values(
                    question=payload.question,
                    sql_text=payload.sql,
                    content_hash=_sha256(f"{payload.question}\n{payload.sql}"),
                    updated_at=_utcnow(),
                )
            )
        return self.get_example(example_id)

    def delete_example(self, example_id: int) -> dict[str, Any]:
        vn = self._build_vanna()
        with self._metadata_connection() as conn:
            row = conn.execute(
                select(self.example_table).where(self.example_table.c.id == example_id)
            ).mappings().first()
            if row is None:
                raise HTTPException(status_code=404, detail="Example not found")

            if row["vanna_training_id"]:
                vn.remove_training_data(row["vanna_training_id"])

            conn.execute(delete(self.example_table).where(self.example_table.c.id == example_id))
        return {"deleted": True, "id": example_id}

    def _sync_documentation(self, vn: ChatBIVanna, result: SyncResult) -> None:
        with self._metadata_connection() as conn:
            rows = conn.execute(select(self.documentation_table)).mappings().all()
            for row in rows:
                # Skip retraining when content hash has not changed since last sync.
                if row["trained_content_hash"] == row["content_hash"] and row["vanna_training_id"]:
                    continue

                previous_training_id = row["vanna_training_id"]
                if previous_training_id:
                    vn.remove_training_data(previous_training_id)
                    result.docs_updated += 1
                else:
                    result.docs_added += 1

                new_training_id = vn.train(documentation=row["content"])
                conn.execute(
                    update(self.documentation_table)
                    .where(self.documentation_table.c.id == row["id"])
                    .values(
                        trained_content_hash=row["content_hash"],
                        vanna_training_id=new_training_id,
                        last_trained_at=_utcnow(),
                    )
                )

    def _sync_examples(self, vn: ChatBIVanna, result: SyncResult) -> None:
        with self._metadata_connection() as conn:
            rows = conn.execute(select(self.example_table)).mappings().all()
            for row in rows:
                # Skip retraining when question/sql content is unchanged.
                if row["trained_content_hash"] == row["content_hash"] and row["vanna_training_id"]:
                    continue

                previous_training_id = row["vanna_training_id"]
                if previous_training_id:
                    vn.remove_training_data(previous_training_id)
                    result.examples_updated += 1
                else:
                    result.examples_added += 1

                new_training_id = vn.train(question=row["question"], sql=row["sql_text"])
                conn.execute(
                    update(self.example_table)
                    .where(self.example_table.c.id == row["id"])
                    .values(
                        trained_content_hash=row["content_hash"],
                        vanna_training_id=new_training_id,
                        last_trained_at=_utcnow(),
                    )
                )

    def _iter_business_tables(self, vn: ChatBIVanna) -> list[str]:
        df_tables = vn.run_sql(f"SHOW TABLES FROM `{self.config.business_database}`")
        if df_tables.empty:
            return []
        table_column = df_tables.columns[0]
        table_names = df_tables[table_column].tolist()
        if self.config.ddl_table_filter:
            table_names = [name for name in table_names if name in self.config.ddl_table_filter]
        return table_names

    def _sync_ddl(self, vn: ChatBIVanna, result: SyncResult) -> None:
        current_ddls: dict[str, tuple[str, str]] = {}
        for table_name in self._iter_business_tables(vn):
            df_ddl = vn.run_sql(f"SHOW CREATE TABLE `{table_name}`")
            ddl_sql = df_ddl.iloc[0, 1]
            current_ddls[table_name] = (ddl_sql, _sha256(ddl_sql))

        with self._metadata_connection() as conn:
            stored_rows = {
                row["table_name"]: row
                for row in conn.execute(select(self.ddl_table)).mappings().all()
            }

            for table_name, (ddl_sql, ddl_hash) in current_ddls.items():
                stored = stored_rows.get(table_name)
                # Unchanged DDL does not need to be re-trained.
                if stored and stored["ddl_hash"] == ddl_hash:
                    result.ddl_unchanged += 1
                    continue

                if stored and stored["vanna_training_id"]:
                    vn.remove_training_data(stored["vanna_training_id"])
                    result.ddl_updated += 1
                else:
                    result.ddl_added += 1

                new_training_id = vn.train(ddl=ddl_sql)
                stmt = insert(self.ddl_table).values(
                    table_name=table_name,
                    ddl_sql=ddl_sql,
                    ddl_hash=ddl_hash,
                    vanna_training_id=new_training_id,
                    last_trained_at=_utcnow(),
                    updated_at=_utcnow(),
                )
                conn.execute(
                    stmt.prefix_with("IGNORE")
                )
                conn.execute(
                    update(self.ddl_table)
                    .where(self.ddl_table.c.table_name == table_name)
                    .values(
                        ddl_sql=ddl_sql,
                        ddl_hash=ddl_hash,
                        vanna_training_id=new_training_id,
                        last_trained_at=_utcnow(),
                        updated_at=_utcnow(),
                    )
                )

            removed_table_names = set(stored_rows) - set(current_ddls)
            for table_name in removed_table_names:
                # Clean up fingerprints for dropped business tables.
                stored = stored_rows[table_name]
                if stored["vanna_training_id"]:
                    vn.remove_training_data(stored["vanna_training_id"])
                conn.execute(delete(self.ddl_table).where(self.ddl_table.c.table_name == table_name))
                result.ddl_deleted += 1

    def sync_training(self, payload: SyncTrainRequest) -> SyncResult:
        vn = self._build_vanna()
        result = SyncResult()

        if payload.sync_docs:
            self._sync_documentation(vn, result)
        if payload.sync_examples:
            self._sync_examples(vn, result)
        if payload.sync_ddl:
            self._sync_ddl(vn, result)

        self._save_sync_log("manual_sync", result)
        return result

    def generate_sql(self, payload: GenerateSqlRequest) -> GenerateSqlResponse:
        # Optional sync is useful when the caller wants "train then ask" in one request.
        if payload.auto_sync_training:
            self.sync_training(SyncTrainRequest())

        vn = self._build_vanna()
        print(f"Generating SQL for question: {payload.question}")
        sql = vn.generate_sql(payload.question)
        print("Generated SQL successfully")
        return GenerateSqlResponse(question=payload.question, sql=sql)

    def list_ddl_fingerprints(self) -> list[dict[str, Any]]:
        with self._metadata_connection() as conn:
            rows = conn.execute(
                select(self.ddl_table).order_by(self.ddl_table.c.table_name)
            ).mappings()
            return [dict(row) for row in rows]


service = ChatBINL2SQLService(ServiceConfig.from_env())
tags_metadata = [
    {"name": "系统", "description": "服务健康检查和文档导航接口。"},
    {"name": "SQL生成", "description": "将自然语言问题转换为 SQL。"},
    {"name": "训练文档", "description": "管理用于训练的业务文档。"},
    {"name": "训练示例", "description": "管理用于训练的问答 SQL 示例。"},
    {"name": "训练同步", "description": "将文档、示例和 DDL 同步到 Vanna 向量训练数据中。"},
]

app = FastAPI(
    title="ChatBI NL2SQL 接口文档",
    version="0.1.0",
    description="用于自然语言转 SQL、训练文档管理、示例管理和增量训练同步的接口服务。",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=tags_metadata,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/swagger")


@app.get("/health", tags=["系统"], summary="健康检查")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "chatbi-nl2sql"}


@app.post(
    "/api/chatbi/v1/sql/generate",
    response_model=GenerateSqlResponse,
    tags=["SQL生成"],
    summary="根据自然语言问题生成 SQL",
)
def generate_sql(payload: GenerateSqlRequest) -> GenerateSqlResponse:
    return service.generate_sql(payload)


@app.get(
    "/api/chatbi/v1/training/documentation",
    response_model=list[TrainingDataResponse],
    tags=["训练文档"],
    summary="查询训练文档列表",
)
def list_documentation() -> list[TrainingDataResponse]:
    return service.list_documentation()


@app.post(
    "/api/chatbi/v1/training/documentation",
    response_model=TrainingDataResponse,
    tags=["训练文档"],
    summary="新增训练文档",
)
def create_documentation(payload: DocumentationCreateRequest) -> TrainingDataResponse:
    return service.create_documentation(payload)


@app.put(
    "/api/chatbi/v1/training/documentation/{doc_id}",
    response_model=TrainingDataResponse,
    tags=["训练文档"],
    summary="更新训练文档",
)
def update_documentation(doc_id: int, payload: DocumentationUpdateRequest) -> TrainingDataResponse:
    return service.update_documentation(doc_id, payload)


@app.delete(
    "/api/chatbi/v1/training/documentation/{doc_id}",
    tags=["训练文档"],
    summary="删除训练文档",
)
def delete_documentation(doc_id: int) -> dict[str, Any]:
    return service.delete_documentation(doc_id)


@app.get(
    "/api/chatbi/v1/training/examples",
    response_model=list[TrainingDataResponse],
    tags=["训练示例"],
    summary="查询训练示例列表",
)
def list_examples() -> list[TrainingDataResponse]:
    return service.list_examples()


@app.post(
    "/api/chatbi/v1/training/examples",
    response_model=TrainingDataResponse,
    tags=["训练示例"],
    summary="新增训练示例",
)
def create_example(payload: ExampleCreateRequest) -> TrainingDataResponse:
    return service.create_example(payload)


@app.put(
    "/api/chatbi/v1/training/examples/{example_id}",
    response_model=TrainingDataResponse,
    tags=["训练示例"],
    summary="更新训练示例",
)
def update_example(example_id: int, payload: ExampleUpdateRequest) -> TrainingDataResponse:
    return service.update_example(example_id, payload)


@app.delete(
    "/api/chatbi/v1/training/examples/{example_id}",
    tags=["训练示例"],
    summary="删除训练示例",
)
def delete_example(example_id: int) -> dict[str, Any]:
    return service.delete_example(example_id)


@app.post(
    "/api/chatbi/v1/train/sync",
    response_model=SyncResult,
    tags=["训练同步"],
    summary="执行训练数据同步",
)
def sync_training(payload: SyncTrainRequest) -> SyncResult:
    return service.sync_training(payload)


@app.get(
    "/api/chatbi/v1/train/ddl-fingerprints",
    tags=["训练同步"],
    summary="查询 DDL 指纹列表",
)
def list_ddl_fingerprints() -> list[dict[str, Any]]:
    return service.list_ddl_fingerprints()
