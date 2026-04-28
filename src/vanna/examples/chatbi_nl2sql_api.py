"""基于 legacy Vanna + Qwen + Chroma + MySQL 的 ChatBI NL2SQL FastAPI 服务。

功能说明：
- 在 sys 数据库中维护多个业务数据库连接配置
- 在 chatbi_sys 中持久化训练文档与示例 SQL
- 按数据库同步 DDL 指纹，仅重训发生变化的表
- 生成 SQL 时按数据库隔离 DDL、文档和示例 SQL，避免互相污染

启动示例：
    uvicorn vanna.examples.chatbi_nl2sql_api:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    JSON,
    Boolean,
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
    text,
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


def _sha256(text_value: str) -> str:
    return hashlib.sha256(text_value.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    return datetime.utcnow()


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少必填环境变量：{name}")
    return value


def _clean_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_database_key(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise ValueError("database_key 不能为空")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_")
    if any(char not in allowed for char in normalized):
        raise ValueError("database_key 仅支持小写字母、数字和下划线")
    return normalized


def _parse_table_filter(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        "ALLOW_LLM_TO_SEE_DATA must be one of: true/false, 1/0, yes/no, on/off"
    )


def _collection_suffix(database_key: str) -> str:
    digest = hashlib.md5(database_key.encode("utf-8")).hexdigest()[:12]  # noqa: S324
    return f"{database_key[:40]}_{digest}"


class ChatBIVanna(ChromaDB_VectorStore, QianWenAI_Chat):
    """兼容 legacy Vanna 的封装，并按数据库拆分 Chroma 集合。"""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        ChromaDB_VectorStore.__init__(self, config=config)
        QianWenAI_Chat.__init__(self, config=config)

        suffix = config.get("collection_suffix")
        if suffix:
            collection_metadata = config.get("collection_metadata")
            self.documentation_collection = self.chroma_client.get_or_create_collection(
                name=f"documentation_{suffix}",
                embedding_function=self.embedding_function,
                metadata=collection_metadata,
            )
            self.ddl_collection = self.chroma_client.get_or_create_collection(
                name=f"ddl_{suffix}",
                embedding_function=self.embedding_function,
                metadata=collection_metadata,
            )
            self.sql_collection = self.chroma_client.get_or_create_collection(
                name=f"sql_{suffix}",
                embedding_function=self.embedding_function,
                metadata=collection_metadata,
            )


class DatabaseConfigCreateRequest(BaseModel):
    database_key: str = Field(description="数据库唯一标识，建议使用稳定的英文键名")
    display_name: str = Field(description="数据库展示名称")
    db_type: str = Field(default="mysql", description="数据库类型，当前仅支持 mysql")
    host: str = Field(description="数据库主机地址")
    port: int = Field(default=3306, description="数据库端口")
    database_name: str = Field(description="业务数据库名或 schema 名")
    username: str = Field(description="数据库用户名")
    password: str = Field(description="数据库密码")
    ddl_table_filter: list[str] = Field(
        default_factory=list,
        description="DDL 指纹同步时使用的表白名单，可为空",
    )
    description: str | None = Field(default=None, description="备注信息")
    is_active: bool = Field(default=True, description="是否启用该数据库")

    model_config = {
        "json_schema_extra": {
            "example": {
                "database_key": "erp_prod",
                "display_name": "ERP生产库",
                "db_type": "mysql",
                "host": "192.168.10.21",
                "port": 3306,
                "database_name": "erp",
                "username": "chatbi_user",
                "password": "your_password",
                "ddl_table_filter": ["order_info", "customer_info", "refund_order"],
                "description": "ERP核心交易库",
                "is_active": True,
            }
        }
    }

    @field_validator("database_key")
    @classmethod
    def validate_database_key(cls, value: str) -> str:
        return _normalize_database_key(value)

    @field_validator("db_type")
    @classmethod
    def validate_db_type(cls, value: str) -> str:
        db_type = value.strip().lower()
        if db_type != "mysql":
            raise ValueError("当前实现仅支持 mysql")
        return db_type

    @field_validator("host", "database_name", "username", "password", "display_name")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        return cleaned

    @field_validator("description")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return _clean_optional_str(value)


class DatabaseConfigUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None)
    db_type: str | None = Field(default=None)
    host: str | None = Field(default=None)
    port: int | None = Field(default=None)
    database_name: str | None = Field(default=None)
    username: str | None = Field(default=None)
    password: str | None = Field(default=None)
    ddl_table_filter: list[str] | None = Field(default=None)
    description: str | None = Field(default=None)
    is_active: bool | None = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "display_name": "ERP生产库-主库",
                "host": "192.168.10.22",
                "port": 3306,
                "database_name": "erp",
                "username": "chatbi_user",
                "password": "new_password",
                "ddl_table_filter": ["order_info", "customer_info"],
                "description": "切换到新主库地址",
                "is_active": True,
            }
        }
    }

    @field_validator("db_type")
    @classmethod
    def validate_optional_db_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        db_type = value.strip().lower()
        if db_type != "mysql":
            raise ValueError("当前实现仅支持 mysql")
        return db_type

    @field_validator("display_name", "host", "database_name", "username", "password")
    @classmethod
    def validate_optional_required_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        return cleaned

    @field_validator("description")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return _clean_optional_str(value)


class DatabaseConfigResponse(BaseModel):
    id: int
    database_key: str
    display_name: str
    db_type: str
    host: str
    port: int
    database_name: str
    username: str
    ddl_table_filter: list[str] = Field(default_factory=list)
    description: str | None = None
    is_active: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None
    has_password: bool = True

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": 1,
                "database_key": "erp_prod",
                "display_name": "ERP生产库",
                "db_type": "mysql",
                "host": "192.168.10.21",
                "port": 3306,
                "database_name": "erp",
                "username": "chatbi_user",
                "ddl_table_filter": ["order_info", "customer_info", "refund_order"],
                "description": "ERP核心交易库",
                "is_active": True,
                "created_at": "2026-04-22T10:00:00",
                "updated_at": "2026-04-22T10:00:00",
                "has_password": True,
            }
        }
    }


class DatabaseConnectionTestResponse(BaseModel):
    database_id: int
    database_key: str
    connected: bool
    message: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "database_id": 1,
                "database_key": "erp_prod",
                "connected": True,
                "message": "连接成功",
            }
        }
    }


class GenerateSqlRequest(BaseModel):
    database_id: int = Field(description="目标数据库 ID")
    question: str = Field(description="需要转换为 SQL 的自然语言问题")
    auto_sync_training: bool = Field(
        default=False,
        description="生成 SQL 前是否自动同步训练数据",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "database_id": 1,
                "question": "查询最近30天订单总金额排名前10的客户",
                "auto_sync_training": False,
            }
        }
    }


class GenerateSqlResponse(BaseModel):
    database_id: int
    question: str
    sql: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "database_id": 1,
                "question": "查询最近30天订单总金额排名前10的客户",
                "sql": "SELECT customer_id, SUM(order_amount) AS total_amount FROM order_info WHERE order_time >= CURDATE() - INTERVAL 30 DAY GROUP BY customer_id ORDER BY total_amount DESC LIMIT 10;",
            }
        }
    }


class DocumentationCreateRequest(BaseModel):
    database_id: int = Field(description="目标数据库 ID")
    content: str = Field(description="用于训练的业务文档内容")

    model_config = {
        "json_schema_extra": {
            "example": {
                "database_id": 1,
                "content": "订单表 order_info 中 status=1 表示已支付，status=2 表示已取消，order_amount 单位为元。",
            }
        }
    }

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("content 不能为空")
        return cleaned


class DocumentationUpdateRequest(BaseModel):
    content: str = Field(description="更新后的训练文档内容")

    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "订单表 order_info 中 status=1 表示已支付，refund_status=2 表示已退款，order_amount 单位为元。",
            }
        }
    }

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("content 不能为空")
        return cleaned


class ExampleCreateRequest(BaseModel):
    database_id: int = Field(description="目标数据库 ID")
    question: str = Field(description="示例自然语言问题")
    sql: str = Field(description="示例问题对应的标准 SQL")

    model_config = {
        "json_schema_extra": {
            "example": {
                "database_id": 1,
                "question": "查询今天新增用户数",
                "sql": "SELECT COUNT(*) AS user_count FROM user_info WHERE DATE(create_time) = CURDATE();",
            }
        }
    }

    @field_validator("question", "sql")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        return cleaned


class ExampleUpdateRequest(BaseModel):
    question: str = Field(description="更新后的示例问题")
    sql: str = Field(description="更新后的示例 SQL")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "查询昨天新增用户数",
                "sql": "SELECT COUNT(*) AS user_count FROM user_info WHERE DATE(create_time) = CURDATE() - INTERVAL 1 DAY;",
            }
        }
    }

    @field_validator("question", "sql")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("字段不能为空")
        return cleaned


class SyncTrainRequest(BaseModel):
    database_id: int = Field(description="鐩爣鏁版嵁搴?ID")
    sync_ddl: bool = Field(default=True, description="鏄惁鍚屾 DDL 鎸囩汗")
    sync_docs: bool = Field(default=True, description="鏄惁鍚屾璁粌鏂囨。")
    sync_examples: bool = Field(default=True, description="鏄惁鍚屾绀轰緥 SQL")
    force_sync: bool = Field(default=False, description="force current DDL rewrite to Chroma")

    model_config = {
        "json_schema_extra": {
            "example": {
                "database_id": 1,
                "sync_ddl": True,
                "sync_docs": True,
                "sync_examples": True,
                "force_sync": False,
            }
        }
    }


class DdlSyncRequest(BaseModel):
    database_id: int = Field(description="鐩爣鏁版嵁搴?ID")
    force_sync: bool = Field(default=False, description="force current DDL rewrite to Chroma")

    model_config = {
        "json_schema_extra": {"example": {"database_id": 1, "force_sync": False}}
    }


class TrainingDataResponse(BaseModel):
    id: int
    database_id: int
    content: str | None = None
    question: str | None = None
    sql: str | None = None
    content_hash: str
    trained_content_hash: str | None = None
    vanna_training_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_trained_at: datetime | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": 12,
                "database_id": 1,
                "content": "订单表 order_info 中 status=1 表示已支付。",
                "question": None,
                "sql": None,
                "content_hash": "5f4dcc3b5aa765d61d8327deb882cf99b8c1f0a6d4f6f2d7f0fca123456789ab",
                "trained_content_hash": "5f4dcc3b5aa765d61d8327deb882cf99b8c1f0a6d4f6f2d7f0fca123456789ab",
                "vanna_training_id": "2c9f-doc",
                "created_at": "2026-04-22T10:05:00",
                "updated_at": "2026-04-22T10:05:00",
                "last_trained_at": "2026-04-22T10:06:00",
            }
        }
    }


class SyncResult(BaseModel):
    database_id: int
    docs_added: int = 0
    docs_updated: int = 0
    docs_deleted: int = 0
    examples_added: int = 0
    examples_updated: int = 0
    examples_deleted: int = 0
    ddl_added: int = 0
    ddl_updated: int = 0
    ddl_deleted: int = 0
    ddl_unchanged: int = 0

    model_config = {
        "json_schema_extra": {
            "example": {
                "database_id": 1,
                "docs_added": 2,
                "docs_updated": 1,
                "docs_deleted": 0,
                "examples_added": 3,
                "examples_updated": 0,
                "examples_deleted": 0,
                "ddl_added": 15,
                "ddl_updated": 2,
                "ddl_deleted": 1,
                "ddl_unchanged": 28,
            }
        }
    }


@dataclass
class ServiceConfig:
    qwen_api_key: str | None
    qwen_model: str
    qwen_base_url: str
    qwen_request_timeout: float
    allow_llm_to_see_data: bool
    chroma_path: str
    response_language: str
    metadata_host: str
    metadata_port: int
    metadata_database: str
    metadata_user: str
    metadata_password: str
    default_database_key: str | None
    default_business_host: str | None
    default_business_port: int | None
    default_business_database: str | None
    default_business_user: str | None
    default_business_password: str | None
    default_ddl_table_filter: list[str]

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        qwen_api_key = _clean_optional_str(
            os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        )

        metadata_host = os.getenv("CHATBI_SYS_HOST") or os.getenv("MYSQL_HOST")
        metadata_user = os.getenv("CHATBI_SYS_USER") or os.getenv("MYSQL_USER")
        metadata_password = os.getenv("CHATBI_SYS_PASSWORD") or os.getenv("MYSQL_PASSWORD")
        if not metadata_host:
            raise RuntimeError("Missing CHATBI_SYS_HOST or MYSQL_HOST")
        if not metadata_user:
            raise RuntimeError("Missing CHATBI_SYS_USER or MYSQL_USER")
        if metadata_password is None:
            raise RuntimeError("Missing CHATBI_SYS_PASSWORD or MYSQL_PASSWORD")

        business_host = _clean_optional_str(os.getenv("MYSQL_HOST"))
        business_database = _clean_optional_str(os.getenv("MYSQL_DATABASE"))
        business_user = _clean_optional_str(os.getenv("MYSQL_USER"))
        business_password = _clean_optional_str(os.getenv("MYSQL_PASSWORD"))
        default_database_key = None
        if business_host and business_database and business_user and business_password:
            default_database_key = _normalize_database_key(
                os.getenv("CHATBI_DEFAULT_DATABASE_KEY", "default")
            )

        return cls(
            qwen_api_key=qwen_api_key,
            qwen_model=os.getenv("QWEN_MODEL", "qwen3.6-plus"),
            qwen_base_url=os.getenv(
                "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            qwen_request_timeout=float(os.getenv("QWEN_HTTP_TIMEOUT_SECONDS", "60")),
            allow_llm_to_see_data=_parse_env_bool(
                os.getenv("ALLOW_LLM_TO_SEE_DATA"), default=False
            ),
            chroma_path=os.getenv(
                "VANNA_CHROMA_PATH", str(Path(".chroma") / "chatbi_nl2sql")
            ),
            response_language=os.getenv("VANNA_RESPONSE_LANGUAGE", "Chinese"),
            metadata_host=metadata_host,
            metadata_port=int(
                os.getenv("CHATBI_SYS_PORT", os.getenv("MYSQL_PORT", "3306"))
            ),
            metadata_database=os.getenv("CHATBI_SYS_DATABASE", "chatbi_sys"),
            metadata_user=metadata_user,
            metadata_password=metadata_password,
            default_database_key=default_database_key,
            default_business_host=business_host,
            default_business_port=int(os.getenv("MYSQL_PORT", "3306"))
            if business_host
            else None,
            default_business_database=business_database,
            default_business_user=business_user,
            default_business_password=business_password,
            default_ddl_table_filter=_parse_table_filter(os.getenv("MYSQL_TRAIN_TABLES")),
        )


class ChatBINL2SQLService:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.metadata = MetaData()

        # 所有可参与 NL2SQL 的业务数据库都会登记在这张总表中。
        self.database_table = Table(
            "chatbi_vanna_database",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True, comment="主键"),
            Column(
                "database_key",
                String(128),
                nullable=False,
                unique=True,
                comment="逻辑数据库唯一键，用于稳定标识一个业务库",
            ),
            Column(
                "display_name",
                String(255),
                nullable=False,
                comment="给运维或业务人员看的数据库展示名称",
            ),
            Column(
                "db_type",
                String(32),
                nullable=False,
                default="mysql",
                comment="数据库类型，当前仅支持 mysql",
            ),
            Column("host", String(255), nullable=False, comment="数据库主机地址"),
            Column("port", Integer, nullable=False, comment="数据库端口"),
            Column(
                "database_name",
                String(255),
                nullable=False,
                comment="业务数据库名或 schema 名",
            ),
            Column("username", String(255), nullable=False, comment="数据库用户名"),
            Column(
                "password",
                Text,
                nullable=False,
                comment="数据库密码，当前版本为明文存储",
            ),
            Column(
                "ddl_table_filter",
                JSON,
                nullable=False,
                default=list,
                comment="DDL 指纹同步时纳入训练的表白名单",
            ),
            Column("description", Text, comment="数据库备注说明"),
            Column(
                "is_active",
                Boolean,
                nullable=False,
                default=True,
                comment="是否可用于训练与 SQL 生成",
            ),
            Column("created_at", DateTime, nullable=False, comment="创建时间"),
            Column("updated_at", DateTime, nullable=False, comment="更新时间"),
            comment="多数据库 NL2SQL 的业务数据库配置表",
        )

        # 人工维护的业务文档，按 database_id 做严格隔离。
        self.documentation_table = Table(
            "chatbi_vanna_documentation",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True, comment="主键"),
            Column(
                "database_id",
                Integer,
                nullable=False,
                comment="所属数据库 ID，对应 chatbi_vanna_database.id",
            ),
            Column("content", Text, nullable=False, comment="业务文档内容"),
            Column(
                "content_hash",
                String(64),
                nullable=False,
                comment="当前文档内容的哈希值",
            ),
            Column(
                "trained_content_hash",
                String(64),
                comment="最近一次成功训练时使用的内容哈希值",
            ),
            Column(
                "vanna_training_id",
                String(128),
                comment="Vanna/Chroma 返回的训练记录 ID",
            ),
            Column("created_at", DateTime, nullable=False, comment="创建时间"),
            Column("updated_at", DateTime, nullable=False, comment="更新时间"),
            Column(
                "last_trained_at",
                DateTime,
                comment="最近一次同步到 Vanna 的时间",
            ),
            comment="训练业务文档表，按数据库隔离",
        )

        # 人工整理的问答 SQL 示例，用于 few-shot 检索增强，按库隔离。
        self.example_table = Table(
            "chatbi_vanna_example_sql",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True, comment="主键"),
            Column(
                "database_id",
                Integer,
                nullable=False,
                comment="所属数据库 ID，对应 chatbi_vanna_database.id",
            ),
            Column("question", Text, nullable=False, comment="示例自然语言问题"),
            Column("sql_text", Text, nullable=False, comment="标准参考 SQL"),
            Column(
                "content_hash",
                String(64),
                nullable=False,
                comment="问题与 SQL 合并后的内容哈希值",
            ),
            Column(
                "trained_content_hash",
                String(64),
                comment="最近一次成功训练时使用的内容哈希值",
            ),
            Column(
                "vanna_training_id",
                String(128),
                comment="Vanna/Chroma 返回的训练记录 ID",
            ),
            Column("created_at", DateTime, nullable=False, comment="创建时间"),
            Column("updated_at", DateTime, nullable=False, comment="更新时间"),
            Column(
                "last_trained_at",
                DateTime,
                comment="最近一次同步到 Vanna 的时间",
            ),
            comment="训练示例 SQL 表，按数据库隔离",
        )

        # DDL 指纹表用于识别变化，只重训发生变更的表结构。
        self.ddl_table = Table(
            "chatbi_vanna_ddl_fingerprint",
            self.metadata,
            Column(
                "table_name",
                String(255),
                primary_key=True,
                comment="内部唯一键，格式为 database_key::source_table_name",
            ),
            Column(
                "database_id",
                Integer,
                nullable=False,
                comment="所属数据库 ID，对应 chatbi_vanna_database.id",
            ),
            Column(
                "source_table_name",
                String(255),
                nullable=False,
                comment="业务数据库中的真实物理表名",
            ),
            Column("ddl_sql", Text, nullable=False, comment="最新的 CREATE TABLE DDL 文本"),
            Column("ddl_hash", String(64), nullable=False, comment="当前 DDL 的哈希值"),
            Column(
                "vanna_training_id",
                String(128),
                comment="Vanna/Chroma 返回的训练记录 ID",
            ),
            Column(
                "last_trained_at",
                DateTime,
                comment="最近一次同步该表 DDL 到 Vanna 的时间",
            ),
            Column(
                "updated_at",
                DateTime,
                nullable=False,
                comment="最近一次从源数据库刷新 DDL 指纹的时间",
            ),
            comment="按数据库做增量 schema 训练的 DDL 指纹表",
        )

        # 训练同步审计日志，方便排查某次同步到底做了什么。
        self.sync_log_table = Table(
            "chatbi_vanna_sync_log",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True, comment="主键"),
            Column(
                "database_id",
                Integer,
                comment="目标数据库 ID，全局或迁移级操作时可为空",
            ),
            Column("sync_type", String(64), nullable=False, comment="同步操作类型"),
            Column("result_json", JSON, nullable=False, comment="同步结果 JSON"),
            Column("created_at", DateTime, nullable=False, comment="创建时间"),
            comment="训练同步操作审计日志表",
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
        self._run_metadata_migrations()
        self._ensure_default_database_from_env()

    def _ensure_metadata_database_exists(self) -> None:
        import pymysql

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

    def _table_exists(self, table_name: str) -> bool:
        with self._metadata_connection() as conn:
            exists = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = :schema_name AND table_name = :table_name
                    """
                ),
                {"schema_name": self.config.metadata_database, "table_name": table_name},
            ).scalar_one()
        return bool(exists)

    def _column_exists(self, table_name: str, column_name: str) -> bool:
        with self._metadata_connection() as conn:
            exists = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM information_schema.columns
                    WHERE table_schema = :schema_name
                      AND table_name = :table_name
                      AND column_name = :column_name
                    """
                ),
                {
                    "schema_name": self.config.metadata_database,
                    "table_name": table_name,
                    "column_name": column_name,
                },
            ).scalar_one()
        return bool(exists)

    def _index_exists(self, table_name: str, index_name: str) -> bool:
        with self._metadata_connection() as conn:
            exists = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM information_schema.statistics
                    WHERE table_schema = :schema_name
                      AND table_name = :table_name
                      AND index_name = :index_name
                    """
                ),
                {
                    "schema_name": self.config.metadata_database,
                    "table_name": table_name,
                    "index_name": index_name,
                },
            ).scalar_one()
        return bool(exists)

    def _run_metadata_migrations(self) -> None:
        """为老版本 sys 表补齐新增列与索引。

        这样旧的单数据库部署可以平滑升级，不需要额外引入独立迁移工具。
        """
        if self._table_exists("chatbi_vanna_documentation"):
            if not self._column_exists("chatbi_vanna_documentation", "database_id"):
                with self._metadata_connection() as conn:
                    conn.execute(
                        text(
                            """
                            ALTER TABLE chatbi_vanna_documentation
                            ADD COLUMN database_id INT NULL AFTER id
                            """
                        )
                    )

        if self._table_exists("chatbi_vanna_example_sql"):
            if not self._column_exists("chatbi_vanna_example_sql", "database_id"):
                with self._metadata_connection() as conn:
                    conn.execute(
                        text(
                            """
                            ALTER TABLE chatbi_vanna_example_sql
                            ADD COLUMN database_id INT NULL AFTER id
                            """
                        )
                    )

        if self._table_exists("chatbi_vanna_ddl_fingerprint"):
            if not self._column_exists("chatbi_vanna_ddl_fingerprint", "database_id"):
                with self._metadata_connection() as conn:
                    conn.execute(
                        text(
                            """
                            ALTER TABLE chatbi_vanna_ddl_fingerprint
                            ADD COLUMN database_id INT NULL AFTER table_name
                            """
                        )
                    )
            if not self._column_exists("chatbi_vanna_ddl_fingerprint", "source_table_name"):
                with self._metadata_connection() as conn:
                    conn.execute(
                        text(
                            """
                            ALTER TABLE chatbi_vanna_ddl_fingerprint
                            ADD COLUMN source_table_name VARCHAR(255) NULL AFTER database_id
                            """
                        )
                    )

        if self._table_exists("chatbi_vanna_sync_log"):
            if not self._column_exists("chatbi_vanna_sync_log", "database_id"):
                with self._metadata_connection() as conn:
                    conn.execute(
                        text(
                            """
                            ALTER TABLE chatbi_vanna_sync_log
                            ADD COLUMN database_id INT NULL AFTER id
                            """
                        )
                    )

        if self._table_exists("chatbi_vanna_documentation") and not self._index_exists(
            "chatbi_vanna_documentation", "idx_chatbi_doc_database_id"
        ):
            with self._metadata_connection() as conn:
                conn.execute(
                    text(
                        """
                        CREATE INDEX idx_chatbi_doc_database_id
                        ON chatbi_vanna_documentation (database_id)
                        """
                    )
                )

        if self._table_exists("chatbi_vanna_example_sql") and not self._index_exists(
            "chatbi_vanna_example_sql", "idx_chatbi_example_database_id"
        ):
            with self._metadata_connection() as conn:
                conn.execute(
                    text(
                        """
                        CREATE INDEX idx_chatbi_example_database_id
                        ON chatbi_vanna_example_sql (database_id)
                        """
                    )
                )

        if self._table_exists("chatbi_vanna_ddl_fingerprint") and not self._index_exists(
            "chatbi_vanna_ddl_fingerprint", "idx_chatbi_ddl_database_id"
        ):
            with self._metadata_connection() as conn:
                conn.execute(
                    text(
                        """
                        CREATE INDEX idx_chatbi_ddl_database_id
                        ON chatbi_vanna_ddl_fingerprint (database_id)
                        """
                    )
                )

    def _ensure_default_database_from_env(self) -> None:
        """基于旧环境变量自动生成一个默认数据库配置。

        这样既保留原来的单库启动方式，又把运行模型迁移到显式数据库注册。
        """
        if not self.config.default_database_key:
            return

        now = _utcnow()
        with self._metadata_connection() as conn:
            row = conn.execute(
                select(self.database_table).where(
                    self.database_table.c.database_key == self.config.default_database_key
                )
            ).mappings().first()
            if row is None:
                result = conn.execute(
                    insert(self.database_table).values(
                        database_key=self.config.default_database_key,
                        display_name=os.getenv("CHATBI_DEFAULT_DATABASE_NAME", "Default MySQL"),
                        db_type="mysql",
                        host=self.config.default_business_host,
                        port=self.config.default_business_port,
                        database_name=self.config.default_business_database,
                        username=self.config.default_business_user,
                        password=self.config.default_business_password,
                        ddl_table_filter=self.config.default_ddl_table_filter,
                        description="Bootstrapped from environment variables",
                        is_active=True,
                        created_at=now,
                        updated_at=now,
                    )
                )
                database_id = result.inserted_primary_key[0]
            else:
                database_id = row["id"]

            self._backfill_legacy_rows_to_database(conn, database_id)

    def _backfill_legacy_rows_to_database(self, conn: Any, database_id: int) -> None:
        # 旧版单数据库数据会回填到默认数据库名下，升级后仍可继续使用。
        if self._column_exists("chatbi_vanna_documentation", "database_id"):
            conn.execute(
                text(
                    """
                    UPDATE chatbi_vanna_documentation
                    SET database_id = :database_id
                    WHERE database_id IS NULL
                    """
                ),
                {"database_id": database_id},
            )

        if self._column_exists("chatbi_vanna_example_sql", "database_id"):
            conn.execute(
                text(
                    """
                    UPDATE chatbi_vanna_example_sql
                    SET database_id = :database_id
                    WHERE database_id IS NULL
                    """
                ),
                {"database_id": database_id},
            )

        if self._column_exists("chatbi_vanna_ddl_fingerprint", "database_id"):
            conn.execute(
                text(
                    """
                    UPDATE chatbi_vanna_ddl_fingerprint
                    SET database_id = :database_id
                    WHERE database_id IS NULL
                    """
                ),
                {"database_id": database_id},
            )

        if self._column_exists("chatbi_vanna_ddl_fingerprint", "source_table_name"):
            conn.execute(
                text(
                    """
                    UPDATE chatbi_vanna_ddl_fingerprint
                    SET source_table_name = table_name
                    WHERE source_table_name IS NULL OR source_table_name = ''
                    """
                )
            )

    def _row_to_database_response(self, row: dict[str, Any]) -> DatabaseConfigResponse:
        return DatabaseConfigResponse(
            id=row["id"],
            database_key=row["database_key"],
            display_name=row["display_name"],
            db_type=row["db_type"],
            host=row["host"],
            port=row["port"],
            database_name=row["database_name"],
            username=row["username"],
            ddl_table_filter=row.get("ddl_table_filter") or [],
            description=row.get("description"),
            is_active=bool(row["is_active"]),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            has_password=bool(row.get("password")),
        )

    def _get_database_row(
        self, database_id: int, *, require_active: bool = True
    ) -> dict[str, Any]:
        with self._metadata_connection() as conn:
            row = conn.execute(
                select(self.database_table).where(self.database_table.c.id == database_id)
            ).mappings().first()
        if row is None:
            raise HTTPException(status_code=404, detail="数据库配置不存在")
        if require_active and not row["is_active"]:
            raise HTTPException(status_code=400, detail="数据库配置已停用")
        if row["db_type"] != "mysql":
            raise HTTPException(
                status_code=400,
                detail=f"数据库类型 `{row['db_type']}` 暂不支持",
            )
        return dict(row)

    def _fingerprint_key(self, database_key: str, table_name: str) -> str:
        # sys 表主键需要全局唯一，而 source_table_name 只需要在单库内唯一。
        return f"{database_key}::{table_name}"

    def _build_vanna(self, database_row: dict[str, Any]) -> ChatBIVanna:
        """构建一个按数据库隔离的 Vanna 实例。

        每个数据库都会绑定独立的 Chroma 集合，避免 DDL、文档、示例 SQL
        被其它数据库误检索到。
        """
        vanna_config = {
            "model": self.config.qwen_model,
            "base_url": self.config.qwen_base_url,
            "request_timeout": self.config.qwen_request_timeout,
            "path": self.config.chroma_path,
            "dialect": "MySQL",
            "language": self.config.response_language,
            "collection_suffix": _collection_suffix(database_row["database_key"]),
        }
        if self.config.qwen_api_key:
            vanna_config["api_key"] = self.config.qwen_api_key

        vn = ChatBIVanna(config=vanna_config)
        vn.connect_to_mysql(
            host=database_row["host"],
            port=int(database_row["port"]),
            dbname=database_row["database_name"],
            user=database_row["username"],
            password=database_row["password"],
        )
        return vn

    def _save_sync_log(self, database_id: int, sync_type: str, result: SyncResult) -> None:
        with self._metadata_connection() as conn:
            conn.execute(
                insert(self.sync_log_table).values(
                    database_id=database_id,
                    sync_type=sync_type,
                    result_json=result.model_dump(),
                    created_at=_utcnow(),
                )
            )

    def list_databases(self) -> list[DatabaseConfigResponse]:
        with self._metadata_connection() as conn:
            rows = conn.execute(select(self.database_table).order_by(self.database_table.c.id))
            return [self._row_to_database_response(dict(row._mapping)) for row in rows]

    def get_database(self, database_id: int) -> DatabaseConfigResponse:
        row = self._get_database_row(database_id, require_active=False)
        return self._row_to_database_response(row)

    def create_database(
        self, payload: DatabaseConfigCreateRequest
    ) -> DatabaseConfigResponse:
        now = _utcnow()
        with self._metadata_connection() as conn:
            existing = conn.execute(
                select(self.database_table.c.id).where(
                    self.database_table.c.database_key == payload.database_key
                )
            ).first()
            if existing is not None:
                raise HTTPException(status_code=409, detail="database_key 已存在")

            result = conn.execute(
                insert(self.database_table).values(
                    database_key=payload.database_key,
                    display_name=payload.display_name,
                    db_type=payload.db_type,
                    host=payload.host,
                    port=payload.port,
                    database_name=payload.database_name,
                    username=payload.username,
                    password=payload.password,
                    ddl_table_filter=payload.ddl_table_filter,
                    description=payload.description,
                    is_active=payload.is_active,
                    created_at=now,
                    updated_at=now,
                )
            )
            database_id = result.inserted_primary_key[0]
        return self.get_database(database_id)

    def update_database(
        self, database_id: int, payload: DatabaseConfigUpdateRequest
    ) -> DatabaseConfigResponse:
        self._get_database_row(database_id, require_active=False)

        values: dict[str, Any] = {}
        for field_name in (
            "display_name",
            "db_type",
            "host",
            "port",
            "database_name",
            "username",
            "password",
            "ddl_table_filter",
            "description",
            "is_active",
        ):
            field_value = getattr(payload, field_name)
            if field_value is not None:
                values[field_name] = field_value

        if not values:
            return self.get_database(database_id)

        values["updated_at"] = _utcnow()
        with self._metadata_connection() as conn:
            conn.execute(
                update(self.database_table)
                .where(self.database_table.c.id == database_id)
                .values(**values)
            )
        return self.get_database(database_id)

    def test_database_connection(self, database_id: int) -> DatabaseConnectionTestResponse:
        database_row = self._get_database_row(database_id, require_active=False)
        try:
            test_engine = self._create_engine(
                host=database_row["host"],
                port=int(database_row["port"]),
                database=database_row["database_name"],
                user=database_row["username"],
                password=database_row["password"],
            )
            with test_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return DatabaseConnectionTestResponse(
                database_id=database_row["id"],
                database_key=database_row["database_key"],
                connected=True,
                message="连接成功",
            )
        except Exception as exc:  # pragma: no cover - depends on real database
            return DatabaseConnectionTestResponse(
                database_id=database_row["id"],
                database_key=database_row["database_key"],
                connected=False,
                message=str(exc),
            )

    def list_documentation(self, database_id: int | None = None) -> list[TrainingDataResponse]:
        stmt = select(self.documentation_table).order_by(self.documentation_table.c.id)
        if database_id is not None:
            self._get_database_row(database_id)
            stmt = stmt.where(self.documentation_table.c.database_id == database_id)

        with self._metadata_connection() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [
            TrainingDataResponse(
                id=row["id"],
                database_id=row["database_id"],
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

    def create_documentation(
        self, payload: DocumentationCreateRequest
    ) -> TrainingDataResponse:
        self._get_database_row(payload.database_id)
        now = _utcnow()
        content_hash = _sha256(payload.content)
        with self._metadata_connection() as conn:
            result = conn.execute(
                insert(self.documentation_table).values(
                    database_id=payload.database_id,
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
            database_id=row["database_id"],
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
        with self._metadata_connection() as conn:
            row = conn.execute(
                select(self.documentation_table).where(self.documentation_table.c.id == doc_id)
            ).mappings().first()
            if row is None:
                raise HTTPException(status_code=404, detail="Documentation not found")

        database_row = self._get_database_row(row["database_id"])
        vn = self._build_vanna(database_row)

        with self._metadata_connection() as conn:
            if row["vanna_training_id"]:
                vn.remove_training_data(row["vanna_training_id"])

            conn.execute(
                delete(self.documentation_table).where(self.documentation_table.c.id == doc_id)
            )
        return {"deleted": True, "id": doc_id}

    def list_examples(self, database_id: int | None = None) -> list[TrainingDataResponse]:
        stmt = select(self.example_table).order_by(self.example_table.c.id)
        if database_id is not None:
            self._get_database_row(database_id)
            stmt = stmt.where(self.example_table.c.database_id == database_id)

        with self._metadata_connection() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [
            TrainingDataResponse(
                id=row["id"],
                database_id=row["database_id"],
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
        self._get_database_row(payload.database_id)
        now = _utcnow()
        content_hash = _sha256(f"{payload.question}\n{payload.sql}")
        with self._metadata_connection() as conn:
            result = conn.execute(
                insert(self.example_table).values(
                    database_id=payload.database_id,
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
            raise HTTPException(status_code=404, detail="训练示例不存在")
        return TrainingDataResponse(
            id=row["id"],
            database_id=row["database_id"],
            question=row["question"],
            sql=row["sql_text"],
            content_hash=row["content_hash"],
            trained_content_hash=row["trained_content_hash"],
            vanna_training_id=row["vanna_training_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_trained_at=row["last_trained_at"],
        )

    def update_example(
        self, example_id: int, payload: ExampleUpdateRequest
    ) -> TrainingDataResponse:
        with self._metadata_connection() as conn:
            row = conn.execute(
                select(self.example_table).where(self.example_table.c.id == example_id)
            ).mappings().first()
            if row is None:
                raise HTTPException(status_code=404, detail="训练示例不存在")

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
        with self._metadata_connection() as conn:
            row = conn.execute(
                select(self.example_table).where(self.example_table.c.id == example_id)
            ).mappings().first()
            if row is None:
                raise HTTPException(status_code=404, detail="训练示例不存在")

        database_row = self._get_database_row(row["database_id"])
        vn = self._build_vanna(database_row)

        with self._metadata_connection() as conn:
            if row["vanna_training_id"]:
                vn.remove_training_data(row["vanna_training_id"])
            conn.execute(delete(self.example_table).where(self.example_table.c.id == example_id))
        return {"deleted": True, "id": example_id}

    def _sync_documentation(
        self, vn: ChatBIVanna, database_id: int, result: SyncResult
    ) -> None:
        with self._metadata_connection() as conn:
            rows = conn.execute(
                select(self.documentation_table).where(
                    self.documentation_table.c.database_id == database_id
                )
            ).mappings().all()

            for row in rows:
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

    def _sync_examples(self, vn: ChatBIVanna, database_id: int, result: SyncResult) -> None:
        with self._metadata_connection() as conn:
            rows = conn.execute(
                select(self.example_table).where(self.example_table.c.database_id == database_id)
            ).mappings().all()

            for row in rows:
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

    def _iter_business_tables(self, vn: ChatBIVanna, database_row: dict[str, Any]) -> list[str]:
        df_tables = vn.run_sql(f"SHOW TABLES FROM `{database_row['database_name']}`")
        if df_tables.empty:
            return []
        table_column = df_tables.columns[0]
        table_names = df_tables[table_column].tolist()
        table_filter = database_row.get("ddl_table_filter") or []
        if table_filter:
            table_names = [name for name in table_names if name in table_filter]
        return table_names

    def _sync_ddl(
        self,
        vn: ChatBIVanna,
        database_row: dict[str, Any],
        result: SyncResult,
        *,
        force_sync: bool = False,
    ) -> None:
        """刷新指定数据库的 DDL 指纹，并仅重训有变化的表。"""
        current_ddls: dict[str, tuple[str, str]] = {}
        for table_name in self._iter_business_tables(vn, database_row):
            df_ddl = vn.run_sql(f"SHOW CREATE TABLE `{table_name}`")
            ddl_sql = df_ddl.iloc[0, 1]
            current_ddls[table_name] = (ddl_sql, _sha256(ddl_sql))

        with self._metadata_connection() as conn:
            stored_rows = {
                row["source_table_name"]: row
                for row in conn.execute(
                    select(self.ddl_table).where(self.ddl_table.c.database_id == database_row["id"])
                ).mappings().all()
            }

            for table_name, (ddl_sql, ddl_hash) in current_ddls.items():
                stored = stored_rows.get(table_name)
                # DDL 哈希一致，说明当前表结构上下文无需重复训练。
                if (
                    not force_sync
                    and stored
                    and stored["ddl_hash"] == ddl_hash
                    and stored["vanna_training_id"]
                ):
                    result.ddl_unchanged += 1
                    continue

                if stored and stored["vanna_training_id"]:
                    vn.remove_training_data(stored["vanna_training_id"])
                    result.ddl_updated += 1
                else:
                    result.ddl_added += 1

                new_training_id = vn.train(ddl=ddl_sql)
                # sys 表里用全局唯一键存储，同时保留原始表名用于展示与查询。
                fingerprint_key = self._fingerprint_key(database_row["database_key"], table_name)

                if stored:
                    conn.execute(
                        update(self.ddl_table)
                        .where(self.ddl_table.c.table_name == stored["table_name"])
                        .values(
                            table_name=fingerprint_key,
                            database_id=database_row["id"],
                            source_table_name=table_name,
                            ddl_sql=ddl_sql,
                            ddl_hash=ddl_hash,
                            vanna_training_id=new_training_id,
                            last_trained_at=_utcnow(),
                            updated_at=_utcnow(),
                        )
                    )
                else:
                    conn.execute(
                        insert(self.ddl_table).values(
                            table_name=fingerprint_key,
                            database_id=database_row["id"],
                            source_table_name=table_name,
                            ddl_sql=ddl_sql,
                            ddl_hash=ddl_hash,
                            vanna_training_id=new_training_id,
                            last_trained_at=_utcnow(),
                            updated_at=_utcnow(),
                        )
                    )

            removed_table_names = set(stored_rows) - set(current_ddls)
            for table_name in removed_table_names:
                stored = stored_rows[table_name]
                if stored["vanna_training_id"]:
                    vn.remove_training_data(stored["vanna_training_id"])
                conn.execute(
                    delete(self.ddl_table).where(self.ddl_table.c.table_name == stored["table_name"])
                )
                result.ddl_deleted += 1

    def sync_training(self, payload: SyncTrainRequest) -> SyncResult:
        """同步指定数据库的文档、示例 SQL 与 DDL 数据。"""
        database_row = self._get_database_row(payload.database_id)
        vn = self._build_vanna(database_row)
        result = SyncResult(database_id=payload.database_id)

        if payload.sync_docs:
            self._sync_documentation(vn, payload.database_id, result)
        if payload.sync_examples:
            self._sync_examples(vn, payload.database_id, result)
        if payload.sync_ddl:
            self._sync_ddl(vn, database_row, result, force_sync=payload.force_sync)

        self._save_sync_log(payload.database_id, "manual_sync", result)
        return result

    def sync_ddl_only(self, payload: DdlSyncRequest) -> SyncResult:
        return self.sync_training(
            SyncTrainRequest(
                database_id=payload.database_id,
                sync_ddl=True,
                sync_docs=False,
                sync_examples=False,
                force_sync=payload.force_sync,
            )
        )

    def generate_sql(self, payload: GenerateSqlRequest) -> GenerateSqlResponse:
        """基于指定数据库生成 SQL。

        选中的数据库同时决定实际 SQL 连接以及 DDL、文档、示例 SQL 的检索范围。
        """
        if payload.auto_sync_training:
            self.sync_training(SyncTrainRequest(database_id=payload.database_id))

        database_row = self._get_database_row(payload.database_id)
        vn = self._build_vanna(database_row)
        sql = vn.generate_sql(
            payload.question,
            allow_llm_to_see_data=self.config.allow_llm_to_see_data,
        )
        return GenerateSqlResponse(
            database_id=payload.database_id,
            question=payload.question,
            sql=sql,
        )

    def list_ddl_fingerprints(
        self, database_id: int | None = None
    ) -> list[dict[str, Any]]:
        stmt = select(self.ddl_table).order_by(self.ddl_table.c.source_table_name)
        if database_id is not None:
            self._get_database_row(database_id)
            stmt = stmt.where(self.ddl_table.c.database_id == database_id)

        with self._metadata_connection() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(row) for row in rows]


service = ChatBINL2SQLService(ServiceConfig.from_env())
tags_metadata = [
    {"name": "系统", "description": "健康检查与文档导航接口"},
    {"name": "数据库配置", "description": "管理可用于 NL2SQL 的业务数据库连接配置"},
    {"name": "SQL生成", "description": "将自然语言问题转换为 SQL"},
    {"name": "训练文档", "description": "管理用于训练的业务文档"},
    {"name": "训练示例", "description": "管理用于训练的问答 SQL 示例"},
    {"name": "训练同步", "description": "将 DDL、文档和示例 SQL 同步到 Vanna"},
]

app = FastAPI(
    title="ChatBI NL2SQL 接口文档",
    version="0.2.0",
    description=(
        "支持多数据库隔离的 NL2SQL 服务。训练数据与 SQL 生成上下文都按 "
        "database_id 进行隔离，避免生产环境多数据库之间互相干扰。"
    ),
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


@app.get(
    "/api/chatbi/v1/databases",
    response_model=list[DatabaseConfigResponse],
    tags=["数据库配置"],
    summary="查询数据库配置列表",
)
def list_databases() -> list[DatabaseConfigResponse]:
    return service.list_databases()


@app.post(
    "/api/chatbi/v1/databases",
    response_model=DatabaseConfigResponse,
    tags=["数据库配置"],
    summary="新增数据库配置",
)
def create_database(payload: DatabaseConfigCreateRequest) -> DatabaseConfigResponse:
    return service.create_database(payload)


@app.get(
    "/api/chatbi/v1/databases/{database_id}",
    response_model=DatabaseConfigResponse,
    tags=["数据库配置"],
    summary="查询单个数据库配置",
)
def get_database(database_id: int) -> DatabaseConfigResponse:
    return service.get_database(database_id)


@app.put(
    "/api/chatbi/v1/databases/{database_id}",
    response_model=DatabaseConfigResponse,
    tags=["数据库配置"],
    summary="更新数据库配置",
)
def update_database(
    database_id: int, payload: DatabaseConfigUpdateRequest
) -> DatabaseConfigResponse:
    return service.update_database(database_id, payload)


@app.post(
    "/api/chatbi/v1/databases/{database_id}/test-connection",
    response_model=DatabaseConnectionTestResponse,
    tags=["数据库配置"],
    summary="测试数据库连接",
)
def test_database_connection(database_id: int) -> DatabaseConnectionTestResponse:
    return service.test_database_connection(database_id)


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
def list_documentation(
    database_id: int | None = Query(default=None)
) -> list[TrainingDataResponse]:
    return service.list_documentation(database_id)


@app.post(
    "/api/chatbi/v1/training/documentation",
    response_model=TrainingDataResponse,
    tags=["训练文档"],
    summary="新增训练文档",
)
def create_documentation(payload: DocumentationCreateRequest) -> TrainingDataResponse:
    return service.create_documentation(payload)


@app.get(
    "/api/chatbi/v1/training/documentation/{doc_id}",
    response_model=TrainingDataResponse,
    tags=["训练文档"],
    summary="查询单条训练文档",
)
def get_documentation(doc_id: int) -> TrainingDataResponse:
    return service.get_documentation(doc_id)


@app.put(
    "/api/chatbi/v1/training/documentation/{doc_id}",
    response_model=TrainingDataResponse,
    tags=["训练文档"],
    summary="更新训练文档",
)
def update_documentation(
    doc_id: int, payload: DocumentationUpdateRequest
) -> TrainingDataResponse:
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
def list_examples(database_id: int | None = Query(default=None)) -> list[TrainingDataResponse]:
    return service.list_examples(database_id)


@app.post(
    "/api/chatbi/v1/training/examples",
    response_model=TrainingDataResponse,
    tags=["训练示例"],
    summary="新增训练示例",
)
def create_example(payload: ExampleCreateRequest) -> TrainingDataResponse:
    return service.create_example(payload)


@app.get(
    "/api/chatbi/v1/training/examples/{example_id}",
    response_model=TrainingDataResponse,
    tags=["训练示例"],
    summary="查询单条训练示例",
)
def get_example(example_id: int) -> TrainingDataResponse:
    return service.get_example(example_id)


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
    summary="同步指定数据库的训练数据",
)
def sync_training(payload: SyncTrainRequest) -> SyncResult:
    return service.sync_training(payload)


@app.post(
    "/api/chatbi/v1/train/ddl/sync",
    response_model=SyncResult,
    tags=["训练同步"],
    summary="仅同步指定数据库的 DDL 指纹",
)
def sync_ddl(payload: DdlSyncRequest) -> SyncResult:
    return service.sync_ddl_only(payload)


@app.get(
    "/api/chatbi/v1/train/ddl-fingerprints",
    tags=["训练同步"],
    summary="查询 DDL 指纹列表",
)
def list_ddl_fingerprints(
    database_id: int | None = Query(default=None),
) -> list[dict[str, Any]]:
    return service.list_ddl_fingerprints(database_id)
