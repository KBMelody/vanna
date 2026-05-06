# ChatBI NL2SQL 离线 Docker 镜像打包指南

> 本文档记录如何将 ChatBI NL2SQL 服务打包为可在内网离线部署的 Docker 镜像。  
> 如需查看 Vanna 官方原版文档，请见 [README_OFFICIAL.md](README_OFFICIAL.md)。

---

## 整体流程概览

```
外网环境（可联网）
    │
    ├── ① 下载 Python wheels（离线依赖包）
    ├── ② 构建 Docker 基础镜像
    ├── ③ 启动容器并预下载 ONNX 模型（联网）
    ├── ④ 测试验证通过后，将容器 commit 为新镜像
    └── ⑤ docker save 导出 tar 包 ───→ 拷贝到内网
                                              │
内网环境（不可联网）                            ▼
    │                                    docker load 导入
    └── ⑥ docker-compose up -d 直接启动
```

---

## 涉及的关键文件

| 文件 | 作用 |
|------|------|
| [`scripts/download-chatbi-wheels.ps1`](scripts/download-chatbi-wheels.ps1) | PowerShell 脚本，负责从 PyPI 下载所有离线依赖的 `.whl` 包到 `wheels/` 目录 |
| [`requirements/chatbi-nl2sql-offline.txt`](requirements/chatbi-nl2sql-offline.txt) | 离线依赖清单，包含运行 NL2SQL 服务所需的全部 Python 包及版本约束 |
| [`Dockerfile.chatbi-nl2sql`](Dockerfile.chatbi-nl2sql) | Docker 镜像构建文件，基于 `python:3.11-slim-bookworm`，使用本地 wheels 离线安装依赖 |
| [`docker-compose.yml`](docker-compose.yml) | 容器编排文件，定义服务启动参数、端口映射、网络、卷挂载等 |
| [`src/vanna/examples/chatbi_nl2sql_api.py`](src/vanna/examples/chatbi_nl2sql_api.py) | 主服务代码，FastAPI 实现的 NL2SQL HTTP 服务 |

---

## 步骤一：下载离线 wheels（外网执行）

在项目根目录下执行 PowerShell 脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download-chatbi-wheels.ps1
```

脚本会自动读取 [`requirements/chatbi-nl2sql-offline.txt`](requirements/chatbi-nl2sql-offline.txt)，并将所有依赖的 `.whl` 包下载到 `wheels/` 目录。

> **注意：** 此步骤需要连接外网（PyPI）。若 wheels/ 目录已存在旧包，建议先删除后再执行，确保版本一致。

---

## 步骤二：构建 Docker 基础镜像（外网执行）

```bash
docker build -f Dockerfile.chatbi-nl2sql -t chatbi-nl2sql:offline-v1 .
```

构建逻辑说明：
- 基础镜像：`python:3.11-slim-bookworm`
- 将 `requirements/chatbi-nl2sql-offline.txt` 和 `wheels/` 复制进镜像
- 使用 `pip install --no-index --find-links=/wheels` **完全离线**安装依赖
- 复制 `src/` 源码到 `/workspace/src/`
- 暴露端口 `8001`，默认启动 `uvicorn` 服务

---

## 步骤三：启动容器并预下载 ONNX 模型（外网执行）

启动容器：

```bash
docker-compose up -d
```

此时容器会运行，但 **Chroma 向量库依赖的 ONNX 模型（`all-MiniLM-L6-v2`）尚未下载**。需要手动触发一次 `generate_sql` 接口，让服务联网下载该模型到容器内的缓存目录：

```bash
# 调用 generate_sql 接口（需要确保环境变量/数据库配置已就绪）
curl -X POST http://localhost:8001/generate_sql \
  -H "Content-Type: application/json" \
  -d '{
    "database_id": 1,
    "question": "查询今天新增订单数",
    "auto_sync_training": false
  }'
```

模型下载路径（容器内，取决于 `XDG_CACHE_HOME`）：

```
/opt/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz
```

> **说明：** 第一次调用 `generate_sql` 时，Chroma 会自动从 HuggingFace 下载该 ONNX embedding 模型。由于内网无法联网，必须在外网环境提前完成此步骤，并将模型一并固化到镜像中。  
> ⚠️ **注意：** 本镜像在 `Dockerfile` 中将运行用户设为了 `appuser`，且 `XDG_CACHE_HOME=/opt/.cache`，因此模型**不会**落在 `/root/.cache/...` 下。如果此处路径写错，内网启动时 Chroma 会重新尝试下载，导致报错。

---

## 步骤四：测试验证

确认服务正常：

```bash
# 查看容器状态
docker ps

# 查看服务日志
docker logs -f chatbi-nl2sql

# 再次调用接口，确认响应正常且无需再次联网下载模型
curl -X POST http://localhost:8001/generate_sql \
  -H "Content-Type: application/json" \
  -d '{
    "database_id": 1,
    "question": "查询最近7天销售额",
    "auto_sync_training": false
  }'
```

确保接口返回预期 SQL 结果，且日志中不再出现模型下载行为，即可进入下一步。

---

## 步骤五：将容器 commit 为固化镜像（外网执行）

当前运行的容器已经包含了预下载好的 ONNX 模型，将其 commit 为一个新镜像，方便保存和迁移：

```bash
# 停止容器（确保状态干净）
docker stop chatbi-nl2sql

# 将当前容器 commit 为新镜像
docker commit chatbi-nl2sql chatbi-nl2sql:offline-v1-final

# 给镜像打标签（可选，便于版本管理）
docker tag chatbi-nl2sql:offline-v1-final chatbi-nl2sql:offline-v1-$(date +%Y%m%d)
```

---

## 步骤六：导出 tar 包（外网执行）

将固化后的镜像导出为 tar 文件，便于拷贝到内网：

```bash
# 保存镜像
docker save -o chatbi-nl2sql-offline-v1.tar chatbi-nl2sql:offline-v1-final

# 压缩（可选，可显著减小体积）
gzip chatbi-nl2sql-offline-v1.tar
# 得到 chatbi-nl2sql-offline-v1.tar.gz
```

---

## 步骤七：内网导入并启动

将 `chatbi-nl2sql-offline-v1.tar`（或 `.tar.gz`）拷贝到内网服务器后执行：

```bash
# 加载镜像
docker load -i chatbi-nl2sql-offline-v1.tar

# 确认镜像已导入
docker images | grep chatbi-nl2sql

# 启动服务（确保 docker-compose.yml 中 image 名称与导入的一致）
docker-compose up -d
```

> **注意：** 内网启动前，请根据实际情况修改 `docker-compose.yml` 中的环境变量、网络配置和卷挂载路径。

---

## 目录结构参考

```
vanna-kbm/
├── scripts/
│   └── download-chatbi-wheels.ps1    # 下载 wheels 脚本
├── requirements/
│   └── chatbi-nl2sql-offline.txt     # 离线依赖清单
├── wheels/                           # 下载后的 .whl 包（gitignore 已排除）
├── src/
│   └── vanna/
│       └── examples/
│           └── chatbi_nl2sql_api.py  # 主服务入口
├── Dockerfile.chatbi-nl2sql          # 镜像构建文件
├── docker-compose.yml                # 容器编排
├── README.md                         # ← 本文件
└── README_OFFICIAL.md                # Vanna 官方原版文档
```

---

## 常见问题

### Q1: `docker-compose up` 报错 `network chatbi-net not found`

`docker-compose.yml` 中使用了外部网络 `chatbi-net`，需提前创建：

```bash
docker network create chatbi-net
```

### Q2: 如何确认 ONNX 模型已经预下载成功？

**第一步：在外网运行的容器里搜索模型真实位置**

由于容器运行用户是 `appuser`，且 `XDG_CACHE_HOME=/opt/.cache`，请使用如下命令全局搜索，不要猜路径：

```bash
# 进入容器
docker exec -it chatbi-nl2sql bash

# 用 find 全局搜索 onnx.tar.gz（可能需要 root 权限，先用 sudo 或 docker exec -u 0）
sudo docker exec -u 0 chatbi-nl2sql find / -name "onnx.tar.gz" 2>/dev/null

# 预期输出示例：
# /opt/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz
```

**第二步：确认文件大小正常**

```bash
docker exec -u 0 chatbi-nl2sql ls -lh /opt/.cache/chroma/onnx_models/all-MiniLM-L6-v2/
# 应看到 onnx.tar.gz，大小约 20MB+
```

**第三步：commit 后确认镜像里确实包含该文件**

```bash
# 用新镜像启动一个临时容器，检查模型是否存在
docker run --rm chatbi-nl2sql:offline-v1-final find /opt/.cache -name "onnx.tar.gz" 2>/dev/null
```

若临时容器里搜不到，说明 **commit 前模型并未真正落在容器文件系统里**（可能被下载到了卷挂载的宿主机目录，或路径不在容器层中）。

---

### Q3: 外网 generate_sql 正常，内网启动后却报错（模型相关）？

这是最容易踩坑的地方，请按以下顺序排查：

#### 排查 1：内网 `docker-compose.yml` 中的 `image` 名称是否已更新？

`docker-compose.yml` 默认写的是：
```yaml
image: chatbi-nl2sql:offline-v1
```

而 commit 后的镜像名是 `chatbi-nl2sql:offline-v1-final`。如果你在内网没有修改 `image` 字段，Docker 会启动**旧的基础镜像**（里面没有预下载的 ONNX 模型），自然报错。

**解决办法**：内网导入镜像后，编辑 `docker-compose.yml`，将 `image` 改为导入的镜像名：
```yaml
image: chatbi-nl2sql:offline-v1-final
```

#### 排查 2：确认内网容器里模型文件存在且路径正确

```bash
# 进入内网运行的容器
docker exec -it chatbi-nl2sql bash

# 查看当前用户和环境变量
whoami        # 应为 appuser
echo $XDG_CACHE_HOME   # 应为 /opt/.cache

# 确认模型文件
ls -lh /opt/.cache/chroma/onnx_models/all-MiniLM-L6-v2/
```

若 `ls` 报 "No such file or directory"，说明模型未被打包进镜像。

#### 排查 3：确认 `appuser` 对缓存目录有读取权限

```bash
docker exec -u 0 chatbi-nl2sql ls -la /opt/.cache/
# 应看到 chroma 目录，且属主为 appuser:appuser
```

若属主是 `root`，`appuser` 可能无法读取，导致 Chroma 认为模型不存在而尝试重新下载。

#### 排查 4：查看内网容器日志，定位具体报错

```bash
docker logs -f chatbi-nl2sql
```

常见报错关键词：
- `ConnectionError`、`HTTPSConnectionPool` → 内网无法联网，Chroma 在尝试下载模型但失败了
- `FileNotFoundError`、`onnx.tar.gz` → 模型文件缺失或路径不对
- `Permission denied` → 用户权限问题

---

### Q4: 如果模型路径还是不对，如何手动修复？

假设你已经在外网容器中确认模型下载到了 `/opt/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz`，但内网仍然报错，可以**在 commit 前手动确保模型位于正确位置**：

```bash
# 外网执行：将模型复制到确保能被 appuser 访问的路径
docker exec -u 0 chatbi-nl2sql mkdir -p /opt/.cache/chroma/onnx_models/all-MiniLM-L6-v2
docker exec -u 0 chatbi-nl2sql cp / somewhere/you/found/onnx.tar.gz /opt/.cache/chroma/onnx_models/all-MiniLM-L6-v2/
docker exec -u 0 chatbi-nl2sql chown -R appuser:appuser /opt/.cache/chroma

# 然后再 commit
docker stop chatbi-nl2sql
docker commit chatbi-nl2sql chatbi-nl2sql:offline-v1-final
```

---

### Q5: wheels 下载脚本执行失败？

- 确认 Python 版本为 3.11（与镜像一致）
- 确认 PowerShell 执行策略已绕过（脚本已自动设置 `-ExecutionPolicy Bypass`）
- 若某些包无 manylinux2014_x86_64 二进制版本，可能需要手动下载对应平台的 wheel

---

## 相关脚本速查

| 场景 | 命令 |
|------|------|
| 下载 wheels | `powershell -ExecutionPolicy Bypass -File .\scripts\download-chatbi-wheels.ps1` |
| 构建镜像 | `docker build -f Dockerfile.chatbi-nl2sql -t chatbi-nl2sql:offline-v1 .` |
| 启动容器 | `docker-compose up -d` |
| 查看日志 | `docker logs -f chatbi-nl2sql` |
| commit 镜像 | `docker commit chatbi-nl2sql chatbi-nl2sql:offline-v1-final` |
| 导出镜像 | `docker save -o chatbi-nl2sql-offline-v1.tar chatbi-nl2sql:offline-v1-final` |
| 导入镜像 | `docker load -i chatbi-nl2sql-offline-v1.tar` |

---

> 本文档由项目维护团队编写，如有更新请同步修改。官方 Vanna 文档请查阅 [README_OFFICIAL.md](README_OFFICIAL.md)。
