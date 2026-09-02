# MR Memory 运行时文件清单

本清单定义插件热部署的源内边界。它只列相对路径和规则；部署校验只比较本次实际修改的
运行时文件，不能提交真实配置、数据库或服务器快照。

## 必需文件

| 路径规则 | 用途 |
| --- | --- |
| `__init__.py` | Python 插件包入口 |
| `main.py` | AstrBot 事件、路由、生命周期与热重载适配 |
| `metadata.yaml` | AstrBot 插件元数据；框架要求的 `version` 字段固定为 `unversioned` |
| `_conf_schema.json` | 插件配置面 |
| `requirements.txt` | 基础运行依赖 |
| `mr_memory/*.py` | 核心数据、在线本地证据服务、后台分层研究运行时与 Web API |
| `pages/console/*` | 已认证管理控制台静态资源 |
| `.astrbot-plugin/i18n/*.json` | AstrBot 插件页面本地化资源 |

`mr_memory/*.py` 的分层运行时模块包括：

- `mr_memory/local_serving.py`：回答前的 source-backed 本地证据 envelope；普通聊天最多
  3000 字符，显式历史回忆最多 12000 字符；
- `mr_memory/snapshot.py`：L0 `RequestSnapshot` 与 revision vector；
- `mr_memory/routing.py`：宿主持有的 L0–L3 路由策略；
- `mr_memory/reader.py`：L2 Evidence Reader 协议；
- `mr_memory/evidence_pack.py`：按来源去重、分层保留并严格限额的 Reader 证据包；
- `mr_memory/orchestrator.py`：生产/实验共用的有界 L3 ECCR；
- `mr_memory/certificate.py`：`EvidenceCertificateV2`；
- `mr_memory/surface.py`：表层编译与回答验证；
- `mr_memory/singleflight.py`：请求内并发合并；
- `mr_memory/evidence_closure.py`、`mr_memory/storage.py`、`mr_memory/service.py`：
  ECCR 契约、schema 17 持久化和异步服务边界。

## 条件文件

| 路径规则 | 条件 |
| --- | --- |
| `requirements-harrier.txt` | 仅当 `embedding_backend=sentence_transformers` 并使用 Harrier 时安装 |

## 不属于运行时部署

- `scripts/`、`tests/`、`dev/` 与 `docs/`；
- `.dev/`、`.test-artifacts/`、pytest 临时目录和任何本地模型缓存；
- `*.db`、`*.db-wal`、`*.db-shm`、日志、Provider 配置、Token、Cookie、私钥和远程配置快照；
- Git 元数据及研究报告中的真实实验产物。

部署只复制本次实际修改且已验证的运行时文件，并确认其现有依赖文件仍存在；校验也只覆盖
这些文件。旧 schema→17 由新代码首次打开各群数据库时执行。只热重载 MR Memory，
不应因此重启 AstrBot 或消息适配器。
