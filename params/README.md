# 参数文件

公开仓库只发**契约形状**，不发具体取值。

| 文件 | 用途 |
|---|---|
| `schema.example.json` | `headshell/params@0` 的字段全集，所有取值留空（`null`） |

具体参数集（产品默认值、各角色的 R&D 参数）在本机私有侧维护，不随本仓库发布。
校验器会拒收任何缺项：`Params.validate()` 对未配置的毫米级参数一律报
`... must be configured (no default is shipped)`。
