# Claude 协作指南

## 角色

Claude 在本项目中适合承担：

- 产品需求梳理；
- Web/UI 交互设计；
- 跨模块方案讨论；
- 文档初稿；
- 大范围代码阅读和原型设计。

但历史对话显示一个重要风险：

> Claude 曾多次基于旧会话或生成 ZIP 的记忆，描述“已有代码”，而当前仓库实际上并未合入这些文件。

因此 Claude 的第一职责是 **以当前仓库为准重新核验**。

---

## 编码前

必须先读：

```text
docs/project/context.md
docs/project/current_state.md
docs/project/decisions.md
docs/project/architecture.md
```

再检查：

```text
project tree
git status
target files
```

不得只依赖历史对话记忆。

---

## 当前优先级

```text
1. Real EC/METAR data
2. Web refinement
3. Simple prediction
4. Formal AI
```

不要主动跳到模型训练。

---

## 实现风格

### Prefer

- 最小改动；
- 保留现有 TypeScript Web；
- 通过 Adapter/Service 接真实数据；
- 提供清晰文件级修改清单；
- 每次交付说明新增/修改/删除文件；
- 对 UI 改动给出业务理由；
- 真实数据驱动后再调整视觉细节。

### Avoid

- 凭历史记忆假设文件存在；
- 一次性生成大量未验证模块；
- 创建 ZIP 后就把能力写成“已完成”；
- 未经确认更改目录结构；
- 把候选算法写死；
- 让前端适配后端临时数据结构。

---

## METAR-specific Rules

如果处理 METAR：

- 原始 TAC 必须保留；
- Parser 未识别 token 不丢报文；
- SQLite 持久化；
- 支持 latest/history/replay；
- 站点配置化；
- 真实 JSON schema 先检查后编码；
- Parser 必须用真实样例测试。

---

## EC-specific Rules

如果处理 EC：

- 先核验真实文件；
- 明确 init/valid/forecast hour；
- 保留 lat/lon；
- 数据变量和特殊值从配置获取；
- 不默认生成彩图；
- 数值场通过 API 给 Web 动态渲染；
- extractor 与 API service 分层。

---

## Web-specific Rules

不要重写现有 Web。

优先：

```text
mockData
   ↓
API Service Adapter
   ↓
real data
```

修改前指出：

- 哪个组件目前从哪里拿数据；
- 新 API 返回什么；
- 哪些 UI 不变；
- fallback 怎么处理。

---

## Deliverable Rule

Claude 如果生成代码：

1. 输出文件树变化；
2. 标明哪些文件真实写入；
3. 运行测试；
4. 不将未合入仓库的 ZIP 当成当前状态；
5. 建议同步更新 `current_state.md`。
