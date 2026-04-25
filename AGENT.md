# AGENT.md

## 项目目标

根据 `https://github.com/AgenticFinLab/PyFi` 搭建一个可在本地运行、但 OCR 和大模型能力均通过远程 API 调用完成的 PaddleOCR-VL + ERNIE 4.5 文档理解项目。

本项目的第一阶段目标是服务 PyFi 风格金融图像理解；第二阶段目标是在项目跑通后，提取一套可复现、可迁移的通用文档智能架构，使同一方法可以扩展到研报、年报、招股书、合同、票据、论文、政务材料、技术文档、医学材料等其他文档场景。

本项目不复刻 PyFi 的训练全流程，而是借鉴 PyFi 的金融图像理解范式，并将其抽象为通用的分层文档理解范式：

- Perception：基础视觉/版面信息识别
- Data Extraction：表格、文字、指标、时间、数值抽取
- Calculation Analysis：财务指标、同比/环比、占比、差值等计算分析
- Pattern Recognition：趋势、异常、结构性变化识别
- Logical Reasoning：结合上下文做多步逻辑推理
- Decision Support / Task Support：在明确风险边界内给出研究型、审阅型或业务辅助型结论

实现路线必须是：

1. 使用 PaddleOCR-VL / PaddleOCR 远程版 API 对 PDF 或图片做 layout parsing。
2. 将 PaddleOCR 输出的 Markdown、表格、图片链接、版面图和结构化结果保存到本地项目输出目录。
3. 使用 ERNIE 4.5 / 文心一言 API 对 OCR 结果进行文档理解、问答生成、分析报告生成或 PyFi 风格多层问题链构造。
4. 在金融文档场景中验证流程有效性，再把领域特定 prompt、schema、评测样例和处理逻辑抽象为可配置模块。
5. 所有运行、依赖和脚本都通过 `uv` 管理在项目本地环境中完成。

## 可复现通用架构目标

项目成功后，必须沉淀出一套独立于金融场景的通用架构。该架构应至少包含以下可复用模块：

- Document Ingestion：接收 PDF、图片或目录输入，判断文件类型，生成稳定任务 ID。
- OCR Layout Parsing：通过 PaddleOCR-VL / PaddleOCR 远程 API 提取 Markdown、版面结构、表格、图片和原始 JSON。
- Evidence Store：保存 OCR 原文、图片、表格、页码、区域信息、下载资源和中间结果，支持后续引用溯源。
- Domain Adapter：按领域加载 prompt、字段 schema、术语表、输出格式和评测规则。
- Reasoning Agent：调用 ERNIE 4.5 / 文心一言，对 OCR 证据进行分层理解、问答、摘要、抽取、计算、核验和报告生成。
- Validation Layer：检查数值一致性、引用完整性、缺失信息、不确定性、路径安全和 API 错误。
- Evaluation Harness：支持 mock 测试、样例集回归、人工标注对比和可复现实验记录。
- Export Layer：导出 Markdown、JSON、HTML 或后续系统可消费的结构化结果。

架构设计必须把“通用能力”和“领域规则”分离：

- 通用能力包括文件读取、远程 OCR、结果保存、证据索引、LLM 调用、日志、错误处理、测试和导出。
- 领域规则包括金融指标、合同条款、论文章节、票据字段、医学术语、政务材料字段等。
- 新增领域时，应优先新增配置、prompt、schema 和样例，不应复制粘贴核心流水线。

首个领域适配器命名为 `finance`，用于复现 PyFi 风格金融图像理解；后续应可以新增 `contract`、`invoice`、`research_paper`、`government_doc` 等适配器。

## 强制环境约束

Claude Code 操作本项目时必须遵守以下环境规则：

- 必须使用 `uv` 管理 Python 环境和依赖。
- 必须使用项目目录内的 `.venv`，不能污染全局 Python 环境。
- 不得执行全局 `pip install ...`。
- 不得依赖系统级 site-packages。
- 不得要求用户手动激活全局 conda、全局 venv 或全局 Python 环境。
- 允许的依赖管理命令示例：

```powershell
uv venv --python 3.12 .venv
uv add requests openai python-dotenv
uv run python -m paddle_pyfi --help
```

如果当前机器没有安装 `uv`，应优先给出本地安装方案，并说明安装后仍必须使用项目内 `.venv`。安装 `uv` 本身之外，不得把业务依赖安装到全局环境。

## 学术合规与提分边界

所有提升效果、提升分数或改进 benchmark 的工作必须限定在学术共同体认可的范围内。不得因为要提分而使用任何学术不允许的方法。

允许的方法包括：

- 改进 OCR 后处理、表格解析、单位规范化、数值校验和引用溯源。
- 基于公开训练集、开发集或自建标注集做 prompt、流程和参数调优。
- 使用验证集做误差分析，并记录改动、指标和实验配置。
- 做消融实验，例如是否启用文档方向分类、图表识别、联网搜索、上下文补充。
- 明确标注模型、数据、API、时间、版本和实验设置。
- 对生成答案做可解释性增强，例如给出引用片段、计算过程和不确定性说明。

禁止的方法包括：

- 使用测试集答案、隐藏 benchmark 标签或人工答案注入来提高分数。
- 在评测阶段泄露标准答案、题目解析、打分规则或测试标签。
- 对测试集反复调 prompt 直到过拟合。
- 把评测集样本混入训练集、few-shot 样例或检索库。
- 伪造实验结果、选择性报告结果或隐瞒失败样本。
- 声称模型能力超过实际 API 输出所支持的范围。
- 在没有证据的情况下给出确定性金融投资建议。

涉及金融分析时，输出必须保持研究辅助定位，不得把模型结果包装成确定投资建议。

## PaddleOCR 远程 API 要求

项目全程必须使用 PaddleOCR 远程 API，不得改成本地 PaddleOCR 推理，不得下载或加载本地 OCR 模型。

API 地址：

```text
https://i0u1edb895ael4d6.aistudio-app.com/layout-parsing
```

认证 token 必须从环境变量读取，不得硬编码到源码、文档或提交历史中。

推荐环境变量：

```powershell
$env:PADDLEOCR_API_URL="https://i0u1edb895ael4d6.aistudio-app.com/layout-parsing"
$env:PADDLEOCR_API_TOKEN="<your-token>"
```

请求逻辑必须等价于以下接口形式，但 token 必须来自环境变量：

```python
import base64
import requests

api_url = os.environ["PADDLEOCR_API_URL"]
token = os.environ["PADDLEOCR_API_TOKEN"]

with open(file_path, "rb") as file:
    file_data = base64.b64encode(file.read()).decode("ascii")

headers = {
    "Authorization": f"token {token}",
    "Content-Type": "application/json",
}

payload = {
    "file": file_data,
    "fileType": file_type,  # PDF 为 0，图片为 1
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useChartRecognition": False,
}

response = requests.post(api_url, json=payload, headers=headers, timeout=300)
response.raise_for_status()
result = response.json()["result"]
```

文件类型规则：

- PDF 文档：`fileType = 0`
- 图片文件：`fileType = 1`
- 支持的图片扩展名至少包含 `.png`、`.jpg`、`.jpeg`、`.bmp`、`.tif`、`.tiff`、`.webp`
- 对未知扩展名必须显式报错，不得猜测上传

输出保存规则：

- 每个输入文件必须生成独立输出目录。
- 保存 PaddleOCR 返回的 Markdown，例如 `doc_0.md`。
- 下载并保存 `markdown.images` 中引用的图片。
- 下载并保存 `outputImages` 中的版面图、表格图或其他结果图。
- 保存原始 JSON 响应，便于复现和排错。
- 下载远程图片时必须检查 HTTP 状态码。
- 本地保存路径必须防止路径穿越，例如不得让 API 返回的 `../../x` 写出输出目录。

## ERNIE 4.5 / 文心一言 API 要求

项目使用 ERNIE 4.5 / 文心一言 API 作为文本推理、文档分析、领域适配和 PyFi 风格问题链生成模型。

API SDK 使用 OpenAI 兼容客户端：

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ERNIE_API_KEY"],
    base_url=os.getenv("ERNIE_BASE_URL", "https://aistudio.baidu.com/llm/lmapi/v3"),
)
```

默认目标模型应使用服务端可用的 ERNIE 4.5 系列模型。模型名必须通过环境变量配置，避免把某个临时模型名写死到业务代码中。

推荐默认模型：

```text
ernie-4.5-turbo-vl
```

推荐环境变量：

```powershell
$env:ERNIE_API_KEY="<your-token>"
$env:ERNIE_BASE_URL="https://aistudio.baidu.com/llm/lmapi/v3"
$env:ERNIE_MODEL="ernie-4.5-turbo-vl"
```

请求逻辑必须等价于以下形式：

```python
chat_completion = client.chat.completions.create(
    model=os.getenv("ERNIE_MODEL", "ernie-4.5-turbo-vl"),
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    stream=True,
    extra_body={
        "web_search": {
            "enable": True,
        }
    },
    max_completion_tokens=65536,
)

for chunk in chat_completion:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if getattr(delta, "reasoning_content", None):
        print(delta.reasoning_content, end="", flush=True)
    elif getattr(delta, "content", None):
        print(delta.content, end="", flush=True)
```

ERNIE 4.5 / 文心一言只能基于 PaddleOCR 输出、用户提供上下文、公开可检索信息和明确的计算过程生成结论。对于 OCR 不确定、表格缺失、图片链接下载失败或数值无法核验的部分，必须在输出中声明不确定性。

如果 API 账号当前不可用 `ernie-4.5-turbo-vl`，允许通过 `ERNIE_MODEL` 切换到同一服务端支持的 ERNIE 4.5 系列模型；切换原因、模型名和运行时间必须记录在实验或运行日志中。

## Agent 工作流

Claude Code 在实现本项目时应按以下顺序执行：

1. 初始化 `uv` 项目结构，创建本地 `.venv`。
2. 建立清晰的源码目录，例如 `src/paddle_pyfi/`。
3. 封装 PaddleOCR 远程 layout parsing client。
4. 封装 ERNIE 4.5 / 文心一言 OpenAI-compatible client。
5. 实现 CLI：
   - `parse`：只做 PaddleOCR 远程解析并保存结果。
   - `analyze`：先 OCR，再调用 ERNIE 4.5 / 文心一言做文档理解分析。
   - `ask`：基于已有 Markdown 或 OCR 结果进行问答。
   - `batch`：批量处理目录中的 PDF 和图片。
   - `export-architecture`：导出当前可复现架构说明、模块边界、配置模板和领域适配说明。
6. 为每个命令提供 `--help`。
7. 添加 `.env.example`，只写占位符，不写真实 token。
8. 添加 README，说明安装、配置、运行、输出结构和如何新增领域适配器。
9. 添加架构文档，例如 `docs/architecture.md`，沉淀通用流程、模块职责、数据流、错误处理、复现实验记录方式和迁移指南。
10. 添加最小测试，覆盖文件类型判断、路径安全、payload 构造、领域适配配置加载和 API client 的可 mock 行为。
11. 使用 `uv run` 执行测试和 CLI smoke test。

## 推荐目录结构

```text
.
├── AGENT.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── src/
│   └── paddle_pyfi/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── paddleocr_client.py
│       ├── ernie_client.py
│       ├── pipeline.py
│       ├── prompts.py
│       ├── architecture.py
│       └── domains/
│           ├── finance.yaml
│           ├── contract.yaml
│           ├── invoice.yaml
│           └── research_paper.yaml
├── docs/
│   └── architecture.md
├── tests/
│   ├── test_file_type.py
│   ├── test_paths.py
│   ├── test_payload.py
│   └── test_domain_adapter.py
└── output/
```

`output/` 应加入 `.gitignore`，避免提交用户文档、OCR 结果和模型输出。

## Prompt 与领域适配设计原则

分析 prompt 必须支持 PyFi 六层能力框架，同时不能把金融领域假设写死到通用流水线中。

通用 prompt 应要求模型输出：

- 输入文件摘要
- 文档类型判断
- 可见文本、表格、图像和版面要点
- 关键字段或指标抽取
- 必要计算、核验或交叉检查过程
- 趋势、异常、冲突和模式
- 可验证证据片段
- 不确定性、OCR 风险和缺失信息
- 面向任务的结论，避免超出证据范围的断言

金融领域 prompt 可额外要求：

- 输入文件摘要
- 可见文本和表格要点
- 关键财务指标抽取
- 必要计算过程
- 趋势、异常和模式
- 可验证证据片段
- 不确定性和 OCR 风险
- 研究型结论，避免投资确定性断言

新增领域时，必须通过领域适配器提供：

- 领域名称和适用文档类型
- 输出 schema
- 领域术语表
- 允许的推理任务
- 禁止的高风险结论
- 示例 prompt
- 评测样例和期望输出

不得要求模型“编造缺失数据”。缺失信息必须标注为缺失。

## 质量要求

- 所有 API 错误都要给出明确错误信息。
- 网络请求必须设置 timeout。
- 大文件处理要给出进度或阶段日志。
- 结果文件名要稳定、可复现、可追踪输入文件。
- 对同名输入文件要避免覆盖，或明确使用时间戳/哈希子目录。
- 单元测试不得真实调用远程 PaddleOCR 或文心一言 API，应使用 mock。
- 集成测试如需真实 API，必须通过环境变量显式开启。
- 通用架构文档必须能让其他文档场景复用同一流水线，只替换领域适配器。
- 金融场景中的特殊逻辑不得散落在核心 OCR client、LLM client 或 CLI 基础设施中。
- 每次对架构的改动都应更新 `docs/architecture.md` 或等价文档，避免代码和架构说明脱节。

## 安全要求

- 不得把 token、Access Token、API Key 写入源码、README、AGENT.md、测试快照或日志。
- 不得把用户上传的 PDF、图片、OCR JSON 和模型输出提交到版本库。
- 日志中只允许打印 token 的存在性，不允许打印 token 明文。
- 对 API 返回的图片 URL 下载结果做状态码和内容类型检查。
- 所有路径写入必须限制在项目输出目录内。

## 完成标准

项目完成时至少应满足：

- `uv run python -m paddle_pyfi --help` 可运行。
- `uv run python -m paddle_pyfi parse <file>` 可调用 PaddleOCR 远程 API 并保存 Markdown、图片和 JSON。
- `uv run python -m paddle_pyfi analyze <file> --domain finance` 可在 OCR 后调用 ERNIE 4.5 / 文心一言生成金融图像理解报告。
- `uv run python -m paddle_pyfi analyze <file> --domain contract` 这类命令可以通过新增领域适配器迁移到其他文档场景。
- `uv run python -m paddle_pyfi export-architecture` 可导出或更新可复现架构说明。
- 所有真实 token 均通过环境变量传入。
- 单元测试可通过 `uv run pytest` 执行。
- README 中给出 Windows PowerShell 下的完整运行示例。
- `docs/architecture.md` 清楚描述 PaddleOCR-VL + ERNIE 4.5 的通用方法、数据流、模块边界、领域适配方式和复现实验记录。
