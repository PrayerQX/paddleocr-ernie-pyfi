# 两轮实验成绩与问题总结

## 1. 实验概览

本文总结了同一份 `pyfi301` 样本集上的两轮实验结果。两轮都使用：

- 数据集：`data/pyfi-600k/pyfi301_manifest.json`
- 样本数：`301`
- 领域：`finance`
- 并发：`2 workers`
- Web Search：关闭
- 最大输出：`8000 max_completion_tokens`

两轮的差异主要在 **PaddleOCR 的参数配置**，而 **大模型都使用同一个 ERNIE 模型**。

## 2. 两轮配置

### 第一轮

- 结果目录：`output-pyfi301-ernie45-text-full-20260422`
- PaddleOCR API：
  - `https://i0u1edb895ael4d6.aistudio-app.com/layout-parsing`
- OCR 预设：
  - `ocr_preset = auto`
  - 实际落盘为 `medium`
- 关键 OCR 参数：
  - `useLayoutDetection = true`
  - `useChartRecognition = true`
  - `useSealRecognition = false`
  - `useOcrForImageBlock = true`
  - `mergeTables = false`
  - `relevelTitles = false`
  - `promptLabel = "chart"`
  - `restructurePages = false`
- ERNIE 模型：
  - `ernie-4.5-21b-a3b`
- ERNIE 输入方式：
  - 文本模型，不支持图片输入
  - 实际表现为：`requested_image_paths` 有值，但 `image_paths` 为空

### 第二轮

- 结果目录：`output-pyfi301-ernie45-baidu-sample-smoke-20260423`
- PaddleOCR API：
  - `https://i0u1edb895ael4d6.aistudio-app.com/layout-parsing`
- OCR 预设：
  - `ocr_preset = baidu_sample`
- 关键 OCR 参数：
  - `useDocOrientationClassify = false`
  - `useDocUnwarping = false`
  - `useLayoutDetection = true`
  - `useChartRecognition = true`
  - `useSealRecognition = true`
  - `useOcrForImageBlock = true`
  - `mergeTables = true`
  - `relevelTitles = true`
  - `promptLabel = "ocr"`
  - `restructurePages = true`
- ERNIE 模型：
  - `ernie-4.5-21b-a3b`
- ERNIE 输入方式：
  - 同样是文本模型，不支持图片输入
  - 实际仍然是：`requested_image_paths` 有值，但 `image_paths` 为空

## 3. 两轮成绩

### 第一轮成绩

- 完成：`301/301`
- 失败：`0`
- 可评分样本：`295`
- 正确数：`163`
- 准确率：`0.552542`
- 缺失预测：`6`

分题型成绩：

- `Decision_support`: `0.629630`
- `Pattern_recognition`: `0.611111`
- `Logical_reasoning`: `0.578947`
- `Data_extraction`: `0.545455`
- `Calculation_analysis`: `0.477273`
- `Perception`: `0.472222`

耗时：

- 平均总耗时：`98.368s`
- 平均 PaddleOCR：`1.821s`
- 平均 ERNIE：`95.061s`

### 第二轮成绩

- 完成：`301/301`
- 失败：`0`
- 可评分样本：`300`
- 正确数：`157`
- 准确率：`0.523333`
- 缺失预测：`1`

分题型成绩：

- `Decision_support`: `0.629630`
- `Logical_reasoning`: `0.586207`
- `Pattern_recognition`: `0.600000`
- `Perception`: `0.500000`
- `Data_extraction`: `0.481013`
- `Calculation_analysis`: `0.377778`

耗时：

- 平均总耗时：`107.116s`
- 平均 PaddleOCR：`21.946s`
- 平均 ERNIE：`83.630s`

## 4. 两轮对比结论

### 总体对比

- 准确率：`0.552542 -> 0.523333`
  - 下降 `0.029209`
- 正确数：`163 -> 157`
  - 少 `6` 题
- 缺失预测：`6 -> 1`
  - 第二轮覆盖率更高
- 平均总耗时：`98.368s -> 107.116s`
  - 第二轮更慢
- 平均 PaddleOCR 耗时：`1.821s -> 21.946s`
  - 第二轮 OCR 明显更重
- 平均 ERNIE 耗时：`95.061s -> 83.630s`
  - 第二轮 ERNIE 更快

### 分题型变化

- 变好：
  - `Logical_reasoning`: `+0.007260`
  - `Perception`: `+0.027778`
- 持平：
  - `Decision_support`: `0`
- 变差：
  - `Calculation_analysis`: `-0.099495`
  - `Data_extraction`: `-0.064442`
  - `Pattern_recognition`: `-0.011111`

### 解释

第二轮更重的 OCR 配置带来了两个效果：

1. 覆盖率变高  
第二轮缺失预测从 `6` 降到 `1`，说明它更容易产出结构化答案。

2. 精确读数和数值稳定性变差  
虽然答案更完整，但在 `Calculation_analysis` 和 `Data_extraction` 上明显退步，说明更重的 OCR 并没有稳定提升数值类和定位类题目的证据质量。

## 5. 主要问题

### 5.1 主要问题一：当前 ERNIE 4.5 只吃文本，不看原图

两轮都使用 `ernie-4.5-21b-a3b`。该模型在当前项目里表现为：

- `image_input_supported = false`
- 原图路径虽然会被记录，但不会真正发给模型

这意味着整个问答流程本质上是：

`原图 -> PaddleOCR -> 文本/表格 -> ERNIE`

而不是：

`原图 + OCR 文本 -> ERNIE`

因此，颜色、图例、空间位置、多子图关系这类强视觉信息很容易在 OCR 转文本时损失掉。

### 5.2 主要问题二：PaddleOCR 转文本后会丢失视觉信息

从错题看，最容易出问题的是：

- 多子图 / 多面板图
- 柱状图
- 泛化图表
- 地图类
- 颜色依赖强的图

这类图在 OCR 后常见问题包括：

- 颜色信息丢失
- 图例和数据系列的绑定关系丢失
- 多个子图被压成一个文本块或表格
- 精确位置关系无法恢复

### 5.3 主要问题三：即使 OCR 给出证据，ERNIE 仍然会判断错

并不是所有错题都来自 OCR。还有一类常见问题是：

- `doc_0.md` 里已经有正确值
- `analysis_finance.json` 的 `extracted_metrics` 也提到了正确值
- 但最终 `choice` 或 `answer` 仍然选错

也就是说，当前链路的错误不是“只有 OCR 的问题”，也包括：

- ERNIE 比较错误
- ERNIE 结论错误
- ERNIE 在多个候选值之间选错

## 6. 错误归因判断

根据错题的启发式分析，整体上更偏向：

- `OCR / 视觉证据不足` 是第一类问题
- `ERNIE 拿到证据但判断错` 是第二类问题

可以概括为：

- 大约 `55%` 的错题更像 OCR 或视觉信息丢失问题
- 大约 `45%` 的错题更像 ERNIE 推理或输出问题

因此，更准确的说法是：

> 目前这条链路的主要短板仍然是“视觉信息被压成文本后损失过大”，但 ERNIE 在已有证据上的比较、抽取和结论能力也并不稳定。

## 7. 最终结论

### 第一轮更适合作为当前基线

原因：

- 总准确率更高
- `Calculation_analysis` 和 `Data_extraction` 更稳
- 虽然缺失预测更多，但总体正确数更多

### 第二轮不是全面变差，而是 trade-off 更明显

第二轮的特点是：

- OCR 更重
- 答案覆盖率更高
- 某些 `Logical_reasoning`、`Perception` 样本更好
- 但数值类和读数类题更差

### 当前最核心的问题不是单一组件，而是链路结构

当前链路的问题顺序大致是：

1. `ERNIE 4.5` 不能直接看原图  
2. `PaddleOCR` 把图像压成文本时损失了关键信息  
3. `ERNIE` 在已有证据基础上仍然会发生判断错误

所以如果后面继续优化，优先级建议是：

1. 先提升“原图信息保留能力”
2. 再优化 OCR 参数
3. 最后再优化 ERNIE 的提示词和输出约束

## 8. 相关文件

- 第一轮评分：[score_report.json](/D:/OneDrive/Desktop/PaddleOCR-VL/output-pyfi301-ernie45-text-full-20260422/score_report.json)
- 第二轮评分：[score_report.json](/D:/OneDrive/Desktop/PaddleOCR-VL/output-pyfi301-ernie45-baidu-sample-smoke-20260423/score_report.json)
- 两轮对比：[comparison_vs_ernie45_text_full.json](/D:/OneDrive/Desktop/PaddleOCR-VL/output-pyfi301-ernie45-baidu-sample-smoke-20260423/comparison_vs_ernie45_text_full.json)
- 详细错题分析：[error_analysis_summary.md](/D:/OneDrive/Desktop/PaddleOCR-VL/output-pyfi301-ernie45-baidu-sample-smoke-20260423/error_analysis_summary.md)
