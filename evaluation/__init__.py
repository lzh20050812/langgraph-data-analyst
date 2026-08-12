"""
Evaluation Module —— 可复现实验与评估模块。

本模块包含两部分：
1. 独立评估脚本（顶层 .py 文件）—— Phase 1-3 已完成的 Agent 评估
   - eval_schema.py      Schema Agent 检索准确率评估
   - eval_sql.py         SQL Agent 生成成功率评估
   - eval_prediction.py  XGBoost + Prophet 模型评估
   - eval_report.py      Report Agent 报告质量评估
   - eval_summary.py     评估层总汇总

2. 实验框架（framework/ 子包）—— 统一基准实验基础设施
   - framework/base.py   实验基类 ExperimentBase + ExperimentResult
   - framework/logger.py 实验日志记录器 ExperimentLogger
   - framework/result.py 结果管理器 ResultManager + save/load 工具
   - framework/config.py 实验配置类 ExperimentConfig
   - metrics/            评价指标注册表与计算函数
   - data/               测试数据集存放目录
   - results/            本地实验结果输出（raw/ 原始 + tables/ 指标表格）
   - logs/               实验运行日志
   - config/             实验配置文件

使用方式：
    # 方式1：运行已有评估脚本（不受影响）
    python -m evaluation.eval_sql

    # 方式2：基于实验框架开发新实验
    from evaluation.framework import ExperimentBase, ExperimentLogger, ResultManager
"""
