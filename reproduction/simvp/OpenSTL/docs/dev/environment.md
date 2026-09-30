# 环境配置（Dev）

## 依赖来源
- `environment.yml`（conda）或 `requirements.txt`（pip）。
- Python <= 3.10.8；核心依赖：torch、timm、lightning、fvcore、xarray==0.19.0 等。

## 安装步骤（上游标准）
```shell
conda env create -f environment.yml
conda activate OpenSTL
python setup.py develop
```

## 当前状态
- reproduction 环境未就绪（`Environment = TODO`）。
- 待：`Environment Working` 状态迁移并留档 command / artifact。
