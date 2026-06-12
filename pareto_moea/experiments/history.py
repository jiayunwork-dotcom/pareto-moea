"""实验记录管理"""

import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class ExperimentRecord:
    """实验记录

    记录单次实验的完整信息，包括算法配置、性能指标和结果引用。
    """

    algorithm_name: str
    problem_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    runtime: float = 0.0
    timestamp: float = field(default_factory=time.time)
    result: Any = None
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self, flatten: bool = False) -> Dict[str, Any]:
        """转换为字典

        Args:
            flatten: 是否扁平化 params 和 metrics

        Returns:
            字典形式的记录
        """
        if not flatten:
            return {
                'record_id': self.record_id,
                'algorithm_name': self.algorithm_name,
                'problem_name': self.problem_name,
                'params': self.params.copy(),
                'metrics': self.metrics.copy(),
                'runtime': self.runtime,
                'timestamp': self.timestamp,
                'result': self.result
            }

        data = {
            'record_id': self.record_id,
            'algorithm_name': self.algorithm_name,
            'problem_name': self.problem_name,
            'runtime': self.runtime,
            'timestamp': self.timestamp
        }
        for k, v in self.params.items():
            data[f'param_{k}'] = v
        for k, v in self.metrics.items():
            data[f'metric_{k}'] = v
        return data


class ExperimentHistory:
    """实验历史记录管理器

    管理多个实验记录，支持添加、查询、筛选和导出功能。
    """

    def __init__(self):
        self._records: List[ExperimentRecord] = []

    def add_record(self, record: ExperimentRecord) -> None:
        """添加实验记录

        Args:
            record: 实验记录对象
        """
        self._records.append(record)

    def add(self,
            algorithm_name: str,
            problem_name: str,
            params: Optional[Dict[str, Any]] = None,
            metrics: Optional[Dict[str, float]] = None,
            runtime: float = 0.0,
            result: Any = None,
            timestamp: Optional[float] = None) -> ExperimentRecord:
        """创建并添加一条实验记录

        Args:
            algorithm_name: 算法名称
            problem_name: 问题名称
            params: 参数配置
            metrics: 性能指标
            runtime: 运行时长（秒）
            result: 结果对象引用
            timestamp: 时间戳（默认当前时间）

        Returns:
            创建的实验记录对象
        """
        record = ExperimentRecord(
            algorithm_name=algorithm_name,
            problem_name=problem_name,
            params=params or {},
            metrics=metrics or {},
            runtime=runtime,
            result=result,
            timestamp=timestamp if timestamp is not None else time.time()
        )
        self.add_record(record)
        return record

    @property
    def records(self) -> List[ExperimentRecord]:
        """所有实验记录列表"""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> ExperimentRecord:
        return self._records[index]

    def __iter__(self):
        return iter(self._records)

    def get_by_id(self, record_id: str) -> Optional[ExperimentRecord]:
        """根据记录ID查找

        Args:
            record_id: 记录ID

        Returns:
            找到的记录，未找到返回 None
        """
        for record in self._records:
            if record.record_id == record_id:
                return record
        return None

    def filter(self,
               algorithm_name: Optional[str] = None,
               problem_name: Optional[str] = None,
               param_filters: Optional[Dict[str, Any]] = None,
               metric_filters: Optional[Dict[str, Any]] = None,
               custom_filter: Optional[Callable[[ExperimentRecord], bool]] = None,
               runtime_min: Optional[float] = None,
               runtime_max: Optional[float] = None,
               time_start: Optional[float] = None,
               time_end: Optional[float] = None) -> 'ExperimentHistory':
        """按条件筛选记录

        Args:
            algorithm_name: 算法名称筛选
            problem_name: 问题名称筛选
            param_filters: 参数筛选字典，键为参数名，值为匹配值或范围
            metric_filters: 指标筛选字典
            custom_filter: 自定义筛选函数
            runtime_min: 最小运行时间
            runtime_max: 最大运行时间
            time_start: 开始时间戳
            time_end: 结束时间戳

        Returns:
            筛选后的 ExperimentHistory
        """
        filtered = []

        for record in self._records:
            if algorithm_name is not None and record.algorithm_name != algorithm_name:
                continue
            if problem_name is not None and record.problem_name != problem_name:
                continue
            if runtime_min is not None and record.runtime < runtime_min:
                continue
            if runtime_max is not None and record.runtime > runtime_max:
                continue
            if time_start is not None and record.timestamp < time_start:
                continue
            if time_end is not None and record.timestamp > time_end:
                continue

            if param_filters:
                match = True
                for k, v in param_filters.items():
                    if k not in record.params or record.params[k] != v:
                        match = False
                        break
                if not match:
                    continue

            if metric_filters:
                match = True
                for k, v in metric_filters.items():
                    if k not in record.metrics or record.metrics[k] != v:
                        match = False
                        break
                if not match:
                    continue

            if custom_filter is not None and not custom_filter(record):
                continue

            filtered.append(record)

        history = ExperimentHistory()
        history._records = filtered
        return history

    def sort_by(self, key: Union[str, Callable[[ExperimentRecord], Any]],
                ascending: bool = True) -> 'ExperimentHistory':
        """排序记录

        Args:
            key: 排序键，可以是字符串（'runtime', 'timestamp', 或参数/指标名）或函数
            ascending: 是否升序

        Returns:
            排序后的 ExperimentHistory
        """
        if callable(key):
            sorted_records = sorted(self._records, key=key, reverse=not ascending)
        else:
            def get_key(record: ExperimentRecord) -> Any:
                if key == 'runtime':
                    return record.runtime
                if key == 'timestamp':
                    return record.timestamp
                if key in record.params:
                    return record.params[key]
                if key in record.metrics:
                    return record.metrics[key]
                return 0
            sorted_records = sorted(self._records, key=get_key, reverse=not ascending)

        history = ExperimentHistory()
        history._records = sorted_records
        return history

    def to_dataframe(self, flatten: bool = True) -> pd.DataFrame:
        """转换为 pandas DataFrame

        Args:
            flatten: 是否扁平化参数和指标列

        Returns:
            DataFrame
        """
        if not self._records:
            return pd.DataFrame()

        data = [record.to_dict(flatten=flatten) for record in self._records]
        df = pd.DataFrame(data)

        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

        return df

    def export_csv(self, filepath: str, flatten: bool = True, **kwargs) -> None:
        """导出为 CSV 文件

        Args:
            filepath: 输出文件路径
            flatten: 是否扁平化参数和指标列
            **kwargs: 传递给 pandas.to_csv 的额外参数
        """
        df = self.to_dataframe(flatten=flatten)
        df.to_csv(filepath, index=False, **kwargs)

    def import_csv(self, filepath: str, **kwargs) -> None:
        """从 CSV 文件导入记录

        Args:
            filepath: CSV 文件路径
            **kwargs: 传递给 pandas.read_csv 的额外参数
        """
        df = pd.read_csv(filepath, **kwargs)

        param_cols = [c for c in df.columns if c.startswith('param_')]
        metric_cols = [c for c in df.columns if c.startswith('metric_')]

        for _, row in df.iterrows():
            params = {col[6:]: row[col] for col in param_cols if pd.notna(row[col])}
            metrics = {col[7:]: row[col] for col in metric_cols if pd.notna(row[col])}

            record = ExperimentRecord(
                record_id=row.get('record_id', str(uuid.uuid4())),
                algorithm_name=row.get('algorithm_name', ''),
                problem_name=row.get('problem_name', ''),
                params=params,
                metrics=metrics,
                runtime=float(row.get('runtime', 0.0)),
                timestamp=float(row.get('timestamp', time.time()))
            )
            self._records.append(record)

    def clear(self) -> None:
        """清空所有记录"""
        self._records.clear()

    def unique_algorithms(self) -> List[str]:
        """获取所有唯一的算法名称"""
        return sorted(list(set(r.algorithm_name for r in self._records)))

    def unique_problems(self) -> List[str]:
        """获取所有唯一的问题名称"""
        return sorted(list(set(r.problem_name for r in self._records)))

    def summary(self) -> pd.DataFrame:
        """生成实验汇总表

        按算法和问题分组，计算各指标的均值、标准差等统计量。

        Returns:
            汇总 DataFrame
        """
        if not self._records:
            return pd.DataFrame()

        df = self.to_dataframe(flatten=True)
        metric_cols = [c for c in df.columns if c.startswith('metric_')]

        if not metric_cols:
            return df.groupby(['algorithm_name', 'problem_name']).size().reset_index(name='count')

        agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in metric_cols}
        agg_dict['runtime'] = ['mean', 'std']

        summary = df.groupby(['algorithm_name', 'problem_name']).agg(agg_dict)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()

        return summary
