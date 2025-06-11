# 碎石分析系统代码优化建议与重构方案

"""
基于原始代码的完整优化重构方案
主要改进:
1. 模块化架构设计
2. 多线程处理优化
3. 算法可配置化
4. 错误处理完善
5. 性能监控集成
6. 数据持久化支持
"""

import sys
import os
import cv2
import numpy as np
import threading
import queue
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ===================== 配置和数据模型 =====================

@dataclass
class AnalysisConfig:
    """分析配置参数"""
    min_contour_area: int = 50
    gaussian_blur_kernel: int = 5
    morphology_kernel_size: int = 3
    canny_lower_threshold: int = 50
    canny_upper_threshold: int = 150
    primary_algorithm: str = "auto"  # auto, edge_detection, color_segmentation, hybrid

    # 颜色分割参数
    red_hsv_ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
    blue_hsv_ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None

    def __post_init__(self):
        if self.red_hsv_ranges is None:
            self.red_hsv_ranges = [
                ((0, 30, 30), (15, 255, 255)),
                ((165, 30, 30), (180, 255, 255))
            ]
        if self.blue_hsv_ranges is None:
            self.blue_hsv_ranges = [((90, 30, 30), (140, 255, 255))]


@dataclass
class AnalysisResult:
    """分析结果数据模型"""
    timestamp: str
    contour_count: int
    largest_area: float
    second_largest_area: float
    largest_perimeter: float
    total_area: float
    area_ratio: float
    contour_areas: List[float]
    algorithm_used: str
    processing_time: float
    image_path: Optional[str] = None

    # 维修评估结果
    crushing_efficiency: float = 0.0
    equipment_status: str = "unknown"
    maintenance_recommendation: str = "需要进一步分析"


# ===================== 算法引擎模块 =====================

class AlgorithmEngine:
    """算法引擎基类"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (self.config.gaussian_blur_kernel, self.config.gaussian_blur_kernel), 0)
        return blurred

    def analyze(self, image: np.ndarray) -> Dict:
        """抽象分析方法"""
        raise NotImplementedError

    def postprocess(self, contours: List, image_shape: Tuple) -> List:
        """后处理过滤小轮廓"""
        return [c for c in contours if cv2.contourArea(c) > self.config.min_contour_area]


class EdgeDetectionAlgorithm(AlgorithmEngine):
    """边缘检测算法"""

    def analyze(self, image: np.ndarray) -> Dict:
        processed = self.preprocess(image)

        # Canny边缘检测
        edges = cv2.Canny(processed, self.config.canny_lower_threshold, self.config.canny_upper_threshold)

        # 形态学操作
        kernel = np.ones((self.config.morphology_kernel_size, self.config.morphology_kernel_size), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = self.postprocess(contours, image.shape)

        return {
            'contours': filtered_contours,
            'algorithm': 'edge_detection',
            'confidence': self._calculate_confidence(filtered_contours)
        }

    def _calculate_confidence(self, contours: List) -> float:
        """计算算法置信度"""
        if not contours:
            return 0.0
        return min(len(contours) / 20.0, 1.0)  # 基于轮廓数量的简单置信度


class ColorSegmentationAlgorithm(AlgorithmEngine):
    """颜色分割算法"""

    def analyze(self, image: np.ndarray) -> Dict:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 创建颜色掩码
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        # 红色掩码
        for lower, upper in self.config.red_hsv_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # 蓝色掩码
        for lower, upper in self.config.blue_hsv_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = self.postprocess(contours, image.shape)

        return {
            'contours': filtered_contours,
            'algorithm': 'color_segmentation',
            'confidence': self._calculate_confidence(filtered_contours, combined_mask)
        }

    def _calculate_confidence(self, contours: List, mask: np.ndarray) -> float:
        """基于掩码质量计算置信度"""
        if not contours:
            return 0.0

        total_pixels = mask.shape[0] * mask.shape[1]
        color_pixels = np.sum(mask > 0)
        color_ratio = color_pixels / total_pixels

        return min(color_ratio * 10, 1.0)  # 基于颜色像素比例


class HybridAlgorithm(AlgorithmEngine):
    """混合算法 - 结合多种方法"""

    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.edge_algo = EdgeDetectionAlgorithm(config)
        self.color_algo = ColorSegmentationAlgorithm(config)

    def analyze(self, image: np.ndarray) -> Dict:
        # 同时运行两种算法
        edge_result = self.edge_algo.analyze(image)
        color_result = self.color_algo.analyze(image)

        # 根据置信度选择最佳结果
        if color_result['confidence'] > edge_result['confidence']:
            primary_result = color_result
            fallback_contours = edge_result['contours']
        else:
            primary_result = edge_result
            fallback_contours = color_result['contours']

        # 合并轮廓（如果需要）
        if len(primary_result['contours']) < 2 and fallback_contours:
            combined_contours = primary_result['contours'] + fallback_contours
            # 去重和重新过滤
            combined_contours = self._remove_duplicate_contours(combined_contours)
            primary_result['contours'] = self.postprocess(combined_contours, image.shape)

        primary_result['algorithm'] = 'hybrid'
        return primary_result

    def _remove_duplicate_contours(self, contours: List) -> List:
        """移除重复轮廓"""
        if not contours:
            return contours

        unique_contours = []
        for contour in contours:
            is_duplicate = False
            for existing in unique_contours:
                # 简单的重复检测 - 比较轮廓中心和面积
                if self._contours_similar(contour, existing):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_contours.append(contour)

        return unique_contours

    def _contours_similar(self, c1, c2, threshold: float = 0.1) -> bool:
        """检查两个轮廓是否相似"""
        try:
            # 比较面积
            area1, area2 = cv2.contourArea(c1), cv2.contourArea(c2)
            if abs(area1 - area2) / max(area1, area2, 1) > threshold:
                return False

            # 比较中心点
            M1 = cv2.moments(c1)
            M2 = cv2.moments(c2)
            if M1["m00"] == 0 or M2["m00"] == 0:
                return False

            cx1, cy1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
            cx2, cy2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])

            distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            return distance < 20  # 像素距离阈值

        except:
            return False


# ===================== 核心分析服务 =====================

class ContourAnalysisService:
    """轮廓分析服务"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.algorithms = {
            'edge_detection': EdgeDetectionAlgorithm(config),
            'color_segmentation': ColorSegmentationAlgorithm(config),
            'hybrid': HybridAlgorithm(config)
        }
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_image(self, image: np.ndarray, algorithm: str = None) -> AnalysisResult:
        """执行图像分析"""
        start_time = datetime.now()

        if algorithm is None:
            algorithm = self.config.primary_algorithm

        # 自动算法选择
        if algorithm == "auto":
            algorithm = self._select_best_algorithm(image)

        # 执行分析
        if algorithm not in self.algorithms:
            algorithm = 'edge_detection'  # 默认算法

        try:
            result = self.algorithms[algorithm].analyze(image)
            contours = result['contours']

            # 计算统计数据
            analysis_result = self._calculate_statistics(contours, algorithm, start_time)

            # 维修评估
            analysis_result = self._assess_maintenance(analysis_result)

            return analysis_result

        except Exception as e:
            self.logger.error(f"分析失败: {str(e)}")
            # 返回空结果
            return AnalysisResult(
                timestamp=datetime.now().isoformat(),
                contour_count=0,
                largest_area=0,
                second_largest_area=0,
                largest_perimeter=0,
                total_area=0,
                area_ratio=0,
                contour_areas=[],
                algorithm_used=algorithm,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    def _select_best_algorithm(self, image: np.ndarray) -> str:
        """自动选择最佳算法"""
        # 简单的图像特征分析
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 检查颜色饱和度
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation)

        # 检查颜色分布
        color_variance = np.var(hsv[:, :, 0])

        if avg_saturation > 50 and color_variance > 200:
            return 'color_segmentation'
        elif avg_saturation < 30:
            return 'edge_detection'
        else:
            return 'hybrid'

    def _calculate_statistics(self, contours: List, algorithm: str, start_time: datetime) -> AnalysisResult:
        """计算统计数据"""
        processing_time = (datetime.now() - start_time).total_seconds()

        if not contours:
            return AnalysisResult(
                timestamp=datetime.now().isoformat(),
                contour_count=0,
                largest_area=0,
                second_largest_area=0,
                largest_perimeter=0,
                total_area=0,
                area_ratio=0,
                contour_areas=[],
                algorithm_used=algorithm,
                processing_time=processing_time
            )

        # 按面积排序
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # 计算面积
        contour_areas = [cv2.contourArea(c) for c in contours]
        total_area = sum(contour_areas)
        largest_area = contour_areas[0] if contour_areas else 0
        second_largest_area = contour_areas[1] if len(contour_areas) > 1 else 0

        # 计算周长
        largest_perimeter = cv2.arcLength(sorted_contours[0], True) if sorted_contours else 0

        # 计算比例
        area_ratio = (largest_area / total_area * 100) if total_area > 0 else 0

        return AnalysisResult(
            timestamp=datetime.now().isoformat(),
            contour_count=len(contours),
            largest_area=largest_area,
            second_largest_area=second_largest_area,
            largest_perimeter=largest_perimeter,
            total_area=total_area,
            area_ratio=area_ratio,
            contour_areas=contour_areas,
            algorithm_used=algorithm,
            processing_time=processing_time
        )

    def _assess_maintenance(self, result: AnalysisResult) -> AnalysisResult:
        """评估维修需求"""
        # 基于轮廓分析结果评估破碎效率
        if result.contour_count == 0:
            result.crushing_efficiency = 0.0
            result.equipment_status = "异常"
            result.maintenance_recommendation = "设备需要立即检查"
        elif result.area_ratio > 50:
            # 大颗粒占比过高，破碎效率低
            result.crushing_efficiency = 60.0
            result.equipment_status = "需要关注"
            result.maintenance_recommendation = "建议检查破碎刀片磨损情况"
        elif result.area_ratio < 10:
            # 颗粒过于细碎
            result.crushing_efficiency = 90.0
            result.equipment_status = "良好"
            result.maintenance_recommendation = "设备运行正常，继续监控"
        else:
            # 正常范围
            result.crushing_efficiency = 85.0
            result.equipment_status = "良好"
            result.maintenance_recommendation = "按计划进行常规维护"

        return result


# ===================== 多线程任务管理 =====================

class AnalysisWorker(QThread):
    """分析工作线程"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)  # AnalysisResult
    error = pyqtSignal(str)

    def __init__(self, image: np.ndarray, config: AnalysisConfig, algorithm: str = None):
        super().__init__()
        self.image = image
        self.config = config
        self.algorithm = algorithm
        self.service = ContourAnalysisService(config)

    def run(self):
        try:
            self.progress.emit(20)
            result = self.service.analyze_image(self.image, self.algorithm)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ===================== 数据持久化 =====================

class DataManager:
    """数据管理器"""

    def __init__(self, data_dir: str = "analysis_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.config_file = self.data_dir / "config.json"
        self.results_file = self.data_dir / "analysis_history.json"

    def save_config(self, config: AnalysisConfig):
        """保存配置"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    def load_config(self) -> AnalysisConfig:
        """加载配置"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return AnalysisConfig(**data)
        return AnalysisConfig()

    def save_result(self, result: AnalysisResult):
        """保存分析结果"""
        results = self.load_results()
        results.append(asdict(result))

        # 只保留最近100条记录
        if len(results) > 100:
            results = results[-100:]

        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def load_results(self) -> List[Dict]:
        """加载历史结果"""
        if self.results_file.exists():
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []


# ===================== 性能监控 =====================

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics = {}

    def start_timer(self, operation: str):
        """开始计时"""
        self.metrics[operation] = {'start': datetime.now()}

    def end_timer(self, operation: str):
        """结束计时"""
        if operation in self.metrics:
            self.metrics[operation]['duration'] = (
                    datetime.now() - self.metrics[operation]['start']
            ).total_seconds()

    def get_metrics(self) -> Dict:
        """获取性能指标"""
        return {k: v.get('duration', 0) for k, v in self.metrics.items()}


# ===================== 主应用程序类重构 =====================

class OptimizedStoneAnalysisApp(QMainWindow):
    """优化后的主应用程序"""

    def __init__(self):
        super().__init__()
        self.setup_logging()
        self.data_manager = DataManager()
        self.config = self.data_manager.load_config()
        self.performance_monitor = PerformanceMonitor()
        self.current_image = None
        self.current_result = None
        self.analysis_worker = None

        self.setup_ui()
        self.setup_status_bar()

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('stone_analysis.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def setup_ui(self):
        """设置优化的用户界面"""
        # 这里会是完整的UI设置代码
        # 包括新的算法选择器、性能监控显示等
        pass

    def setup_status_bar(self):
        """设置状态栏"""
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def analyze_image_optimized(self):
        """优化的图像分析"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return

        # 显示进度
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 创建工作线程
        algorithm = self.get_selected_algorithm()
        self.analysis_worker = AnalysisWorker(self.current_image, self.config, algorithm)
        self.analysis_worker.progress.connect(self.progress_bar.setValue)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.error.connect(self.on_analysis_error)

        # 开始分析
        self.performance_monitor.start_timer('full_analysis')
        self.analysis_worker.start()

    def on_analysis_finished(self, result: AnalysisResult):
        """分析完成处理"""
        self.performance_monitor.end_timer('full_analysis')
        self.progress_bar.setVisible(False)

        self.current_result = result
        self.data_manager.save_result(result)

        # 更新UI显示
        self.update_analysis_display(result)

        # 显示性能指标
        metrics = self.performance_monitor.get_metrics()
        self.logger.info(f"分析完成，性能指标: {metrics}")

    def on_analysis_error(self, error_msg: str):
        """分析错误处理"""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "分析错误", f"分析过程中发生错误:\n{error_msg}")
        self.logger.error(f"分析错误: {error_msg}")

    def get_selected_algorithm(self) -> str:
        """获取选中的算法"""
        # 这里会从UI获取用户选择的算法
        return self.config.primary_algorithm

    def update_analysis_display(self, result: AnalysisResult):
        """更新分析结果显示"""
        # 更新统计数据、图表、报告等
        pass

    def closeEvent(self, event):
        """关闭事件处理"""
        # 保存配置
        self.data_manager.save_config(self.config)

        # 停止工作线程
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.terminate()
            self.analysis_worker.wait()

        event.accept()


# ===================== 使用示例 =====================

def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序信息
    app.setApplicationName("智能碎石分析系统")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("石料分析研究所")

    # 创建主窗口
    window = OptimizedStoneAnalysisApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# ===================== 配置文件示例 =====================

# analysis_config.yaml
"""
analysis:
  min_contour_area: 50
  gaussian_blur_kernel: 5
  morphology_kernel_size: 3
  canny_lower_threshold: 50
  canny_upper_threshold: 150
  primary_algorithm: "auto"

color_segmentation:
  red_hsv_ranges:
    - [[0, 30, 30], [15, 255, 255]]
    - [[165, 30, 30], [180, 255, 255]]
  blue_hsv_ranges:
    - [[90, 30, 30], [140, 255, 255]]

maintenance_thresholds:
  excellent_efficiency: 90
  good_efficiency: 80
  poor_efficiency: 60
  critical_area_ratio: 50
  optimal_area_ratio: 20

performance:
  max_image_size: [2048, 2048]
  thread_pool_size: 4
  cache_size: 100
"""