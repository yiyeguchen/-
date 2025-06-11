#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于计算机视觉的碎石粒度智能分析系统
UI模块 (app_ui.py)

项目名称: 基于计算机视觉的碎石粒度智能分析系统
版本: 1.0
作者: QS GROUP集团
描述: 包含所有Qt控件的定义、布局和UI相关的逻辑
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import csv  # 新增：用于CSV导出功能
import os   # 新增：用于批量处理功能

# 导入自定义模块
from image_analyzer import perform_contour_analysis, create_contour_result_image
from utilities import (
    create_demo_image, display_image_in_label, validate_image_file,
    get_supported_image_formats, format_area_value, format_perimeter_value,
    format_percentage_value
)


# 新增：参数设置对话框类
class SettingsDialog(QDialog):
    """
    参数设置对话框
    允许用户调整核心分析算法的参数
    """
    
    def __init__(self, parent=None, current_params=None):
        super().__init__(parent)
        self.setWindowTitle('⚙️ Настройка параметров анализа')
        self.setFixedSize(400, 300)
        self.setModal(True)
        
        # 设置默认参数
        if current_params is None:
            current_params = {'canny_t1': 50, 'canny_t2': 200, 'min_area': 50}
        
        self.setup_ui(current_params)
    
    def setup_ui(self, current_params):
        """设置对话框界面"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题
        title_label = QLabel('Настройка параметров контурного анализа')
        title_label.setFont(QFont('Segoe UI', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 参数设置组
        params_group = QGroupBox('Параметры алгоритма')
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(15)
        
        # Canny阈值1
        self.canny_t1_spinbox = QSpinBox()
        self.canny_t1_spinbox.setRange(0, 255)
        self.canny_t1_spinbox.setValue(current_params['canny_t1'])
        self.canny_t1_spinbox.setSuffix(' (нижний порог)')
        params_layout.addRow('Canny порог 1:', self.canny_t1_spinbox)
        
        # Canny阈值2
        self.canny_t2_spinbox = QSpinBox()
        self.canny_t2_spinbox.setRange(0, 255)
        self.canny_t2_spinbox.setValue(current_params['canny_t2'])
        self.canny_t2_spinbox.setSuffix(' (верхний порог)')
        params_layout.addRow('Canny порог 2:', self.canny_t2_spinbox)
        
        # 最小轮廓面积
        self.min_area_spinbox = QSpinBox()
        self.min_area_spinbox.setRange(0, 1000)
        self.min_area_spinbox.setValue(current_params['min_area'])
        self.min_area_spinbox.setSuffix(' пикселей²')
        params_layout.addRow('Минимальная площадь:', self.min_area_spinbox)
        
        layout.addWidget(params_group)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton('✓ Применить')
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton('✗ Отмена')
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        layout.addLayout(button_layout)
    
    def get_parameters(self):
        """获取用户设置的参数"""
        return {
            'canny_t1': self.canny_t1_spinbox.value(),
            'canny_t2': self.canny_t2_spinbox.value(),
            'min_area': self.min_area_spinbox.value()
        }


class StoneAnalysisDemo(QMainWindow):
    """
    碎石分析演示主窗口类
    负责创建和管理整个用户界面
    """
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        
        # 新增：分析参数存储
        self.analysis_params = {
            'canny_t1': 50,
            'canny_t2': 200,
            'min_area': 50
        }
        
        # 新增：单位标定参数
        self.scale_ratio = 1.0  # 默认1像素=1单位
        
        # 新增：详细结果存储（用于CSV导出）
        self.detailed_results = []
        
        self.setup_fonts()
        self.initUI()
    
    def setup_fonts(self):
        """设置统一的俄文字体方案"""
        self.title_font = QFont("Segoe UI", 18, QFont.Bold)  # 从14增大到18
        self.standard_font = QFont("Segoe UI", 14, QFont.Normal)  # 从11增大到14
        self.small_font = QFont("Segoe UI", 12, QFont.Normal)  # 从9增大到12
        self.data_font = QFont("Segoe UI", 15, QFont.Medium)  # 从12增大到15
        self.button_font = QFont("Segoe UI", 14, QFont.Medium)  # 从11增大到14
        
        # 设置matplotlib字体
        plt.rcParams['font.family'] = ['Segoe UI', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    
    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle('Система интеллектуального анализа щебня - Модуль контурного анализа')
        self.setGeometry(100, 100, 1600, 900)
        self.setFont(self.standard_font)
        
        # 设置样式表
        self.setStyleSheet(self._get_stylesheet())
        
        # 创建主布局
        self._create_main_layout()
        
        # 设置状态栏
        status_bar = self.statusBar()
        status_bar.setFont(self.small_font)
        status_bar.showMessage('Готов к контурному анализу изображений щебня')
    
    def _get_stylesheet(self):
        """获取应用程序样式表"""
        return """
            QMainWindow { 
                background-color: #f8f9fa; 
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 500;
                font-family: 'Segoe UI', Arial, sans-serif;
                min-height: 20px;
            }
            QPushButton:hover { 
                background-color: #45a049; 
            }
            QPushButton:pressed { 
                background-color: #3d8b40; 
            }
            QPushButton:disabled { 
                background-color: #cccccc; 
                color: #888888;
            }
            QGroupBox {
                font-weight: 600;
                font-size: 16px;
                font-family: 'Segoe UI', Arial, sans-serif;
                border: 2px solid #e1e5e9;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: white;
                color: #2c3e50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 5px 15px 5px 15px;
                background-color: white;
                border-radius: 5px;
                color: #2c3e50;
            }
            QLabel {
                font-family: 'Segoe UI', Arial, sans-serif;
                color: #2c3e50;
            }
            QTextEdit {
                font-family: 'Segoe UI', 'Consolas', monospace;
                font-size: 17px;
                line-height: 1.4;
            }
            QStatusBar {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 17px;
                color: #5a6c7d;
            }
        """
    
    def _create_main_layout(self):
        """创建主布局"""
        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 顶部控制栏
        control_layout = self._create_control_panel()
        main_layout.addLayout(control_layout)
        
        # 主要内容区域
        content_splitter = self._create_content_area()
        main_layout.addWidget(content_splitter)
    
    def _create_control_panel(self):
        """创建顶部控制面板"""
        # 主控制布局
        main_control_layout = QVBoxLayout()
        main_control_layout.setSpacing(10)
        
        # 第一行：基础功能按钮
        first_row_layout = QHBoxLayout()
        first_row_layout.setSpacing(15)
        
        # 加载图像按钮
        self.load_btn = QPushButton('📁 Загрузить изображение')
        self.load_btn.setFont(self.button_font)
        self.load_btn.clicked.connect(self.load_image)
        
        # 分析按钮
        self.analyze_btn = QPushButton('🔍 Контурный анализ')
        self.analyze_btn.setFont(self.button_font)
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        
        # 演示样本按钮
        self.demo_btn = QPushButton('🎯 Демо-образец')
        self.demo_btn.setFont(self.button_font)
        self.demo_btn.clicked.connect(self.load_demo_sample)
        
        # 重置按钮
        self.reset_btn = QPushButton('🔄 Сброс анализа')
        self.reset_btn.setFont(self.button_font)
        self.reset_btn.clicked.connect(self.reset_analysis)
        self.reset_btn.setEnabled(False)
        
        first_row_layout.addWidget(self.load_btn)
        first_row_layout.addWidget(self.analyze_btn)
        first_row_layout.addWidget(self.demo_btn)
        first_row_layout.addWidget(self.reset_btn)
        first_row_layout.addStretch()
        
        # 第二行：高级功能按钮
        second_row_layout = QHBoxLayout()
        second_row_layout.setSpacing(15)
        
        # 新增：参数设置按钮
        self.settings_btn = QPushButton('⚙️ Настройка параметров')
        self.settings_btn.setFont(self.button_font)
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 500;
                font-family: 'Segoe UI', Arial, sans-serif;
                min-height: 20px;
            }
            QPushButton:hover { 
                background-color: #138496; 
            }
            QPushButton:pressed { 
                background-color: #117a8b; 
            }
        """)
        self.settings_btn.clicked.connect(self.open_settings_dialog)
        
        # 新增：批量处理按钮
        self.batch_btn = QPushButton('📁 Пакетная обработка')
        self.batch_btn.setFont(self.button_font)
        self.batch_btn.setStyleSheet("""
            QPushButton {
                background-color: #fd7e14;
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 500;
                font-family: 'Segoe UI', Arial, sans-serif;
                min-height: 20px;
            }
            QPushButton:hover { 
                background-color: #e8690b; 
            }
            QPushButton:pressed { 
                background-color: #d35400; 
            }
        """)
        self.batch_btn.clicked.connect(self.batch_process_images)
        
        # 新增：导出CSV按钮
        self.export_btn = QPushButton('💾 Экспорт в CSV')
        self.export_btn.setFont(self.button_font)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 500;
                font-family: 'Segoe UI', Arial, sans-serif;
                min-height: 20px;
            }
            QPushButton:hover { 
                background-color: #218838; 
            }
            QPushButton:pressed { 
                background-color: #1e7e34; 
            }
            QPushButton:disabled { 
                background-color: #cccccc; 
                color: #888888;
            }
        """)
        self.export_btn.clicked.connect(self.export_to_csv)
        self.export_btn.setEnabled(False)  # 默认禁用
        
        # 新增：单位标定按钮
        self.calibration_btn = QPushButton('📏 Калибровка')
        self.calibration_btn.setFont(self.button_font)
        self.calibration_btn.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 500;
                font-family: 'Segoe UI', Arial, sans-serif;
                min-height: 20px;
            }
            QPushButton:hover { 
                background-color: #5a32a3; 
            }
            QPushButton:pressed { 
                background-color: #4c2a85; 
            }
        """)
        self.calibration_btn.clicked.connect(self.open_calibration_dialog)
        
        second_row_layout.addWidget(self.settings_btn)
        second_row_layout.addWidget(self.batch_btn)
        second_row_layout.addWidget(self.export_btn)
        second_row_layout.addWidget(self.calibration_btn)
        second_row_layout.addStretch()
        
        # 添加两行到主布局
        main_control_layout.addLayout(first_row_layout)
        main_control_layout.addLayout(second_row_layout)
        
        return main_control_layout
    
    def _create_content_area(self):
        """创建主要内容区域"""
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setHandleWidth(8)
        content_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #e1e5e9;
                border-radius: 4px;
            }
            QSplitter::handle:hover {
                background-color: #cbd5db;
            }
        """)
        
        # 左侧：图像显示
        self._create_image_panel(content_splitter)
        
        # 右侧：分析结果
        self._create_analysis_panel(content_splitter)
        
        content_splitter.setSizes([800, 800])
        return content_splitter
    
    def _create_image_panel(self, parent):
        """创建图像显示面板"""
        image_widget = QWidget()
        layout = QVBoxLayout(image_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 原始图像组
        original_group = QGroupBox("📷 Исходное изображение")
        original_group.setFont(self.title_font)
        original_layout = QVBoxLayout(original_group)
        original_layout.setContentsMargins(15, 25, 15, 15)
        
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(350, 250)
        self.original_label.setFont(self.standard_font)
        self.original_label.setStyleSheet("""
            border: 3px dashed #cbd5db;
            background-color: #f8f9fa;
            border-radius: 12px;
            color: #6c757d;
            font-size: 17px;
            padding: 20px;
        """)
        self.original_label.setText("Нажмите для загрузки изображения\nили используйте демо-образец")
        
        original_layout.addWidget(self.original_label)
        
        # 处理结果组
        result_group = QGroupBox("🎯 Результаты контурного анализа")
        result_group.setFont(self.title_font)
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(15, 25, 15, 15)
        
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(350, 250)
        self.result_label.setFont(self.standard_font)
        self.result_label.setStyleSheet("""
            border: 3px dashed #cbd5db;
            background-color: #f8f9fa;
            border-radius: 12px;
            color: #6c757d;
            font-size: 17px;
            padding: 20px;
        """)
        self.result_label.setText("Результаты контурного анализа\nбудут отображены здесь")
        
        result_layout.addWidget(self.result_label)
        
        layout.addWidget(original_group)
        layout.addWidget(result_group)
        
        parent.addWidget(image_widget)
    
    def _create_analysis_panel(self, parent):
        """创建分析结果面板 - 选项卡式布局"""
        analysis_widget = QWidget()
        layout = QVBoxLayout(analysis_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建选项卡控件
        self.tab_widget = QTabWidget()
        self.tab_widget.setFont(self.standard_font)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f8f9fa;
                border: 2px solid #e9ecef;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 20px;
                margin-right: 2px;
                font-weight: 600;
                color: #495057;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #007bff;
                border-color: #007bff;
            }
            QTabBar::tab:hover {
                background-color: #e9ecef;
            }
        """)
        
        # 创建多个选项卡页面
        self._create_summary_tab()
        self._create_chart_tab()
        self._create_pie_chart_tab()
        self._create_scatter_plot_tab()
        self._create_box_plot_tab()
        self._create_comparison_tab()
        self._create_report_tab()
        
        layout.addWidget(self.tab_widget)
        parent.addWidget(analysis_widget)
    
    def _create_summary_tab(self):
        """创建统计摘要选项卡页面"""
        summary_widget = QWidget()
        layout = QVBoxLayout(summary_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 统计信息组
        stats_group = self._create_statistics_group()
        layout.addWidget(stats_group)
        
        # 颜色编码说明组
        legend_group = self._create_legend_group()
        layout.addWidget(legend_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        self.tab_widget.addTab(summary_widget, "📊 Сводка")
    
    def _create_chart_tab(self):
        """创建图表选项卡页面"""
        chart_widget = QWidget()
        layout = QVBoxLayout(chart_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 图表组
        chart_group = self._create_chart_group()
        layout.addWidget(chart_group)
        
        self.tab_widget.addTab(chart_widget, "📈 График")
    
    def _create_report_tab(self):
        """创建详细报告选项卡页面"""
        report_widget = QWidget()
        layout = QVBoxLayout(report_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 创建更专业的报告显示区域
        self.report_text = QTextEdit()
        self.report_text.setFont(QFont("Consolas", 13))  # 使用等宽字体，增大字体大小
        self.report_text.setStyleSheet("""
            QTextEdit {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 20px;
                background-color: #f8f9fa;
                line-height: 1.6;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)
        self.report_text.setPlainText("Ожидание загрузки изображения для анализа...")
        
        layout.addWidget(self.report_text)
        
        self.tab_widget.addTab(report_widget, "📋 Отчет")
    
    def _create_statistics_group(self):
        """创建统计信息组"""
        stats_group = QGroupBox("📊 Статистика контуров")
        stats_group.setFont(self.title_font)
        stats_layout = QGridLayout(stats_group)
        stats_layout.setContentsMargins(20, 30, 20, 20)
        stats_layout.setSpacing(15)
        
        # 统计标签
        self.stats = {
            "Общее количество контуров": QLabel("--"),
            "Площадь максимального контура": QLabel("--"),
            "Периметр максимального контура": QLabel("--"),
            "Отношение площадей": QLabel("--")
        }
        
        row = 0
        for name, label in self.stats.items():
            name_label = QLabel(name + ":")
            name_label.setFont(self.standard_font)
            name_label.setStyleSheet("font-weight: 600; color: #495057;")
            
            label.setFont(self.data_font)
            label.setStyleSheet(
                "color: #007bff; font-weight: 600; padding: 5px 10px; "
                "background-color: #f8f9ff; border-radius: 6px;"
            )
            
            stats_layout.addWidget(name_label, row, 0)
            stats_layout.addWidget(label, row, 1)
            row += 1
        
        return stats_group
    
    def _create_legend_group(self):
        """创建颜色编码说明组"""
        legend_group = QGroupBox("🎨 Цветовая кодировка")
        legend_group.setFont(self.title_font)
        legend_layout = QVBoxLayout(legend_group)
        legend_layout.setContentsMargins(20, 25, 20, 20)
        
        red_label = QLabel("🔴 Красный: Максимальный контур по площади")
        red_label.setFont(self.standard_font)
        red_label.setStyleSheet("color: #dc3545; font-weight: 500; padding: 5px;")
        
        blue_label = QLabel("🔵 Синий: Второй по величине контур")
        blue_label.setFont(self.standard_font)
        blue_label.setStyleSheet("color: #007bff; font-weight: 500; padding: 5px;")
        
        legend_layout.addWidget(red_label)
        legend_layout.addWidget(blue_label)
        
        return legend_group
    
    def _create_chart_group(self):
        """创建图表组"""
        chart_group = QGroupBox("📈 Анализ распределения площадей")
        chart_group.setFont(self.title_font)
        chart_layout = QVBoxLayout(chart_group)
        chart_layout.setContentsMargins(15, 25, 15, 15)
        
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.figure.patch.set_facecolor('white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(300)
        chart_layout.addWidget(self.canvas)
        
        return chart_group
    
    def _create_pie_chart_tab(self):
        """创建饼图选项卡页面"""
        pie_widget = QWidget()
        layout = QVBoxLayout(pie_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 饼图组
        pie_group = QGroupBox("🥧 Круговая диаграмма распределения")
        pie_group.setFont(self.title_font)
        pie_layout = QVBoxLayout(pie_group)
        pie_layout.setContentsMargins(15, 25, 15, 15)
        
        self.pie_figure = Figure(figsize=(8, 6), dpi=100)
        self.pie_figure.patch.set_facecolor('white')
        self.pie_canvas = FigureCanvas(self.pie_figure)
        self.pie_canvas.setMinimumHeight(400)
        pie_layout.addWidget(self.pie_canvas)
        
        layout.addWidget(pie_group)
        
        self.tab_widget.addTab(pie_widget, "🥧 Круговая")
    
    def _create_scatter_plot_tab(self):
        """创建散点图选项卡页面"""
        scatter_widget = QWidget()
        layout = QVBoxLayout(scatter_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 散点图组
        scatter_group = QGroupBox("📊 Диаграмма рассеяния (Площадь vs Периметр)")
        scatter_group.setFont(self.title_font)
        scatter_layout = QVBoxLayout(scatter_group)
        scatter_layout.setContentsMargins(15, 25, 15, 15)
        
        self.scatter_figure = Figure(figsize=(8, 6), dpi=100)
        self.scatter_figure.patch.set_facecolor('white')
        self.scatter_canvas = FigureCanvas(self.scatter_figure)
        self.scatter_canvas.setMinimumHeight(400)
        scatter_layout.addWidget(self.scatter_canvas)
        
        layout.addWidget(scatter_group)
        
        self.tab_widget.addTab(scatter_widget, "📊 Рассеяние")
    
    def _create_box_plot_tab(self):
        """创建箱线图选项卡页面"""
        box_widget = QWidget()
        layout = QVBoxLayout(box_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 箱线图组
        box_group = QGroupBox("📦 Ящичная диаграмма распределения")
        box_group.setFont(self.title_font)
        box_layout = QVBoxLayout(box_group)
        box_layout.setContentsMargins(15, 25, 15, 15)
        
        self.box_figure = Figure(figsize=(8, 6), dpi=100)
        self.box_figure.patch.set_facecolor('white')
        self.box_canvas = FigureCanvas(self.box_figure)
        self.box_canvas.setMinimumHeight(400)
        box_layout.addWidget(self.box_canvas)
        
        layout.addWidget(box_group)
        
        self.tab_widget.addTab(box_widget, "📦 Ящичная")
    
    def _create_comparison_tab(self):
        """创建对比分析选项卡页面"""
        comparison_widget = QWidget()
        layout = QVBoxLayout(comparison_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 对比分析组
        comparison_group = QGroupBox("⚖️ Сравнительный анализ фракций")
        comparison_group.setFont(self.title_font)
        comparison_layout = QVBoxLayout(comparison_group)
        comparison_layout.setContentsMargins(15, 25, 15, 15)
        
        self.comparison_figure = Figure(figsize=(10, 8), dpi=100)
        self.comparison_figure.patch.set_facecolor('white')
        self.comparison_canvas = FigureCanvas(self.comparison_figure)
        self.comparison_canvas.setMinimumHeight(500)
        comparison_layout.addWidget(self.comparison_canvas)
        
        layout.addWidget(comparison_group)
        
        self.tab_widget.addTab(comparison_widget, "⚖️ Сравнение")
    

    
    # 事件处理方法
    def load_image(self):
        """加载图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение щебня", "",
            get_supported_image_formats()
        )
        
        if file_path:
            try:
                # 验证图像文件
                is_valid, error_msg = validate_image_file(file_path)
                if not is_valid:
                    QMessageBox.critical(self, "Ошибка", error_msg)
                    return
                
                # 读取图像
                img_np = np.fromfile(file_path, dtype=np.uint8)
                self.current_image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                
                if self.current_image is not None:
                    # 保存当前图像路径
                    self.current_image_path = file_path
                    # 保存原始图像用于报告
                    self.original_image = self.current_image.copy()
                    
                    display_image_in_label(self.current_image, self.original_label)
                    self.analyze_btn.setEnabled(True)
                    self.reset_btn.setEnabled(True)
                    self.statusBar().showMessage(f'Загружено: {file_path.split("/")[-1]}')
                else:
                    QMessageBox.critical(self, "Ошибка", "Невозможно прочитать файл изображения")
                    
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки: {str(e)}")
    
    def load_demo_sample(self):
        """加载演示样本"""
        try:
            demo_image = create_demo_image()
            self.current_image = demo_image
            # 设置演示样本的路径信息
            self.current_image_path = "demo_sample.png"
            # 保存原始图像用于报告
            self.original_image = demo_image.copy()
            
            display_image_in_label(demo_image, self.original_label)
            self.analyze_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.statusBar().showMessage('Загружен демо-образец для контурного анализа')
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка создания демо-образца: {str(e)}")
    
    def reset_analysis(self):
        """重置分析结果"""
        if self.current_image is not None:
            # 重置显示为原始图像
            display_image_in_label(self.current_image, self.result_label)
            
            # 重置统计信息
            for label in self.stats.values():
                label.setText("--")
            
            # 清空图表
            self.figure.clear()
            self.canvas.draw()
            
            # 重置报告
            self.report_text.setPlainText("Анализ сброшен. Нажмите 'Контурный анализ' для повторного анализа.")
            
            self.statusBar().showMessage('Анализ сброшен')
    
    def analyze_image(self):
        """执行轮廓分析"""
        if self.current_image is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение!")
            return
        
        try:
            self.statusBar().showMessage('Выполняется контурный анализ...')
            QApplication.processEvents()
            
            # 新增：使用用户设置的参数执行核心轮廓分析
            contour_results = perform_contour_analysis(
                self.current_image,
                canny_t1=self.analysis_params['canny_t1'],
                canny_t2=self.analysis_params['canny_t2'],
                min_area=self.analysis_params['min_area']
            )
            
            # 新增：存储详细结果用于CSV导出
            self.detailed_results = contour_results.get('detailed_contours', [])
            
            # 更新显示
            self.update_analysis_display(contour_results)
            
            # 创建并显示结果图像
            processed_img = create_contour_result_image(self.current_image, contour_results)
            display_image_in_label(processed_img, self.result_label)
            
            # 新增：启用导出按钮
            self.export_btn.setEnabled(True)
            
            self.statusBar().showMessage('Контурный анализ успешно завершен')
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка анализа", f"Ошибка в процессе контурного анализа: {str(e)}")
    
    def update_analysis_display(self, contour_results):
        """更新分析结果显示"""
        # 更新统计信息
        self.stats["Общее количество контуров"].setText(str(contour_results["contour_count"]))
        self.stats["Площадь максимального контура"].setText(format_area_value(contour_results['largest_area']))
        self.stats["Периметр максимального контура"].setText(format_perimeter_value(contour_results['largest_perimeter']))
        self.stats["Отношение площадей"].setText(format_percentage_value(contour_results['area_ratio']))
        
        # 更新图表
        self.update_contour_charts(contour_results)
        
        # 更新新增的图表
        self.update_pie_chart(contour_results)
        self.update_scatter_plot(contour_results)
        self.update_box_plot(contour_results)
        self.update_comparison_chart(contour_results)
        
        # 更新报告
        self.update_contour_report(contour_results)
    
    def update_contour_charts(self, contour_results):
        """更新轮廓分析图表"""
        self.figure.clear()
        
        # 创建面积分布直方图
        ax = self.figure.add_subplot(1, 1, 1)
        
        # 从轮廓计算面积
        contour_areas = [cv2.contourArea(cnt) for cnt in contour_results["all_contours"]]
        if contour_areas:
            # 根据标定比例转换单位
            if self.scale_ratio > 1.0:
                # 转换为物理单位
                areas_display = [area / (self.scale_ratio ** 2) for area in contour_areas]
                largest_area_display = contour_results["largest_area"] / (self.scale_ratio ** 2)
                second_largest_area_display = contour_results["second_largest_area"] / (self.scale_ratio ** 2)
                area_unit = 'mm²'
            else:
                # 保持像素单位
                areas_display = contour_areas
                largest_area_display = contour_results["largest_area"]
                second_largest_area_display = contour_results["second_largest_area"]
                area_unit = 'px²'
            
            # 创建直方图
            n, bins, patches = ax.hist(areas_display, bins=20, alpha=0.7, 
                                     color='#6c757d', edgecolor='white')
            
            # 标记最大面积（红色）
            ax.axvline(x=largest_area_display, color='#dc3545', linestyle='--', linewidth=2,
                      label=f'Максимальная площадь: {largest_area_display:.1f} {area_unit}')
            
            # 标记第二大面积（蓝色）
            if second_largest_area_display > 0:
                ax.axvline(x=second_largest_area_display, color='#007bff', linestyle='--', linewidth=2,
                          label=f'Вторая по величине: {second_largest_area_display:.1f} {area_unit}')
            
            ax.set_xlabel(f"Площадь контура ({area_unit})", fontdict={'family': 'Segoe UI', 'size': 13})
            ax.set_ylabel("Количество контуров", fontdict={'family': 'Segoe UI', 'size': 13})
            ax.set_title("Распределение площадей контуров",
                        fontdict={'family': 'Segoe UI', 'size': 14, 'weight': 'bold'})
            ax.legend()
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafbfc')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_contour_report(self, contour_results):
        """更新轮廓分析报告"""
        # 根据标定比例转换单位
        if self.scale_ratio > 1.0:
            # 转换为物理单位
            largest_area_display = contour_results['largest_area'] / (self.scale_ratio ** 2)
            largest_perimeter_display = contour_results['largest_perimeter'] / self.scale_ratio
            second_largest_area_display = contour_results['second_largest_area'] / (self.scale_ratio ** 2)
            second_largest_perimeter_display = contour_results.get('second_largest_perimeter', 0) / self.scale_ratio
            total_area_display = contour_results['total_area'] / (self.scale_ratio ** 2)
            avg_area_display = total_area_display / contour_results['contour_count'] if contour_results['contour_count'] > 0 else 0
            area_unit = 'mm²'
            length_unit = 'mm'
            scale_info = f"\n📏 КАЛИБРОВКА: 1 мм = {self.scale_ratio:.2f} пикселей"
        else:
            # 保持像素单位
            largest_area_display = contour_results['largest_area']
            largest_perimeter_display = contour_results['largest_perimeter']
            second_largest_area_display = contour_results['second_largest_area']
            second_largest_perimeter_display = contour_results.get('second_largest_perimeter', 0)
            total_area_display = contour_results['total_area']
            avg_area_display = total_area_display / contour_results['contour_count'] if contour_results['contour_count'] > 0 else 0
            area_unit = 'px²'
            length_unit = 'px'
            scale_info = ""
        
        # 计算第二大轮廓的面积占比
        second_area_ratio = 0
        if contour_results['total_area'] > 0:
            second_area_ratio = (contour_results['second_largest_area'] / contour_results['total_area']) * 100
        
        # 生成详细的专业技术报告
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""📊 ТЕХНИЧЕСКИЙ ОТЧЕТ ПО АНАЛИЗУ ФРАКЦИЙ ЩЕБНЯ
{'═' * 80}

📋 ОБЩАЯ ИНФОРМАЦИЯ:
  • Дата и время анализа: {current_time}
  • Файл изображения: {os.path.basename(self.current_image_path) if hasattr(self, 'current_image_path') else 'Неизвестно'}
  • Разрешение изображения: {self.original_image.shape[1]}×{self.original_image.shape[0]} пикселей{scale_info}

🔬 ПАРАМЕТРЫ АЛГОРИТМА:
  • Метод обнаружения контуров: Двухэтапный анализ
    - Этап 1: Пороговая обработка (THRESH_BINARY + OTSU)
    - Этап 2: Детектор границ Canny
  • Пороги Canny: Нижний={self.analysis_params['canny_t1']}, Верхний={self.analysis_params['canny_t2']}
  • Минимальная площадь фильтрации: {self.analysis_params['min_area']} пикселей
  • Метод аппроксимации контуров: CHAIN_APPROX_SIMPLE
  • Морфологические операции: Применены для улучшения качества

📊 РЕЗУЛЬТАТЫ КОЛИЧЕСТВЕННОГО АНАЛИЗА:
  • Общее количество обнаруженных контуров: {contour_results['contour_count']}
  • Суммарная площадь всех фракций: {total_area_display:.2f} {area_unit}
  • Средняя площадь фракции: {avg_area_display:.2f} {area_unit}
  • Стандартное отклонение площадей: Требует дополнительного расчета

🎯 АНАЛИЗ ДОМИНИРУЮЩИХ ФРАКЦИЙ:

🔴 КРУПНЕЙШАЯ ФРАКЦИЯ (выделена красным цветом):
  • Площадь поверхности: {largest_area_display:.2f} {area_unit}
  • Периметр: {largest_perimeter_display:.2f} {length_unit}
  • Процентная доля от общей площади: {contour_results['area_ratio']:.2f}%
  • Эквивалентный диаметр: {(4 * largest_area_display / 3.14159) ** 0.5:.2f} {length_unit}

🔵 ВТОРАЯ ПО РАЗМЕРУ ФРАКЦИЯ (выделена синим цветом):
  • Площадь поверхности: {second_largest_area_display:.2f} {area_unit}
  • Периметр: {second_largest_perimeter_display:.2f} {length_unit}
  • Процентная доля от общей площади: {second_area_ratio:.2f}%
  • Эквивалентный диаметр: {(4 * second_largest_area_display / 3.14159) ** 0.5:.2f} {length_unit}

📈 СТАТИСТИЧЕСКИЙ АНАЛИЗ РАСПРЕДЕЛЕНИЯ:
  • Коэффициент неоднородности: {(largest_area_display / avg_area_display):.2f}
  • Отношение крупнейшей к второй фракции: {(largest_area_display / second_largest_area_display):.2f}
  • Индекс концентрации (топ-2 фракции): {(contour_results['area_ratio'] + second_area_ratio):.1f}%

🎨 СИСТЕМА ЦВЕТОВОГО КОДИРОВАНИЯ:
  🔴 Красный контур - максимальная площадь (доминирующая фракция)
  🔵 Синий контур - вторая по величине площадь
  ⚪ Тонкие красные линии - все остальные обнаруженные контуры

🔍 КАЧЕСТВО ОБРАБОТКИ:
  • Четкость границ: Высокая (алгоритм Canny)
  • Точность сегментации: Оптимизирована для щебня
  • Устойчивость к шумам: Морфологическая фильтрация применена

📋 ТЕХНИЧЕСКОЕ ЗАКЛЮЧЕНИЕ:
Анализ фракционного состава щебня успешно завершен. Обнаружено 
{contour_results['contour_count']} отдельных фракций. Доминирующая фракция 
составляет {contour_results['area_ratio']:.2f}% от общей площади образца, 
что указывает на {'высокую' if contour_results['area_ratio'] > 15 else 'умеренную' if contour_results['area_ratio'] > 8 else 'низкую'} 
степень неоднородности материала.

⚠️  ПРИМЕЧАНИЯ:
  • Результаты действительны для данного ракурса съемки
  • Рекомендуется анализ нескольких проекций для полной оценки
  • Калибровка масштаба повышает точность измерений

{'─' * 80}
📧 Отчет сгенерирован системой компьютерного зрения v2.0"""
        
        self.report_text.setPlainText(report)
    
    # 新增：参数设置功能
    def open_settings_dialog(self):
        """打开参数设置对话框"""
        dialog = SettingsDialog(self, self.analysis_params)
        if dialog.exec_() == QDialog.Accepted:
            # 更新参数
            self.analysis_params = dialog.get_parameters()
            self.statusBar().showMessage(f'Параметры обновлены: Canny({self.analysis_params["canny_t1"]}, {self.analysis_params["canny_t2"]}), Мин.площадь: {self.analysis_params["min_area"]}')
    
    # 新增：批量处理功能
    def batch_process_images(self):
        """批量处理多张图像"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите изображения для пакетной обработки",
            "",
            "Image files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if not file_paths:
            return
        
        try:
            # 创建进度对话框
            progress = QProgressDialog("Обработка изображений...", "Отмена", 0, len(file_paths), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            
            batch_results = []
            
            for i, file_path in enumerate(file_paths):
                if progress.wasCanceled():
                    break
                
                progress.setValue(i)
                progress.setLabelText(f"Обработка: {os.path.basename(file_path)}")
                QApplication.processEvents()
                
                # 读取图像
                image = cv2.imread(file_path)
                if image is None:
                    continue
                
                # 分析图像
                contour_results = perform_contour_analysis(
                    image,
                    canny_t1=self.analysis_params['canny_t1'],
                    canny_t2=self.analysis_params['canny_t2'],
                    min_area=self.analysis_params['min_area']
                )
                
                # 存储结果
                batch_results.append({
                    'filename': os.path.basename(file_path),
                    'contour_count': contour_results['contour_count'],
                    'largest_area': contour_results['largest_area'],
                    'largest_perimeter': contour_results['largest_perimeter'],
                    'total_area': contour_results['total_area'],
                    'area_ratio': contour_results['area_ratio']
                })
            
            progress.setValue(len(file_paths))
            
            # 显示批量处理结果
            self.display_batch_results(batch_results)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при пакетной обработке: {str(e)}")
    
    def display_batch_results(self, batch_results):
        """显示批量处理结果"""
        if not batch_results:
            return
        
        # 计算汇总统计
        total_files = len(batch_results)
        avg_contours = np.mean([r['contour_count'] for r in batch_results])
        avg_largest_area = np.mean([r['largest_area'] for r in batch_results])
        max_area_file = max(batch_results, key=lambda x: x['largest_area'])
        
        # 生成批量处理报告
        report = f"""📁 ОТЧЕТ ПАКЕТНОЙ ОБРАБОТКИ
{'═' * 50}

📊 ОБЩАЯ СТАТИСТИКА:
  • Обработано файлов: {total_files}
  • Среднее количество контуров: {avg_contours:.1f}
  • Средняя максимальная площадь: {avg_largest_area:.0f} px²
  • Файл с наибольшей площадью: {max_area_file['filename']} ({max_area_file['largest_area']:.0f} px²)

📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:
"""
        
        for result in batch_results:
            report += f"\n🔸 {result['filename']}:\n"
            report += f"   Контуры: {result['contour_count']}, "
            report += f"Макс.площадь: {result['largest_area']:.0f} px², "
            report += f"Доля: {result['area_ratio']:.1f}%\n"
        
        self.report_text.setPlainText(report)
        self.statusBar().showMessage(f'Пакетная обработка завершена: {total_files} файлов')
    
    # 新增：导出CSV功能
    def export_to_csv(self):
        """导出分析结果到CSV文件"""
        if not self.detailed_results:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для экспорта. Сначала выполните анализ.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результаты в CSV",
            "contour_analysis_results.csv",
            "CSV files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['ID', 'Area', 'Perimeter', 'Area_mm2', 'Perimeter_mm']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 写入表头
                writer.writeheader()
                
                # 写入数据
                for i, contour_data in enumerate(self.detailed_results, 1):
                    area_px = contour_data['area']
                    perimeter_px = contour_data['perimeter']
                    
                    # 转换为物理单位
                    area_mm = area_px / (self.scale_ratio ** 2)
                    perimeter_mm = perimeter_px / self.scale_ratio
                    
                    writer.writerow({
                        'ID': i,
                        'Area': f"{area_px:.2f}",
                        'Perimeter': f"{perimeter_px:.2f}",
                        'Area_mm2': f"{area_mm:.2f}",
                        'Perimeter_mm': f"{perimeter_mm:.2f}"
                    })
            
            QMessageBox.information(self, "Успех", f"Результаты успешно экспортированы в:\n{file_path}")
            self.statusBar().showMessage(f'Данные экспортированы: {len(self.detailed_results)} контуров')
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при экспорте: {str(e)}")
    
    # 新增：单位标定功能
    def open_calibration_dialog(self):
        """打开单位标定对话框"""
        pixels_per_mm, ok = QInputDialog.getDouble(
            self,
            "📏 Калибровка единиц измерения",
            "Введите количество пикселей в 1 мм:\n(например: 50.0 означает 1мм = 50 пикселей)",
            self.scale_ratio,
            0.1,
            10000.0,
            2
        )
        
        if ok and pixels_per_mm > 0:
            self.scale_ratio = pixels_per_mm
            
            # 更新显示单位
            self.update_display_units()
            
            QMessageBox.information(
                self,
                "Калибровка завершена",
                f"Масштаб установлен: 1 мм = {self.scale_ratio:.2f} пикселей\n\nВсе результаты теперь отображаются в миллиметрах."
            )
            
            self.statusBar().showMessage(f'Калибровка: 1мм = {self.scale_ratio:.2f}px')
    
    def update_display_units(self):
        """更新显示单位"""
        # 这里可以添加更新所有显示单位的逻辑
        # 由于当前的显示更新在analyze_image中进行，这里主要是为了扩展性
        pass
    
    def update_pie_chart(self, contour_results):
        """更新饼图"""
        self.pie_figure.clear()
        
        # 计算轮廓面积
        contour_areas = [cv2.contourArea(cnt) for cnt in contour_results["all_contours"]]
        if not contour_areas:
            return
        
        # 根据面积大小分类
        total_area = sum(contour_areas)
        large_threshold = np.percentile(contour_areas, 75)  # 75%分位数
        medium_threshold = np.percentile(contour_areas, 25)  # 25%分位数
        
        large_count = sum(1 for area in contour_areas if area >= large_threshold)
        medium_count = sum(1 for area in contour_areas if medium_threshold <= area < large_threshold)
        small_count = sum(1 for area in contour_areas if area < medium_threshold)
        
        # 创建饼图
        ax = self.pie_figure.add_subplot(1, 1, 1)
        
        sizes = [large_count, medium_count, small_count]
        labels = ['Крупные фракции', 'Средние фракции', 'Мелкие фракции']
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        explode = (0.08, 0, 0)  # выделить крупные частицы
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                          autopct='%1.1f%%', shadow=True, startangle=90)
        
        # настройка шрифта, избежание наложения
        for text in texts:
            text.set_fontsize(11)
            text.set_fontfamily('DejaVu Sans')
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')
            autotext.set_fontfamily('DejaVu Sans')
        
        ax.set_title('Распределение размеров частиц', 
                    fontdict={'family': 'DejaVu Sans', 'size': 13, 'weight': 'bold'}, pad=15)
        
        self.pie_figure.tight_layout(pad=2.0)
        self.pie_canvas.draw()
    
    def update_scatter_plot(self, contour_results):
        """更新散点图"""
        self.scatter_figure.clear()
        
        # 计算轮廓面积和周长
        contour_areas = [cv2.contourArea(cnt) for cnt in contour_results["all_contours"]]
        contour_perimeters = [cv2.arcLength(cnt, True) for cnt in contour_results["all_contours"]]
        
        if not contour_areas:
            return
        
        # 根据标定比例转换单位
        if self.scale_ratio > 1.0:
            areas_display = [area / (self.scale_ratio ** 2) for area in contour_areas]
            perimeters_display = [perimeter / self.scale_ratio for perimeter in contour_perimeters]
            area_unit = 'mm²'
            perimeter_unit = 'mm'
        else:
            areas_display = contour_areas
            perimeters_display = contour_perimeters
            area_unit = 'px²'
            perimeter_unit = 'px'
        
        # 创建散点图
        ax = self.scatter_figure.add_subplot(1, 1, 1)
        
        # 根据面积大小设置颜色
        colors = plt.cm.viridis(np.array(areas_display) / max(areas_display))
        
        scatter = ax.scatter(areas_display, perimeters_display, c=colors, alpha=0.6, s=50)
        
        ax.set_xlabel(f'Площадь ({area_unit})', fontdict={'family': 'DejaVu Sans', 'size': 11})
        ax.set_ylabel(f'Периметр ({perimeter_unit})', fontdict={'family': 'DejaVu Sans', 'size': 11})
        ax.set_title('Соотношение площади и периметра',
                    fontdict={'family': 'DejaVu Sans', 'size': 12, 'weight': 'bold'}, pad=12)
        
        # 添加趋势线
        if len(areas_display) > 1:
            z = np.polyfit(areas_display, perimeters_display, 1)
            p = np.poly1d(z)
            ax.plot(sorted(areas_display), p(sorted(areas_display)), "r--", alpha=0.8, linewidth=2)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafbfc')
        
        # 添加颜色条
        cbar = self.scatter_figure.colorbar(scatter, ax=ax)
        cbar.set_label(f'Площадь ({area_unit})', fontdict={'family': 'DejaVu Sans', 'size': 10})
        cbar.ax.tick_params(labelsize=9)
        
        # 设置刻度标签字体
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        self.scatter_figure.tight_layout(pad=2.0)
        self.scatter_canvas.draw()
    
    def update_box_plot(self, contour_results):
        """更新箱线图"""
        self.box_figure.clear()
        
        # 计算轮廓面积和周长
        contour_areas = [cv2.contourArea(cnt) for cnt in contour_results["all_contours"]]
        contour_perimeters = [cv2.arcLength(cnt, True) for cnt in contour_results["all_contours"]]
        
        if not contour_areas:
            return
        
        # 根据标定比例转换单位
        if self.scale_ratio > 1.0:
            areas_display = [area / (self.scale_ratio ** 2) for area in contour_areas]
            perimeters_display = [perimeter / self.scale_ratio for perimeter in contour_perimeters]
            unit_suffix = ' (mm²/mm)'
        else:
            areas_display = contour_areas
            perimeters_display = contour_perimeters
            unit_suffix = ' (px²/px)'
        
        # 创建箱线图
        ax = self.box_figure.add_subplot(1, 1, 1)
        
        # подготовка данных
        data = [areas_display, perimeters_display]
        labels = [f'Площадь{unit_suffix}', f'Периметр{unit_suffix}']
        
        # нормализация данных для сравнения
        normalized_areas = (np.array(areas_display) - np.mean(areas_display)) / np.std(areas_display)
        normalized_perimeters = (np.array(perimeters_display) - np.mean(perimeters_display)) / np.std(perimeters_display)
        
        box_data = [normalized_areas, normalized_perimeters]
        box_labels = ['Площадь\n(нормализованная)', 'Периметр\n(нормализованный)']
        
        # 创建箱线图
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                       notch=True, showmeans=True)
        
        # 设置颜色
        colors = ['#ff6b6b', '#4ecdc4']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 设置样式
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color='#2c3e50', linewidth=1.5)
        
        plt.setp(bp['means'], marker='D', markerfacecolor='white', 
                markeredgecolor='#2c3e50', markersize=6)
        
        ax.set_ylabel('Нормализованные значения', fontdict={'family': 'DejaVu Sans', 'size': 11})
        ax.set_title('Диаграмма распределения характеристик',
                    fontdict={'family': 'DejaVu Sans', 'size': 12, 'weight': 'bold'}, pad=12)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafbfc')
        
        # 设置刻度标签字体
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        self.box_figure.tight_layout(pad=2.0)
        self.box_canvas.draw()
    
    def update_comparison_chart(self, contour_results):
        """更新对比分析图表"""
        self.comparison_figure.clear()
        
        # 计算轮廓面积
        contour_areas = [cv2.contourArea(cnt) for cnt in contour_results["all_contours"]]
        if not contour_areas:
            return
        
        # 根据标定比例转换单位
        if self.scale_ratio > 1.0:
            areas_display = [area / (self.scale_ratio ** 2) for area in contour_areas]
            unit = 'mm²'
        else:
            areas_display = contour_areas
            unit = 'px²'
        
        # 创建2x2子图布局，增加间距避免重叠
        gs = self.comparison_figure.add_gridspec(2, 2, hspace=0.45, wspace=0.35, 
                                                top=0.92, bottom=0.08, left=0.08, right=0.95)
        
        # 1. гистограмма распределения площадей
        ax1 = self.comparison_figure.add_subplot(gs[0, 0])
        n, bins, patches = ax1.hist(areas_display, bins=12, alpha=0.7, color='#3498db', edgecolor='white')
        ax1.set_xlabel(f'Площадь ({unit})', fontsize=9, fontfamily='DejaVu Sans')
        ax1.set_ylabel('Количество', fontsize=9, fontfamily='DejaVu Sans')
        ax1.set_title('Распределение площадей', fontsize=10, weight='bold', fontfamily='DejaVu Sans', pad=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        
        # 2. кумулятивное распределение
        ax2 = self.comparison_figure.add_subplot(gs[0, 1])
        sorted_areas = np.sort(areas_display)
        cumulative = np.arange(1, len(sorted_areas) + 1) / len(sorted_areas) * 100
        ax2.plot(sorted_areas, cumulative, color='#e74c3c', linewidth=2.5)
        ax2.set_xlabel(f'Площадь ({unit})', fontsize=9, fontfamily='DejaVu Sans')
        ax2.set_ylabel('Кумулятивный %', fontsize=9, fontfamily='DejaVu Sans')
        ax2.set_title('Кумулятивное распределение', fontsize=10, weight='bold', fontfamily='DejaVu Sans', pad=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        
        # 3. статистика по категориям
        ax3 = self.comparison_figure.add_subplot(gs[1, 0])
        
        # классификация по площади
        percentiles = [25, 50, 75, 90]
        thresholds = [np.percentile(areas_display, p) for p in percentiles]
        
        categories = ['0-25%', '25-50%', '50-75%', '75-90%', '90-100%']
        counts = []
        
        for i in range(len(categories)):
            if i == 0:
                count = sum(1 for area in areas_display if area <= thresholds[0])
            elif i == len(categories) - 1:
                count = sum(1 for area in areas_display if area > thresholds[-1])
            else:
                count = sum(1 for area in areas_display if thresholds[i-1] < area <= thresholds[i])
            counts.append(count)
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        bars = ax3.bar(categories, counts, color=colors, alpha=0.8)
        ax3.set_xlabel('Процентильные группы', fontsize=9, fontfamily='DejaVu Sans')
        ax3.set_ylabel('Количество', fontsize=9, fontfamily='DejaVu Sans')
        ax3.set_title('Распределение по группам', fontsize=10, weight='bold', fontfamily='DejaVu Sans', pad=8)
        ax3.tick_params(axis='both', which='major', labelsize=8)
        
        # поворот меток оси x для избежания наложения
        ax3.tick_params(axis='x', rotation=45)
        
        # добавление числовых меток
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.02,
                    f'{count}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 4. статистическая сводка
        ax4 = self.comparison_figure.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        # расчет статистической информации, оптимизация формата текста
        stats_text = f"""📊 СТАТИСТИЧЕСКАЯ СВОДКА

• Общее количество: {len(areas_display)}
• Средняя площадь: {np.mean(areas_display):.1f} {unit}
• Медианная площадь: {np.median(areas_display):.1f} {unit}
• Стандартное отклонение: {np.std(areas_display):.1f} {unit}
• Минимальная площадь: {np.min(areas_display):.1f} {unit}
• Максимальная площадь: {np.max(areas_display):.1f} {unit}
• Коэффициент вариации: {(np.std(areas_display)/np.mean(areas_display)*100):.1f}%"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='DejaVu Sans', linespacing=1.3,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', 
                         edgecolor='#dee2e6', alpha=0.9, linewidth=1))
        
        # настройка положения и размера главного заголовка
        self.comparison_figure.suptitle('Комплексный анализ фракций щебня', 
                                       fontsize=12, weight='bold', y=1.08, fontfamily='DejaVu Sans')
        
        self.comparison_canvas.draw()