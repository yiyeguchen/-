import cv2
import numpy as np
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class StoneAnalysisDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.setup_fonts()
        self.initUI()

    def setup_fonts(self):
        """设置统一的俄文字体方案"""
        self.title_font = QFont("Segoe UI", 14, QFont.Bold)
        self.standard_font = QFont("Segoe UI", 11, QFont.Normal)
        self.small_font = QFont("Segoe UI", 9, QFont.Normal)
        self.data_font = QFont("Segoe UI", 12, QFont.Medium)
        self.button_font = QFont("Segoe UI", 11, QFont.Medium)

        plt.rcParams['font.family'] = ['Segoe UI', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def initUI(self):
        self.setWindowTitle('Система интеллектуального анализа щебня - Модуль контурного анализа')
        self.setGeometry(100, 100, 1600, 900)
        self.setFont(self.standard_font)

        self.setStyleSheet("""
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
                font-size: 12px;
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
                font-size: 13px;
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
                font-size: 11px;
                line-height: 1.4;
            }
            QStatusBar {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                color: #5a6c7d;
            }
        """)

        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 顶部控制栏
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)

        self.load_btn = QPushButton('📁 Загрузить изображение')
        self.load_btn.setFont(self.button_font)
        self.load_btn.clicked.connect(self.load_image)

        self.analyze_btn = QPushButton('🔍 Контурный анализ')
        self.analyze_btn.setFont(self.button_font)
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)

        self.demo_btn = QPushButton('🎯 Демо-образец')
        self.demo_btn.setFont(self.button_font)
        self.demo_btn.clicked.connect(self.load_demo_sample)

        self.reset_btn = QPushButton('🔄 Сброс анализа')
        self.reset_btn.setFont(self.button_font)
        self.reset_btn.clicked.connect(self.reset_analysis)
        self.reset_btn.setEnabled(False)

        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.analyze_btn)
        control_layout.addWidget(self.demo_btn)
        control_layout.addWidget(self.reset_btn)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # 主要内容区域
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
        self.create_image_panel(content_splitter)

        # 右侧：分析结果
        self.create_analysis_panel(content_splitter)

        content_splitter.setSizes([800, 800])
        main_layout.addWidget(content_splitter)

        # 状态栏
        status_bar = self.statusBar()
        status_bar.setFont(self.small_font)
        status_bar.showMessage('Готов к контурному анализу изображений щебня')

    def create_image_panel(self, parent):
        """创建优化的图像显示面板"""
        image_widget = QWidget()
        layout = QVBoxLayout(image_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # 原始图像
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
            font-size: 14px;
            padding: 20px;
        """)
        self.original_label.setText("Нажмите для загрузки изображения\nили используйте демо-образец")

        original_layout.addWidget(self.original_label)

        # 处理结果
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
            font-size: 14px;
            padding: 20px;
        """)
        self.result_label.setText("Результаты контурного анализа\nбудут отображены здесь")

        result_layout.addWidget(self.result_label)

        layout.addWidget(original_group)
        layout.addWidget(result_group)

        parent.addWidget(image_widget)

    def create_analysis_panel(self, parent):
        """创建分析结果面板"""
        analysis_widget = QWidget()
        layout = QVBoxLayout(analysis_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # 轮廓统计信息
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
                "color: #007bff; font-weight: 600; padding: 5px 10px; background-color: #f8f9ff; border-radius: 6px;")

            stats_layout.addWidget(name_label, row, 0)
            stats_layout.addWidget(label, row, 1)
            row += 1

        layout.addWidget(stats_group)

        # 颜色编码说明
        legend_group = QGroupBox("🎨 Цветовая кодировка")
        legend_group.setFont(self.title_font)
        legend_layout = QVBoxLayout(legend_group)
        legend_layout.setContentsMargins(20, 25, 20, 20)

        red_label = QLabel("🔴 Красный: Все базовые контуры")
        red_label.setFont(self.standard_font)
        red_label.setStyleSheet("color: #dc3545; font-weight: 500; padding: 5px;")

        blue_label = QLabel("🔵 Синий: Максимальный контур по площади")
        blue_label.setFont(self.standard_font)
        blue_label.setStyleSheet("color: #007bff; font-weight: 500; padding: 5px;")

        green_label = QLabel("🟢 Зеленый: Второй по величине контур")
        green_label.setFont(self.standard_font)
        green_label.setStyleSheet("color: #28a745; font-weight: 500; padding: 5px;")

        legend_layout.addWidget(red_label)
        legend_layout.addWidget(blue_label)
        legend_layout.addWidget(green_label)

        layout.addWidget(legend_group)

        # 图表区域
        chart_group = QGroupBox("📈 Анализ распределения площадей")
        chart_group.setFont(self.title_font)
        chart_layout = QVBoxLayout(chart_group)
        chart_layout.setContentsMargins(15, 25, 15, 15)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.figure.patch.set_facecolor('white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(300)
        chart_layout.addWidget(self.canvas)

        layout.addWidget(chart_group)

        # 详细报告
        report_group = QGroupBox("📋 Технический отчет")
        report_group.setFont(self.title_font)
        report_layout = QVBoxLayout(report_group)
        report_layout.setContentsMargins(15, 25, 15, 15)

        self.report_text = QTextEdit()
        self.report_text.setMaximumHeight(150)
        self.report_text.setFont(self.standard_font)
        self.report_text.setStyleSheet("""
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            background-color: #f8f9fa;
            line-height: 1.5;
        """)
        self.report_text.setPlainText("Ожидание загрузки изображения для анализа...")

        report_layout.addWidget(self.report_text)
        layout.addWidget(report_group)

        parent.addWidget(analysis_widget)

    def load_image(self):
        """加载图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение щебня", "",
            "Файлы изображений (*.png *.jpg *.jpeg *.bmp);;Все файлы (*)"
        )

        if file_path:
            try:
                img_np = np.fromfile(file_path, dtype=np.uint8)
                self.current_image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                if self.current_image is not None:
                    self.display_image(self.current_image, self.original_label)
                    self.analyze_btn.setEnabled(True)
                    self.reset_btn.setEnabled(True)
                    self.statusBar().showMessage(f'Загружено: {file_path.split("/")[-1]}')
                else:
                    QMessageBox.critical(self, "Ошибка", "Невозможно прочитать файл изображения")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки: {str(e)}")

    def load_demo_sample(self):
        """加载演示样本"""
        demo_image = self.create_demo_image()
        self.current_image = demo_image
        self.display_image(demo_image, self.original_label)
        self.analyze_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.statusBar().showMessage('Загружен демо-образец для контурного анализа')

    def create_demo_image(self):
        """创建模拟碎石图像"""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 200
        np.random.seed(42)

        colors = [
            (120, 120, 120),  # 灰色
            (180, 180, 180),  # 浅灰色
            (100, 150, 200),  # 蓝灰色
            (150, 130, 100),  # 褐色
            (200, 200, 200),  # 白色
        ]

        for _ in range(150):
            x = np.random.randint(20, 580)
            y = np.random.randint(20, 380)
            size = np.random.randint(8, 25)
            color = colors[np.random.randint(0, len(colors))]

            cv2.ellipse(img, (x, y), (size, int(size * 0.8)),
                        np.random.randint(0, 180), 0, 360, color, -1)
            cv2.ellipse(img, (x, y), (size, int(size * 0.8)),
                        np.random.randint(0, 180), 0, 360, (80, 80, 80), 1)

        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def reset_analysis(self):
        """重置分析结果"""
        if self.current_image is not None:
            # 重置显示为原始图像
            self.display_image(self.current_image, self.result_label)

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
        """执行专用轮廓分析"""
        if self.current_image is None:
            return

        try:
            self.statusBar().showMessage('Выполняется контурный анализ...')
            QApplication.processEvents()

            # 执行核心轮廓分析管线
            contour_results = self.perform_contour_analysis(self.current_image)

            # 更新显示
            self.update_analysis_display(contour_results)

            # 创建并显示专用轮廓结果图像（无文本叠加）
            processed_img = self.create_contour_result_image(self.current_image, contour_results)
            self.display_image(processed_img, self.result_label)

            self.statusBar().showMessage('Контурный анализ успешно завершен')

        except Exception as e:
            QMessageBox.critical(self, "Ошибка анализа", f"Ошибка в процессе контурного анализа: {str(e)}")

    def perform_contour_analysis(self, img):
        """
        核心轮廓分析管线
        实现模块I和模块II的功能，检测两个最大的轮廓
        """
        # 预处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用Canny边缘检测，类似main.py的方法
        edges = cv2.Canny(gray, 50, 200)
        
        # 模块I: 基础轮廓提取
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 过滤小面积轮廓
        min_area = 50
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        # 按面积排序轮廓（降序）
        sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

        # 模块II: 获取两个最大面积轮廓
        largest_contour = sorted_contours[0] if len(sorted_contours) > 0 else None
        second_largest_contour = sorted_contours[1] if len(sorted_contours) > 1 else None

        contour_areas = []
        for contour in valid_contours:
            area = cv2.contourArea(contour)
            contour_areas.append(area)

        # 计算统计数据
        total_area = sum(contour_areas)
        largest_area = cv2.contourArea(largest_contour) if largest_contour is not None else 0
        second_largest_area = cv2.contourArea(second_largest_contour) if second_largest_contour is not None else 0
        largest_perimeter = cv2.arcLength(largest_contour, True) if largest_contour is not None else 0
        area_ratio = (largest_area / total_area * 100) if total_area > 0 else 0

        return {
            "all_contours": valid_contours,
            "largest_contour": largest_contour,
            "second_largest_contour": second_largest_contour,
            "contour_count": len(valid_contours),
            "largest_area": largest_area,
            "second_largest_area": second_largest_area,
            "largest_perimeter": largest_perimeter,
            "total_area": total_area,
            "area_ratio": area_ratio,
            "contour_areas": contour_areas
        }

    def create_contour_result_image(self, img, contour_results):
        """
        创建专用轮廓结果图像
        显示两个最大的轮廓：
        - 红色基础轮廓
        - 蓝色最大面积轮廓
        - 绿色第二大面积轮廓
        """
        # 创建原始图像的副本
        result_img = img.copy()

        # 模块I可视化: 红色基础轮廓 (RGB: 255,0,0)
        all_contours = contour_results["all_contours"]
        cv2.drawContours(result_img, all_contours, -1, (0, 0, 255), 2)  # OpenCV使用BGR格式

        # 模块II可视化: 蓝色最大面积轮廓 (RGB: 0,0,255)
        largest_contour = contour_results["largest_contour"]
        if largest_contour is not None:
            cv2.drawContours(result_img, [largest_contour], -1, (255, 0, 0), 10)  # 蓝色，粗线条突出显示

        # 绿色第二大面积轮廓 (RGB: 0,255,0)
        second_largest_contour = contour_results["second_largest_contour"]
        if second_largest_contour is not None:
            cv2.drawContours(result_img, [second_largest_contour], -1, (0, 255, 0), 10)  # 绿色，粗线条突出显示

        # 严格约束：不添加任何文本或其他标记
        return result_img

    def update_analysis_display(self, contour_results):
        """更新分析结果显示"""
        # 更新统计信息
        self.stats["Общее количество контуров"].setText(str(contour_results["contour_count"]))
        self.stats["Площадь максимального контура"].setText(f"{contour_results['largest_area']:.0f} px²")
        self.stats["Периметр максимального контура"].setText(f"{contour_results['largest_perimeter']:.1f} px")
        self.stats["Отношение площадей"].setText(f"{contour_results['area_ratio']:.1f}%")

        # 更新图表
        self.update_contour_charts(contour_results)

        # 更新报告
        self.update_contour_report(contour_results)

    def update_contour_charts(self, contour_results):
        """更新轮廓分析图表"""
        self.figure.clear()

        # 创建面积分布直方图
        ax = self.figure.add_subplot(1, 1, 1)

        contour_areas = contour_results["contour_areas"]
        if contour_areas:
            # 创建直方图
            n, bins, patches = ax.hist(contour_areas, bins=20, alpha=0.7, color='#6c757d', edgecolor='white')

            # 标记最大面积
            largest_area = contour_results["largest_area"]
            ax.axvline(x=largest_area, color='#007bff', linestyle='--', linewidth=2,
                       label=f'Максимальная площадь: {largest_area:.0f} px²')
            
            # 标记第二大面积
            second_largest_area = contour_results["second_largest_area"]
            if second_largest_area > 0:
                ax.axvline(x=second_largest_area, color='#28a745', linestyle='--', linewidth=2,
                           label=f'Вторая по величине: {second_largest_area:.0f} px²')

            ax.set_xlabel("Площадь контура (пиксели²)", fontdict={'family': 'Segoe UI', 'size': 10})
            ax.set_ylabel("Количество контуров", fontdict={'family': 'Segoe UI', 'size': 10})
            ax.set_title("Распределение площадей контуров",
                         fontdict={'family': 'Segoe UI', 'size': 11, 'weight': 'bold'})
            ax.legend()
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafbfc')

        self.figure.tight_layout()
        self.canvas.draw()

    def update_contour_report(self, contour_results):
        """更新轮廓分析报告"""
        report = f"""🔍 ОТЧЕТ КОНТУРНОГО АНАЛИЗА
{'═' * 45}

📊 СТАТИСТИКА ОБНАРУЖЕНИЯ:
  • Общее количество контуров: {contour_results['contour_count']} шт.
  • Максимальная площадь: {contour_results['largest_area']:.0f} пикселей²
  • Максимальный периметр: {contour_results['largest_perimeter']:.1f} пикселей
  • Доля максимального контура: {contour_results['area_ratio']:.1f}%

🎯 ДВА КРУПНЕЙШИХ КОНТУРА:
  🔵 Максимальный контур (синий):
    - Площадь: {contour_results['largest_area']:.0f} px²
    - Периметр: {contour_results['largest_perimeter']:.1f} px
    - Доля от общей площади: {contour_results['area_ratio']:.1f}%
  
  🟢 Второй по величине (зеленый):
    - Площадь: {contour_results['second_largest_area']:.0f} px²
    - Доля от общей площади: {(contour_results['second_largest_area']/contour_results['total_area']*100 if contour_results['total_area'] > 0 else 0):.1f}%

🎨 ЦВЕТОВАЯ КОДИРОВКА:
  🔴 Красные контуры: Все обнаруженные базовые контуры
  🔵 Синий контур: Контур с максимальной площадью
  🟢 Зеленый контур: Второй по величине контур

📈 АНАЛИЗ РАСПРЕДЕЛЕНИЯ:
  • Средняя площадь: {np.mean(contour_results['contour_areas']):.1f} px²
  • Общая площадь: {contour_results['total_area']:.0f} px²
  • Стандартное отклонение: {np.std(contour_results['contour_areas']):.1f} px²

🔬 ЗАКЛЮЧЕНИЕ:
Контурный анализ успешно выполнен. Доминирующий объект 
составляет {contour_results['area_ratio']:.1f}% от общей площади."""

        self.report_text.setPlainText(report)

    def display_image(self, cv_img, label):
        """在标签中显示图像"""
        if cv_img is None:
            return

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width

        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        label_size = label.size()
        scaled_pixmap = pixmap.scaled(
            int(label_size.width() * 0.85),
            int(label_size.height() * 0.85),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        label.setPixmap(scaled_pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setApplicationName("Система контурного анализа щебня")
    app.setApplicationVersion("2.0 Специализированный модуль")
    app.setFont(QFont("Segoe UI", 10))

    window = StoneAnalysisDemo()
    window.show()

    sys.exit(app.exec_())