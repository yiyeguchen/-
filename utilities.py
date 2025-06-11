#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于计算机视觉的碎石粒度智能分析系统
工具模块 (utilities.py)

项目名称: 基于计算机视觉的碎石粒度智能分析系统
版本: 1.0
作者: QS GROUP集团
描述: 提供辅助函数和工具功能，包括演示数据生成等
"""

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


def create_demo_image():
    """
    创建模拟碎石图像用于演示和测试
    
    Returns:
        numpy.ndarray: 生成的演示图像 (BGR格式)
    """
    # 创建基础背景图像
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # 设置随机种子以确保可重现的结果
    np.random.seed(42)
    
    # 定义碎石颜色调色板
    colors = [
        (120, 120, 120),  # 灰色
        (180, 180, 180),  # 浅灰色
        (100, 150, 200),  # 蓝灰色
        (150, 130, 100),  # 褐色
        (200, 200, 200),  # 白色
    ]
    
    # 生成150个随机椭圆形碎石
    for _ in range(150):
        # 随机位置
        x = np.random.randint(20, 580)
        y = np.random.randint(20, 380)
        
        # 随机大小
        size = np.random.randint(8, 25)
        
        # 随机颜色
        color = colors[np.random.randint(0, len(colors))]
        
        # 绘制填充椭圆（碎石主体）
        cv2.ellipse(img, (x, y), (size, int(size * 0.8)),
                    np.random.randint(0, 180), 0, 360, color, -1)
        
        # 绘制椭圆边框（增加真实感）
        cv2.ellipse(img, (x, y), (size, int(size * 0.8)),
                    np.random.randint(0, 180), 0, 360, (80, 80, 80), 1)
    
    # 添加噪声以增加真实感
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def convert_cv_to_qimage(cv_img):
    """
    将OpenCV图像转换为QImage格式
    
    Args:
        cv_img (numpy.ndarray): OpenCV图像 (BGR格式)
        
    Returns:
        QImage: 转换后的QImage对象
    """
    if cv_img is None:
        return None
        
    # 转换BGR到RGB
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width, channel = rgb_image.shape
    bytes_per_line = 3 * width
    
    # 创建QImage
    q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    return q_image


def scale_pixmap_to_label(pixmap, label, scale_factor=0.85):
    """
    将QPixmap缩放以适应标签大小
    
    Args:
        pixmap (QPixmap): 要缩放的像素图
        label (QLabel): 目标标签
        scale_factor (float): 缩放因子 (默认0.85)
        
    Returns:
        QPixmap: 缩放后的像素图
    """
    label_size = label.size()
    scaled_pixmap = pixmap.scaled(
        int(label_size.width() * scale_factor),
        int(label_size.height() * scale_factor),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )
    return scaled_pixmap


def display_image_in_label(cv_img, label):
    """
    在QLabel中显示OpenCV图像
    
    Args:
        cv_img (numpy.ndarray): OpenCV图像 (BGR格式)
        label (QLabel): 目标标签控件
    """
    if cv_img is None:
        return
        
    # 转换为QImage
    q_image = convert_cv_to_qimage(cv_img)
    if q_image is None:
        return
        
    # 转换为QPixmap并缩放
    pixmap = QPixmap.fromImage(q_image)
    scaled_pixmap = scale_pixmap_to_label(pixmap, label)
    
    # 设置到标签
    label.setPixmap(scaled_pixmap)


def validate_image_file(file_path):
    """
    验证图像文件是否有效
    
    Args:
        file_path (str): 图像文件路径
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # 尝试读取文件
        img_np = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        if img is None:
            return False, "无法解码图像文件"
            
        # 检查图像尺寸
        if img.shape[0] < 50 or img.shape[1] < 50:
            return False, "图像尺寸过小"
            
        return True, ""
        
    except Exception as e:
        return False, f"文件读取错误: {str(e)}"


def get_supported_image_formats():
    """
    获取支持的图像格式列表
    
    Returns:
        str: 文件对话框格式字符串
    """
    return "Файлы изображений (*.png *.jpg *.jpeg *.bmp);;Все файлы (*)"


def format_area_value(area):
    """
    格式化面积值显示
    
    Args:
        area (float): 面积值
        
    Returns:
        str: 格式化的面积字符串
    """
    if area >= 1000:
        return f"{area:.0f} px²"
    else:
        return f"{area:.1f} px²"


def format_perimeter_value(perimeter):
    """
    格式化周长值显示
    
    Args:
        perimeter (float): 周长值
        
    Returns:
        str: 格式化的周长字符串
    """
    return f"{perimeter:.1f} px"


def format_percentage_value(percentage):
    """
    格式化百分比值显示
    
    Args:
        percentage (float): 百分比值
        
    Returns:
        str: 格式化的百分比字符串
    """
    return f"{percentage:.1f}%"