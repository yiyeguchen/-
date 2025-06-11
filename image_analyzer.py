#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于计算机视觉的碎石粒度智能分析系统
核心分析模块 (image_analyzer.py)

项目名称: 基于计算机视觉的碎石粒度智能分析系统
版本: 1.0
作者: QS GROUP集团
描述: 包含所有与OpenCV相关的图像处理函数，实现轮廓检测、分析和排序等核心功能
"""

import cv2
import numpy as np


def perform_contour_analysis(image, canny_t1=50, canny_t2=200, min_area=50):
    """
    对输入图像执行轮廓分析
    参考main1.py的两阶段处理方法：
    1. 使用阈值处理找到所有轮廓
    2. 使用Canny边缘检测找到主要轮廓
    
    Args:
        image: 输入的BGR图像
        canny_t1: Canny边缘检测的低阈值 (新增参数)
        canny_t2: Canny边缘检测的高阈值 (新增参数)
        min_area: 最小轮廓面积阈值 (新增参数)
        
    Returns:
        dict: 包含分析结果的字典，新增detailed_contours字段
    """
    # 第一阶段：使用阈值处理找到所有轮廓（参考main1.py第一部分）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 寻找所有轮廓
    all_contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 第二阶段：使用Canny边缘检测找到主要轮廓（参考main1.py第二部分）
    edges = cv2.Canny(gray, canny_t1, canny_t2)
    main_contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # 过滤小轮廓 (使用传入的最小面积参数)
    filtered_contours = [cnt for cnt in main_contours if cv2.contourArea(cnt) > min_area]
    
    if not filtered_contours:
        return {
            'contour_count': 0,
            'largest_area': 0,
            'largest_perimeter': 0,
            'second_largest_area': 0,
            'second_largest_perimeter': 0,
            'total_area': 0,
            'area_ratio': 0,
            'largest_contour': None,
            'second_largest_contour': None,
            'all_contours': [],
            'detailed_contours': []  # 新增：详细轮廓数据
        }
    
    # 按面积排序轮廓（降序）
    sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
    
    # 获取最大和第五大轮廓（参考main1.py的实现）
    largest_contour = sorted_contours[0]
    # 如果有足够的轮廓，取第五大的；否则取第二大的
    if len(sorted_contours) >= 5:
        second_largest_contour = sorted_contours[4]  # 第五大轮廓
    elif len(sorted_contours) > 1:
        second_largest_contour = sorted_contours[1]  # 第二大轮廓
    else:
        second_largest_contour = None
    
    # 计算统计数据
    largest_area = cv2.contourArea(largest_contour)
    largest_perimeter = cv2.arcLength(largest_contour, True)
    
    second_largest_area = cv2.contourArea(second_largest_contour) if second_largest_contour is not None else 0
    second_largest_perimeter = cv2.arcLength(second_largest_contour, True) if second_largest_contour is not None else 0
    
    total_area = sum(cv2.contourArea(cnt) for cnt in filtered_contours)
    area_ratio = (largest_area / total_area * 100) if total_area > 0 else 0
    
    # 新增：生成详细轮廓数据用于CSV导出
    detailed_contours = []
    for i, contour in enumerate(sorted_contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        detailed_contours.append({
            'id': i + 1,
            'area': area,
            'perimeter': perimeter
        })
    
    return {
        'contour_count': len(filtered_contours),
        'largest_area': largest_area,
        'largest_perimeter': largest_perimeter,
        'second_largest_area': second_largest_area,
        'second_largest_perimeter': second_largest_perimeter,
        'total_area': total_area,
        'area_ratio': area_ratio,
        'largest_contour': largest_contour,
        'second_largest_contour': second_largest_contour,
        'all_contours': filtered_contours,
        'detailed_contours': detailed_contours  # 新增：详细轮廓数据
    }


def create_contour_result_image(img, contour_results):
    """
    创建专用轮廓结果图像
    参考main1.py的实现：
    1. 先绘制所有轮廓（红色细线）
    2. 突出显示最大面积轮廓（红色粗线）
    3. 突出显示第五大面积轮廓（蓝色粗线）
    
    Args:
        img (numpy.ndarray): 原始输入图像 (BGR格式)
        contour_results (dict): 轮廓分析结果字典
        
    Returns:
        numpy.ndarray: 带有轮廓标记的结果图像 (BGR格式)
    """
    # 创建原始图像的副本
    result_img = img.copy()
    
    # 第一步：绘制所有轮廓（红色细线，参考main1.py第一部分）
    all_contours = contour_results["all_contours"]
    for i, contour in enumerate(all_contours):
        if i == 0:  # 跳过第一个轮廓（通常是整个图像边界）
            continue
        cv2.drawContours(result_img, [contour], 0, (0, 0, 255), 2)
    
    # 第二步：突出显示最大面积轮廓（红色粗线）
    largest_contour = contour_results["largest_contour"]
    if largest_contour is not None:
        cv2.drawContours(result_img, [largest_contour], -1, (0, 0, 255), 10)
    
    # 第三步：突出显示第五大面积轮廓（蓝色粗线）
    second_largest_contour = contour_results["second_largest_contour"]
    if second_largest_contour is not None:
        cv2.drawContours(result_img, [second_largest_contour], -1, (255, 0, 0), 10)
    
    return result_img


def analyze_contour_properties(contour):
    """
    分析单个轮廓的详细属性
    
    Args:
        contour (numpy.ndarray): 轮廓数据
        
    Returns:
        dict: 轮廓属性字典，包含面积、周长、质心等信息
    """
    if contour is None:
        return None
        
    # 计算基本属性
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # 计算质心
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0
    
    # 计算边界矩形
    x, y, w, h = cv2.boundingRect(contour)
    
    # 计算最小外接圆
    (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
    
    # 计算凸包
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    # 计算凸性缺陷比率
    solidity = area / hull_area if hull_area > 0 else 0
    
    # 计算长宽比
    aspect_ratio = w / h if h > 0 else 0
    
    # 计算矩形度
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    
    return {
        "area": area,
        "perimeter": perimeter,
        "centroid": (cx, cy),
        "bounding_rect": (x, y, w, h),
        "min_enclosing_circle": ((circle_x, circle_y), radius),
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "hull_area": hull_area
    }


def filter_contours_by_area(contours, min_area=50, max_area=None):
    """
    根据面积过滤轮廓
    
    Args:
        contours (list): 轮廓列表
        min_area (float): 最小面积阈值
        max_area (float): 最大面积阈值 (None表示无限制)
        
    Returns:
        list: 过滤后的轮廓列表
    """
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 检查最小面积
        if area < min_area:
            continue
            
        # 检查最大面积
        if max_area is not None and area > max_area:
            continue
            
        filtered_contours.append(contour)
    
    return filtered_contours


def sort_contours_by_area(contours, reverse=True):
    """
    按面积对轮廓进行排序
    
    Args:
        contours (list): 轮廓列表
        reverse (bool): 是否降序排列 (True为降序，False为升序)
        
    Returns:
        list: 排序后的轮廓列表
    """
    return sorted(contours, key=cv2.contourArea, reverse=reverse)


def get_contour_statistics(contours):
    """
    计算轮廓集合的统计信息
    
    Args:
        contours (list): 轮廓列表
        
    Returns:
        dict: 统计信息字典
    """
    if not contours:
        return {
            "count": 0,
            "total_area": 0,
            "mean_area": 0,
            "std_area": 0,
            "min_area": 0,
            "max_area": 0,
            "median_area": 0
        }
    
    # 计算所有面积
    areas = [cv2.contourArea(contour) for contour in contours]
    
    return {
        "count": len(contours),
        "total_area": sum(areas),
        "mean_area": np.mean(areas),
        "std_area": np.std(areas),
        "min_area": min(areas),
        "max_area": max(areas),
        "median_area": np.median(areas)
    }


def preprocess_image(img, blur_kernel_size=5, canny_low=50, canny_high=200):
    """
    图像预处理管线
    
    Args:
        img (numpy.ndarray): 输入图像 (BGR格式)
        blur_kernel_size (int): 高斯模糊核大小
        canny_low (int): Canny边缘检测低阈值
        canny_high (int): Canny边缘检测高阈值
        
    Returns:
        tuple: (灰度图, 边缘图)
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    
    # Canny边缘检测
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    return gray, edges


def detect_and_analyze_contours(img, min_area=50, max_contours=None):
    """
    完整的轮廓检测和分析管线
    
    Args:
        img (numpy.ndarray): 输入图像 (BGR格式)
        min_area (float): 最小轮廓面积阈值
        max_contours (int): 最大返回轮廓数量 (None表示无限制)
        
    Returns:
        dict: 完整的分析结果
    """
    # 预处理
    gray, edges = preprocess_image(img)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # 过滤轮廓
    valid_contours = filter_contours_by_area(contours, min_area)
    
    # 排序轮廓
    sorted_contours = sort_contours_by_area(valid_contours)
    
    # 限制轮廓数量
    if max_contours is not None:
        sorted_contours = sorted_contours[:max_contours]
    
    # 计算统计信息
    stats = get_contour_statistics(sorted_contours)
    
    # 分析前两个最大轮廓
    largest_contour = sorted_contours[0] if len(sorted_contours) > 0 else None
    second_largest_contour = sorted_contours[1] if len(sorted_contours) > 1 else None
    
    largest_props = analyze_contour_properties(largest_contour)
    second_largest_props = analyze_contour_properties(second_largest_contour)
    
    return {
        "preprocessed": {
            "gray": gray,
            "edges": edges
        },
        "contours": {
            "all": sorted_contours,
            "largest": largest_contour,
            "second_largest": second_largest_contour
        },
        "properties": {
            "largest": largest_props,
            "second_largest": second_largest_props
        },
        "statistics": stats
    }