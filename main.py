#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于计算机视觉的碎石粒度智能分析系统
主入口文件 (main.py)

项目名称: 基于计算机视觉的碎石粒度智能分析系统
版本: 1.0
作者: QS GROUP集团
描述: 应用程序主入口，负责启动PyQt5应用程序
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

# 导入主窗口类
from app_ui import StoneAnalysisDemo


def main():
    """
    应用程序主入口函数
    """
    # 启用高DPI支持（必须在QApplication创建之前设置）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # 创建QApplication实例
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("Система интеллектуального анализа щебня")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("QS GROUP")
    
    # 创建并显示主窗口
    window = StoneAnalysisDemo()
    window.show()
    
    # 启动事件循环
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
