# Документ о технологическом стеке и архитектуре
## Интеллектуальная система анализа гранулометрического состава щебня на основе компьютерного зрения

**Версия документа**: 1.0  
**Дата создания**: 2024 год  
**Кодовое название проекта**: Stone-Vision-Analyzer  
**Ответственная команда**: QS GROUP  

---

## 1. Обзор технологического стека

### 1.1 Выбор технологического стека

Данная система построена на основе современного технологического стека Python, обеспечивающего высокую производительность, стабильность и расширяемость. Выбор технологий основан на принципах зрелости экосистемы, активности сообщества и соответствия требованиям проекта.

#### 1.1.1 Язык программирования
**Python 3.8+**
- **Обоснование выбора**: Богатая экосистема научных вычислений, отличная поддержка компьютерного зрения
- **Основные преимущества**: Простота разработки, богатые библиотеки, активное сообщество
- **Версионная стратегия**: Поддержка Python 3.8 и выше для обеспечения совместимости

#### 1.1.2 GUI-фреймворк
**PyQt5**
- **Обоснование выбора**: Зрелый кроссплатформенный GUI-фреймворк с богатыми возможностями
- **Основные преимущества**: Нативный внешний вид, высокая производительность, полная функциональность
- **Альтернативы**: Tkinter (слишком простой), PyQt6 (проблемы совместимости), Kivy (не подходит для настольных приложений)

#### 1.1.3 Компьютерное зрение
**OpenCV 4.5+**
- **Обоснование выбора**: Ведущая библиотека компьютерного зрения с полными алгоритмами
- **Основные преимущества**: Высокая производительность, богатые алгоритмы, отличная документация
- **Ключевые модули**: cv2.threshold, cv2.Canny, cv2.findContours, cv2.contourArea

#### 1.1.4 Численные вычисления
**NumPy 1.19+**
- **Обоснование выбора**: Основа экосистемы научных вычислений Python
- **Основные преимущества**: Высокопроизводительные массивы, богатые математические функции
- **Роль в проекте**: Обработка данных изображений, статистические вычисления

#### 1.1.5 Визуализация данных
**Matplotlib 3.3+**
- **Обоснование выбора**: Стандартная библиотека визуализации Python
- **Основные преимущества**: Гибкие возможности настройки, поддержка множественных форматов вывода
- **Ключевые модули**: pyplot, figure, axes для создания статистических графиков

#### 1.1.6 Обработка изображений
**Pillow (PIL) 8.0+**
- **Обоснование выбора**: Стандартная библиотека обработки изображений Python
- **Основные преимущества**: Поддержка множественных форматов, простой API
- **Роль в проекте**: Загрузка изображений, преобразование форматов, базовые операции

### 1.2 Детальный анализ выбора технологий

#### 1.2.1 Анализ выбора языка программирования

**Python vs. C++**
- **Преимущества Python**: Быстрая разработка, богатые библиотеки, простота сопровождения
- **Преимущества C++**: Высокая производительность, контроль памяти
- **Решение**: Python + оптимизированные библиотеки (OpenCV с C++ ядром) обеспечивают баланс

**Python vs. Java**
- **Преимущества Python**: Превосходство в области научных вычислений, простота синтаксиса
- **Преимущества Java**: Кроссплатформенность, производительность
- **Решение**: Экосистема Python более подходит для задач компьютерного зрения

#### 1.2.2 Анализ выбора GUI-фреймворка

**PyQt5 vs. Tkinter**
- **Преимущества PyQt5**: Современный внешний вид, богатые виджеты, высокая производительность
- **Преимущества Tkinter**: Встроенность, простота
- **Решение**: PyQt5 лучше подходит для профессиональных приложений

**PyQt5 vs. Web-технологии (Electron)**
- **Преимущества PyQt5**: Нативная производительность, отсутствие зависимости от браузера
- **Преимущества Web**: Кроссплатформенность, современный UI
- **Решение**: Настольное приложение больше подходит для профессиональных инструментов

---

## 2. Проектирование слоистой архитектуры

### 2.1 Обзор архитектуры

Система принимает классическую четырехслойную архитектуру, обеспечивающую четкое разделение ответственности, высокую модульность и удобство сопровождения.

```
┌─────────────────────────────────────────────────────────────┐
│                    Слой представления                       │
│                 (Presentation Layer)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Главное   │  │  Диалоговые │  │    Компоненты       │ │
│  │    окно     │  │    окна     │  │   визуализации     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Слой бизнес-логики                       │
│                 (Business Logic Layer)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Управление │  │ Управление  │  │   Управление        │ │
│  │ изображениями│  │  анализом   │  │   отчетами          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Слой алгоритмических сервисов             │
│                (Algorithm Service Layer)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Обнаружение│  │Статистический│  │    Генерация        │ │
│  │   контуров  │  │   анализ    │  │    графиков         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Слой доступа к данным                     │
│                 (Data Access Layer)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Загрузка  │  │  Кэширование│  │     Экспорт         │ │
│  │ изображений │  │   данных    │  │    результатов      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Слой представления (Presentation Layer)

#### 2.2.1 Ответственности
- Отображение пользовательского интерфейса и взаимодействие с пользователем
- Обработка пользовательского ввода и событий
- Отображение результатов анализа и визуализация данных
- Управление состоянием интерфейса и пользовательским опытом

#### 2.2.2 Основные компоненты

**MainWindow (Главное окно)**
- Основной контейнер приложения
- Управление меню и панелями инструментов
- Координация между различными компонентами интерфейса

**ImageDisplayWidget (Виджет отображения изображений)**
- Отображение исходных изображений и результатов анализа
- Поддержка масштабирования и панорамирования
- Наложение аннотаций контуров

**AnalysisControlPanel (Панель управления анализом)**
- Настройка параметров анализа
- Управление процессом анализа
- Отображение прогресса выполнения

**StatisticsDisplayWidget (Виджет отображения статистики)**
- Отображение статистических результатов
- Интерактивные графики и диаграммы
- Экспорт статистических данных

#### 2.2.3 Паттерны проектирования
- **MVP (Model-View-Presenter)**: Разделение логики представления и бизнес-логики
- **Observer Pattern**: Автоматическое обновление интерфейса при изменении данных
- **Command Pattern**: Инкапсуляция пользовательских операций

### 2.3 Слой бизнес-логики (Business Logic Layer)

#### 2.3.1 Ответственности
- Координация рабочих процессов анализа
- Управление состоянием приложения
- Валидация пользовательского ввода
- Обработка бизнес-правил и логики

#### 2.3.2 Основные компоненты

**ImageManager (Менеджер изображений)**
```python
class ImageManager:
    def load_image(self, file_path: str) -> np.ndarray
    def validate_image(self, image: np.ndarray) -> bool
    def preprocess_image(self, image: np.ndarray) -> np.ndarray
    def get_image_info(self, image: np.ndarray) -> dict
```

**AnalysisController (Контроллер анализа)**
```python
class AnalysisController:
    def start_analysis(self, image: np.ndarray, params: dict) -> AnalysisResult
    def validate_parameters(self, params: dict) -> bool
    def monitor_progress(self) -> float
    def cancel_analysis(self) -> None
```

**ReportManager (Менеджер отчетов)**
```python
class ReportManager:
    def generate_report(self, analysis_result: AnalysisResult) -> Report
    def export_report(self, report: Report, format: str) -> str
    def get_report_templates(self) -> List[ReportTemplate]
```

#### 2.3.3 Паттерны проектирования
- **Facade Pattern**: Упрощение интерфейса для сложных подсистем
- **Strategy Pattern**: Различные стратегии анализа и генерации отчетов
- **Template Method**: Стандартизированные рабочие процессы

### 2.4 Слой алгоритмических сервисов (Algorithm Service Layer)

#### 2.4.1 Ответственности
- Реализация основных алгоритмов компьютерного зрения
- Статистический анализ и обработка данных
- Генерация визуализаций и графиков
- Оптимизация производительности алгоритмов

#### 2.4.2 Основные компоненты

**ContourDetector (Детектор контуров)**
```python
class ContourDetector:
    def detect_contours(self, image: np.ndarray, params: dict) -> List[Contour]
    def filter_contours(self, contours: List[Contour], min_area: float) -> List[Contour]
    def sort_contours(self, contours: List[Contour], method: str) -> List[Contour]
    def calculate_contour_properties(self, contour: Contour) -> ContourProperties
```

**StatisticalAnalyzer (Статистический анализатор)**
```python
class StatisticalAnalyzer:
    def calculate_basic_stats(self, areas: List[float]) -> BasicStatistics
    def generate_histogram(self, data: List[float], bins: int) -> Histogram
    def calculate_distribution(self, data: List[float]) -> Distribution
    def perform_size_classification(self, areas: List[float]) -> Classification
```

**VisualizationEngine (Движок визуализации)**
```python
class VisualizationEngine:
    def create_histogram_plot(self, data: List[float]) -> Figure
    def create_cumulative_plot(self, data: List[float]) -> Figure
    def create_size_distribution_plot(self, classification: Classification) -> Figure
    def annotate_contours(self, image: np.ndarray, contours: List[Contour]) -> np.ndarray
```

#### 2.4.3 Алгоритмические стратегии
- **Многоэтапное обнаружение**: Комбинация пороговой обработки и обнаружения границ Canny
- **Адаптивная фильтрация**: Динамическая настройка параметров на основе характеристик изображения
- **Оптимизированная обработка**: Использование векторизованных операций NumPy

### 2.5 Слой доступа к данным (Data Access Layer)

#### 2.5.1 Ответственности
- Загрузка и сохранение файлов изображений
- Кэширование промежуточных результатов
- Экспорт результатов анализа
- Управление конфигурацией приложения

#### 2.5.2 Основные компоненты

**FileHandler (Обработчик файлов)**
```python
class FileHandler:
    def load_image_file(self, file_path: str) -> np.ndarray
    def save_image_file(self, image: np.ndarray, file_path: str) -> bool
    def get_supported_formats(self) -> List[str]
    def validate_file_format(self, file_path: str) -> bool
```

**CacheManager (Менеджер кэша)**
```python
class CacheManager:
    def cache_analysis_result(self, key: str, result: AnalysisResult) -> None
    def get_cached_result(self, key: str) -> Optional[AnalysisResult]
    def clear_cache(self) -> None
    def get_cache_size(self) -> int
```

**ConfigurationManager (Менеджер конфигурации)**
```python
class ConfigurationManager:
    def load_config(self) -> Configuration
    def save_config(self, config: Configuration) -> None
    def get_default_config(self) -> Configuration
    def validate_config(self, config: Configuration) -> bool
```

---

## 3. Проектирование основных модулей

### 3.1 Модуль main.py

#### 3.1.1 Ответственности
- Инициализация приложения и настройка среды
- Создание основных объектов системы
- Обработка аргументов командной строки
- Управление жизненным циклом приложения

#### 3.1.2 Структура кода
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интеллектуальная система анализа гранулометрического состава щебня
Главный модуль запуска приложения
"""

import sys
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTranslator, QLocale

from app_ui import StoneAnalysisApp
from utils import setup_logging, load_configuration

def main():
    """Главная функция запуска приложения"""
    # Настройка логирования
    setup_logging()
    
    # Создание приложения Qt
    app = QApplication(sys.argv)
    
    # Настройка интернационализации
    setup_internationalization(app)
    
    # Загрузка конфигурации
    config = load_configuration()
    
    # Создание главного окна
    main_window = StoneAnalysisApp(config)
    main_window.show()
    
    # Запуск цикла событий
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
```

### 3.2 Модуль app_ui.py

#### 3.2.1 Ответственности
- Определение пользовательского интерфейса
- Обработка пользовательских событий
- Координация между различными компонентами UI
- Управление состоянием интерфейса

#### 3.2.2 Архитектурные принципы
- **Разделение ответственности**: UI-логика отделена от бизнес-логики
- **Модульность**: Каждый виджет инкапсулирует свою функциональность
- **Расширяемость**: Легкое добавление новых компонентов интерфейса
- **Тестируемость**: Четкие интерфейсы для модульного тестирования

### 3.3 Модуль image_analyzer.py

#### 3.3.1 Ответственности
- Реализация алгоритмов обнаружения контуров
- Статистический анализ результатов
- Оптимизация производительности обработки
- Обработка ошибок и исключительных ситуаций

#### 3.3.2 Ключевые алгоритмы

**Алгоритм обнаружения контуров**
```python
def perform_contour_analysis(image, canny_low=50, canny_high=150, min_area=100):
    """
    Выполнение анализа контуров с многоэтапной обработкой
    
    Args:
        image: Входное изображение
        canny_low: Нижний порог для обнаружения границ Canny
        canny_high: Верхний порог для обнаружения границ Canny
        min_area: Минимальная площадь контура
    
    Returns:
        Результаты анализа контуров
    """
    # Этап 1: Предварительная обработка
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Этап 2: Пороговая обработка
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Этап 3: Обнаружение границ Canny
    edges = cv2.Canny(thresh, canny_low, canny_high)
    
    # Этап 4: Поиск контуров
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Этап 5: Фильтрация и анализ
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    return analyze_contours(filtered_contours)
```

### 3.4 Модуль utils.py

#### 3.4.1 Ответственности
- Предоставление вспомогательных функций
- Управление конфигурацией системы
- Утилиты для обработки данных
- Общие инструменты и константы

#### 3.4.2 Основные категории утилит
- **Утилиты файловой системы**: Операции с файлами и путями
- **Утилиты обработки данных**: Преобразование и валидация данных
- **Утилиты логирования**: Настройка и управление логами
- **Утилиты конфигурации**: Загрузка и сохранение настроек

---

## 4. Проектирование межмодульного взаимодействия

### 4.1 Паттерн Observer + Механизм сигналов-слотов

#### 4.1.1 Обоснование выбора
Использование комбинации паттерна Observer и механизма сигналов-слотов PyQt5 обеспечивает:
- **Слабую связанность**: Модули не зависят напрямую друг от друга
- **Асинхронность**: Неблокирующее взаимодействие между компонентами
- **Расширяемость**: Легкое добавление новых обработчиков событий
- **Тестируемость**: Возможность изолированного тестирования модулей

#### 4.1.2 Схема взаимодействия

```
┌─────────────────┐    сигналы     ┌─────────────────┐
│   UI-компоненты │ ──────────────► │ Контроллеры     │
│                 │                │ бизнес-логики   │
└─────────────────┘                └─────────────────┘
         ▲                                   │
         │                                   ▼
         │ обновления                ┌─────────────────┐
         │ интерфейса                │ Алгоритмические │
         │                           │    сервисы      │
         │                           └─────────────────┘
         │                                   │
         │                                   ▼
┌─────────────────┐    события     ┌─────────────────┐
│ Менеджер событий│ ◄────────────── │ Слой доступа    │
│   (EventBus)    │                │   к данным      │
└─────────────────┘                └─────────────────┘
```

#### 4.1.3 Реализация EventBus

```python
class EventBus(QObject):
    """Центральная шина событий для межмодульного взаимодействия"""
    
    # Определение сигналов
    image_loaded = pyqtSignal(np.ndarray, str)  # изображение, путь к файлу
    analysis_started = pyqtSignal(dict)         # параметры анализа
    analysis_progress = pyqtSignal(int)         # прогресс в процентах
    analysis_completed = pyqtSignal(object)     # результаты анализа
    analysis_error = pyqtSignal(str)            # сообщение об ошибке
    
    def __init__(self):
        super().__init__()
        self._instance = None
    
    @classmethod
    def get_instance(cls):
        """Singleton паттерн для глобального доступа"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

### 4.2 Интерфейсы модулей

#### 4.2.1 Интерфейс анализатора изображений

```python
from abc import ABC, abstractmethod

class ImageAnalyzerInterface(ABC):
    """Абстрактный интерфейс для анализаторов изображений"""
    
    @abstractmethod
    def analyze(self, image: np.ndarray, parameters: dict) -> AnalysisResult:
        """Выполнение анализа изображения"""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: dict) -> bool:
        """Валидация параметров анализа"""
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> dict:
        """Получение параметров по умолчанию"""
        pass
```

#### 4.2.2 Интерфейс генератора отчетов

```python
class ReportGeneratorInterface(ABC):
    """Абстрактный интерфейс для генераторов отчетов"""
    
    @abstractmethod
    def generate(self, analysis_result: AnalysisResult, template: str) -> Report:
        """Генерация отчета"""
        pass
    
    @abstractmethod
    def export(self, report: Report, format: str, file_path: str) -> bool:
        """Экспорт отчета в файл"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Получение поддерживаемых форматов"""
        pass
```

### 4.3 Управление зависимостями

#### 4.3.1 Dependency Injection Container

```python
class DIContainer:
    """Контейнер для управления зависимостями"""
    
    def __init__(self):
        self._services = {}
        self._singletons = {}
    
    def register(self, interface, implementation, singleton=False):
        """Регистрация сервиса"""
        self._services[interface] = {
            'implementation': implementation,
            'singleton': singleton
        }
    
    def resolve(self, interface):
        """Разрешение зависимости"""
        if interface not in self._services:
            raise ValueError(f"Service {interface} not registered")
        
        service_info = self._services[interface]
        
        if service_info['singleton']:
            if interface not in self._singletons:
                self._singletons[interface] = service_info['implementation']()
            return self._singletons[interface]
        
        return service_info['implementation']()
```

---

## 5. Ключевые технические решения

### 5.1 Алгоритм обнаружения контуров

#### 5.1.1 Многоэтапная стратегия обработки

**Этап 1: Предварительная обработка**
- Преобразование в оттенки серого
- Шумоподавление (при необходимости)
- Нормализация контраста

**Этап 2: Пороговая обработка**
- Использование адаптивной пороговой обработки Otsu
- Автоматическое определение оптимального порога
- Бинаризация изображения

**Этап 3: Обнаружение границ**
- Применение алгоритма Canny с настраиваемыми порогами
- Двойная пороговая обработка для подавления шума
- Связывание границ по гистерезису

**Этап 4: Поиск и фильтрация контуров**
- Использование алгоритма Suzuki-Abe для поиска контуров
- Фильтрация по минимальной площади
- Удаление вложенных контуров

#### 5.1.2 Оптимизация производительности

```python
def optimized_contour_detection(image, params):
    """Оптимизированное обнаружение контуров"""
    
    # Предварительная проверка размера изображения
    if image.shape[0] * image.shape[1] > MAX_IMAGE_SIZE:
        # Масштабирование для ускорения обработки
        scale_factor = calculate_scale_factor(image.shape)
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    
    # Векторизованная обработка с NumPy
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Кэширование промежуточных результатов
    cache_key = generate_cache_key(image, params)
    if cache_key in contour_cache:
        return contour_cache[cache_key]
    
    # Основная обработка
    result = perform_contour_analysis(gray, **params)
    
    # Сохранение в кэш
    contour_cache[cache_key] = result
    
    return result
```

### 5.2 Обработка ошибок и система логирования

#### 5.2.1 Иерархия исключений

```python
class StoneAnalysisException(Exception):
    """Базовое исключение для системы анализа"""
    pass

class ImageProcessingError(StoneAnalysisException):
    """Ошибки обработки изображений"""
    pass

class AnalysisParameterError(StoneAnalysisException):
    """Ошибки параметров анализа"""
    pass

class FileOperationError(StoneAnalysisException):
    """Ошибки файловых операций"""
    pass
```

#### 5.2.2 Система логирования

```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Настройка системы логирования"""
    
    # Создание форматтера
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Настройка файлового логгера
    file_handler = RotatingFileHandler(
        'stone_analysis.log', 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Настройка консольного логгера
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
```

### 5.3 Система конфигурации

#### 5.3.1 Структура конфигурации

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AnalysisConfig:
    """Конфигурация параметров анализа"""
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    min_contour_area: float = 100.0
    max_contour_area: float = 10000.0
    gaussian_blur_kernel: int = 5

@dataclass
class UIConfig:
    """Конфигурация пользовательского интерфейса"""
    language: str = 'ru'
    theme: str = 'default'
    window_width: int = 1200
    window_height: int = 800
    auto_save_results: bool = True

@dataclass
class SystemConfig:
    """Общая конфигурация системы"""
    analysis: AnalysisConfig
    ui: UIConfig
    cache_size_mb: int = 100
    max_image_size_mb: int = 50
    temp_directory: str = './temp'
```

---

## 6. Схема развертывания и эксплуатации

### 6.1 Стратегия упаковки

#### 6.1.1 PyInstaller для создания исполняемых файлов

```python
# build.spec
a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('assets/*', 'assets'),
        ('config/*', 'config'),
        ('docs/*', 'docs')
    ],
    hiddenimports=[
        'PyQt5.QtCore',
        'PyQt5.QtGui', 
        'PyQt5.QtWidgets',
        'cv2',
        'numpy',
        'matplotlib'
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='StoneAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon='assets/icon.ico'
)
```

#### 6.1.2 Управление версиями

```python
# version.py
VERSION_MAJOR = 1
VERSION_MINOR = 0
VERSION_PATCH = 0
VERSION_BUILD = 'release'

VERSION_STRING = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}-{VERSION_BUILD}"

def get_version_info():
    """Получение информации о версии"""
    return {
        'version': VERSION_STRING,
        'major': VERSION_MAJOR,
        'minor': VERSION_MINOR,
        'patch': VERSION_PATCH,
        'build': VERSION_BUILD
    }
```

### 6.2 Обеспечение качества

#### 6.2.1 Стратегия тестирования

**Модульные тесты**
```python
import unittest
import numpy as np
from image_analyzer import perform_contour_analysis

class TestContourAnalysis(unittest.TestCase):
    
    def setUp(self):
        """Настройка тестовых данных"""
        self.test_image = self.create_test_image()
        self.default_params = {
            'canny_low': 50,
            'canny_high': 150,
            'min_area': 100
        }
    
    def test_contour_detection_basic(self):
        """Тест базового обнаружения контуров"""
        result = perform_contour_analysis(self.test_image, **self.default_params)
        
        self.assertIsNotNone(result)
        self.assertGreater(len(result['contours']), 0)
        self.assertIn('statistics', result)
    
    def test_parameter_validation(self):
        """Тест валидации параметров"""
        invalid_params = {'canny_low': -1, 'canny_high': 300, 'min_area': -10}
        
        with self.assertRaises(ValueError):
            perform_contour_analysis(self.test_image, **invalid_params)
```

**Интеграционные тесты**
```python
class TestSystemIntegration(unittest.TestCase):
    
    def test_full_analysis_workflow(self):
        """Тест полного рабочего процесса анализа"""
        # Загрузка изображения
        image_path = 'test_data/sample_image.jpg'
        image = load_image(image_path)
        
        # Выполнение анализа
        result = perform_full_analysis(image)
        
        # Проверка результатов
        self.assertIn('contours', result)
        self.assertIn('statistics', result)
        self.assertIn('visualization', result)
        
        # Генерация отчета
        report = generate_report(result)
        self.assertIsNotNone(report)
```

#### 6.2.2 Непрерывная интеграция

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: windows-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ --cov=src/ --cov-report=xml
    
    - name: Run linting
      run: |
        flake8 src/ tests/
        pylint src/
    
    - name: Build executable
      run: |
        pyinstaller build.spec
```

---

## 7. Проектирование расширяемости и сопровождения

### 7.1 Архитектура плагинов

#### 7.1.1 Интерфейс плагинов

```python
from abc import ABC, abstractmethod

class AnalysisPlugin(ABC):
    """Базовый интерфейс для плагинов анализа"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Получение имени плагина"""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Получение версии плагина"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Получение описания плагина"""
        pass
    
    @abstractmethod
    def analyze(self, image: np.ndarray, params: dict) -> dict:
        """Выполнение анализа"""
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> dict:
        """Получение параметров по умолчанию"""
        pass
```

#### 7.1.2 Менеджер плагинов

```python
class PluginManager:
    """Менеджер для управления плагинами"""
    
    def __init__(self):
        self._plugins = {}
        self._plugin_directory = './plugins'
    
    def load_plugins(self):
        """Загрузка всех плагинов из директории"""
        plugin_files = glob.glob(os.path.join(self._plugin_directory, '*.py'))
        
        for plugin_file in plugin_files:
            try:
                plugin = self._load_plugin_from_file(plugin_file)
                self._plugins[plugin.get_name()] = plugin
                logging.info(f"Loaded plugin: {plugin.get_name()}")
            except Exception as e:
                logging.error(f"Failed to load plugin {plugin_file}: {e}")
    
    def get_plugin(self, name: str) -> AnalysisPlugin:
        """Получение плагина по имени"""
        return self._plugins.get(name)
    
    def get_available_plugins(self) -> List[str]:
        """Получение списка доступных плагинов"""
        return list(self._plugins.keys())
```

### 7.2 Система управления конфигурацией

#### 7.2.1 Конфигурационные файлы

```yaml
# config/default.yaml
analysis:
  contour_detection:
    canny_low_threshold: 50
    canny_high_threshold: 150
    min_contour_area: 100.0
    gaussian_blur_kernel: 5
  
  statistical_analysis:
    histogram_bins: 20
    size_classification_ranges:
      - [0, 25]      # Мелкая фракция
      - [25, 50]     # Средняя фракция  
      - [50, 100]    # Крупная фракция

ui:
  language: 'ru'
  theme: 'default'
  window:
    width: 1200
    height: 800
    resizable: true
  
  auto_save:
    enabled: true
    interval_minutes: 5

system:
  cache:
    size_mb: 100
    cleanup_interval_hours: 24
  
  logging:
    level: 'INFO'
    file_size_mb: 10
    backup_count: 5
  
  performance:
    max_image_size_mb: 50
    parallel_processing: true
    thread_count: 4
```

#### 7.2.2 Динамическая конфигурация

```python
class ConfigurationManager:
    """Менеджер динамической конфигурации"""
    
    def __init__(self):
        self._config = {}
        self._watchers = []
        self._config_file = 'config/user.yaml'
    
    def load_configuration(self):
        """Загрузка конфигурации из файлов"""
        # Загрузка базовой конфигурации
        base_config = self._load_yaml('config/default.yaml')
        
        # Загрузка пользовательской конфигурации
        user_config = self._load_yaml(self._config_file)
        
        # Слияние конфигураций
        self._config = self._merge_configs(base_config, user_config)
    
    def get(self, key_path: str, default=None):
        """Получение значения конфигурации по пути"""
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Установка значения конфигурации"""
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self._notify_watchers(key_path, value)
    
    def watch(self, key_path: str, callback):
        """Подписка на изменения конфигурации"""
        self._watchers.append((key_path, callback))
```

