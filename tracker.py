import csv
import os
from tkinter import filedialog, messagebox
import cv2
from matplotlib import pyplot as plt
import mediapipe as mp
from statistics import mean
import numpy as np
import time
from datetime import datetime
from collections import deque
import pandas as pd
from scipy.spatial import distance as dist
import colorama
import seaborn as sns

# 
import random
from enum import Enum
import tkinter as tk

# Перечисление размеров кнопок (используется Enum для создания именованных констант)
class Length(Enum):
    MIN = 25  # Минимальный размер 25 пикселей
    SMALL = 50  # Маленький размер 50 пикселей
    MEDIUM = 100  # Средний размер 100 пикселей
    LARGE = 150  # Большой размер 150 пикселей
    EXTRA_LARGE = 200  # Очень большой размер 200 пикселей
    MAX = 250  # Максимальный размер 250 пикселей

class EyeTracker:
    def __init__(self, camera=0, res="1280x1080", cam_res=None, fixation=False, blink=False, fast=True, folder_path=None,file_path=None, duration=5, ma_trshhd=0.65, ma_width=5, exp_koef=0.5):
        
        colorama.init()  # Инициализируем colorama

        # Переменная главного цикла
        self.window_holder = True

        # Для обработки нажатых клавиш
        self.key = None

        # Ширина и высота окна
        self.screen_w, self.screen_h = map(int, res.split('x'))

        self.cam = self.open_camera(camera, fast, cam_res)
        self.log(f"Установленное разрешение вебкамеры: {self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)}", "INFO", "tracker")

        # Модель FaceMash от Mediapipe
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        
        # Переменный путей
        self.folder_path = folder_path
        self.file_path = file_path

        # Служебные переменные-флаги
        self.img_read = False
        self.calibrated = False

        # Счетчик имени папки
        self.save_counter = 0
        self.save_folder_name = "PercepTest0"
        
        # Переменная продолжительности испытания
        self.duration = duration

        # Калибровка и другие переменные
        self.frame_center_x = self.screen_w // 2
        self.frame_center_y = self.screen_h // 2
        self.frame_left_top_corner_x = 0
        self.frame_left_top_corner_y = 0
        self.frame_right_bottom_corner_x = self.screen_w
        self.frame_right_bottom_corner_y = self.screen_h
        self.focused = 0
        self.left_top_corner_x = None
        self.left_top_corner_y = None
        self.right_bottom_corner_x = None
        self.right_bottom_corner_y = None
        self.nose_calib_x = []
        self.nose_calib_y = []
        self.nose_calib_mean_x = None
        self.nose_calib_mean_x = None

        # Параметры окружности зоны взгляда
        # self.gaze_area_color = (0, 255, 0)
        self.gaze_area_color= (200, 200, 200)
        self.gaze_area_border = 2
                
        # Индексы точек глаз для определения морганий
        self.left_eye_indices = [468, 33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [473, 362, 385, 387, 263, 373, 380]
        self.left_eye_center = None
        self.right_eye_center = None
        self.nose = None

        # Очередь для хранения последних позиций взгляда для сглаживания (МА)
        self.smooth_x = deque(maxlen=ma_width)
        self.smooth_y = deque(maxlen=ma_width)
        self.smooth_threshold = ma_trshhd
        self.smooth_frame_pos_x = None
        self.smooth_frame_pos_y = None

        # Параметры для сглаживания перемещения точки взгляда
        self.alpha = exp_koef  # Коэффициент сглаживания 35
        self.smooth_frame_pos_x_exp = None
        self.smooth_frame_pos_y_exp = None

        # Параметры для радиуса круга взгляда
        self.min_radius = 50           # Минимальный радиус круга 10
        self.max_radius = 120           # Максимальный радиус круга

        
        # PerceptTest vars
        self.running = False  # Флаг работы программы
        self.paused = False  # Флаг паузы
        self.waiting = False  # Флаг ожидания
        self.time_pause = 0  # Время паузы
        self.start_wait = 0  # Время начала ожидания
        self.temp_frame = None
        self.esc = False
        
        # Переменные для цветов и фигур
        self.i_color_button = int(random.randint(0, 6))  # Индекс текущего цвета кнопки (случайный)
        self.i_color_background = 0  # Индекс текущего цвета фона (по умолчанию - первый цвет)
        self.figure_type = int(random.randint(1, 5))  # Тип фигуры, которая будет отрисовываться (случайный)
        self.figure_list = []  # Список для хранения данных о нажатиях (реакциях пользователя)

        # Переменные для размеров кнопок
        self.button_width, self.button_height = 0, 0  # Размеры кнопки (будут установлены позже)
        self.button_x, self.button_y = 0, 0  # Координаты кнопки (будут установлены позже)

        # Цвета кнопок и фона (в BGR формате для OpenCV)
        # self.name_color_array_button = ["Красный", "Оранжевый", "Жёлтый", "Зелёный", "Синий", "Индиго", "Фиолетовый", "Чёрный", "Белый"]  # Названия цветов кнопок
        self.name_color_array_button = ["Красный", "Оранжевый", "Жёлтый", "Зелёный", "Синий", "Индиго", "Фиолетовый"]  # Названия цветов кнопок
        self.name_color_array_background = ["Чёрный", "Белый"]  # Названия цветов фона
        # OpenCV использует BGR формат вместо RGB
        # self.array_colors_button = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 128, 0), (255, 0, 0), (130, 0, 75), (128, 0, 128), (0, 0, 0), (255, 255, 255)]  # BGR значения цветов кнопок
        self.array_colors_button = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 128, 0), (255, 0, 0), (130, 0, 75), (128, 0, 128)]  # BGR значения цветов кнопок
        self.array_colors_background = [(0, 0, 0), (255, 255, 255)]  # BGR значения цветов фона
        
        self.time_logs = []  # Список для журналов времени нажатий

        # Словарь для фигур - связывает номер типа фигуры с его названием
        self.figure_names = {1: "Эллипс", 2: "Круг", 3: "Ромб", 4: "Прямоугольник", 5: "Квадрат"}

        # Генерация начальных координат и размеров
        self.button_width, self.button_height = self.get_random_dimensions()  # Получаем случайные размеры кнопки
        # Вычисляем случайные координаты кнопки (с учетом её размеров, чтобы она не выходила за границы окна)
        self.button_x, self.button_y = random.randint(self.button_width // 2, self.screen_w - self.button_width // 2), random.randint(self.button_height // 2, self.screen_h - self.button_height // 2)
        

    def show_loader(self):
        """Функция для отображения экрана загрузки"""
        width, height = self.screen_w, self.screen_h
        loader_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        text = "loading, wait..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height - text_size[1]) // 2
        
        cv2.putText(loader_img, text, (text_x, text_y+20), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Eye tracker', loader_img)
        cv2.waitKey(1)  # Нужно, чтобы окно обновилось
    
    def open_camera(self, c, f, cam_res):
        """Функция для открытия камеры"""
        self.show_loader()  # Показываем загрузочный экран
        
        # Режим подключения к камере
        if f:
            cam = cv2.VideoCapture(c, cv2.CAP_DSHOW)
            if cam_res != None:
                w, h = map(int, cam_res.split('x'))
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        else:
            cam = cv2.VideoCapture(c)
            if cam_res != None:
                w, h = map(int, cam_res.split('x'))
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        start_time = time.time()
        while not cam.isOpened():
            if time.time() - start_time > 100:  # Таймаут 100 секунд
                self.log(f"Ошибка подключения к камере... (timeout)", "ERROR", "tracker")
                cv2.destroyWindow('Eye tracker')
                return
        
        # cv2.destroyWindow("Loader window")  # Закрываем загрузочный экран
        return cam
    

    # Функция отрисовки точек, полученных от модели FaceMesh Mediapipe
    def draw_landmarks(self, landmarks, frame_w, frame_h):
        
        left_eye_landmarks = []
        right_eye_landmarks = []
        
        if not self.running:
            # Извлекаем координаты левого глаза EAR
            for idx in self.left_eye_indices:
                x = landmarks[idx].x * frame_w
                y = landmarks[idx].y * frame_h
                if idx == 468:
                    cv2.circle(self.frame, (int(x), int(y)), 2, (0, 0, 255))
                else:
                    left_eye_landmarks.append((x, y))
                    cv2.circle(self.frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            # Извлекаем координаты правого глаза EAR
            for idx in self.right_eye_indices:
                x = landmarks[idx].x * frame_w
                y = landmarks[idx].y * frame_h
                if idx == 473:
                    cv2.circle(self.frame, (int(x), int(y)), 2, (0, 0, 255))
                else:
                    right_eye_landmarks.append((x, y))
                    cv2.circle(self.frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        # Рисуем кончик носа
        self.nose = landmarks[1]
        cv2.circle(self.frame, (int(self.nose.x * frame_w), int(self.nose.y * frame_h)), 2, (0, 255, 255), 2)

        self.left_eye_center, self.right_eye_center = landmarks[468], landmarks[473]

        return left_eye_landmarks, right_eye_landmarks
    

    # Функция калибровки трекера по взгляду пользователя
    def calibrate(self, frame_w, frame_h):

        cv2.circle(self.frame, (self.frame_center_x, self.frame_center_y), 2, (0, 0, 255), 2)

        # Калибровка по центру
        if self.focused == 0:
            cv2.circle(self.frame, (self.frame_center_x, self.frame_center_y), 10, (0, 0, 255), 10)
            cv2.putText(self.frame, "look at CENTER and press SPACE", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 0), 2)
        
            if self.key == ord(' '):
                self.focused += 1
        
        # Калибровка по левому верхнему углу
        elif self.focused == 1:
            cv2.circle(self.frame, (self.frame_left_top_corner_x+10, self.frame_left_top_corner_y+10), 10, (0, 0, 255), 10)
            cv2.putText(self.frame, "look at UPPER LEFT and press SPACE", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 0), 2)
            
            if self.key == ord(' '):
                self.left_top_corner_x, self.left_top_corner_y = self.calibrate_coords("Координаты глаз при калибровке по левому верхнему углу: ")
        
        # Калибровка по правому нижнему углу
        elif self.focused == 2:
            cv2.circle(self.frame, (self.frame_right_bottom_corner_x-10, self.frame_right_bottom_corner_y-10), 10, (0, 0, 255), 10)
            cv2.putText(self.frame, "look at BOTTOM RIGHT and press SPACE", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 0), 2)
            
            if self.key == ord(' '):
                self.right_bottom_corner_x, self.right_bottom_corner_y = self.calibrate_coords("Координаты глаз при калибровке по правому нижниму углу: ")
                # Точка калибровки носа
                self.nose_calib_mean_x = int(mean(self.nose_calib_x) * frame_w)
                self.nose_calib_mean_y = int(mean(self.nose_calib_y) * frame_h)

                self.calibrated = True
        

    # Получение координат зрачков и из рассчет (для калибровки)
    def calibrate_coords(self, text):
        pulp_mid_x = (self.left_eye_center.x + self.right_eye_center.x) / 2
        pulp_mid_y = (self.left_eye_center.y + self.right_eye_center.y) / 2
        self.nose_calib_x.append(self.nose.x)
        self.nose_calib_y.append(self.nose.y)
        self.log(f"{text} сохранены", "INFO", "tracker")
        self.focused += 1
        return pulp_mid_x, pulp_mid_y
    
    # Функция рассчета координат взгляда
    def calculate_gaze(self, frame_w, frame_h):
            # Получение координат глаз в кадре
            pos_x = (self.left_eye_center.x + self.right_eye_center.x) / 2
            pos_y = (self.left_eye_center.y + self.right_eye_center.y) / 2

            # Рассчет координат взгляда
            frame_pos_x = (pos_x - self.left_top_corner_x) * self.frame_right_bottom_corner_x / (self.right_bottom_corner_x - self.left_top_corner_x)
            frame_pos_y = (pos_y - self.left_top_corner_y) * self.frame_right_bottom_corner_y / (self.right_bottom_corner_y - self.left_top_corner_y)

            # Заполнение очередей для сглаживания
            if len(self.smooth_x) == self.smooth_x.maxlen:
                median_x = mean(self.smooth_x)
                median_y = mean(self.smooth_y)
                if (abs(frame_pos_x - median_x) > self.frame_right_bottom_corner_x * self.smooth_threshold) or \
                (abs(frame_pos_y - median_y) > self.frame_right_bottom_corner_y * self.smooth_threshold):
                    # Очистить очередь при превышении порога
                    self.smooth_x.clear()
                    self.smooth_y.clear()
                self.smooth_x.append(frame_pos_x)
                self.smooth_y.append(frame_pos_y)
            else:
                # Добавляем текущие значения в очередь
                self.smooth_x.append(frame_pos_x)
                self.smooth_y.append(frame_pos_y)

            # Рассчитываем усредненные значения взгляда (МА)
            self.smooth_frame_pos_x = int(mean(self.smooth_x))
            self.smooth_frame_pos_y = int(mean(self.smooth_y))

            # Применяем экспоненциальное сглаживание
            if self.smooth_frame_pos_x_exp is None:
                self.smooth_frame_pos_x_exp = self.smooth_frame_pos_x
                self.smooth_frame_pos_y_exp = self.smooth_frame_pos_y
            else:
                self.smooth_frame_pos_x_exp = int(self.alpha * self.smooth_frame_pos_x + (1 - self.alpha) * self.smooth_frame_pos_x_exp)
                self.smooth_frame_pos_y_exp = int(self.alpha * self.smooth_frame_pos_y + (1 - self.alpha) * self.smooth_frame_pos_y_exp)

            # Логика расчета радиуса
            queue_length = len(self.smooth_x)  # Текущая длина очереди
            max_queue_length = self.smooth_x.maxlen  # Максимальная длина очереди

            if queue_length == 0:
                # Если очередь только что сброшена
                self.radius = self.max_radius
            elif queue_length < max_queue_length:
                # Плавное уменьшение радиуса от max_radius до min_radius
                progress = queue_length / max_queue_length  # Доля заполнения очереди (0..1)
                self.radius = int(self.max_radius - (self.max_radius - self.min_radius) * progress)
            else:
                # Очередь заполнена, рассчитываем радиус по разбросу
                std_x = np.std(self.smooth_x)
                std_y = np.std(self.smooth_y)
                spread = np.sqrt(std_x**2 + std_y**2)
                radius = int(spread / 2)
                self.radius = max(self.min_radius, min(radius, self.max_radius))

            # Применяем экспоненциально сглаженные координаты
            self.smooth_frame_pos_x = self.smooth_frame_pos_x_exp
            self.smooth_frame_pos_y = self.smooth_frame_pos_y_exp

            # Создаем полупрозрачный круг
            overlay = self.frame.copy()  # Копия кадра для наложения
             # Рисуем круг на фрейме
            cv2.circle(self.frame, (self.smooth_frame_pos_x, self.smooth_frame_pos_y), self.radius, self.gaze_area_color, self.gaze_area_border)

            # Накладываем круг с прозрачностью 30% (alpha = 0.3 для круга, beta = 0.7 для фона)
            alpha = 0.05  # 30% прозрачности для круга
            beta = 1.0 - alpha  # 70% для фона
            cv2.addWeighted(overlay, alpha, self.frame, beta, 0.0, self.frame)

           


    # Функция рассчета указателя зоны взгляда (которая ушла за пределы видимой зоны окна)
    def draw_gaze_direction(self, gaze_x, gaze_y):

        # Центр экрана
        center_x = self.frame_center_x
        center_y = self.frame_center_y

        # Проверка, находится ли расчетная точка взгляда за границами экрана
        if 0 <= gaze_x <= self.frame_right_bottom_corner_x and 0 <= gaze_y <= self.frame_right_bottom_corner_y:
            return  # Точка внутри экрана, выход

        # Определяем разницу по X и по Y от центра экрана
        dx = gaze_x - center_x
        dy = gaze_y - center_y

        # Определяем крутость наклона луча из центра до точки взгляда (отношение изменения ординат к изменению абсцисс)
        if dx != 0:
            slope = dy / dx
        else:
            slope = float('inf')  # Если dx = 0, линия вертикальная

        # Взгляд направлен в право
        if dx > 0:
            intersect_x = self.frame_right_bottom_corner_x                                  # Пересечение с правой границей
            intersect_y = center_y + slope * (self.frame_right_bottom_corner_x - center_x)  # Рассчет высоты пересечения
        else:
            intersect_x = 0                                     # Пересечение с левой границей
            intersect_y = center_y + slope * (0 - center_x)     # Рассчет высоты пересечения

        # Проверяем, если пересечение по Y выходит за верх или низ экрана
        if not (0 <= intersect_y <= self.frame_right_bottom_corner_y):
            if dy > 0:
                intersect_y = self.frame_right_bottom_corner_y  # Пересечение с нижней границей
                intersect_x = center_x + (self.frame_right_bottom_corner_y - center_y) / slope
            else:
                intersect_y = 0                                 # Пересечение с верхней границей
                intersect_x = center_x + (0 - center_y) / slope

        # Рассчитываем расстояние от расчетной точки до границы экрана
        distance_to_border = np.sqrt((gaze_x - intersect_x) ** 2 + (gaze_y - intersect_y) ** 2)

        # Перестроение радиуса точки в зависимости от расстояния до границы
        max_distance = 3500
        min_radius = 10
        max_radius = 50
        radius = max_radius - (max_radius - min_radius) * min(distance_to_border, max_distance) / max_distance
        radius = int(radius)  # Преобразуем в целое число для рисования

        # Рисуем точку на пересечении границы экрана
        cv2.circle(self.frame, (int(intersect_x), int(intersect_y)), radius, (0, 0, 255), -1)


    
    def log(self, message, level="INFO", source="NA"):
        """Функция для красивого вывода логов"""
        colors = {
            "INFO": colorama.Fore.CYAN,   # Голубой
            "WARNING": colorama.Fore.YELLOW,  # Желтый
            "ERROR": colorama.Fore.RED,   # Красный
            "SUCCESS": colorama.Fore.GREEN,  # Зеленый
            "TESTING": colorama.Fore.MAGENTA,  # Фиолетовый
        }
        
        now = datetime.now().strftime("%H:%M:%S")  # Получаем текущее время (часы:минуты:секунды)
        color = colors.get(level, colorama.Fore.WHITE)  # Получаем цвет (по умолчанию белый)
        
        print(f"{color}[{now}] [{level}] ({source}) {message}{colorama.Style.RESET_ALL}")

    
    # БЛОК PERCEPTEST CV 
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def get_random_dimensions(self):
        """ Генерирует случайные размеры кнопки из предопределенных значений. """
        # Выбираем случайное значение из перечисления Length и возвращаем его как размер кнопки
        # random.choice выбирает случайный элемент из списка
        # list(Length) преобразует перечисление в список его элементов
        # value извлекает числовое значение из выбранного элемента перечисления
        return random.choice(list(Length)).value, random.choice(list(Length)).value

    def draw_figure(self, img, x, y, w, h, shape_type):
        """ Отрисовывает фигуру на экране в заданных координатах и с заданными размерами. """
        # Получаем цвет для фигуры из массива цветов по текущему индексу
        color = self.array_colors_button[self.i_color_button]
        
        # В зависимости от типа фигуры вызываем соответствующий метод отрисовки OpenCV
        if shape_type == 1:
            # Эллипс - задаем центр (x, y) и половинные размеры (w/2, h/2), угол поворота 0, начальный и конечный углы 0 и 360
            cv2.ellipse(img, (x, y), (w//2, h//2), 0, 0, 360, color, -1)  # -1 означает заполненную фигуру
        elif shape_type == 2:
            # Круг - задаем центр (x, y) и радиус, равный половине минимального из ширины и высоты
            cv2.circle(img, (x, y), min(w, h) // 2, color, -1)
        elif shape_type == 3:
            # Ромб - задаем четыре точки - вверх, вправо, вниз, влево от центра
            points = np.array([(x, y + h // 2), (x + w // 2, y), (x, y - h // 2), (x - w // 2, y)], np.int32)
            # Преобразуем массив точек в формат, требуемый для fillPoly
            points = points.reshape((-1, 1, 2))
            # Закрашиваем многоугольник по заданным точкам
            cv2.fillPoly(img, [points], color)
        elif shape_type == 4:
            # Прямоугольник - задаем верхний левый угол (x-w/2, y-h/2) и нижний правый угол (x+w/2, y+h/2)
            cv2.rectangle(img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, -1)
        elif shape_type == 5:
            # Квадрат - аналогично прямоугольнику, но стороны равны (используем минимальное из ширины и высоты)
            size = min(w, h)
            cv2.rectangle(img, (x - size // 2, y - size // 2), (x + size // 2, y + size // 2), color, -1)
        # Возвращаем изображение с нарисованной фигурой
        return img

    def get_random_value(self):
        """ Выбирает случайные значения для цвета кнопки, цвета фона и типа фигуры. """
        # Выбираем случайный цвет фона и его индекс
        # enumerate создает пары (индекс, элемент) из списка
        # random.choice выбирает случайную пару из списка
        background_index, background_value = random.choice(list(enumerate(self.name_color_array_background)))

        # Убираем этот цвет из списка цветов кнопки, если он есть
        # Это нужно, чтобы цвет кнопки отличался от цвета фона
        filtered_colors = [color for color in self.name_color_array_button if color != background_value]

        # Выбираем случайный цвет кнопки из оставшихся
        button_value = random.choice(filtered_colors)
        # Получаем индекс выбранного цвета в исходном списке цветов кнопки
        button_index = self.name_color_array_button.index(button_value)

        # Возвращаем индексы цветов и случайный тип фигуры (от 1 до 5)
        return button_index, background_index, random.choice([1, 2, 3, 4, 5])

    def save_to_csv(self, data):
        """ Сохраняет данные в CSV-файл. """

        # Определение номера папки для испытания
        while os.path.exists(os.path.join(self.folder_path, self.save_folder_name)):
            self.save_counter += 1
            self.save_folder_name = f"PercepTest{self.save_counter}"
            
        # Создаём новую папку под испытание
        os.makedirs(os.path.join(self.folder_path, self.save_folder_name))
        filename = f'{self.folder_path}/{self.save_folder_name}/perceptest_data.csv'
        try:
            # Открываем файл для добавления ('a') в кодировке utf-8
            with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
                # Определяем заголовки колонок для CSV файла
                fieldnames = ['t', 'color', 'figure', 'background', 'size_h', 'size_w']
                # Создаем объект DictWriter, который записывает словари в формате CSV
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Если файл пустой (позиция указателя 0), записываем заголовок
                if csvfile.tell() == 0:
                    writer.writeheader()
                
                # Записываем все записи данных
                writer.writerows(data)
            
            # Выводим сообщение об успешном сохранении
            self.log(f"Данные сохранены в {filename}", "SUCCESS", "PercepTest")
        except IOError:
            # В случае ошибки выводим сообщение
            self.log("Ошибка при записи в файл csv", "ERROR", "PercepTest")


    def plot_histogram(self, data, xlabel, title):
        """ Строит гистограмму. """
        # Создаем новую фигуру размером 10x6 дюймов
        plt.figure(figsize=(10, 6))
        
        # Создаем столбчатую диаграмму с ключами из data в качестве меток на оси X
        # и значениями из data в качестве высоты столбцов
        plt.bar(data.keys(), data.values(), color='skyblue', edgecolor='black')
        
        # Устанавливаем название оси X
        plt.xlabel(xlabel)
        
        # Устанавливаем название оси Y
        plt.ylabel("Среднее время реакции (сек)")
        
        # Устанавливаем заголовок гистограммы
        plt.title(title)
        
        # Поворачиваем метки на оси X на 45 градусов для лучшей читаемости
        plt.xticks(rotation=45)
        
        # Добавляем сетку по оси Y с пунктирными линиями и прозрачностью 0.7
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        savename = f'{self.folder_path}/{self.save_folder_name}/{title}.png'
        plt.savefig(savename, dpi=300)
        self.log(f"Гистограмма '{title}' готова", "SUCCESS", "PercepTest")
        
        # Отображаем гистограмму
        plt.show()
    
    def color_t_values(self):
        """ Получаем время реакции для каждой кнопки в зависимости от цвета. """
        # Создаем пустой словарь для хранения времени реакции по цветам кнопок
        color_t_values = {}
        # Для каждого цвета кнопки собираем все значения времени реакции
        for color in self.name_color_array_button:
            # Отбираем записи, где цвет фигуры совпадает с заданным цветом
            # и извлекаем значение времени реакции
            color_t_values[color] = [f['t'] for f in self.figure_list if f['color'] == color]
        return color_t_values

    def color_t_values_background(self):
        """ Получаем время реакции в зависимости от цвета фона. """
        # Создаем пустой словарь для хранения времени реакции по цветам фона
        color_t_values_background = {}
        # Для каждого цвета фона собираем все значения времени реакции
        for color in self.name_color_array_background:
            # Отбираем записи, где цвет фона совпадает с заданным цветом
            # и извлекаем значение времени реакции
            color_t_values_background[color] = [f['t'] for f in self.figure_list if f['background'] == color]
        return color_t_values_background

    def size_t_values(self):
        """ Получаем время реакции в зависимости от размера фигуры. """
        # Создаем пустой словарь для хранения времени реакции по размерам фигур
        size_t_values = {}
        # Для каждого возможного размера фигуры собираем все значения времени реакции
        for size in [25, 50, 100, 150, 200, 250]:
            # Отбираем записи, где высота фигуры совпадает с заданным размером
            # и извлекаем значение времени реакции
            size_t_values[str(size)] = [f['t'] for f in self.figure_list if f['size_h'] == size]
        return size_t_values

    def figure_t_values(self):
        """ Получаем время реакции в зависимости от формы фигуры. """
        # Создаем пустой словарь для хранения времени реакции по формам фигур
        figure_t_values = {}
        # Для каждого типа фигуры собираем все значения времени реакции
        for key, name in self.figure_names.items():
            # Отбираем записи, где тип фигуры совпадает с заданным типом
            # и извлекаем значение времени реакции
            figure_t_values[name] = [f['t'] for f in self.figure_list if f['figure'] == name]
        return figure_t_values

    def calculate_avg_t(self, values_dict):
        """ Вычисляет среднее время реакции по категориям. """
        # Создаем пустой словарь для хранения средних значений
        avg_t = {}
        
        # Для каждого ключа и списка значений времени в переданном словаре
        for key, times in values_dict.items():
            # Преобразуем строковые значения времени в числовые, заменяя запятые на точки
            # Проверяем, что значение является строкой, числом или может быть преобразовано
            numeric_times = [float(t.replace(',', '.')) for t in times if isinstance(t, (int, float, str)) and t]
            
            # Вычисляем среднее значение, если список не пустой
            # Если список пустой, устанавливаем среднее значение в 0
            avg_t[key] = sum(numeric_times) / len(numeric_times) if numeric_times else 0
        
        # Возвращаем словарь средних значений
        return avg_t
    
    def plot_area_vs_time(self):
        """ Строит график зависимости времени реакции от площади фигуры (scatter plot). """
        # Преобразуем figure_list в DataFrame для удобства
        df = pd.DataFrame(self.figure_list)
        df['area'] = df['size_h'] * df['size_w']  # Вычисляем площадь
        df['t'] = df['t'].astype(float)  # Убеждаемся, что время в float

        plt.figure(figsize=(10, 6))
        plt.scatter(df['area'], df['t'], color='teal', alpha=0.6, edgecolor='black')
        plt.xlabel("Площадь фигуры (size_h × size_w)")
        plt.ylabel("Время реакции (сек)")
        plt.title("Время реакции в зависимости от площади фигуры")
        plt.grid(True, linestyle='--', alpha=0.7)

        savename = f'{self.folder_path}/{self.save_folder_name}/Время реакции vs Площадь.png'
        plt.savefig(savename, dpi=300)
        self.log("График 'Время реакции vs Площадь' готов", "SUCCESS", "PercepTest")
        plt.show()

    def plot_color_background_heatmap(self):
        """ Строит тепловую карту времени реакции для комбинаций цвета фигуры и фона. """
        df = pd.DataFrame(self.figure_list)
        df['t'] = df['t'].astype(float)

        # Создаем сводную таблицу со средним временем реакции
        pivot_table = df.pivot_table(values='t', index='color', columns='background', aggfunc='mean')

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': 'Среднее время реакции (сек)'})
        plt.xlabel("Цвет фона")
        plt.ylabel("Цвет фигуры")
        plt.title("Среднее время реакции для комбинаций цвета фигуры и фона")

        savename = f'{self.folder_path}/{self.save_folder_name}/Тепловая карта Цвет фигуры vs Фон.png'
        plt.savefig(savename, dpi=300)
        self.log("Тепловая карта 'Цвет фигуры vs Фон' готова", "SUCCESS", "PercepTest")
        plt.show()

    def plot_boxplot_by_figure(self):
        """ Строит ящики с усами для времени реакции по типам фигур. """
        df = pd.DataFrame(self.figure_list)
        df['t'] = df['t'].astype(float)

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='figure', y='t', data=df, palette='Set2')
        plt.xlabel("Форма фигуры")
        plt.ylabel("Время реакции (сек)")
        plt.title("Распределение времени реакции по формам фигур")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        savename = f'{self.folder_path}/{self.save_folder_name}/Ящики с усами по формам.png'
        plt.savefig(savename, dpi=300)
        self.log("График 'Ящики с усами по формам' готов", "SUCCESS", "PercepTest")
        plt.show()
    

    # Функция ожидания следующей фигуры
    def expectation(self, text, delay_time):
        """ Функция ожидания с отображением текста. """

        # Продолжаем цикл, пока не пройдет указанное время
        if int(time.time() * 1000) - self.start_wait < delay_time:
            # Создаем пустое изображение с нужным цветом фона
            img = np.ones((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            # Заполняем изображение цветом фона
            img[:] = self.array_colors_background[self.i_color_background]
            
            # Выбираем цвет текста, контрастный с фоном
            # Инвертируем каждую компоненту цвета фона (255 - цвет)
            bg_color = self.array_colors_background[self.i_color_background]
            text_color = (255 - bg_color[0], 255 - bg_color[1], 255 - bg_color[2])
            
            # Настраиваем параметры текста
            font = cv2.FONT_HERSHEY_SIMPLEX  # Шрифт
            font_scale = 1  # Масштаб шрифта
            thickness = 2  # Толщина линий текста
            
            # Определяем размер текста для его центрирования
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            # Вычисляем координаты для центрирования текста
            text_x = (self.screen_w - text_size[0]) // 2
            text_y = (self.screen_h + text_size[1]) // 2
            
            # Добавляем текст на изображение
            cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)
            
            return img
        
        else:
            self.waiting = False
            self.running = True
            # Обновляем время начала для новой фигуры
            self.start_time = int(time.time() * 1000)
            return None

    def show_text_on_screen(self, text):
        """ Функция для отображения текста на экране. """
        # Создаем изображение с нужным цветом фона
        img = np.ones((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        # Заполняем изображение цветом фона
        img[:] = self.array_colors_background[self.i_color_background]
        
        # Настраиваем параметры текста
        font = cv2.FONT_HERSHEY_SIMPLEX  # Шрифт
        font_scale = 1  # Масштаб шрифта
        thickness = 2  # Толщина линий текста
        text_color = (255, 255, 255)  # Цвет текста (белый)
        
        # Определяем размер текста для его центрирования
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        # Вычисляем координаты для центрирования текста
        text_x = (self.screen_w - text_size[0]) // 2
        text_y = (self.screen_h + text_size[1]) // 2
        
        # Добавляем текст на изображение
        cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Отображаем текст с инструкциями управления внизу экрана
        controls_text = "ESC - finish test | 'k' - recalibrate"
        # Определяем размер текста инструкций
        controls_size = cv2.getTextSize(controls_text, font, 0.7, 1)[0]
        # Вычисляем координаты для центрирования инструкций
        controls_x = (self.screen_w - controls_size[0]) // 2
        controls_y = self.screen_h - 50
        # Добавляем текст инструкций на изображение
        cv2.putText(img, controls_text, (controls_x, controls_y), font, 0.7, text_color, 1)
        
        return img


    # Функция генерации фрейма с фигурой
    def show_figure(self):
        """ Функция генерации фрейма с фигурой. """
        
        # Создаем пустое изображение с цветом фона
        img = np.ones((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        

        # Заполняем изображение текущим цветом фона
        img[:] = self.array_colors_background[self.i_color_background]
        # Рисуем фигуру заданного типа в заданных координатах
        img = self.draw_figure(img, self.button_x, self.button_y, self.button_width, self.button_height, self.figure_type)
        
        # Добавляем инструкцию в верхнем левом углу
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Используем контрастный цвет относительно фона
        bg_color = self.array_colors_background[self.i_color_background]
        text_color = (255 - bg_color[0], 255 - bg_color[1], 255 - bg_color[2])
        # Выводим текст с инструкцией
        cv2.putText(img, "Look at the figure (gaze area should cross the figure)", (20, 30), font, 0.6, text_color, 1)

        # Отображаем текст с инструкциями управления внизу экрана
        controls_text = "ESC - finish test | SPACE - pause | 'k' - recalibrate"
        # Определяем размер текста инструкций
        controls_size = cv2.getTextSize(controls_text, font, 0.6, 1)[0]
        # Вычисляем координаты для центрирования инструкций
        controls_x = (self.screen_w - controls_size[0]) // 2
        controls_y = self.screen_h - 70
        # Добавляем текст инструкций на изображение
        cv2.putText(img, controls_text, (controls_x, controls_y), font, 0.6, text_color, 1)

        return img

    
    # Функция проверки пересечения фигуры с точкой взгляда
    def was_intersection(self):
        """ Функция проверки пересечения фигуры с точкой взгляда. """
        # Записываем время реакции
        current_time = int(time.time() * 1000)
        # Вычисляем время, прошедшее с момента появления фигуры
        elapsed_time_ms = current_time - self.start_time
        elapsed_time_seconds = elapsed_time_ms // 1000  # Секунды
        remaining_ms = elapsed_time_ms % 1000  # Миллисекунды
        time_click = f"{elapsed_time_seconds}.{remaining_ms:03d}"  # Форматируем время
        self.log(f"Время реакции: {time_click} секунд", "TESTING", "PercepTest")

        # Добавляем запись о реакции в список
        self.figure_list.append({
            't': time_click,  # Время реакции
            'color': self.name_color_array_button[self.i_color_button],  # Цвет фигуры
            'figure': self.figure_names[self.figure_type],  # Тип фигуры
            'background': self.name_color_array_background[self.i_color_background],  # Цвет фона
            'size_h': self.button_height,  # Высота фигуры
            'size_w': self.button_width  # Ширина фигуры
        })

        # Генерируем новые случайные параметры для следующей фигуры
        self.i_color_button, self.i_color_background, self.figure_type = self.get_random_value()
        self.button_width, self.button_height = self.get_random_dimensions()
        self.button_x, self.button_y = random.randint(self.button_width // 2, self.screen_w - self.button_width // 2), random.randint(self.button_height // 2, self.screen_h - self.button_height // 2)

        # Добавляем запись в журнал времени
        self.time_logs.append({
            "start": self.start_time,  # Время начала показа фигуры
            "stop": current_time,  # Время нажатия клавиши
            "time": time_click  # Время реакции
        })

        # Генерация случайной паузы
        self.time_pause = random.randint(100, 3000)
        # Запоминаем время начала ожидания в миллисекундах
        self.start_wait = int(time.time() * 1000)
        self.waiting = True
    
    # Функция сохранения данных в CSV и построения гистограмм
    def make_percep_charts(self):
        # Сохранение в CSV
        self.save_to_csv(self.figure_list)

        # Расчет среднего времени реакции
        color_avg_t = self.calculate_avg_t(self.color_t_values())
        color_avg_t_background = self.calculate_avg_t(self.color_t_values_background())
        size_avg_t = self.calculate_avg_t(self.size_t_values())
        figure_avg_t = self.calculate_avg_t(self.figure_t_values())

        # Строим гистограммы
        self.plot_histogram(color_avg_t, "Цвет кнопки", "Среднее время реакции для каждого цвета кнопки")
        self.plot_histogram(color_avg_t_background, "Цвет фона", "Среднее время реакции для каждого цвета фона")
        self.plot_histogram(size_avg_t, "Размер фигуры", "Среднее время реакции для каждого размера фигуры")
        self.plot_histogram(figure_avg_t, "Форма фигуры", "Среднее время реакции для каждой формы фигуры") 

        self.plot_area_vs_time()
        self.plot_color_background_heatmap()
        self.plot_boxplot_by_figure()


    # ГЛАВНЫЙ РАБОЧИЙ ЦИКЛ ПРОГРАММЫ 
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def start(self):
        self.log(f"Трекер запущен и начал работу", "INFO", "tracker")
        # Главный цикл
        while self.window_holder:
            
            _, self.frame = self.cam.read()                         # Получаем кадр с вебки
            frame_h, frame_w, _ = self.frame.shape

            # Вычисляем новую ширину кадра, сохраняя соотношение сторон
            new_frame_h = self.screen_h  # Высота кадра будет равна высоте изображения
            aspect_ratio = frame_w / frame_h  # Соотношение сторон веб-камеры
            new_frame_w = int(new_frame_h * aspect_ratio)  # Новая ширина кадра

            self.frame = cv2.flip(self.frame, 1)                    # Отражаем кадр

            # Масштабируем кадр с сохранением пропорций
            self.frame = cv2.resize(self.frame, (new_frame_w, new_frame_h))
            # Создаем черный фон (канвас) с размерами изображения
            canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

            # Вычисляем координаты для центрирования по ширине
            x_offset = (self.screen_w - new_frame_w) // 2

            # Вставляем кадр с веб-камеры по центру (с чёрными полями, если нужно)
            canvas[:, x_offset:x_offset+new_frame_w] = self.frame

            self.frame = canvas

            frame_h, frame_w, _ = self.frame.shape

            # self.frame = cv2.resize(self.frame, (self.screen_w, self.screen_h))       # Ресайз под разрешение
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            output = self.face_mesh.process(rgb_frame)              # Получаем разметку лица
            landmark_points = output.multi_face_landmarks
            

            if self.calibrated:
                if self.temp_frame is not None:
                    self.frame = self.temp_frame.copy()

            # Если есть лицо в кадре
            if landmark_points:

                # Выводим ключевые точки лица на экран
                self.draw_landmarks(landmark_points[0].landmark, frame_w, frame_h)

                # Вывод центра кадра во время калибровки
                if self.focused < 3:
                    # Производим калибровку
                    self.calibrate(frame_w, frame_h)

                else:

                    # Показываем начальный экран с инструкциями
                    # Если пользователь нажал ESC, завершаем работу
                    if not self.running or self.paused:
                        self.frame = self.show_text_on_screen("Press ENTER to Start | ESC to Exit")
                        self.temp_frame = self.frame.copy()
                    
                    if self.waiting and self.paused != True:
                        # Показываем экран ожидания
                        # Если пользователь прервал ожидание, завершаем работу
                        frame = self.expectation("WAITING", self.time_pause)
                        if frame is not None:
                            self.frame = frame
                            self.temp_frame = frame.copy()
                        else:
                            self.waiting = False
                            self.time_pause = 0
                            self.start_wait = 0
                    elif self.running and self.paused != True:
                        self.frame = self.show_figure()
                        intersection, distance = self.check_gaze_figure_intersection()
                        # Выводим координаты сглаженной позиции взгляда
                        if intersection:
                            self.was_intersection()
                        else:
                            cv2.putText(self.frame, f'Distance: {round(distance, 2)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_4)

                    # Отображаем точку калибровки носа
                    cv2.circle(self.frame, (self.nose_calib_mean_x, self.nose_calib_mean_y), 2, (0, 0, 255), 2)
                    # Отображаем точку пользователя
                    cv2.circle(self.frame, (int(self.nose.x * frame_w), int(self.nose.y * frame_h)), 2, (0, 255, 255), 2)

                    # Рассчитываем точку взгляда
                    self.calculate_gaze(frame_w, frame_h)

                    # Отрисовываем точку взгляда
                    self.draw_gaze_direction(self.smooth_frame_pos_x_exp, self.smooth_frame_pos_y_exp)

                    
            cv2.imshow('Eye tracker', self.frame)

            # ОБРАБОТКА НАЖАТИЯ КЛАВИШ 
            # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            # Ожидание нажатия клавиши
            self.key = cv2.waitKey(1) & 0xFF

            # Если нажата 'k' - Начать калибровку
            if self.key == ord('k'):
                self.focused = 0
                self.nose_calib_x = []
                self.nose_calib_y = []
                self.fixations = []
                self.calibrated = False
                self.temp_frame = None
                self.running = False
            # Если нажата 'Enter' - Начать 
            elif self.key == 13:
                self.paused = False
                self.running = True
                self.waiting = True
                # Генерируем случайное время паузы от 100 до 3000 мс
                self.time_pause = random.randint(100, 3000)
                # Запоминаем время начала ожидания в миллисекундах
                self.start_wait = int(time.time() * 1000)
            # Если нажата 'p' - установка паузы
            elif self.key == ord(' ') and self.calibrated:
                self.paused = True
                self.running = False

            # БЛОК РАБОТЫ ТЕСТИРОВАНИЯ 
            # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            # Если нажата 'q' - Завершить работу
            if self.key == ord('q'):  
                self.log(f"Трекер завершает работу", "SUCCESS", "tracker")
                break
            elif self.key == 27:  # ESC - завершение теста и показ результатов
                self.log(f"Трекер завершает работу, построение графиков", "SUCCESS", "tracker")
                self.esc = True
                break                

        self.cam.release()
        cv2.destroyAllWindows()

        if self.esc:
            # Если были собраны данные, сохраняем и анализируем их
            if self.figure_list:
                self.log(f"Тест завершен. Собрано {len(self.figure_list)} точек данных.", "SUCCESS", "PercepTest")
                self.make_percep_charts()


    def check_gaze_figure_intersection(self):
        """
        Проверяет пересечение зоны взгляда (окружности) с геометрической фигурой на экране.
        
        Возвращает:
        - bool: True, если зона взгляда пересекается с фигурой, иначе False
        - float: Расстояние от центра зоны взгляда до ближайшей точки фигуры
                (отрицательное, если центр внутри фигуры; положительное, если снаружи)
        """
        # Координаты и радиус зоны взгляда
        gaze_x, gaze_y = self.smooth_frame_pos_x, self.smooth_frame_pos_y
        gaze_radius = self.radius

        # Координаты и параметры фигуры
        fig_x, fig_y = self.button_x, self.button_y
        fig_width, fig_height = self.button_width, self.button_height
        fig_type = self.figure_type

        # Вспомогательная функция для вычисления евклидова расстояния между точками
        def euclidean_distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        # Универсальная функция для расчета расстояния до ближайшей точки на границе
        def distance_to_boundary():
            if fig_type == 1:  # Эллипс
                # Преобразуем в пространство эллипса
                dx = (gaze_x - fig_x) / (fig_width / 2)
                dy = (gaze_y - fig_y) / (fig_height / 2)
                norm_dist = np.sqrt(dx**2 + dy**2)

                if norm_dist <= 1:  # Центр внутри эллипса
                    # Находим ближайшую точку на границе эллипса
                    if norm_dist > 0:
                        boundary_x = fig_x + (dx / norm_dist) * (fig_width / 2)
                        boundary_y = fig_y + (dy / norm_dist) * (fig_height / 2)
                    else:  # Центр в середине
                        boundary_x, boundary_y = fig_x, fig_y + fig_height / 2
                    distance = -euclidean_distance((gaze_x, gaze_y), (boundary_x, boundary_y))
                else:  # Центр снаружи
                    boundary_x = fig_x + (dx / norm_dist) * (fig_width / 2)
                    boundary_y = fig_y + (dy / norm_dist) * (fig_height / 2)
                    distance = euclidean_distance((gaze_x, gaze_y), (boundary_x, boundary_y))
                return distance

            elif fig_type == 2:  # Круг
                fig_radius = min(fig_width, fig_height) / 2
                center_dist = euclidean_distance((gaze_x, gaze_y), (fig_x, fig_y))
                if center_dist <= fig_radius:  # Центр внутри
                    return -(fig_radius - center_dist)
                return center_dist - fig_radius

            elif fig_type == 3:  # Ромб
                # Вершины ромба
                vertices = [
                    (fig_x, fig_y - fig_height / 2),  # верх
                    (fig_x + fig_width / 2, fig_y),    # право
                    (fig_x, fig_y + fig_height / 2),   # низ
                    (fig_x - fig_width / 2, fig_y)     # лево
                ]

                # Проверка, внутри ли центр ромба (Ray Casting)
                def is_inside_polygon(point, poly):
                    x, y = point
                    n = len(poly)
                    inside = False
                    for i in range(n):
                        x1, y1 = poly[i]
                        x2, y2 = poly[(i + 1) % n]
                        if ((y1 <= y < y2) or (y2 <= y < y1)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                            inside = not inside
                    return inside

                # Расстояние до ближайшей стороны
                def distance_to_segment(p, v1, v2):
                    # Проекция точки p на отрезок v1-v2
                    line_vec = (v2[0] - v1[0], v2[1] - v1[1])
                    point_vec = (p[0] - v1[0], p[1] - v1[1])
                    line_len = euclidean_distance(v1, v2)
                    if line_len == 0:
                        return euclidean_distance(p, v1)
                    t = max(0, min(1, (point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]) / (line_len ** 2)))
                    projection = (v1[0] + t * line_vec[0], v1[1] + t * line_vec[1])
                    return euclidean_distance(p, projection)

                is_inside = is_inside_polygon((gaze_x, gaze_y), vertices)
                distances = [distance_to_segment((gaze_x, gaze_y), vertices[i], vertices[(i + 1) % 4]) 
                             for i in range(4)]
                min_dist = min(distances)
                return -min_dist if is_inside else min_dist

            elif fig_type in (4, 5):  # Прямоугольник или квадрат
                if fig_type == 5:  # Квадрат
                    size = min(fig_width, fig_height)
                    half_size = size / 2
                    left, right = fig_x - half_size, fig_x + half_size
                    top, bottom = fig_y - half_size, fig_y + half_size
                else:  # Прямоугольник
                    left, right = fig_x - fig_width / 2, fig_x + fig_width / 2
                    top, bottom = fig_y - fig_height / 2, fig_y + fig_height / 2

                # Ближайшая точка на границе
                closest_x = max(left, min(gaze_x, right))
                closest_y = max(top, min(gaze_y, bottom))
                dist = euclidean_distance((gaze_x, gaze_y), (closest_x, closest_y))
                is_inside = (left <= gaze_x <= right) and (top <= gaze_y <= bottom)
                if is_inside:
                    # Расстояние до ближайшей границы
                    distances = [gaze_x - left, right - gaze_x, gaze_y - top, bottom - gaze_y]
                    return -min(distances)
                return dist

            else:
                self.log(f"Неизвестный тип фигуры: {fig_type}", "WARNING", "PercepTest")
                return float('inf')

        # Основная логика
        distance = distance_to_boundary()
        intersects = distance <= gaze_radius
        actual_distance = distance - gaze_radius if distance > 0 else distance
        return intersects, actual_distance

if __name__ == "__main__":
    tracker = EyeTracker()
    tracker.start()