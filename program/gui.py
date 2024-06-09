from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
import csv
import os
import pandas as pd
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QGraphicsDropShadowEffect, QPlainTextEdit
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt
import re
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QPainterPath, QPen, QFont
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QFrame
import datetime
from scipy import spatial
import chardet
import main
from PyQt5 import QtCore
import webbrowser
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import QPropertyAnimation, QRect
from functools import partial


from main import location_json
from main import street_json
from main import residential_complex_json
from main import district_json
from main import loaded_model
from main import df2
from main import df1

_translate = QtCore.QCoreApplication.translate

class CustomWidget1(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setGeometry(QtCore.QRect(10, 373, 322, 231))
        self.setMinimumSize(QtCore.QSize(310, 0))
        self.setMaximumSize(QtCore.QSize(322, 500))
        self.setStyleSheet("background-color: rgb(248, 247, 252);\n"
                           "border: none;\n"
                           "border-radius: 21px;\n"
                           "")
        self.setObjectName("widget_40")

        self.x = -34
        self.y = 3 
        
        self.y_coords = [120, 142, 155]
        self.txt = [159.0, 158.2, 156.6]


    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(Qt.NoPen)

        painter.setBrush(QtGui.QColor(248, 247, 252))  # серый цвет
        painter.drawRoundedRect(0, 0, 322, 231, 12, 12)

        # Серый прямоугольник с закругленными углами
        painter.setBrush(QtGui.QColor(244, 243, 249))  # серый цвет 244, 243, 249
        painter.drawRoundedRect(16, 61, 290, 141, 12, 12)

        # Добавление кривой
        path = QPainterPath()
        path.moveTo(50 + self.x, self.y_coords[0] + self.y)
        path.lineTo(197 + self.x, self.y_coords[1] + self.y)
        path.lineTo(340 + self.x, self.y_coords[2] + self.y)
        path.lineTo(340 + self.x, 192 + self.y)
        path.cubicTo(340 + self.x, 186 + self.y, 340 + self.x, 198 + self.y, 328 + self.x, 198 + self.y)
        path.lineTo(56 + self.x, 198 + self.y)
        path.cubicTo(62 + self.x, 198 + self.y, 50 + self.x, 198 + self.y, 50 + self.x, 186 + self.y)
        path.lineTo(50 + self.x, 154 + self.y)
        painter.setBrush(QColor(231, 238, 255))
        painter.drawPath(path)

        # Добавление линии
        painter = QPainter()
        color = QColor(57, 106, 234)
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(color, 4, Qt.SolidLine))
        painter.setBrush(color)
        painter.drawLine(50 + self.x, self.y_coords[0] + self.y, 197 + self.x, self.y_coords[1] + self.y)
        painter.drawLine(197 + self.x, self.y_coords[1] + self.y, 340 + self.x, self.y_coords[2] + self.y)

        # Добавление трех точек на полотно
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(57, 106, 234))
        painter.drawEllipse(194 + self.x, self.y_coords[1] + self.y - 4, 7, 7)  # центральная точка по середине
        painter.drawEllipse(47 + self.x, self.y_coords[0] + self.y - 4, 7, 7)   # точка слева по краю
        painter.drawEllipse(337 + self.x, self.y_coords[2] + self.y - 4, 7, 7)  # точка справа по краю

        # Добавление текста над тремя точками
        painter.setPen(QColor(57, 106, 234))
        font = QFont('Decorative', 9, weight=QFont.Bold)
        painter.setFont(font)

        painter.drawText(47 + self.x + 5, self.y_coords[0] + self.y - 15, "{:.1f} тыс, р".format(self.txt[0]))  # Текст над точкой слева

        text = "{:.1f} тыс, р".format(self.txt[1])
        text_width = painter.fontMetrics().width(text)
        painter.drawText(194 + self.x - round(text_width / 2) + 4, self.y_coords[1] + self.y - 15, text)  # Текст над центральной точкой

        text = "{:.1f} тыс, р".format(self.txt[2])
        text_width = painter.fontMetrics().width(text)
        painter.drawText(337 + self.x - text_width - 5, self.y_coords[2] + self.y - 15, text)  # Текст над точкой справа


class CustomWidget2(QtWidgets.QWidget):
        def __init__(self, parent=None):
                super().__init__(parent)  
                self.setGeometry(QtCore.QRect(10, 622, 322, 231))
                self.setMinimumSize(QtCore.QSize(310, 0))
                self.setMaximumSize(QtCore.QSize(322, 500))
                self.setStyleSheet("background-color: rgb(248, 247, 252);\n"
                                "border: none;\n"
                                "border-radius: 21px;\n"
                                "")
                self.setObjectName("widget_45")

                self.x = -34
                self.y = 3

                self.y_coords2 = [120, 142, 155]
                self.txt2 = [156.6, 153.2, 151.4]


        def paintEvent(self, event):
                painter = QtGui.QPainter(self)
                painter.setPen(Qt.NoPen)

                painter.setBrush(QtGui.QColor(248,247,252)) # серый цвет
                painter.drawRoundedRect(0, 0, 322, 231, 12, 12)

                # Серый прямоугольник с закругленными углами
                painter.setBrush(QtGui.QColor(244, 243, 249)) # серый цвет 244, 243, 249
                painter.drawRoundedRect(16, 61, 290, 141, 12, 12)

                # Добавление кривой
                path = QPainterPath()
                path.moveTo(50+self.x, self.y_coords2[0]+self.y)
                path.lineTo(197+self.x, self.y_coords2[1]+self.y)
                path.lineTo(340+self.x, self.y_coords2[2]+self.y)
                path.lineTo(340+self.x, 192+self.y)
                path.cubicTo(340+self.x, 186+self.y, 340+self.x, 198+self.y, 328+self.x, 198+self.y)
                path.lineTo(56+self.x, 198+self.y)
                path.cubicTo(62+self.x, 198+self.y, 50+self.x, 198+self.y, 50+self.x, 186+self.y)
                path.lineTo(50+self.x, 154+self.y)
                painter.setBrush(QColor(226, 208, 235))
                painter.drawPath(path)

                # Добавление линии
                painter = QPainter()
                color = QColor(142, 49, 175)
                painter.begin(self)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setPen(QPen(color, 4, Qt.SolidLine))
                painter.setBrush(color)
                painter.drawLine(50+self.x, self.y_coords2[0]+self.y, 197+self.x, self.y_coords2[1]+self.y)
                painter.drawLine(197+self.x, self.y_coords2[1]+self.y, 340+self.x, self.y_coords2[2]+self.y)

                # Добавление трех точек на полотно
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(142, 49, 175))
                painter.drawEllipse(194+self.x, self.y_coords2[1]+self.y - 4, 7, 7)  # центральная точка по середине
                painter.drawEllipse(47+self.x, self.y_coords2[0]+self.y - 4, 7, 7)    # точка слева по краю
                painter.drawEllipse(337+self.x, self.y_coords2[2]+self.y - 4, 7, 7)   # точка справа по краю

                # Добавление текста над тремя точками
                painter.setPen(QColor(142, 49, 175))
                font = QFont('Decorative', 9, weight=QFont.Bold)
                painter.setFont(font)

                painter.drawText(47+self.x+5, self.y_coords2[0]+self.y - 15, "{:.1f} тыс, р".format(self.txt2[0]))    # Текст над точкой слева

                text = "{:.1f} тыс, р".format(self.txt2[1])
                text_width = painter.fontMetrics().width(text)
                painter.drawText(194 + self.x - round(text_width / 2) + 4, self.y_coords2[1] + self.y - 15, text)  # Текст над центральной точкой

                text = "{:.1f} тыс, р".format(self.txt2[2])
                text_width = painter.fontMetrics().width(text)
                painter.drawText(337 + self.x - text_width - 5, self.y_coords2[2] + self.y - 15, text)   # Текст над точкой справа



class Ui_MainWindow(QWidget):
    def move_widget_15(self):
        self.close.show()
        self.anim = QPropertyAnimation(self.widget_15, b"geometry")
        self.anim.setDuration(450)  # Продолжительность анимации в миллисекундах
        self.anim.setStartValue(QRect(0, 820, self.widget_15.width(), self.widget_15.height()))
        self.anim.setEndValue(QRect(0, 410, self.widget_15.width(), self.widget_15.height()))  # Конечные координаты
        self.anim.start()


    def close_button_clicked(self):
        self.close.hide()
        self.anim = QPropertyAnimation(self.widget_15, b"geometry")
        self.anim.setDuration(450)  # Продолжительность анимации в миллисекундах
        self.anim.setStartValue(QRect(0, 410, self.widget_15.width(), self.widget_15.height()))
        self.anim.setEndValue(QRect(0, 820, self.widget_15.width(), self.widget_15.height()))  # Конечные координаты
        self.anim.start()


    def addSlotToHistory2(self):
        dataset = pd.read_csv("info.csv")
        hi = len(dataset)
        if hi > 0:
            self.empty_4.hide()
        else:
            self.empty_4.show()

        calc_2 = (67 * hi) + ((hi - 1) * 12) + 5

        if hi == 1:
              self.scrollArea_6.setFixedHeight(91)
        elif hi == 2:
              self.scrollArea_6.setFixedHeight(171)
        elif hi == 3:
              self.scrollArea_6.setFixedHeight(245)
        elif hi == 4:
              self.scrollArea_6.setFixedHeight(315)
              

        for i in range(hi - 1, -1, -1):
            item_id = dataset.iloc[i, 0]  # Предположим, что первый столбец содержит уникальные идентификаторы элементов
            if item_id in self.added_items:
                return  # Пропускаем добавление элемента, если он уже добавлен

            self.added_items.add(item_id)  # Добавляем идентификатор в набор добавленных

            frame = QFrame()
            frame.setStyleSheet("background-color: rgb(235, 233, 244);\n"
                                "border: none;\n"
                                "border-radius: 17px;")
            self.pushButton_545 = QtWidgets.QPushButton(frame)
            self.pushButton_545.setGeometry(QtCore.QRect(0, 0, 290, 67))
            self.pushButton_545.setStyleSheet("QPushButton {\n"
                                              "background-color: rgba(255, 255, 255, 0);\n"
                                              "border-radius: 17px;\n"
                                              "}\n"
                                              "\n"
                                              "QPushButton:pressed {\n"
                                              "background-color: rgba(0, 0, 0, 7)\n"
                                              "}")
            self.pushButton_545.setText("")
            self.pushButton_545.setObjectName("pushButton_5")
            self.layoutWidget_1138 = QtWidgets.QWidget(frame)
            self.layoutWidget_1138.setGeometry(QtCore.QRect(0, 3, 286, 66))
            self.layoutWidget_1138.setObjectName("layoutWidget_1138")
            self.verticalLayout_288 = QtWidgets.QVBoxLayout(self.layoutWidget_1138)
            self.verticalLayout_288.setContentsMargins(15, 8, 0, 14)
            self.verticalLayout_288.setSpacing(0)
            self.verticalLayout_288.setObjectName("verticalLayout_288")
            self.label_1992 = QtWidgets.QLabel(self.layoutWidget_1138)
            self.label_1992.setMaximumSize(QtCore.QSize(270, 16777215))
            font = QtGui.QFont()
            font.setPointSize(10)
            font.setBold(True)
            font.setWeight(75)
            self.label_1992.setFont(font)
            self.label_1992.setObjectName("label_1992")
            self.label_1992.setText(f"<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">{dataset.iloc[i, 1]} - {dataset.iloc[i, 5]} м², {dataset.iloc[i, 14]} млн, р</span></p></body></html>")
            self.verticalLayout_288.addWidget(self.label_1992)
            self.label_1212 = QtWidgets.QLabel(self.layoutWidget_1138)
            self.label_1212.setMaximumSize(QtCore.QSize(270, 16777215))
            self.label_1212.setObjectName("label_1212")

            if dataset.iloc[i, 8] == 1:
                coll = "комната"
            elif dataset.iloc[i, 8] in [2, 3, 4]:
                coll = "комнаты"
            else:
                coll = "комнат"

            if dataset.iloc[i, 13] != "Отсутствует":
                compl = f"{dataset.iloc[i, 12]}, {dataset.iloc[i, 13]}"
            else:
                compl = f"{dataset.iloc[i, 12]}"

            text = f"{dataset.iloc[i, 10]}, {compl}, {dataset.iloc[i, 4]}/{dataset.iloc[i, 3]} этаж, {dataset.iloc[i, 8]} {coll}, {dataset.iloc[i, 9]} год"
            wrapped_text = self.wrap_text(text)  # вызов функции переносящей слова на следующую строку
            self.label_1212.setText(f"<html><head/><body><p><span style=\" color:#a0a0a0;\">{wrapped_text}</span></p></body></html>")

            self.verticalLayout_288.addWidget(self.label_1212)

            self.layoutWidget_1138.raise_()
            self.pushButton_545.raise_()

            frame.setMinimumSize(290, 67)  # Устанавливаем минимальный размер фрейма (ширина, высота)
            frame.setMaximumSize(290, 67)  # Устанавливаем максимальный размер фрейма (ширина, высота)
            self.verticalLayout_13.addWidget(frame)


    def wrap_text(self, text, limit=45): # перенос слов
        words = text.split()
        lines = []
        line = ''
        
        for word in words:
                if len(line) + len(word) <= limit:
                        line += ' ' + word if line else word
                else:
                        lines.append(line)
                        line = word
                
        if line:
                lines.append(line)
                
        return '<br/>'.join(lines)
    

    def upd_inform(self):
        dataset = pd.read_csv("info.csv")
        hi = len(dataset)
        i = hi - 1
        if i < 0:
              return
        
        self.pushButton_5.clicked.connect(partial(self.opening_information, i))

        self.label_9.setText(f"<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">{dataset.iloc[i, 1]} - {dataset.iloc[i, 5]} м², {dataset.iloc[i, 14]} млн, р</span></p></body></html>")
       
        # Правильное отображение приписки
        if dataset.iloc[i, 8] == 1:
                coll = "комната"
        elif dataset.iloc[i, 8] == 2 or dataset.iloc[i, 8] == 3 or dataset.iloc[i, 8] == 4:
                coll = "комнаты"
        else:
                coll = "комнат"
        
        # Проверка на заполненность жк
        if dataset.iloc[i, 13] != "Отсутствует":
                compl = f"{dataset.iloc[i, 12]}, {dataset.iloc[i, 13]}"
        else:
                compl = f"{dataset.iloc[i, 12]}"

                
        text = f"{dataset.iloc[i, 10]}, {compl}, {dataset.iloc[i, 4]}/{dataset.iloc[i, 3]} этаж, {dataset.iloc[i, 8]} {coll}, {dataset.iloc[i, 9]} год"
        wrapped_text = self.wrap_text(text) # вызов функции переносящей слова на следующую строку
        self.label_12.setText(f"<html><head/><body><p><span style=\" color:#a0a0a0;\">{wrapped_text}</span></p></body></html>")

        if i > 1:
                self.label_16.setText(f"<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">{dataset.iloc[i - 1, 1]} - {dataset.iloc[i - 1, 5]} м², {dataset.iloc[i - 1, 14]} млн, р</span></p></body></html>")
        
                # Правильное отображение приписки
                if dataset.iloc[i - 1, 8] == 1:
                        coll = "комната"
                elif dataset.iloc[i - 1, 8] == 2 or dataset.iloc[i - 1, 8] == 3 or dataset.iloc[i - 1, 8] == 4:
                        coll = "комнаты"
                else:
                        coll = "комнат"
                
                # Проверка на заполненность жк
                if dataset.iloc[i, 13] != "Отсутствует":
                        compl = f"{dataset.iloc[i - 1, 12]}, {dataset.iloc[i - 1, 13]}"
                else:
                        compl = f"{dataset.iloc[i - 1, 12]}"

                        
                text = f"{dataset.iloc[i - 1, 10]}, {compl}, {dataset.iloc[i - 1, 4]}/{dataset.iloc[i - 1, 3]} этаж, {dataset.iloc[i - 1, 8]} {coll}, {dataset.iloc[i - 1, 9]} год"
                wrapped_text = self.wrap_text(text) # вызов функции переносящей слова на следующую строку

                self.pushButton_9.clicked.connect(partial(self.opening_information, i - 1))

                self.label_17.setText(f"<html><head/><body><p><span style=\" color:#a0a0a0;\">{wrapped_text}</span></p></body></html>")
        

    def addSlotToHistory(self):
        dataset = pd.read_csv("info.csv")
        hi = len(dataset)
        if hi > 0:
              self.empty_5.hide()
        else:
              self.empty_5.show()

        calc_2 = (67 * hi) + ((hi - 1) * 12) + 5 # увеличение высоты виджетам

        self.layoutWidget_5.setGeometry(QtCore.QRect(7, 80, 311, calc_2)) # задаётся высота вертикального лаяута

        if hi > 6:
            self.background.setGeometry(QtCore.QRect(1, 135, 322, 127 + calc_2))
            self.foreground.setGeometry(QtCore.QRect(0, 30, 322, 97 + calc_2))

            self.scrollAreaWidgetContents_3.setMinimumSize(QtCore.QSize(0, 300 + calc_2))
            self.frame_28.setMinimumSize(QtCore.QSize(0, 1000 + calc_2))
        else:
            self.background.setGeometry(QtCore.QRect(1, 135, 322, 594))
            self.foreground.setGeometry(QtCore.QRect(0, 30, 322, 564))
            self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 349, 790))

        for i in range(hi - 1, -1, -1):
                frame = QFrame()
                frame.setStyleSheet("background-color: rgb(235, 233, 244);\n"
                "border: none;\n"
                "border-radius: 17px;")
                self.pushButton_555 = QtWidgets.QPushButton(frame)
                self.pushButton_555.setGeometry(QtCore.QRect(0, 0, 290, 67))
                self.pushButton_555.setStyleSheet("QPushButton {\n"
        "background-color: rgba(255, 255, 255, 0);\n"
        "border-radius: 17px;\n"
        "}\n"
        "\n"
        "QPushButton:pressed {\n"
        "background-color: rgba(0, 0, 0, 7)\n"
        "}")
                self.pushButton_555.setText("")
                self.pushButton_555.setObjectName("pushButton_555")
                self.layoutWidget_138 = QtWidgets.QWidget(frame)
                self.layoutWidget_138.setGeometry(QtCore.QRect(0, 3, 286, 66))
                self.layoutWidget_138.setObjectName("layoutWidget_138")
                self.verticalLayout_88 = QtWidgets.QVBoxLayout(self.layoutWidget_138)
                self.verticalLayout_88.setContentsMargins(15, 8, 0, 14)
                self.verticalLayout_88.setSpacing(0)
                self.verticalLayout_88.setObjectName("verticalLayout_88")
                self.label_99 = QtWidgets.QLabel(self.layoutWidget_138)
                self.label_99.setMaximumSize(QtCore.QSize(270, 16777215))
                font = QtGui.QFont()
                font.setPointSize(10)
                font.setBold(True)
                font.setWeight(75)
                self.label_99.setFont(font)
                self.label_99.setObjectName("label_99")
                self.label_99.setText(f"<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">{dataset.iloc[i, 1]} - {dataset.iloc[i, 5]} м², {dataset.iloc[i, 14]} млн, р</span></p></body></html>")
                self.verticalLayout_88.addWidget(self.label_99)
                self.label_112 = QtWidgets.QLabel(self.layoutWidget_138)
                self.label_112.setMaximumSize(QtCore.QSize(270, 16777215))
                self.label_112.setObjectName("label_112")

                # Правильное отображение приписки
                if dataset.iloc[i, 8] == 1:
                        coll = "комната"
                elif dataset.iloc[i, 8] == 2 or dataset.iloc[i, 8] == 3 or dataset.iloc[i, 8] == 4:
                        coll = "комнаты"
                else:
                        coll = "комнат"
                
                # Проверка на заполненность жк
                if dataset.iloc[i, 13] != "Отсутствует":
                     compl = f"{dataset.iloc[i, 12]}, {dataset.iloc[i, 13]}"
                else:
                     compl = f"{dataset.iloc[i, 12]}"

                     
                text = f"{dataset.iloc[i, 10]}, {compl}, {dataset.iloc[i, 4]}/{dataset.iloc[i, 3]} этаж, {dataset.iloc[i, 8]} {coll}, {dataset.iloc[i, 9]} год"
                wrapped_text = self.wrap_text(text) # вызов функции переносящей слова на следующую строку
                self.label_112.setText(f"<html><head/><body><p><span style=\" color:#a0a0a0;\">{wrapped_text}</span></p></body></html>")
                
                self.verticalLayout_88.addWidget(self.label_112)
                
                self.layoutWidget_138.raise_()
                self.pushButton_555.raise_()
                
                self.pushButton_555.clicked.connect(partial(self.opening_information, i))

                frame.setMinimumSize(290, 67)  # Устанавливаем минимальный размер фрейма (ширина, высота)
                frame.setMaximumSize(290, 67)  # Устанавливаем максимальный размер фрейма (ширина, высота)
                self.verticalLayout_40.addWidget(frame)


    def changeColor(self):
        sender = self.sender()

        if self.currentButton:
            self.currentButton.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border-radius: 5px;")  # Сброс стилей предыдущей активной кнопки

        self.currentButton = sender
        self.currentButton.setStyleSheet("background-color: rgb(213, 213, 213);\n"
"border-radius: 5px;")

        self.number_of_rooms = int(self.currentButton.text().replace('+', ''))


#     def open_page_info(self):
#     открытие информации по квартире через датасет info csv


    def processing_info(self):
        # Проверка значений на заполненность
        try:
                # Текущие значения
                self.locat = self.comboBox.currentText()
                self.locat = location_json.get(self.locat)

                self.floor = self.label_69.text()
                self.floors_count = self.label_63.text()
                self.rooms_count = self.number_of_rooms
                self.tot_meters = self.label_72.text()

                self.district = self.comboBox_4.currentText()
                self.district = district_json.get(self.district)

                self.street = self.comboBox_3.currentText()
                self.street = street_json.get(self.street)

                self.residential_complex = self.comboBox_5.currentText()
                self.residential_complex = residential_complex_json.get(self.residential_complex)

                self.living_meters = self.label_66.text()
                self.kitchen_meters = self.label_58.text()
                self.year_of_construction = self.comboBox_6.currentText()

                # Признаки для обучения
                self.spisok = [self.locat,self.floor,self.floors_count,
                self.rooms_count,self.tot_meters,
                self.district,self.street,self.residential_complex,
                self.living_meters,self.kitchen_meters,self.year_of_construction]
                
                # Прогноз
                self.price_per_m2 = round(loaded_model.predict(self.spisok))

                # Стоимость квартиры
                self.price = int(self.price_per_m2) * int(self.tot_meters)

                # Количество строк в датасете
                self.df = pd.read_csv('info.csv')
                self.num_rows = self.df.shape[0] + 1

                # Текущее время
                self.now_data = datetime.datetime.now()
                self.now_data = self.now_data.strftime("%d.%m.%y %H:%M")

                # Похожие квартиры
                # сортировка
                an = self.comboBox.currentText()
                self.df1_sorted = df1.loc[df1['location'] == self.locat]

                # Преобразование целевого значения в float
                self.tot_meters = int(self.tot_meters)
                self.rooms_count = int(self.rooms_count)

                # Вычисляем абсолютное отклонение от целевого значения
                self.df1_sorted['diff_meters'] = np.abs(self.df1_sorted['total_meters'] - self.tot_meters)
                self.df1_sorted['diff_rooms'] = np.abs(self.df1_sorted['rooms_count'] - self.rooms_count)

                # Сортируем строки по абсолютному отклонению
                df1_sorted = self.df1_sorted.sort_values(['diff_meters', 'diff_rooms'])

                # Выводим две наиболее схожие строки
                similar_apartments = df1_sorted.head(2)
                similar_apartments = similar_apartments.reset_index(drop=True)

                # Присваиваем значения переменным
                similar_apartment1 = similar_apartments.loc[0, 'url']
                similar_apartment2 = similar_apartments.loc[1, 'url']

                # Прогнозы
                self.price_4 = round(df2.loc[df2['location'] == an, 'price_4'].values[0], 1)
                self.price_2 = round(df2.loc[df2['location'] == an, 'price_2'].values[0], 1)
                self.feb_apr = round(df2.loc[df2['location'] == an, 'feb_apr, %'].values[0], 1)
                self.apr_may = round(df2.loc[df2['location'] == an, 'apr_may, %'].values[0], 1)
                self.jun_jul = round(df2.loc[df2['location'] == an, 'jun_jul, %'].values[0], 1)
                self.jul_aug = round(df2.loc[df2['location'] == an, 'jul_aug, %'].values[0], 1)

                # Оценка
                floors_count = int(self.floors_count)
                floor =  int(self.floor)
                total_meters =  int(self.tot_meters)
                living_meters =  int(self.living_meters)
                kitchen_meters =  int(self.kitchen_meters)
                rooms_count =  int(self.rooms_count)
                year_of_construction =  int(self.year_of_construction)
                price_per_m2 =  int(self.price_per_m2)

                score = main.calculate_score(floors_count, floor, total_meters, living_meters, 
                                        kitchen_meters, rooms_count, year_of_construction,
                                        price_per_m2)
                

                # Новый список для записи
                self.price = round(float(str(self.price / 1000000).replace('.0', '')), 1)
                      
                if len(str(abs(self.price))) > 4:
                        self.price = "{:.1f}".format(self.price / 10**(len(str(abs(int(self.price)))) - 1))

                      


                new_row = [self.num_rows, self.now_data, score, int(self.floors_count), int(self.floor), self.tot_meters, 
                        int(self.living_meters), int(self.kitchen_meters), self.rooms_count, int(self.year_of_construction),
                        self.comboBox.currentText(), self.comboBox_4.currentText(), self.comboBox_3.currentText(),
                        self.comboBox_5.currentText(), self.price, self.price_4, self.price_2, similar_apartment1,
                        similar_apartment2, self.price_per_m2, self.feb_apr, self.apr_may, self.jun_jul, self.jul_aug]


                with open('info.csv', 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                        writer.writerow(new_row)
                
                while self.verticalLayout_40.count() > 0:
                        self.verticalLayout_40.takeAt(0)

                _translate = QtCore.QCoreApplication.translate
                with open('info.csv', 'r') as file:
                        self.frame.hide()
                        self.frame_2.hide()
                        self.empty.show()

                        num_lines = len(file.readlines()) - 1
                        if num_lines == 0:
                                num_lines = ""
                        self.pushButton_3.setText(_translate("MainWindow", f"ВСЕ {num_lines}"))

                        if num_lines == 1:
                                self.frame.show()  # показать frame
                                self.empty.hide()
                        
                        if num_lines >= 2:
                                self.frame_2.show()
                                self.frame.show()
                                self.empty.hide()

                self.upd_inform()
                self.addSlotToHistory()
                self.addSlotToHistory2()
                self.empty_4.hide()

                self.opening_information(self.num_rows - 1)

        except Exception:
                return

    def opening_information(self, id_i):
        self.stackedWidget.setCurrentWidget(self.pahe_info) # Открытие страницы
        dataset = pd.read_csv("info.csv")
        df1 = main.df1

        self.label_337.setText(f"{dataset.iloc[id_i, 14]} млн, р")
        self.label_340.setText(f"{round(int(dataset.iloc[id_i, 19])/1000, 1)} тыс, р")

        if float(dataset.iloc[id_i, 15]) > 0:
              ant1 = "▲"
        else:
              ant1 = "▼"
              
        if float(dataset.iloc[id_i, 16]) > 0:
              ant2 = "▲"
        else:
              ant2 = "▼"
        

        self.label_342.setText(f"<html><head/><body><p>Стоимость м² этой квартиры изменилась <br/>на <span style=\" color:#396aea;\">{ant1}{dataset.iloc[id_i, 15]}%</span> за 4 месяца</p></body></html>")
        self.label_344.setText(f"<html><head/><body><p>Вероятное изменение стоимости м² этой<br/>квартиры в течении 2 месяцев <span style=\" color:#aa64c3;\">{ant2}{dataset.iloc[id_i, 16]}%</span></p></body></html>")
        self.label_333.setText(f"{dataset.iloc[id_i, 8]}-комн. квартира · {dataset.iloc[id_i, 5]} м²")
        self.label_334.setText(f"{dataset.iloc[id_i, 10]}, {dataset.iloc[id_i, 12]}, {dataset.iloc[id_i, 4]}/{dataset.iloc[id_i, 3]} этаж")


        filtered_df = df1.loc[df1['url'] == dataset.iloc[id_i, 17]]  # Фильтрация строк
        m2 = int(filtered_df.iloc[0, 7])
        pr = float(filtered_df.iloc[0, 10])/1000000
        ct = int(filtered_df.iloc[0, 1])
        ct = [k for k, v in location_json.items() if v == ct][0]
        sr = int(filtered_df.iloc[0, 12])
        sr = [k for k, v in main.street_dictionary.items() if v == sr][0]
        ds = int(filtered_df.iloc[0, 11])
        ds = [k for k, v in main.district_dictionary.items() if v == ds][0]
        cmp = int(filtered_df.iloc[0, 13])
        cmp = [k for k, v in main.residential_complex_dictionary.items() if v == cmp][0]
        fl = int(filtered_df.iloc[0, 2])
        flc = int(filtered_df.iloc[0, 3])

        if cmp == "Отсутствует":
              cmp = ""
        else:
              cmp = f", {cmp}"

        if ds == "Отсутствует":
              ds = ""
        else:
              ds = f", {ds}"   

        text = f"{ct}, {sr}{cmp}{ds}, {fl}/{flc} этаж"
        if len(text) > 64:
                prof = len(text) - 64
                ds = ds[:-prof]
                text = f"{ct}, {sr}{cmp}{ds}, {fl}/{flc} этаж"
                
        text = f"{text[:21]}\n{text[21:42]}\n{text[42:]}"
        self.pushButton_8.clicked.connect(lambda: webbrowser.open(dataset.iloc[id_i, 17]))
        self.label_349.setText(f"{m2} м², {pr} млн, руб")
        self.label_350.setText(text)


        filtered_df = df1.loc[df1['url'] == dataset.iloc[id_i, 18]]  # Фильтрация строк
        m2 = int(filtered_df.iloc[0, 7])
        pr = float(filtered_df.iloc[0, 10])/1000000
        ct = int(filtered_df.iloc[0, 1])
        ct = [k for k, v in location_json.items() if v == ct][0]
        sr = int(filtered_df.iloc[0, 12])
        sr = [k for k, v in main.street_dictionary.items() if v == sr][0]
        ds = int(filtered_df.iloc[0, 11])
        ds = [k for k, v in main.district_dictionary.items() if v == ds][0]
        cmp = int(filtered_df.iloc[0, 13])
        cmp = [k for k, v in main.residential_complex_dictionary.items() if v == cmp][0]
        fl = int(filtered_df.iloc[0, 2])
        flc = int(filtered_df.iloc[0, 3])

        if cmp == "Отсутствует":
              cmp = ""
        else:
              cmp = f", {cmp}"

        if ds == "Отсутствует":
              ds = ""
        else:
              ds = f", {ds}"   

        text = f"{ct}, {sr}{cmp}{ds}, {fl}/{flc} этаж"
        if len(text) > 64:
                prof = len(text) - 64
                ds = ds[:-prof]
                text = f"{ct}, {sr}{cmp}{ds}, {fl}/{flc} этаж"

        text = f"{text[:21]}\n{text[21:42]}\n{text[42:]}"
        self.pushButton_10.clicked.connect(lambda: webbrowser.open(dataset.iloc[id_i, 18]))
        self.label_351.setText(f"{m2} м², {pr} млн, руб")
        self.label_352.setText(text)


#############################################################

        a1 = float(dataset.iloc[id_i, 19])

        if float(dataset.iloc[id_i, 21]) > 0:
                a2 = a1 - (a1 * (float(dataset.iloc[id_i, 21]) / 100))
        elif float(dataset.iloc[id_i, 21]) < 0:
                a2 = a1 + (a1 * (float(dataset.iloc[id_i, 21]) / 100))
        else:
                a2 = a1

        if float(dataset.iloc[id_i, 20]) > 0:
                a3 = a2 - (a2 * (float(dataset.iloc[id_i, 20]) / 100))
        elif float(dataset.iloc[id_i, 20]) < 0:
                a3 = a2 + (a2 * (float(dataset.iloc[id_i, 20]) / 100))
        else:
                a3 = a2

        self.txt = [round(a1/1000, 1), round(a2/1000, 1), round(a3/1000, 1)]

        print(self.txt)

        # находим наибольшее значение
        max_value = max(a1, a2, a3)
        # находим наименьшее значение
        min_value = min(a1, a2, a3)

        # присваиваем новые значения
        a1 = 155 if a1 == min_value else 120 if a1 == max_value else 142
        a2 = 155 if a2 == min_value else 120 if a2 == max_value else 142
        a3 = 155 if a3 == min_value else 120 if a3 == max_value else 142

        # self.y_coords = [a1, a2, a3]
        self.y_coords = [120, 120, 120]

        print(self.y_coords)

#############################################################


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateProgressBar)
        self.timer.start(2) # Искусственная загрузка для подгрузки данных
        self.values = list(range(1, 1001))
        self.current_index = 0


    def updateProgressBar(self):
        if self.current_index < len(self.values):
            self.progressBar.setValue(self.values[self.current_index])
            self.current_index += 1
        else:
            self.timer.stop()
            self.progress.hide()


    def applyShadowEffect(self, widgets):
        for widget in widgets:
            shadow = QGraphicsDropShadowEffect()
            shadow.setColor(QColor(0, 0, 0, 50))
            shadow.setBlurRadius(5)
            shadow.setXOffset(0) # Смещение сбоку
            shadow.setYOffset(2) # Смещение снизу
            widget.setGraphicsEffect(shadow)


    def add_commas(self, line_edit):
        current_text = line_edit.text()
        
        if current_text:
                current_value = re.sub(r'\D', "", current_text)
                formatted_value = "{:,}".format(int(current_value))
                line_edit.blockSignals(True)
                line_edit.setText(formatted_value)
                line_edit.setCursorPosition(len(formatted_value))
                line_edit.blockSignals(False)


    def comboBox1Changed(self, text):
        if text in street_json: #  текст = текущее значение
                values = street_json[text] # заполнить опираясь на текст
                self.comboBox_3.clear()
                self.comboBox_3.addItems(values)


    def comboBox2Changed(self, text):
        if text in residential_complex_json:
                values = residential_complex_json[text]
                if len(values) > 1:
                        new_values = []
                        for value in values:
                                if value != "Отсутствует":
                                        new_values.append(value)
                        self.comboBox_5.clear()
                        self.comboBox_5.addItems(new_values)       
                else:
                        self.comboBox_5.clear()
                        self.comboBox_5.addItems(values)


    def comboBox3Changed(self, text):
        if text in district_json:
                values = district_json[text]
                if len(values) > 1:
                        new_values = []
                        for value in values:
                                if value != "Отсутствует":
                                        new_values.append(value)
                        self.comboBox_4.clear()
                        self.comboBox_4.addItems(new_values)       
                else:
                        self.comboBox_4.clear()
                        self.comboBox_4.addItems(values)



    def setupUi(self, MainWindow):
        self.added_items = set()
        icon_dir = os.path.join(os.path.dirname(__file__), '..', 'program', 'icons')
        icon_path = os.path.join(icon_dir, 'Chevron Down.png').replace('\\', '/')
        validator = QIntValidator()
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(360, 800)
        MainWindow.setMinimumSize(QtCore.QSize(360, 800))
        MainWindow.setMaximumSize(QtCore.QSize(360, 800))
        MainWindow.setStyleSheet("background-color: rgb(230, 230, 230);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setMinimumSize(QtCore.QSize(360, 800))
        self.centralwidget.setMaximumSize(QtCore.QSize(1080, 1400))
        font = QtGui.QFont()
        font.setKerning(True)
        self.centralwidget.setFont(font)
        self.centralwidget.setStyleSheet("background-color:rgb(230, 230, 230)")
        self.centralwidget.setObjectName("centralwidget")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 0, 360, 1200))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy)
        self.stackedWidget.setMinimumSize(QtCore.QSize(360, 800))
        self.stackedWidget.setMaximumSize(QtCore.QSize(360, 2400))
        self.stackedWidget.setBaseSize(QtCore.QSize(300, 0))
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_main = QtWidgets.QWidget()
        self.page_main.setMinimumSize(QtCore.QSize(360, 0))
        self.page_main.setMaximumSize(QtCore.QSize(360, 16777215))
        self.page_main.setObjectName("page_main")
        self.text_main = QtWidgets.QLabel(self.page_main)
        self.text_main.setGeometry(QtCore.QRect(20, 84, 322, 40))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.text_main.setFont(font)
        self.text_main.setObjectName("text_main")
        self.apartment_comparison = QtWidgets.QFrame(self.page_main)
        self.apartment_comparison.setGeometry(QtCore.QRect(19, 450, 322, 67))
        self.apartment_comparison.setMinimumSize(QtCore.QSize(310, 67))
        self.apartment_comparison.setMaximumSize(QtCore.QSize(335, 67))
        self.apartment_comparison.setSizeIncrement(QtCore.QSize(335, 67))
        self.apartment_comparison.setBaseSize(QtCore.QSize(335, 57))
        self.apartment_comparison.setStyleSheet("background-color: qlineargradient(spread:reflect, x1:0.256, y1:0.25, x2:0.966, y2:1, stop:0 rgba(47, 22, 148, 255), stop:1 rgba(106, 45, 143, 255));\n"
"border: none;\n"
"border-radius: 21px;")
        self.apartment_comparison.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.apartment_comparison.setFrameShadow(QtWidgets.QFrame.Raised)
        self.apartment_comparison.setObjectName("apartment_comparison")
        self.pushButton_2 = QtWidgets.QPushButton(self.apartment_comparison)
        self.pushButton_2.setGeometry(QtCore.QRect(0, 0, 321, 67))
        self.pushButton_2.setMinimumSize(QtCore.QSize(310, 67))
        self.pushButton_2.setMaximumSize(QtCore.QSize(321, 67))
        self.pushButton_2.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color:rgba(255, 255, 255, 25)\n"
"}")
        self.pushButton_2.setText("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_comparison))
        self.layoutWidget_36 = QtWidgets.QWidget(self.apartment_comparison)
        self.layoutWidget_36.setGeometry(QtCore.QRect(14, 0, 305, 69))
        self.layoutWidget_36.setObjectName("layoutWidget_36")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.layoutWidget_36)
        self.horizontalLayout_6.setContentsMargins(0, 12, 7, 0)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setContentsMargins(0, 4, -1, 18)
        self.verticalLayout_9.setSpacing(0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_10 = QtWidgets.QLabel(self.layoutWidget_36)
        self.label_10.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.label_10.setObjectName("label_10")
        self.verticalLayout_9.addWidget(self.label_10)
        self.label_11 = QtWidgets.QLabel(self.layoutWidget_36)
        self.label_11.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.label_11.setObjectName("label_11")
        self.verticalLayout_9.addWidget(self.label_11)
        self.horizontalLayout_6.addLayout(self.verticalLayout_9)
        self.label_15 = QtWidgets.QLabel(self.layoutWidget_36)
        self.label_15.setMaximumSize(QtCore.QSize(50, 56))
        self.label_15.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.label_15.setText("")
        self.label_15.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, 'Group2.png')))
        self.label_15.setScaledContents(True)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_6.addWidget(self.label_15)
        self.layoutWidget_36.raise_()
        self.pushButton_2.raise_()
        self.recent_estimates = QtWidgets.QWidget(self.page_main)
        self.recent_estimates.setGeometry(QtCore.QRect(19, 530, 322, 255))
        self.recent_estimates.setMinimumSize(QtCore.QSize(310, 255))
        self.recent_estimates.setMaximumSize(QtCore.QSize(335, 255))
        self.recent_estimates.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;")
        self.recent_estimates.setObjectName("recent_estimates")
        self.layoutWidget_37 = QtWidgets.QWidget(self.recent_estimates)
        self.layoutWidget_37.setGeometry(QtCore.QRect(20, 12, 281, 71))
        self.layoutWidget_37.setObjectName("layoutWidget_37")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget_37)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label = QtWidgets.QLabel(self.layoutWidget_37)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_5.addWidget(self.label, 0, QtCore.Qt.AlignLeft)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget_37)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_5.addWidget(self.label_2, 0, QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.horizontalLayout_2.addLayout(self.verticalLayout_5)
        self.pushButton_3 = QtWidgets.QPushButton(self.layoutWidget_37)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("QPushButton {\n"
"color:rgb(96, 178, 255)\n"
"}\n"
"\n"
"QPushButton:pressed {color: rgb(255, 41, 152);}")
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_2.addWidget(self.pushButton_3, 0, QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.frame = QtWidgets.QFrame(self.recent_estimates)
        self.frame.setGeometry(QtCore.QRect(16, 92, 290, 67))
        self.frame.setMinimumSize(QtCore.QSize(290, 67))
        self.frame.setMaximumSize(QtCore.QSize(290, 67))
        self.frame.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame.hide()
        self.pushButton_5 = QtWidgets.QPushButton(self.frame)
        self.pushButton_5.setGeometry(QtCore.QRect(0, 0, 290, 67))
        self.pushButton_5.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border-radius: 17px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton_5.setText("")
        self.pushButton_5.setObjectName("pushButton_5")
        self.layoutWidget_38 = QtWidgets.QWidget(self.frame)
        self.layoutWidget_38.setGeometry(QtCore.QRect(0, 3, 286, 66))
        self.layoutWidget_38.setObjectName("layoutWidget_38")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.layoutWidget_38)
        self.verticalLayout_8.setContentsMargins(15, 8, 0, 14)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_9 = QtWidgets.QLabel(self.layoutWidget_38)
        self.label_9.setMaximumSize(QtCore.QSize(270, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_8.addWidget(self.label_9)
        self.label_12 = QtWidgets.QLabel(self.layoutWidget_38)
        self.label_12.setMaximumSize(QtCore.QSize(270, 16777215))
        self.label_12.setObjectName("label_12")
        self.verticalLayout_8.addWidget(self.label_12)
        self.layoutWidget_38.raise_()
        self.pushButton_5.raise_()
        self.frame_2 = QtWidgets.QFrame(self.recent_estimates)
        self.frame_2.setGeometry(QtCore.QRect(16, 171, 290, 67))
        self.frame_2.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_2.setMaximumSize(QtCore.QSize(290, 67))
        self.frame_2.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.frame_2.hide()
        self.pushButton_6 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_6.setGeometry(QtCore.QRect(0, 0, 290, 67))
        self.pushButton_6.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border-radius: 17px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton_6.setText("")
        self.pushButton_6.setObjectName("pushButton_6")
        self.frame_5 = QtWidgets.QFrame(self.frame_2)
        self.frame_5.setGeometry(QtCore.QRect(0, 0, 290, 67))
        self.frame_5.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_5.setMaximumSize(QtCore.QSize(290, 67))
        self.frame_5.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.pushButton_9 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_9.setGeometry(QtCore.QRect(0, 0, 290, 67))
        self.pushButton_9.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border-radius: 17px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton_9.setText("")
        self.pushButton_9.setObjectName("pushButton_9")
        self.layoutWidget_53 = QtWidgets.QWidget(self.frame_5)
        self.layoutWidget_53.setGeometry(QtCore.QRect(0, 3, 257, 66))
        self.layoutWidget_53.setObjectName("layoutWidget_53")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.layoutWidget_53)
        self.verticalLayout_11.setContentsMargins(15, 8, 0, 14)
        self.verticalLayout_11.setSpacing(0)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.label_16 = QtWidgets.QLabel(self.layoutWidget_53)
        self.label_16.setMaximumSize(QtCore.QSize(270, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.verticalLayout_11.addWidget(self.label_16)
        self.label_17 = QtWidgets.QLabel(self.layoutWidget_53)
        self.label_17.setMaximumSize(QtCore.QSize(270, 16777215))
        self.label_17.setObjectName("label_17")
        self.verticalLayout_11.addWidget(self.label_17)
        self.layoutWidget_53.raise_()
        self.pushButton_9.raise_()
        self.empty = QtWidgets.QLabel(self.recent_estimates)
        self.empty.setGeometry(QtCore.QRect(0, 90, 321, 151))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.empty.setFont(font)
        self.empty.setStyleSheet("color: rgb(131, 131, 131);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.empty.setAlignment(QtCore.Qt.AlignCenter)
        self.empty.setObjectName("empty")
        self.layoutWidget_37.raise_()
        self.empty.raise_()
        self.frame.raise_()
        self.frame_2.raise_()
        self.widget_14 = QtWidgets.QWidget(self.page_main)
        self.widget_14.setGeometry(QtCore.QRect(19, 145, 322, 291))
        self.widget_14.setStyleSheet("background-color: rgb(81, 36, 146);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_14.setObjectName("widget_14")
        self.widget = QtWidgets.QWidget(self.widget_14)
        self.widget.setGeometry(QtCore.QRect(0, 30, 322, 261))
        self.widget.setMinimumSize(QtCore.QSize(310, 182))
        self.widget.setMaximumSize(QtCore.QSize(335, 400))
        self.widget.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget.setObjectName("widget")
        self.layoutWidget_54 = QtWidgets.QWidget(self.widget)
        self.layoutWidget_54.setGeometry(QtCore.QRect(0, 0, 321, 261))
        self.layoutWidget_54.setObjectName("layoutWidget_54")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.layoutWidget_54)
        self.verticalLayout_4.setContentsMargins(16, 3, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.calculation = QtWidgets.QFrame(self.layoutWidget_54)
        self.calculation.setMinimumSize(QtCore.QSize(0, 67))
        self.calculation.setMaximumSize(QtCore.QSize(290, 67))
        self.calculation.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.calculation.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.calculation.setFrameShadow(QtWidgets.QFrame.Raised)
        self.calculation.setObjectName("calculation")
        self.pushButton = QtWidgets.QPushButton(self.calculation)
        self.pushButton.setGeometry(QtCore.QRect(0, 0, 290, 67))
        self.pushButton.setMinimumSize(QtCore.QSize(290, 67))
        self.pushButton.setMaximumSize(QtCore.QSize(290, 67))
        self.pushButton.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border-radius: 17px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton.setText("")
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.pahe_signs))
        self.layoutWidget_55 = QtWidgets.QWidget(self.calculation)
        self.layoutWidget_55.setGeometry(QtCore.QRect(0, 0, 272, 65))
        self.layoutWidget_55.setObjectName("layoutWidget_55")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget_55)
        self.horizontalLayout_3.setContentsMargins(11, 5, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.layoutWidget_55)
        self.label_3.setMaximumSize(QtCore.QSize(36, 36))
        self.label_3.setStyleSheet("")
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "icon _receipt_tax_ 2.png")))
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setContentsMargins(-1, 12, -1, 15)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_5 = QtWidgets.QLabel(self.layoutWidget_55)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("")
        self.label_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_6.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.layoutWidget_55)
        self.label_6.setStyleSheet("")
        self.label_6.setObjectName("label_6")
        self.verticalLayout_6.addWidget(self.label_6)
        self.horizontalLayout_3.addLayout(self.verticalLayout_6)
        self.layoutWidget_55.raise_()
        self.pushButton.raise_()
        self.verticalLayout_4.addWidget(self.calculation)
        self.tax = QtWidgets.QFrame(self.layoutWidget_54)
        self.tax.setMinimumSize(QtCore.QSize(0, 67))
        self.tax.setMaximumSize(QtCore.QSize(290, 67))
        self.tax.setSizeIncrement(QtCore.QSize(290, 67))
        self.tax.setBaseSize(QtCore.QSize(290, 67))
        self.tax.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.tax.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tax.setFrameShadow(QtWidgets.QFrame.Raised)
        self.tax.setObjectName("tax")
        self.pushButton_4 = QtWidgets.QPushButton(self.tax)
        self.pushButton_4.setGeometry(QtCore.QRect(0, 0, 290, 67))
        self.pushButton_4.setMinimumSize(QtCore.QSize(290, 67))
        self.pushButton_4.setMaximumSize(QtCore.QSize(290, 67))
        self.pushButton_4.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border-radius: 17px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton_4.setText("")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_calculator))
        self.layoutWidget_56 = QtWidgets.QWidget(self.tax)
        self.layoutWidget_56.setGeometry(QtCore.QRect(0, 0, 279, 68))
        self.layoutWidget_56.setObjectName("layoutWidget_56")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget_56)
        self.horizontalLayout_4.setContentsMargins(11, 5, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget_56)
        self.label_4.setMaximumSize(QtCore.QSize(36, 36))
        self.label_4.setStyleSheet("")
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "icon_receipt_tax_.png")))
        self.label_4.setScaledContents(True)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setContentsMargins(-1, 12, -1, 15)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_7 = QtWidgets.QLabel(self.layoutWidget_56)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("")
        self.label_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_7.addWidget(self.label_7)
        self.label_8 = QtWidgets.QLabel(self.layoutWidget_56)
        self.label_8.setStyleSheet("")
        self.label_8.setObjectName("label_8")
        self.verticalLayout_7.addWidget(self.label_8)
        self.horizontalLayout_4.addLayout(self.verticalLayout_7)
        self.layoutWidget_56.raise_()
        self.pushButton_4.raise_()
        self.verticalLayout_4.addWidget(self.tax)
        self.tax_2 = QtWidgets.QFrame(self.layoutWidget_54)
        self.tax_2.setMinimumSize(QtCore.QSize(0, 67))
        self.tax_2.setMaximumSize(QtCore.QSize(290, 67))
        self.tax_2.setSizeIncrement(QtCore.QSize(290, 67))
        self.tax_2.setBaseSize(QtCore.QSize(290, 67))
        self.tax_2.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.tax_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tax_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.tax_2.setObjectName("tax_2")
        self.pushButton_7 = QtWidgets.QPushButton(self.tax_2)
        self.pushButton_7.setGeometry(QtCore.QRect(0, 0, 290, 67))
        self.pushButton_7.setMinimumSize(QtCore.QSize(290, 67))
        self.pushButton_7.setMaximumSize(QtCore.QSize(290, 67))
        self.pushButton_7.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border-radius: 17px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton_7.setText("")
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_7.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_mortgage))
        self.layoutWidget_58 = QtWidgets.QWidget(self.tax_2)
        self.layoutWidget_58.setGeometry(QtCore.QRect(0, 0, 279, 68))
        self.layoutWidget_58.setObjectName("layoutWidget_58")
        self.horizontalLayout_52 = QtWidgets.QHBoxLayout(self.layoutWidget_58)
        self.horizontalLayout_52.setContentsMargins(11, 5, 0, 0)
        self.horizontalLayout_52.setObjectName("horizontalLayout_52")
        self.label_13 = QtWidgets.QLabel(self.layoutWidget_58)
        self.label_13.setMaximumSize(QtCore.QSize(36, 36))
        self.label_13.setStyleSheet("")
        self.label_13.setText("")
        self.label_13.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Debt.png")))
        self.label_13.setScaledContents(True)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_52.addWidget(self.label_13)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setContentsMargins(-1, 12, -1, 15)
        self.verticalLayout_10.setSpacing(0)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_14 = QtWidgets.QLabel(self.layoutWidget_58)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setStyleSheet("")
        self.label_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_14.setObjectName("label_14")
        self.verticalLayout_10.addWidget(self.label_14)
        self.label_82 = QtWidgets.QLabel(self.layoutWidget_58)
        self.label_82.setStyleSheet("")
        self.label_82.setObjectName("label_82")
        self.verticalLayout_10.addWidget(self.label_82)
        self.horizontalLayout_52.addLayout(self.verticalLayout_10)
        self.layoutWidget_58.raise_()
        self.pushButton_7.raise_()
        self.verticalLayout_4.addWidget(self.tax_2)
        self.layoutWidget_57 = QtWidgets.QWidget(self.page_main)
        self.layoutWidget_57.setGeometry(QtCore.QRect(0, 0, 361, 61))
        self.layoutWidget_57.setObjectName("layoutWidget_57")
        self.horizontalLayout_53 = QtWidgets.QHBoxLayout(self.layoutWidget_57)
        self.horizontalLayout_53.setContentsMargins(10, 3, 10, 0)
        self.horizontalLayout_53.setSpacing(0)
        self.horizontalLayout_53.setObjectName("horizontalLayout_53")
        self.logo_8 = QtWidgets.QLabel(self.layoutWidget_57)
        self.logo_8.setMinimumSize(QtCore.QSize(40, 44))
        self.logo_8.setMaximumSize(QtCore.QSize(40, 44))
        self.logo_8.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.logo_8.setText("")
        self.logo_8.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Logo.png")))
        self.logo_8.setScaledContents(True)
        self.logo_8.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.logo_8.setObjectName("logo_8")
        self.horizontalLayout_53.addWidget(self.logo_8)
        self.pushButton_35 = QtWidgets.QPushButton(self.layoutWidget_57)
        self.pushButton_35.setMinimumSize(QtCore.QSize(38, 38))
        self.pushButton_35.setMaximumSize(QtCore.QSize(38, 38))
        self.pushButton_35.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.pushButton_35.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Help.png")))
        self.pushButton_35.setIcon(icon)
        self.pushButton_35.setIconSize(QtCore.QSize(38, 38))
        self.pushButton_35.setObjectName("pushButton_35")
        self.horizontalLayout_53.addWidget(self.pushButton_35, 0, QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.pushButton_35.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.pahe_help))

        self.avatar = QtWidgets.QLabel(self.page_main)
        self.avatar.setGeometry(QtCore.QRect(0, 0, 360, 800))
        self.avatar.setText("")
        self.avatar.setStyleSheet("background-color: transparent;")  # Непрозрачный фон
        self.avatar.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Avatar.png")))
        self.avatar.setScaledContents(True)
        self.avatar.setObjectName("avatar")
        self.avatar.raise_()
        self.avatar.mousePressEvent = lambda event: self.avatar.hide()

        self.logo = QtWidgets.QLabel(self.page_main)
        self.logo.setGeometry(QtCore.QRect(0, 0, 360, 800))
        self.logo.setText("")
        self.logo.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "screensaver.png")))
        self.logo.setScaledContents(True)
        self.logo.setObjectName("logo")

        # Создаем QTimer для скрытия логотипа
        self.timer = QTimer()
        self.timer.timeout.connect(self.hide_logo)
        self.timer.start(3000)  # Запускаем таймер на 3 секунды

        self.stackedWidget.addWidget(self.page_main)
        self.page_comparison = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.page_comparison.sizePolicy().hasHeightForWidth())
        self.page_comparison.setSizePolicy(sizePolicy)
        self.page_comparison.setObjectName("page_comparison")
        self.scrollArea_4 = QtWidgets.QScrollArea(self.page_comparison)
        self.scrollArea_4.setGeometry(QtCore.QRect(0, 8, 361, 800))
        self.scrollArea_4.setMinimumSize(QtCore.QSize(0, 800))
        self.scrollArea_4.setMaximumSize(QtCore.QSize(16777215, 800))
        self.scrollArea_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.scrollArea_4.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollArea_4.setObjectName("scrollArea_4")
        self.scrollAreaWidgetContents_4 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_4.setGeometry(QtCore.QRect(0, 0, 359, 1770))
        self.scrollAreaWidgetContents_4.setMinimumSize(QtCore.QSize(0, 1770))
        self.scrollAreaWidgetContents_4.setMaximumSize(QtCore.QSize(360, 1770))
        self.scrollAreaWidgetContents_4.setObjectName("scrollAreaWidgetContents_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_4)
        self.gridLayout_4.setContentsMargins(8, 0, 0, 0)
        self.gridLayout_4.setSpacing(0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.frame_30 = QtWidgets.QFrame(self.scrollAreaWidgetContents_4)
        self.frame_30.setMaximumSize(QtCore.QSize(16777215, 1800))
        self.frame_30.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_30.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_30.setObjectName("frame_30")
        self.text_main_7 = QtWidgets.QLabel(self.frame_30)
        self.text_main_7.setGeometry(QtCore.QRect(9, 79, 341, 33))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.text_main_7.setFont(font)
        self.text_main_7.setObjectName("text_main_7")
        self.layoutWidget_39 = QtWidgets.QWidget(self.frame_30)
        self.layoutWidget_39.setGeometry(QtCore.QRect(10, 120, 331, 1631))
        self.layoutWidget_39.setObjectName("layoutWidget_39")
        self.verticalLayout_59 = QtWidgets.QVBoxLayout(self.layoutWidget_39)
        self.verticalLayout_59.setContentsMargins(0, 6, 0, 0)
        self.verticalLayout_59.setObjectName("verticalLayout_59")
        self.widget_10 = QtWidgets.QWidget(self.layoutWidget_39)
        self.widget_10.setMinimumSize(QtCore.QSize(310, 0))
        self.widget_10.setMaximumSize(QtCore.QSize(322, 161))
        self.widget_10.setStyleSheet("background-color: rgb(216, 117, 117);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_10.setObjectName("widget_10")
        self.widget_11 = QtWidgets.QWidget(self.widget_10)
        self.widget_11.setGeometry(QtCore.QRect(0, 30, 322, 131))
        self.widget_11.setMinimumSize(QtCore.QSize(310, 0))
        self.widget_11.setMaximumSize(QtCore.QSize(335, 200))
        self.widget_11.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_11.setObjectName("widget_11")
        self.layoutWidget_40 = QtWidgets.QWidget(self.widget_11)
        self.layoutWidget_40.setGeometry(QtCore.QRect(0, 0, 321, 131))
        self.layoutWidget_40.setObjectName("layoutWidget_40")
        self.verticalLayout_41 = QtWidgets.QVBoxLayout(self.layoutWidget_40)
        self.verticalLayout_41.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_41.setObjectName("verticalLayout_41")
        self.label_87 = QtWidgets.QLabel(self.layoutWidget_40)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_87.setFont(font)
        self.label_87.setObjectName("label_87")
        self.verticalLayout_41.addWidget(self.label_87)
        self.horizontalLayout_35 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_35.setSpacing(10)
        self.horizontalLayout_35.setObjectName("horizontalLayout_35")
        self.widget_12 = QtWidgets.QWidget(self.layoutWidget_40)
        self.widget_12.setMinimumSize(QtCore.QSize(0, 70))
        self.widget_12.setMaximumSize(QtCore.QSize(142, 70))
        self.widget_12.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 12px;")
        self.widget_12.setObjectName("widget_12")
        self.pushButton_29 = QtWidgets.QPushButton(self.widget_12)
        self.pushButton_29.setGeometry(QtCore.QRect(0, -1, 142, 70))
        self.pushButton_29.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border: none;\n"
"border-radius: 12px;\n"
"}\n"
"\n")
        self.pushButton_29.setText("")
        self.pushButton_29.setObjectName("pushButton_29")
        self.empty_3 = QtWidgets.QLabel(self.widget_12)
        self.empty_3.setEnabled(True)
        self.empty_3.setGeometry(QtCore.QRect(0, 0, 141, 71))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.empty_3.setFont(font)
        self.empty_3.setStyleSheet("color: rgb(131, 131, 131);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.empty_3.setAlignment(QtCore.Qt.AlignCenter)
        self.empty_3.setObjectName("empty_3")
        self.layoutWidget_41 = QtWidgets.QWidget(self.widget_12)
        self.layoutWidget_41.setGeometry(QtCore.QRect(7, 3, 133, 61))
        self.layoutWidget_41.setObjectName("layoutWidget_41")
        self.verticalLayout_42 = QtWidgets.QVBoxLayout(self.layoutWidget_41)
        self.verticalLayout_42.setContentsMargins(0, 5, 0, 3)
        self.verticalLayout_42.setSpacing(0)
        self.verticalLayout_42.setObjectName("verticalLayout_42")
        self.label_88 = QtWidgets.QLabel(self.layoutWidget_41)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_88.setFont(font)
        self.label_88.setObjectName("label_88")
        self.verticalLayout_42.addWidget(self.label_88)
        self.label_89 = QtWidgets.QLabel(self.layoutWidget_41)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_89.setFont(font)
        self.label_89.setObjectName("label_89")
        self.verticalLayout_42.addWidget(self.label_89)

        # Скрытие элекментов verticalLayout_42
        self.label_88.hide()
        self.label_89.hide()

        self.layoutWidget_41.raise_()
        self.empty_3.raise_()
        self.pushButton_29.raise_()
        self.pushButton_29.clicked.connect(self.move_widget_15)
        self.horizontalLayout_35.addWidget(self.widget_12)
        self.widget_13 = QtWidgets.QWidget(self.layoutWidget_40)
        self.widget_13.setMinimumSize(QtCore.QSize(0, 70))
        self.widget_13.setMaximumSize(QtCore.QSize(16777215, 70))
        self.widget_13.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 12px;")
        self.widget_13.setObjectName("widget_13")
        self.pushButton_30 = QtWidgets.QPushButton(self.widget_13)
        self.pushButton_30.setGeometry(QtCore.QRect(0, -1, 142, 70))
        self.pushButton_30.setMaximumSize(QtCore.QSize(142, 16777215))
        self.pushButton_30.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border: none;\n"
"border-radius: 12px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton_30.setText("")
        self.pushButton_30.setObjectName("pushButton_30")
        self.layoutWidget_42 = QtWidgets.QWidget(self.widget_13)
        self.layoutWidget_42.setGeometry(QtCore.QRect(7, 3, 134, 61))
        self.layoutWidget_42.setObjectName("layoutWidget_42")
        self.verticalLayout_43 = QtWidgets.QVBoxLayout(self.layoutWidget_42)
        self.verticalLayout_43.setContentsMargins(0, 5, 0, 3)
        self.verticalLayout_43.setSpacing(0)
        self.verticalLayout_43.setObjectName("verticalLayout_43")
        self.label_90 = QtWidgets.QLabel(self.layoutWidget_42)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_90.setFont(font)
        self.label_90.setObjectName("label_90")
        self.verticalLayout_43.addWidget(self.label_90)
        self.label_91 = QtWidgets.QLabel(self.layoutWidget_42)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_91.setFont(font)
        self.label_91.setObjectName("label_91")
        self.verticalLayout_43.addWidget(self.label_91)
        
        # Скрытие элекментов verticalLayout_43
        self.label_90.hide()
        self.label_91.hide()

        self.empty_2 = QtWidgets.QLabel(self.widget_13)
        self.empty_2.setEnabled(True)
        self.empty_2.setGeometry(QtCore.QRect(0, 0, 141, 71))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.empty_2.setFont(font)
        self.empty_2.setStyleSheet("color: rgb(131, 131, 131);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.empty_2.setAlignment(QtCore.Qt.AlignCenter)
        self.empty_2.setObjectName("empty_2")
        self.layoutWidget_42.raise_()
        self.empty_2.raise_()
        self.pushButton_30.raise_()
        self.horizontalLayout_35.addWidget(self.widget_13)
        self.verticalLayout_41.addLayout(self.horizontalLayout_35)
        self.verticalLayout_59.addWidget(self.widget_10)
        self.sign = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign.setMinimumSize(QtCore.QSize(322, 0))
        self.sign.setMaximumSize(QtCore.QSize(322, 80))
        self.sign.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign.setObjectName("sign")
        self.layoutWidget_43 = QtWidgets.QWidget(self.sign)
        self.layoutWidget_43.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_43.setObjectName("layoutWidget_43")
        self.verticalLayout_44 = QtWidgets.QVBoxLayout(self.layoutWidget_43)

        self.verticalLayout_44.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_44.setSpacing(5)
        self.verticalLayout_44.setObjectName("verticalLayout_44")
        self.label_92 = QtWidgets.QLabel(self.layoutWidget_43)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_92.setFont(font)
        self.label_92.setObjectName("label_92")
        self.verticalLayout_44.addWidget(self.label_92)
        self.horizontalLayout_36 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_36.setSpacing(10)
        self.horizontalLayout_36.setObjectName("horizontalLayout_36")
        self.label_95 = QtWidgets.QLabel(self.layoutWidget_43)
        self.label_95.setMinimumSize(QtCore.QSize(142, 26))
        self.label_95.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_95.setFont(font)
        self.label_95.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_95.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_95.setText("")
        self.label_95.setObjectName("label_95")
        self.horizontalLayout_36.addWidget(self.label_95)
        self.label_96 = QtWidgets.QLabel(self.layoutWidget_43)
        self.label_96.setMinimumSize(QtCore.QSize(142, 26))
        self.label_96.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_96.setFont(font)
        self.label_96.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_96.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_96.setText("")
        self.label_96.setObjectName("label_96")
        self.horizontalLayout_36.addWidget(self.label_96)
        self.verticalLayout_44.addLayout(self.horizontalLayout_36)
        self.verticalLayout_59.addWidget(self.sign)
        self.sign_2 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_2.setMinimumSize(QtCore.QSize(322, 0))
        self.sign_2.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_2.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_2.setObjectName("sign_2")
        self.layoutWidget_44 = QtWidgets.QWidget(self.sign_2)
        self.layoutWidget_44.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_44.setObjectName("layoutWidget_44")
        self.verticalLayout_45 = QtWidgets.QVBoxLayout(self.layoutWidget_44)
        self.verticalLayout_45.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_45.setSpacing(5)
        self.verticalLayout_45.setObjectName("verticalLayout_45")
        self.label_93 = QtWidgets.QLabel(self.layoutWidget_44)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_93.setFont(font)
        self.label_93.setObjectName("label_93")
        self.verticalLayout_45.addWidget(self.label_93)
        self.horizontalLayout_37 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_37.setSpacing(10)
        self.horizontalLayout_37.setObjectName("horizontalLayout_37")
        self.label_97 = QtWidgets.QLabel(self.layoutWidget_44)
        self.label_97.setMinimumSize(QtCore.QSize(142, 26))
        self.label_97.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_97.setFont(font)
        self.label_97.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_97.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_97.setText("")
        self.label_97.setObjectName("label_97")
        self.horizontalLayout_37.addWidget(self.label_97)
        self.label_98 = QtWidgets.QLabel(self.layoutWidget_44)
        self.label_98.setMinimumSize(QtCore.QSize(142, 26))
        self.label_98.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_98.setFont(font)
        self.label_98.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_98.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_98.setText("")
        self.label_98.setObjectName("label_98")
        self.horizontalLayout_37.addWidget(self.label_98)
        self.verticalLayout_45.addLayout(self.horizontalLayout_37)
        self.verticalLayout_59.addWidget(self.sign_2)
        self.sign_3 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_3.setMinimumSize(QtCore.QSize(310, 0))
        self.sign_3.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_3.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_3.setObjectName("sign_3")
        self.layoutWidget_45 = QtWidgets.QWidget(self.sign_3)
        self.layoutWidget_45.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_45.setObjectName("layoutWidget_45")
        self.verticalLayout_46 = QtWidgets.QVBoxLayout(self.layoutWidget_45)
        self.verticalLayout_46.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_46.setSpacing(5)
        self.verticalLayout_46.setObjectName("verticalLayout_46")
        self.label_94 = QtWidgets.QLabel(self.layoutWidget_45)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_94.setFont(font)
        self.label_94.setObjectName("label_94")
        self.verticalLayout_46.addWidget(self.label_94)
        self.horizontalLayout_38 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_38.setSpacing(10)
        self.horizontalLayout_38.setObjectName("horizontalLayout_38")
        self.label_99 = QtWidgets.QLabel(self.layoutWidget_45)
        self.label_99.setMinimumSize(QtCore.QSize(142, 26))
        self.label_99.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_99.setFont(font)
        self.label_99.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_99.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_99.setText("")
        self.label_99.setObjectName("label_99")
        self.horizontalLayout_38.addWidget(self.label_99)
        self.label_100 = QtWidgets.QLabel(self.layoutWidget_45)
        self.label_100.setMinimumSize(QtCore.QSize(142, 26))
        self.label_100.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_100.setFont(font)
        self.label_100.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_100.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_100.setText("")
        self.label_100.setObjectName("label_100")
        self.horizontalLayout_38.addWidget(self.label_100)
        self.verticalLayout_46.addLayout(self.horizontalLayout_38)
        self.verticalLayout_59.addWidget(self.sign_3)
        self.sign_4 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_4.setMinimumSize(QtCore.QSize(310, 0))
        self.sign_4.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_4.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_4.setObjectName("sign_4")
        self.layoutWidget_46 = QtWidgets.QWidget(self.sign_4)
        self.layoutWidget_46.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_46.setObjectName("layoutWidget_46")
        self.verticalLayout_47 = QtWidgets.QVBoxLayout(self.layoutWidget_46)
        self.verticalLayout_47.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_47.setSpacing(5)
        self.verticalLayout_47.setObjectName("verticalLayout_47")
        self.label_101 = QtWidgets.QLabel(self.layoutWidget_46)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_101.setFont(font)
        self.label_101.setObjectName("label_101")
        self.verticalLayout_47.addWidget(self.label_101)
        self.horizontalLayout_39 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_39.setSpacing(10)
        self.horizontalLayout_39.setObjectName("horizontalLayout_39")
        self.label_102 = QtWidgets.QLabel(self.layoutWidget_46)
        self.label_102.setMinimumSize(QtCore.QSize(142, 26))
        self.label_102.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_102.setFont(font)
        self.label_102.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_102.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_102.setText("")
        self.label_102.setObjectName("label_102")
        self.horizontalLayout_39.addWidget(self.label_102)
        self.label_103 = QtWidgets.QLabel(self.layoutWidget_46)
        self.label_103.setMinimumSize(QtCore.QSize(142, 26))
        self.label_103.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_103.setFont(font)
        self.label_103.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_103.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_103.setText("")
        self.label_103.setObjectName("label_103")
        self.horizontalLayout_39.addWidget(self.label_103)
        self.verticalLayout_47.addLayout(self.horizontalLayout_39)
        self.verticalLayout_59.addWidget(self.sign_4)
        self.sign_13 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_13.setMinimumSize(QtCore.QSize(310, 0))
        self.sign_13.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_13.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_13.setObjectName("sign_13")
        self.layoutWidget_50 = QtWidgets.QWidget(self.sign_13)
        self.layoutWidget_50.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_50.setObjectName("layoutWidget_50")
        self.verticalLayout_56 = QtWidgets.QVBoxLayout(self.layoutWidget_50)
        self.verticalLayout_56.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_56.setSpacing(5)
        self.verticalLayout_56.setObjectName("verticalLayout_56")
        self.label_128 = QtWidgets.QLabel(self.layoutWidget_50)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_128.setFont(font)
        self.label_128.setObjectName("label_128")
        self.verticalLayout_56.addWidget(self.label_128)
        self.horizontalLayout_49 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_49.setSpacing(10)
        self.horizontalLayout_49.setObjectName("horizontalLayout_49")
        self.label_129 = QtWidgets.QLabel(self.layoutWidget_50)
        self.label_129.setMinimumSize(QtCore.QSize(142, 26))
        self.label_129.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_129.setFont(font)
        self.label_129.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_129.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_129.setText("")
        self.label_129.setObjectName("label_129")
        self.horizontalLayout_49.addWidget(self.label_129)
        self.label_130 = QtWidgets.QLabel(self.layoutWidget_50)
        self.label_130.setMinimumSize(QtCore.QSize(142, 26))
        self.label_130.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_130.setFont(font)
        self.label_130.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_130.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_130.setText("")
        self.label_130.setObjectName("label_130")
        self.horizontalLayout_49.addWidget(self.label_130)
        self.verticalLayout_56.addLayout(self.horizontalLayout_49)
        self.verticalLayout_59.addWidget(self.sign_13)
        self.sign_6 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_6.setMinimumSize(QtCore.QSize(310, 0))
        self.sign_6.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_6.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_6.setObjectName("sign_6")
        self.layoutWidget_47 = QtWidgets.QWidget(self.sign_6)
        self.layoutWidget_47.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_47.setObjectName("layoutWidget_47")
        self.verticalLayout_49 = QtWidgets.QVBoxLayout(self.layoutWidget_47)
        self.verticalLayout_49.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_49.setSpacing(5)
        self.verticalLayout_49.setObjectName("verticalLayout_49")
        self.label_107 = QtWidgets.QLabel(self.layoutWidget_47)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_107.setFont(font)
        self.label_107.setObjectName("label_107")
        self.verticalLayout_49.addWidget(self.label_107)
        self.horizontalLayout_41 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_41.setSpacing(10)
        self.horizontalLayout_41.setObjectName("horizontalLayout_41")
        self.label_108 = QtWidgets.QLabel(self.layoutWidget_47)
        self.label_108.setMinimumSize(QtCore.QSize(142, 26))
        self.label_108.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_108.setFont(font)
        self.label_108.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_108.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_108.setText("")
        self.label_108.setObjectName("label_108")
        self.horizontalLayout_41.addWidget(self.label_108)
        self.label_109 = QtWidgets.QLabel(self.layoutWidget_47)
        self.label_109.setMinimumSize(QtCore.QSize(142, 26))
        self.label_109.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_109.setFont(font)
        self.label_109.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_109.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_109.setText("")
        self.label_109.setObjectName("label_109")
        self.horizontalLayout_41.addWidget(self.label_109)
        self.verticalLayout_49.addLayout(self.horizontalLayout_41)
        self.verticalLayout_59.addWidget(self.sign_6)
        self.sign_5 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_5.setMinimumSize(QtCore.QSize(310, 0))
        self.sign_5.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_5.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_5.setObjectName("sign_5")
        self.layoutWidget_49 = QtWidgets.QWidget(self.sign_5)
        self.layoutWidget_49.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_49.setObjectName("layoutWidget_49")
        self.verticalLayout_48 = QtWidgets.QVBoxLayout(self.layoutWidget_49)
        self.verticalLayout_48.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_48.setSpacing(5)
        self.verticalLayout_48.setObjectName("verticalLayout_48")
        self.label_104 = QtWidgets.QLabel(self.layoutWidget_49)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_104.setFont(font)
        self.label_104.setObjectName("label_104")
        self.verticalLayout_48.addWidget(self.label_104)
        self.horizontalLayout_40 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_40.setSpacing(10)
        self.horizontalLayout_40.setObjectName("horizontalLayout_40")
        self.label_105 = QtWidgets.QLabel(self.layoutWidget_49)
        self.label_105.setMinimumSize(QtCore.QSize(142, 26))
        self.label_105.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_105.setFont(font)
        self.label_105.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_105.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_105.setText("")
        self.label_105.setObjectName("label_105")
        self.horizontalLayout_40.addWidget(self.label_105)
        self.label_106 = QtWidgets.QLabel(self.layoutWidget_49)
        self.label_106.setMinimumSize(QtCore.QSize(142, 26))
        self.label_106.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_106.setFont(font)
        self.label_106.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_106.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_106.setText("")
        self.label_106.setObjectName("label_106")
        self.horizontalLayout_40.addWidget(self.label_106)
        self.verticalLayout_48.addLayout(self.horizontalLayout_40)
        self.verticalLayout_59.addWidget(self.sign_5)
        self.sign_12 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_12.setMinimumSize(QtCore.QSize(310, 0))
        self.sign_12.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_12.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_12.setObjectName("sign_12")
        self.layoutWidget_51 = QtWidgets.QWidget(self.sign_12)
        self.layoutWidget_51.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_51.setObjectName("layoutWidget_51")
        self.verticalLayout_55 = QtWidgets.QVBoxLayout(self.layoutWidget_51)
        self.verticalLayout_55.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_55.setSpacing(5)
        self.verticalLayout_55.setObjectName("verticalLayout_55")
        self.label_125 = QtWidgets.QLabel(self.layoutWidget_51)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_125.setFont(font)
        self.label_125.setObjectName("label_125")
        self.verticalLayout_55.addWidget(self.label_125)
        self.horizontalLayout_48 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_48.setSpacing(10)
        self.horizontalLayout_48.setObjectName("horizontalLayout_48")
        self.label_126 = QtWidgets.QLabel(self.layoutWidget_51)
        self.label_126.setMinimumSize(QtCore.QSize(142, 26))
        self.label_126.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_126.setFont(font)
        self.label_126.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_126.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_126.setText("")
        self.label_126.setObjectName("label_126")
        self.horizontalLayout_48.addWidget(self.label_126)
        self.label_127 = QtWidgets.QLabel(self.layoutWidget_51)
        self.label_127.setMinimumSize(QtCore.QSize(142, 26))
        self.label_127.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_127.setFont(font)
        self.label_127.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_127.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_127.setText("")
        self.label_127.setObjectName("label_127")
        self.horizontalLayout_48.addWidget(self.label_127)
        self.verticalLayout_55.addLayout(self.horizontalLayout_48)
        self.verticalLayout_59.addWidget(self.sign_12)
        self.sign_14 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_14.setMinimumSize(QtCore.QSize(310, 0))
        self.sign_14.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_14.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_14.setObjectName("sign_14")
        self.layoutWidget_52 = QtWidgets.QWidget(self.sign_14)
        self.layoutWidget_52.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_52.setObjectName("layoutWidget_52")
        self.verticalLayout_57 = QtWidgets.QVBoxLayout(self.layoutWidget_52)
        self.verticalLayout_57.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_57.setSpacing(5)
        self.verticalLayout_57.setObjectName("verticalLayout_57")
        self.label_131 = QtWidgets.QLabel(self.layoutWidget_52)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_131.setFont(font)
        self.label_131.setObjectName("label_131")
        self.verticalLayout_57.addWidget(self.label_131)
        self.horizontalLayout_50 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_50.setSpacing(10)
        self.horizontalLayout_50.setObjectName("horizontalLayout_50")
        self.label_132 = QtWidgets.QLabel(self.layoutWidget_52)
        self.label_132.setMinimumSize(QtCore.QSize(142, 26))
        self.label_132.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_132.setFont(font)
        self.label_132.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_132.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_132.setText("")
        self.label_132.setObjectName("label_132")
        self.horizontalLayout_50.addWidget(self.label_132)
        self.label_133 = QtWidgets.QLabel(self.layoutWidget_52)
        self.label_133.setMinimumSize(QtCore.QSize(142, 26))
        self.label_133.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_133.setFont(font)
        self.label_133.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_133.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_133.setText("")
        self.label_133.setObjectName("label_133")
        self.horizontalLayout_50.addWidget(self.label_133)
        self.verticalLayout_57.addLayout(self.horizontalLayout_50)
        self.verticalLayout_59.addWidget(self.sign_14)
        self.sign_10 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_10.setMinimumSize(QtCore.QSize(310, 0))
        self.sign_10.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_10.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_10.setObjectName("sign_10")
        self.layoutWidget_59 = QtWidgets.QWidget(self.sign_10)
        self.layoutWidget_59.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_59.setObjectName("layoutWidget_59")
        self.verticalLayout_53 = QtWidgets.QVBoxLayout(self.layoutWidget_59)
        self.verticalLayout_53.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_53.setSpacing(5)
        self.verticalLayout_53.setObjectName("verticalLayout_53")
        self.label_119 = QtWidgets.QLabel(self.layoutWidget_59)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_119.setFont(font)
        self.label_119.setObjectName("label_119")
        self.verticalLayout_53.addWidget(self.label_119)
        self.horizontalLayout_45 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_45.setSpacing(10)
        self.horizontalLayout_45.setObjectName("horizontalLayout_45")
        self.label_120 = QtWidgets.QLabel(self.layoutWidget_59)
        self.label_120.setMinimumSize(QtCore.QSize(142, 26))
        self.label_120.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_120.setFont(font)
        self.label_120.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_120.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_120.setText("")
        self.label_120.setObjectName("label_120")
        self.horizontalLayout_45.addWidget(self.label_120)
        self.label_121 = QtWidgets.QLabel(self.layoutWidget_59)
        self.label_121.setMinimumSize(QtCore.QSize(142, 26))
        self.label_121.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_121.setFont(font)
        self.label_121.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_121.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_121.setText("")
        self.label_121.setObjectName("label_121")
        self.horizontalLayout_45.addWidget(self.label_121)
        self.verticalLayout_53.addLayout(self.horizontalLayout_45)
        self.verticalLayout_59.addWidget(self.sign_10)
        self.sign_7 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_7.setMinimumSize(QtCore.QSize(310, 0))
        self.sign_7.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_7.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_7.setObjectName("sign_7")
        self.layoutWidget_60 = QtWidgets.QWidget(self.sign_7)
        self.layoutWidget_60.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_60.setObjectName("layoutWidget_60")
        self.verticalLayout_50 = QtWidgets.QVBoxLayout(self.layoutWidget_60)
        self.verticalLayout_50.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_50.setSpacing(5)
        self.verticalLayout_50.setObjectName("verticalLayout_50")
        self.label_110 = QtWidgets.QLabel(self.layoutWidget_60)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_110.setFont(font)
        self.label_110.setObjectName("label_110")
        self.verticalLayout_50.addWidget(self.label_110)
        self.horizontalLayout_42 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_42.setSpacing(10)
        self.horizontalLayout_42.setObjectName("horizontalLayout_42")
        self.label_111 = QtWidgets.QLabel(self.layoutWidget_60)
        self.label_111.setMinimumSize(QtCore.QSize(142, 26))
        self.label_111.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_111.setFont(font)
        self.label_111.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_111.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_111.setText("")
        self.label_111.setObjectName("label_111")
        self.horizontalLayout_42.addWidget(self.label_111)
        self.label_112 = QtWidgets.QLabel(self.layoutWidget_60)
        self.label_112.setMinimumSize(QtCore.QSize(142, 26))
        self.label_112.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_112.setFont(font)
        self.label_112.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_112.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_112.setText("")
        self.label_112.setObjectName("label_112")
        self.horizontalLayout_42.addWidget(self.label_112)
        self.verticalLayout_50.addLayout(self.horizontalLayout_42)
        self.verticalLayout_59.addWidget(self.sign_7)
        self.sign_11 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_11.setMinimumSize(QtCore.QSize(310, 0))
        self.sign_11.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_11.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_11.setObjectName("sign_11")
        self.layoutWidget_61 = QtWidgets.QWidget(self.sign_11)
        self.layoutWidget_61.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_61.setObjectName("layoutWidget_61")
        self.verticalLayout_54 = QtWidgets.QVBoxLayout(self.layoutWidget_61)
        self.verticalLayout_54.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_54.setSpacing(5)
        self.verticalLayout_54.setObjectName("verticalLayout_54")
        self.label_122 = QtWidgets.QLabel(self.layoutWidget_61)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_122.setFont(font)
        self.label_122.setObjectName("label_122")
        self.verticalLayout_54.addWidget(self.label_122)
        self.horizontalLayout_46 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_46.setSpacing(10)
        self.horizontalLayout_46.setObjectName("horizontalLayout_46")
        self.label_123 = QtWidgets.QLabel(self.layoutWidget_61)
        self.label_123.setMinimumSize(QtCore.QSize(142, 26))
        self.label_123.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_123.setFont(font)
        self.label_123.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_123.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_123.setText("")
        self.label_123.setObjectName("label_123")
        self.horizontalLayout_46.addWidget(self.label_123)
        self.label_124 = QtWidgets.QLabel(self.layoutWidget_61)
        self.label_124.setMinimumSize(QtCore.QSize(142, 26))
        self.label_124.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_124.setFont(font)
        self.label_124.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_124.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_124.setText("")
        self.label_124.setObjectName("label_124")
        self.horizontalLayout_46.addWidget(self.label_124)
        self.verticalLayout_54.addLayout(self.horizontalLayout_46)
        self.verticalLayout_59.addWidget(self.sign_11)
        self.sign_15 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_15.setMinimumSize(QtCore.QSize(310, 0))
        self.sign_15.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_15.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_15.setObjectName("sign_15")
        self.layoutWidget_62 = QtWidgets.QWidget(self.sign_15)
        self.layoutWidget_62.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_62.setObjectName("layoutWidget_62")
        self.verticalLayout_58 = QtWidgets.QVBoxLayout(self.layoutWidget_62)
        self.verticalLayout_58.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_58.setSpacing(5)
        self.verticalLayout_58.setObjectName("verticalLayout_58")
        self.label_134 = QtWidgets.QLabel(self.layoutWidget_62)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_134.setFont(font)
        self.label_134.setObjectName("label_134")
        self.verticalLayout_58.addWidget(self.label_134)
        self.horizontalLayout_51 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_51.setSpacing(10)
        self.horizontalLayout_51.setObjectName("horizontalLayout_51")
        self.label_135 = QtWidgets.QLabel(self.layoutWidget_62)
        self.label_135.setMinimumSize(QtCore.QSize(142, 26))
        self.label_135.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_135.setFont(font)
        self.label_135.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_135.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_135.setText("")
        self.label_135.setObjectName("label_135")
        self.horizontalLayout_51.addWidget(self.label_135)
        self.label_136 = QtWidgets.QLabel(self.layoutWidget_62)
        self.label_136.setMinimumSize(QtCore.QSize(142, 26))
        self.label_136.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_136.setFont(font)
        self.label_136.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_136.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_136.setText("")
        self.label_136.setObjectName("label_136")
        self.horizontalLayout_51.addWidget(self.label_136)
        self.verticalLayout_58.addLayout(self.horizontalLayout_51)
        self.verticalLayout_59.addWidget(self.sign_15)
        self.sign_8 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_8.setMinimumSize(QtCore.QSize(310, 0))
        self.sign_8.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_8.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_8.setObjectName("sign_8")
        self.layoutWidget_63 = QtWidgets.QWidget(self.sign_8)
        self.layoutWidget_63.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_63.setObjectName("layoutWidget_63")
        self.verticalLayout_51 = QtWidgets.QVBoxLayout(self.layoutWidget_63)
        self.verticalLayout_51.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_51.setSpacing(5)
        self.verticalLayout_51.setObjectName("verticalLayout_51")
        self.label_113 = QtWidgets.QLabel(self.layoutWidget_63)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_113.setFont(font)
        self.label_113.setObjectName("label_113")
        self.verticalLayout_51.addWidget(self.label_113)
        self.horizontalLayout_43 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_43.setSpacing(10)
        self.horizontalLayout_43.setObjectName("horizontalLayout_43")
        self.label_114 = QtWidgets.QLabel(self.layoutWidget_63)
        self.label_114.setMinimumSize(QtCore.QSize(142, 26))
        self.label_114.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_114.setFont(font)
        self.label_114.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_114.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_114.setText("")
        self.label_114.setObjectName("label_114")
        self.horizontalLayout_43.addWidget(self.label_114)
        self.label_115 = QtWidgets.QLabel(self.layoutWidget_63)
        self.label_115.setMinimumSize(QtCore.QSize(142, 26))
        self.label_115.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_115.setFont(font)
        self.label_115.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_115.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_115.setText("")
        self.label_115.setObjectName("label_115")
        self.horizontalLayout_43.addWidget(self.label_115)
        self.verticalLayout_51.addLayout(self.horizontalLayout_43)
        self.verticalLayout_59.addWidget(self.sign_8)
        self.sign_9 = QtWidgets.QWidget(self.layoutWidget_39)
        self.sign_9.setMinimumSize(QtCore.QSize(310, 0))
        self.sign_9.setMaximumSize(QtCore.QSize(322, 80))
        self.sign_9.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.sign_9.setObjectName("sign_9")
        self.layoutWidget_64 = QtWidgets.QWidget(self.sign_9)
        self.layoutWidget_64.setGeometry(QtCore.QRect(0, 0, 321, 81))
        self.layoutWidget_64.setObjectName("layoutWidget_64")
        self.verticalLayout_52 = QtWidgets.QVBoxLayout(self.layoutWidget_64)
        self.verticalLayout_52.setContentsMargins(15, 9, 15, 15)
        self.verticalLayout_52.setSpacing(5)
        self.verticalLayout_52.setObjectName("verticalLayout_52")
        self.label_116 = QtWidgets.QLabel(self.layoutWidget_64)
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_116.setFont(font)
        self.label_116.setObjectName("label_116")
        self.verticalLayout_52.addWidget(self.label_116)
        self.horizontalLayout_44 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_44.setSpacing(10)
        self.horizontalLayout_44.setObjectName("horizontalLayout_44")
        self.label_117 = QtWidgets.QLabel(self.layoutWidget_64)
        self.label_117.setMinimumSize(QtCore.QSize(142, 26))
        self.label_117.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_117.setFont(font)
        self.label_117.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_117.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_117.setText("")
        self.label_117.setObjectName("label_117")
        self.horizontalLayout_44.addWidget(self.label_117)
        self.label_118 = QtWidgets.QLabel(self.layoutWidget_64)
        self.label_118.setMinimumSize(QtCore.QSize(142, 26))
        self.label_118.setMaximumSize(QtCore.QSize(142, 26))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_118.setFont(font)
        self.label_118.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 8px;")
        self.label_118.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_118.setText("")
        self.label_118.setObjectName("label_118")
        self.horizontalLayout_44.addWidget(self.label_118)
        self.verticalLayout_52.addLayout(self.horizontalLayout_44)
        self.verticalLayout_59.addWidget(self.sign_9)
        self.layoutWidget_66 = QtWidgets.QWidget(self.frame_30)
        self.layoutWidget_66.setGeometry(QtCore.QRect(-10, -10, 361, 61))
        self.layoutWidget_66.setObjectName("layoutWidget_66")
        self.horizontalLayout_54 = QtWidgets.QHBoxLayout(self.layoutWidget_66)
        self.horizontalLayout_54.setContentsMargins(11, 5, 9, 0)
        self.horizontalLayout_54.setSpacing(0)
        self.horizontalLayout_54.setObjectName("horizontalLayout_54")
        self.logo_9 = QtWidgets.QLabel(self.layoutWidget_66)
        self.logo_9.setMinimumSize(QtCore.QSize(40, 44))
        self.logo_9.setMaximumSize(QtCore.QSize(40, 44))
        self.logo_9.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.logo_9.setText("")
        self.logo_9.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Logo.png")))
        self.logo_9.setScaledContents(True)
        self.logo_9.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.logo_9.setObjectName("logo_9")
        self.horizontalLayout_54.addWidget(self.logo_9)
        self.pushButton_36 = QtWidgets.QPushButton(self.layoutWidget_66)
        self.pushButton_36.setMinimumSize(QtCore.QSize(38, 38))
        self.pushButton_36.setMaximumSize(QtCore.QSize(38, 38))
        self.pushButton_36.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.pushButton_36.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(os.path.join(icon_dir, "icon_exit.png")).scaled(40, 44, QtCore.Qt.IgnoreAspectRatio))
        self.pushButton_36.setIcon(icon1)
        self.pushButton_36.setIconSize(QtCore.QSize(38, 38))
        self.pushButton_36.setObjectName("pushButton_36")
        self.pushButton_36.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_main))
        self.horizontalLayout_54.addWidget(self.pushButton_36, 0, QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.gridLayout_4.addWidget(self.frame_30, 0, 0, 1, 1)
        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_4)
        self.widget_15 = QtWidgets.QWidget(self.page_comparison)
        self.widget_15.setGeometry(QtCore.QRect(410, 410, 360, 390))
        self.widget_15.setMinimumSize(QtCore.QSize(360, 390))
        self.widget_15.setMaximumSize(QtCore.QSize(360, 390))
        self.widget_15.setStyleSheet("background-color: #1E5D21;\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_15.setObjectName("widget_15")
        self.recent_estimates_4 = QtWidgets.QWidget(self.widget_15)
        self.recent_estimates_4.setGeometry(QtCore.QRect(0, 15, 360, 400))
        self.recent_estimates_4.setMinimumSize(QtCore.QSize(360, 400))
        self.recent_estimates_4.setMaximumSize(QtCore.QSize(360, 400))
        self.recent_estimates_4.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;")
        self.recent_estimates_4.setObjectName("recent_estimates_4")
        self.layoutWidget_71 = QtWidgets.QWidget(self.recent_estimates_4)
        self.layoutWidget_71.setGeometry(QtCore.QRect(10, 10, 261, 31))
        self.layoutWidget_71.setObjectName("layoutWidget_71")
        self.verticalLayout_60 = QtWidgets.QVBoxLayout(self.layoutWidget_71)
        self.verticalLayout_60.setContentsMargins(16, 0, 0, 0)
        self.verticalLayout_60.setSpacing(6)
        self.verticalLayout_60.setObjectName("verticalLayout_60")
        self.label_153 = QtWidgets.QLabel(self.layoutWidget_71)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_153.setFont(font)
        self.label_153.setObjectName("label_153")
        self.verticalLayout_60.addWidget(self.label_153, 0, QtCore.Qt.AlignLeft)
        self.scrollArea_6 = QtWidgets.QScrollArea(self.recent_estimates_4)
        self.scrollArea_6.setGeometry(QtCore.QRect(12, 42, 335, 321))
        self.scrollArea_6.setMinimumSize(QtCore.QSize(335, 0))
        self.scrollArea_6.setMaximumSize(QtCore.QSize(335, 16777215))
        self.scrollArea_6.setStyleSheet("QScrollArea {\n"
"background-color: rgba(235, 233, 244, 0);\n"
"border-radius: 6px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"border: none;\n"
"background: rgb(152, 152, 152);\n"
"width: 6px;\n"
"border-radius: 3px;\n"
"margin-top: 10px;\n"
"margin-bottom: 10px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"background: #648B66;\n"
"min-height: 20px;\n"
"border-radius: 3px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"border: none;\n"
"background: transparent;\n"
"height: 0px;\n"
"subcontrol-position: bottom;\n"
"subcontrol-origin: margin;\n"
"border-radius: 3px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"border: none;\n"
"background: transparent;\n"
"height: 0px;\n"
"subcontrol-position: top;\n"
"subcontrol-origin: margin;\n"
"border-radius: 3px;\n"
"}\n"
"")
        self.scrollArea_6.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea_6.setWidgetResizable(True)
        self.scrollArea_6.setObjectName("scrollArea_6")
        self.scrollAreaWidgetContents_5 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_5.setGeometry(QtCore.QRect(0, 0, 320, 321))
        self.scrollAreaWidgetContents_5.setMinimumSize(QtCore.QSize(320, 0))
        self.scrollAreaWidgetContents_5.setMaximumSize(QtCore.QSize(320, 16777215))
        self.scrollAreaWidgetContents_5.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.scrollAreaWidgetContents_5.setObjectName("scrollAreaWidgetContents_5")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_5)
        self.gridLayout_5.setVerticalSpacing(10)
        self.gridLayout_5.setObjectName("gridLayout_5")


        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.verticalLayout_13.setSpacing(12)
        self.gridLayout_5.addLayout(self.verticalLayout_13, 0, 0, 1, 1)

        self.scrollArea_6.setWidget(self.scrollAreaWidgetContents_5)
        self.empty_5 = QtWidgets.QLabel(self.recent_estimates_4)
        self.empty_5.setEnabled(True)
        self.empty_5.setGeometry(QtCore.QRect(10, 40, 341, 321))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.empty_5.setFont(font)
        self.empty_5.setStyleSheet("color: rgb(131, 131, 131);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.empty_5.setAlignment(QtCore.Qt.AlignCenter)
        self.empty_5.setObjectName("empty_5")
        self.label_155 = QtWidgets.QLabel(self.widget_15)
        self.label_155.setGeometry(QtCore.QRect(155, 6, 57, 3))
        self.label_155.setStyleSheet("background-color: rgb(181, 181, 181);\n"
"border-radius: 2px;")
        self.label_155.setText("")
        self.label_155.setObjectName("label_155")
        self.label_167 = QtWidgets.QLabel(self.page_comparison)
        self.label_167.setGeometry(QtCore.QRect(325, -10, 21, 61))
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.label_167.setFont(font)
        self.label_167.setStyleSheet("background-color: rgba(255, 255, 255, 0);\n"
"color: rgb(255, 255, 255)")
        self.label_167.setObjectName("label_167")
        self.label_167.setEnabled(False)
        self.close = QtWidgets.QPushButton(self.page_comparison)
        self.close.setGeometry(QtCore.QRect(0, 0, 361, 801))
        self.close.setStyleSheet("background-color: rgba(83, 83, 83, 200);")
        self.close.setText("")
        self.close.setObjectName("close")
        self.scrollArea_4.raise_()
        self.close.raise_()
        self.close.hide()
        self.close.clicked.connect(self.close_button_clicked)
        self.widget_15.raise_()
        self.label_167.raise_()
        self.label_167.hide()
        self.stackedWidget.addWidget(self.page_comparison)
        self.page_calculator = QtWidgets.QWidget()
        self.page_calculator.setMinimumSize(QtCore.QSize(340, 800))
        self.page_calculator.setMaximumSize(QtCore.QSize(1080, 2400))
        self.page_calculator.setObjectName("page_calculator")
        self.widget_4 = QtWidgets.QWidget(self.page_calculator)
        self.widget_4.setGeometry(QtCore.QRect(19, 540, 322, 190))
        self.widget_4.setMinimumSize(QtCore.QSize(310, 190))
        self.widget_4.setMaximumSize(QtCore.QSize(335, 195))
        self.widget_4.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_4.setObjectName("widget_4")
        self.layoutWidget = QtWidgets.QWidget(self.widget_4)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 1, 321, 191))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_15.setContentsMargins(16, 0, 0, 3)
        self.verticalLayout_15.setSpacing(0)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.frame_10 = QtWidgets.QFrame(self.layoutWidget)
        self.frame_10.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_10.setMaximumSize(QtCore.QSize(292, 67))
        self.frame_10.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.layoutWidget_11 = QtWidgets.QWidget(self.frame_10)
        self.layoutWidget_11.setGeometry(QtCore.QRect(0, 0, 281, 61))
        self.layoutWidget_11.setObjectName("layoutWidget_11")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.layoutWidget_11)
        self.horizontalLayout_12.setContentsMargins(13, 13, 0, 6)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.label_28 = QtWidgets.QLabel(self.layoutWidget_11)
        self.label_28.setMaximumSize(QtCore.QSize(36, 36))
        self.label_28.setText("")
        self.label_28.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Finish Flag.png")))
        self.label_28.setScaledContents(True)
        self.label_28.setObjectName("label_28")
        self.horizontalLayout_12.addWidget(self.label_28)
        self.verticalLayout_20 = QtWidgets.QVBoxLayout()
        self.verticalLayout_20.setContentsMargins(9, 0, 10, 0)
        self.verticalLayout_20.setObjectName("verticalLayout_20")
        self.label_29 = QtWidgets.QLabel(self.layoutWidget_11)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_29.setFont(font)
        self.label_29.setObjectName("label_29")
        self.verticalLayout_20.addWidget(self.label_29)
        self.label_18 = QtWidgets.QLabel(self.layoutWidget_11)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_18.setFont(font)
        self.label_18.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 5px;\n"
"padding-left: 5px;")
        self.label_18.setText("")
        self.label_18.setObjectName("label_18")
        self.verticalLayout_20.addWidget(self.label_18)
        self.horizontalLayout_12.addLayout(self.verticalLayout_20)
        self.verticalLayout_15.addWidget(self.frame_10)
        self.pushButton_12 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_12.setMinimumSize(QtCore.QSize(290, 67))
        self.pushButton_12.setMaximumSize(QtCore.QSize(292, 67))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_12.setFont(font)
        self.pushButton_12.setStyleSheet("QPushButton {\n"
"background-color: rgb(130, 198, 95);\n"
"border: none;\n"
"border-radius: 17px;\n"
"color: rgb(255, 255, 255)\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgb(109, 166, 80);\n"
"}")
        self.pushButton_12.setObjectName("pushButton_12")
        self.verticalLayout_15.addWidget(self.pushButton_12)
        self.widget_2 = QtWidgets.QWidget(self.page_calculator)
        self.widget_2.setGeometry(QtCore.QRect(19, 145, 322, 382))
        self.widget_2.setMinimumSize(QtCore.QSize(310, 382))
        self.widget_2.setMaximumSize(QtCore.QSize(335, 352))
        self.widget_2.setStyleSheet("background-color: rgb(137, 211, 103);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_2.setObjectName("widget_2")
        self.widget_3 = QtWidgets.QWidget(self.widget_2)
        self.widget_3.setGeometry(QtCore.QRect(0, 30, 324, 352))
        self.widget_3.setMinimumSize(QtCore.QSize(310, 352))
        self.widget_3.setMaximumSize(QtCore.QSize(335, 352))
        self.widget_3.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_3.setObjectName("widget_3")
        self.layoutWidget_6 = QtWidgets.QWidget(self.widget_3)
        self.layoutWidget_6.setGeometry(QtCore.QRect(0, 0, 321, 351))
        self.layoutWidget_6.setObjectName("layoutWidget_6")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.layoutWidget_6)
        self.verticalLayout_14.setContentsMargins(16, 7, 0, 4)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.frame_6 = QtWidgets.QFrame(self.layoutWidget_6)
        self.frame_6.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_6.setMaximumSize(QtCore.QSize(292, 67))
        self.frame_6.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.layoutWidget_10 = QtWidgets.QWidget(self.frame_6)
        self.layoutWidget_10.setGeometry(QtCore.QRect(0, 0, 281, 59))
        self.layoutWidget_10.setObjectName("layoutWidget_10")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.layoutWidget_10)
        self.horizontalLayout_11.setContentsMargins(13, 13, 0, 6)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_26 = QtWidgets.QLabel(self.layoutWidget_10)
        self.label_26.setMaximumSize(QtCore.QSize(36, 36))
        self.label_26.setText("")
        self.label_26.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Ruler.png")))
        self.label_26.setScaledContents(True)
        self.label_26.setObjectName("label_26")
        self.horizontalLayout_11.addWidget(self.label_26)
        self.verticalLayout_19 = QtWidgets.QVBoxLayout()
        self.verticalLayout_19.setContentsMargins(9, 0, 10, 0)
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.label_27 = QtWidgets.QLabel(self.layoutWidget_10)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_27.setFont(font)
        self.label_27.setObjectName("label_27")
        self.verticalLayout_19.addWidget(self.label_27)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.layoutWidget_10)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_5.setFont(font)
        self.lineEdit_5.setWhatsThis("")
        self.lineEdit_5.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 5px;\n"
"padding-left: 5px;")
        self.lineEdit_5.setInputMethodHints(QtCore.Qt.ImhNone)
        self.lineEdit_5.setInputMask("")
        self.lineEdit_5.setText("")
        self.lineEdit_5.setMaxLength(6)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.verticalLayout_19.addWidget(self.lineEdit_5)
        self.lineEdit_5.setValidator(validator)
        self.horizontalLayout_11.addLayout(self.verticalLayout_19)
        self.verticalLayout_14.addWidget(self.frame_6)
        self.frame_7 = QtWidgets.QFrame(self.layoutWidget_6)
        self.frame_7.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_7.setMaximumSize(QtCore.QSize(292, 67))
        self.frame_7.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.layoutWidget_7 = QtWidgets.QWidget(self.frame_7)
        self.layoutWidget_7.setGeometry(QtCore.QRect(0, 0, 281, 59))
        self.layoutWidget_7.setObjectName("layoutWidget_7")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.layoutWidget_7)
        self.horizontalLayout_8.setContentsMargins(13, 13, 0, 6)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_20 = QtWidgets.QLabel(self.layoutWidget_7)
        self.label_20.setMaximumSize(QtCore.QSize(36, 36))
        self.label_20.setText("")
        self.label_20.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Cost.png")))
        self.label_20.setScaledContents(True)
        self.label_20.setObjectName("label_20")
        self.horizontalLayout_8.addWidget(self.label_20)
        self.verticalLayout_16 = QtWidgets.QVBoxLayout()
        self.verticalLayout_16.setContentsMargins(9, 0, 10, 0)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.label_21 = QtWidgets.QLabel(self.layoutWidget_7)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.verticalLayout_16.addWidget(self.label_21)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.layoutWidget_7)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 5px;\n"
"padding-left: 5px;")
        self.lineEdit_2.setMaxLength(11)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.verticalLayout_16.addWidget(self.lineEdit_2)
        self.lineEdit_2.setValidator(validator)

        self.lineEdit_2.textChanged.connect(lambda: self.add_commas(self.lineEdit_2))

        self.horizontalLayout_8.addLayout(self.verticalLayout_16)
        self.verticalLayout_14.addWidget(self.frame_7)
        self.frame_9 = QtWidgets.QFrame(self.layoutWidget_6)
        self.frame_9.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_9.setMaximumSize(QtCore.QSize(292, 67))
        self.frame_9.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.layoutWidget_12 = QtWidgets.QWidget(self.frame_9)
        self.layoutWidget_12.setGeometry(QtCore.QRect(0, 0, 281, 59))
        self.layoutWidget_12.setObjectName("layoutWidget_12")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.layoutWidget_12)
        self.horizontalLayout_13.setContentsMargins(13, 13, 0, 6)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label_30 = QtWidgets.QLabel(self.layoutWidget_12)
        self.label_30.setMaximumSize(QtCore.QSize(36, 36))
        self.label_30.setText("")
        self.label_30.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Ratio.png")))
        self.label_30.setScaledContents(True)
        self.label_30.setObjectName("label_30")
        self.horizontalLayout_13.addWidget(self.label_30)
        self.frame_12 = QtWidgets.QFrame(self.layoutWidget_12)
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.horizontalSlider_3 = QtWidgets.QSlider(self.frame_12)
        self.horizontalSlider_3.setGeometry(QtCore.QRect(50, 20, 161, 20))
        self.horizontalSlider_3.setStyleSheet("QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    background: #3d82db;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    height: 15px;\n"
"    width: 15px;\n"
"    border-radius: 7px;\n"
"    margin: -5px 0;\n"
"    background: rgb(61, 130, 219);\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #c4c4c4;\n"
"}")
        self.horizontalSlider_3.setMinimum(1)
        self.horizontalSlider_3.setMaximum(20)
        self.horizontalSlider_3.setSingleStep(1)
        self.horizontalSlider_3.setPageStep(1)
        self.horizontalSlider_3.setProperty("value", 4)
        self.horizontalSlider_3.setSliderPosition(4)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.label_31 = QtWidgets.QLabel(self.frame_12)
        self.label_31.setGeometry(QtCore.QRect(10, 0, 206, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_31.setFont(font)
        self.label_31.setObjectName("label_31")
        self.label_22 = QtWidgets.QLabel(self.frame_12)
        self.label_22.setGeometry(QtCore.QRect(10, 22, 31, 16))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_22.sizePolicy().hasHeightForWidth())
        self.label_22.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_22.setFont(font)
        self.label_22.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;")
        self.label_22.setText("")
        self.label_22.setTextFormat(QtCore.Qt.PlainText)
        self.label_22.setScaledContents(False)
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setWordWrap(False)
        self.label_22.setObjectName("label_22")
        self.horizontalSlider_3.valueChanged.connect(lambda value: self.label_22.setText(str(value/10)))
        self.horizontalLayout_13.addWidget(self.frame_12)
        self.verticalLayout_14.addWidget(self.frame_9)
        self.frame_8 = QtWidgets.QFrame(self.layoutWidget_6)
        self.frame_8.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_8.setMaximumSize(QtCore.QSize(292, 67))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.frame_8.setFont(font)
        self.frame_8.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.layoutWidget_8 = QtWidgets.QWidget(self.frame_8)
        self.layoutWidget_8.setGeometry(QtCore.QRect(0, 0, 281, 59))
        self.layoutWidget_8.setObjectName("layoutWidget_8")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.layoutWidget_8)
        self.horizontalLayout_9.setContentsMargins(13, 13, 0, 6)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_23 = QtWidgets.QLabel(self.layoutWidget_8)
        self.label_23.setMaximumSize(QtCore.QSize(36, 36))
        self.label_23.setText("")
        self.label_23.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Discount.png")))
        self.label_23.setScaledContents(True)
        self.label_23.setObjectName("label_23")
        self.horizontalLayout_9.addWidget(self.label_23)
        self.verticalLayout_17 = QtWidgets.QVBoxLayout()
        self.verticalLayout_17.setContentsMargins(9, 0, 10, 0)
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.label_24 = QtWidgets.QLabel(self.layoutWidget_8)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.verticalLayout_17.addWidget(self.label_24)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.layoutWidget_8)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 5px;\n"
"padding-left: 5px;")
        self.lineEdit_3.setMaxLength(5)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.verticalLayout_17.addWidget(self.lineEdit_3)
        self.lineEdit_3.setValidator(validator)
        self.horizontalLayout_9.addLayout(self.verticalLayout_17)
        self.verticalLayout_14.addWidget(self.frame_8)
        self.text_main_2 = QtWidgets.QLabel(self.page_calculator)
        self.text_main_2.setGeometry(QtCore.QRect(19, 88, 191, 33))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.text_main_2.setFont(font)
        self.text_main_2.setObjectName("text_main_2")
        self.layoutWidget_67 = QtWidgets.QWidget(self.page_calculator)
        self.layoutWidget_67.setGeometry(QtCore.QRect(0, 0, 361, 61))
        self.layoutWidget_67.setObjectName("layoutWidget_67")
        self.horizontalLayout_55 = QtWidgets.QHBoxLayout(self.layoutWidget_67)
        self.horizontalLayout_55.setContentsMargins(10, 3, 10, 0)
        self.horizontalLayout_55.setSpacing(0)
        self.horizontalLayout_55.setObjectName("horizontalLayout_55")
        self.logo_10 = QtWidgets.QLabel(self.layoutWidget_67)
        self.logo_10.setMinimumSize(QtCore.QSize(40, 44))
        self.logo_10.setMaximumSize(QtCore.QSize(40, 44))
        self.logo_10.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.logo_10.setText("")
        self.logo_10.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Logo.png")))
        self.logo_10.setScaledContents(True)
        self.logo_10.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.logo_10.setObjectName("logo_10")
        self.horizontalLayout_55.addWidget(self.logo_10)
        self.pushButton_37 = QtWidgets.QPushButton(self.layoutWidget_67)
        self.pushButton_37.setMinimumSize(QtCore.QSize(38, 38))
        self.pushButton_37.setMaximumSize(QtCore.QSize(38, 38))
        self.pushButton_37.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.pushButton_37.setText("")
        self.pushButton_37.setIcon(icon1)
        self.pushButton_37.setIconSize(QtCore.QSize(38, 38))
        self.pushButton_37.setObjectName("pushButton_37")
        self.pushButton_37.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_main))
        self.horizontalLayout_55.addWidget(self.pushButton_37, 0, QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.stackedWidget.addWidget(self.page_calculator)
        self.pahe_help = QtWidgets.QWidget()
        self.pahe_help.setObjectName("pahe_help")
        self.widget_5 = QtWidgets.QWidget(self.pahe_help)
        self.widget_5.setGeometry(QtCore.QRect(19, 145, 322, 540))
        self.widget_5.setMinimumSize(QtCore.QSize(310, 540))
        self.widget_5.setMaximumSize(QtCore.QSize(335, 510))
        self.widget_5.setStyleSheet("background-color: rgb(38, 90, 113);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_5.setObjectName("widget_5")
        self.recent_estimates_2 = QtWidgets.QWidget(self.widget_5)
        self.recent_estimates_2.setGeometry(QtCore.QRect(0, 30, 322, 510))
        self.recent_estimates_2.setMinimumSize(QtCore.QSize(290, 510))
        self.recent_estimates_2.setMaximumSize(QtCore.QSize(335, 510))
        self.recent_estimates_2.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;")
        self.recent_estimates_2.setObjectName("recent_estimates_2")
        self.layoutWidget_13 = QtWidgets.QWidget(self.recent_estimates_2)
        self.layoutWidget_13.setGeometry(QtCore.QRect(0, 0, 321, 91))
        self.layoutWidget_13.setObjectName("layoutWidget_13")
        self.verticalLayout_21 = QtWidgets.QVBoxLayout(self.layoutWidget_13)
        self.verticalLayout_21.setContentsMargins(16, 16, 0, 0)
        self.verticalLayout_21.setSpacing(6)
        self.verticalLayout_21.setObjectName("verticalLayout_21")
        self.label_19 = QtWidgets.QLabel(self.layoutWidget_13)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.verticalLayout_21.addWidget(self.label_19, 0, QtCore.Qt.AlignLeft)
        self.label_25 = QtWidgets.QLabel(self.layoutWidget_13)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_25.setFont(font)
        self.label_25.setObjectName("label_25")
        self.verticalLayout_21.addWidget(self.label_25, 0, QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.layoutWidget_4 = QtWidgets.QWidget(self.recent_estimates_2)
        self.layoutWidget_4.setGeometry(QtCore.QRect(1, 100, 321, 401))
        self.layoutWidget_4.setObjectName("layoutWidget_4")
        self.verticalLayout_25 = QtWidgets.QVBoxLayout(self.layoutWidget_4)
        self.verticalLayout_25.setContentsMargins(15, 0, 0, 0)
        self.verticalLayout_25.setObjectName("verticalLayout_25")
        self.calculation_3 = QtWidgets.QFrame(self.layoutWidget_4)
        self.calculation_3.setMinimumSize(QtCore.QSize(0, 67))
        self.calculation_3.setMaximumSize(QtCore.QSize(290, 67))
        self.calculation_3.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.calculation_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.calculation_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.calculation_3.setObjectName("calculation_3")
        self.pushButton_15 = QtWidgets.QPushButton(self.calculation_3)
        self.pushButton_15.setGeometry(QtCore.QRect(0, 0, 290, 67))
        self.pushButton_15.setMinimumSize(QtCore.QSize(290, 67))
        self.pushButton_15.setMaximumSize(QtCore.QSize(290, 67))
        self.pushButton_15.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border-radius: 17px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton_15.setText("")
        self.pushButton_15.setObjectName("pushButton_15")
        self.pushButton_15.clicked.connect(self.clear_data)

        self.layoutWidget_18 = QtWidgets.QWidget(self.calculation_3)
        self.layoutWidget_18.setGeometry(QtCore.QRect(11, 0, 261, 71))
        self.layoutWidget_18.setObjectName("layoutWidget_18")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.layoutWidget_18)
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_41 = QtWidgets.QLabel(self.layoutWidget_18)
        self.label_41.setMaximumSize(QtCore.QSize(36, 36))
        self.label_41.setStyleSheet("")
        self.label_41.setText("")
        self.label_41.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Eraser.png")))
        self.label_41.setScaledContents(True)
        self.label_41.setObjectName("label_41")
        self.horizontalLayout_14.addWidget(self.label_41)
        self.verticalLayout_27 = QtWidgets.QVBoxLayout()
        self.verticalLayout_27.setContentsMargins(5, 12, -1, 15)
        self.verticalLayout_27.setSpacing(0)
        self.verticalLayout_27.setObjectName("verticalLayout_27")
        self.label_42 = QtWidgets.QLabel(self.layoutWidget_18)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_42.setFont(font)
        self.label_42.setStyleSheet("")
        self.label_42.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_42.setObjectName("label_42")
        self.verticalLayout_27.addWidget(self.label_42)
        self.label_43 = QtWidgets.QLabel(self.layoutWidget_18)
        self.label_43.setStyleSheet("")
        self.label_43.setObjectName("label_43")
        self.verticalLayout_27.addWidget(self.label_43)
        self.horizontalLayout_14.addLayout(self.verticalLayout_27)
        self.layoutWidget_18.raise_()
        self.pushButton_15.raise_()
        self.verticalLayout_25.addWidget(self.calculation_3)
        self.calculation_2 = QtWidgets.QFrame(self.layoutWidget_4)
        self.calculation_2.setMinimumSize(QtCore.QSize(0, 67))
        self.calculation_2.setMaximumSize(QtCore.QSize(290, 67))
        self.calculation_2.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.calculation_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.calculation_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.calculation_2.setObjectName("calculation_2")
        self.pushButton_14 = QtWidgets.QPushButton(self.calculation_2)
        self.pushButton_14.setGeometry(QtCore.QRect(0, 0, 290, 67))
        self.pushButton_14.setMinimumSize(QtCore.QSize(290, 67))
        self.pushButton_14.setMaximumSize(QtCore.QSize(290, 67))
        self.pushButton_14.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border-radius: 17px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton_14.setText("")
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_14.clicked.connect(self.open_gmail)
        self.layoutWidget_17 = QtWidgets.QWidget(self.calculation_2)
        self.layoutWidget_17.setGeometry(QtCore.QRect(11, 0, 261, 71))
        self.layoutWidget_17.setObjectName("layoutWidget_17")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.layoutWidget_17)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_38 = QtWidgets.QLabel(self.layoutWidget_17)
        self.label_38.setMaximumSize(QtCore.QSize(36, 36))
        self.label_38.setStyleSheet("")
        self.label_38.setText("")
        self.label_38.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Connected People.png")))
        self.label_38.setScaledContents(True)
        self.label_38.setObjectName("label_38")
        self.horizontalLayout_10.addWidget(self.label_38)
        self.verticalLayout_26 = QtWidgets.QVBoxLayout()
        self.verticalLayout_26.setContentsMargins(5, 12, -1, 15)
        self.verticalLayout_26.setSpacing(0)
        self.verticalLayout_26.setObjectName("verticalLayout_26")
        self.label_39 = QtWidgets.QLabel(self.layoutWidget_17)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_39.setFont(font)
        self.label_39.setStyleSheet("")
        self.label_39.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_39.setObjectName("label_39")
        self.verticalLayout_26.addWidget(self.label_39)
        self.label_40 = QtWidgets.QLabel(self.layoutWidget_17)
        self.label_40.setStyleSheet("")
        self.label_40.setObjectName("label_40")
        self.verticalLayout_26.addWidget(self.label_40)
        self.horizontalLayout_10.addLayout(self.verticalLayout_26)
        self.layoutWidget_17.raise_()
        self.pushButton_14.raise_()
        self.verticalLayout_25.addWidget(self.calculation_2)
        self.calculation_4 = QtWidgets.QFrame(self.layoutWidget_4)
        self.calculation_4.setMinimumSize(QtCore.QSize(0, 67))
        self.calculation_4.setMaximumSize(QtCore.QSize(290, 67))
        self.calculation_4.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.calculation_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.calculation_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.calculation_4.setObjectName("calculation_4")
        self.pushButton_16 = QtWidgets.QPushButton(self.calculation_4)
        self.pushButton_16.setGeometry(QtCore.QRect(0, 0, 290, 67))
        self.pushButton_16.setMinimumSize(QtCore.QSize(290, 67))
        self.pushButton_16.setMaximumSize(QtCore.QSize(290, 67))
        self.pushButton_16.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border-radius: 17px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton_16.setText("")
        self.pushButton_16.setObjectName("pushButton_16")
        self.pushButton_16.clicked.connect(self.open_website)
        self.layoutWidget_19 = QtWidgets.QWidget(self.calculation_4)
        self.layoutWidget_19.setGeometry(QtCore.QRect(11, 0, 261, 71))
        self.layoutWidget_19.setObjectName("layoutWidget_19")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.layoutWidget_19)
        self.horizontalLayout_15.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.label_44 = QtWidgets.QLabel(self.layoutWidget_19)
        self.label_44.setMaximumSize(QtCore.QSize(36, 36))
        self.label_44.setStyleSheet("")
        self.label_44.setText("")
        self.label_44.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Caretaker.png")))
        self.label_44.setScaledContents(True)
        self.label_44.setObjectName("label_44")
        self.horizontalLayout_15.addWidget(self.label_44)
        self.verticalLayout_28 = QtWidgets.QVBoxLayout()
        self.verticalLayout_28.setContentsMargins(5, 12, -1, 15)
        self.verticalLayout_28.setSpacing(0)
        self.verticalLayout_28.setObjectName("verticalLayout_28")
        self.label_45 = QtWidgets.QLabel(self.layoutWidget_19)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_45.setFont(font)
        self.label_45.setStyleSheet("")
        self.label_45.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_45.setObjectName("label_45")
        self.verticalLayout_28.addWidget(self.label_45)
        self.label_46 = QtWidgets.QLabel(self.layoutWidget_19)
        self.label_46.setStyleSheet("")
        self.label_46.setObjectName("label_46")
        self.verticalLayout_28.addWidget(self.label_46)
        self.horizontalLayout_15.addLayout(self.verticalLayout_28)
        self.layoutWidget_19.raise_()
        self.pushButton_16.raise_()
        self.verticalLayout_25.addWidget(self.calculation_4)
        self.calculation_5 = QtWidgets.QFrame(self.layoutWidget_4)
        self.calculation_5.setMinimumSize(QtCore.QSize(0, 67))
        self.calculation_5.setMaximumSize(QtCore.QSize(290, 67))
        self.calculation_5.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.calculation_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.calculation_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.calculation_5.setObjectName("calculation_5")
        self.pushButton_17 = QtWidgets.QPushButton(self.calculation_5)
        self.pushButton_17.setGeometry(QtCore.QRect(0, 0, 290, 67))
        self.pushButton_17.setMinimumSize(QtCore.QSize(290, 67))
        self.pushButton_17.setMaximumSize(QtCore.QSize(290, 67))
        self.pushButton_17.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border-radius: 17px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton_17.setText("")
        self.pushButton_17.setObjectName("pushButton_17")
        self.layoutWidget_20 = QtWidgets.QWidget(self.calculation_5)
        self.layoutWidget_20.setGeometry(QtCore.QRect(11, 0, 261, 65))
        self.layoutWidget_20.setObjectName("layoutWidget_20")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.layoutWidget_20)
        self.horizontalLayout_16.setContentsMargins(0, 5, 0, 0)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.label_47 = QtWidgets.QLabel(self.layoutWidget_20)
        self.label_47.setMaximumSize(QtCore.QSize(36, 36))
        self.label_47.setStyleSheet("")
        self.label_47.setText("")
        self.label_47.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Black and White.png")))
        self.label_47.setScaledContents(True)
        self.label_47.setObjectName("label_47")
        self.horizontalLayout_16.addWidget(self.label_47)
        self.verticalLayout_29 = QtWidgets.QVBoxLayout()
        self.verticalLayout_29.setContentsMargins(5, 12, -1, 15)
        self.verticalLayout_29.setSpacing(0)
        self.verticalLayout_29.setObjectName("verticalLayout_29")
        self.label_48 = QtWidgets.QLabel(self.layoutWidget_20)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_48.setFont(font)
        self.label_48.setStyleSheet("")
        self.label_48.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_48.setObjectName("label_48")
        self.verticalLayout_29.addWidget(self.label_48)
        self.label_49 = QtWidgets.QLabel(self.layoutWidget_20)
        self.label_49.setStyleSheet("")
        self.label_49.setObjectName("label_49")
        self.verticalLayout_29.addWidget(self.label_49)
        self.horizontalLayout_16.addLayout(self.verticalLayout_29)
        self.layoutWidget_20.raise_()
        self.pushButton_17.raise_()
        self.verticalLayout_25.addWidget(self.calculation_5)
        self.calculation_6 = QtWidgets.QFrame(self.layoutWidget_4)
        self.calculation_6.setMinimumSize(QtCore.QSize(0, 67))
        self.calculation_6.setMaximumSize(QtCore.QSize(290, 67))
        self.calculation_6.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.calculation_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.calculation_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.calculation_6.setObjectName("calculation_6")
        self.pushButton_18 = QtWidgets.QPushButton(self.calculation_6)
        self.pushButton_18.setGeometry(QtCore.QRect(0, 0, 290, 67))
        self.pushButton_18.setMinimumSize(QtCore.QSize(290, 67))
        self.pushButton_18.setMaximumSize(QtCore.QSize(290, 67))
        self.pushButton_18.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border-radius: 17px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton_18.setText("")
        self.pushButton_18.setObjectName("pushButton_18")
        self.pushButton_18.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.pahe_about))
        self.layoutWidget_21 = QtWidgets.QWidget(self.calculation_6)
        self.layoutWidget_21.setGeometry(QtCore.QRect(11, 0, 261, 65))
        self.layoutWidget_21.setObjectName("layoutWidget_21")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.layoutWidget_21)
        self.horizontalLayout_17.setContentsMargins(0, 5, 0, 0)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.label_50 = QtWidgets.QLabel(self.layoutWidget_21)
        self.label_50.setMaximumSize(QtCore.QSize(36, 36))
        self.label_50.setStyleSheet("")
        self.label_50.setText("")
        self.label_50.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Info Squared.png")))
        self.label_50.setScaledContents(True)
        self.label_50.setObjectName("label_50")
        self.horizontalLayout_17.addWidget(self.label_50)
        self.verticalLayout_30 = QtWidgets.QVBoxLayout()
        self.verticalLayout_30.setContentsMargins(5, 12, -1, 15)
        self.verticalLayout_30.setSpacing(0)
        self.verticalLayout_30.setObjectName("verticalLayout_30")
        self.label_51 = QtWidgets.QLabel(self.layoutWidget_21)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_51.setFont(font)
        self.label_51.setStyleSheet("")
        self.label_51.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_51.setObjectName("label_51")
        self.verticalLayout_30.addWidget(self.label_51)
        self.label_52 = QtWidgets.QLabel(self.layoutWidget_21)
        self.label_52.setStyleSheet("")
        self.label_52.setObjectName("label_52")
        self.verticalLayout_30.addWidget(self.label_52)
        self.horizontalLayout_17.addLayout(self.verticalLayout_30)
        self.layoutWidget_21.raise_()
        self.pushButton_18.raise_()
        self.verticalLayout_25.addWidget(self.calculation_6)
        self.text_main_3 = QtWidgets.QLabel(self.pahe_help)
        self.text_main_3.setGeometry(QtCore.QRect(19, 88, 319, 33))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.text_main_3.setFont(font)
        self.text_main_3.setObjectName("text_main_3")
        self.layoutWidget_68 = QtWidgets.QWidget(self.pahe_help)
        self.layoutWidget_68.setGeometry(QtCore.QRect(0, 0, 361, 61))
        self.layoutWidget_68.setObjectName("layoutWidget_68")
        self.horizontalLayout_56 = QtWidgets.QHBoxLayout(self.layoutWidget_68)
        self.horizontalLayout_56.setContentsMargins(10, 3, 10, 0)
        self.horizontalLayout_56.setSpacing(0)
        self.horizontalLayout_56.setObjectName("horizontalLayout_56")
        self.logo_11 = QtWidgets.QLabel(self.layoutWidget_68)
        self.logo_11.setMinimumSize(QtCore.QSize(40, 44))
        self.logo_11.setMaximumSize(QtCore.QSize(40, 44))
        self.logo_11.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.logo_11.setText("")
        self.logo_11.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Logo.png")))
        self.logo_11.setScaledContents(True)
        self.logo_11.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.logo_11.setObjectName("logo_11")
        self.horizontalLayout_56.addWidget(self.logo_11)
        self.pushButton_38 = QtWidgets.QPushButton(self.layoutWidget_68)
        self.pushButton_38.setMinimumSize(QtCore.QSize(38, 38))
        self.pushButton_38.setMaximumSize(QtCore.QSize(38, 38))
        self.pushButton_38.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.pushButton_38.setText("")
        self.pushButton_38.setIcon(icon1)
        self.pushButton_38.setIconSize(QtCore.QSize(38, 38))
        self.pushButton_38.setObjectName("pushButton_38")
        self.pushButton_38.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_main))
        self.horizontalLayout_56.addWidget(self.pushButton_38, 0, QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.stackedWidget.addWidget(self.pahe_help)
        self.pahe_about = QtWidgets.QWidget()
        self.pahe_about.setObjectName("pahe_about")
        self.text_main_4 = QtWidgets.QLabel(self.pahe_about)
        self.text_main_4.setGeometry(QtCore.QRect(19, 88, 218, 33))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.text_main_4.setFont(font)
        self.text_main_4.setObjectName("text_main_4")
        self.widget_7 = QtWidgets.QWidget(self.pahe_about)
        self.widget_7.setGeometry(QtCore.QRect(19, 145, 322, 85))
        self.widget_7.setMinimumSize(QtCore.QSize(310, 85))
        self.widget_7.setMaximumSize(QtCore.QSize(335, 85))
        self.widget_7.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;")
        self.widget_7.setObjectName("widget_7")
        self.layoutWidget_2 = QtWidgets.QWidget(self.widget_7)
        self.layoutWidget_2.setGeometry(QtCore.QRect(16, 12, 283, 63))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.verticalLayout_22 = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_22.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_22.setObjectName("verticalLayout_22")
        self.label_34 = QtWidgets.QLabel(self.layoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_34.setFont(font)
        self.label_34.setObjectName("label_34")
        self.verticalLayout_22.addWidget(self.label_34)
        self.label_35 = QtWidgets.QLabel(self.layoutWidget_2)
        self.label_35.setObjectName("label_35")
        self.verticalLayout_22.addWidget(self.label_35)
        self.widget_6 = QtWidgets.QWidget(self.pahe_about)
        self.widget_6.setGeometry(QtCore.QRect(19, 245, 322, 520))
        self.widget_6.setMinimumSize(QtCore.QSize(310, 520))
        self.widget_6.setMaximumSize(QtCore.QSize(335, 520))
        self.widget_6.setStyleSheet("background-color: rgb(74, 125, 226);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_6.setObjectName("widget_6")
        self.recent_estimates_3 = QtWidgets.QWidget(self.widget_6)
        self.recent_estimates_3.setGeometry(QtCore.QRect(0, 30, 322, 490))
        self.recent_estimates_3.setMinimumSize(QtCore.QSize(290, 400))
        self.recent_estimates_3.setMaximumSize(QtCore.QSize(335, 510))
        self.recent_estimates_3.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;")
        self.recent_estimates_3.setObjectName("recent_estimates_3")
        self.layoutWidget_14 = QtWidgets.QWidget(self.recent_estimates_3)
        self.layoutWidget_14.setGeometry(QtCore.QRect(10, 0, 271, 71))
        self.layoutWidget_14.setObjectName("layoutWidget_14")
        self.verticalLayout_23 = QtWidgets.QVBoxLayout(self.layoutWidget_14)
        self.verticalLayout_23.setContentsMargins(16, 16, 0, 0)
        self.verticalLayout_23.setSpacing(6)
        self.verticalLayout_23.setObjectName("verticalLayout_23")
        self.label_32 = QtWidgets.QLabel(self.layoutWidget_14)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_32.setFont(font)
        self.label_32.setObjectName("label_32")
        self.verticalLayout_23.addWidget(self.label_32, 0, QtCore.Qt.AlignLeft)
        self.label_33 = QtWidgets.QLabel(self.layoutWidget_14)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_33.setFont(font)
        self.label_33.setObjectName("label_33")
        self.verticalLayout_23.addWidget(self.label_33, 0, QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.calculation_7 = QtWidgets.QFrame(self.recent_estimates_3)
        self.calculation_7.setGeometry(QtCore.QRect(15, 75, 291, 401))
        self.calculation_7.setMinimumSize(QtCore.QSize(260, 67))
        self.calculation_7.setMaximumSize(QtCore.QSize(310, 6700000))
        self.calculation_7.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 12px;")
        self.calculation_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.calculation_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.calculation_7.setObjectName("calculation_7")
        self.scrollArea_2 = QtWidgets.QScrollArea(self.calculation_7)
        self.scrollArea_2.setGeometry(QtCore.QRect(10, 8, 271, 381))
        self.scrollArea_2.setMaximumSize(QtCore.QSize(271, 381))
        self.scrollArea_2.setStyleSheet("QScrollArea {\n"
"background-color: rgb(235, 233, 244);\n"
"border-radius: 6px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"border: none;\n"
"background: transparent;\n"
"width: 6px;\n"
"border-radius: 3px;\n"
"margin-top: 10px;\n"
"margin-bottom: 10px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"background: rgb(185, 197, 226);\n"
"min-height: 20px;\n"
"border-radius: 3px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"border: none;\n"
"background: transparent;\n"
"height: 0px;\n"
"subcontrol-position: bottom;\n"
"subcontrol-origin: margin;\n"
"border-radius: 3px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"border: none;\n"
"background: transparent;\n"
"height: 0px;\n"
"subcontrol-position: top;\n"
"subcontrol-origin: margin;\n"
"border-radius: 3px;\n"
"}\n"
"")
        self.scrollArea_2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 268, 629))
        self.scrollAreaWidgetContents_2.setMaximumSize(QtCore.QSize(270, 16777215))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.gridLayout = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout.setObjectName("gridLayout")
        self.label_55 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_55.setMaximumSize(QtCore.QSize(250, 16777215))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(235, 233, 244))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(235, 233, 244))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(235, 233, 244))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(235, 233, 244))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(235, 233, 244))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(235, 233, 244))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(235, 233, 244))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(235, 233, 244))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(235, 233, 244))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.label_55.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_55.setFont(font)
        self.label_55.setStyleSheet("background-color: rgb(235, 233, 244);")
        self.label_55.setObjectName("label_55")
        self.gridLayout.addWidget(self.label_55, 0, 0, 1, 1)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.layoutWidget_69 = QtWidgets.QWidget(self.pahe_about)
        self.layoutWidget_69.setGeometry(QtCore.QRect(0, 0, 361, 61))
        self.layoutWidget_69.setObjectName("layoutWidget_69")
        self.horizontalLayout_57 = QtWidgets.QHBoxLayout(self.layoutWidget_69)
        self.horizontalLayout_57.setContentsMargins(10, 3, 10, 0)
        self.horizontalLayout_57.setSpacing(0)
        self.horizontalLayout_57.setObjectName("horizontalLayout_57")
        self.logo_12 = QtWidgets.QLabel(self.layoutWidget_69)
        self.logo_12.setMinimumSize(QtCore.QSize(40, 44))
        self.logo_12.setMaximumSize(QtCore.QSize(40, 44))
        self.logo_12.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.logo_12.setText("")
        self.logo_12.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Logo.png")))
        self.logo_12.setScaledContents(True)
        self.logo_12.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.logo_12.setObjectName("logo_12")
        self.horizontalLayout_57.addWidget(self.logo_12)
        self.pushButton_39 = QtWidgets.QPushButton(self.layoutWidget_69)
        self.pushButton_39.setMinimumSize(QtCore.QSize(38, 38))
        self.pushButton_39.setMaximumSize(QtCore.QSize(38, 38))
        self.pushButton_39.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.pushButton_39.setText("")
        self.pushButton_39.setIcon(icon1)
        self.pushButton_39.setIconSize(QtCore.QSize(38, 38))
        self.pushButton_39.setObjectName("pushButton_39")
        self.pushButton_39.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.pahe_help))
        self.horizontalLayout_57.addWidget(self.pushButton_39, 0, QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.stackedWidget.addWidget(self.pahe_about)
        self.pahe_signs = QtWidgets.QWidget()
        self.pahe_signs.setObjectName("pahe_signs")
        self.scrollArea = QtWidgets.QScrollArea(self.pahe_signs)
        self.scrollArea.setGeometry(QtCore.QRect(0, 8, 361, 800))
        self.scrollArea.setMaximumSize(QtCore.QSize(16777215, 800))
        self.scrollArea.setFrameShadow(QtWidgets.QFrame.Raised)
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 359, 1200))
        self.scrollAreaWidgetContents.setMinimumSize(QtCore.QSize(0, 1200))
        self.scrollAreaWidgetContents.setMaximumSize(QtCore.QSize(360, 1200))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_2.setContentsMargins(8, 0, 0, 0)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame_11 = QtWidgets.QFrame(self.scrollAreaWidgetContents)
        self.frame_11.setMaximumSize(QtCore.QSize(16777215, 1215))
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.widget_8 = QtWidgets.QWidget(self.frame_11)
        self.widget_8.setGeometry(QtCore.QRect(10, 135, 322, 955))
        self.widget_8.setMinimumSize(QtCore.QSize(310, 382))
        self.widget_8.setMaximumSize(QtCore.QSize(335, 1056))
        self.widget_8.setStyleSheet("background-color: rgb(222, 146, 58);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_8.setObjectName("widget_8")
        self.widget_9 = QtWidgets.QWidget(self.widget_8)
        self.widget_9.setGeometry(QtCore.QRect(0, 30, 322, 925))
        self.widget_9.setMinimumSize(QtCore.QSize(310, 352))
        self.widget_9.setMaximumSize(QtCore.QSize(335, 1056))
        self.widget_9.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_9.setObjectName("widget_9")
        self.layoutWidget_15 = QtWidgets.QWidget(self.widget_9)
        self.layoutWidget_15.setGeometry(QtCore.QRect(0, 0, 321, 921))
        self.layoutWidget_15.setObjectName("layoutWidget_15")
        self.verticalLayout_24 = QtWidgets.QVBoxLayout(self.layoutWidget_15)
        self.verticalLayout_24.setContentsMargins(16, 10, 0, 4)
        self.verticalLayout_24.setObjectName("verticalLayout_24")
        self.label_73 = QtWidgets.QLabel(self.layoutWidget_15)
        self.label_73.setMaximumSize(QtCore.QSize(16777215, 24))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label_73.setFont(font)
        self.label_73.setObjectName("label_73")
        self.verticalLayout_24.addWidget(self.label_73)
        self.frame_13 = QtWidgets.QFrame(self.layoutWidget_15)
        self.frame_13.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_13.setMaximumSize(QtCore.QSize(290, 67))
        self.frame_13.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.layoutWidget_16 = QtWidgets.QWidget(self.frame_13)
        self.layoutWidget_16.setGeometry(QtCore.QRect(10, 0, 271, 71))
        self.layoutWidget_16.setObjectName("layoutWidget_16")
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout(self.layoutWidget_16)
        self.horizontalLayout_20.setContentsMargins(4, 5, 0, 6)
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.label_36 = QtWidgets.QLabel(self.layoutWidget_16)
        self.label_36.setMaximumSize(QtCore.QSize(36, 36))
        self.label_36.setText("")
        self.label_36.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "City.png")))
        self.label_36.setScaledContents(True)
        self.label_36.setObjectName("label_36")
        self.horizontalLayout_20.addWidget(self.label_36)
        self.verticalLayout_31 = QtWidgets.QVBoxLayout()
        self.verticalLayout_31.setContentsMargins(7, 0, 10, 9)
        self.verticalLayout_31.setSpacing(3)
        self.verticalLayout_31.setObjectName("verticalLayout_31")
        self.label_37 = QtWidgets.QLabel(self.layoutWidget_16)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_37.setFont(font)
        self.label_37.setObjectName("label_37")
        self.verticalLayout_31.addWidget(self.label_37)
        self.comboBox = QtWidgets.QComboBox(self.layoutWidget_16)
        self.comboBox.setMaximumSize(QtCore.QSize(16777215, 24))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.comboBox.setFont(font)
        self.comboBox.setStyleSheet("QComboBox QAbstractItemView {\n"
"  color: Transparency;\n"
"    border: 2px solid orange;\n"
"     background-color: rgb(243, 227, 214);\n"
"    selection-background-color: rgb(222, 146, 58);\n"
"    alternate-background-color: lightgray;\n"
"    border-radius: 0px;\n"
"}\n"
"\n"
"QComboBox:on {\n"
"    border: 2px solid orange;\n"
"     padding-left: 5px;\n"
"    border-bottom-left-radius: 0px;\n"
"    border-bottom-right-radius: 0px;\n"
"}\n"
"\n"
"QComboBox {\n"
"  background-color: rgb(248, 247, 252);\n"
"  color: rgb(68, 68, 68);\n"
"    border-radius: 10px;\n"
"  color: rgb(39, 39, 39);\n"
"    padding-left: 7px;\n"
"    height: 16px;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: right;\n"
"    border-radius: 0px;\n"
"    width: 20px;\n"
"   height: 16px;\n"
"}\n"
"\n"
"QComboBox::drop-down:on {\n"
"    padding-right: -2px;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
f"   image: url(\"{icon_path}\");\n"
"    width: 10px;\n"
"    height: 10px;\n"
"}\n"
"\n"
"QComboBox::down-arrow:on {\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QScrollArea {\n"
"background-color: rgb(248, 247, 252);\n"
"border-radius: 0px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"border: none;\n"
"background-color: rgb(248, 247, 252);\n"
"width: 6px;\n"
"border-radius: 0px;\n"
"margin-top: 0;\n"
"margin-bottom: 0;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"min-height: 20px;\n"
"border-radius: 0px;\n"
"background: rgb(222, 163, 96);\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"border: none;\n"
"background: rgb(248, 247, 252);\n"
"height: 0px;\n"
"subcontrol-position: bottom;\n"
"subcontrol-origin: margin;\n"
"border-radius: 0px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"border: none;\n"
"background: transparent;\n"
"height: 0px;\n"
"subcontrol-position: top;\n"
"subcontrol-origin: margin;\n"
"border-radius: 0px;\n"
"}")
        self.comboBox.setMaxVisibleItems(8)
        self.comboBox.setIconSize(QtCore.QSize(10, 10))
        self.comboBox.setObjectName("comboBox")


        # Заполнение комбобокса
        location_keys = list(sorted(location_json.keys()))
        for key in location_keys:
                self.comboBox.addItem(key)
        self.comboBox.currentIndexChanged[str].connect(self.comboBox1Changed)
        

        # Установка цвета строчкам дял удобного чтения
        for i in range(self.comboBox.count()):
            if i % 2 == 0:
                color = QColor(240,207,178)
            else:
                color = QColor(243, 227, 214)
            self.comboBox.setItemData(i, color, role=Qt.BackgroundRole)

        self.verticalLayout_31.addWidget(self.comboBox)
        self.horizontalLayout_20.addLayout(self.verticalLayout_31)
        self.verticalLayout_24.addWidget(self.frame_13)
        self.frame_14 = QtWidgets.QFrame(self.layoutWidget_15)
        self.frame_14.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_14.setMaximumSize(QtCore.QSize(290, 67))
        self.frame_14.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.layoutWidget_22 = QtWidgets.QWidget(self.frame_14)
        self.layoutWidget_22.setGeometry(QtCore.QRect(10, 0, 271, 71))
        self.layoutWidget_22.setObjectName("layoutWidget_22")
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout(self.layoutWidget_22)
        self.horizontalLayout_21.setContentsMargins(4, 5, 0, 6)
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.label_53 = QtWidgets.QLabel(self.layoutWidget_22)
        self.label_53.setMaximumSize(QtCore.QSize(36, 36))
        self.label_53.setText("")
        self.label_53.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Streets.png")))
        self.label_53.setScaledContents(True)
        self.label_53.setObjectName("label_53")
        self.horizontalLayout_21.addWidget(self.label_53)
        self.verticalLayout_32 = QtWidgets.QVBoxLayout()
        self.verticalLayout_32.setContentsMargins(7, 0, 10, 9)
        self.verticalLayout_32.setSpacing(3)
        self.verticalLayout_32.setObjectName("verticalLayout_32")
        self.label_54 = QtWidgets.QLabel(self.layoutWidget_22)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_54.setFont(font)
        self.label_54.setObjectName("label_54")
        self.verticalLayout_32.addWidget(self.label_54)
        self.comboBox_3 = QtWidgets.QComboBox(self.layoutWidget_22)
        self.comboBox_3.setMaximumSize(QtCore.QSize(16777215, 24))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.comboBox_3.setFont(font)
        self.comboBox_3.setStyleSheet("QComboBox QAbstractItemView {\n"
"  color: Transparency;\n"
"    border: 2px solid orange;\n"
"     background-color: rgb(243, 227, 214);\n"
"    selection-background-color: rgb(222, 146, 58);\n"
"    alternate-background-color: lightgray;\n"
"    border-radius: 0px;\n"
"}\n"
"\n"
"QComboBox:on {\n"
"    border: 2px solid orange;\n"
"     padding-left: 5px;\n"
"    border-bottom-left-radius: 0px;\n"
"    border-bottom-right-radius: 0px;\n"
"}\n"
"\n"
"QComboBox {\n"
"  background-color: rgb(248, 247, 252);\n"
"  color: rgb(68, 68, 68);\n"
"    border-radius: 10px;\n"
"  color: rgb(39, 39, 39);\n"
"    padding-left: 7px;\n"
"    height: 16px;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: right;\n"
"    border-radius: 0px;\n"
"    width: 20px;\n"
"   height: 16px;\n"
"}\n"
"\n"
"QComboBox::drop-down:on {\n"
"    padding-right: -2px;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
f"   image: url(\"{icon_path}\");\n"
"    width: 10px;\n"
"    height: 10px;\n"
"}\n"
"\n"
"QComboBox::down-arrow:on {\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QScrollArea {\n"
"background-color: rgb(248, 247, 252);\n"
"border-radius: 0px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"border: none;\n"
"background-color: rgb(248, 247, 252);\n"
"width: 6px;\n"
"border-radius: 0px;\n"
"margin-top: 0;\n"
"margin-bottom: 0;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"min-height: 20px;\n"
"border-radius: 0px;\n"
"background: rgb(222, 163, 96);\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"border: none;\n"
"background: rgb(248, 247, 252);\n"
"height: 0px;\n"
"subcontrol-position: bottom;\n"
"subcontrol-origin: margin;\n"
"border-radius: 0px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"border: none;\n"
"background: transparent;\n"
"height: 0px;\n"
"subcontrol-position: top;\n"
"subcontrol-origin: margin;\n"
"border-radius: 0px;\n"
"}")
        self.comboBox_3.setMaxVisibleItems(8)
        self.comboBox_3.setIconSize(QtCore.QSize(10, 10))
        self.comboBox_3.setObjectName("comboBox_3")

        # Установка цвета строчкам дял удобного чтения
        for i in range(self.comboBox_3.count()):
            if i % 2 == 0:
                color = QColor(240,207,178)
            else:
                color = QColor(243, 227, 214)
            self.comboBox_3.setItemData(i, color, role=Qt.BackgroundRole)

        self.verticalLayout_32.addWidget(self.comboBox_3)
        self.horizontalLayout_21.addLayout(self.verticalLayout_32)
        self.verticalLayout_24.addWidget(self.frame_14)
        self.frame_26 = QtWidgets.QFrame(self.layoutWidget_15)
        self.frame_26.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_26.setMaximumSize(QtCore.QSize(290, 67))
        self.frame_26.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_26.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_26.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_26.setObjectName("frame_26")
        self.layoutWidget_30 = QtWidgets.QWidget(self.frame_26)
        self.layoutWidget_30.setGeometry(QtCore.QRect(10, 0, 271, 71))
        self.layoutWidget_30.setObjectName("layoutWidget_30")
        self.horizontalLayout_29 = QtWidgets.QHBoxLayout(self.layoutWidget_30)
        self.horizontalLayout_29.setContentsMargins(4, 5, 0, 6)
        self.horizontalLayout_29.setObjectName("horizontalLayout_29")
        self.label_75 = QtWidgets.QLabel(self.layoutWidget_30)
        self.label_75.setMaximumSize(QtCore.QSize(36, 36))
        self.label_75.setText("")
        self.label_75.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Farm 2.png")))
        self.label_75.setScaledContents(True)
        self.label_75.setObjectName("label_75")
        self.horizontalLayout_29.addWidget(self.label_75)
        self.verticalLayout_35 = QtWidgets.QVBoxLayout()
        self.verticalLayout_35.setContentsMargins(7, 0, 10, 9)
        self.verticalLayout_35.setSpacing(3)
        self.verticalLayout_35.setObjectName("verticalLayout_35")
        self.label_76 = QtWidgets.QLabel(self.layoutWidget_30)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_76.setFont(font)
        self.label_76.setObjectName("label_76")
        self.verticalLayout_35.addWidget(self.label_76)
        self.comboBox_4 = QtWidgets.QComboBox(self.layoutWidget_30)
        self.comboBox_4.setMaximumSize(QtCore.QSize(16777215, 24))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.comboBox_4.setFont(font)
        self.comboBox_4.setStyleSheet("QComboBox QAbstractItemView {\n"
"  color: Transparency;\n"
"    border: 2px solid orange;\n"
"     background-color: rgb(243, 227, 214);\n"
"    selection-background-color: rgb(222, 146, 58);\n"
"    alternate-background-color: lightgray;\n"
"    border-radius: 0px;\n"
"}\n"
"\n"
"QComboBox:on {\n"
"    border: 2px solid orange;\n"
"     padding-left: 5px;\n"
"    border-bottom-left-radius: 0px;\n"
"    border-bottom-right-radius: 0px;\n"
"}\n"
"\n"
"QComboBox {\n"
"  background-color: rgb(248, 247, 252);\n"
"  color: rgb(68, 68, 68);\n"
"    border-radius: 10px;\n"
"  color: rgb(39, 39, 39);\n"
"    padding-left: 7px;\n"
"    height: 16px;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: right;\n"
"    border-radius: 0px;\n"
"    width: 20px;\n"
"   height: 16px;\n"
"}\n"
"\n"
"QComboBox::drop-down:on {\n"
"    padding-right: -2px;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
f"   image: url(\"{icon_path}\");\n"
"    width: 10px;\n"
"    height: 10px;\n"
"}\n"
"\n"
"QComboBox::down-arrow:on {\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QScrollArea {\n"
"background-color: rgb(248, 247, 252);\n"
"border-radius: 0px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"border: none;\n"
"background-color: rgb(248, 247, 252);\n"
"width: 6px;\n"
"border-radius: 0px;\n"
"margin-top: 0;\n"
"margin-bottom: 0;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"min-height: 20px;\n"
"border-radius: 0px;\n"
"background: rgb(222, 163, 96);\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"border: none;\n"
"background: rgb(248, 247, 252);\n"
"height: 0px;\n"
"subcontrol-position: bottom;\n"
"subcontrol-origin: margin;\n"
"border-radius: 0px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"border: none;\n"
"background: transparent;\n"
"height: 0px;\n"
"subcontrol-position: top;\n"
"subcontrol-origin: margin;\n"
"border-radius: 0px;\n"
"}")
        self.comboBox_4.setMaxVisibleItems(8)
        self.comboBox_4.setIconSize(QtCore.QSize(10, 10))
        self.comboBox_4.setObjectName("comboBox_4")

        # Установка цвета строчкам дял удобного чтения
        for i in range(self.comboBox_4.count()):
            if i % 2 == 0:
                color = QColor(240,207,178)
            else:
                color = QColor(243, 227, 214)
            self.comboBox_4.setItemData(i, color, role=Qt.BackgroundRole)

        self.verticalLayout_35.addWidget(self.comboBox_4)
        self.comboBox_3.currentIndexChanged[str].connect(self.comboBox2Changed)
        self.comboBox_3.currentIndexChanged[str].connect(self.comboBox3Changed)
        self.horizontalLayout_29.addLayout(self.verticalLayout_35)
        self.verticalLayout_24.addWidget(self.frame_26)
        self.frame_27 = QtWidgets.QFrame(self.layoutWidget_15)
        self.frame_27.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_27.setMaximumSize(QtCore.QSize(290, 67))
        self.frame_27.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_27.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_27.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_27.setObjectName("frame_27")
        self.layoutWidget_31 = QtWidgets.QWidget(self.frame_27)
        self.layoutWidget_31.setGeometry(QtCore.QRect(10, 0, 271, 71))
        self.layoutWidget_31.setObjectName("layoutWidget_31")
        self.horizontalLayout_30 = QtWidgets.QHBoxLayout(self.layoutWidget_31)
        self.horizontalLayout_30.setContentsMargins(4, 5, 0, 6)
        self.horizontalLayout_30.setObjectName("horizontalLayout_30")
        self.label_77 = QtWidgets.QLabel(self.layoutWidget_31)
        self.label_77.setMaximumSize(QtCore.QSize(36, 36))
        self.label_77.setText("")
        self.label_77.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Property.png")))
        self.label_77.setScaledContents(True)
        self.label_77.setObjectName("label_77")
        self.horizontalLayout_30.addWidget(self.label_77)
        self.verticalLayout_36 = QtWidgets.QVBoxLayout()
        self.verticalLayout_36.setContentsMargins(7, 0, 10, 9)
        self.verticalLayout_36.setSpacing(3)
        self.verticalLayout_36.setObjectName("verticalLayout_36")
        self.label_78 = QtWidgets.QLabel(self.layoutWidget_31)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_78.setFont(font)
        self.label_78.setObjectName("label_78")
        self.verticalLayout_36.addWidget(self.label_78)
        self.comboBox_5 = QtWidgets.QComboBox(self.layoutWidget_31)
        self.comboBox_5.setMaximumSize(QtCore.QSize(16777215, 24))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.comboBox_5.setFont(font)
        self.comboBox_5.setStyleSheet("QComboBox QAbstractItemView {\n"
"  color: Transparency;\n"
"    border: 2px solid orange;\n"
"     background-color: rgb(243, 227, 214);\n"
"    selection-background-color: rgb(222, 146, 58);\n"
"    alternate-background-color: lightgray;\n"
"    border-radius: 0px;\n"
"}\n"
"\n"
"QComboBox:on {\n"
"    border: 2px solid orange;\n"
"     padding-left: 5px;\n"
"    border-bottom-left-radius: 0px;\n"
"    border-bottom-right-radius: 0px;\n"
"}\n"
"\n"
"QComboBox {\n"
"  background-color: rgb(248, 247, 252);\n"
"  color: rgb(68, 68, 68);\n"
"    border-radius: 10px;\n"
"  color: rgb(39, 39, 39);\n"
"    padding-left: 7px;\n"
"    height: 16px;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: right;\n"
"    border-radius: 0px;\n"
"    width: 20px;\n"
"   height: 16px;\n"
"}\n"
"\n"
"QComboBox::drop-down:on {\n"
"    padding-right: -2px;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
f"   image: url(\"{icon_path}\");\n"
"    width: 10px;\n"
"    height: 10px;\n"
"}\n"
"\n"
"QComboBox::down-arrow:on {\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QScrollArea {\n"
"background-color: rgb(248, 247, 252);\n"
"border-radius: 0px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"border: none;\n"
"background-color: rgb(248, 247, 252);\n"
"width: 6px;\n"
"border-radius: 0px;\n"
"margin-top: 0;\n"
"margin-bottom: 0;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"min-height: 20px;\n"
"border-radius: 0px;\n"
"background: rgb(222, 163, 96);\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"border: none;\n"
"background: rgb(248, 247, 252);\n"
"height: 0px;\n"
"subcontrol-position: bottom;\n"
"subcontrol-origin: margin;\n"
"border-radius: 0px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"border: none;\n"
"background: transparent;\n"
"height: 0px;\n"
"subcontrol-position: top;\n"
"subcontrol-origin: margin;\n"
"border-radius: 0px;\n"
"}")
        self.comboBox_5.setMaxVisibleItems(8)
        self.comboBox_5.setIconSize(QtCore.QSize(10, 10))
        self.comboBox_5.setObjectName("comboBox_5")

        # Установка цвета строчкам дял удобного чтения
        for i in range(self.comboBox_5.count()):
            if i % 2 == 0:
                color = QColor(240,207,178)
            else:
                color = QColor(243, 227, 214)
            self.comboBox_5.setItemData(i, color, role=Qt.BackgroundRole)

        self.verticalLayout_36.addWidget(self.comboBox_5)
        self.horizontalLayout_30.addLayout(self.verticalLayout_36)
        self.verticalLayout_24.addWidget(self.frame_27)
        self.frame_17 = QtWidgets.QFrame(self.layoutWidget_15)
        self.frame_17.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_17.setMaximumSize(QtCore.QSize(290, 67))
        self.frame_17.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_17.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_17.setObjectName("frame_17")
        self.layoutWidget_25 = QtWidgets.QWidget(self.frame_17)
        self.layoutWidget_25.setGeometry(QtCore.QRect(0, 0, 281, 59))
        self.layoutWidget_25.setObjectName("layoutWidget_25")
        self.horizontalLayout_24 = QtWidgets.QHBoxLayout(self.layoutWidget_25)
        self.horizontalLayout_24.setContentsMargins(13, 13, 0, 6)
        self.horizontalLayout_24.setObjectName("horizontalLayout_24")
        self.label_61 = QtWidgets.QLabel(self.layoutWidget_25)
        self.label_61.setMaximumSize(QtCore.QSize(36, 36))
        self.label_61.setText("")
        self.label_61.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Stationery.png")))
        self.label_61.setScaledContents(True)
        self.label_61.setObjectName("label_61")
        self.horizontalLayout_24.addWidget(self.label_61)
        self.frame_18 = QtWidgets.QFrame(self.layoutWidget_25)
        self.frame_18.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_18.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_18.setObjectName("frame_18")
        self.horizontalSlider_5 = QtWidgets.QSlider(self.frame_18)
        self.horizontalSlider_5.setGeometry(QtCore.QRect(50, 20, 161, 20))
        self.horizontalSlider_5.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.horizontalSlider_5.setStyleSheet("QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    background: rgb(222, 146, 58);\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    height: 15px;\n"
"    width: 15px;\n"
"    border-radius: 7px;\n"
"    margin: -5px 0;\n"
"    background:#de923a;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #c4c4c4;\n"
"}")
        self.horizontalSlider_5.setMinimum(1)
        self.horizontalSlider_5.setMaximum(50)
        self.horizontalSlider_5.setSingleStep(1)
        self.horizontalSlider_5.setPageStep(1)
        self.horizontalSlider_5.setProperty("value", 8)
        self.horizontalSlider_5.setSliderPosition(8)
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_5.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_5.setObjectName("horizontalSlider_5")
        self.label_62 = QtWidgets.QLabel(self.frame_18)
        self.label_62.setGeometry(QtCore.QRect(10, 0, 206, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_62.setFont(font)
        self.label_62.setObjectName("label_62")
        self.label_63 = QtWidgets.QLabel(self.frame_18)
        self.label_63.setGeometry(QtCore.QRect(10, 22, 31, 16))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_63.sizePolicy().hasHeightForWidth())
        self.label_63.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_63.setFont(font)
        self.label_63.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;")
        self.label_63.setText("")
        self.label_63.setTextFormat(QtCore.Qt.PlainText)
        self.label_63.setScaledContents(False)
        self.label_63.setAlignment(QtCore.Qt.AlignCenter)
        self.label_63.setWordWrap(False)
        self.label_63.setObjectName("label_63")
        self.horizontalSlider_5.valueChanged.connect(lambda value: self.label_63.setText(str(value)))
        self.horizontalLayout_24.addWidget(self.frame_18)
        self.verticalLayout_24.addWidget(self.frame_17)
        self.frame_21 = QtWidgets.QFrame(self.layoutWidget_15)
        self.frame_21.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_21.setMaximumSize(QtCore.QSize(290, 67))
        self.frame_21.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_21.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_21.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_21.setObjectName("frame_21")
        self.layoutWidget_27 = QtWidgets.QWidget(self.frame_21)
        self.layoutWidget_27.setGeometry(QtCore.QRect(0, 0, 281, 59))
        self.layoutWidget_27.setObjectName("layoutWidget_27")
        self.horizontalLayout_26 = QtWidgets.QHBoxLayout(self.layoutWidget_27)
        self.horizontalLayout_26.setContentsMargins(13, 13, 0, 6)
        self.horizontalLayout_26.setObjectName("horizontalLayout_26")
        self.label_67 = QtWidgets.QLabel(self.layoutWidget_27)
        self.label_67.setMaximumSize(QtCore.QSize(36, 36))
        self.label_67.setText("")
        self.label_67.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Closed Window.png")))
        self.label_67.setScaledContents(True)
        self.label_67.setObjectName("label_67")
        self.horizontalLayout_26.addWidget(self.label_67)
        self.frame_22 = QtWidgets.QFrame(self.layoutWidget_27)
        self.frame_22.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_22.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_22.setObjectName("frame_22")
        self.horizontalSlider_7 = QtWidgets.QSlider(self.frame_22)
        self.horizontalSlider_7.setGeometry(QtCore.QRect(50, 20, 161, 20))
        self.horizontalSlider_7.setStyleSheet("QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    background: rgb(222, 146, 58);\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    height: 15px;\n"
"    width: 15px;\n"
"    border-radius: 7px;\n"
"    margin: -5px 0;\n"
"    background:#de923a;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #c4c4c4;\n"
"}")
        self.horizontalSlider_7.setMinimum(1)
        self.horizontalSlider_7.setMaximum(50)
        self.horizontalSlider_7.setSingleStep(1)
        self.horizontalSlider_7.setPageStep(1)
        self.horizontalSlider_7.setProperty("value", 4)
        self.horizontalSlider_7.setSliderPosition(4)
        self.horizontalSlider_7.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_7.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_7.setObjectName("horizontalSlider_7")
        self.label_68 = QtWidgets.QLabel(self.frame_22)
        self.label_68.setGeometry(QtCore.QRect(10, 0, 206, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_68.setFont(font)
        self.label_68.setObjectName("label_68")
        self.label_69 = QtWidgets.QLabel(self.frame_22)
        self.label_69.setGeometry(QtCore.QRect(10, 22, 31, 16))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_69.sizePolicy().hasHeightForWidth())
        self.label_69.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_69.setFont(font)
        self.label_69.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;")
        self.label_69.setText("")
        self.label_69.setTextFormat(QtCore.Qt.PlainText)
        self.label_69.setScaledContents(False)
        self.label_69.setAlignment(QtCore.Qt.AlignCenter)
        self.label_69.setWordWrap(False)
        self.label_69.setObjectName("label_69")
        self.horizontalSlider_7.valueChanged.connect(lambda value: self.label_69.setText(str(value)))
        self.horizontalLayout_26.addWidget(self.frame_22)
        self.verticalLayout_24.addWidget(self.frame_21)
        self.frame_23 = QtWidgets.QFrame(self.layoutWidget_15)
        self.frame_23.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_23.setMaximumSize(QtCore.QSize(290, 67))
        self.frame_23.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_23.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_23.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_23.setObjectName("frame_23")
        self.layoutWidget_28 = QtWidgets.QWidget(self.frame_23)
        self.layoutWidget_28.setGeometry(QtCore.QRect(0, 0, 281, 59))
        self.layoutWidget_28.setObjectName("layoutWidget_28")
        self.horizontalLayout_27 = QtWidgets.QHBoxLayout(self.layoutWidget_28)
        self.horizontalLayout_27.setContentsMargins(13, 13, 0, 6)
        self.horizontalLayout_27.setObjectName("horizontalLayout_27")
        self.label_70 = QtWidgets.QLabel(self.layoutWidget_28)
        self.label_70.setMaximumSize(QtCore.QSize(36, 36))
        self.label_70.setText("")
        self.label_70.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Room.png")))
        self.label_70.setScaledContents(True)
        self.label_70.setObjectName("label_70")
        self.horizontalLayout_27.addWidget(self.label_70)
        self.frame_24 = QtWidgets.QFrame(self.layoutWidget_28)
        self.frame_24.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_24.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_24.setObjectName("frame_24")
        self.horizontalSlider_8 = QtWidgets.QSlider(self.frame_24)
        self.horizontalSlider_8.setGeometry(QtCore.QRect(50, 20, 161, 20))
        self.horizontalSlider_8.setStyleSheet("QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    background: rgb(222, 146, 58);\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    height: 15px;\n"
"    width: 15px;\n"
"    border-radius: 7px;\n"
"    margin: -5px 0;\n"
"    background:#de923a;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #c4c4c4;\n"
"}")
        self.horizontalSlider_8.setMinimum(20)
        self.horizontalSlider_8.setMaximum(120)
        self.horizontalSlider_8.setSingleStep(1)
        self.horizontalSlider_8.setPageStep(1)
        self.horizontalSlider_8.setProperty("value", 30)
        self.horizontalSlider_8.setSliderPosition(30)
        self.horizontalSlider_8.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_8.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_8.setObjectName("horizontalSlider_8")
        self.label_71 = QtWidgets.QLabel(self.frame_24)
        self.label_71.setGeometry(QtCore.QRect(10, 0, 206, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_71.setFont(font)
        self.label_71.setObjectName("label_71")
        self.label_72 = QtWidgets.QLabel(self.frame_24)
        self.label_72.setGeometry(QtCore.QRect(10, 22, 31, 16))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_72.sizePolicy().hasHeightForWidth())
        self.label_72.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_72.setFont(font)
        self.label_72.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;")
        self.label_72.setText("")
        self.label_72.setTextFormat(QtCore.Qt.PlainText)
        self.label_72.setScaledContents(False)
        self.label_72.setAlignment(QtCore.Qt.AlignCenter)
        self.label_72.setWordWrap(False)
        self.label_72.setObjectName("label_72")
        self.horizontalSlider_8.valueChanged.connect(lambda value: self.label_72.setText(str(value)))
        self.horizontalLayout_27.addWidget(self.frame_24)
        self.verticalLayout_24.addWidget(self.frame_23)
        self.frame_19 = QtWidgets.QFrame(self.layoutWidget_15)
        self.frame_19.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_19.setMaximumSize(QtCore.QSize(290, 67))
        self.frame_19.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_19.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_19.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_19.setObjectName("frame_19")
        self.layoutWidget_26 = QtWidgets.QWidget(self.frame_19)
        self.layoutWidget_26.setGeometry(QtCore.QRect(0, 0, 281, 59))
        self.layoutWidget_26.setObjectName("layoutWidget_26")
        self.horizontalLayout_25 = QtWidgets.QHBoxLayout(self.layoutWidget_26)
        self.horizontalLayout_25.setContentsMargins(13, 13, 0, 6)
        self.horizontalLayout_25.setObjectName("horizontalLayout_25")
        self.label_64 = QtWidgets.QLabel(self.layoutWidget_26)
        self.label_64.setMaximumSize(QtCore.QSize(36, 36))
        self.label_64.setText("")
        self.label_64.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Double Bed.png")))
        self.label_64.setScaledContents(True)
        self.label_64.setObjectName("label_64")
        self.horizontalLayout_25.addWidget(self.label_64)
        self.frame_20 = QtWidgets.QFrame(self.layoutWidget_26)
        self.frame_20.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_20.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_20.setObjectName("frame_20")
        self.horizontalSlider_6 = QtWidgets.QSlider(self.frame_20)
        self.horizontalSlider_6.setGeometry(QtCore.QRect(50, 20, 161, 20))
        self.horizontalSlider_6.setStyleSheet("QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    background: rgb(222, 146, 58);\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    height: 15px;\n"
"    width: 15px;\n"
"    border-radius: 7px;\n"
"    margin: -5px 0;\n"
"    background:#de923a;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #c4c4c4;\n"
"}")
        self.horizontalSlider_6.setMinimum(17)
        self.horizontalSlider_6.setMaximum(80)
        self.horizontalSlider_6.setSingleStep(1)
        self.horizontalSlider_6.setPageStep(1)
        self.horizontalSlider_6.setProperty("value", 20)
        self.horizontalSlider_6.setSliderPosition(20)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_6.setObjectName("horizontalSlider_6")
        self.label_65 = QtWidgets.QLabel(self.frame_20)
        self.label_65.setGeometry(QtCore.QRect(10, 0, 206, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_65.setFont(font)
        self.label_65.setObjectName("label_65")
        self.label_66 = QtWidgets.QLabel(self.frame_20)
        self.label_66.setGeometry(QtCore.QRect(10, 22, 31, 16))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_66.sizePolicy().hasHeightForWidth())
        self.label_66.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_66.setFont(font)
        self.label_66.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;")
        self.label_66.setText("")
        self.label_66.setTextFormat(QtCore.Qt.PlainText)
        self.label_66.setScaledContents(False)
        self.label_66.setAlignment(QtCore.Qt.AlignCenter)
        self.label_66.setWordWrap(False)
        self.label_66.setObjectName("label_66")
        self.horizontalSlider_6.valueChanged.connect(lambda value: self.label_66.setText(str(value)))
        self.horizontalLayout_25.addWidget(self.frame_20)
        self.verticalLayout_24.addWidget(self.frame_19)
        self.frame_15 = QtWidgets.QFrame(self.layoutWidget_15)
        self.frame_15.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_15.setMaximumSize(QtCore.QSize(290, 67))
        self.frame_15.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.layoutWidget_23 = QtWidgets.QWidget(self.frame_15)
        self.layoutWidget_23.setGeometry(QtCore.QRect(0, 0, 281, 59))
        self.layoutWidget_23.setObjectName("layoutWidget_23")
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout(self.layoutWidget_23)
        self.horizontalLayout_22.setContentsMargins(13, 13, 0, 6)
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.label_56 = QtWidgets.QLabel(self.layoutWidget_23)
        self.label_56.setMaximumSize(QtCore.QSize(36, 36))
        self.label_56.setText("")
        self.label_56.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Cooker.png")))
        self.label_56.setScaledContents(True)
        self.label_56.setObjectName("label_56")
        self.horizontalLayout_22.addWidget(self.label_56)
        self.frame_16 = QtWidgets.QFrame(self.layoutWidget_23)
        self.frame_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_16.setObjectName("frame_16")
        self.horizontalSlider_4 = QtWidgets.QSlider(self.frame_16)
        self.horizontalSlider_4.setGeometry(QtCore.QRect(50, 20, 161, 20))
        self.horizontalSlider_4.setStyleSheet("QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    background: rgb(222, 146, 58);\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    height: 15px;\n"
"    width: 15px;\n"
"    border-radius: 7px;\n"
"    margin: -5px 0;\n"
"    background:#de923a;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #c4c4c4;\n"
"}")
        self.horizontalSlider_4.setMinimum(3)
        self.horizontalSlider_4.setMaximum(30)
        self.horizontalSlider_4.setSingleStep(1)
        self.horizontalSlider_4.setPageStep(1)
        self.horizontalSlider_4.setProperty("value", 6)
        self.horizontalSlider_4.setSliderPosition(6)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.label_57 = QtWidgets.QLabel(self.frame_16)
        self.label_57.setGeometry(QtCore.QRect(10, 0, 206, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_57.setFont(font)
        self.label_57.setObjectName("label_57")
        self.label_58 = QtWidgets.QLabel(self.frame_16)
        self.label_58.setGeometry(QtCore.QRect(10, 22, 31, 16))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_58.sizePolicy().hasHeightForWidth())
        self.label_58.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_58.setFont(font)
        self.label_58.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;")
        self.label_58.setText("")
        self.label_58.setTextFormat(QtCore.Qt.PlainText)
        self.label_58.setScaledContents(False)
        self.label_58.setAlignment(QtCore.Qt.AlignCenter)
        self.label_58.setWordWrap(False)
        self.label_58.setObjectName("label_58")
        self.horizontalSlider_4.valueChanged.connect(lambda value: self.label_58.setText(str(value)))
        self.horizontalLayout_22.addWidget(self.frame_16)
        self.verticalLayout_24.addWidget(self.frame_15)
        self.frame_29 = QtWidgets.QFrame(self.layoutWidget_15)
        self.frame_29.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_29.setMaximumSize(QtCore.QSize(290, 67))
        self.frame_29.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_29.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_29.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_29.setObjectName("frame_29")
        self.layoutWidget_32 = QtWidgets.QWidget(self.frame_29)
        self.layoutWidget_32.setGeometry(QtCore.QRect(10, 0, 271, 71))
        self.layoutWidget_32.setObjectName("layoutWidget_32")
        self.horizontalLayout_31 = QtWidgets.QHBoxLayout(self.layoutWidget_32)
        self.horizontalLayout_31.setContentsMargins(4, 5, 0, 6)
        self.horizontalLayout_31.setObjectName("horizontalLayout_31")
        self.label_79 = QtWidgets.QLabel(self.layoutWidget_32)
        self.label_79.setMaximumSize(QtCore.QSize(36, 36))
        self.label_79.setText("")
        self.label_79.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Lease.png")))
        self.label_79.setScaledContents(True)
        self.label_79.setObjectName("label_79")
        self.horizontalLayout_31.addWidget(self.label_79)
        self.verticalLayout_37 = QtWidgets.QVBoxLayout()
        self.verticalLayout_37.setContentsMargins(7, 0, 10, 9)
        self.verticalLayout_37.setSpacing(3)
        self.verticalLayout_37.setObjectName("verticalLayout_37")
        self.label_80 = QtWidgets.QLabel(self.layoutWidget_32)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_80.setFont(font)
        self.label_80.setObjectName("label_80")
        self.verticalLayout_37.addWidget(self.label_80)
        self.comboBox_6 = QtWidgets.QComboBox(self.layoutWidget_32)
        self.comboBox_6.setMaximumSize(QtCore.QSize(16777215, 24))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.comboBox_6.setFont(font)
        self.comboBox_6.setStyleSheet("QComboBox QAbstractItemView {\n"
"  color: Transparency;\n"
"    border: 2px solid orange;\n"
"     background-color: rgb(243, 227, 214);\n"
"    selection-background-color: rgb(222, 146, 58);\n"
"    alternate-background-color: lightgray;\n"
"    border-radius: 0px;\n"
"}\n"
"\n"
"QComboBox:on {\n"
"    border: 2px solid orange;\n"
"     padding-left: 5px;\n"
"    border-bottom-left-radius: 0px;\n"
"    border-bottom-right-radius: 0px;\n"
"}\n"
"\n"
"QComboBox {\n"
"  background-color: rgb(248, 247, 252);\n"
"  color: rgb(68, 68, 68);\n"
"    border-radius: 10px;\n"
"  color: rgb(39, 39, 39);\n"
"    padding-left: 7px;\n"
"    height: 16px;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: right;\n"
"    border-radius: 0px;\n"
"    width: 20px;\n"
"   height: 16px;\n"
"}\n"
"\n"
"QComboBox::drop-down:on {\n"
"    padding-right: -2px;\n"
"}\n"
"\n"
"QComboBox::down-arrow {\n"
f"   image: url(\"{icon_path}\");\n"
"    width: 10px;\n"
"    height: 10px;\n"
"}\n"
"\n"
"QComboBox::down-arrow:on {\n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QScrollArea {\n"
"background-color: rgb(248, 247, 252);\n"
"border-radius: 0px;\n"
"}\n"
"\n"
"QScrollBar:vertical {\n"
"border: none;\n"
"background-color: rgb(248, 247, 252);\n"
"width: 6px;\n"
"border-radius: 0px;\n"
"margin-top: 0;\n"
"margin-bottom: 0;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical {\n"
"min-height: 20px;\n"
"border-radius: 0px;\n"
"background: rgb(222, 163, 96);\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"border: none;\n"
"background: rgb(248, 247, 252);\n"
"height: 0px;\n"
"subcontrol-position: bottom;\n"
"subcontrol-origin: margin;\n"
"border-radius: 0px;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"border: none;\n"
"background: transparent;\n"
"height: 0px;\n"
"subcontrol-position: top;\n"
"subcontrol-origin: margin;\n"
"border-radius: 0px;\n"
"}")
        self.comboBox_6.setMaxVisibleItems(8)
        self.comboBox_6.setIconSize(QtCore.QSize(10, 10))
        self.comboBox_6.setObjectName("comboBox_6")

        for i in range(2024, 1915, -1):
                self.comboBox_6.addItem(str(i))

        # Установка цвета строчкам дял удобного чтения
        for i in range(self.comboBox_6.count()):
            if i % 2 == 0:
                color = QColor(240,207,178)
            else:
                color = QColor(243, 227, 214)
            self.comboBox_6.setItemData(i, color, role=Qt.BackgroundRole)

        self.verticalLayout_37.addWidget(self.comboBox_6)

        self.horizontalLayout_31.addLayout(self.verticalLayout_37)
        self.verticalLayout_24.addWidget(self.frame_29)
        self.frame_25 = QtWidgets.QFrame(self.layoutWidget_15)
        self.frame_25.setMinimumSize(QtCore.QSize(290, 67))
        self.frame_25.setMaximumSize(QtCore.QSize(290, 67))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.frame_25.setFont(font)
        self.frame_25.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 17px;")
        self.frame_25.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_25.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_25.setObjectName("frame_25")
        self.layoutWidget_24 = QtWidgets.QWidget(self.frame_25)
        self.layoutWidget_24.setGeometry(QtCore.QRect(10, -2, 271, 71))
        self.layoutWidget_24.setObjectName("layoutWidget_24")
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout(self.layoutWidget_24)
        self.horizontalLayout_23.setContentsMargins(5, 11, 0, 6)
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.label_59 = QtWidgets.QLabel(self.layoutWidget_24)
        self.label_59.setMaximumSize(QtCore.QSize(36, 36))
        self.label_59.setText("")
        self.label_59.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Studio Floor Plan.png")))
        self.label_59.setScaledContents(True)
        self.label_59.setObjectName("label_59")
        self.horizontalLayout_23.addWidget(self.label_59)
        self.verticalLayout_33 = QtWidgets.QVBoxLayout()
        self.verticalLayout_33.setContentsMargins(7, 0, 10, 7)
        self.verticalLayout_33.setObjectName("verticalLayout_33")
        self.label_60 = QtWidgets.QLabel(self.layoutWidget_24)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_60.setFont(font)
        self.label_60.setObjectName("label_60")
        self.verticalLayout_33.addWidget(self.label_60)
        self.horizontalLayout_28 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_28.setSpacing(8)
        self.horizontalLayout_28.setObjectName("horizontalLayout_28")
        self.pushButton_24 = QtWidgets.QPushButton(self.layoutWidget_24)
        self.pushButton_24.setMinimumSize(QtCore.QSize(22, 22))
        self.pushButton_24.setMaximumSize(QtCore.QSize(22, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_24.setFont(font)
        self.pushButton_24.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;")
        self.pushButton_24.setObjectName("pushButton_24")
        self.horizontalLayout_28.addWidget(self.pushButton_24)
        self.pushButton_24.clicked.connect(self.changeColor)
        self.pushButton_25 = QtWidgets.QPushButton(self.layoutWidget_24)
        self.pushButton_25.setMinimumSize(QtCore.QSize(22, 22))
        self.pushButton_25.setMaximumSize(QtCore.QSize(22, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_25.setFont(font)
        self.pushButton_25.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;")
        self.pushButton_25.setObjectName("pushButton_25")
        self.horizontalLayout_28.addWidget(self.pushButton_25)
        self.pushButton_25.clicked.connect(self.changeColor)
        self.pushButton_23 = QtWidgets.QPushButton(self.layoutWidget_24)
        self.pushButton_23.setMinimumSize(QtCore.QSize(22, 22))
        self.pushButton_23.setMaximumSize(QtCore.QSize(22, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_23.setFont(font)
        self.pushButton_23.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;")
        self.pushButton_23.setObjectName("pushButton_23")
        self.horizontalLayout_28.addWidget(self.pushButton_23)
        self.pushButton_23.clicked.connect(self.changeColor)
        self.pushButton_22 = QtWidgets.QPushButton(self.layoutWidget_24)
        self.pushButton_22.setMinimumSize(QtCore.QSize(22, 22))
        self.pushButton_22.setMaximumSize(QtCore.QSize(22, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_22.setFont(font)
        self.pushButton_22.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;")
        self.pushButton_22.setObjectName("pushButton_22")
        self.horizontalLayout_28.addWidget(self.pushButton_22)
        self.pushButton_22.clicked.connect(self.changeColor)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_28.addItem(spacerItem)
        self.verticalLayout_33.addLayout(self.horizontalLayout_28)
        self.horizontalLayout_23.addLayout(self.verticalLayout_33)
        self.verticalLayout_24.addWidget(self.frame_25)
        self.text_main_5 = QtWidgets.QLabel(self.frame_11)
        self.text_main_5.setGeometry(QtCore.QRect(10, 79, 341, 33))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.text_main_5.setFont(font)
        self.text_main_5.setObjectName("text_main_5")
        self.pushButton_21 = QtWidgets.QPushButton(self.frame_11)
        self.pushButton_21.setGeometry(QtCore.QRect(10, 1105, 322, 67))
        self.pushButton_21.setMinimumSize(QtCore.QSize(290, 67))
        self.pushButton_21.setMaximumSize(QtCore.QSize(322, 67))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_21.setFont(font)
        self.pushButton_21.setStyleSheet("QPushButton {\n"
"background-color: rgb(222, 146, 58);\n"
"border: none;\n"
"border-radius: 17px;\n"
"color: rgb(255, 255, 255)\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgb(197, 129, 51);\n"
"}")
        self.pushButton_21.setObjectName("pushButton_21")
        self.pushButton_21.clicked.connect(self.processing_info)

        self.layoutWidget_70 = QtWidgets.QWidget(self.frame_11)
        self.layoutWidget_70.setGeometry(QtCore.QRect(0, -12, 351, 61))
        self.layoutWidget_70.setObjectName("layoutWidget_70")
        self.horizontalLayout_59 = QtWidgets.QHBoxLayout(self.layoutWidget_70)
        self.horizontalLayout_59.setContentsMargins(1, 9, 9, 0)
        self.horizontalLayout_59.setSpacing(0)
        self.horizontalLayout_59.setObjectName("horizontalLayout_59")
        self.logo_14 = QtWidgets.QLabel(self.layoutWidget_70)
        self.logo_14.setMinimumSize(QtCore.QSize(40, 44))
        self.logo_14.setMaximumSize(QtCore.QSize(40, 44))
        self.logo_14.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.logo_14.setText("")
        self.logo_14.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Logo.png")))
        self.logo_14.setScaledContents(True)
        self.logo_14.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.logo_14.setObjectName("logo_14")
        self.horizontalLayout_59.addWidget(self.logo_14)
        self.pushButton_41 = QtWidgets.QPushButton(self.layoutWidget_70)
        self.pushButton_41.setMinimumSize(QtCore.QSize(38, 38))
        self.pushButton_41.setMaximumSize(QtCore.QSize(38, 38))
        self.pushButton_41.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.pushButton_41.setText("")
        self.pushButton_41.setIcon(icon1)
        self.pushButton_41.setIconSize(QtCore.QSize(38, 38))
        self.pushButton_41.setObjectName("pushButton_41")
        self.pushButton_41.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_main))
        self.horizontalLayout_59.addWidget(self.pushButton_41, 0, QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.gridLayout_2.addWidget(self.frame_11, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.stackedWidget.addWidget(self.pahe_signs)
        self.pahe_history = QtWidgets.QWidget()
        self.pahe_history.setObjectName("pahe_history")
        self.scrollArea_3 = QtWidgets.QScrollArea(self.pahe_history)
        self.scrollArea_3.setGeometry(QtCore.QRect(8, 1, 351, 800))
        self.scrollArea_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.scrollArea_3.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea_3.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollArea_3.setObjectName("scrollArea_3")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 349, 790))
        self.scrollAreaWidgetContents_3.setMinimumSize(QtCore.QSize(349, 790))
        self.scrollAreaWidgetContents_3.setMaximumSize(QtCore.QSize(349, 16777215))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame_28 = QtWidgets.QFrame(self.scrollAreaWidgetContents_3)
        self.frame_28.setMinimumSize(QtCore.QSize(0, 1000))
        self.frame_28.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.frame_28.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_28.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_28.setObjectName("frame_28")
        self.layoutWidget_29 = QtWidgets.QWidget(self.frame_28)
        self.layoutWidget_29.setGeometry(QtCore.QRect(-10, -10, 351, 61))
        self.layoutWidget_29.setObjectName("layoutWidget_29")
        self.horizontalLayout_32 = QtWidgets.QHBoxLayout(self.layoutWidget_29)
        self.horizontalLayout_32.setContentsMargins(2, 1, 8, 0)
        self.horizontalLayout_32.setSpacing(0)
        self.horizontalLayout_32.setObjectName("horizontalLayout_32")
        self.logo_6 = QtWidgets.QLabel(self.layoutWidget_29)
        self.logo_6.setMinimumSize(QtCore.QSize(40, 44))
        self.logo_6.setMaximumSize(QtCore.QSize(40, 44))
        self.logo_6.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.logo_6.setText("")
        self.logo_6.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Logo.png")))
        self.logo_6.setScaledContents(True)
        self.logo_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.logo_6.setObjectName("logo_6")
        self.horizontalLayout_32.addWidget(self.logo_6)
        self.pushButton_26 = QtWidgets.QPushButton(self.layoutWidget_29)
        self.pushButton_26.setMinimumSize(QtCore.QSize(38, 38))
        self.pushButton_26.setMaximumSize(QtCore.QSize(38, 38))
        self.pushButton_26.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.pushButton_26.setText("")
        self.pushButton_26.setIcon(icon1)
        self.pushButton_26.setIconSize(QtCore.QSize(38, 38))
        self.pushButton_26.setObjectName("pushButton_26")
        self.pushButton_26.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_main))
        self.horizontalLayout_32.addWidget(self.pushButton_26, 0, QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.text_main_6 = QtWidgets.QLabel(self.frame_28)
        self.text_main_6.setGeometry(QtCore.QRect(9, 77, 291, 33))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.text_main_6.setFont(font)
        self.text_main_6.setObjectName("text_main_6")
        self.background = QtWidgets.QWidget(self.frame_28)
        self.background.setGeometry(QtCore.QRect(1, 135, 322, 607))
        self.background.setMinimumSize(QtCore.QSize(310, 0))
        self.background.setMaximumSize(QtCore.QSize(335, 16777215))
        self.background.setStyleSheet("background-color: rgb(245, 223, 28);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.background.setObjectName("background")
        self.foreground = QtWidgets.QWidget(self.background)
        self.foreground.setGeometry(QtCore.QRect(0, 30, 322, 577))
        self.foreground.setMinimumSize(QtCore.QSize(290, 0))
        self.foreground.setMaximumSize(QtCore.QSize(335, 16777215))
        self.foreground.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;")
        self.foreground.setObjectName("foreground")
        self.layoutWidget_33 = QtWidgets.QWidget(self.foreground)
        self.layoutWidget_33.setGeometry(QtCore.QRect(10, 0, 282, 72))
        self.layoutWidget_33.setObjectName("layoutWidget_33")
        self.verticalLayout_34 = QtWidgets.QVBoxLayout(self.layoutWidget_33)
        self.verticalLayout_34.setContentsMargins(10, 16, 0, 0)
        self.verticalLayout_34.setSpacing(6)
        self.verticalLayout_34.setObjectName("verticalLayout_34")
        self.label_74 = QtWidgets.QLabel(self.layoutWidget_33)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_74.setFont(font)
        self.label_74.setObjectName("label_74")
        self.verticalLayout_34.addWidget(self.label_74, 0, QtCore.Qt.AlignLeft)
        self.label_81 = QtWidgets.QLabel(self.layoutWidget_33)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_81.setFont(font)
        self.label_81.setObjectName("label_81")
        self.verticalLayout_34.addWidget(self.label_81, 0, QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.layoutWidget_5 = QtWidgets.QWidget(self.foreground)
        self.layoutWidget_5.setGeometry(QtCore.QRect(7, 80, 311, 81))
        self.layoutWidget_5.setObjectName("layoutWidget_5")
        self.verticalLayout_40 = QtWidgets.QVBoxLayout(self.layoutWidget_5)
        self.verticalLayout_40.setContentsMargins(9, 5, 0, 0)
        self.verticalLayout_40.setSpacing(12)
        self.verticalLayout_40.setObjectName("verticalLayout_40")
        self.empty_4 = QtWidgets.QLabel(self.foreground)
        self.empty_4.setGeometry(QtCore.QRect(0, -1, 322, 601))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.empty_4.setFont(font)
        self.empty_4.setStyleSheet("color: rgb(131, 131, 131);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.empty_4.setAlignment(QtCore.Qt.AlignCenter)
        self.empty_4.setObjectName("empty_4")
        self.empty_4.raise_()
        self.layoutWidget_5.raise_()
        self.layoutWidget_33.raise_()
        self.gridLayout_3.addWidget(self.frame_28, 0, 0, 1, 1, QtCore.Qt.AlignTop)
        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)
        self.stackedWidget.addWidget(self.pahe_history)
        self.page_mortgage = QtWidgets.QWidget()
        self.page_mortgage.setMinimumSize(QtCore.QSize(340, 1040))
        self.page_mortgage.setMaximumSize(QtCore.QSize(1080, 1040))
        self.page_mortgage.setObjectName("page_mortgage")
        self.scrollArea_5 = QtWidgets.QScrollArea(self.page_mortgage)
        self.scrollArea_5.setGeometry(QtCore.QRect(0, 9, 361, 800))
        self.scrollArea_5.setMinimumSize(QtCore.QSize(0, 800))
        self.scrollArea_5.setMaximumSize(QtCore.QSize(16777215, 1040))
        self.scrollArea_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.scrollArea_5.setLineWidth(1)
        self.scrollArea_5.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea_5.setWidgetResizable(True)
        self.scrollArea_5.setObjectName("scrollArea_5")
        self.scrollAreaWidgetContents_6 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_6.setGeometry(QtCore.QRect(0, 0, 359, 1040))
        self.scrollAreaWidgetContents_6.setMinimumSize(QtCore.QSize(0, 1040))
        self.scrollAreaWidgetContents_6.setMaximumSize(QtCore.QSize(360, 1040))
        self.scrollAreaWidgetContents_6.setObjectName("scrollAreaWidgetContents_6")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_6)
        self.gridLayout_6.setContentsMargins(8, 0, 0, 0)
        self.gridLayout_6.setSpacing(0)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.frame_32 = QtWidgets.QFrame(self.scrollAreaWidgetContents_6)
        self.frame_32.setMinimumSize(QtCore.QSize(0, 1040))
        self.frame_32.setMaximumSize(QtCore.QSize(16777215, 1040))
        self.frame_32.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_32.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_32.setObjectName("frame_32")
        self.text_main_9 = QtWidgets.QLabel(self.frame_32)
        self.text_main_9.setGeometry(QtCore.QRect(10, 78, 341, 33))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.text_main_9.setFont(font)
        self.text_main_9.setObjectName("text_main_9")
        self.layoutWidget_108 = QtWidgets.QWidget(self.frame_32)
        self.layoutWidget_108.setGeometry(QtCore.QRect(-10, -10, 361, 61))
        self.layoutWidget_108.setObjectName("layoutWidget_108")
        self.horizontalLayout_91 = QtWidgets.QHBoxLayout(self.layoutWidget_108)
        self.horizontalLayout_91.setContentsMargins(11, 3, 9, 0)
        self.horizontalLayout_91.setSpacing(0)
        self.horizontalLayout_91.setObjectName("horizontalLayout_91")
        self.logo_15 = QtWidgets.QLabel(self.layoutWidget_108)
        self.logo_15.setMinimumSize(QtCore.QSize(40, 44))
        self.logo_15.setMaximumSize(QtCore.QSize(40, 44))
        self.logo_15.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.logo_15.setText("")
        self.logo_15.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Logo.png")))
        self.logo_15.setScaledContents(True)
        self.logo_15.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.logo_15.setObjectName("logo_15")
        self.horizontalLayout_91.addWidget(self.logo_15)
        self.pushButton_42 = QtWidgets.QPushButton(self.layoutWidget_108)
        self.pushButton_42.setMinimumSize(QtCore.QSize(38, 38))
        self.pushButton_42.setMaximumSize(QtCore.QSize(38, 38))
        self.pushButton_42.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.pushButton_42.setText("")
        self.pushButton_42.setIcon(icon1)
        self.pushButton_42.setIconSize(QtCore.QSize(38, 38))
        self.pushButton_42.setObjectName("pushButton_42")
        self.pushButton_42.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_main))
        self.horizontalLayout_91.addWidget(self.pushButton_42, 0, QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.widget_19 = QtWidgets.QWidget(self.frame_32)
        self.widget_19.setGeometry(QtCore.QRect(10, 137, 322, 442))
        self.widget_19.setMinimumSize(QtCore.QSize(310, 0))
        self.widget_19.setMaximumSize(QtCore.QSize(322, 500))
        self.widget_19.setStyleSheet("background-color: #2F986C;\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_19.setObjectName("widget_19")
        self.widget_20 = QtWidgets.QWidget(self.widget_19)
        self.widget_20.setGeometry(QtCore.QRect(0, 30, 322, 412))
        self.widget_20.setMinimumSize(QtCore.QSize(310, 0))
        self.widget_20.setMaximumSize(QtCore.QSize(335, 500))
        self.widget_20.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_20.setObjectName("widget_20")
        self.layoutWidget_90 = QtWidgets.QWidget(self.widget_20)
        self.layoutWidget_90.setGeometry(QtCore.QRect(0, 0, 322, 411))
        self.layoutWidget_90.setObjectName("layoutWidget_90")
        self.verticalLayout_80 = QtWidgets.QVBoxLayout(self.layoutWidget_90)
        self.verticalLayout_80.setContentsMargins(16, 9, 14, 15)
        self.verticalLayout_80.setSpacing(13)
        self.verticalLayout_80.setObjectName("verticalLayout_80")
        self.label_187 = QtWidgets.QLabel(self.layoutWidget_90)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_187.setFont(font)
        self.label_187.setObjectName("label_187")
        self.verticalLayout_80.addWidget(self.label_187)
        self.frame_33 = QtWidgets.QFrame(self.layoutWidget_90)
        self.frame_33.setMinimumSize(QtCore.QSize(290, 76))
        self.frame_33.setMaximumSize(QtCore.QSize(290, 76))
        self.frame_33.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 12px;")
        self.frame_33.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_33.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_33.setObjectName("frame_33")
        self.label_188 = QtWidgets.QLabel(self.frame_33)
        self.label_188.setGeometry(QtCore.QRect(15, 21, 36, 36))
        self.label_188.setMaximumSize(QtCore.QSize(36, 36))
        self.label_188.setText("")
        self.label_188.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Ruble.png")))
        self.label_188.setScaledContents(True)
        self.label_188.setObjectName("label_188")
        self.label_189 = QtWidgets.QLabel(self.frame_33)
        self.label_189.setGeometry(QtCore.QRect(64, -15, 209, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_189.setFont(font)
        self.label_189.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.label_189.setObjectName("label_189")
        self.layoutWidget1 = QtWidgets.QWidget(self.frame_33)
        self.layoutWidget1.setGeometry(QtCore.QRect(58, 30, 221, 41))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(5, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_191 = QtWidgets.QLineEdit(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_191.sizePolicy().hasHeightForWidth())
        self.label_191.setSizePolicy(sizePolicy)
        self.label_191.setMinimumSize(QtCore.QSize(207, 31))
        self.label_191.setMaximumSize(QtCore.QSize(207, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_191.setFont(font)
        self.label_191.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;\n"
"padding-left: 5px;\n"
"padding-bottom: 2px;")
        self.label_191.setMaxLength(11)
        self.label_191.setObjectName("label_191")
        self.verticalLayout.addWidget(self.label_191)
        self.label_191.setValidator(validator)
        self.label_191.textChanged.connect(lambda: self.add_commas(self.label_191))
        self.horizontalSlider_9 = QtWidgets.QSlider(self.layoutWidget1)
        self.horizontalSlider_9.setMinimumSize(QtCore.QSize(207, 20))
        self.horizontalSlider_9.setMaximumSize(QtCore.QSize(207, 20))
        self.horizontalSlider_9.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.horizontalSlider_9.setStyleSheet("QSlider {\n"
"    \n"
"    background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    background: #2FB57D;\n"
"    border-top-right-radius: 0px;\n"
"    border-bottom-right-radius: 4px;\n"
"    border-top-left-radius: 0; \n"
"    border-bottom-left-radius: 4; \n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    height: 15px;\n"
"    width: 15px;\n"
"    border-radius: 7px;\n"
"    margin: -5px 0;\n"
"    background:#2FB57D;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #c4c4c4;\n"
"    border-top-right-radius: 0px;\n"
"    border-bottom-right-radius: 4px;\n"
"    border-top-left-radius: 0; \n"
"    border-bottom-left-radius: 4; \n"
"}")
        self.horizontalSlider_9.setMinimum(1)
        self.horizontalSlider_9.setMaximum(100000000)
        self.horizontalSlider_9.setSingleStep(1)
        self.horizontalSlider_9.setPageStep(1)
        self.horizontalSlider_9.setProperty("value", 7000000)
        self.horizontalSlider_9.setSliderPosition(8)
        self.horizontalSlider_9.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_9.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_9.setObjectName("horizontalSlider_9")
        self.verticalLayout.addWidget(self.horizontalSlider_9)
        self.horizontalSlider_9.valueChanged.connect(lambda value: self.label_191.setText("{:,}".format(value)))
        self.verticalLayout_80.addWidget(self.frame_33)
        self.frame_36 = QtWidgets.QFrame(self.layoutWidget_90)
        self.frame_36.setMinimumSize(QtCore.QSize(290, 76))
        self.frame_36.setMaximumSize(QtCore.QSize(290, 76))
        self.frame_36.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 12px;")
        self.frame_36.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_36.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_36.setObjectName("frame_36")
        self.label_194 = QtWidgets.QLabel(self.frame_36)
        self.label_194.setGeometry(QtCore.QRect(15, 21, 36, 36))
        self.label_194.setMaximumSize(QtCore.QSize(36, 36))
        self.label_194.setText("")
        self.label_194.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Donate.png")))
        self.label_194.setScaledContents(True)
        self.label_194.setObjectName("label_194")
        self.label_195 = QtWidgets.QLabel(self.frame_36)
        self.label_195.setGeometry(QtCore.QRect(64, -15, 209, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_195.setFont(font)
        self.label_195.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.label_195.setObjectName("label_195")
        self.layoutWidget_3 = QtWidgets.QWidget(self.frame_36)
        self.layoutWidget_3.setGeometry(QtCore.QRect(58, 30, 221, 41))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget_3)
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_2.setContentsMargins(5, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_196 = QtWidgets.QLineEdit(self.layoutWidget_3)
        self.label_196.setMinimumSize(QtCore.QSize(207, 31))
        self.label_196.setMaximumSize(QtCore.QSize(207, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_196.setFont(font)
        self.label_196.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;\n"
"padding-left: 5px;\n"
"padding-bottom: 2px;")
        self.label_196.setMaxLength(11)
        self.label_196.setObjectName("label_196")
        self.verticalLayout_2.addWidget(self.label_196)
        self.label_196.setValidator(validator)

        self.label_196.textChanged.connect(lambda: self.add_commas(self.label_196))

        self.horizontalSlider_11 = QtWidgets.QSlider(self.layoutWidget_3)
        self.horizontalSlider_11.setMinimumSize(QtCore.QSize(207, 20))
        self.horizontalSlider_11.setMaximumSize(QtCore.QSize(207, 20))
        self.horizontalSlider_11.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.horizontalSlider_11.setStyleSheet("QSlider {\n"
"    \n"
"    background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    background: #2FB57D;\n"
"    border-top-right-radius: 0px;\n"
"    border-bottom-right-radius: 4px;\n"
"    border-top-left-radius: 0; \n"
"    border-bottom-left-radius: 4; \n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    height: 15px;\n"
"    width: 15px;\n"
"    border-radius: 7px;\n"
"    margin: -5px 0;\n"
"    background:#2FB57D;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #c4c4c4;\n"
"    border-top-right-radius: 0px;\n"
"    border-bottom-right-radius: 4px;\n"
"    border-top-left-radius: 0; \n"
"    border-bottom-left-radius: 4; \n"
"}")
        self.horizontalSlider_11.setMinimum(0)
        self.horizontalSlider_11.setMaximum(100000000)
        self.horizontalSlider_11.setSingleStep(1)
        self.horizontalSlider_11.setPageStep(1)
        self.horizontalSlider_11.setProperty("value", 7000000)
        self.horizontalSlider_11.setSliderPosition(8)
        self.horizontalSlider_11.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_11.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_11.setObjectName("horizontalSlider_11")
        self.verticalLayout_2.addWidget(self.horizontalSlider_11)
        self.horizontalSlider_11.valueChanged.connect(lambda value: self.label_196.setText("{:,}".format(value)))
        self.verticalLayout_80.addWidget(self.frame_36)
        self.frame_37 = QtWidgets.QFrame(self.layoutWidget_90)
        self.frame_37.setMinimumSize(QtCore.QSize(290, 76))
        self.frame_37.setMaximumSize(QtCore.QSize(290, 76))
        self.frame_37.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 12px;")
        self.frame_37.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_37.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_37.setObjectName("frame_37")
        self.label_197 = QtWidgets.QLabel(self.frame_37)
        self.label_197.setGeometry(QtCore.QRect(15, 21, 36, 36))
        self.label_197.setMaximumSize(QtCore.QSize(36, 36))
        self.label_197.setText("")
        self.label_197.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Realtime.png")))
        self.label_197.setScaledContents(True)
        self.label_197.setObjectName("label_197")
        self.label_198 = QtWidgets.QLabel(self.frame_37)
        self.label_198.setGeometry(QtCore.QRect(64, -15, 209, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_198.setFont(font)
        self.label_198.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.label_198.setObjectName("label_198")
        self.layoutWidget_9 = QtWidgets.QWidget(self.frame_37)
        self.layoutWidget_9.setGeometry(QtCore.QRect(58, 30, 221, 41))
        self.layoutWidget_9.setObjectName("layoutWidget_9")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget_9)
        self.verticalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_3.setContentsMargins(5, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_199 = QtWidgets.QLabel(self.layoutWidget_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_199.sizePolicy().hasHeightForWidth())
        self.label_199.setSizePolicy(sizePolicy)
        self.label_199.setMinimumSize(QtCore.QSize(207, 31))
        self.label_199.setMaximumSize(QtCore.QSize(207, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_199.setFont(font)
        self.label_199.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;\n"
"padding-left: 5px;\n"
"padding-bottom: 2px;")
        self.label_199.setText("")
        self.label_199.setTextFormat(QtCore.Qt.PlainText)
        self.label_199.setScaledContents(False)
        self.label_199.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_199.setWordWrap(False)
        self.label_199.setObjectName("label_199")
        self.verticalLayout_3.addWidget(self.label_199)
        self.horizontalSlider_12 = QtWidgets.QSlider(self.layoutWidget_9)
        self.horizontalSlider_12.setMinimumSize(QtCore.QSize(207, 20))
        self.horizontalSlider_12.setMaximumSize(QtCore.QSize(207, 20))
        self.horizontalSlider_12.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.horizontalSlider_12.setStyleSheet("QSlider {\n"
"    \n"
"    background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    background: #2FB57D;\n"
"    border-top-right-radius: 0px;\n"
"    border-bottom-right-radius: 4px;\n"
"    border-top-left-radius: 0; \n"
"    border-bottom-left-radius: 4; \n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    height: 15px;\n"
"    width: 15px;\n"
"    border-radius: 7px;\n"
"    margin: -5px 0;\n"
"    background:#2FB57D;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #c4c4c4;\n"
"    border-top-right-radius: 0px;\n"
"    border-bottom-right-radius: 4px;\n"
"    border-top-left-radius: 0; \n"
"    border-bottom-left-radius: 4; \n"
"}")
        self.horizontalSlider_12.setMinimum(1)
        self.horizontalSlider_12.setMaximum(30)
        self.horizontalSlider_12.setSingleStep(5)
        self.horizontalSlider_12.setPageStep(5)
        self.horizontalSlider_12.setProperty("value", 15)
        self.horizontalSlider_12.setSliderPosition(15)
        self.horizontalSlider_12.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_12.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_12.setObjectName("horizontalSlider_12")
        self.verticalLayout_3.addWidget(self.horizontalSlider_12)
        self.horizontalSlider_12.valueChanged.connect(lambda value: self.label_199.setText(str(value) + (" года" 
                if value == 2 or value == 3 or value == 4 else " год" if value == 1 else " лет")))
        
        self.verticalLayout_80.addWidget(self.frame_37)
        self.frame_38 = QtWidgets.QFrame(self.layoutWidget_90)
        self.frame_38.setMinimumSize(QtCore.QSize(290, 76))
        self.frame_38.setMaximumSize(QtCore.QSize(290, 76))
        self.frame_38.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 12px;")
        self.frame_38.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_38.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_38.setObjectName("frame_38")
        self.label_200 = QtWidgets.QLabel(self.frame_38)
        self.label_200.setGeometry(QtCore.QRect(15, 21, 36, 36))
        self.label_200.setMaximumSize(QtCore.QSize(36, 36))
        self.label_200.setText("")
        self.label_200.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Get a Discount.png")))
        self.label_200.setScaledContents(True)
        self.label_200.setObjectName("label_200")
        self.label_201 = QtWidgets.QLabel(self.frame_38)
        self.label_201.setGeometry(QtCore.QRect(64, -15, 209, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_201.setFont(font)
        self.label_201.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.label_201.setObjectName("label_201")
        self.layoutWidget_48 = QtWidgets.QWidget(self.frame_38)
        self.layoutWidget_48.setGeometry(QtCore.QRect(58, 30, 221, 41))
        self.layoutWidget_48.setObjectName("layoutWidget_48")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.layoutWidget_48)
        self.verticalLayout_12.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_12.setContentsMargins(5, 0, 0, 0)
        self.verticalLayout_12.setSpacing(0)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.label_202 = QtWidgets.QLabel(self.layoutWidget_48)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_202.sizePolicy().hasHeightForWidth())
        self.label_202.setSizePolicy(sizePolicy)
        self.label_202.setMinimumSize(QtCore.QSize(207, 31))
        self.label_202.setMaximumSize(QtCore.QSize(207, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_202.setFont(font)
        self.label_202.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;\n"
"padding-left: 5px;\n"
"padding-bottom: 2px;")
        self.label_202.setText("")
        self.label_202.setTextFormat(QtCore.Qt.PlainText)
        self.label_202.setScaledContents(False)
        self.label_202.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_202.setWordWrap(False)
        self.label_202.setObjectName("label_202")
        self.verticalLayout_12.addWidget(self.label_202)
        self.horizontalSlider_13 = QtWidgets.QSlider(self.layoutWidget_48)
        self.horizontalSlider_13.setMinimumSize(QtCore.QSize(207, 20))
        self.horizontalSlider_13.setMaximumSize(QtCore.QSize(207, 20))
        self.horizontalSlider_13.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.horizontalSlider_13.setStyleSheet("QSlider {\n"
"    \n"
"    background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"    height: 5px;\n"
"    background: #2FB57D;\n"
"    border-top-right-radius: 0px;\n"
"    border-bottom-right-radius: 4px;\n"
"    border-top-left-radius: 0; \n"
"    border-bottom-left-radius: 4; \n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    height: 15px;\n"
"    width: 15px;\n"
"    border-radius: 7px;\n"
"    margin: -5px 0;\n"
"    background:#2FB57D;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: #c4c4c4;\n"
"    border-top-right-radius: 0px;\n"
"    border-bottom-right-radius: 4px;\n"
"    border-top-left-radius: 0; \n"
"    border-bottom-left-radius: 4; \n"
"}")
        self.horizontalSlider_13.setMinimum(1)
        self.horizontalSlider_13.setMaximum(300)
        self.horizontalSlider_13.setSingleStep(1)
        self.horizontalSlider_13.setPageStep(1)
        self.horizontalSlider_13.setProperty("value", 10)
        self.horizontalSlider_13.setSliderPosition(10)
        self.horizontalSlider_13.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_13.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_13.setObjectName("horizontalSlider_13")
        self.verticalLayout_12.addWidget(self.horizontalSlider_13)
        self.horizontalSlider_13.valueChanged.connect(lambda value: self.label_202.setText(str(value/10)))
        self.verticalLayout_80.addWidget(self.frame_38)
        self.widget_21 = QtWidgets.QWidget(self.frame_32)
        self.widget_21.setGeometry(QtCore.QRect(10, 590, 322, 421))
        self.widget_21.setMinimumSize(QtCore.QSize(310, 0))
        self.widget_21.setMaximumSize(QtCore.QSize(322, 500))
        self.widget_21.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_21.setObjectName("widget_21")
        self.widget_22 = QtWidgets.QWidget(self.widget_21)
        self.widget_22.setGeometry(QtCore.QRect(16, 51, 290, 221))
        self.widget_22.setMinimumSize(QtCore.QSize(200, 0))
        self.widget_22.setMaximumSize(QtCore.QSize(335, 500))
        self.widget_22.setStyleSheet("background-color: rgb(235, 233, 244);\n"
"border: none;\n"
"border-radius: 12px;\n"
"")
        self.widget_22.setObjectName("widget_22")
        self.label_137 = QtWidgets.QLabel(self.widget_22)
        self.label_137.setGeometry(QtCore.QRect(16, 16, 131, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_137.setFont(font)
        self.label_137.setStyleSheet("color: rgb(20, 20, 20)")
        self.label_137.setObjectName("label_137")
        self.label_138 = QtWidgets.QLabel(self.widget_22)
        self.label_138.setGeometry(QtCore.QRect(14, 40, 261, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_138.setFont(font)
        self.label_138.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;\n"
"padding-left: 5px;\n"
"padding-bottom: 2px;\n"
"color: rgb(106, 106, 106);")
        self.label_138.setText("")
        self.label_138.setObjectName("label_138")
        self.label_139 = QtWidgets.QLabel(self.widget_22)
        self.label_139.setGeometry(QtCore.QRect(262, 40, 21, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_139.setFont(font)
        self.label_139.setStyleSheet("color: rgb(106, 106, 106);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_139.setObjectName("label_139")
        self.label_140 = QtWidgets.QLabel(self.widget_22)
        self.label_140.setGeometry(QtCore.QRect(14, 90, 261, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_140.setFont(font)
        self.label_140.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;\n"
"padding-left: 5px;\n"
"padding-bottom: 2px;\n"
"color: rgb(106, 106, 106);")
        self.label_140.setText("")
        self.label_140.setObjectName("label_140")
        self.label_141 = QtWidgets.QLabel(self.widget_22)
        self.label_141.setGeometry(QtCore.QRect(16, 66, 181, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_141.setFont(font)
        self.label_141.setStyleSheet("color: rgb(20, 20, 20)")
        self.label_141.setObjectName("label_141")
        self.label_142 = QtWidgets.QLabel(self.widget_22)
        self.label_142.setGeometry(QtCore.QRect(262, 90, 21, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_142.setFont(font)
        self.label_142.setStyleSheet("color: rgb(106, 106, 106);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_142.setObjectName("label_142")
        self.label_143 = QtWidgets.QLabel(self.widget_22)
        self.label_143.setGeometry(QtCore.QRect(14, 140, 261, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_143.setFont(font)
        self.label_143.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;\n"
"padding-left: 5px;\n"
"padding-bottom: 2px;\n"
"color: rgb(106, 106, 106);")
        self.label_143.setText("")
        self.label_143.setObjectName("label_143")
        self.label_144 = QtWidgets.QLabel(self.widget_22)
        self.label_144.setGeometry(QtCore.QRect(16, 116, 191, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_144.setFont(font)
        self.label_144.setStyleSheet("color: rgb(20, 20, 20)")
        self.label_144.setObjectName("label_144")
        self.label_145 = QtWidgets.QLabel(self.widget_22)
        self.label_145.setGeometry(QtCore.QRect(262, 140, 21, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_145.setFont(font)
        self.label_145.setStyleSheet("color: rgb(106, 106, 106);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_145.setObjectName("label_145")
        self.label_146 = QtWidgets.QLabel(self.widget_22)
        self.label_146.setGeometry(QtCore.QRect(14, 190, 261, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_146.setFont(font)
        self.label_146.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 5px;\n"
"padding-left: 5px;\n"
"padding-bottom: 2px;\n"
"color: rgb(106, 106, 106);")
        self.label_146.setText("")
        self.label_146.setObjectName("label_146")
        self.label_147 = QtWidgets.QLabel(self.widget_22)
        self.label_147.setGeometry(QtCore.QRect(16, 166, 141, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_147.setFont(font)
        self.label_147.setStyleSheet("color: rgb(20, 20, 20)")
        self.label_147.setObjectName("label_147")
        self.label_148 = QtWidgets.QLabel(self.widget_22)
        self.label_148.setGeometry(QtCore.QRect(262, 190, 21, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_148.setFont(font)
        self.label_148.setStyleSheet("color: rgb(106, 106, 106);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_148.setObjectName("label_148")
        self.label_190 = QtWidgets.QLabel(self.widget_21)
        self.label_190.setGeometry(QtCore.QRect(16, 10, 292, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_190.setFont(font)
        self.label_190.setStyleSheet("color: rgb(20, 20, 20);")
        self.label_190.setObjectName("label_190")
        self.pushButton_13 = QtWidgets.QPushButton(self.widget_21)
        self.pushButton_13.setGeometry(QtCore.QRect(16, 335, 292, 67))
        self.pushButton_13.setMinimumSize(QtCore.QSize(290, 67))
        self.pushButton_13.setMaximumSize(QtCore.QSize(292, 67))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_13.setFont(font)
        self.pushButton_13.setStyleSheet("QPushButton {\n"
"background-color: rgb(47, 152, 108);\n"
"border: none;\n"
"border-radius: 17px;\n"
"color: rgb(255, 255, 255)\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgb(43, 139, 97);\n"
"}")
        self.pushButton_13.setObjectName("pushButton_13")
        self.horizontalSlider_10 = QtWidgets.QSlider(self.widget_21)
        self.horizontalSlider_10.setGeometry(QtCore.QRect(16, 300, 291, 20))
        self.horizontalSlider_10.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.horizontalSlider_10.setStyleSheet("QSlider::groove:horizontal {\n"
"    height:10px;\n"
"    background: rgb(60, 156, 132);\n"
"    border-radius: 5px;\n"
"}\n"
"\n"
"\n"
"QSlider::add-page:horizontal {\n"
"    background: rgb(222, 146, 58);\n"
"    border-radius: 5px;\n"
"}")
        self.horizontalSlider_10.setMinimum(1)
        self.horizontalSlider_10.setMaximum(100)
        self.horizontalSlider_10.setSingleStep(1)
        self.horizontalSlider_10.setPageStep(1)
        self.horizontalSlider_10.setProperty("value", 30)
        self.horizontalSlider_10.setSliderPosition(30)
        self.horizontalSlider_10.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_10.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.horizontalSlider_10.setObjectName("horizontalSlider_10")
        self.label_149 = QtWidgets.QLabel(self.widget_21)
        self.label_149.setGeometry(QtCore.QRect(16, 284, 11, 11))
        self.label_149.setStyleSheet("background-color: rgb(60, 156, 132);\n"
"border-radius: 5px;")
        self.label_149.setText("")
        self.label_149.setObjectName("label_149")
        self.label_150 = QtWidgets.QLabel(self.widget_21)
        self.label_150.setGeometry(QtCore.QRect(34, 281, 51, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_150.setFont(font)
        self.label_150.setStyleSheet("color: rgb(83, 83, 83)")
        self.label_150.setObjectName("label_150")
        self.label_151 = QtWidgets.QLabel(self.widget_21)
        self.label_151.setGeometry(QtCore.QRect(118, 281, 81, 16))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_151.setFont(font)
        self.label_151.setStyleSheet("color: rgb(83, 83, 83)")
        self.label_151.setObjectName("label_151")
        self.label_152 = QtWidgets.QLabel(self.widget_21)
        self.label_152.setGeometry(QtCore.QRect(100, 284, 11, 11))
        self.label_152.setStyleSheet("background-color: rgb(222, 146, 58);\n"
"border-radius: 5px;")
        self.label_152.setText("")
        self.label_152.setObjectName("label_152")
        self.gridLayout_6.addWidget(self.frame_32, 0, 0, 1, 1)
        self.scrollArea_5.setWidget(self.scrollAreaWidgetContents_6)
        self.stackedWidget.addWidget(self.page_mortgage)
        self.pahe_info = QtWidgets.QWidget()
        self.pahe_info.setObjectName("pahe_info")
        self.scrollArea_11 = QtWidgets.QScrollArea(self.pahe_info)
        self.scrollArea_11.setGeometry(QtCore.QRect(0, 9, 361, 800))
        self.scrollArea_11.setMinimumSize(QtCore.QSize(0, 800))
        self.scrollArea_11.setMaximumSize(QtCore.QSize(16777215, 800))
        self.scrollArea_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.scrollArea_11.setLineWidth(1)
        self.scrollArea_11.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea_11.setWidgetResizable(True)
        self.scrollArea_11.setObjectName("scrollArea_11")
        self.scrollAreaWidgetContents_11 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_11.setGeometry(QtCore.QRect(0, 0, 359, 1045))
        self.scrollAreaWidgetContents_11.setMinimumSize(QtCore.QSize(0, 1045))
        self.scrollAreaWidgetContents_11.setMaximumSize(QtCore.QSize(360, 1045))
        self.scrollAreaWidgetContents_11.setObjectName("scrollAreaWidgetContents_11")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_11)
        self.gridLayout_11.setContentsMargins(8, 0, 0, 0)
        self.gridLayout_11.setSpacing(0)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.frame_67 = QtWidgets.QFrame(self.scrollAreaWidgetContents_11)
        self.frame_67.setMinimumSize(QtCore.QSize(0, 1200))
        self.frame_67.setMaximumSize(QtCore.QSize(16777215, 1200))
        self.frame_67.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_67.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_67.setObjectName("frame_67")
        self.text_main_17 = QtWidgets.QLabel(self.frame_67)
        self.text_main_17.setGeometry(QtCore.QRect(10, 78, 341, 33))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.text_main_17.setFont(font)
        self.text_main_17.setObjectName("text_main_17")
        self.layoutWidget_144 = QtWidgets.QWidget(self.frame_67)
        self.layoutWidget_144.setGeometry(QtCore.QRect(-10, -10, 361, 61))
        self.layoutWidget_144.setObjectName("layoutWidget_144")
        self.horizontalLayout_108 = QtWidgets.QHBoxLayout(self.layoutWidget_144)
        self.horizontalLayout_108.setContentsMargins(11, 3, 9, 0)
        self.horizontalLayout_108.setSpacing(0)
        self.horizontalLayout_108.setObjectName("horizontalLayout_108")
        self.logo_22 = QtWidgets.QLabel(self.layoutWidget_144)
        self.logo_22.setMinimumSize(QtCore.QSize(40, 44))
        self.logo_22.setMaximumSize(QtCore.QSize(40, 44))
        self.logo_22.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.logo_22.setText("")
        self.logo_22.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Logo.png")))
        self.logo_22.setScaledContents(True)
        self.logo_22.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.logo_22.setObjectName("logo_22")
        self.horizontalLayout_108.addWidget(self.logo_22)
        self.pushButton_65 = QtWidgets.QPushButton(self.layoutWidget_144)
        self.pushButton_65.setMinimumSize(QtCore.QSize(38, 38))
        self.pushButton_65.setMaximumSize(QtCore.QSize(38, 38))
        self.pushButton_65.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.pushButton_65.setText("")
        self.pushButton_65.setIcon(icon1)
        self.pushButton_65.setIconSize(QtCore.QSize(38, 38))
        self.pushButton_65.setObjectName("pushButton_65")
        self.pushButton_65.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.page_main))
        self.pushButton_65.clicked.connect(lambda:self.progress.show())

        self.horizontalLayout_108.addWidget(self.pushButton_65, 0, QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.widget_37 = QtWidgets.QWidget(self.frame_67)
        self.widget_37.setGeometry(QtCore.QRect(10, 137, 322, 81))
        self.widget_37.setMinimumSize(QtCore.QSize(310, 0))
        self.widget_37.setMaximumSize(QtCore.QSize(322, 500))
        self.widget_37.setStyleSheet("background-color: rgb(191, 117, 216);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_37.setObjectName("widget_37")
        self.widget_38 = QtWidgets.QWidget(self.widget_37)
        self.widget_38.setGeometry(QtCore.QRect(0, 30, 322, 51))
        self.widget_38.setMinimumSize(QtCore.QSize(310, 0))
        self.widget_38.setMaximumSize(QtCore.QSize(335, 500))
        self.widget_38.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_38.setObjectName("widget_38")
        self.layoutWidget_145 = QtWidgets.QWidget(self.widget_38)
        self.layoutWidget_145.setGeometry(QtCore.QRect(0, 0, 321, 52))
        self.layoutWidget_145.setObjectName("layoutWidget_145")
        self.verticalLayout_117 = QtWidgets.QVBoxLayout(self.layoutWidget_145)
        self.verticalLayout_117.setContentsMargins(16, 10, 14, 11)
        self.verticalLayout_117.setSpacing(4)
        self.verticalLayout_117.setObjectName("verticalLayout_117")
        self.label_333 = QtWidgets.QLabel(self.layoutWidget_145)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_333.setFont(font)
        self.label_333.setObjectName("label_333")
        self.verticalLayout_117.addWidget(self.label_333)
        self.label_334 = QtWidgets.QLabel(self.layoutWidget_145)
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.label_334.setFont(font)
        self.label_334.setStyleSheet("color: rgb(161, 161, 161);")
        self.label_334.setObjectName("label_334")
        self.verticalLayout_117.addWidget(self.label_334)
        self.widget_39 = QtWidgets.QWidget(self.frame_67)
        self.widget_39.setGeometry(QtCore.QRect(10, 235, 322, 121))
        self.widget_39.setMinimumSize(QtCore.QSize(310, 0))
        self.widget_39.setMaximumSize(QtCore.QSize(322, 500))
        self.widget_39.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_39.setObjectName("widget_39")
        self.label_335 = QtWidgets.QLabel(self.widget_39)
        self.label_335.setGeometry(QtCore.QRect(16, 35, 291, 21))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.label_335.setFont(font)
        self.label_335.setStyleSheet("color: rgb(161, 161, 161);")
        self.label_335.setObjectName("label_335")
        self.label_336 = QtWidgets.QLabel(self.widget_39)
        self.label_336.setGeometry(QtCore.QRect(16, 14, 291, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_336.setFont(font)
        self.label_336.setObjectName("label_336")
        self.layoutWidget2 = QtWidgets.QWidget(self.widget_39)
        self.layoutWidget2.setGeometry(QtCore.QRect(10, 60, 301, 31))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout.setContentsMargins(5, 0, 5, 0)
        self.horizontalLayout.setSpacing(16)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_337 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_337.setMaximumSize(QtCore.QSize(16777215, 24))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_337.setFont(font)
        self.label_337.setStyleSheet("background-color: rgb(210, 245, 209);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 7px;\n"
"padding-left: 5px;")
        self.label_337.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_337.setObjectName("label_337")
        self.horizontalLayout.addWidget(self.label_337)
        self.label_340 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_340.setMaximumSize(QtCore.QSize(16777215, 24))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_340.setFont(font)
        self.label_340.setStyleSheet("background-color: rgb(210, 245, 209);\n"
"color: rgb(27, 27, 27);\n"
"border-radius: 7px;\n"
"padding-left: 5px;")
        self.label_340.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_340.setObjectName("label_340")
        self.horizontalLayout.addWidget(self.label_340)
        self.layoutWidget3 = QtWidgets.QWidget(self.widget_39)
        self.layoutWidget3.setGeometry(QtCore.QRect(10, 89, 301, 21))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.horizontalLayout_109 = QtWidgets.QHBoxLayout(self.layoutWidget3)
        self.horizontalLayout_109.setContentsMargins(5, 0, 5, 0)
        self.horizontalLayout_109.setSpacing(16)
        self.horizontalLayout_109.setObjectName("horizontalLayout_109")
        self.label_338 = QtWidgets.QLabel(self.layoutWidget3)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_338.setFont(font)
        self.label_338.setStyleSheet("color: rgb(151, 151, 151)")
        self.label_338.setAlignment(QtCore.Qt.AlignCenter)
        self.label_338.setObjectName("label_338")
        self.horizontalLayout_109.addWidget(self.label_338)
        self.label_339 = QtWidgets.QLabel(self.layoutWidget3)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_339.setFont(font)
        self.label_339.setStyleSheet("color: rgb(151, 151, 151)")
        self.label_339.setAlignment(QtCore.Qt.AlignCenter)
        self.label_339.setObjectName("label_339")
        self.horizontalLayout_109.addWidget(self.label_339)

        self.widget_40 = CustomWidget1(self.frame_67)  # QtWidgets.QWidget(self.frame_67)

        self.label_342 = QtWidgets.QLabel(self.widget_40)
        self.label_342.setGeometry(QtCore.QRect(16, 12, 291, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_342.setFont(font)
        self.label_342.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.label_342.setObjectName("label_342")
        self.layoutWidget4 = QtWidgets.QWidget(self.widget_40)
        self.layoutWidget4.setGeometry(QtCore.QRect(20, 200, 281, 31))
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.horizontalLayout_110 = QtWidgets.QHBoxLayout(self.layoutWidget4)
        self.horizontalLayout_110.setContentsMargins(4, 0, 0, 0)
        self.horizontalLayout_110.setObjectName("horizontalLayout_110")
        self.label_353 = QtWidgets.QLabel(self.layoutWidget4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_353.setFont(font)
        self.label_353.setStyleSheet("color: rgb(161, 161, 161);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_353.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_353.setObjectName("label_353")
        self.horizontalLayout_110.addWidget(self.label_353)
        self.label_354 = QtWidgets.QLabel(self.layoutWidget4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_354.setFont(font)
        self.label_354.setStyleSheet("color: rgb(161, 161, 161);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_354.setAlignment(QtCore.Qt.AlignCenter)
        self.label_354.setObjectName("label_354")
        self.horizontalLayout_110.addWidget(self.label_354)
        self.label_355 = QtWidgets.QLabel(self.layoutWidget4)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_355.setFont(font)
        self.label_355.setStyleSheet("color: rgb(161, 161, 161);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_355.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_355.setObjectName("label_355")
        self.horizontalLayout_110.addWidget(self.label_355)

#         self.graphicsView = QtWidgets.QGraphicsView(self.widget_40)
#         self.graphicsView.setGeometry(QtCore.QRect(16, 61, 290, 141))
#         self.graphicsView.setStyleSheet("background-color: rgb(244, 243, 249); \n"
# "border-radius: 12px")
#         self.graphicsView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
#         self.graphicsView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
#         self.graphicsView.setObjectName("graphicsView")

        self.widget_41 = QtWidgets.QWidget(self.frame_67)
        self.widget_41.setGeometry(QtCore.QRect(10, 870, 322, 144))
        self.widget_41.setMinimumSize(QtCore.QSize(310, 0))
        self.widget_41.setMaximumSize(QtCore.QSize(322, 500))
        self.widget_41.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_41.setObjectName("widget_41")
        self.label_347 = QtWidgets.QLabel(self.widget_41)
        self.label_347.setGeometry(QtCore.QRect(16, 30, 291, 21))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.label_347.setFont(font)
        self.label_347.setStyleSheet("color: rgb(161, 161, 161);")
        self.label_347.setObjectName("label_347")
        self.label_348 = QtWidgets.QLabel(self.widget_41)
        self.label_348.setGeometry(QtCore.QRect(16, 13, 291, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_348.setFont(font)
        self.label_348.setObjectName("label_348")
        self.layoutWidget_148 = QtWidgets.QWidget(self.widget_41)
        self.layoutWidget_148.setGeometry(QtCore.QRect(10, 60, 301, 71))
        self.layoutWidget_148.setObjectName("layoutWidget_148")
        self.horizontalLayout_112 = QtWidgets.QHBoxLayout(self.layoutWidget_148)
        self.horizontalLayout_112.setContentsMargins(5, 0, 5, 0)
        self.horizontalLayout_112.setSpacing(10)
        self.horizontalLayout_112.setObjectName("horizontalLayout_112")
        self.widget_42 = QtWidgets.QWidget(self.layoutWidget_148)
        self.widget_42.setMaximumSize(QtCore.QSize(16777215, 66))
        self.widget_42.setStyleSheet("background-color: rgb(250, 250, 204);\n"
"border-radius: 12px")
        self.widget_42.setObjectName("widget_42")
        self.label_349 = QtWidgets.QLabel(self.widget_42)
        self.label_349.setGeometry(QtCore.QRect(8, 5, 131, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_349.setFont(font)
        self.label_349.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_349.setObjectName("label_349")
        self.label_350 = QtWidgets.QLabel(self.widget_42)
        self.label_350.setGeometry(QtCore.QRect(8, 20, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.label_350.setFont(font)
        self.label_350.setStyleSheet("color: rgb(143, 143, 143);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_350.setObjectName("label_350")
        self.pushButton_8 = QtWidgets.QPushButton(self.widget_42)
        self.pushButton_8.setGeometry(QtCore.QRect(0, 0, 141, 66))
        self.pushButton_8.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border-radius: 12px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton_8.setText("")
        self.pushButton_8.setObjectName("pushButton_8")
        self.horizontalLayout_112.addWidget(self.widget_42)
        self.widget_43 = QtWidgets.QWidget(self.layoutWidget_148)
        self.widget_43.setMaximumSize(QtCore.QSize(16777215, 66))
        self.widget_43.setStyleSheet("background-color: rgb(250, 250, 204);\n"
"border-radius: 12px")
        self.widget_43.setObjectName("widget_43")
        self.label_351 = QtWidgets.QLabel(self.widget_43)
        self.label_351.setGeometry(QtCore.QRect(6, 5, 131, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_351.setFont(font)
        self.label_351.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_351.setObjectName("label_351")
        self.label_352 = QtWidgets.QLabel(self.widget_43)
        self.label_352.setGeometry(QtCore.QRect(6, 20, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.label_352.setFont(font)
        self.label_352.setStyleSheet("color: rgb(143, 143, 143);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_352.setObjectName("label_352")
        self.pushButton_10 = QtWidgets.QPushButton(self.widget_43)
        self.pushButton_10.setGeometry(QtCore.QRect(0, 0, 140, 66))
        self.pushButton_10.setStyleSheet("QPushButton {\n"
"background-color: rgba(255, 255, 255, 0);\n"
"border-radius: 12px;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"background-color: rgba(0, 0, 0, 7)\n"
"}")
        self.pushButton_10.setText("")
        self.pushButton_10.setObjectName("pushButton_10")
        self.horizontalLayout_112.addWidget(self.widget_43)

        self.widget_45 = CustomWidget2(self.frame_67)

        self.widget_45.setGeometry(QtCore.QRect(10, 622, 322, 231))
        self.widget_45.setMinimumSize(QtCore.QSize(310, 0))
        self.widget_45.setMaximumSize(QtCore.QSize(322, 500))
        self.widget_45.setStyleSheet("background-color: rgb(248, 247, 252);\n"
"border: none;\n"
"border-radius: 21px;\n"
"")
        self.widget_45.setObjectName("widget_45")
        self.label_344 = QtWidgets.QLabel(self.widget_45)
        self.label_344.setGeometry(QtCore.QRect(16, 12, 291, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_344.setFont(font)
        self.label_344.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.label_344.setObjectName("label_344")
        self.layoutWidget_65 = QtWidgets.QWidget(self.widget_45)
        self.layoutWidget_65.setGeometry(QtCore.QRect(20, 200, 281, 31))
        self.layoutWidget_65.setObjectName("layoutWidget_65")
        self.horizontalLayout_113 = QtWidgets.QHBoxLayout(self.layoutWidget_65)
        self.horizontalLayout_113.setContentsMargins(4, 0, 0, 0)
        self.horizontalLayout_113.setObjectName("horizontalLayout_113")
        self.label_359 = QtWidgets.QLabel(self.layoutWidget_65)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_359.setFont(font)
        self.label_359.setStyleSheet("color: rgb(161, 161, 161);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_359.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_359.setObjectName("label_359")
        self.horizontalLayout_113.addWidget(self.label_359)
        self.label_360 = QtWidgets.QLabel(self.layoutWidget_65)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_360.setFont(font)
        self.label_360.setStyleSheet("color: rgb(161, 161, 161);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_360.setAlignment(QtCore.Qt.AlignCenter)
        self.label_360.setObjectName("label_360")
        self.horizontalLayout_113.addWidget(self.label_360)
        self.label_361 = QtWidgets.QLabel(self.layoutWidget_65)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_361.setFont(font)
        self.label_361.setStyleSheet("color: rgb(161, 161, 161);\n"
"background-color: rgba(255, 255, 255, 0);")
        self.label_361.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_361.setObjectName("label_361")
        self.horizontalLayout_113.addWidget(self.label_361)
#         self.graphicsView_2 = QtWidgets.QGraphicsView(self.widget_45)
#         self.graphicsView_2.setGeometry(QtCore.QRect(16, 61, 290, 141))
#         self.graphicsView_2.setStyleSheet("background-color: rgb(244, 243, 249); \n"
# "border-radius: 12px")
#         self.graphicsView_2.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
#         self.graphicsView_2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
#         self.graphicsView_2.setObjectName("graphicsView_2")
        self.progress = QtWidgets.QWidget(self.frame_67)
        self.progress.setEnabled(True)
        self.progress.setGeometry(QtCore.QRect(-10, -1, 361, 791))
        self.progress.setObjectName("progress")
        self.logo_13 = QtWidgets.QLabel(self.progress)
        self.logo_13.setGeometry(QtCore.QRect(145, 300, 70, 77))
        self.logo_13.setMinimumSize(QtCore.QSize(70, 77))
        self.logo_13.setMaximumSize(QtCore.QSize(60, 66))
        self.logo_13.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.logo_13.setText("")
        self.logo_13.setPixmap(QtGui.QPixmap(os.path.join(icon_dir, "Logo.png")))
        self.logo_13.setScaledContents(True)
        self.logo_13.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.logo_13.setObjectName("logo_13")
        self.label_154 = QtWidgets.QLabel(self.progress)
        self.label_154.setGeometry(QtCore.QRect(5, 420, 351, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_154.setFont(font)
        self.label_154.setStyleSheet("color: rgb(86, 86, 86);")
        self.label_154.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_154.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_154.setAlignment(QtCore.Qt.AlignCenter)
        self.label_154.setObjectName("label_154")
        self.progressBar = QtWidgets.QProgressBar(self.progress)
        self.progressBar.setGeometry(QtCore.QRect(85, 400, 191, 7))
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 7))
        self.progressBar.setStyleSheet("QProgressBar {\n"
"    border: none;\n"
"    border-radius: 3px;\n"
"    background-color: #E7EEFF;\n"
"    height: 5px;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"    background-color: #396AEA;\n"
"    border-radius: 3px;\n"
"}")
        self.progressBar.setMaximum(1000)
        self.progressBar.setProperty("value", 1)
        self.progressBar.setTextVisible(False)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_11.addWidget(self.frame_67, 0, 0, 1, 1)
        self.scrollArea_11.setWidget(self.scrollAreaWidgetContents_11)
        self.stackedWidget.addWidget(self.pahe_info)
        self.pushButton_11 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_11.setGeometry(QtCore.QRect(966, 1620, 44, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_11.setFont(font)
        self.pushButton_11.setStyleSheet("QPushButton {\n"
"color:rgb(96, 178, 255)\n"
"}\n"
"\n"
"QPushButton:pressed {color: rgb(255, 41, 152);}")
        self.pushButton_11.setObjectName("pushButton_11")
        MainWindow.setCentralWidget(self.centralwidget)
        self.text_main.setBuddy(self.text_main)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Анализ и прогнозирование"))
        self.text_main.setText(_translate("MainWindow", " Главная"))
        self.label_10.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; color:#e6e6e6;\">Сравните квартиры между собой</span></p></body></html>"))
        self.label_11.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#acacac;\">Выберите наиболее подходящий вариант</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#1e1e1e;\">Последние оценки</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a1a1a1;\">Получите возможность вернуться <br/>к прошлым анализам</span></p></body></html>"))
        

        with open('info.csv', 'r') as file:
                num_lines = len(file.readlines()) - 1
                if num_lines == 0:
                        num_lines = ""
                self.pushButton_3.setText(_translate("MainWindow", f"ВСЕ {num_lines}"))

        self.pushButton_3.clicked.connect(lambda:self.stackedWidget.setCurrentWidget(self.pahe_history))

        self.label_9.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">08.01.2024 - 120 м², 15.7 млн, руб</span></p></body></html>"))
        self.label_12.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a0a0a0;\">Долгопрудный, Пацаева, ЖК Бригантина, 7/20 <br/>этаж, 3 комнаты, 2020 год</span></p></body></html>"))
        self.label_16.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">08.01.2024 - 120 м², 15.7 млн, руб</span></p></body></html>"))
        self.label_17.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a0a0a0;\">Долгопрудный, Пацаева, ЖК Бригантина, 7/20 <br/>этаж, 3 комнаты, 2020 год</span></p></body></html>"))
        self.empty.setText(_translate("MainWindow", "История пуста"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p>Рассчитать стоимость квартиры</p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a1a1a1;\">Оцените стоимость квартиры в два клика</span></p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p>Налоговый калькулятор</p></body></html>"))
        self.label_8.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a1a1a1;\">Рассчитайте кадастровый вычет</span></p></body></html>"))
        self.label_14.setText(_translate("MainWindow", "<html><head/><body><p>Ипотечный калькулятор</p></body></html>"))
        self.label_82.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a1a1a1;\">Рассчитать ставку, сумму, срок платежа</span></p></body></html>"))
        self.text_main_7.setText(_translate("MainWindow", " Сравнение квартир"))
        self.label_87.setText(_translate("MainWindow", "Выберите объекты"))
        self.empty_3.setText(_translate("MainWindow", "Выбрать"))
        self.label_88.setText(_translate("MainWindow", "120 м², 15.7 млн, руб"))
        self.label_89.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a1a1a1;\">Долгопрудный, Пацаева, <br/>ЖК Бригантина, 7/20 <br/>этаж</span></p></body></html>"))
        self.label_90.setText(_translate("MainWindow", "120 м², 15.7 млн, руб"))
        self.label_91.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a1a1a1;\">Долгопрудный, Пацаева, <br/>ЖК Бригантина, 7/20 <br/>этаж</span></p></body></html>"))
        self.empty_2.setText(_translate("MainWindow", "Выбрать"))
        self.label_92.setText(_translate("MainWindow", "Оценка квартир"))
        self.label_93.setText(_translate("MainWindow", "Количество этажей в доме"))
        self.label_94.setText(_translate("MainWindow", "Этаж проживания"))
        self.label_101.setText(_translate("MainWindow", "Общая площадь"))
        self.label_128.setText(_translate("MainWindow", "Жилая площадь"))
        self.label_107.setText(_translate("MainWindow", "Кухонная площадь"))
        self.label_104.setText(_translate("MainWindow", "Количество комнат"))
        self.label_125.setText(_translate("MainWindow", "Год сдачи"))
        self.label_131.setText(_translate("MainWindow", "Город"))
        self.label_119.setText(_translate("MainWindow", "Район"))
        self.label_110.setText(_translate("MainWindow", "Улица"))
        self.label_122.setText(_translate("MainWindow", "Жилой комплекс"))
        self.label_134.setText(_translate("MainWindow", "Средняя цена"))
        self.label_113.setText(_translate("MainWindow", "Изменение цены за 4 месяца"))
        self.label_116.setText(_translate("MainWindow", "Изменение цены через 2 месяца"))
        self.label_153.setText(_translate("MainWindow", "<html><head/><body><p>Выбор объекта</p></body></html>"))
        self.empty_5.setText(_translate("MainWindow", "Список пуст"))
        self.label_167.setText(_translate("MainWindow", "x"))
        self.label_29.setText(_translate("MainWindow", "Результат, руб"))
        self.pushButton_12.setText(_translate("MainWindow", "Рассчитать налог"))
        self.label_27.setText(_translate("MainWindow", "Общая площадь объекта, м²"))
        self.label_21.setText(_translate("MainWindow", "Кадастровая стоимость, руб"))
        self.label_31.setText(_translate("MainWindow", "Ставка, от 0.1% до 2.0%"))
        self.label_24.setText(_translate("MainWindow", "Налоговый вычет, м²"))
        self.text_main_2.setText(_translate("MainWindow", " Калькулятор"))
        self.label_19.setText(_translate("MainWindow", "<html><head/><body><p>Настройки</p></body></html>"))
        self.label_25.setText(_translate("MainWindow", "<html><head/><body><p align=\"justify\"><span style=\" color:#a1a1a1;\">Настройки - ключ к тому, чтобы приложение <br/>стало идеальным инструментом, соответству-<br/>ющим всем потребностям пользователей</span></p></body></html>"))
        self.label_42.setText(_translate("MainWindow", "<html><head/><body><p>Очистить историю оценок</p></body></html>"))
        self.label_43.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a1a1a1;\">Все ранее проведённые анализы <br/>будут удалены</span></p></body></html>"))
        self.label_39.setText(_translate("MainWindow", "<html><head/><body><p>Обратная связь</p></body></html>"))
        self.label_40.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a1a1a1;\">Сообщить о проблемах, предложениях <br/>для улучшения приложения</span></p></body></html>"))
        self.label_45.setText(_translate("MainWindow", "<html><head/><body><p>Вопросы риелтору</p></body></html>"))
        self.label_46.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a1a1a1;\">Получить профессиональную помощь <br/>от экспертов в области недвижимости</span></p></body></html>"))
        self.label_48.setText(_translate("MainWindow", "<html><head/><body><p>Оформление</p></body></html>"))
        self.label_49.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a1a1a1;\">Сменить внешний вид приложения</span></p></body></html>"))
        self.label_51.setText(_translate("MainWindow", "<html><head/><body><p>О приложении</p></body></html>"))
        self.label_52.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a1a1a1;\">Версия 1.0.0</span></p></body></html>"))
        self.text_main_3.setText(_translate("MainWindow", " Помощь"))
        self.text_main_4.setText(_translate("MainWindow", " О приложении"))
        self.label_34.setText(_translate("MainWindow", "Дипломная работа"))
        self.label_35.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a1a1a1;\">На тему ”Разработка многофункционального приложе-<br/>ния для оценки стоимости жилых помещений в Москве<br/>и Московской обл.”. Выполнил Анохин Егор Сергеевич</span></p></body></html>"))
        self.label_32.setText(_translate("MainWindow", "<html><head/><body><p>Исходный код</p></body></html>"))
        self.label_33.setText(_translate("MainWindow", "<html><head/><body><p align=\"justify\"><span style=\" color:#a1a1a1;\">Версия 1.0.0</span></p></body></html>"))
        with open('source_code.txt', 'r', encoding='utf-8') as file:
                text = file.read()
        self.label_55.setText(_translate("MainWindow", text))
        self.label_73.setText(_translate("MainWindow", "Введите критерии объекта"))
        self.label_37.setText(_translate("MainWindow", "Город"))
        self.label_54.setText(_translate("MainWindow", "Улица"))
        self.label_76.setText(_translate("MainWindow", "Район"))
        self.label_78.setText(_translate("MainWindow", "Жилой комплекс"))
        self.label_62.setText(_translate("MainWindow", "Этажность дома"))
        self.label_68.setText(_translate("MainWindow", "Этаж квартиры"))
        self.label_71.setText(_translate("MainWindow", "Общая площадь"))
        self.label_65.setText(_translate("MainWindow", "Жилая площадь"))
        self.label_57.setText(_translate("MainWindow", "Кухонная площадь"))
        self.label_80.setText(_translate("MainWindow", "Год сдачи"))
        self.label_60.setText(_translate("MainWindow", "Количество комнат"))
        self.pushButton_24.setText(_translate("MainWindow", "1"))
        self.pushButton_25.setText(_translate("MainWindow", "2"))
        self.pushButton_23.setText(_translate("MainWindow", "3"))
        self.pushButton_22.setText(_translate("MainWindow", "4+"))
        self.text_main_5.setText(_translate("MainWindow", " Оценка"))
        self.pushButton_21.setText(_translate("MainWindow", "Оценить стоимость!"))
        self.text_main_6.setText(_translate("MainWindow", "Последние анализы"))
        self.label_74.setText(_translate("MainWindow", "<html><head/><body><p>История</p></body></html>"))
        self.label_81.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#a1a1a1;\">Изучение рынка недвижимости - ключ к муд-<br/>рому выбору дома мечты, избегая ошибки</span></p></body></html>"))
        self.empty_4.setText(_translate("MainWindow", "История пуста"))
        self.text_main_9.setText(_translate("MainWindow", " Расчёт ипотеки"))
        self.label_187.setText(_translate("MainWindow", "Свои значения"))
        self.label_189.setText(_translate("MainWindow", "Стоимость недвижимости, руб"))
        self.label_195.setText(_translate("MainWindow", "Первоначальный взнос, руб"))
        self.label_198.setText(_translate("MainWindow", "Срок"))
        self.label_201.setText(_translate("MainWindow", "Ставка, %"))
        self.label_137.setText(_translate("MainWindow", "Сумма кредита"))
        self.label_139.setText(_translate("MainWindow", "₽"))
        self.label_141.setText(_translate("MainWindow", "Ежемесячный платёж"))
        self.label_142.setText(_translate("MainWindow", "₽"))
        self.label_144.setText(_translate("MainWindow", "Переплата по кредиту"))
        self.label_145.setText(_translate("MainWindow", "₽"))
        self.label_147.setText(_translate("MainWindow", "Общая выплата"))
        self.label_148.setText(_translate("MainWindow", "₽"))
        self.label_190.setText(_translate("MainWindow", "Результаты расчёта"))
        self.pushButton_13.setText(_translate("MainWindow", "Рассчитать ипотеку"))
        self.label_150.setText(_translate("MainWindow", "Кредит"))
        self.label_151.setText(_translate("MainWindow", "Переплата"))
        self.text_main_17.setText(_translate("MainWindow", " Информация о квартире"))
        self.label_333.setText(_translate("MainWindow", "3-комн. квартира · 120 м²"))
        self.label_334.setText(_translate("MainWindow", "Долгопрудный, Пацаева, ЖК Бригантина, 7/20 этаж"))
        self.label_335.setText(_translate("MainWindow", "Рассчитано, используя цены сделок и похожие \n"
"объявления, в том числе недавно снятые"))
        self.label_336.setText(_translate("MainWindow", "Оценка стоимости жилья"))
        self.label_337.setText(_translate("MainWindow", "12.5 млн, р"))
        self.label_340.setText(_translate("MainWindow", "12.5-13.0 млн, р"))
        self.label_338.setText(_translate("MainWindow", "Полная стоимость"))
        self.label_339.setText(_translate("MainWindow", "Квадратный метр"))
        self.label_342.setText(_translate("MainWindow", "<html><head/><body><p>Стоимость этой квартиры изменилась <br/>на <span style=\" color:#396aea;\">▲18%</span> за 4 месяца</p></body></html>"))
        self.label_353.setText(_translate("MainWindow", "фев 20"))
        self.label_354.setText(_translate("MainWindow", "апр 20"))
        self.label_355.setText(_translate("MainWindow", "июнь 20"))
        self.label_347.setText(_translate("MainWindow", "При нажатии переносит на объявление сайта ЦИАН"))
        self.label_348.setText(_translate("MainWindow", "Похожие квартиры"))
        self.label_349.setText(_translate("MainWindow", "120 м², 15.7 млн, руб"))
        self.label_350.setText(_translate("MainWindow", "Долгопрудный, Пацаев\n"
"а, ЖК Бригантина, 7/20 \n"
"этаж"))
        self.label_351.setText(_translate("MainWindow", "120 м², 15.7 млн, руб"))
        self.label_352.setText(_translate("MainWindow", "Долгопрудный, Пацаев\n"
"а, ЖК Бригантина, 7/20 \n"
"этаж"))
        self.label_344.setText(_translate("MainWindow", "<html><head/><body><p>Вероятное изменение стоимости этой <br/>квартиры в течении 2 месяцев <span style=\" color:#aa64c3;\">▲2%</span></p></body></html>"))
        self.label_359.setText(_translate("MainWindow", "июнь 20"))
        self.label_360.setText(_translate("MainWindow", "июль 20"))
        self.label_361.setText(_translate("MainWindow", "авг 20"))
        self.label_154.setText(_translate("MainWindow", "Анализ параметров квартиры"))
        self.pushButton_11.setText(_translate("MainWindow", "ВСЕ 6"))

        self.upd_inform()
        self.addSlotToHistory()

        self.addSlotToHistory2()

        self.currentButton = None

        with open('info.csv', 'r') as file:
            csv_reader = csv.reader(file)
            rows = list(csv_reader)
            
            if len(rows) >= 2:
                self.frame.show()  # показать frame
                self.empty.hide()
            
            if len(rows) >= 3:
                self.frame_2.show()  # показать frame_2

        widgets_to_apply_shadow = [self.widget_19, self.widget_21, self.widget_37, self.widget_39, 
                                   self.widget_40, self.widget_45, self.widget_41, self.widget_42, 
                                   self.widget_43, self.widget_14, self.recent_estimates, 
                                   self.widget_10, self.sign_9, self.sign_8, self.sign_7, 
                                   self.sign_6, self.sign_5, self.sign_4, self.sign_3, self.sign_2, 
                                   self.sign, self.sign_15, self.sign_14, self.sign_13, self.sign_12, 
                                   self.sign_11, self.sign_10, self.widget_2, self.widget_4, 
                                   self.widget_5, self.widget_7, self.widget_6, self.widget_8, 
                                   self.pushButton_21, self.background, self.pushButton_13]
        self.applyShadowEffect(widgets_to_apply_shadow)

    def hide_logo(self):
        self.logo.hide()


    def clearLayout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                self.clearLayout(item.layout())


    def clear_data(self):
        df = pd.read_csv('info.csv')
        df = df.head(1)
        df.to_csv('info.csv', index=False)
        
        # Очищаем все строки, кроме первой
        with open('info.csv', 'r') as file:
            lines = file.readlines()
        with open('info.csv', 'w') as file:
            file.write(lines[0])

        while self.verticalLayout_40.count() > 0:
                self.verticalLayout_40.takeAt(0)

        self.clearLayout(self.verticalLayout_13)
        self.added_items.clear()

        _translate = QtCore.QCoreApplication.translate
        with open('info.csv', 'r') as file:
                self.frame.hide()
                self.frame_2.hide()
                self.empty.show()

                num_lines = len(file.readlines()) - 1
                if num_lines == 0:
                        num_lines = ""
                self.pushButton_3.setText(_translate("MainWindow", f"ВСЕ {num_lines}"))


                self.frame.hide()  # показать frame
                self.empty.show()
                self.frame_2.hide()
                
        self.addSlotToHistory()
        self.addSlotToHistory2()

    def open_website(self):
        url = "https://www.cian.ru/podbor-rieltora/"
        QDesktopServices.openUrl(QUrl(url))
        
    def open_gmail(self):
        url = "https://mail.google.com/mail/u/0/?view=cm&fs=1&tf=1&to=minin5742@gmail.com"
        QDesktopServices.openUrl(QUrl(url))

