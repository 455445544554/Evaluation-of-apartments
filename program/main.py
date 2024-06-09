import gui
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import re
import os
from PyQt5 import QtGui
import string
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import gui_test
import warnings
import pickle
import json
import pandas as pd


# Загрузка сохраненной модели из файла
model_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Diplom', 'model')

model_path = os.path.join(model_dir, 'model.pkl')
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Загрузка базы с динамикой цен
model_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Diplom', 'datasets')
dataset_path = os.path.join(model_dir, 'price_dynamics.csv')
df2 = pd.read_csv(dataset_path)

# Загрузка базы с обработанным датасетом
model_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Diplom', 'datasets')
dataset_path = os.path.join(model_dir, 'dataset.csv')
df1 = pd.read_csv(dataset_path)

# Загрузка зависимостей
dic_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Diplom', 'dependencies')

dic_path = os.path.join(dic_dir, 'street.json')
with open(dic_path, 'rb') as file:
    street_json = json.load(file)

dic_path = os.path.join(dic_dir, 'district.json')
with open(dic_path, 'rb') as file:
    district_json = json.load(file)

dic_path = os.path.join(dic_dir, 'residential_complex.json')
with open(dic_path, 'rb') as file:
    residential_complex_json = json.load(file)


# Загрузка словарей с категоризированными данными
dic_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Diplom', 'categorization')

dic_path = os.path.join(dic_dir, 'location_dictionary.json')
with open(dic_path, 'r') as file:
    location_json = json.load(file)

dic_path = os.path.join(dic_dir, 'street_dictionary.json')
with open(dic_path, 'r') as file:
    street_dictionary = json.load(file)

dic_path = os.path.join(dic_dir, 'district_dictionary.json')
with open(dic_path, 'r') as file:
    district_dictionary = json.load(file)

dic_path = os.path.join(dic_dir, 'residential_complex_dictionary.json')
with open(dic_path, 'r') as file:
    residential_complex_dictionary = json.load(file)


def calculate_score(floors_count, floor, total_meters, living_meters, 
                    kitchen_meters, rooms_count, year_of_construction,
                    price_per_m2):
    score = 0

    # Оценка по количеству этажей
    if floors_count <= 4:
        score += 1
    elif floors_count <= 12:
        score += 2
    else:
        score += 1

    # Оценка по этажу
    if floor <= 2:
        score += 1
    elif floor <= 9:
        score += 2
    else:
        score -= 2

    # Оценка по общей площади
    if total_meters >= 100:
        score += 3
    elif total_meters >= 50:
        score += 2
    else:
        score -= 1

    # Оценка по площади жилой зоны
    if living_meters >= 50:
        score += 2
    elif living_meters >= 30:
        score += 1
    else:
        score -= 1

    # Оценка по площади кухни
    if kitchen_meters >= 20:
        score += 2
    elif kitchen_meters >= 10:
        score += 1
    else:
        score -= 1

    # Оценка по количеству комнат
    if rooms_count >= 3:
        score += 2
    elif rooms_count >= 2:
        score += 1
    else:
        score += 1

    # Оценка по году постройки
    if year_of_construction >= 2010:
        score += 3
    elif year_of_construction >= 1992:
        score += 2
    else:
        score += 1

    # Оценка по цене за кв.м
    if price_per_m2 <= 100000:
        score += 1
    elif price_per_m2 <= 150000:
        score += 2
    else:
        score += 3

    return score



if __name__ == "__main__":
    app = QApplication(sys.argv)

    icon_dir = os.path.join(os.path.dirname(__file__), '..', 'program', 'icons')
    logo_path = os.path.join(icon_dir, 'Logo.png')

    window = QMainWindow()
    ui = gui.Ui_MainWindow()
    ui.setupUi(window)

    window.setWindowIcon(QtGui.QIcon(logo_path))  # Set the window icon


    def remove_commas_and_dots(s):
        return re.sub(r'\D', "", s)


    def calculation_of_the_result():
        if ui.label_191.text() and ui.label_196.text() and ui.label_199.text() and ui.label_202.text():
            sn_str = ui.label_191.text() # стоимость
            pv_str = ui.label_196.text() # взнос
            ppk_str = ui.label_199.text() # срок
            ov_str = ui.label_202.text() # ставка

            sn = int(remove_commas_and_dots(sn_str)) # стоимость
            pv = int(remove_commas_and_dots(pv_str)) # взнос
            ppk = int(remove_commas_and_dots(ppk_str)) # срок
            ov = float(remove_commas_and_dots(ov_str)) / 10 # ставка

            summa_kredita = sn - pv

            mesyachnaya_plata = round(summa_kredita * ((((ov/100)/12) * (1 + (ov/100)/12) ** (ppk*12)) 
                                                       / ((1 + (ov/100)/12) ** (ppk*12) - 1)))

            pereplata_po_kreditu = (mesyachnaya_plata * ppk * 12) - summa_kredita

            obshaya_vyplata = mesyachnaya_plata * ppk * 12

            ui.horizontalSlider_10.setMaximum(obshaya_vyplata)
            ui.horizontalSlider_10.setProperty("value", summa_kredita)

            if summa_kredita <= 0 or mesyachnaya_plata <= 0 or pereplata_po_kreditu <= 0 or obshaya_vyplata <= 0:
                ui.label_138.setText("Ошибка данных")
                ui.label_140.setText("Ошибка данных")
                ui.label_143.setText("Ошибка данных")
                ui.label_146.setText("Ошибка данных")
            else:
                ui.label_138.setText("{:,}".format(summa_kredita))
                ui.label_140.setText("{:,}".format(mesyachnaya_plata))
                ui.label_143.setText("{:,}".format(pereplata_po_kreditu))
                ui.label_146.setText("{:,}".format(obshaya_vyplata))

    ui.pushButton_13.clicked.connect(calculation_of_the_result)


    def handle_calculation():
        if ui.lineEdit_2.text() and ui.lineEdit_5.text() and ui.lineEdit_3.text() and ui.label_22.text():
            s = float(''.join(filter(str.isdigit, ui.lineEdit_2.text())))
            m = float(ui.lineEdit_5.text())
            v = float(ui.lineEdit_3.text())
            p = float(ui.label_22.text())

            result = round(int(s - ((s / m) * v)) * (p / 100))

            if result <= 0:
                result = "Ошибка данных"
            else:
                result = "{:,}".format(result)

            ui.label_18.setText(result)
    
    ui.pushButton_12.clicked.connect(lambda: handle_calculation())

    window.show()
    sys.exit(app.exec_())