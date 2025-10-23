<div align="center">
<h1>Gaze Response Tracker</h1>
</div>
<div align="center">
<p>Десктопное приложение для регистрации реакции взгляда человека на появляющиеся геометрические объекты различной цветовой составляющей и контрастности</p>

<img src="https://img.shields.io/pypi/pyversions/mediapipe" alt="Swift Language">  ![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/savelevvaa/gaze-tracker-app/total)  ![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-closed/savelevvaa/gaze-tracker-app)  ![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-pr-closed/savelevvaa/gaze-tracker-app)

</div>

## 🎯 Описание приложения для отслеживания реакции взгляда пользователя
Gaze Response Tracker — это инструмент, предназначенный для отслеживания скорости реакции человека на появляющиеся объекты на экране. Проект в корне использует тот же подход что и **gaze-tracker-app:** интерполяция координат зрачков в координаты точки взгляда на экране.

В отличии от **gaze-tracker-app** в данном проекте коэфициенты сгляживания, использующиеся при расчете зоны взгляда сильно уменьшены, для того чтобы не образовывался лаг при расчете координат и мы собирали данные, больше приближенные к реальности.

В результат работы данного приложения можно использовать для исследований в области изучения влияния разноконтрастных графических структур на реакции человеческого взгляда, анализирую собранные данные и построенные по ним инфографике (см. ниже).


## 👨🏻‍💻  Демонстрация работы программы
<div align="center">
<img src="assets/gazeresponse.gif" width="90%">
</div>


## 🌟 Возможности
- 🔍 **Отслеживание реакций взгляда в реальном времени**
- 📊 **Визуализация собранных данных**
- 🖥️ **Простой интерфейс**


## 🛠 Установка
1. Клонируйте репозиторий:
   ```sh
   git clone https://github.com/savelevvaa/gaze-response.git
   ```
2. Перейдите в папку с проектом:
   ```sh
   cd gaze-response
   ```
3. Создайте и активируйте виртуальное окружение (Python <= 3.12):
   ```sh
   python -m venv venv
   venv\Scripts\activate  # Для Mac: source venv/bin/activate
   ```
4. Установите зависимости:
   ```sh
   pip install -r requirements.txt
   ```

## 🚀 Запуск приложения
Запустите скрипты приложения интерпретатором Python следующей командой:
```sh
python app.py
```


## 📸 Инфографика по собранным данным

Главное меню приложения             |  Меню выбора файла для исследования 
:-------------------------:|:-------------------------:
![alt text](assets/img1.png)  |  ![alt text](assets/img2.png)

Справка по работе прилоежения  |  Окно конфигурации запуска трекера 
:-------------------------:|:-------------------------:
![alt text](assets/img3.png)  |  ![alt text](assets/img4.png)

Калибровка работы трекера  |  Проведения испытания  
:-------------------------:|:-------------------------:
![alt text](assets/img5.png)  |  ![alt text](assets/img6.png)

Диаграмма рассеивания взгляда  |  
:-------------------------:|
![alt text](assets/img7.png)  | 



## 🤝 Вклад в проект
Буду рад новым идеям и предложениям! Открывайте issues и отправляйте pull requests.

## 📬 Контакты
По вопросам пишите на [savelevvaa@mail.ru](mailto:savelevvaa@mail.ru) или создавайте issue.
