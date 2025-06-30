## Установка и Запуск

1.  **Клонируйте репозиторий (если применимо) или скачайте файлы.**
    Если вы клонировали репозиторий:
    ```bash
    git clone https://github.com/itssemen/ai_time_wizard
    cd time_wizard
    ```

2.  **Настройка и запуск приложения:**
    *   Перейдите https://id.yandex.ru/security/app-passwords -> Календарь CalDAV -> запомните пароль
        Перейдите в директорию `backend`:
        ```bash
        cd backend
        #сгенерируйте ключ для Flask
        python3 -c "import secrets; print(secrets.token_hex(16))"
        touch .env
        nano .env
        #Добавьте ключи.
        #Первый - почта для которой сгенерирован пароль здесь https://id.yandex.ru/security/app-passwords
        #Второй - пароль для CalDAV, который сгенерирован здесь https://id.yandex.ru/security/app-passwords
        #Третий - результат python3 -c "import secrets; print(secrets.token_hex(16))"
        #Затем Ctrl+O, Enter, Ctrl+X
        
        ```
    *   Создать и активировать виртуальное окружение:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   Установите зависимости:
        ```bash
        pip install Flask
        pip install -r requirements.txt
        pip install caldav
        ```
    *   Запустите Flask-сервер:
        ```bash
        python app.py
        ```
    *   Приложение будет доступно по адресу `http://127.0.0.1:5000` в вашем веб-браузере. Сервер Flask теперь обслуживает и фронтенд (index.html) и бэкенд API.