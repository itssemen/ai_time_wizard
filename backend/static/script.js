document.addEventListener('DOMContentLoaded', () => {
    // Элементы интерфейса
    const taskInput = document.getElementById('taskInput');
    const optimizeButton = document.getElementById('optimizeButton');
    const resultsContainer = document.getElementById('resultsContainer');
    const taskList = document.getElementById('taskList');
    const addToCalendarButton = document.getElementById('addToCalendarButton');
    const connectCalendarButton = document.getElementById('connectCalendarButton');
    const authStatus = document.getElementById('authStatus');

    // Проверка статуса авторизации при загрузке
    checkAuthStatus();

    // Обработчик кнопки "Оптимизировать"
    optimizeButton.addEventListener('click', async () => {
        const inputText = taskInput.value.trim();
        if (!inputText) {
            alert('Пожалуйста, введите ваши задачи.');
            return;
        }

        try {
            const response = await fetch('/api/optimize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: inputText })
            });
            
            const data = await response.json();
            
            if (data.error) {
                showError(data.error);
            } else {
                window.currentScheduleData = data.schedule;
                displaySchedule(data.schedule);
            }
        } catch (error) {
            console.error('Error:', error);
            showError('Произошла ошибка при обращении к серверу');
        }
    });

    // Обработчик кнопки "Добавить в календарь"
    addToCalendarButton.addEventListener('click', async () => {
        if (!window.currentScheduleData) {
            alert('Сначала оптимизируйте расписание');
            return;
        }

        try {
            const response = await fetch('/api/add_to_calendar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ schedule: window.currentScheduleData })
            });
            
            const data = await response.json();
            
            if (data.error) {
                if (data.error === "Not authorized with Yandex") {
                    // Перенаправляем на авторизацию
                    window.location.href = '/auth/yandex';
                } else {
                    alert(`Ошибка: ${data.error}`);
                }
            } else {
                alert(data.message);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Произошла ошибка при добавлении в календарь');
        }
    });

    // Обработчик кнопки "Подключить календарь"
    connectCalendarButton.addEventListener('click', () => {
        window.location.href = '/auth/yandex';
    });

    // Функция проверки статуса авторизации
    async function checkAuthStatus() {
        try {
            const response = await fetch('/auth/status');
            const data = await response.json();
            
            if (data.authenticated) {
                authStatus.textContent = '✓ Календарь подключен';
                authStatus.style.color = 'green';
                connectCalendarButton.style.display = 'none';
            } else {
                authStatus.textContent = 'Календарь не подключен';
                authStatus.style.color = 'red';
            }
        } catch (error) {
            console.error('Auth check error:', error);
        }
    }

    // Функция отображения расписания
    function displaySchedule(schedule) {
        taskList.innerHTML = '';

        if (!schedule || schedule.length === 0) {
            taskList.innerHTML = '<p>Не удалось распознать задачи. Попробуйте другой формат.</p>';
            resultsContainer.style.display = 'block';
            return;
        }

        schedule.forEach(item => {
            const card = document.createElement('div');
            card.className = 'task-card';

            const timeP = document.createElement('p');
            timeP.innerHTML = `<strong>Время:</strong> ${item.time}`;

            const nameP = document.createElement('p');
            nameP.innerHTML = `<strong>Задача:</strong> ${item.task}`;

            card.appendChild(timeP);
            card.appendChild(nameP);
            taskList.appendChild(card);
        });

        resultsContainer.style.display = 'block';
    }

    // Функция показа ошибок
    function showError(message) {
        taskList.innerHTML = `<p class="error">${message}</p>`;
        resultsContainer.style.display = 'block';
    }
});
