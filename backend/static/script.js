// document.addEventListener('DOMContentLoaded', () => {
//     const taskInput = document.getElementById('taskInput');
//     const optimizeButton = document.getElementById('optimizeButton');
//     const resultsContainer = document.getElementById('resultsContainer');
//     const taskList = document.getElementById('taskList');
//     const addToCalendarButton = document.getElementById('addToCalendarButton');
//     const connectCalendarButton = document.getElementById('connectCalendarButton');

//     optimizeButton.addEventListener('click', () => {
//         const inputText = taskInput.value.trim();
//         if (!inputText) {
//             alert('Пожалуйста, введите ваши задачи.');
//             return;
//         }

//         // Вызов API бэкенда для оптимизации
//         fetch('/api/optimize', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify({ text: inputText }),
//         })
//         .then(response => response.json())
//         .then(data => {
//             if (data.error) {
//                 alert(`Ошибка: ${data.error}`);
//                 taskList.innerHTML = `<p>Ошибка обработки запроса: ${data.error}</p>`;
//                 resultsContainer.style.display = 'block';
//             } else {
//                 displaySchedule(data.schedule);
//             }
//         })
//         .catch(error => {
//             console.error('Error calling optimize API:', error);
//             alert('Произошла ошибка при обращении к серверу.');
//             taskList.innerHTML = '<p>Произошла ошибка при обращении к серверу. Попробуйте позже.</p>';
//             resultsContainer.style.display = 'block';
//         });
//     });

//     addToCalendarButton.addEventListener('click', () => {
//         // Временно (MVP): Показываем alert
//         // В будущем здесь будет вызов fetch('/api/add_to_calendar', ...)
//         // и передача текущего расписания (которое нужно где-то сохранить после получения от /api/optimize)
//         alert('Готово! Задачи добавлены в ваш календарь 🎉');
//         // Пример, как можно было бы отправить данные (понадобится хранить scheduleData)
//         /*
//         if (window.currentScheduleData) {
//             fetch('/api/add_to_calendar', {
//                 method: 'POST',
//                 headers: { 'Content-Type': 'application/json' },
//                 body: JSON.stringify({ schedule: window.currentScheduleData })
//             })
//             .then(response => response.json())
//             .then(data => {
//                 if (data.status === 'success') {
//                     alert(data.message);
//                 } else {
//                     alert(`Ошибка добавления в календарь: ${data.error || 'Неизвестная ошибка'}`);
//                 }
//             })
//             .catch(error => {
//                 console.error('Error calling add_to_calendar API:', error);
//                 alert('Произошла ошибка при добавлении в календарь.');
//             });
//         } else {
//             alert('Сначала оптимизируйте расписание.');
//         }
//         */
//     });

//     connectCalendarButton.addEventListener('click', () => {
//         // Временно (MVP): Меняем текст кнопки
//         connectCalendarButton.textContent = 'Календарь подключен ✅';
//         connectCalendarButton.disabled = true;
//     });

//     // Удаляем локальную функцию parseTasks, так как парсинг теперь на бэкенде
//     // function parseTasks(text) { ... }

//     // Удаляем локальную функцию formatTime, если она больше нигде не нужна
//     // (в данном MVP она не нужна, т.к. время приходит с бэкенда)
//     // function formatTime(date) { ... }

//     function displaySchedule(schedule) {
//         taskList.innerHTML = ''; // Очищаем предыдущие результаты

//         if (schedule.length === 0) {
//             taskList.innerHTML = '<p>Не удалось распознать задачи. Попробуйте другой формат.</p>';
//             resultsContainer.style.display = 'block';
//             return;
//         }

//         schedule.forEach(item => {
//             const card = document.createElement('div');
//             card.className = 'task-card';

//             const timeP = document.createElement('p');
//             timeP.innerHTML = `Время: <span class="task-time">${item.time}</span>`;

//             const nameP = document.createElement('p');
//             nameP.innerHTML = `Название задачи: <span class="task-name">${item.task}</span>`;

//             card.appendChild(timeP);
//             card.appendChild(nameP);
//             taskList.appendChild(card);
//         });

//         resultsContainer.style.display = 'block';
//     }
// });

addToCalendarButton.addEventListener('click', () => {
    if (!window.currentScheduleData) {
        alert('Сначала оптимизируйте расписание');
        return;
    }

    fetch('/api/add_to_calendar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ schedule: window.currentScheduleData })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            if (data.error === "Not authorized with Yandex") {
                window.location.href = '/auth/yandex';
            } else {
                alert(`Ошибка: ${data.error}`);
            }
        } else {
            alert(data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Произошла ошибка');
    });
});
