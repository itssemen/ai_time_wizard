
document.addEventListener('DOMContentLoaded', () => {
    const taskInput = document.getElementById('taskInput');
    const optimizeBtn = document.getElementById('optimizeButton');
    const addToCalendarBtn = document.getElementById('addToCalendarButton');
    const resultsContainer = document.getElementById('resultsContainer');
    const taskList = document.getElementById('taskList');
    const calendarStatus = document.getElementById('calendarStatus');


    let currentSchedule = null;


    checkCalendarStatus();


    optimizeBtn.addEventListener('click', async () => {
        const text = taskInput.value.trim();
        if (!text) {
            alert('Пожалуйста, введите задачи');
            return;
        }

        try {
            const response = await fetch('/api/optimize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            
            const data = await response.json();
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            currentSchedule = data.schedule;
            renderSchedule(data.schedule);
            resultsContainer.style.display = 'block';
        } catch (error) {
            console.error('Optimize error:', error);
            showError('Ошибка сервера при обработке задач');
        }
    });


    addToCalendarBtn.addEventListener('click', async () => {
        if (!currentSchedule || currentSchedule.length === 0) {
            alert('Сначала создайте расписание');
            return;
        }

        try {
            const response = await fetch('/api/add_to_calendar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ schedule: currentSchedule })
            });
            
            const data = await response.json();
            
            if (data.results.some(r => r.status === 'error')) {
                const errors = data.results
                    .filter(r => r.status === 'error')
                    .map(r => `• ${r.task}: ${r.error}`)
                    .join('\n');
                
                alert(`Не все задачи добавлены:\n${errors}`);
            } else {
                alert('Все задачи успешно добавлены в календарь!');
            }
            

            checkCalendarStatus();
        } catch (error) {
            console.error('Calendar add error:', error);
            alert('Ошибка при добавлении в календарь');
        }
    });


    async function checkCalendarStatus() {
        try {
            const response = await fetch('/api/calendar_status');
            const data = await response.json();
            
            calendarStatus.textContent = data.message;
            calendarStatus.style.color = data.connected ? 'green' : 'red';
            

            addToCalendarBtn.style.display = data.connected ? 'block' : 'none';
        } catch (error) {
            console.error('Status check error:', error);
            calendarStatus.textContent = 'Ошибка проверки статуса';
            calendarStatus.style.color = 'red';
        }
    }


    function renderSchedule(schedule) {
        taskList.innerHTML = '';
        
        schedule.forEach(item => {
            const taskEl = document.createElement('div');
            taskEl.className = 'task-item';
            taskEl.innerHTML = `
                <div class="task-time">${item.time}</div>
                <div class="task-name">${item.task}</div>
            `;
            taskList.appendChild(taskEl);
        });
    }


    function showError(message) {
        taskList.innerHTML = `<div class="error-message">${message}</div>`;
        resultsContainer.style.display = 'block';
    }
});
