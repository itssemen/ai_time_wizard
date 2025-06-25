# Simple Frontend/Backend Project

This project demonstrates a simple setup with a Python Flask backend and an HTML/JavaScript frontend.

## Prerequisites

- Python 3.x installed
- pip (Python package installer) installed

## Setup and Running

1.  **Clone the repository (if applicable) or download the files.**

git clone https://github.com/itssemen/time_wizard
cd time_wizard

2.  **Backend Setup:**
    *   Navigate to the `backend` directory:
        ```bash
        cd backend
        ```
    *   It's recommended to create and activate a virtual environment:
        ```bash
        python -m venv venv
        # On Windows
        venv\\Scripts\\activate
        # On macOS/Linux
        source venv/bin/activate
        ```
    *   Install Flask:
        ```bash
        pip install Flask
        ```
    *   Run the backend server:
        ```bash
        python app.py
        ```
        The backend will be running on `http://127.0.0.1:5000`.

3.  **Frontend Setup:**
    *   Open a new terminal window/tab.
    *   Navigate to the `frontend` directory:
        ```bash
        cd frontend
        ```
    *   You need a simple HTTP server to serve the `index.html` file and allow the `fetch` request to the backend to work correctly (due to Cross-Origin Resource Sharing policies if you just open the file directly in the browser). Python's built-in HTTP server is sufficient for this.
        *   If you have Python 3:
            ```bash
            python -m http.server 8000
            ```
        *   If you have Python 2 (less common nowadays):
            ```bash
            python -m SimpleHTTPServer 8000
            ```
    *   Open your web browser and go to `http://localhost:8000`.

## How it Works

*   The **backend** (`backend/app.py`) is a Flask application that serves a simple JSON response at the `/api/data` endpoint.
*   The **frontend** (`frontend/index.html`) is a basic HTML page that uses JavaScript's `fetch` API to request data from the backend's `/api/data` endpoint and displays the message on the page.

## Stopping the Servers

*   To stop the backend Flask server, go to its terminal window and press `Ctrl+C`.
*   To stop the frontend HTTP server, go to its terminal window and press `Ctrl+C`.
*   If you used a virtual environment, you can deactivate it:
    ```bash
    deactivate
    ```
