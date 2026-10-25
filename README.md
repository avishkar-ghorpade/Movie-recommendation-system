# Movie Recommendation System

## 1. Project Overview

The Movie Recommendation System is a machine learning-based web application designed to recommend movies based on user preferences, ratings, and movie similarities.

The system provides a simple web interface where users can explore movies and receive recommendations.

## 2. Objectives

- Recommend relevant movies based on user preferences.
- Apply Machine Learning techniques to movie data.
- Analyze movie ratings and similarities.
- Provide a user-friendly web interface.
- Demonstrate practical applications of Machine Learning.

## 3. Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Dataset:** CSV movie and rating files
- **Development Tool:** Visual Studio Code

## 4. Project Structure

```text
StoryWorld/
├── backend/
│   ├── app.py
│   ├── make_data.py
│   ├── movies.csv
│   ├── ratings.csv
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   └── ml.html
├── notebooks/
│   └── analysis.ipynb
├── .gitignore
└── README.md
```

## 5. Main Features

- Movie dataset processing
- Movie recommendation functionality
- Rating-based movie analysis
- Machine Learning experimentation
- Web-based user interface
- Movie data analysis

## 6. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/avishkar-ghorpade/Movie-recommendation-system.git
```

### Step 2: Open the Project Folder

```bash
cd Movie-recommendation-system
```

### Step 3: Create a Virtual Environment

```bash
py -m venv .venv
```

### Step 4: Activate the Environment on Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### Step 6: Run the Application

```bash
py backend/app.py
```

Open the local URL displayed in the terminal. It may be `http://127.0.0.1:5000`.

## 7. Dataset

The project includes CSV files containing movie information and ratings. These files support movie analysis and recommendation processing.

## 8. Expected Outcome

The application helps users discover movies that match their interests using the available movie data and recommendation logic.

## 9. Future Enhancements

- Personalized recommendations based on user history
- Improved recommendation algorithms
- Movie posters and detailed movie information
- Cloud deployment
- Responsive design for mobile devices

## 10. Author

**Project:** Movie Recommendation System

**Purpose:** Academic and Machine Learning Project

## 11. License

This project is intended for educational and learning purposes.
