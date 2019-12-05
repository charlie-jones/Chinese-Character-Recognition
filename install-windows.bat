py -m pip install --user --upgrade pip
py -m pip install --user virtualenv
py -m virtualenv env


CALL ".\env\Scripts\activate.bat"
pip install Flask
set FLASK_APP=backendMain.py
set FLASK_ENV=development
flask run --host=0.0.0.0