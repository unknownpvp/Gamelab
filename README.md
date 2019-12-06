GameLab
Deployment instructions:

For hosting on server:  
Go to PythonAnywhere.com and register for an account to host.  
Download the git repository or git clone.  
Create a "Manual" web app with Python 3.7  
Set "Source code" and "Working directory" to be /home/yourname/GameLab  
Set virtualenv to the default myvirtualenv
Open the WSGI configuration file, set the path variable as path = '/home/yourname/GameLab'  
Remove the comments at the bottom for flask  
Change the last line to 'from mining import app as application'  
Reload the web app, and it should be available at https://yourname.pythonanywhere.com.  

For hosting locally:  
Set up virtual environment and download flask.  
Download the git repository or git clone.  
Download the dependacies and run the commands:  
python3 -m pip install --user virtualenv  
python3 -m venv env  
source env/bin/activate  
pip install flask  
pip install nltk  
pip install pandas  
Run the following command - python mining.py.  
Open http://127.0.0.1:5000 in web browser.  
