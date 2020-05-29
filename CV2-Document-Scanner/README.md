# CV2-Document-Scanner
A simple document scanner using python's CV2 library.

Manual:
    It is recommended to make a new virtual environment for this program. You will need virtualenv installed for this to work.
    
    To install virtualenv, run the following command:
        pip install virtualenv
        
    Windows users should also run:
        pip install virtualenvwrapper-win
    
    
    Start by running this command ([venv] is the name of your virtual environment):
        mkvirtualenv [venv] -p python3
    
    Make sure to work on your virtual environment:
        workon [venv]
    
    Install the required libraries on your virtual environment:
        pip install -r requirements.txt
    
    Run the document scanner ([filename] is the name of the image to read):
        python "Document Scanner.py" [filename]
    
    'h' will display the help menu