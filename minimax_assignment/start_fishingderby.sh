export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
. /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv -p /usr/bin/python3 fishingderby
python3 main.py settings.yml
