#!/bin/bash

INVENV=$(python -c 'import sys; print ("1" if sys.prefix != sys.base_prefix else "0")')

if [ "$INVENV" == 1 ]
then
  pip install -r requirements.txt
else
  source .env/bin/activate
  pip install -r requirements.txt
fi
