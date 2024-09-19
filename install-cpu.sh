#!/bin/bash

INVENV=$(python -c 'import sys; print ("1" if sys.prefix != sys.base_prefix else "0")')

if [ "$INVENV" == 1 ]
then
  pip install --requirements requirements.txt
else
  source .env/bin/activate
  pip install --requirements requirements.txt
fi