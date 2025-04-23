#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)

# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $baseDir/../src
rm -rf ./dist/*
# python setup.py sdist upload -r pypi
python setup_touchtext.py sdist
twine upload --repository pypi dist/*

if [ $? -eq '0' ]; then
    rm -rf *.egg-info/
fi