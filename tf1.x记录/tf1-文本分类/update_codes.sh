#!/bin/bash

if [[ ! -d "code" ]]; then
  mkdir code
fi

if [[ ! -d "dist" ]]; then
  mkdir dist
fi

bucket='fds://gzg/code_package'
code_file='text-cls-1.0.tar.gz'

rm -f code/*
rm -f dist/*

cp ./*.py code/
fdscli rm ${bucket}${code_file}
python setup.py sdist --format=gztar
fdscli cp dist/${code_file} ${bucket}
