FROM python:latest
MAINTAINER sugyeong.jo@unist.ac.kr

RUN pip install --upgrade pip
RUN pip install pandas
RUN pip install numpy
RUN pip install mip
