FROM python:3.8-slim

ADD . .

WORKDIR .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 2000

CMD ["gunicorn", "-w", "4","-b","0.0.0.0:2000", "geoinformation_processor:app" ]