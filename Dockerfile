# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./main.py /code/

#
COPY ./RPC_Model.h5 /code/

#
RUN mkdir /code/static

# 
CMD ["fastapi", "run", "main.py", "--port", "80"]