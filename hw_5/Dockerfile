FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

# Set the working directory in the container
WORKDIR /app

# Copy the new Pipfile and Pipfile.lock to the container
COPY ["Pipfile", "Pipfile.lock", "model1.bin", "dv.bin", "./"]

# Install dependencies from the Pipenv file
RUN pipenv install --system --deploy

# Copy your Flask script to the container
COPY ["predict.py", "./"]

# Expose the port on which your Flask app will run
EXPOSE 9696

# Run your Flask app with Gunicorn
ENTRYPOINT ["pipenv", "run", "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
