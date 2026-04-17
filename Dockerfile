#Use slim python 3.11 base image
FROM python:3.11-slim

#Set working directory
WORKDIR /app

#Needed for MLflow packages and optimization
#rm -rf /var/lib/apt/lists/* to reduce image size of cached files
#installs curl for this python version
#gcc and libgomp1 are needed for MLflow packages that require compilation and optimization
#libgomp1 is the GNU Offloading and Multi Processing Runtime Library, which is required for parallel processing and 
#optimization in MLflow packages that utilize multi-threading or multi-processing capabilities.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* 

#Copy requirements file
COPY requirements.txt .

#Install requirements
RUN pip install --no-cache-dir -r requirements.txt

#Copy app source code
COPY app.py .

# Copy only needed model files
COPY models/regression_nn.py ./models/
COPY models/preprocessor.joblib ./models/

# Copy only needed data taxi zone lookup file
COPY data/taxi_zone_lookup.csv ./data/

#Exposes port 8000 for FastAPI
EXPOSE 8000

#Starting FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]