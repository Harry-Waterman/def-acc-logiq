FROM node:20-bullseye

WORKDIR /app

COPY package*.json ./
RUN npm install --production

COPY src ./src

ENV DATASET_PATH=/workspace/dataset/Nigerian_Fraud.csv
ENV MLC_MODEL_ID=Qwen3-0.6B-q4f16_1-MLC
ENV MLC_SERVER_URL=http://mlc-server:8080

CMD ["npm", "run", "benchmark"]

