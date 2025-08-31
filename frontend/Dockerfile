# frontend/Dockerfile
FROM node:20-alpine

WORKDIR /app

COPY package.json ./
COPY yarn.lock ./  # or pnpm-lock.yaml if using pnpm

# install deps
RUN corepack enable && corepack prepare pnpm@9.6.0 --activate
RUN pnpm install

COPY . .

EXPOSE 5173

CMD ["pnpm", "dev", "--host", "0.0.0.0"]

