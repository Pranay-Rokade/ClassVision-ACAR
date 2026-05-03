# ============================================================
# ClassVision-ACAR — Frontend Dockerfile
# Multi-stage build: Node (build) → Nginx (serve)
# ============================================================

# ---- Stage 1: Build the React app ----
FROM node:18-alpine AS build

WORKDIR /app

# Copy package files first for layer caching
COPY package.json package-lock.json ./
RUN npm ci

# Copy all source code
COPY . .

# Set the API URL to empty string — in Docker, nginx proxies
# /auth/*, /classify/*, /analysis/* to the backend container,
# so the frontend just uses relative URLs (same origin).
ENV VITE_API_URL=""

# Build production bundle
RUN npm run build

# ---- Stage 2: Serve with Nginx ----
FROM nginx:alpine

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built files from Stage 1
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
