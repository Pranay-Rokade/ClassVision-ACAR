# ============================================================
# ClassVision-ACAR — Frontend Dockerfile
# Multi-stage build: Node (build) → Nginx (serve)
# ============================================================

# ---- Stage 1: Build the React app ----
# Using node:18 (Debian) instead of Alpine to avoid rollup native binary issues on ARM64
FROM node:18 AS build

WORKDIR /app

# Copy package files first for layer caching
COPY package.json package-lock.json ./

# Delete lockfile and reinstall — the macOS-generated lockfile doesn't
# include Linux ARM64 native binaries for rollup/vite.
RUN rm -f package-lock.json && npm install

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
