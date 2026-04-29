# syntax=docker/dockerfile:1.7

# ============================================================================
# Stage 1 — Install deps
# ============================================================================
FROM node:20-alpine AS deps
WORKDIR /app

# Alpine ships musl; some Next.js / SWC bits need libc. The package below
# pulls in what's needed without bloating the image.
RUN apk add --no-cache libc6-compat

COPY web/frontend/package.json web/frontend/package-lock.json* ./
RUN npm ci


# ============================================================================
# Stage 2 — Build (next build with standalone output)
# ============================================================================
FROM node:20-alpine AS builder
WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY web/frontend/ ./

# NEXT_PUBLIC_* values are baked at build time. The Express backend URL
# comes from the compose file (default http://localhost:3001 for dev).
ARG NEXT_PUBLIC_API_URL=http://localhost:3001
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

RUN npm run build


# ============================================================================
# Stage 3 — Runtime (standalone server, no node_modules needed)
# ============================================================================
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN apk add --no-cache curl tini \
    && addgroup --system --gid 1001 nodejs \
    && adduser  --system --uid 1001 nextjs

# .next/standalone contains a self-contained server.js + minimal deps.
# .next/static and public/ must be copied alongside it (Next docs).
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static     ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public           ./public

USER nextjs

EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT} || exit 1

ENTRYPOINT ["/sbin/tini", "--"]
CMD ["node", "server.js"]
