FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Node.js 22 LTS from NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Working directory (RunPod convention)
WORKDIR /workspace

# Entrypoint
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
