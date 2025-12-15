FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install git (required by setuptools-scm for version detection)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the project into the image
COPY . /app

# Set working directory
WORKDIR /app

# Install Python dependencies (skip dev dependencies)
RUN uv sync --locked --no-dev

# Add the virtual environment to PATH so we use the installed packages
ENV PATH="/app/.venv/bin:$PATH"

# Expose Chainlit default port
EXPOSE 8000

# Expose MSAL authentication callback port
EXPOSE 42071

# Run the Chainlit app directly from the venv
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]