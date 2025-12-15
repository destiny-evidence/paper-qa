# Docker Setup for Chainlit UI

This guide explains how the DESTINY repo search agent UI in a docker container.

## Quick Start

### Using Docker Compose (Recommended)

1. **Build and run the container:**
   ```bash
   docker compose up --build
   ```

2. **Access the UI:**
   Open your browser to http://localhost:8000

3. **Stop the container:**
   ```bash
   docker compose down
   ```

### Using Docker Directly (Untested)

1. **Build the image:**
   ```bash
   docker build -t paper-qa-chainlit .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 -p 42071:42071 --env-file .env paper-qa-chainlit
   ```

## Environment Variables

The application requires several environment variables for API access. These are automatically loaded from your `.env` file in the project's root directory when using docker-compose.

If you haven't created one yet, make sure to do so!

**Required variables:**
- `AZURE_API_BASE` - Azure OpenAI endpoint
- `AZURE_API_KEY` - Azure OpenAI API key
- `DESTINY_API_URL` - DESTINY API URL
- `DESTINY_CLIENT_ID` - DESTINY OAuth client ID
- `DESTINY_AUTHORITY` - DESTINY OAuth authority
- `DESTINY_LOGIN_HINT` - User email for DESTINY
- `DESTINY_SCOPES` - DESTINY API scopes

## Data Persistence

Docker Compose sets up volumes for persistent data:
- `chainlit-data` - Chainlit configuration and session data
- `chainlit-files` - Uploaded files
- `chainlit-public` - Public assets

To view or backup this data:
```bash
docker volume ls
docker volume inspect paper-qa-chainlit-data
```

## Troubleshooting

### Port already in use
If port 8000 is already in use, change it in docker-compose.yml:
```yaml
ports:
  - "8080:8000"  # Use port 8080 on host
```

### Authentication issues with DESTINY
The DESTINY OAuth flow requires interactive authentication using MSAL. The container exposes port 42071 for the authentication callback.

**How it works:**
1. When you start the container and visit `localhost:8000` in your browser, MSAL will print an authentication URL in your terminal.
2. Copy the second URL and open it in your **host machine's browser** (not inside the container)
3. Complete the Microsoft login
4. The callback will be sent to `localhost:42071` which is forwarded to the container
5. Authentication completes and the token is cached

**If authentication fails:**
- Ensure port 42071 is not blocked by your firewall
- Check that the port mapping is correct: `docker compose ps`
- Ensure you're visiting the **second** authentication URL from your terminal
- Ensure you visit `localhost:8000` in your browser before looking for the auth link your terminal

### View logs
```bash
docker compose logs -f chainlit-ui
```

## Cleanup

Remove all containers and volumes:
```bash
docker compose down -v
```

Remove the image:
```bash
docker rmi paper-qa-chainlit
```