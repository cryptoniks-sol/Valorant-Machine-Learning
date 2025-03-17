# StatsVLR-API

## ✨ Features

- Scraping data from vlr.gg efficiently and securely.
- Providing a simple and intuitive API to access player and team information.
- Regular updates to keep the data current and relevant.
- Lightweight and easy to integrate into other projects.

## 🎯 Usage

Once you have the API server up and running, you can interact with it using HTTP requests. The API provides endpoints for accessing players', teams', and match history data. Here's a basic example of how to use the API with cURL:

```bash
# Get information about a specific player
curl -X GET http://localhost:5000/api/v1/players/{playerid}

# Get information about a specific team
curl -X GET http://localhost:5000/api/v1/teams/{teamid}

# Get match history of a specific team
curl -X GET http://localhost:5000/api/v1/match-history/{teamid}
```

## 📚 API Endpoints

The following are the main endpoints provided by the API:

### Players

- `GET /api/v1/players` - Retrieve information about all players.
  - **Parameters:**
    - `page` (default: `1`) - Current page number.
    - `limit` (default: `10`) - Limit of results per page. Use `all` to get all players.
    - `event_series` (default: `all`) - Event group ID.
    - `event` (default: `all`) - Specific event ID.
    - `region` (default: `all`) - Filter by region (`na`, `eu`, `ap`, `jp`, `sa`, `oce`, `mn`, `gc`).
    - `country` (default: `all`) - Filter by country (e.g., `co`, `es`).
    - `minrounds` (default: `200`) - Minimum number of rounds played.
    - `minrating` (default: `1550`) - Minimum rating.
    - `agent` (default: `all`) - Filter by agent (e.g., `astra`, `jett`).
    - `map` (default: `all`) - Filter by map (map ID).
    - `timespan` (default: `60d`) - Time period (`30d`, `60d`, `90d`, `all`).
  - **Example Request:**
    ```bash
    curl -X GET "http://localhost:5000/api/v1/players?limit=3&country=co"
    ```

- `GET /api/v1/players/{playerid}` - Retrieve information about a specific player.
  - **Parameters:**
    - `playerid` (Required) - ID of the player to consult.

### Teams

- `GET /api/v1/teams` - Retrieve information about all teams.
  - **Parameters:**
    - `page` (default: `1`) - Current page number.
    - `limit` (default: `10`) - Limit of results per page. Use `all` to get all teams.
    - `region` (default: `all`) - Filter by region (`na`, `eu`, `br`, `ap`, `kr`, `ch`, `jp`, `lan`, `las`, `oce`, `mn`, `gc`).
  - **Example Request:**
    ```bash
    curl -X GET "http://localhost:5000/api/v1/teams?region=lan"
    ```

- `GET /api/v1/teams/{teamid}` - Retrieve detailed information about a specific team.
  - **Parameters:**
    - `teamid` (Required) - ID of the team to consult.

### Match History

- `GET /api/v1/match-history/{teamid}` - Retrieve match history of a specific team.
  - **Parameters:**
    - `teamid` (Required) - ID of the team to fetch match history for.
  - **Example Request:**
    ```bash
    curl -X GET "http://localhost:5000/api/v1/match-history/{teamid}"
    ```


### Events

- `GET /api/v1/events` - Retrieve information about all events.
  - **Parameters:**
    - `page` (default: `1`) - Current page number.
    - `status` (default: `all`) - Filter events by their status (`ongoing`, `upcoming`, `completed`, or `all`).
    - `region` (default: `all`) - Filter by region (`na`, `eu`, `br`, `ap`, `kr`, `ch`, `jp`, `lan`, `las`, `oce`, `mn`, `gc`).
  - **Example Request:**
    ```bash
    curl -X GET "http://localhost:5000/api/v1/events?page=1&status=upcoming&region=all"
    ```

### Matches

- `GET /api/v1/matches` - Retrieve information about upcoming matches or matches currently being played.

### Results

- `GET /api/v1/results` - Retrieve information about past match results.
  - **Parameters:**
    - `page` (default: `1`) - Current page number.
  - **Example Request:**
    ```bash
    curl -X GET "http://localhost:5000/api/v1/results?page=1"
    ```

## 🔗 Base URL

All API requests should be prefixed with:
```
http://localhost:5000/api/v1/
```