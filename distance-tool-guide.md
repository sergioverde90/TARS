# Building a Distance Tool for LLMs (Nominatim + HERE API)

## 1. Overview

The full flow for calculating distances between two places:

```
User: "How long from Triana to Nervión?"
        ↓
1. Nominatim  →  "Triana, Sevilla"        →  lat: 37.3826, lon: -6.0022
2. Nominatim  →  "Nervión, Sevilla"       →  lat: 37.3774, lon: -5.9692
        ↓
3. HERE API   →  driving route with live traffic
        ↓
Result: "2.4 km, ~8 min (with current traffic)"
```

---

## 2. MCP Server (Node.js)

You can build a lightweight MCP server that exposes a `calculate_distance` tool. Your LLM can then call it during conversations.

```javascript
// distance-mcp-server.js
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "distance-server", version: "1.0.0" });

server.tool(
  "calculate_distance",
  "Calculate the distance between two coordinates (lat/lng) in km or miles",
  {
    lat1: z.number().describe("Latitude of point A"),
    lon1: z.number().describe("Longitude of point A"),
    lat2: z.number().describe("Latitude of point B"),
    lon2: z.number().describe("Longitude of point B"),
    unit: z.enum(["km", "miles"]).default("km"),
  },
  async ({ lat1, lon1, lat2, lon2, unit }) => {
    const R = unit === "km" ? 6371 : 3958.8;
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) ** 2 +
      Math.cos((lat1 * Math.PI) / 180) *
        Math.cos((lat2 * Math.PI) / 180) *
        Math.sin(dLon / 2) ** 2;
    const distance = R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return {
      content: [{ type: "text", text: `Distance: ${distance.toFixed(2)} ${unit}` }],
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

To use it, add it to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "distance": {
      "command": "node",
      "args": ["/path/to/distance-mcp-server.js"]
    }
  }
}
```

---

## 3. Geocoding with Nominatim

To get coordinates from a place name, use the free Nominatim API (no API key required).

### cURL Example

```bash
curl 'https://nominatim.openstreetmap.org/search?q=Triana%2C%20Sevilla&format=json&limit=1' \
  -H 'User-Agent: my-distance-app/1.0'
```

### Sample Response

```json
[
  {
    "place_id": 423877308,
    "licence": "Data © OpenStreetMap contributors, ODbL 1.0.",
    "osm_type": "relation",
    "osm_id": 342563,
    "lat": "37.3886303",
    "lon": "-5.9953403",
    "class": "boundary",
    "type": "administrative",
    "place_rank": 16,
    "importance": 0.7297329708417591,
    "addresstype": "city",
    "name": "Sevilla",
    "display_name": "Sevilla, Andalucía, España",
    "boundingbox": ["37.3002036", "37.4529579", "-6.0329183", "-5.8191571"]
  }
]
```

### Key Fields

| Field | What it means |
|---|---|
| `lat` / `lon` | The coordinates you need ✅ |
| `display_name` | Human-readable full address |
| `place_rank` | Lower = more specific (16 = city, 10 = province) |
| `importance` | Higher = more relevant |
| `addresstype` | What kind of place: `city`, `province`, `road`, etc. |
| `boundingbox` | Bounding box of the area `[minLat, maxLat, minLon, maxLon]` |

> **Tip:** Nominatim sorts results by `importance` descending, so `data[0]` is always the best match.

### Geocoding Function

```javascript
async function geocode(address) {
  const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(address)}&format=json&limit=1`;
  const res = await fetch(url, {
    headers: { "User-Agent": "my-distance-app/1.0" }
  });
  const data = await res.json();

  if (!data.length) throw new Error(`Address not found: "${address}"`);

  const place = data[0];
  return {
    lat: parseFloat(place.lat),
    lon: parseFloat(place.lon),
    display_name: place.display_name,
    type: place.addresstype,
  };
}
```

---

## 4. Routing with HERE API (with Traffic)

OSRM does **not** support real-time traffic — it uses static OpenStreetMap data. For live traffic, use the HERE Routing API.

### Traffic-aware APIs Comparison

| Service | Traffic | Free Tier | API Key |
|---|---|---|---|
| **OSRM** | ❌ None | Unlimited | ❌ None needed |
| **Google Maps** | ✅ Live | 40k/month | ✅ Required |
| **HERE** | ✅ Live | 250k/month | ✅ Required |
| **TomTom** | ✅ Live | 2,500/day | ✅ Required |

### cURL Example

```bash
curl 'https://router.hereapi.com/v8/routes?transportMode=car&origin=37.3826,-6.0022&destination=37.3774,-5.9692&departureTime=now&return=summary&apikey=YOUR_API_KEY'
```

### Useful Query Parameters

| Param | Example | What it does |
|---|---|---|
| `overview` | `overview=false` | Skip geometry (lighter response) |
| `steps` | `steps=true` | Include turn-by-turn directions |
| `alternatives` | `alternatives=true` | Return alternative routes |
| `departureTime` | `now` | Use live traffic |

### ⚠️ Important: OSRM uses lon,lat (not lat,lon)

```javascript
// ✅ Correct
`${place.lon},${place.lat}`

// ❌ Wrong
`${place.lat},${place.lon}`
```

---

## 5. Full Node.js Script (Nominatim + HERE)

```javascript
const HERE_API_KEY = "YOUR_API_KEY";

async function geocode(placeName) {
  const query = `${placeName}, Sevilla`;
  const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=1`;

  const res = await fetch(url, {
    headers: { "User-Agent": "my-distance-app/1.0" },
  });
  const data = await res.json();

  if (!data.length) throw new Error(`Place not found: "${placeName}"`);

  return {
    lat: parseFloat(data[0].lat),
    lon: parseFloat(data[0].lon),
    display_name: data[0].display_name,
  };
}

async function getRoute(originCoords, destinationCoords) {
  const { lat: lat1, lon: lon1 } = originCoords;
  const { lat: lat2, lon: lon2 } = destinationCoords;

  const url = `https://router.hereapi.com/v8/routes?transportMode=car&origin=${lat1},${lon1}&destination=${lat2},${lon2}&departureTime=now&return=summary&apikey=${HERE_API_KEY}`;

  const res = await fetch(url);
  const data = await res.json();

  if (!data.routes?.length) throw new Error("No route found");

  const summary = data.routes[0].sections[0].summary;
  return {
    distance_km: (summary.length / 1000).toFixed(2),
    duration_min: Math.round(summary.duration / 60),
    duration_traffic_min: Math.round((summary.typicalDuration ?? summary.duration) / 60),
  };
}

async function calculateDistance(placeA, placeB) {
  console.log(`Geocoding "${placeA}"...`);
  const origin = await geocode(placeA);
  console.log(`→ ${origin.display_name} (${origin.lat}, ${origin.lon})`);

  console.log(`Geocoding "${placeB}"...`);
  const destination = await geocode(placeB);
  console.log(`→ ${destination.display_name} (${destination.lat}, ${destination.lon})`);

  console.log("Fetching route with traffic...");
  const route = await getRoute(origin, destination);

  return {
    from: origin.display_name,
    to: destination.display_name,
    distance_km: route.distance_km,
    duration_min: route.duration_min,
    duration_traffic_min: route.duration_traffic_min,
  };
}

// --- Run it ---
const result = await calculateDistance("Triana", "Nervión");

console.log("\n--- Result ---");
console.log(`From:     ${result.from}`);
console.log(`To:       ${result.to}`);
console.log(`Distance: ${result.distance_km} km`);
console.log(`Duration: ${result.duration_min} min (typical)`);
console.log(`Traffic:  ${result.duration_traffic_min} min (with current traffic)`);
```

### Sample Output

```
Geocoding "Triana"...
→ Triana, Sevilla, Andalucía, España (37.3826, -6.0022)
Geocoding "Nervión"...
→ Nervión, Sevilla, Andalucía, España (37.3774, -5.9692)
Fetching route with traffic...

--- Result ---
From:     Triana, Sevilla, Andalucía, España
To:       Nervión, Sevilla, Andalucía, España
Distance: 2.40 km
Duration: 7 min (typical)
Traffic:  11 min (with current traffic)
```

---

## 6. Next Step: Wrap into a Full MCP Server

The `calculateDistance` function above is the core logic. The next step is to wrap it into an MCP server tool so your LLM can call it automatically during conversations — combining the geocoding, routing, and traffic steps into a single tool call like:

```
calculate_distance("Triana", "Nervión")
→ 2.4 km, 11 min with current traffic
```
