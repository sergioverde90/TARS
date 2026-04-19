package com.tars.tools;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.tars.config.TarsProperties;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.util.List;
import java.util.Map;

@Component
public class DistanceTool {

    private static final Logger log = LoggerFactory.getLogger(DistanceTool.class);

    private final RestClient http;
    private final String apiKey;
    private final ObjectMapper mapper = new ObjectMapper();

    public DistanceTool(TarsProperties props) {
        this.apiKey = props.openrouteservice().apiKey();
        this.http = RestClient.builder()
            .baseUrl(props.openrouteservice().baseUrl())
            .defaultHeader("Authorization", apiKey)
            .defaultHeader(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE)
            .build();
    }

    @Tool("Calculates the driving distance and estimated travel time between two places by car")
    public String drivingDistance(
        @P("origin place name or address") String from,
        @P("destination place name or address") String to
    ) {
        try {
            double[] originCoords = geocode(from);
            double[] destCoords = geocode(to);

            Map<String, Object> body = Map.of(
                "coordinates", List.of(
                    List.of(originCoords[0], originCoords[1]),
                    List.of(destCoords[0], destCoords[1])
                )
            );

            String response = http.post()
                .uri("/v2/directions/driving-car")
                .contentType(MediaType.APPLICATION_JSON)
                .body(body)
                .retrieve()
                .body(String.class);

            JsonNode root = mapper.readTree(response);
            JsonNode summary = root.path("routes").get(0).path("summary");
            double distanceKm = summary.path("distance").asDouble() / 1000.0;
            double durationMin = summary.path("duration").asDouble() / 60.0;

            return String.format("Driving from %s to %s: %.1f km, approximately %.0f minutes by car.",
                from, to, distanceKm, durationMin);

        } catch (Exception e) {
            log.error("DistanceTool error for {} → {}: {}", from, to, e.getMessage());
            return "Could not calculate route from " + from + " to " + to + ": " + e.getMessage();
        }
    }

    private double[] geocode(String place) throws Exception {
        String response = http.get()
            .uri(uriBuilder -> uriBuilder
                .path("/geocode/search")
                .queryParam("api_key", apiKey)
                .queryParam("text", place)
                .queryParam("size", 1)
                .build())
            .retrieve()
            .body(String.class);

        JsonNode features = mapper.readTree(response).path("features");
        if (features.isEmpty()) throw new RuntimeException("Place not found: " + place);
        JsonNode coords = features.get(0).path("geometry").path("coordinates");
        return new double[]{ coords.get(0).asDouble(), coords.get(1).asDouble() };
    }
}
