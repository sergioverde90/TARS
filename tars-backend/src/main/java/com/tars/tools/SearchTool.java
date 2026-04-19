package com.tars.tools;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.tars.config.TarsProperties;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.util.Map;

@Component
public class SearchTool {

    private static final Logger log = LoggerFactory.getLogger(SearchTool.class);

    private final RestClient http;
    private final String apiKey;
    private final ObjectMapper mapper = new ObjectMapper();

    public SearchTool(TarsProperties props) {
        this.apiKey = props.tavily().apiKey();
        this.http = RestClient.builder()
            .baseUrl(props.tavily().baseUrl())
            .build();
    }

    @Tool("Searches the web for current information on any topic. Use when the user asks about recent news, facts, or anything that may have changed.")
    public String webSearch(@P("search query") String query) {
        try {
            Map<String, Object> body = Map.of(
                "api_key", apiKey,
                "query", query,
                "max_results", 3,
                "search_depth", "basic"
            );

            String response = http.post()
                .uri("/search")
                .contentType(MediaType.APPLICATION_JSON)
                .body(body)
                .retrieve()
                .body(String.class);

            JsonNode results = mapper.readTree(response).path("results");
            if (results.isEmpty()) return "No results found for: " + query;

            StringBuilder sb = new StringBuilder();
            for (JsonNode result : results) {
                sb.append("- ").append(result.path("title").asText())
                  .append(": ").append(result.path("content").asText(), 0,
                      Math.min(200, result.path("content").asText().length()))
                  .append("\n");
            }
            return sb.toString().trim();

        } catch (Exception e) {
            log.error("SearchTool error for '{}': {}", query, e.getMessage());
            return "Search failed for '" + query + "': " + e.getMessage();
        }
    }
}
