package com.tars.config;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.tars.chat.ToolMetadata;
import com.tars.tools.*;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

@Configuration
public class TarsConfiguration {

    @Bean
    public ObjectMapper objectMapper() {
        return new ObjectMapper();
    }

    @Bean
    public Map<String, Function<String, String>> toolHandlers(
            ObjectMapper mapper, TimeTool timeTool,
            DistanceTool distanceTool,
            SearchTool searchTool,
            WebsiteScrappingTool websiteScrappingTool,
            YahooFinanceTool yahooFinanceTool){
        Map<String, Function<String, String>> toolHandlers = new HashMap<>();
        toolHandlers.put("currentTime", args -> timeTool.currentTime());
        toolHandlers.put("lookupTicker", args -> {
            try {
                JsonNode node = mapper.readTree(args);
                return yahooFinanceTool.lookupTicker(node.path("companyName").asText());
            } catch (JsonProcessingException e) {
                throw new RuntimeException(e);
            }
        });
        toolHandlers.put("getStockPrice", args -> {
            try {
                JsonNode node = mapper.readTree(args);
                return yahooFinanceTool.getStockPrice(node.path("ticker").asText());
            } catch (JsonProcessingException e) {
                throw new RuntimeException(e);
            }
        });
        toolHandlers.put("scrape", args -> {
            try {
                JsonNode node = mapper.readTree(args);
                return websiteScrappingTool.scrape(node.path("url").asText());
            } catch (JsonProcessingException e) {
                throw new RuntimeException(e);
            }
        });
        toolHandlers.put("drivingDistance", args -> {
            try {
                JsonNode node = mapper.readTree(args);
                return distanceTool.drivingDistance(
                        node.path("from").asText(),
                        node.path("to").asText()
                );
            } catch (Exception e) {
                return "Error parsing tool args: " + e.getMessage();
            }
        });
        toolHandlers.put("webSearch", args -> {
            try {
                JsonNode node = mapper.readTree(args);
                return searchTool.webSearch(node.path("query").asText());
            } catch (Exception e) {
                return "Error parsing tool args: " + e.getMessage();
            }
        });

        return toolHandlers;
    }

    @Bean
    public List<ToolMetadata> toolMetadata() {
        return List.of(
            // currentTime — no parameters
            new ToolMetadata(
                "currentTime",
                "Returns the current date and time in the user's local timezone (Europe/Madrid, Spain)",
                Map.of("type", "object", "properties", Map.of(), "required", List.of())
            ),
            // lookupTicker
            new ToolMetadata(
                "lookupTicker",
                "Look up the stock ticker symbol for a company name (e.g., 'Apple' -> 'AAPL')",
                Map.of(
                    "type", "object",
                    "properties", Map.of(
                        "companyName", Map.of("type", "string", "description", "The full name of the company")
                    ),
                    "required", List.of("companyName")
                )
            ),
            // getStockPrice
            new ToolMetadata(
                "getStockPrice",
                "Get the current stock price and currency for a specific ticker symbol (e.g., 'AAPL')",
                Map.of(
                    "type", "object",
                    "properties", Map.of(
                        "ticker", Map.of("type", "string", "description", "The stock ticker symbol, e.g., AAPL")
                    ),
                    "required", List.of("ticker")
                )
            ),
            // scrape
            new ToolMetadata(
                "scrape",
                "Fetches the content of a web page at the given URL and returns it as clean markdown text, including the page title, main content, and any relevant structured information. Use this when you need to read the actual content of a specific URL.",
                Map.of(
                    "type", "object",
                    "properties", Map.of(
                        "url", Map.of("type", "string", "description", "the full URL to fetch, including https://")
                    ),
                    "required", List.of("url")
                )
            ),
            // drivingDistance
            new ToolMetadata(
                "drivingDistance",
                "Calculates the driving distance and estimated travel time between two places by car",
                Map.of(
                    "type", "object",
                    "properties", Map.of(
                        "from", Map.of("type", "string", "description", "origin place name or address"),
                        "to", Map.of("type", "string", "description", "destination place name or address")
                    ),
                    "required", List.of("from", "to")
                )
            ),
            // webSearch
            new ToolMetadata(
                "webSearch",
                "Searches the web for current information on any topic. Use when the user asks about recent news, facts, or anything that may have changed.",
                Map.of(
                    "type", "object",
                    "properties", Map.of(
                        "query", Map.of("type", "string", "description", "search query")
                    ),
                    "required", List.of("query")
                )
            )
        );
    }

}
