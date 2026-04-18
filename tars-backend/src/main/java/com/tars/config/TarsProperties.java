package com.tars.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "tars")
public record TarsProperties(
    OpenRouteService openrouteservice,
    Tavily tavily
) {
    public record OpenRouteService(String apiKey, String baseUrl) {}
    public record Tavily(String apiKey, String baseUrl) {}
}
