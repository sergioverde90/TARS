package com.tars.tools;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

@Component
public class WebsiteScrappingTool {

    private final HttpClient httpClient;

    public WebsiteScrappingTool() {
        this.httpClient = HttpClient.newHttpClient();
    }

    @Tool("Fetches the content of a web page at the given URL and returns it as clean markdown text, including the page title, main content, and any relevant structured information. Use this when you need to read the actual content of a specific URL.")
    public String scrape(@P("the full URL to fetch, including https://") String url) {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("https://r.jina.ai/" + url))
                    .header("Accept", "text/plain")
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            String content = response.body();

            // Trim for LLM context
            if (content.length() > 8000) {
                content = content.substring(0, 8000) + "...";
            }

            return content;

        } catch (Exception e) {
            throw new RuntimeException("Scrape failed for: " + url, e);
        }
    }

}