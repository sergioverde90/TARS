package com.tars.tools;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Component;

@Component
public class YahooFinanceTool {

    private final HttpClient client = HttpClient.newHttpClient();
    private final ObjectMapper mapper = new ObjectMapper();
    private final String USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36";

    @Tool("Look up the stock ticker symbol for a company name (e.g., 'Apple' -> 'AAPL')")
    public String lookupTicker(@P("The full name of the company") String companyName) {
        try {
            String url = "https://query2.finance.yahoo.com/v1/finance/search?q="
                    + companyName.replace(" ", "%20");

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("User-Agent", USER_AGENT)
                    .GET()
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            JsonNode root = mapper.readTree(response.body());

            // Return a summary of the first few matches for the LLM to choose from
            return root.path("quotes").get(0).toString();
        } catch (Exception e) {
            return "Error finding ticker: " + e.getMessage();
        }
    }

    @Tool("Get the current stock price and currency for a specific ticker symbol (e.g., 'AAPL')")
    public String getStockPrice(@P("The stock ticker symbol, e.g., AAPL") String ticker) {
        try {
            String url = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker;

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("User-Agent", USER_AGENT)
                    .GET()
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            JsonNode root = mapper.readTree(response.body());

            JsonNode meta = root.path("chart").path("result").get(0).path("meta");
            double price = meta.path("regularMarketPrice").asDouble();
            String currency = meta.path("currency").asText();

            return String.format("The current price of %s is %.2f %s", ticker, price, currency);
        } catch (Exception e) {
            return "Error fetching price for " + ticker + ": " + e.getMessage();
        }
    }
}
